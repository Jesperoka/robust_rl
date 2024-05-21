import threading
import multiprocessing
import os
from jax import jit
from jax.tree_util import Partial
from jax._src.pjit import JitWrapped
from jax._src.stages import Compiled
import numpy as np
import cv2
from traceback import format_exc
from functools import partial
from jax.numpy import logical_and, logical_or
from typing import Callable, NamedTuple, Self, TypeAlias
from panda_py import Panda, PandaContext, controllers
from panda_py.libfranka import Robot, RobotState, Duration, Gripper, GripperState, Torques, set_current_thread_to_highest_scheduler_priority
from websockets.sync.client import connect
from enum import IntEnum
from time import clock_gettime_ns, CLOCK_BOOTTIME, sleep
from json import dumps
from environments.physical import PandaLimits
from inference.setup_for_inference import setup_camera, setup_env, load_policies, setup_panda, observe, policy_inference, reset_goal_pos, rotation_velocity_modifier
from inference.processing import LowPassFilter
from inference.controllers import arm_spline_tracking_controller
from environments.A_to_B_jax import A_to_B
from environments.options import EnvironmentOptions as _EnvOpts
from pyrealsense2 import pipeline
from pupil_apriltags import Detector, Detection


ZEUS_URI = "ws://192.168.4.1:8765"

ZeusMode = IntEnum("ZeusMode", ["STANDBY", "ACT", "CONTINUE"], start=0)

class ZeusMessage:
    new = False
    msg = {"A": 0.0, "B": 0.0, "C": 0.0, "D": ZeusMode.STANDBY}
    mtx = threading.Lock()

    def write(self, A: float, B: float, C: float, D: ZeusMode) -> None:
        with self.mtx:
            self.msg.update([("A", A), ("B", B), ("C", C), ("D", D)])
            self.new = True

    def read(self) -> dict | bool:
        with self.mtx:
            if not self.new:
                return False

            self.new = False

            return self.msg

# WARNING: UNFINISHED
class ControllerData:
    mtx = threading.Lock()

    def __init__(self,
                 b0: np.ndarray = np.array(PandaLimits().q_start),
                 b1: np.ndarray = np.array(PandaLimits().q_start),
                 b2: np.ndarray = np.array(PandaLimits().q_start),
                 b3: np.ndarray = np.array(PandaLimits().q_start),
                 t: float = 0.0
                 ) -> None:

        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.t = t

    def update_ctrl_points(self, b_new: np.ndarray) -> None:
        with self.mtx:
            self.b0 = self.b1
            self.b1 = self.b2
            self.b2 = self.b3
            self.b3 = b_new

    def update_time(self, duration: float) -> None: # NOTE: can remove mutex if time is only updated by control callback
        with self.mtx:
            self.t = min(self.t + duration, 1.0)

    def zero_time(self) -> None:
        with self.mtx:
            self.t = 0.0

    def read(self):
        with self.mtx:
            return self.b0, self.b1, self.b2, self.b3, self.t

    def copy(self) -> Self:
        with self.mtx:
            return self.__class__(self.b0, self.b1, self.b2, self.b3, self.t)

class _gripper_state:
    mtx = threading.Lock()
    state: GripperState

    def __init__(self, state: GripperState) -> None:
        self.state = state

    def update(self, state: GripperState) -> None:
        with self.mtx:
            self.state = state

    def read(self) -> GripperState:
        with self.mtx:
            return self.state


class Pose(NamedTuple):
    R: np.ndarray = np.eye(3)
    t: np.ndarray = np.zeros((3, 1))

class PoseEstimates:
    car_pose = Pose()
    floor_pose = Pose()
    mtx = threading.Lock()

    def update_data(self, data) -> None:
        with self.mtx:
            self.car_pose = data[0] if data[0] is not None else self.car_pose
            self.floor_pose = data[1] if data[1] is not None else self.floor_pose

    def get_data(self) -> tuple[Pose, Pose]:
        with self.mtx:
            return self.car_pose, self.floor_pose

VizData: TypeAlias = Pose

class Visualization:
    mtx = threading.Lock()

    frame: np.ndarray
    K: np.ndarray
    dist_coeffs: np.ndarray

    data: list[VizData] = [VizData(), VizData()]

    def __init__(self, w, h, K, dist_coeffs) -> None:
        self.frame = np.zeros((h, w), dtype=np.uint8)
        self.K = K
        self.dist_coeffs = dist_coeffs

    def update_data(self, data, frame) -> None:
        with self.mtx:
            self.data = data
            self.frame = frame

    def draw_axes(self) -> None:
        with self.mtx:
            for data in self.data:
                if data is None: continue
                rotV, _ = cv2.Rodrigues(data.R)
                points = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1], [0, 0, 0]], dtype=np.float32).reshape(-1, 3)
                axisPoints, _ = cv2.projectPoints(points, rotV, data.t, self.K, self.dist_coeffs)
                axisPoints = axisPoints.astype(int)
                self.frame = cv2.line(self.frame, axisPoints[3].ravel(), tuple(axisPoints[0].ravel()), (255,0,0), 3)
                self.frame = cv2.line(self.frame, axisPoints[3].ravel(), tuple(axisPoints[1].ravel()), (0,255,0), 3)
                self.frame = cv2.line(self.frame, axisPoints[3].ravel(), tuple(axisPoints[2].ravel()), (0,0,255), 3)

    def imshow(self) -> None:
        with self.mtx:
            cv2.imshow("Realsense Camera", self.frame)

def websocket_client(
        msg: ZeusMessage,
        msg_event: threading.Event,
        exit_event: threading.Event,
        num_stop_cmds: int = 5,
        num_act_cmds: int = 5,
    ) -> None:

    with connect(ZEUS_URI) as socket:
        print("Connected to ZeusCar")
        while not exit_event.is_set():
            result = msg_event.wait(timeout=0.5)
            if not result: continue

            if m := msg.read():
                for _ in range(num_act_cmds):
                    socket.send(dumps(m))
                msg_event.clear()

        for _ in range(num_stop_cmds):
            socket.send(dumps(ZeusMessage().msg))

    print("\nDisconnected from ZeusCar...")


def apriltag_detection(
    pipe: pipeline,
    cam_params: tuple[float, float, float, float],
    detector: Detector,
    ) -> tuple[list[VizData | None], np.ndarray, bool]:

    frame = pipe.wait_for_frames().get_infrared_frame()
    got_frame = True if frame else False
    image = np.asarray(frame.data, dtype=np.uint8)

    detections: list[Detection] = detector.detect(
        image,
        estimate_tag_pose=True,
        camera_params=cam_params,
        tag_size=0.1858975
    ) # type: ignore[assignment]

    car_detection = list(filter(lambda d: d.tag_id == 0, detections))
    floor_detection = list(filter(lambda d: d.tag_id == 1, detections))

    viz_data: list[VizData | None] = [None, None]
    if car_detection:
        viz_data[0] = VizData(
            R=car_detection[0].pose_R,      # type: ignore[assignment]
            t=car_detection[0].pose_t       # type: ignore[assignment]
        )

    if floor_detection:
        viz_data[1] = VizData(
            R=floor_detection[0].pose_R,    # type: ignore[assignment]
            t=floor_detection[0].pose_t     # type: ignore[assignment]
        )

    return viz_data, image, got_frame


def _ctrl_callback(
    ctrl_data: ControllerData, controller: Callable,
    robot_state: RobotState, dt: float
    ) -> Torques:

    b0, b1, b2, b3, t = ctrl_data.read()
    ctrl_data.update_time(dt) # TODO: remove mutexes

    tau = controller(
        dt=dt,
        t=t,
        q=np.array(robot_state.q),
        dq=np.array(robot_state.dq),
        ddq=np.array(robot_state.ddq_d),
        b0=b0,
        b1=b1,
        b2=b2,
        b3=b3
    )

    q_ref = np.array(PandaLimits().q_start)

    KP = np.array([200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0], dtype=np.float32)
    KD = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)

    tau = KP*(q_ref - np.array(robot_state.q)) + KD*(np.zeros_like(robot_state.dq))


    # return Torques(tau.squeeze())
    return tau.squeeze()


def start_inner_loop_threads(
    pipe: pipeline,
    cam_params: tuple[float, float, float, float],
    detector: Detector,
    gripper: Gripper,
    gripper_state: _gripper_state,
    panda: Panda,
    ctx: PandaContext,
    controller: Compiled,
    ctrl_data: ControllerData,
    pose_estimates: PoseEstimates,
    vis: Visualization,
    vis_event: threading.Event,
    gripper_event: threading.Event,
    reset_event: threading.Event,
    exit_event: threading.Event
    ) -> tuple[multiprocessing.Process, threading.Thread, threading.Thread]:

    def panda_ctrl(
        panda: Panda,
        ctrl_data: ControllerData,
        exit_event: threading.Event,
        reset_event: threading.Event
        ):

        panda_ctrl_callback = partial(_ctrl_callback, ctrl_data, controller)
        panda.move_to_start()
        ctrl = controllers.AppliedTorque()
        res = set_current_thread_to_highest_scheduler_priority("couldn't set priority")
        panda.start_controller(ctrl)

        with ctx:
            while ctx.ok() and not exit_event.is_set() and not reset_event.is_set():
                # dt =
                state = panda.get_state()
                tau = _ctrl_callback(ctrl_data, controller, state, 0.001)
                tau = panda_ctrl_callback(state, Duration(1))
                ctrl.set_control(tau)

        panda.stop_controller()
        panda.move_to_start()
        print("Stopped Panda controller...")


    def gripper_state_reader(
        gripper: Gripper,
        gripper_state: _gripper_state,
        ):
        while not exit_event.is_set() and not reset_event.is_set():
            gs = gripper.read_once()
            gripper_state.update(gs)

            sleep(0.1)

    def gripper_ctrl(
            gripper: Gripper,
            exit_event: threading.Event,
            reset_event: threading.Event
        ):

        while not exit_event.is_set() and not reset_event.is_set():
            result = gripper_event.wait(timeout=0.5)
            if not result: continue

            gripper.move(width=0.08, speed=0.05)
            gripper_event.clear()

        gripper.stop()
        print("Stopped Gripper controller...")

    # TODO: rename once I found out which are best as processes/threads
    panda_ctrl_process = multiprocessing.Process(target=panda_ctrl, args=(panda, ctrl_data, exit_event, reset_event))
    gripper_ctrl_thread = threading.Thread(target=gripper_ctrl, args=(gripper, exit_event, reset_event))
    gripper_state_reader_thread = threading.Thread(target=gripper_state_reader, args=(gripper, gripper_state))

    gripper_ctrl_thread.start()
    panda_ctrl_process.start()
    gripper_state_reader_thread.start()

    return panda_ctrl_process, gripper_ctrl_thread, gripper_state_reader_thread


def pick_up_ball(
    gripper: Gripper,
    panda: Panda,
    pre_pick_up_joints: np.ndarray = np.array([2.443083308018442, 0.7390330600388322, -0.12036630424252516, -2.0156737524990036, 0.1133777970969677, 2.6755371474425, 1.4384864832311868]),
    pick_up_joints: np.ndarray = np.array([2.3437016375420385, 0.8570955322834483, -0.033515140057655705, -1.9930821339289344, 0.124799286352258, 2.8073062164783473, 1.3742789834092062])
    ) -> None:

    # panda.move_to_start()
    gripper.move(width=0.08, speed=0.05)
    panda.move_to_joint_position(pre_pick_up_joints)
    panda.move_to_joint_position(pick_up_joints, speed_factor=0.01)

    grasped = False
    while not grasped:
        grasped = gripper.grasp(width=0.045, speed=0.01, force=140, epsilon_inner=0.005, epsilon_outer=0.005)
    print("Grasped:", grasped)

    panda.move_to_joint_position(pre_pick_up_joints, speed_factor=0.005)
    panda.move_to_start()


# WARNING: UNFINISHED
def loop_body(
    pipe: pipeline,
    cam_params: tuple[float, float, float, float],
    detector: Detector,
    env: A_to_B,
    decode_obs: Callable,
    apply_fns: tuple[Callable, ...],
    gripper: Gripper,
    gripper_state: _gripper_state,
    panda: Panda,
    msg: ZeusMessage,
    vis: Visualization,
    msg_event: threading.Event,
    exit_event: threading.Event,
    vis_event: threading.Event,
    gripper_event: threading.Event,
    ctrl_data: ControllerData,
    pose_estimates: PoseEstimates,
    inference_carry: tuple,
    ctrl_time_ns: float = 0.04,
    car_goal_radius: float = 0.1,
    ball_goal_radius: float = 0.1,
    num_agents: int = 2,
    lowpass_filter: LowPassFilter = LowPassFilter(input_shape=_EnvOpts(None).obs_min.shape, history_length=10, bandlimit_hz=0.75, sample_rate_hz=1.0/0.04) # type: ignore[arg-type]
    ) -> tuple[tuple, bool, bool]:

    (
        actors,
        actor_hidden_states,
        previous_done,
        prev_q_car,
        prev_qd_car,
        ball_released,
        p_goal,
        dt,
    ) = inference_carry

    car_pose, floor_pose = pose_estimates.get_data()

    obs, aux = observe(
        env,
        p_goal,
        car_pose.R,
        car_pose.t,
        floor_pose.R,
        floor_pose.t,
        prev_q_car,
        prev_qd_car,
        panda.get_state(),
        gripper_state.read(),
        not ball_released,
        dt
    )

    obs = lowpass_filter(obs)
    (
        q_car, q_arm, q_gripper, p_ball,
        qd_car, qd_arm, qd_gripper, pd_ball,
        p_goal,
        dc_goal,
        dcc_0, dcc_1, dcc_2, dcc_3,
        dgc_0, dgc_1, dgc_2, dgc_3,
        dbc_0, dbc_1, dbc_2, dbc_3,
        db_target
     ) = decode_obs(obs)

    done = logical_or(dc_goal <= car_goal_radius, db_target <= ball_goal_radius)
    # done = logical_and(done, np.array(False))
    print(q_car, dc_goal, db_target)

    actions, actor_hidden_states = policy_inference(
        num_agents,
        apply_fns,
        previous_done,
        actors,
        actor_hidden_states,
        obs,
    )
    a_car, a_arm = actions
    magnitude, angle, rot_vel = a_car
    b_new, a_gripper = a_arm[0:7], a_arm[7]

    # WARNING: temporary
    magnitude = 0.0
    angle = np.mod(-np.pi/2.0, 2*np.pi)
    rot_vel = -1.0

    # Force only releasing once
    a_gripper = 0.0 if ball_released else a_gripper
    ball_released = True if a_gripper >= 0.0 else ball_released

    velocity = rotation_velocity_modifier(magnitude, rot_vel)
    msg.write(float(velocity), float(angle), float(rot_vel), ZeusMode.ACT)
    msg_event.set()

    # TOOD: ctrl_data.update_release_timing(a_gripper)
    ctrl_data.update_ctrl_points(b_new)
    ctrl_data.zero_time()

    # Inner loop
    start_ns = clock_gettime_ns(CLOCK_BOOTTIME)
    while clock_gettime_ns(CLOCK_BOOTTIME) - start_ns < ctrl_time_ns:
        if a_gripper >= 0.0 and not ball_released and (clock_gettime_ns(CLOCK_BOOTTIME) - start_ns)/1e9 >= a_gripper:
            gripper_event.set()

    # frame = pipe.wait_for_frames().get_infrared_frame()
    data, frame, got_frame = apriltag_detection(pipe, cam_params, detector)
    if got_frame:
        pose_estimates.update_data(data)
        vis.update_data(data, frame)

    vis.draw_axes()
    vis.imshow()
    if cv2.waitKey(4) & 0xFF == ord('q'):
        exit_event.set()
        return (None, None, None, None, None, None, None, None), True, True

    return (
        actors,
        actor_hidden_states,
        done,
        q_car,
        qd_car,
        ball_released,
        p_goal,
        clock_gettime_ns(CLOCK_BOOTTIME) - start_ns # dt
    ), bool(done), False



def main() -> None:
    rnn_hidden_size = 16
    rnn_fc_size = 64
    env, decode_obs = setup_env() # TODO: return all env functions I need
    actors, actor_hidden_states, apply_fns = load_policies(env, rnn_hidden_size, rnn_fc_size, "checkpoint_LATEST")
    pipe, cam_params, dist_coeffs, K, R, t, width, height, detector = setup_camera()
    panda, panda_ctx, gripper, desk = setup_panda()
    gripper_state = _gripper_state(gripper.read_once())

    msg = ZeusMessage()
    vis = Visualization(width, height, K, dist_coeffs)
    msg_event = threading.Event()
    vis_event = threading.Event()
    gripper_event = threading.Event()
    reset_event = threading.Event()
    exit_event = threading.Event()

    _loop_body = partial(
        loop_body,
        pipe,
        cam_params,
        detector,
        env,
        decode_obs,
        apply_fns,
        gripper,
        gripper_state,
        panda,
        msg,
        vis,
        msg_event,
        exit_event,
        vis_event,
        gripper_event,
    )

    _ = ControllerData()
    __ = panda.get_state()
    b0, b1, b2, b3, t = _.read()
    controller = jit(partial(arm_spline_tracking_controller, vel_margin=0.05), static_argnames=("vel_margin", )).lower(
        dt=0.001, vel_margin=0.05, t=t, q=np.array(__.q), qd=np.array(__.dq), qdd=np.array(__.ddq_d), b0=b0, b1=b1, b2=b2, b3=b3
    ).compile()

    websocket_client_thread = threading.Thread(target=websocket_client, args=(msg, msg_event, exit_event))
    websocket_client_thread.start()

    print("Starting inference...")
    count = 1
    all_rollouts_done = False
    while not all_rollouts_done: # TODO: exit_event.is_set()
        print("Resetting before rollout...")
        reset_event.clear()

        ctrl_data = ControllerData()
        pose_estimates = PoseEstimates()

        loop_body_fn = partial(
            _loop_body,
            ctrl_data,
            pose_estimates
        )

        # Need to do a warm up run before starting inner loop threads
        if count > 1:
            pick_up_ball(gripper, panda)

            panda_ctrl_process, gripper_ctrl_thread, gripper_state_reader_thread = start_inner_loop_threads(
                pipe,
                cam_params,
                detector,
                gripper,
                gripper_state,
                panda,
                panda_ctx,
                controller,
                ctrl_data,
                pose_estimates,
                vis,
                vis_event,
                gripper_event,
                reset_event,
                exit_event
            )

        p_goal = reset_goal_pos(env.car_limits.x_min, env.car_limits.x_max, env.car_limits.y_min, env.car_limits.y_max)
        print("p_goal:", p_goal)

        inference_carry = (
            actors,
            actor_hidden_states,
            np.array([True]),               # previous_done
            np.zeros(3, dtype=np.float32),  # prev_q_car
            np.zeros(3, dtype=np.float32),  # prev_qd_car
            False,                          # ball_released
            p_goal,
            99999.9                         # dt
        )

        print("Running rollout...")
        rollout_count = 0
        current_rollout_done = False
        while not current_rollout_done and rollout_count < 100_000_000:

            # We catch all exceptions to be able to clean up some stuff (had problems with Ubuntu hanging)
            try:
                inference_carry, current_rollout_done, all_rollouts_done = loop_body_fn(inference_carry)
            except Exception as e:
                exit_event.set()
                current_rollout_done, all_rollouts_done = True, True
                print(f"Exception caught in loop_body(), stopping...\n\n{format_exc()}")

            rollout_count += 1

        print("Rollout done.")

        # Stop arm and car
        reset_event.set()
        msg.write(0, 0, 0, ZeusMode.STANDBY)
        msg_event.set()

        exit_event.set()

        if count > 1:
            panda_ctrl_process.join()
            detection_thread.join()
            gripper_ctrl_thread.join()
            gripper_state_reader_thread.join()

        all_rollouts_done = True if count >= 4 else False # TODO: proper logic for multiple rollouts
        count += 1

        msg_event.clear()
        vis_event.clear()
        gripper_event.clear()
        reset_event.clear()
        exit_event.clear() # TODO: shouldn't clear exit

    exit_event.set()
    pipe.stop()
    websocket_client_thread.join()

    # Relinquish control of Panda
    panda.move_to_start()
    panda.get_robot().stop()
    desk.deactivate_fci()
    desk.release_control()



if __name__ == "__main__":
    main()
