from torch import set_num_threads, set_num_interop_threads
set_num_threads(1)
set_num_interop_threads(1)
import torch.multiprocessing as multiprocessing
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

import os
import threading

from multiprocessing.sharedctypes import SynchronizedBase, Value
from multiprocessing.synchronize import Event as mp_Event
from jax import jit
from jax._src.stages import Compiled
import numpy as np
import cv2
from traceback import format_exc
from functools import partial
from jax.numpy import logical_and, logical_or
from typing import Callable, NamedTuple, TypeAlias
from panda_py import Panda, PandaContext, controllers
from panda_py.libfranka import RobotState, Gripper, GripperState, set_current_thread_to_highest_scheduler_priority
from websockets.sync.client import connect
from enum import IntEnum
from time import clock_gettime_ns, CLOCK_BOOTTIME, sleep
from json import dumps
from environments.physical import PandaLimits
from inference.setup_for_inference import (
        setup_env,
        setup_camera,
        setup_yolo,
        load_policies,
        setup_panda,
        observe,
        policy_inference,
        reset_goal_pos,
        rotation_velocity_modifier,
        ball_pos_cam_frame,
)
from inference.processing import LowPassFilter
from inference.controllers import arm_spline_tracking_controller
from environments.A_to_B_jax import A_to_B
from environments.options import EnvironmentOptions as _EnvOpts
from pyrealsense2 import pipeline
from pupil_apriltags import Detector, Detection


ZEUS_URI = "ws://192.168.4.1:8765"
BALL_DETECTION_CPUS = {0}
PANDA_CPUS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11}

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

# could refactor this to use shared buffer arrays and values for compatability with not using os.fork()
class ControllerData:
    mtx = multiprocessing.RLock()

    def __init__(
        self,
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

    def update_time(self, duration: float) -> None:
        with self.mtx:
            self.t = min(self.t + duration, 1.0)

    def zero_time(self) -> None:
        with self.mtx:
            self.t = 0.0

    def read(self):
        with self.mtx:
            return self.b0, self.b1, self.b2, self.b3, self.t

    def reset(self):
        with self.mtx:
            self.b0 = np.array(PandaLimits().q_start)
            self.b1 = np.array(PandaLimits().q_start)
            self.b2 = np.array(PandaLimits().q_start)
            self.b3 = np.array(PandaLimits().q_start)
            self.t = 0.0


class _gripper_state:
    mtx = threading.Lock()
    state: GripperState
    gripping = Value("b", True)

    def __init__(self, state: GripperState) -> None:
        self.state = state

    def update(self, state: GripperState) -> None:
        with self.mtx:
            self.state = state

    def read(self) -> GripperState:
        with self.mtx:
            return self.state

def _ball_pos_shared_array(dims=3) -> np.ndarray:
    buffer = multiprocessing.Array("f", dims)
    arr = np.frombuffer(buffer.get_obj(), dtype=np.float32)
    np.copyto(arr, np.zeros(dims, dtype=np.float32))

    return arr

def _latest_frame_shared_array(width: int, height: int, channels: int | None=None) -> np.ndarray:
    if channels is not None:
        buffer = multiprocessing.Array("B", width*height*channels)
        arr = np.frombuffer(buffer.get_obj(), dtype=np.uint8).reshape((height, width, channels))
        np.copyto(arr, np.zeros((height, width, channels), dtype=np.uint8))
    else:
        buffer = multiprocessing.Array("B", width*height)
        arr = np.frombuffer(buffer.get_obj(), dtype=np.uint8).reshape((height, width))
        np.copyto(arr, np.zeros((height, width), dtype=np.uint8))

    return arr

class Pose(NamedTuple):
    R: np.ndarray = np.eye(3)
    t: np.ndarray = np.zeros((3, 1))

class PoseEstimates:
    mtx = threading.Lock()

    car_pose = Pose()
    floor_pose = Pose()

    def update_poses(self, data) -> None:
        with self.mtx:
            self.car_pose = data[0] if data[0] is not None else self.car_pose
            self.floor_pose = data[1] if data[1] is not None else self.floor_pose

    def get_data(self) -> tuple[Pose, Pose]:
        with self.mtx:
            return self.car_pose, self.floor_pose

    def reset(self) -> None:
        with self.mtx:
            self.car_pose = Pose()
            self.floor_pose = Pose()

VizData: TypeAlias = Pose

class Visualization:
    mtx = threading.Lock()
    p_mtx = multiprocessing.Lock()

    _clean_frame: np.ndarray
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
        exit_event: mp_Event,
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


def _ctrl_fn(
    controller: Callable,
    ctrl_data: ControllerData,
    robot_state: RobotState,
    dt: float
    ) -> np.ndarray:

    b0, b1, b2, b3, t = ctrl_data.read()
    ctrl_data.update_time(dt)

    # TODO: fix qd vs dq naming
    tau = controller(
        dt=dt,
        t=t,
        q=np.array(robot_state.q),
        qd=np.array(robot_state.dq),
        qdd=np.array(robot_state.ddq_d),
        b0=b0,
        b1=b1,
        b2=b2,
        b3=b3
    )

    q_ref = np.array(PandaLimits().q_start)

    KP = np.array([200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0], dtype=np.float32)
    KD = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)

    tau = KP*(q_ref - np.array(robot_state.q)) + KD*(np.zeros_like(robot_state.dq) - np.array(robot_state.dq))

    return tau.squeeze()


def start_inner_control_loops(
    gripper: Gripper,
    gripper_state: _gripper_state,
    panda: Panda,
    ctx: PandaContext,
    controller: Compiled,
    ctrl_data: ControllerData,
    gripper_event: threading.Event,
    ) -> threading.Thread:

    def gripper_state_reader(
        gripper: Gripper,
        gripper_state: _gripper_state,
        ):
        print("Starting Gripper state reader daemon thread...")
        while True:
            gripper_state.update(gripper.read_once())
            sleep(0.5)

    def gripper_ctrl(
            gripper: Gripper,
            gripper_event: threading.Event,
        ):
        print("Starting Gripper control daemon thread...")
        while True:
            result = gripper_event.wait(timeout=0.5)
            if not result: continue

            gripper.move(width=0.08, speed=0.05)
            gripper_event.clear()

    gripper_ctrl_thread = threading.Thread(target=gripper_ctrl, args=(gripper, gripper_event), daemon=True)
    gripper_ctrl_thread.start()

    gripper_state_reader_thread = threading.Thread(target=gripper_state_reader, args=(gripper, gripper_state), daemon=True)
    gripper_state_reader_thread.start()

    return gripper_ctrl_thread

# needs to be defined at module level
def estimate_ball_pos(
    gripping: SynchronizedBase,
    ball_pos: np.ndarray,
    latest_frame: np.ndarray,
    process_ready_event: mp_Event
    ) -> None:
    print("Starting ball observation process...")

    os.sched_setaffinity(0, BALL_DETECTION_CPUS)
    _ball_pos_cam_frame = ball_pos_cam_frame.lower(np.ones(4)).compile()

    detect_ball = setup_yolo()

    no_detects = 0
    max_no_detects = 200
    process_ready_event.set()
    while no_detects < max_no_detects:
        if gripping:
            sleep(0.08)
            continue

        result = detect_ball(latest_frame)

        if result.shape[0] == 0:
            no_detects += 1
            if no_detects >= max_no_detects:
                print("Aborting ball observation process...")
                exit(1)
            sleep(0.08)
            continue

        np.copyto(ball_pos, _ball_pos_cam_frame(result))
        print("Ball position:", ball_pos)
        no_detects = 0
        sleep(0.08)


# TODO: move directly to main
def start_ball_observation_process(
    gripping: SynchronizedBase,
    ball_pos: np.ndarray, # shared buffer array
    latest_frame: np.ndarray, # shared buffer array
    reset_event: mp_Event,
    exit_event: mp_Event,
    process_ready_event: mp_Event
    ) -> multiprocessing.Process:

    ball_obs_process = multiprocessing.Process(
        target=estimate_ball_pos,
        args=(
            gripping,
            ball_pos,
            latest_frame,
            process_ready_event
        ),
        daemon=True
    )
    ball_obs_process.start()
    process_ready_event.wait()
    process_ready_event.clear()

    return ball_obs_process


def pick_up_ball(
    gripper: Gripper,
    panda: Panda,
    pre_pick_up_joints_0: np.ndarray = np.array([2.741776827299804, 0.16191385076966203, -0.3059633437904001, -1.2027111798002006, 0.10539678035842047, 1.3743739019785552, 1.6465634071555602]),
    pre_pick_up_joints_1: np.ndarray = np.array([2.727414517997034, 0.8530083795133824, -0.25646261388795416, -1.8753472193500451, 0.30701742740697596, 2.7107548903624212, 1.5277980511732814]),
    pick_up_joints: np.ndarray = np.array([2.6813547066882646, 1.2087113865530301, -0.2518414849705166, -1.5724787159137747, 0.30743288654751244, 2.7203699969450628, 1.5277652654397402]),
    post_pick_up_joints_0: np.ndarray = np.array([2.727414517997034, 0.8530083795133824, -0.25646261388795416, -1.8753472193500451, 0.30701742740697596, 2.7107548903624212, 1.5277980511732814]),
    post_pick_up_joints_1: np.ndarray = np.array([2.741776827299804, 0.16191385076966203, -0.3059633437904001, -1.2027111798002006, 0.10539678035842047, 1.3743739019785552, 1.6465634071555602]),
    post_pick_up_joints_2: np.ndarray = np.array([1.6953011181718531, -0.5849034175014706, -0.09494162156490818, -1.9984340891085173, -0.03546912060512437, 1.467547558578196, 0.8601144378731647]
),
    ) -> None:

    gripper.move(width=0.08, speed=0.05)
    panda.move_to_joint_position(pre_pick_up_joints_0)
    panda.move_to_joint_position(pre_pick_up_joints_1)
    panda.move_to_joint_position(pick_up_joints, speed_factor=0.1)

    grasped = False
    while not grasped:
        grasped = gripper.grasp(width=0.045, speed=0.01, force=140, epsilon_inner=0.005, epsilon_outer=0.005)
    print("Grasped:", grasped)

    panda.move_to_joint_position(post_pick_up_joints_0, speed_factor=0.1)
    panda.move_to_joint_position(post_pick_up_joints_1)
    panda.move_to_joint_position(post_pick_up_joints_2)
    panda.move_to_start()


def loop_body(
    # partial() before all rollouts
    pipe: pipeline,
    cam_params: tuple[float, float, float, float],
    detector: Detector,
    env: A_to_B,
    decode_obs: Callable,
    apply_fns: tuple[Callable, ...],
    gripper: Gripper,
    gripper_state: _gripper_state,
    panda: Panda,
    panda_low_level_ctrl: Callable,
    ball_pos: np.ndarray, # shared buffer array
    latest_frame: np.ndarray, # shared buffer array
    msg: ZeusMessage,
    vis: Visualization,
    msg_event: threading.Event,
    gripper_event: threading.Event,
    exit_event: mp_Event,

    # Argument to loop_body() during rollouts
    inference_carry: tuple,
    torque_controller: controllers.AppliedTorque,
    ctrl_data: ControllerData,
    pose_estimates: PoseEstimates,

    # Constants
    ctrl_time_ns: float = 0.04,
    car_goal_radius: float = 0.1,
    ball_goal_radius: float = 0.1,
    num_agents: int = 2,
    lowpass_filter: LowPassFilter = LowPassFilter(input_shape=_EnvOpts(None).obs_min.shape, history_length=10, bandlimit_hz=0.75, sample_rate_hz=1.0/0.04) # type: ignore[arg-type]
    ) -> tuple[tuple, bool, bool]:

    start_ns = clock_gettime_ns(CLOCK_BOOTTIME)
    atleast_once = True

    (
        actors,
        actor_hidden_states,
        previous_done,
        prev_q_car,
        prev_qd_car,
        prev_p_ball,
        prev_pd_ball,
        ball_released,
        p_goal,
        dt,
        warmup
    ) = inference_carry

    car_pose, floor_pose = pose_estimates.get_data()

    robot_state = panda.get_state()
    obs, aux = observe(
        env,
        p_goal,
        car_pose.R,
        car_pose.t,
        floor_pose.R,
        floor_pose.t,
        prev_q_car,
        prev_qd_car,
        robot_state,
        gripper_state.read(),
        not ball_released,
        ball_pos,
        prev_p_ball,
        prev_pd_ball,
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

    print(p_ball)
    print(pd_ball)

    done = logical_or(dc_goal <= car_goal_radius, db_target <= ball_goal_radius)
    # print(dc_goal, q_car[0:2], p_goal)
    # print(db_target, p_ball, p_goal)
    # done = logical_and(done, np.array(False))

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

    velocity = rotation_velocity_modifier(magnitude, rot_vel)

    # Send actions to agents
    msg.write(float(velocity), float(angle), float(rot_vel), ZeusMode.ACT)
    msg_event.set()
    ctrl_data.update_ctrl_points(b_new) # todo: remove mutexes now that we're not async for this
    ctrl_data.zero_time() # todo: remove mutexes now that we're not async for this

    # Force only releasing once
    a_gripper = 0.0 if ball_released else a_gripper
    ball_released = True if a_gripper >= 0.0 else ball_released
    gripper_state.gripping.value = not ball_released

    # Inner loop
    t_ns = clock_gettime_ns(CLOCK_BOOTTIME)
    while t_ns - start_ns < ctrl_time_ns or atleast_once:
        atleast_once = False

        # Gripper release logic
        if a_gripper >= 0.0 and not ball_released and (t_ns - start_ns)/1e9 >= a_gripper:
            gripper_event.set()

        if not warmup:
            state = panda.get_state()
            dt = (clock_gettime_ns(CLOCK_BOOTTIME) - t_ns) / 1e9
            tau = panda_low_level_ctrl(ctrl_data, state, dt)
            torque_controller.set_control(tau)

        t_ns = clock_gettime_ns(CLOCK_BOOTTIME)


    data, frame, got_frame = apriltag_detection(pipe, cam_params, detector)
    if got_frame:
        pose_estimates.update_poses(data)
        vis.update_data(data, frame)
        np.copyto(latest_frame, frame)

    vis.draw_axes()
    vis.imshow()
    if cv2.waitKey(4) & 0xFF == ord('q'):
        exit_event.set()
        return (None, None, None, None, None, None, None, None), True, True

    dt = (clock_gettime_ns(CLOCK_BOOTTIME) - start_ns)/1e9

    return (
        actors,
        actor_hidden_states,
        done,
        q_car,
        qd_car,
        p_ball,
        pd_ball,
        ball_released,
        p_goal,
        dt,
        warmup
    ), bool(done), False


def main() -> None:
    os.sched_setaffinity(0, {5})

    rnn_hidden_size = 16
    rnn_fc_size = 64
    env, decode_obs = setup_env() # TODO: return all env functions I need
    actors, actor_hidden_states, apply_fns = load_policies(env, rnn_hidden_size, rnn_fc_size, "checkpoint_LATEST")
    pipe, cam_params, dist_coeffs, K, R, t, width, height, detector = setup_camera()
    panda, panda_ctx, gripper, desk = setup_panda()

    gripper_state = _gripper_state(gripper.read_once())
    torque_controller = controllers.AppliedTorque()
    # set_current_thread_to_highest_scheduler_priority("couldn't set priority")

    # Shared memory arrays
    ball_pos = _ball_pos_shared_array(3)
    latest_frame = _latest_frame_shared_array(width, height)

    msg = ZeusMessage()
    vis = Visualization(width, height, K, dist_coeffs)
    msg_event = threading.Event()
    gripper_event = threading.Event()
    reset_event = multiprocessing.Event()
    exit_event = multiprocessing.Event()
    process_ready_event = multiprocessing.Event()

    _ = ControllerData()
    __ = panda.get_state()
    b0, b1, b2, b3, t = _.read()
    # TODO: fix qd vs dq naming
    controller = jit(partial(arm_spline_tracking_controller, vel_margin=0.05), static_argnames=("vel_margin", )).lower(
        dt=0.001, vel_margin=0.05, t=t, q=np.array(__.q), qd=np.array(__.dq), qdd=np.array(__.ddq_d), b0=b0, b1=b1, b2=b2, b3=b3
    ).compile()

    # TODO: pass to partial
    ctrl_data = ControllerData()
    pose_estimates = PoseEstimates()

    _loop_body = partial(
        loop_body,
        pipe=pipe,
        cam_params=cam_params,
        detector=detector,
        env=env,
        decode_obs=decode_obs,
        apply_fns=apply_fns,
        gripper=gripper,
        gripper_state=gripper_state,
        panda=panda,
        panda_low_level_ctrl=partial(_ctrl_fn, controller),
        ball_pos=ball_pos,
        latest_frame=latest_frame,
        msg=msg,
        vis=vis,
        msg_event=msg_event,
        gripper_event=gripper_event,
        exit_event=exit_event
    )

    websocket_client_thread = threading.Thread(target=websocket_client, args=(msg, msg_event, exit_event))
    websocket_client_thread.start()

    # TODO: start directly here
    ball_detection_process = start_ball_observation_process(
        gripper_state.gripping,
        ball_pos,
        latest_frame,
        reset_event,
        exit_event,
        process_ready_event,
    )

    # TODO: rename and start directly here
    gripper_ctrl_thread = start_inner_control_loops(
        gripper,
        gripper_state,
        panda,
        panda_ctx,
        controller,
        ctrl_data,
        gripper_event,
    )

    print("Starting inference...")
    count = 1
    all_rollouts_done = False
    while not all_rollouts_done and not exit_event.is_set():
        print("Resetting before rollout...")
        gripper_event.clear()
        reset_event.clear()

        ctrl_data.reset()
        pose_estimates.reset()

        p_goal = reset_goal_pos(env.car_limits.x_min, env.car_limits.x_max, env.car_limits.y_min, env.car_limits.y_max)
        print("p_goal:", p_goal)

        inference_carry = (
            actors,
            actor_hidden_states,
            np.array([True]),               # previous_done
            np.zeros(3, dtype=np.float32),  # prev_q_car
            np.zeros(3, dtype=np.float32),  # prev_qd_car
            np.zeros(3, dtype=np.float32),  # prev_p_ball
            np.zeros(3, dtype=np.float32),  # prev_pd_ball
            False,                          # ball_released
            p_goal,
            99999.9,                        # dt
            count == 1                      # warmup
        )

        # Need to do a warm up run before starting inner loop threads
        if count > 1:
            pick_up_ball(gripper, panda)
            panda.start_controller(torque_controller)

        print("Running rollout...")
        rollout_count = 0
        current_rollout_done = False
        with panda_ctx:
            while not current_rollout_done and rollout_count < 10_000_000 and panda_ctx.ok():

                # We catch all exceptions to be able to clean up some stuff (had problems with Ubuntu hanging)
                try:
                    inference_carry, current_rollout_done, all_rollouts_done = _loop_body(
                        inference_carry=inference_carry,
                        torque_controller=torque_controller,
                        ctrl_data=ctrl_data,
                        pose_estimates=pose_estimates,
                    )

                    # Warmup
                    if count == 1 and rollout_count < 5:
                        current_rollout_done = False
                    elif count == 1 and rollout_count >= 5:
                        current_rollout_done = True

                except Exception as e:

                    exit_event.set()
                    current_rollout_done, all_rollouts_done = True, True
                    print(f"Exception caught in loop_body(), stopping...\n\n{format_exc()}")

                rollout_count += 1

        print("Rollout done.")
        msg.write(0, 0, 0, ZeusMode.STANDBY)
        msg_event.set()
        panda.stop_controller()
        reset_event.set()
        cv2.destroyAllWindows()

        all_rollouts_done = True if count >= 4 else False # TODO: proper logic for multiple rollouts
        count += 1

    cv2.destroyAllWindows()
    exit_event.set()
    pipe.stop()
    websocket_client_thread.join(timeout=10)
    gripper_ctrl_thread.join(timeout=10)
    ball_detection_process.join(timeout=10)

    # Relinquish control of Panda
    panda.move_to_start()
    panda.get_robot().stop()
    desk.deactivate_fci()
    desk.release_control()



if __name__ == "__main__":
    main()
