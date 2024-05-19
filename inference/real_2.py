from functools import partial
import threading
import numpy as np
import cv2
from typing import Callable, NamedTuple, TypeAlias
from panda_py import Panda, PandaContext
from panda_py.libfranka import RobotState, Duration, Gripper
from websockets.sync.client import connect
from enum import IntEnum
from time import clock_gettime_ns, CLOCK_BOOTTIME
from json import dumps
from environments.physical import PandaLimits
from inference.setup_for_inference import setup_camera, setup_env, load_policies, setup_panda, observe, policy_inference
from inference.processing import LowPassFilter
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
    b0: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    b3: np.ndarray

    t: float = 0.0
    mtx = threading.Lock()

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
    frame: np.ndarray
    K: np.ndarray
    dist_coeffs: np.ndarray

    data: list[VizData] = [VizData(), VizData()]
    mtx = threading.Lock()

    def __init__(self, w, h, K, dist_coeffs) -> None:
        self.frame = np.zeros((h, w), dtype=np.uint8)
        self.K = K
        self.dist_coeffs = dist_coeffs

    def update_data(self, data) -> None:
        with self.mtx:
            self.data = data

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
            msg_event.wait()
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
    ) -> list[VizData | None]:

    frame = pipe.wait_for_frames().get_infrared_frame()
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

    return viz_data


# WARNING: temporary
def panda_ctrl_callback(robot_state: RobotState, duration: Duration):
    q_ref = PandaLimits.q_start
    KP = np.array([200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0], dtype=np.float32)
    KD = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)

    tau = KP*(q_ref - robot_state.q) + KD*(np.zeros_like(robot_state.dq))

    return tau


def start_inner_loop_threads(
    pipe: pipeline,
    cam_params: tuple[float, float, float, float],
    detector: Detector,
    panda: Panda,
    ctx: PandaContext,
    panda_ctrl_callback: Callable[[RobotState, Duration], np.ndarray],
    pose_estimates: PoseEstimates,
    vis: Visualization,
    vis_event: threading.Event,
    detection_event: threading.Event,
    reset_event: threading.Event,
    exit_event: threading.Event
    ) -> tuple[threading.Thread, threading.Thread]:

    def tag_detection():
        while not exit_event.is_set():
            detection_event.wait()
            data = apriltag_detection(pipe, cam_params, detector)
            pose_estimates.update_data(data)
            vis.update_data(data)
            vis_event.set()
            detection_event.clear()

    def panda_ctrl():
        panda.move_to_start()
        with ctx:
            while ctx.ok() and not exit_event.is_set() and not reset_event.is_set():
                panda.control(panda_ctrl_callback)

            panda.stop_controller()
            panda.move_to_start()

    # TODO: implement
    def gripper_ctrl():
        pass

    detection_thread = threading.Thread(target=tag_detection, daemon=True)
    panda_ctrl_thread = threading.Thread(target=panda_ctrl)

    detection_thread.start()
    panda_ctrl_thread.start()

    return detection_thread, panda_ctrl_thread


def start_viz_daemon_threads(
        vis: Visualization,
        vis_event: threading.Event
    ) -> None:

    def show():
        while True:
            vis.imshow()

    def update():
        while True:
            vis_event.wait()
            vis.draw_axes()
            vis_event.clear()

    show_thread = threading.Thread(target=show, daemon=True)
    update_thread = threading.Thread(target=update, daemon=True)

    show_thread.start()
    update_thread.start()

# WARNING: UNFINISHED
def loop_body(
    env: A_to_B,
    panda: Panda,
    gripper: Gripper,
    ctrl_data: ControllerData,
    pose_estimates: PoseEstimates,
    msg: ZeusMessage,
    vis: Visualization,
    msg_event: threading.Event,
    exit_event: threading.Event,
    vis_event: threading.Event,
    detection_event: threading.Event,
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

    obs, aux = observe(
        env,
        p_goal,
        pose_estimates,
        prev_q_car,
        prev_qd_car,
        panda.get_state(),
        gripper.read_once(),
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
     ) = env.decode_observation(obs)

    done = (dc_goal <= car_goal_radius) or (db_target <= ball_goal_radius)

    actions, actor_hidden_states = policy_inference(
        num_agents,
        actors,
        actor_hidden_states,
        obs,
        previous_done
    )
    a_car, a_arm = actions
    magnitude, angle, rot_vel = a_car
    b_new, a_gripper = a_arm[0:7], a_arm[7]

    # Force only releasing once
    a_gripper = 0.0 if ball_released else a_gripper
    ball_released = True if a_gripper >= 0.0 else ball_released

    msg.write(magnitude, angle, rot_vel, ZeusMode.ACT)
    msg_event.set()

    # TOOD: ctrl_data.update_release_timing(a_gripper)
    ctrl_data.update_ctrl_points(b_new)
    ctrl_data.zero_time()

    # Inner loop
    start_ns = clock_gettime_ns(CLOCK_BOOTTIME)
    while clock_gettime_ns(CLOCK_BOOTTIME) - start_ns < ctrl_time_ns:
        detection_event.set()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit_event.set()
        return (None, None, None, None, None, None), True, True # type: ignore[assignment]

    return (
        actors,
        actor_hidden_states,
        done,
        q_car,
        qd_car,
        ball_released,
        p_goal,
        clock_gettime_ns(CLOCK_BOOTTIME) - start_ns
    ), bool(done), False

# TODO: jit
def rotation_velocity_modifier(velocity, omega):
    return np.clip(velocity - np.abs(omega), 0.0, velocity)


def main() -> None:
    rnn_hidden_size = 16
    rnn_fc_size = 64
    env = setup_env() # TODO: return all env functions I need
    actors, actor_hidden_states = load_policies(env, rnn_hidden_size, rnn_fc_size, "checkpoint_LATEST")
    pipe, cam_params, dist_coeffs, K, R, t, width, height, detector = setup_camera()
    panda, panda_ctx, gripper = setup_panda()

    msg = ZeusMessage()
    vis = Visualization(width, height, K, dist_coeffs)
    ctrl_data = ControllerData()
    msg_event = threading.Event()
    vis_event = threading.Event()
    detection_event = threading.Event()
    reset_event = threading.Event()
    exit_event = threading.Event()

    loop_body_fn = partial(
        loop_body,
        env,
        panda,
        gripper,
        ctrl_data,
        PoseEstimates(),
        msg,
        vis,
        msg_event,
        exit_event,
        vis_event,
        detection_event
    )

    websocket_client_thread = threading.Thread(target=websocket_client, args=(msg, msg_event, exit_event))
    websocket_client_thread.start()
    start_viz_daemon_threads(vis, vis_event)

    all_rollouts_done = False
    while not all_rollouts_done:
        reset_event.clear()

        detection_thread, panda_ctrl_thread = start_inner_loop_threads(
            pipe,
            cam_params,
            detector,
            panda,
            panda_ctx,
            panda_ctrl_callback,
            PoseEstimates(),
            vis,
            vis_event,
            detection_event,
            reset_event,
            exit_event
        )

        inference_carry = (actors, actor_hidden_states, True)
        current_rollout_done = False
        while not current_rollout_done:

            inference_carry, current_rollout_done, all_rollouts_done = loop_body_fn(inference_carry)

        reset_event.set()
        all_rollouts_done = True # TODO: logic for multiple rollouts
        detection_thread.join()
        panda_ctrl_thread.join()

    websocket_client_thread.join()


if __name__ == "__main__":
    main()
