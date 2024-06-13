from jax.random import PRNGKey, split
import torch.multiprocessing as multiprocessing
from torch import set_num_threads, set_num_interop_threads
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

import os
import threading
import reproducibility_globals

from multiprocessing.sharedctypes import SynchronizedBase, Value
from multiprocessing.synchronize import Event as mp_Event
from jax import jit
from jax._src.stages import Compiled
import numpy as np
import cv2
from traceback import format_exc
from functools import partial
from jax.numpy import logical_and, logical_or, cos, sin
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
        rotational_velocity_modifier,
        ball_pos_cam_frame,
        action_to_reference,
        reference_to_torque,
        observe_ball_ballistic,
        pd_control_to_start
)
from inference.processing import LowPassFilter
from inference.controllers import arm_spline_tracking_controller
from environments.A_to_B_jax import A_to_B
from environments.options import EnvironmentOptions as _EnvOpts
from pyrealsense2 import pipeline
from pupil_apriltags import Detector, Detection


ZEUS_URI = "ws://192.168.4.1:8765"
BALL_DETECTION_CPUS = {0} #, 1, 2, 3}
PANDA_CPUS = {1, 2, 3, 4, 5, 6}

set_num_threads(len(BALL_DETECTION_CPUS))
set_num_interop_threads(len(BALL_DETECTION_CPUS))


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

def _ball_pos_shared_array(dims=3) -> tuple[np.ndarray, SynchronizedBase]:
    buffer = multiprocessing.Array("f", dims)
    arr = np.frombuffer(buffer.get_obj(), dtype=np.float32)
    np.copyto(arr, np.zeros(dims, dtype=np.float32))

    return arr, buffer

def _latest_frame_shared_array(width: int, height: int, channels: int | None=None) -> tuple[np.ndarray, SynchronizedBase]:
    if channels is not None:
        buffer = multiprocessing.Array("B", width*height*channels)
        arr = np.frombuffer(buffer.get_obj(), dtype=np.uint8).reshape((height, width, channels))
        np.copyto(arr, np.zeros((height, width, channels), dtype=np.uint8))
    else:
        buffer = multiprocessing.Array("B", width*height)
        arr = np.frombuffer(buffer.get_obj(), dtype=np.uint8).reshape((height, width))
        np.copyto(arr, np.zeros((height, width), dtype=np.uint8))

    return arr, buffer

class Pose(NamedTuple):
    R: np.ndarray = np.eye(3)
    t: np.ndarray = np.zeros((3, 1))

class PoseEstimates:
    # mtx = threading.Lock()

    car_pose = Pose()
    floor_pose = Pose()

    def update_poses(self, data, warmup: bool=True) -> None:
        # with self.mtx:
        self.car_pose = data[0] if data[0] is not None else self.car_pose
        if warmup:
            self.floor_pose = data[1] if data[1] is not None else self.floor_pose

    def get_data(self) -> tuple[Pose, Pose]:
        # with self.mtx:
        return self.car_pose, self.floor_pose

    def reset(self) -> None:
        # with self.mtx:
        self.car_pose = Pose()
        self.floor_pose = Pose()

VizData: TypeAlias = Pose

class Visualization:
    # mtx = threading.Lock()
    # p_mtx = multiprocessing.Lock()

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
        # with self.mtx:
        self.data[0] = data[0] if data[0] is not None else self.data[0]
        self.data[1] = data[1] if data[1] is not None else self.data[1]
        self.frame = frame

    def draw_axes(self) -> None:
        # with self.mtx:
        for data in self.data:
            if data is None: continue
            rotV, _ = cv2.Rodrigues(data.R)
            points = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 0.2], [0, 0, 0]], dtype=np.float32).reshape(-1, 3) # z axis is drawn inverted
            axisPoints, _ = cv2.projectPoints(points, rotV, data.t, self.K, self.dist_coeffs)
            axisPoints = axisPoints.astype(int)
            self.frame = cv2.line(self.frame, axisPoints[3].ravel(), tuple(axisPoints[0].ravel()), (255,255,255), 3)
            self.frame = cv2.line(self.frame, axisPoints[3].ravel(), tuple(axisPoints[1].ravel()), (0,0,0), 3)
            self.frame = cv2.line(self.frame, axisPoints[3].ravel(), tuple(axisPoints[2].ravel()), (0,0,0), 3)

    def draw_ball(self, p_ball: np.ndarray) -> None:
        offset = np.array([-0.1575, 0.0, -0.095])
        # with self.mtx:
        if self.data[1] is not None:
            data = self.data[1]
            rotV, _ = cv2.Rodrigues(data.R)
            shifted_p_ball = p_ball + offset
            ball_pos_3d = np.array([shifted_p_ball], dtype=np.float32).reshape(-1, 3)
            ball_pos_2d, _ = cv2.projectPoints(ball_pos_3d, rotV, data.t, self.K, self.dist_coeffs)
            ball_pos_2d = ball_pos_2d.astype(int)
            self.frame = cv2.circle(self.frame, tuple(ball_pos_2d.ravel()), 15, (200, 200, 75), 2)

    def circle_points(self, p: np.ndarray, r: float, N = 10) -> np.ndarray:
        points = np.array([
                r*np.cos(np.linspace(0, 2*np.pi, N)) + p[0],
                r*np.sin(np.linspace(0, 2*np.pi, N)) + p[1],
                0.0*np.ones(N)
            ], dtype=np.float32).T

        return points

    def draw_goal(self, p_goal: np.ndarray) -> None:
        offset = np.array([-0.1575+0.075, 0.0])
        # with self.mtx:
        if self.data[1] is not None:
            data = self.data[1]
            rotV, _ = cv2.Rodrigues(data.R)
            goal_pos_3d = self.circle_points(p_goal + offset, 0.1)
            goal_pos_3d = np.array([goal_pos_3d], dtype=np.float32).reshape(-1, 3)
            goal_pos_2d, _ = cv2.projectPoints(goal_pos_3d, rotV, data.t, self.K, self.dist_coeffs)
            goal_pos_2d = goal_pos_2d.astype(int)
            self.frame = cv2.polylines(self.frame, [goal_pos_2d], True, (75, 200, 75), 2)

    def draw_car_action(self, orientation: float, magnitude: float, angle: float) -> None:
        # draw an arrow representing the car action from the car reference frame
        # also draw and arrow representing the car action from the world reference frame
        if self.data[0] is not None:
            data = self.data[0]
            rotV, _ = cv2.Rodrigues(data.R)
            vx_local = magnitude*cos(angle)
            vy_local = magnitude*sin(angle)
            arrow_3d = np.array([[0, 0, 0], [0.2*vx_local, 0.2*vy_local, 0]], dtype=np.float32).reshape(-1, 3)
            arrow_2d, _ = cv2.projectPoints(arrow_3d, rotV, data.t, self.K, self.dist_coeffs)
            arrow_2d = arrow_2d.astype(int)
            self.frame = cv2.arrowedLine(self.frame, tuple(arrow_2d[0].ravel()), tuple(arrow_2d[1].ravel()), (255, 0, 0), 2)

        if self.data[1] is not None:
            data = self.data[1]
            rotV, _ = cv2.Rodrigues(data.R)
            vx_local = magnitude*cos(angle)
            vy_local = magnitude*sin(angle)
            vx_world = vx_local*cos(orientation) - vy_local*sin(orientation)
            vy_world = vx_local*sin(orientation) + vy_local*cos(orientation)
            arrow_3d = np.array([[0, 0, 0], [0.1*vx_world, 0.1*vy_world, 0]], dtype=np.float32).reshape(-1, 3)
            arrow_2d, _ = cv2.projectPoints(arrow_3d, rotV, data.t, self.K, self.dist_coeffs)
            arrow_2d = arrow_2d.astype(int)
            self.frame = cv2.arrowedLine(self.frame, tuple(arrow_2d[0].ravel()), tuple(arrow_2d[1].ravel()), (0, 0, 255), 2)

    def draw_car_velocity(self, vx: float, vy: float) -> None:
        offset = np.array([-0.1575, 0.0, 0.0])
        if self.data[0] is not None and self.data[1] is not None:
            data = self.data[1]
            t_car = self.data[0].t.squeeze() + offset
            rotV, _ = cv2.Rodrigues(data.R)
            x0 = t_car[0]
            y0 = t_car[1]
            vx = vx*0.1
            vy = vy*0.1
            arrow_3d = np.array([
                [x0, y0, 0],
                [x0 + vx, y0 + vy, 0]
            ], dtype=np.float32).reshape(-1, 3)
            arrow_2d, _ = cv2.projectPoints(arrow_3d, rotV, data.t, self.K, self.dist_coeffs)
            arrow_2d = arrow_2d.astype(int)
            self.frame = cv2.arrowedLine(self.frame, tuple(arrow_2d[0].ravel()), tuple(arrow_2d[1].ravel()), (0, 255, 0), 2)


    def imshow(self) -> None:
        # with self.mtx:
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
    ) -> tuple[list[VizData | None], np.ndarray, bool, np.ndarray]:

    # frame = pipe.wait_for_frames().get_color_frame()
    frame = pipe.wait_for_frames().get_infrared_frame()
    got_frame = True if frame else False
    unaltered_image = np.asarray(frame.data, dtype=np.uint8)
    image = unaltered_image
    # image = cv2.cvtColor(unaltered_image, cv2.COLOR_RGB2GRAY)
    # image = cv2.equalizeHist(image)
    # image = cv2.convertScaleAbs(image, alpha=0.5, beta=0.0)

    detections: list[Detection] = detector.detect(
        image,
        estimate_tag_pose=True,
        camera_params=cam_params,
        tag_size=0.1475#0.1858975
    ) # type: ignore[assignment]

    car_detection = list(filter(lambda d: d.tag_id == 0, detections))
    floor_detection = list(filter(lambda d: d.tag_id == 1, detections))

    viz_data: list[VizData | None] = [None, None]

    flip_yz = np.array([
        [1, 0, 0],
        [0, -1, 0],
        [0, 0, -1]
    ], dtype=np.float32)

    if car_detection:
        viz_data[0] = VizData(
            R = car_detection[0].pose_R @ flip_yz,  # type: ignore[assignment]
            t = car_detection[0].pose_t             # type: ignore[assignment]
        )

    if floor_detection:
        viz_data[1] = VizData(
            R = floor_detection[0].pose_R @ flip_yz,    # type: ignore[assignment]
            t = floor_detection[0].pose_t               # type: ignore[assignment]
        )

    return viz_data, image, got_frame, unaltered_image




def compute_panda_torques(
    panda_action_to_reference: Compiled,
    panda_reference_to_torque: Compiled,
    robot_state: RobotState,
    action: np.ndarray,
    ) -> np.ndarray:

    reference = panda_action_to_reference(action)
    tau = panda_reference_to_torque(reference, np.array(robot_state.q), np.array(robot_state.dq))

    return tau


def start_gripper_control(
    gripper: Gripper,
    gripper_state: _gripper_state,
    gripper_event: threading.Event,
    ) -> threading.Thread:

    def gripper_state_reader(
        gripper: Gripper,
        gripper_state: _gripper_state,
        ):
        print("Starting Gripper state reader daemon thread...")
        while True:
            try:
                gripper_state.update(gripper.read_once())
                sleep(0.5)
            except Exception as e:
                print(f"Exception caught in gripper_state_reader():\n\n{format_exc()}")
                sleep(0.5)

    def gripper_ctrl(
            gripper: Gripper,
            gripper_event: threading.Event,
        ):
        print("Starting Gripper control daemon thread...")
        set_current_thread_to_highest_scheduler_priority(error_message="Couldn't set priority")
        num_retries = 10
        while True:
            result = gripper_event.wait(timeout=0.5)
            if not result: continue

            gripper.move(width=0.08, speed=1.0)
            gripper_event.clear()
            # print("Ball released")

    gripper_ctrl_thread = threading.Thread(target=gripper_ctrl, args=(gripper, gripper_event), daemon=True)
    gripper_ctrl_thread.start()

    gripper_state_reader_thread = threading.Thread(target=gripper_state_reader, args=(gripper, gripper_state), daemon=True)
    gripper_state_reader_thread.start()

    return gripper_ctrl_thread

# needs to be defined at module level
def estimate_ball_pos(
    gripping: SynchronizedBase,
    ball_pos: np.ndarray,
    ball_buffer: SynchronizedBase,
    latest_frame: np.ndarray,
    frame_buffer: SynchronizedBase,
    process_ready_event: mp_Event,
    exit_event: mp_Event,
    cam_params: tuple[float, ...]
    ) -> None:

    os.sched_setaffinity(0, BALL_DETECTION_CPUS)

    print("Starting ball observation process...")
    ball_pos = np.frombuffer(ball_buffer.get_obj(), dtype=np.float32).reshape(ball_pos.shape)
    latest_frame = np.frombuffer(frame_buffer.get_obj(), dtype=np.uint8).reshape(latest_frame.shape)

    # from torch import set_num_threads, set_num_interop_threads
    # set_num_threads(len(BALL_DETECTION_CPUS))
    # set_num_interop_threads(len(BALL_DETECTION_CPUS))

    _ball_pos_cam_frame = ball_pos_cam_frame.lower(np.ones((4,)), cam_params).compile()

    detect_ball = setup_yolo()

    no_detects = 0
    max_no_detects = 20000
    process_ready_event.set()
    while no_detects < max_no_detects and not exit_event.is_set():
        if gripping.value:
            sleep(0.008)
            continue

        # with frame_buffer.get_lock():
        _latest_frame = np.copy(latest_frame)

        # cv2.imshow("Ball Detection", _latest_frame)
        # if cv2.waitKey(4) & 0xFF == ord('q'):
        #     exit_event.set()
        #     return

        result = detect_ball(_latest_frame)

        if result.shape[0] == 0:
            no_detects += 1
            if no_detects >= max_no_detects:
                print("Aborting ball observation process...")
                exit(1)
            sleep(0.008)
            continue

        # with ball_buffer.get_lock():
        np.copyto(ball_pos, _ball_pos_cam_frame(result, cam_params))
            # print("Ball position:", ball_pos)

        no_detects = 0
        sleep(0.008)


def start_ball_observation_process(
    gripping: SynchronizedBase,
    ball_pos: np.ndarray,           # shared memory
    ball_buffer: SynchronizedBase,  # shared memory
    latest_frame: np.ndarray,       # shared memory
    frame_buffer: SynchronizedBase, # shared memory
    exit_event: mp_Event,
    process_ready_event: mp_Event,
    cam_params: tuple[float, ...]
    ) -> multiprocessing.Process:

    ball_obs_process = multiprocessing.Process(
        target=estimate_ball_pos,
        args=(
            gripping,
            ball_pos,
            ball_buffer,
            latest_frame,
            frame_buffer,
            process_ready_event,
            exit_event,
            cam_params
        ),
        daemon=False # can't be daemon process because of torch model setup
    )
    ball_obs_process.start()
    process_ready_event.wait()
    process_ready_event.clear()

    return ball_obs_process


def pick_up_ball(
    gripper: Gripper,
    panda: Panda,
    pre_pick_up_joints_0: np.ndarray = np.array([2.8236212650113583, -0.05107980247129473, -0.380029254294423, -2.069614593940868, -0.019233986661573988, 2.0426735146045685, 2.100047396952907]),
    pre_pick_up_joints_1: np.ndarray = np.array([2.796583685997808, 0.9800072787017697, -0.26372293091522275, -1.8068627915620958, 0.3522799074314543, 2.7211871241728467, 1.526388245557745]),
    pick_up_joints: np.ndarray = np.array([2.795450960834821, 1.1965057161732724, -0.26524958326105486, -1.652138797442118, 0.3542163000404835, 2.7812191962401074, 1.5686535192877051]),
    post_pick_up_joints_0: np.ndarray = np.array([2.796583685997808, 0.9800072787017697, -0.26372293091522275, -1.8068627915620958, 0.3522799074314543, 2.7211871241728467, 1.526388245557745]),
    post_pick_up_joints_1: np.ndarray = np.array([2.8236212650113583, -0.05107980247129473, -0.380029254294423, -2.069614593940868, -0.019233986661573988, 2.0426735146045685, 2.100047396952907]),
    ) -> None:

    gripper.move(width=0.08, speed=1.0)
    panda.move_to_joint_position(pre_pick_up_joints_0, speed_factor=0.4)
    panda.move_to_joint_position(pre_pick_up_joints_1, speed_factor=0.2)
    panda.move_to_joint_position(pick_up_joints, speed_factor=0.1)

    grasped = False
    while not grasped:
        grasped = gripper.grasp(width=0.045, speed=0.01, force=140, epsilon_inner=0.005, epsilon_outer=0.005)
    print("Grasped:", grasped)

    panda.move_to_joint_position(post_pick_up_joints_0, speed_factor=0.2)
    panda.move_to_joint_position(post_pick_up_joints_1, speed_factor=0.4)
    # panda.move_to_start()
    panda.move_to_joint_position(PandaLimits().q_start.at[3].set(-2.65619449), speed_factor=0.2) # type: ignore[]

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
    compute_panda_torques: Callable,
    ball_pos: np.ndarray, # shared buffer array
    ball_buffer: SynchronizedBase,
    latest_frame: np.ndarray, # shared buffer array
    frame_buffer: SynchronizedBase,
    msg: ZeusMessage,
    vis: Visualization,
    msg_event: threading.Event,
    gripper_event: threading.Event,
    exit_event: mp_Event,

    # Argument to loop_body() during rollouts
    inference_carry: tuple,
    torque_controller: controllers.AppliedTorque,
    pose_estimates: PoseEstimates,

    # Constants
    ctrl_time_ns: int = int(0.04*1e9),
    car_goal_radius: float = 0.1,
    ball_goal_radius: float = 0.1,
    num_agents: int = 2,
    # lowpass_filter_q_car: LowPassFilter = LowPassFilter(input_shape=(3, ), history_length=30, bandlimit_hz=1.0, sample_rate_hz=1.0/0.04),
    lowpass_filter_qd_car: LowPassFilter = LowPassFilter(input_shape=(3, ), history_length=30, bandlimit_hz=1.25, sample_rate_hz=1.0/0.04),
    # lowpass_filter_p_ball: LowPassFilter = LowPassFilter(input_shape=(3,), history_length=30, bandlimit_hz=1.5, sample_rate_hz=1.0/0.04),
    lowpass_filter_pd_ball: LowPassFilter = LowPassFilter(input_shape=(3,), history_length=30, bandlimit_hz=1.5, sample_rate_hz=1.0/0.04),
    lowpass_filter_floor_R: LowPassFilter = LowPassFilter(input_shape=(3, 3), history_length=100, bandlimit_hz=0.005, sample_rate_hz=1.0/0.04),
    lowpass_filter_floor_t: LowPassFilter = LowPassFilter(input_shape=(3, 1), history_length=100, bandlimit_hz=0.005, sample_rate_hz=1.0/0.04),
    # lowpass_filter_car_R: LowPassFilter = LowPassFilter(input_shape=(3, 3), history_length=30, bandlimit_hz=1.0, sample_rate_hz=1.0/0.04),
    # lowpass_filter_car_t: LowPassFilter = LowPassFilter(input_shape=(3, 1), history_length=30, bandlimit_hz=1.0, sample_rate_hz=1.0/0.04),
    ) -> tuple[tuple, bool, bool]:

    start_ns = clock_gettime_ns(CLOCK_BOOTTIME)

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
        warmup,
        prng
    ) = inference_carry

    prng, _prng = split(prng)

    car_pose, floor_pose = pose_estimates.get_data()

    car_R = car_pose.R
    car_t = car_pose.t
    if not warmup:
        # car_R = lowpass_filter_car_R(car_pose.R)
        # car_t = lowpass_filter_car_t(car_pose.t)
        floor_R = lowpass_filter_floor_R(floor_pose.R)
        floor_t = lowpass_filter_floor_t(floor_pose.t)
    else:
        floor_R = floor_pose.R
        floor_t = floor_pose.t


    robot_state = panda.get_state()
    obs, aux = observe(
        env,
        p_goal,
        car_R,
        car_t,
        floor_R,
        floor_t,
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

    (
        q_car, q_arm, q_gripper, p_ball,
        qd_car, qd_arm, qd_gripper, pd_ball,
        p_goal,
        dc_goal,
        db_target
     ) = decode_obs(obs)

    # q_car = lowpass_filter_q_car(q_car)
    qd_car = lowpass_filter_qd_car(qd_car)
    # p_ball = lowpass_filter_p_ball(p_ba1.0
    pd_ball = lowpass_filter_pd_ball(pd_ball)

    done = logical_or(
        dc_goal <= car_goal_radius,
        logical_or(
            db_target <= ball_goal_radius,
            p_ball[2] <= 0.1,
        )
    )

    # Policy inference
    actions, actor_hidden_states = policy_inference(
        _prng,
        num_agents,
        apply_fns,
        previous_done,
        actors,
        actor_hidden_states,
        obs,
    )
    a_zeus, a_panda = actions

    a_zeus = env.scale_action(a_zeus, env.act_space_car.low, env.act_space_car.high)
    a_panda = env.scale_action(a_panda, env.act_space_arm.low, env.act_space_arm.high) # includes gripper

    magnitude, angle, rot_vel = a_zeus


    rot_vel = rotational_velocity_modifier(magnitude)*rot_vel

    a_arm, a_gripper = a_panda[0:3], a_panda[-1]

    # Send actions to agents
    msg.write(round(float(magnitude), 2), round(float(angle), 2), round(float(rot_vel), 2), ZeusMode.ACT)
    msg_event.set()

    # Force only releasing once
    a_gripper = -1.0 if ball_released else a_gripper
    ball_released = True if a_gripper >= 0.0 else ball_released
    gripper_state.gripping.value = not ball_released

    # Inner loop
    eps = int(0.0005*1e9)
    atleast = 1
    t_ns = clock_gettime_ns(CLOCK_BOOTTIME)
    while t_ns - start_ns < (ctrl_time_ns - eps) or (atleast > 0):
        atleast -= 1

        # Panda control
        if not warmup:
            state = panda.get_state()
            tau = compute_panda_torques(state, a_arm)
            # if atleast == 0: print("\ntau: ", tau, "\na_arm: ", a_arm, "\nqd: ", np.round(np.array(state.dq)[[0, 3, 5]], 2))
            torque_controller.set_control(tau)

        # Gripper control
        delay_compensation = 1.0
        if a_gripper >= 0.0 and (t_ns - start_ns) >= delay_compensation*a_gripper*ctrl_time_ns:
            gripper_event.set()

        # Ball ballistic dead reckoning
        # if ball_released or warmup:
        #     _dt = (clock_gettime_ns(CLOCK_BOOTTIME) - t_ns) / 1e9
        #     p_ball, pd_ball = observe_ball_ballistic(p_ball, pd_ball, _dt)
            # p_ball = p_ball + _dt * pd_ball

        # sleep(0.001)
        t_ns = clock_gettime_ns(CLOCK_BOOTTIME)

    data, frame, got_frame, unaltered_image = apriltag_detection(pipe, cam_params, detector)
    if got_frame:
        pose_estimates.update_poses(data, warmup)
        # vis.update_data(data, frame)

        # with frame_buffer.get_lock():
        # np.copyto(latest_frame, unaltered_image)

    # if not ball_released:
        # with ball_buffer.get_lock():
        # np.copyto(ball_pos, p_ball)

    # print(q_car, qd_car)
    # vis.draw_axes()
    # vis.draw_ball(p_ball)
    # vis.draw_goal(p_goal)
    # vis.draw_car_action(q_car[2], magnitude, angle)
    # vis.draw_car_velocity(qd_car[0], qd_car[1])
    # vis.imshow()
    # if cv2.waitKey(40) & 0xFF == ord('q'):
    #     exit_event.set()
    #     return (None, None, None, None, None, None, None, None), True, True

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
        warmup,
        prng
    ), bool(done), False


def main() -> None:
    # os.sched_setaffinity(0, PANDA_CPUS)
    # sys.setswitchinterval(interval)

    env, decode_obs = setup_env()
    rnn_hidden_size = 8 # this value must be correct for the model being loaded
    rnn_fc_size = 64    # this value must be correct for the model being loaded
    actors, actor_hidden_states, apply_fns = load_policies(env, rnn_hidden_size, rnn_fc_size, "checkpoint_LATEST")
    rng = PRNGKey(reproducibility_globals.PRNG_SEED)
    pipe, cam_params, dist_coeffs, K, R, t, width, height, detector = setup_camera()
    panda, panda_ctx, gripper, desk = setup_panda()

    gripper_state = _gripper_state(gripper.read_once())
    torque_controller = controllers.AppliedTorque()
    # set_current_thread_to_highest_scheduler_priority("couldn't set priority")

    # Shared memory arrays
    ball_pos, ball_buffer = _ball_pos_shared_array(3)
    latest_frame, frame_buffer = _latest_frame_shared_array(width, height, 3)

    msg = ZeusMessage()
    vis = Visualization(width, height, K, dist_coeffs)
    msg_event = threading.Event()
    gripper_event = threading.Event()
    reset_event = multiprocessing.Event()
    exit_event = multiprocessing.Event()
    process_ready_event = multiprocessing.Event()

    # Ahead of time compilation of controller subroutines
    panda_action_to_reference = action_to_reference.lower(
        np.zeros(3, dtype=np.float32)
    ).compile()

    panda_reference_to_torque = reference_to_torque.lower(
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
        np.zeros(7, dtype=np.float32),
    ).compile()

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
        compute_panda_torques=partial(compute_panda_torques, panda_action_to_reference, panda_reference_to_torque),
        ball_pos=ball_pos,
        ball_buffer=ball_buffer,
        latest_frame=latest_frame,
        frame_buffer=frame_buffer,
        msg=msg,
        vis=vis,
        msg_event=msg_event,
        gripper_event=gripper_event,
        exit_event=exit_event
    )

    websocket_client_thread = threading.Thread(target=websocket_client, args=(msg, msg_event, exit_event))
    websocket_client_thread.start()

    # ball_detection_process = start_ball_observation_process(
    #     gripper_state.gripping,
    #     ball_pos,
    #     ball_buffer,
    #     latest_frame,
    #     frame_buffer,
    #     exit_event,
    #     process_ready_event,
    #     cam_params
    # )

    gripper_ctrl_thread = start_gripper_control(
        gripper,
        gripper_state,
        gripper_event,
    )

    init_inference_carry = (
        actors, actor_hidden_states,
        np.array([True]), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32),
        np.array([0.40647, -0.0013068, 1.3482]), np.zeros(3, dtype=np.float32), False,
        0.9*reset_goal_pos(env.car_limits.x_min, env.car_limits.x_max, env.car_limits.y_min, env.car_limits.y_max),
        0.04, True, rng)

    print("Starting inference...")
    count = 1
    all_rollouts_done = False
    while not all_rollouts_done and not exit_event.is_set():
        print("Resetting before rollout...")
        gripper_event.clear()
        reset_event.clear()
        pose_estimates.reset()

        p_goal = 0.8*reset_goal_pos(env.car_limits.x_min, env.car_limits.x_max, env.car_limits.y_min, env.car_limits.y_max)
        print("p_goal:", p_goal)

        inference_carry = (
            actors,
            actor_hidden_states,
            np.array([True]),               # previous_done
            inference_carry[3] if not count == 1 else init_inference_carry[3],  # prev_q_car
            np.zeros(3, dtype=np.float32),  # prev_qd_car
            np.array([0.40647, -0.0013068, 1.3482]),     # prev_p_ball
            np.zeros(3, dtype=np.float32),  # prev_pd_ball
            False,                          # ball_released
            p_goal,
            0.04,                           # dt
            count == 1,                      # warmup
            inference_carry[-1] if not count == 1 else init_inference_carry[-1]  # prng
        )

        # Need to do a warm up run before starting inner loop threads
        if count > 1:
            panda.stop_controller()
            pick_up_ball(gripper, panda)
            panda.start_controller(torque_controller) # HERE

        print("Running rollout...")
        rollout_count = 0
        current_rollout_done = False
        with panda_ctx:
            while not current_rollout_done and rollout_count < 50_000_000 and panda_ctx.ok():

                # We catch all exceptions to be able to clean up some stuff (had problems with Ubuntu hanging)
                try:
                    inference_carry, current_rollout_done, all_rollouts_done = _loop_body(
                        inference_carry=inference_carry,
                        torque_controller=torque_controller,
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

            # Slow down Panda before stopping controller
            start_ns = clock_gettime_ns(CLOCK_BOOTTIME)
            while clock_gettime_ns(CLOCK_BOOTTIME) - start_ns < int(0.75*1e9):
                state = panda.get_state()
                # print("after rollout dq:", state.dq)
                torque_controller.set_control(
                    pd_control_to_start(np.array(state.q), np.array(state.dq))
                )

            # TODO: if sentence to check if panda is in control
            print("Stopping Panda controller...")
            panda.stop_controller()
            print("Panda controller stopped.")
            panda.move_to_joint_position(PandaLimits().q_start, speed_factor=0.2) # type: ignore[]

            reset_event.set()
            # cv2.destroyAllWindows()

            all_rollouts_done = True if count >= 15 else False # TODO: proper logic for multiple rollouts
            count += 1

    cv2.destroyAllWindows()
    exit_event.set()

    print("Stopping RealSense pipeline...")
    pipe.stop()
    print("RealSense pipeline stopped.")

    print("Waiting for threads to join...")
    websocket_client_thread.join(timeout=10)
    gripper_ctrl_thread.join(timeout=10)
    # ball_detection_process.join(timeout=10)
    print("Threads joined.")

    # Relinquish control of Panda
    # TODO: if sentence to check if panda is in control
    print("Moving Panda to start position...")
    panda.move_to_start()
    print("Panda moved to start position.")

    print("Stopping Panda...")
    panda.get_robot().stop()
    print("Panda stopped.")

    print("Releasing control...")
    desk.deactivate_fci()   # type: ignore[attr-defined]
    desk.release_control()  # type: ignore[attr-defined]
    print("Control released.")



if __name__ == "__main__":
    main()
