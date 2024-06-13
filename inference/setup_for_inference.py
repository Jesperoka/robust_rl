from jax._src.random import KeyArray
import pyrealsense2 as rs
import reproducibility_globals
from typing import Callable
from pupil_apriltags import Detector
from os.path import dirname, abspath, join
from functools import partial
from numpy import array as np_array, eye, ndarray, zeros, float32, random
from cv2 import INTER_AREA, cvtColor, COLOR_GRAY2BGR, resize
from cv2.typing import MatLike
from mujoco import MjModel, mjtObj, mj_name2id # type: ignore[attr-defined]
from mujoco.mjx import Model, put_model
from environments.A_to_B_jax import A_to_B
from environments.options import EnvironmentOptions
from numba import njit
from jax import Array, device_get, jit
from jax.lax import cond
from jax.numpy import newaxis, array, mod, arctan2, min, abs, sum, clip, sign, pi, concatenate
from jax.numpy.linalg import norm
from jax.random import split, PRNGKey
from jax.tree_util import tree_map
from orbax.checkpoint import Checkpointer, StandardCheckpointHandler, args
from algorithms.utils import ActorInput, MultiActorRNN, FakeTrainState, initialize_actors
from inference.controllers import gripper_ctrl, minimal_pos_controller
from environments.reward_functions import simple_curriculum_reward
from environments.physical import PandaLimits, ZeusDimensions
from panda_py import Desk, Panda, PandaContext
from panda_py.libfranka import Gripper, GripperState, RobotMode, RobotState
from ultralytics.models import YOLO
# from torch.jit import optimize_for_inference, script



import pdb

# Setup functions
# --------------------------------------------------------------------------------
CURRENT_DIR = dirname(abspath(__file__))
CHECKPOINT_DIR = join(CURRENT_DIR, "..", "trained_policies", "checkpoints")
CHECKPOINT_FILE = "_IN_TRAINING_with_vars_5118__param_dicts__fc_64_rnn_8"
# CHECKPOINT_FILE = "checkpoint_LATEST_with_vars_param_dicts__fc_64_rnn_8"
MODEL_DIR = "mujoco_models"
MODEL_FILE = "scene.xml"
SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
FILEPATH = abspath(join(dirname(__file__), "../", "sens.txt"))
CTRL_FREQUENCY = 1000; assert CTRL_FREQUENCY == 1000
MAX_RUNTIME = 30.0
BALL_DETECTION_MODEL = "yolov8s-worldv2.pt"

# Camera pre-processing parameters
HEIGHT = 480
WIDTH = 848
CROP_X_LEFT = 152
CROP_X_RIGHT = 88
CROP_Y_TOP = 0
REDUCTION = 3
IMGSZ = (HEIGHT-CROP_Y_TOP-32*REDUCTION, WIDTH-CROP_X_LEFT-CROP_X_RIGHT-32*REDUCTION)
SCALE_X = (IMGSZ[1] - 32.0*REDUCTION) / float(IMGSZ[1])
SCALE_Y = (IMGSZ[0] - 32.0*REDUCTION) / float(IMGSZ[0])

def setup_env():
    scene = join(CURRENT_DIR, "..", MODEL_DIR, MODEL_FILE)

    model: MjModel = MjModel.from_xml_path(scene)
    mjx_model: Model = put_model(model)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")

    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn           = partial(simple_curriculum_reward, 1),
        arm_ctrl            = minimal_pos_controller,
        arm_act_min         = array([-2.0, -2.0, -2.5]),
        arm_act_max         = array([2.0, 2.0, 2.5]),
        gripper_ctrl        = gripper_ctrl,
        goal_radius         = 0.05,
        steps_per_ctrl      = 20,
        time_limit          = 3.0,
    )

    env = A_to_B(mjx_model, None, grip_site_id, options) # type: ignore[assignment]

    return env, jit(env.decode_observation)


def setup_camera() -> tuple[rs.pipeline, tuple[float, float, float, float], MatLike, MatLike, MatLike, MatLike, int, int, Detector]:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    sensor_idx = 1 # 0 color 1 infra
    cfg.enable_stream(rs.stream.infrared, sensor_idx, 848, 480, rs.format.y8, 90)
    # cfg.enable_stream(rs.stream.color, sensor_idx, WIDTH, HEIGHT, rs.format.rgb8, 30)
    profile = pipe.start(cfg)
    device = profile.get_device()
    sensor = device.query_sensors()[0]

    sensor.set_option(rs.option.emitter_enabled, 0)
    sensor.set_option(rs.option.laser_power, 0)
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.gain, 16) # default is 16
    sensor.set_option(rs.option.exposure, 3300)

    cam_intrinsics = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
    # cam_intrinsics = profile.get_stream(rs.stream.color, 0).as_video_stream_profile().get_intrinsics()

    fx: float = cam_intrinsics.fx
    fy: float = cam_intrinsics.fy
    cx: float = cam_intrinsics.ppx
    cy: float = cam_intrinsics.ppy

    cam_params: tuple[float, ...]= (fx, fy, cx, cy)

    dist_coeffs: MatLike = np_array(cam_intrinsics.coeffs, dtype=float32)
    width: int = cam_intrinsics.width
    height: int = cam_intrinsics.height

    K = np_array([[fx, 0, cx],
               [0, fy, cy],
               [0, 0, 1]], dtype=float32)

    R = eye(3, dtype=float32)
    t = zeros((3, 1), dtype=float32)

    detector: Detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=0,
        decode_sharpening=0.0,
    )

    return pipe, cam_params, dist_coeffs, K, R, t, width, height, detector

def setup_yolo():
    model = YOLO(BALL_DETECTION_MODEL, task="detect")

    model.set_classes([
        "stressball", "baseball", "softball",
        "sportsball", "white ball", "gray ball",
        "grey ball", "small white ball", "small ball",
        "tiny ball", "ball", "fast-moving ball",
        "fastball", "moving ball", "blurry ball",
        "large ball", "tiny ball", "toy ball",
        "rubber ball", "industrial ball", "flying ball",
        "bouncing ball", "rolling ball", "spinning ball",
        "rotating ball", "floating ball", "hovering ball",
        "sphere", "round ball", "orb",
    ])

    model.compile(mode="max-autotune-no-cudagraphs", backend="inductor")

    _model_forward = partial(
        model.predict,
        stream=False,
        conf=0.02,
        imgsz=IMGSZ,
        show=False,
        agnostic_nms=True,
        max_det=1,
        augment=True,
        iou=0.4,
        verbose=False
    )

    def model_forward(img):
        img = img[CROP_Y_TOP:, CROP_X_LEFT:-CROP_X_RIGHT]
        img = resize(img, IMGSZ[::-1], interpolation=INTER_AREA)

        return _model_forward(source=img)[0].boxes.xywh.squeeze().numpy()

    return model_forward


def load_policies(env, rnn_hidden_size: int, rnn_fc_size: int, checkpoint_file: str = CHECKPOINT_FILE):
    rng = PRNGKey(reproducibility_globals.PRNG_SEED)
    obs_size = env.obs_space.sample().shape[0]
    act_sizes = tree_map(lambda space: space.sample().shape[0], env.act_spaces, is_leaf=lambda x: not isinstance(x, tuple))
    num_envs, num_agents, lr, max_grad_norm = 1, 2, 1e-3, 0.5

    sequence_length, num_envs = 1, 1
    assert sequence_length == 1 and num_envs == 1
    action_rngs = tuple(split(rng))

    # Instantiate checkpointer
    checkpointer = Checkpointer(StandardCheckpointHandler())

    # Init actors and forward functions
    actors, actor_hidden_states = initialize_actors(action_rngs, num_envs, num_agents, obs_size, act_sizes, lr, max_grad_norm, rnn_hidden_size, rnn_fc_size)
    apply_fns = tuple(jit(partial(ts.apply_fn, train=False), static_argnames=("train",)) for ts in actors.train_states) # type: ignore[attr-defined]

    # Restore parameter and variable dicts
    print("Loading policies from: ", join(CHECKPOINT_DIR, CHECKPOINT_FILE))
    abstract_state = {"actor_"+str(i): (device_get(ts.params), device_get(var)) for i, (ts, var) in enumerate(zip(actors.train_states, actors.vars))}
    restored_state = checkpointer.restore(
            join(CHECKPOINT_DIR, CHECKPOINT_FILE),
            args=args.StandardRestore(abstract_state)
    )
    restored_actors = actors
    restored_actors.train_states = tuple(FakeTrainState(params=params) for (params, _) in restored_state.values())
    restored_actors.vars = tuple(vars for (_, vars) in restored_state.values())

    return restored_actors, actor_hidden_states, apply_fns


def setup_panda() -> tuple[Panda, PandaContext, Gripper, Desk]:
    with open(FILEPATH, 'r') as file:
        username = file.readline().strip()
        password = file.readline().strip()

    print("Connecting to Desk")
    desk = Desk(SHOP_FLOOR_IP, username, password)

    while not desk.has_control():
        key = input("Another user has control\nPress any key (except q) when control has been relinquished to unlock FCI.\nOr press q to exit: ")
        if key == "q":
            exit(0)
        desk.take_control()

    print("Unlocking joints if locked")
    desk.unlock()

    print("Activating FCI")
    desk.activate_fci()

    print("Conneting to Panda")
    panda = Panda(SHOP_FLOOR_IP)
    panda.recover()
    panda_ctx = panda.create_context(frequency=CTRL_FREQUENCY, max_runtime=MAX_RUNTIME)
    assert panda.get_state().robot_mode == RobotMode.kIdle, f"Cannot run while in robot_mode: {panda.get_state().robot_mode}, should be {RobotMode.kIdle}"
    panda.move_to_start()

    print("Connecting to Franka Hand")
    gripper = Gripper(SHOP_FLOOR_IP)
    gripper.homing()

    return panda, panda_ctx, gripper, desk
# --------------------------------------------------------------------------------



# Inference functions
# --------------------------------------------------------------------------------
@partial(jit, static_argnames=("num_agents", "apply_fns"))
def policy_inference(
    rng: KeyArray,
    num_agents: int,
    apply_fns: Callable,
    done: Array,
    actors: MultiActorRNN,
    actor_hidden_states: tuple[Array, ...],
    observation: Array,
    ) -> tuple[Array, tuple[Array, ...]]:

    inputs = tuple(
            ActorInput(observation[newaxis, :][newaxis, :], done[newaxis, :])
            for _ in range(num_agents)
    )

    actor_hidden_states, policies = zip(*tree_map(
        lambda ts, vars, hs, inputs, apply_fn: apply_fn({"params": ts.params, "vars": vars}, hs, inputs, train=False),
            actors.train_states,
            actors.vars,
            actor_hidden_states,
            inputs,
            apply_fns,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
    ))
    actions = tree_map(lambda policy: policy.mode.squeeze(), policies, is_leaf=lambda x: not isinstance(x, tuple))
    # actions = tree_map(lambda policy: policy.sample(seed=rng).squeeze(), policies, is_leaf=lambda x: not isinstance(x, tuple))

    return actions, actor_hidden_states


# This function takes a 3-element action vector and returns a 7-element reference vector
# where joints 0, 3, and 5 are set to the corresponding action values (rad/s).
# The other joints are set to the start joint angles.
@jit
def action_to_reference(action: Array) -> Array:
    q_start = PandaLimits().q_start

    return q_start.at[0].set(action[0]).at[3].set(action[1]).at[5].set(action[2])

# This function takes a 7-element reference vector and returns the controller torques
# where joints 0, 3, and 5 are velocity controlled and the other joints are position controlled.
@jit
def reference_to_torque(reference: Array, q: Array, qd: Array) -> Array:
    ref_min = array([-2.0, -1.7628, -2.8973, -2.0, -2.8973, -2.5, -2.8973], dtype=float32)
    ref_max = array([ 2.0,  1.7628,  2.8973,  2.0, 2.8973, 2.5,  2.8973], dtype=float32)
    reference = clip(reference, ref_min, ref_max)

    tau_min = array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0], dtype=float32)
    tau_max = array([ 87.0,  87.0,  87.0,  87.0,  12.0,  12.0,  12.0], dtype=float32)

    tau = clip(array([
               14.0 * (reference[0] - qd[0]), #- sign(qd[0])*(1.5*qd[0])**2,
               200.0 * (reference[1] - q[1]) - 20.0 * qd[1],
               200.0 * (reference[2] - q[2]) - 20.0 * qd[2],
               14.0 * (reference[3] - qd[3]), #- sign(qd[3])*(1.5*qd[3])**2,
               100.0 * (reference[4] - q[4]) - 10.0 * qd[4],
               14.0 * (reference[5] - qd[5]), #- sign(qd[5])*(1.5*qd[5])**2,
               30.0 * (reference[6] - q[6]) - 2.0 * qd[6]
              ], dtype=float32), 0.2*tau_min, 0.2*tau_max).squeeze()

    return tau

# Used to slow down the robot before resetting
@jit
def pd_control_to_start(q: Array, qd: Array) -> Array:
    q_start = PandaLimits().q_start

    tau_min = array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0], dtype=float32)
    tau_max = array([ 87.0,  87.0,  87.0,  87.0,  12.0,  12.0,  12.0], dtype=float32)

    # Kp = array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0], dtype=float32)
    Kp = zeros(7, dtype=float32)
    Kd = array([18.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0], dtype=float32)

    tau = clip((Kp * (q_start - q) - Kd * qd), 0.2*tau_min, 0.2*tau_max)

    return tau
pd_control_to_start = pd_control_to_start.lower(zeros(7, dtype=float32), zeros(7, dtype=float32)).compile() # type: ignore[assignment]

# uses stateful random number generation
def reset_goal_pos(x_min: float, x_max: float, y_min: float, y_max: float) -> Array:
    return array([random.uniform(x_min, x_max), random.uniform(y_min, y_max)], dtype=float32)

@jit
def angle_velocity_modifier(theta: Array) -> Array:
    return 0.5 + 0.5*( abs( ( mod(theta, (pi/2.0)) ) - (pi/4.0) ) / (pi/4.0) )

@jit
def rotational_velocity_modifier(magnitude: Array) -> Array:
    return clip(0.1/(abs(magnitude) + 0.1), 0.0, 1)

@jit
def finite_difference(x, prev_x, dt):
    return (1.0/dt) * (x - prev_x)


@partial(jit, static_argnames=("car_angle_offset", "car_max_abs_vel", "car_pos_offset_x", "car_pos_offset_y", "car_pos_offset_z"))
def observe_car(
    floor_R: Array,
    floor_t: Array,
    car_R: Array,
    car_t: Array,
    prev_q_car: Array,
    prev_qd_car: Array,
    dt: float,
    car_angle_offset: float = 0.0,
    car_max_abs_vel:  Array = array([2.0, 2.0, 2.0]),
    car_pos_offset_x: float = 0.0,
    car_pos_offset_y: float = 0.0,
    car_pos_offset_z: float = 0.0
    ) -> tuple[Array, Array, Array, Array]:

    # TODO: rename _"ground"_ to _"floor"_
    # Compute the relative pose of the car with respect to the ground frame
    car_t_ground_frame = floor_R.T @ (car_t - floor_t)
    car_R_ground_frame = floor_R.T @ car_R

    x_dir = array((1.0, 0.0))
    x_dir_rotated = car_R_ground_frame[0:2, 0:2] @ x_dir # approximate rotation about z-axis
    _theta = arctan2(x_dir[0]*x_dir_rotated[1] - x_dir[1]*x_dir_rotated[0], x_dir[0]*x_dir_rotated[0] + x_dir[1]*x_dir_rotated[1])
    theta = mod(_theta + car_angle_offset, 2*pi)

    q_car = array((
        car_t_ground_frame[0][0] + car_pos_offset_x,
        car_t_ground_frame[1][0] + car_pos_offset_y,
        theta
    ))

    qd_car = finite_difference(q_car, prev_q_car, dt)# + prev_qd_car) / 2.0 # Average of previous computed velocity and positional finite differences

    def discontinuity():
        return -sign(qd_car[2]) * 0.5 * min(array([
            abs(finite_difference(q_car[2]-0.1, prev_q_car[2], dt)),
            abs(finite_difference(q_car[2]+0.1, prev_q_car[2], dt))
        ]))

    qd_car = clip(
        qd_car.at[2].set(
            cond(abs(qd_car[2]) > 5.0, discontinuity, lambda: qd_car[2]) # Angle-wrap discontinuity correction
    ), -car_max_abs_vel, car_max_abs_vel)

    return q_car, qd_car, car_R_ground_frame, car_t_ground_frame


# @njit
def observe_arm(robot_state: RobotState) -> tuple[Array, Array]:
    return array(robot_state.q, dtype=float32), array(robot_state.dq, dtype=float32)


# @njit
def observe_gripper(gripper_state: GripperState, gripping: bool) -> tuple[Array, Array]:
    q_gripper = 0.5*gripper_state.width
    vel = 0.0 if (gripping or gripper_state.width >= 0.075) else 0.05
    return array([q_gripper, q_gripper], dtype=float32), array([vel, vel], dtype=float32)


@partial(jit, static_argnames=("env", ))
def observe_relative_distances(
    q_car: Array,
    p_goal: Array,
    p_ball: Array,
    env: A_to_B
    ) -> Array:

    # 2D Euclidean distance drom car to the goal
    dc_goal = norm(p_goal[0:2] - q_car[0:2], ord=2)[newaxis]

    # 3D Euclidean distance from ball to the car (target)
    db_target = norm(array([q_car[0], q_car[1], env.playing_area.floor_height + ZeusDimensions.target_height]) - p_ball[0:3], ord=2)[newaxis]

    return array([
        dc_goal,
        db_target
    ], dtype=float32).squeeze()


@jit
def concat_obs(
    q_car: Array,
    q_arm: Array,
    q_gripper: Array,
    p_ball: Array,
    qd_car: Array,
    qd_arm: Array,
    qd_gripper: Array,
    pd_ball: Array,
    p_goal: Array,
    relative_distances: Array
    ) -> Array:

    return concatenate([
        q_car, q_arm, q_gripper, p_ball,        # poses
        qd_car, qd_arm, qd_gripper, pd_ball,    # velocities
        p_goal,                                 # car goal
        relative_distances
    ], axis=0, dtype=float32)


@partial(jit, static_argnames=("diameter", ))
def ball_pos_cam_frame(
    xywh: ndarray,
    cam_params: tuple[float, ...],
    diameter=0.062,
    ) -> Array:

    fx, fy, cx, cy = cam_params

    # Shift principal point based on cropping
    cx -= CROP_X_LEFT
    cy -= CROP_Y_TOP

    # Scale principal point and focal lengths based on resize ratio
    fx *= SCALE_X
    fy *= SCALE_Y
    cx *= SCALE_X
    cy *= SCALE_Y

    # Scale the pixel diameter to correct for error in bouding box predictions
    scale_correction = (2.6227 / 2.7532)# 1.0 # DISABLED (2.2537/3.4757)

    pixel_diameter = sum(xywh[2:]) / 2.0
    pixel_diameter *= scale_correction

    x, y = xywh[0], xywh[1]
    z = (diameter * fx) / pixel_diameter
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy

    return array([x, y, z])


def ball_pos_gripping(
        robot_state: RobotState,
        prev_p_ball: ndarray,
        prev_pd_ball: ndarray,
        dt: float,
        ball_pos_offset_x: float = 0.0,
        ball_pos_offset_y: float = 0.0,
        ball_pos_offset_z: float = 0.0
    ) -> tuple[ndarray, ndarray]:

    _ball_pos_offset_z = ball_pos_offset_z + 0.9
    p_EE = np_array(robot_state.O_T_EE, dtype=float32).reshape((4,4), order="F") # column-major format
    offset = np_array([ball_pos_offset_x, ball_pos_offset_y, _ball_pos_offset_z], dtype=float32)

    p_ball = (
        p_EE[0:3,-1]
        + offset
    )
    # pd_ball = np_array(robot_state.O_dP_EE_c[0:3], dtype=float32)
    pd_ball = finite_difference(p_ball, prev_p_ball, dt)# + prev_pd_ball) / 2.0 # Average of previous computed velocity and positional finite differences
    # print("p_ball: ", p_ball, "pd_ball: ", pd_ball)

    return p_ball, pd_ball


@partial(jit, static_argnames=("ball_pos_offset_x", "ball_pos_offset_y", "ball_pos_offset_z"))
def observe_ball(
    floor_R: Array,
    floor_t: Array,
    p_ball_cam_frame: ndarray,
    prev_p_ball: Array,
    prev_pd_ball: Array,
    dt: float,
    ball_pos_offset_x: float = 0.0,
    ball_pos_offset_y: float = 0.0,
    ball_pos_offset_z: float = 0.0
    ) -> tuple[Array, Array]:

    p_ball = floor_R.T @ (p_ball_cam_frame - floor_t.squeeze())
    # p_ball[2] = -p_ball[2]

    p_ball = p_ball + array([ball_pos_offset_x, ball_pos_offset_y, ball_pos_offset_z], dtype=float32)

    pd_ball = (finite_difference(p_ball, prev_p_ball, dt) + prev_pd_ball) / 2.0 # Average of previous computed velocity and positional finite differences

    return p_ball, pd_ball

@partial(jit, static_argnames=("gravity", ))
def observe_ball_ballistic(position, velocity, timestep, gravity=array([0, 0, -9.81])):
    velocity = velocity + gravity * timestep
    position = position + velocity * timestep + 0.5 * gravity * timestep**2

    return position, velocity

def observe(
    env: A_to_B,  # partial()
    p_goal: ndarray,
    car_R: ndarray,
    car_t: ndarray,
    floor_R: ndarray,
    floor_t: ndarray,
    prev_q_car: ndarray,
    prev_qd_car: ndarray,
    robot_state: RobotState,
    gripper_state: GripperState,
    gripping: bool,
    p_ball_cam_frame: ndarray,
    prev_p_ball: ndarray,
    prev_pd_ball: ndarray,
    dt: float,
    pos_offset_x: float = 0.1575-0.075,
    pos_offset_y: float = 0.0,
    pos_offset_z: float = 0.095
    ) -> tuple[Array, tuple[Array, ...]]:

    q_car, qd_car, car_R_gf, car_t_gf = observe_car(
        floor_R, floor_t, car_R, car_t, prev_q_car, prev_qd_car, dt,
        car_pos_offset_x=pos_offset_x,
        car_pos_offset_y=pos_offset_y,
        car_pos_offset_z=pos_offset_z
    )
    q_arm, qd_arm = observe_arm(robot_state)
    q_gripper, qd_gripper = observe_gripper(gripper_state, gripping)

    if gripping:
        p_ball, pd_ball = ball_pos_gripping(
            robot_state,
            prev_p_ball,
            prev_pd_ball,
            dt,
            ball_pos_offset_x=pos_offset_x,
            ball_pos_offset_y=pos_offset_y,
            ball_pos_offset_z=pos_offset_z
        )
    else:
        # NOTE: using ballistic ball observation
        # p_ball, pd_ball = observe_ball(
        #     floor_R, floor_t, p_ball_cam_frame, prev_p_ball, prev_pd_ball, dt,
        #     ball_pos_offset_x=pos_offset_x,
        #     ball_pos_offset_y=pos_offset_y,
        #     ball_pos_offset_z=pos_offset_z
        # )
        p_ball, pd_ball = observe_ball_ballistic(prev_p_ball, prev_pd_ball, dt)

    relative_distances = observe_relative_distances(q_car, p_goal, p_ball, env)

    obs = concat_obs(q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, pd_ball, p_goal, relative_distances)

    return obs, (car_R_gf, car_t_gf)
# --------------------------------------------------------------------------------
