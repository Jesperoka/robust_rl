import pyrealsense2 as rs
import reproducibility_globals
from typing import Callable
from pupil_apriltags import Detector
from os.path import dirname, abspath, join
from functools import partial
from numpy import array as np_array, eye, ndarray, zeros, float32, random
from cv2.typing import MatLike
from mujoco import MjModel, mjtObj, mj_name2id # type: ignore[attr-defined]
from mujoco.mjx import Model, put_model
from environments.A_to_B_jax import A_to_B
from environments.options import EnvironmentOptions
from numba import njit
from jax import Array, ShapeDtypeStruct, device_get, devices, jit
from jax.lax import cond
from jax.sharding import Mesh, PartitionSpec, NamedSharding, SingleDeviceSharding
from jax.numpy import newaxis, array, mod, arctan2, min, abs, clip, sign, pi, concatenate
from jax.numpy.linalg import norm
from jax.random import split, PRNGKey
from jax.tree_util import tree_map
from orbax.checkpoint import CheckpointManager, Checkpointer, PyTreeCheckpointHandler, PyTreeCheckpointer, RestoreArgs, StandardCheckpointHandler, args, checkpoint_utils
from algorithms.utils import ActorInput, MultiActorRNN, FakeTrainState, initialize_actors
from inference.controllers import gripper_ctrl, arm_spline_tracking_controller
from environments.reward_functions import curriculum_reward
from environments.physical import ZeusDimensions
from panda_py import Desk, Panda, PandaContext
from panda_py.libfranka import Gripper, GripperState, RobotMode, RobotState


import pdb

# Setup functions
# --------------------------------------------------------------------------------
CURRENT_DIR = dirname(abspath(__file__))
CHECKPOINT_DIR = join(CURRENT_DIR, "..", "trained_policies", "checkpoints")
CHECKPOINT_FILE = "checkpoint_LATEST"
MODEL_DIR = "mujoco_models"
MODEL_FILE = "scene.xml"
SHOP_FLOOR_IP = "10.0.0.2"  # hostname for the workshop floor, i.e. the Franka Emika Desk
FILEPATH = abspath(join(dirname(__file__), "../", "sens.txt"))
CTRL_FREQUENCY = 1000; assert CTRL_FREQUENCY == 1000
MAX_RUNTIME = 30.0


def setup_env():
    scene = join(CURRENT_DIR, "..", MODEL_DIR, MODEL_FILE)

    model: MjModel = MjModel.from_xml_path(scene)
    mjx_model: Model = put_model(model)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")

    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn           = partial(curriculum_reward, 1),
        arm_low_level_ctrl  = arm_spline_tracking_controller,
        gripper_ctrl        = gripper_ctrl,
        goal_radius         = 0.1,
        steps_per_ctrl      = 20,
        time_limit          = 3.0,
    )

    env = A_to_B(mjx_model, None, grip_site_id, options) # type: ignore[assignment]

    return env, jit(env.decode_observation)


def setup_camera() -> tuple[rs.pipeline, tuple[float, float, float, float], MatLike, MatLike, MatLike, MatLike, int, int, Detector]:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
    profile = pipe.start(cfg)
    device = profile.get_device()
    stereo_sensor = device.query_sensors()[0]

    stereo_sensor.set_option(rs.option.emitter_enabled, 0)
    stereo_sensor.set_option(rs.option.laser_power, 0)
    stereo_sensor.set_option(rs.option.enable_auto_exposure, 0)
    stereo_sensor.set_option(rs.option.gain, 16) # default is 16 (had 32 before)
    stereo_sensor.set_option(rs.option.exposure, 3300)

    cam_intrinsics = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()

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
        nthreads=4,
        quad_decimate=1.5,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    return pipe, cam_params, dist_coeffs, K, R, t, width, height, detector


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

    # restore state dicts
    abstract_state = {"actor_"+str(i): device_get(ts.params) for i, ts in enumerate(actors.train_states)}
    restored_state = checkpointer.restore(
            join(CHECKPOINT_DIR, CHECKPOINT_FILE+"_param_dicts__fc_"+str(rnn_fc_size)+"_rnn_"+str(rnn_hidden_size)),
            args=args.StandardRestore(abstract_state)
    )
    # create actors with restored state dicts
    restored_actors = actors
    restored_actors.train_states = tuple(FakeTrainState(params=params) for params in restored_state.values())

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

    print("Connection to Franka Hand")
    gripper = Gripper(SHOP_FLOOR_IP)
    gripper.homing()

    return panda, panda_ctx, gripper, desk
# --------------------------------------------------------------------------------



# Inference functions
# --------------------------------------------------------------------------------
@partial(jit, static_argnames=("num_agents", "apply_fns"))
def policy_inference(
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

    _, policies = zip(*tree_map(
        lambda ts, vars, hs, inputs, apply_fn: apply_fn({"params": ts.params, "vars": vars}, hs, inputs, train=False),
            actors.train_states,
            actors.vars,
            actor_hidden_states,
            inputs,
            apply_fns,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
    ))
    actions = tree_map(lambda policy: policy.mode().squeeze(), policies, is_leaf=lambda x: not isinstance(x, tuple))

    return actions, actor_hidden_states

# uses stateful random number generation
def reset_goal_pos(x_min: float, x_max: float, y_min: float, y_max: float) -> Array:
    return array([random.uniform(x_min, x_max), random.uniform(y_min, y_max)], dtype=float32)

@jit
def rotation_velocity_modifier(velocity, omega):
    return clip(velocity - abs(omega), 0.0, velocity)

@jit
def finite_difference(x, prev_x, dt):
    return (1.0/dt) * (x - prev_x)


@partial(jit, static_argnames=("car_angle_offset", "car_max_abs_vel", "car_pos_offset_x", "car_pos_offset_y"))
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
    car_pos_offset_y: float = 0.0
    ) -> tuple[Array, Array, Array, Array]:

    # Compute the relative pose of the car with respect to the ground frame
    car_t_ground_frame = floor_R.T @ (car_t - floor_t)
    car_R_ground_frame = floor_R.T @ car_R

    x_dir = array((1.0, 0.0))
    x_dir_rotated = car_R_ground_frame[0:2, 0:2] @ x_dir # approximate rotation about z-axis
    _theta = arctan2(x_dir[0]*x_dir_rotated[1] - x_dir[1]*x_dir_rotated[0], x_dir[0]*x_dir_rotated[0] + x_dir[1]*x_dir_rotated[1])
    theta = mod(_theta + car_angle_offset, 2*pi)

    q_car = array((car_t_ground_frame[0][0], car_t_ground_frame[1][0], theta))

    qd_car = (finite_difference(q_car, prev_q_car, dt) + prev_qd_car) / 2.0 # Average of previous computed velocity and positional finite differences

    def discontinuity():
        return -sign(qd_car[2]) * min(array([
            abs(finite_difference(qd_car[2]-2*pi, prev_qd_car[2], dt)),
            abs(finite_difference(qd_car[2]+2*pi, prev_qd_car[2], dt))
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
    # Corners of the playing area
    corner_0 = array([env.car_limits.x_min, env.car_limits.y_min, env.playing_area.floor_height], dtype=float32)
    corner_1 = array([env.car_limits.x_min, env.car_limits.y_max, env.playing_area.floor_height], dtype=float32)
    corner_2 = array([env.car_limits.x_max, env.car_limits.y_min, env.playing_area.floor_height], dtype=float32)
    corner_3 = array([env.car_limits.x_max, env.car_limits.y_max, env.playing_area.floor_height], dtype=float32)

    # 2D Euclidean distance from car to the corners of the playing area
    dcc_0 = norm(corner_0[0:2] - q_car[0:2], ord=2)[newaxis]
    dcc_1 = norm(corner_1[0:2] - q_car[0:2], ord=2)[newaxis]
    dcc_2 = norm(corner_2[0:2] - q_car[0:2], ord=2)[newaxis]
    dcc_3 = norm(corner_3[0:2] - q_car[0:2], ord=2)[newaxis]

    # 2D Euclidean distance from goal to the corners of the playing area
    dgc_0 = norm(corner_0[0:2] - p_goal, ord=2)[newaxis]
    dgc_1 = norm(corner_1[0:2] - p_goal, ord=2)[newaxis]
    dgc_2 = norm(corner_2[0:2] - p_goal, ord=2)[newaxis]
    dgc_3 = norm(corner_3[0:2] - p_goal, ord=2)[newaxis]

    # 3D Euclidean distance from ball to the corners of the playing area
    dbc_0 = norm(corner_0 - p_ball, ord=2)[newaxis]
    dbc_1 = norm(corner_1 - p_ball, ord=2)[newaxis]
    dbc_2 = norm(corner_2 - p_ball, ord=2)[newaxis]
    dbc_3 = norm(corner_3 - p_ball, ord=2)[newaxis]

    # 2D Euclidean distance drom car to the goal
    dc_goal = norm(p_goal[0:2] - q_car[0:2], ord=2)[newaxis]

    # 3D Euclidean distance from ball to the car (target)
    db_target = norm(array([q_car[0], q_car[1], env.playing_area.floor_height + ZeusDimensions.target_height]) - p_ball[0:3], ord=2)[newaxis]

    return array([
        dc_goal,
        dcc_0, dcc_1, dcc_2, dcc_3,
        dgc_0, dgc_1, dgc_2, dgc_3,
        dbc_0, dbc_1, dbc_2, dbc_3,
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
    dt: float,
    car_pos_offset_x: float = 0.0,
    car_pos_offset_y: float = 0.0
    ) -> tuple[Array, tuple[Array, ...]]:

    q_car, qd_car, car_R_gf, car_t_gf = observe_car(
        floor_R, floor_t, car_R, car_t, prev_q_car, prev_qd_car, dt,
        car_pos_offset_x=car_pos_offset_x,
        car_pos_offset_y=car_pos_offset_y
    )
    q_arm, qd_arm = observe_arm(robot_state)
    q_gripper, qd_gripper = observe_gripper(gripper_state, gripping)

    # TODO: ball tracking
    # p_ball, pd_ball = observe_ball()
    p_ball = array([0.0, 0.0, 1.3])
    pd_ball = array([0.0, 0.0, 0.0])

    relative_distances = observe_relative_distances(q_car, p_goal, p_ball, env)

    obs = concat_obs(q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, pd_ball, p_goal, relative_distances)

    return obs, (car_R_gf, car_t_gf)
# --------------------------------------------------------------------------------
