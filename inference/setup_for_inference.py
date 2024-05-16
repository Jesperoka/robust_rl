import pyrealsense2 as rs
import reproducibility_globals
from pupil_apriltags import Detector
from os.path import dirname, abspath, join
from functools import partial
from numpy import array as np_array, eye, zeros, float32
from cv2.typing import MatLike
from mujoco import MjModel, mjtObj, mj_name2id # type: ignore[attr-defined]
from mujoco.mjx import Model, put_model
from environments.A_to_B_jax import A_to_B
from environments.options import EnvironmentOptions
from numba import njit
from jax import Array, jit
from jax.numpy import newaxis, array, mod, arctan2, min, abs, clip, sign, pi, concatenate
from jax.numpy.linalg import norm
from jax.random import split, PRNGKey
from jax.tree_util import tree_map
from orbax.checkpoints import Checkpointer, PyTreeCheckpointHandler, args
from algorithms.utils import ActorInput, initialize_actors
from inference.controllers import gripper_ctrl, arm_spline_tracking_controller
from environments.reward_functions import curriculum_reward
from environments.physical import ZeusDimensions
from panda_py.libfranka import GripperState, RobotState



# Setup functions
# --------------------------------------------------------------------------------
CURRENT_DIR = dirname(abspath(__file__))
CHECKPOINT_DIR = join(CURRENT_DIR, "..", "trained_policies", "checkpoints")
CHECKPOINT_FILE = "checkpoint_LATEST"
MODEL_DIR = "mujoco_models"
MODEL_FILE = "scene.xml"

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

    return env


def setup_camera() -> tuple[rs.pipline, tuple[float, ...], MatLike, MatLike, MatLike, MatLike, int, int, Detector]:
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
    stereo_sensor.set_option(rs.option.gain, 32) # default is 16
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
    act_sizes = (space.sample().shape[0] for space in env.act_spaces)
    num_envs, num_agents, lr, max_grad_norm = 1, 2, 1e-3, 0.5

    sequence_length, num_envs = 1, 1
    assert sequence_length == 1 and num_envs == 1
    action_rngs = tuple(split(rng))
    actors, actor_hidden_states = initialize_actors(action_rngs, num_envs, num_agents, obs_size, act_sizes, lr, max_grad_norm, rnn_hidden_size, rnn_fc_size)
    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    restored_actors = checkpointer.restore(join(CHECKPOINT_DIR, checkpoint_file), state=actors, args=args.PyTreeRestore(actors))

    return restored_actors, actor_hidden_states
# --------------------------------------------------------------------------------



# Inference functions
# --------------------------------------------------------------------------------
@partial(jit, static_argnames=("num_agents", "done"))
def policy_inference(num_agents, actors, actor_hidden_states, observation, done):
    inputs = tuple(
            ActorInput(observation[newaxis, :][newaxis, :], done[newaxis, :])
            for _ in range(num_agents)
    )

    _, policies = zip(*tree_map(
        lambda ts, vars, hs, inputs: ts.apply_fn({"params": ts.params, "vars": vars}, hs, inputs, train=False),
            actors.train_states,
            actors.vars,
            actor_hidden_states,
            inputs,
        is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
    ))
    actions = tree_map(lambda policy: policy.mode().squeeze(), policies, is_leaf=lambda x: not isinstance(x, tuple))

    return actions, actor_hidden_states


@jit
def finite_difference(x, prev_x, dt):
    return (1.0/dt) * (x - prev_x)


@partial(jit, static_argnames=("angle_offset", "max_abs_vel"))
def observe_car(
    ground_R: Array,
    ground_t: Array,
    car_R: Array,
    car_t: Array,
    prev_q_car: Array,
    prev_qd_car: Array,
    dt: float,
    angle_offset: float = pi,
    max_abs_vel:  Array = array([1.0, 1.0, 1.0])
    ) -> tuple[Array, Array, Array, Array]:

    # Compute the relative pose of the car with respect to the ground frame
    car_t_ground_frame = ground_R.T @ (car_t - ground_t)
    car_R_ground_frame = ground_R.T @ car_R

    x_dir = array((1.0, 0.0))
    x_dir_rotated = car_R_ground_frame[0:2, 0:2] @ x_dir # approximate rotation about z-axis
    _theta = -arctan2(x_dir_rotated[1]*x_dir[0] - x_dir_rotated[0]*x_dir[1], x_dir_rotated[0]*x_dir[0] + x_dir_rotated[1]*x_dir[1])
    theta = mod(_theta + angle_offset, 2*pi)

    q_car = array((car_t_ground_frame[0][0], -car_t_ground_frame[1][0], theta)) # car_x == cam_z, car_y == -cam_x

    qd_car = (finite_difference(q_car, prev_q_car, dt) + prev_qd_car) / 2.0 # Average of previous computed velocity and positional finite differences
    qd_car[2] = qd_car[2] if abs(qd_car[2]) <= 5.0 else -sign(qd_car[2])*min(array([abs(finite_difference(qd_car[2]-2*pi, prev_qd_car[2], dt)), abs(finite_difference(qd_car[2]+2*pi, prev_qd_car[2], dt))]))
    qd_car = clip(qd_car, -max_abs_vel, max_abs_vel)

    return q_car, qd_car, car_R_ground_frame, car_t_ground_frame


@njit
def observe_arm(robot_state: RobotState) -> tuple[Array, Array, Array]:
    return array(robot_state.q, dtype=float32), array(robot_state.dq, dtype=float32), array(robot_state.ddq_d, dtype=float32)


@njit
def observe_gripper(gripper_state: GripperState, gripping: bool) -> tuple[Array, Array]:
    q_gripper = 0.5*gripper_state.width
    vel = 0.0 if (gripping or gripper_state.width >= 0.075) else 0.05
    return array([q_gripper, q_gripper], dtype=float32), array([vel, vel], dtype=float32)


@partial(jit, static_argnames=("env",))
def observe_relative_distances(
    q_car: Array,
    p_goal: Array,
    p_ball: Array,
    env: A_to_B
    ) -> list[Array]:
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

    return [
        dc_goal,
        dcc_0, dcc_1, dcc_2, dcc_3,
        dgc_0, dgc_1, dgc_2, dgc_3,
        dbc_0, dbc_1, dbc_2, dbc_3,
        db_target
    ]


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
    p_goal: Array,
    ground_R: Array,
    ground_t: Array,
    car_R: Array,
    car_t: Array,
    prev_q_car: Array,
    prev_qd_car: Array,
    robot_state: RobotState,
    gripper_state: GripperState,
    gripping: bool,
    dt: float
    ) -> tuple[Array, tuple[Array, ...]]:

    q_car, qd_car, car_R_gf, car_t_gf = observe_car(ground_R, ground_t, car_R, car_t, prev_q_car, prev_qd_car, dt)
    q_arm, qd_arm, qdd_d_arm = observe_arm(robot_state)
    q_gripper, qd_gripper = observe_gripper(gripper_state, gripping)

    # TODO: ball tracking
    # p_ball, pd_ball = observe_ball()
    p_ball = array([0.0, 0.0, 1.3])
    pd_ball = array([0.0, 0.0, 0.0])

    relative_distances = observe_relative_distances(q_car, p_goal, p_ball, env)

    obs = concat_obs(q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, pd_ball, p_goal, relative_distances)

    return obs, (car_R_gf, car_t_gf)
# --------------------------------------------------------------------------------
