import asyncio
import websockets
from json import dumps

import numpy as np
from inference import pyrealsense2 as rs
import cv2
import reproducibility_globals

from pupil_apriltags import Detector, Detection

from jax import tree_map, jit, device_get
from jax.random import PRNGKey, split
from mujoco import MjModel, mj_name2id, mjtObj
from mujoco.mjx import Model, put_model
from os.path import join, dirname, abspath
from time import monotonic_ns 
from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler, args 
from algorithms.utils import initialize_actors, ActorInput
from inference.controllers import arm_fixed_pose, gripper_always_grip 
from environments.reward_functions import car_only_negative_distance 
from environments.options import EnvironmentOptions
from environments.physical import PlayingArea, PandaLimits, ZeusLimits
from environments.A_to_B_jax import A_to_B
from numba import jit as njit 


from pprint import pprint

CURRENT_DIR = dirname(abspath(__file__))
CHECKPOINT_DIR = join(CURRENT_DIR, "..", "trained_policies", "checkpoints")
CHECKPOINT_FILE = "zeus_rnn_32"
MODEL_DIR = "mujoco_models"
MODEL_FILE = "scene.xml"


def draw_axes(img, R, t, K):
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1], [0, 0, 0]]).reshape(-1, 3) # inverted z axis
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    axisPoints = axisPoints.astype(int)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[2].ravel()), (0,0,255), 3)

    return img


def setup_env():
    scene = join(CURRENT_DIR, "..", MODEL_DIR, MODEL_FILE)

    model: MjModel = MjModel.from_xml_path(scene)
    mjx_model: Model = put_model(model)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")

    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = car_only_negative_distance,
        arm_ctrl       = arm_fixed_pose,
        gripper_ctrl   = gripper_always_grip,
        goal_radius    = 0.1,
        steps_per_ctrl = 20,
        time_limit     = 3.0,
        num_envs       = 1,
        prng_seed      = reproducibility_globals.PRNG_SEED,
        act_min        = np.concatenate([ZeusLimits().a_min, PandaLimits().tau_min, np.array([-1.0])], axis=0),
        act_max        = np.concatenate([ZeusLimits().a_max, PandaLimits().tau_max, np.array([1.0])], axis=0)
    )

    env = A_to_B(mjx_model, None, grip_site_id, options)

    return env



def load_policies(rng, obs_size, act_sizes, num_envs, num_agents, rnn_hidden_size, rnn_fc_size, lr, max_grad_norm):
    sequence_length, num_envs = 1, 1
    assert sequence_length == 1 and num_envs == 1
    action_rngs = tuple(split(rng))
    actors, actor_hidden_states = initialize_actors(action_rngs, num_envs, num_agents, obs_size, act_sizes, lr, max_grad_norm, rnn_hidden_size, rnn_fc_size)
    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    restored_actors = checkpointer.restore(join(CHECKPOINT_DIR, CHECKPOINT_FILE), state=actors, args=args.PyTreeRestore(actors))

    return restored_actors, actor_hidden_states
    

def policy_inference(num_agents, actors, actor_hidden_states, observation, done):
    inputs = tuple(
            ActorInput(observation[np.newaxis, :][np.newaxis, :], done[np.newaxis, :])
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


def setup_camera():
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

    K = np.float32([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]])

    R = np.eye(3)
    t = np.zeros((3, 1))

    return pipe, cam_params, K, R, t


@njit
def finite_difference(x, prev_x, dt):
    return (x - prev_x) / dt

@njit
def observe_car(ground_R, ground_t, car_R, car_t, prev_q_car, prev_qd_car, dt):
    # Reference frames of ground are pre-detected and known
    
    # Compute the relative pose of the car with respect to the ground frame
    car_t_ground_frame = ground_R.T @ (car_t - ground_t)
    car_R_ground_frame = ground_R.T @ car_R
    theta = np.arccos(0.5 * (np.trace(car_R_ground_frame) - 1.0))

    q_car = np.array((-car_t_ground_frame[0][0], car_t_ground_frame[1][0], theta)) # car_x == cam_z, car_y == -cam_x
    qd_car = (finite_difference(q_car, prev_q_car, dt) + prev_qd_car ) / 2.0 # Average of previous computed velocity and positional finite differences

    return q_car, qd_car, (car_R_ground_frame, car_t_ground_frame)

Q_START = np.array(device_get(PandaLimits().q_start))

# @njit
def observe(
        ground_R, 
        ground_t, 
        car_R, 
        car_t, 
        prev_q_car, 
        prev_qd_car, 
        dt
        ):

    q_car, qd_car, aux = observe_car(ground_R, ground_t, car_R, car_t, prev_q_car, prev_qd_car, dt)

    # The rest are dummy observations for now
    q_arm = Q_START
    qd_arm = np.zeros(7)
    q_gripper = np.array([0.2, 0.2])
    qd_gripper = np.zeros(2)
    p_ball = np.array([0.1, 0.0, 1.3])
    pd_ball = np.zeros(3)

    p_goal = np.array([-0.288717, -0.524066])
    dc_goal = np.array([np.linalg.norm(p_goal[0:2] - q_car[0:2], ord=2)])
     
    obs = np.concatenate((
        q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, pd_ball, p_goal, dc_goal, #dcc_0, dcc_1, dcc_2, dcc_3, dgc_0, dgc_1, dgc_2, dgc_3
    ), axis=0)

    return obs, aux


async def websocket_client(queue, uri="ws://192.168.4.1:8765"): # ESP32-CAM IP address when it creates an Access Point
    last_message = {} 
    async with websockets.connect(uri) as websocket:
        while True:
            if not queue.empty():
                message = queue.get_nowait()
                last_message = message
            else:
                message = last_message

            await websocket.send(dumps(message) + ">")
            await asyncio.sleep(0.005)


def main():
    env = setup_env()

    ctrl_time = 0.04
    ctrl_time_ns = int(ctrl_time * 1e9)

    ground_R = np.array([[-0.0584163 ,  0.99816874,  0.01570677], 
                         [-0.27988767, -0.03147855,  0.95951654], 
                         [ 0.95825384,  0.05165528,  0.28121398]])

    ground_t = np.array([[0.17442545], [0.58795911], [1.22108916]])

    prev_q_car = np.zeros(3)
    prev_qd_car = np.zeros(3)

    rng = PRNGKey(reproducibility_globals.PRNG_SEED)
    obs_size = env.obs_space.sample().shape[0]
    act_sizes = (space.sample().shape[0] for space in env.act_spaces)
    num_envs = 1
    num_agents = 2
    rnn_hidden_size = 32 
    rnn_fc_size = 64 
    lr = 1e-3
    max_grad_norm = 0.5

    actors, actor_hidden_states = load_policies(rng, obs_size, act_sizes, num_envs, num_agents, rnn_hidden_size, rnn_fc_size, lr, max_grad_norm)

    detector: Detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=0.0,
        quad_sigma=1.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    pipe, cam_params, K, car_R, car_t = setup_camera()
    frame = pipe.wait_for_frames().get_infrared_frame()

    car_command_queue = asyncio.Queue()
    loop = asyncio.get_event_loop()
    loop.create_task(websocket_client(car_command_queue))

    print("Running april-tag detection...\nPress 'q' to exit")
    done = False
    while not done:
        observation, aux = observe(ground_R, ground_t, car_R, car_t, prev_q_car, prev_qd_car, ctrl_time) 
        (q_car, _, _, _, qd_car, *_) = env.decode_observation(observation)
        prev_q_car = q_car
        prev_qd_car = qd_car

        terminal = np.array([False])
        actions, actor_hidden_states = jit(policy_inference, static_argnums=0)(num_agents, actors, actor_hidden_states, observation, terminal)
        action = np.concatenate(actions, axis=-1)

        car_orientation = q_car[2]

        print(f"Car x: {q_car[0]}, Car y: {q_car[1]}, Car theta: {q_car[2]}")

        ctrl = jit(env.compute_controls)(car_orientation, observation, action)
        
        car_command = {"A": ctrl[0], "B": ctrl[1], "C": ctrl[2], "D": 1 }
        car_command_queue.put_nowait(car_command)

        start_ns = monotonic_ns()
        while monotonic_ns() - start_ns < ctrl_time_ns:

            _frame = pipe.wait_for_frames().get_infrared_frame()
            frame = _frame if _frame else frame

            image = np.asarray(frame.data, dtype=np.uint8)
            detections: list[Detection] = detector.detect(image, estimate_tag_pose=True, camera_params=cam_params, tag_size=0.14) # type: ignore[assignment]

            if len(detections) > 0:
                car_R = detections[0].pose_R
                car_t = detections[0].pose_t

            frame = draw_axes(image, car_R, car_t, K)

            print("\n\n::::: ", car_R, car_t, " :::::\n\n")


            cv2.imshow("image", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting...")
                done = True 

    pipe.stop()


if __name__ == "__main__":
    main()
