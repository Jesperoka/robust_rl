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
from time import clock_gettime_ns, CLOCK_REALTIME, CLOCK_MONOTONIC
from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler, args 
from algorithms.utils import initialize_actors, ActorInput
from inference.controllers import arm_fixed_pose, gripper_always_grip 
from inference.processing import LowPassFilter
from environments.reward_functions import car_only_negative_distance 
from environments.options import EnvironmentOptions
from environments.physical import PlayingArea, PandaLimits, ZeusLimits
from environments.A_to_B_jax import A_to_B
from numba import jit as njit 


from pprint import pprint

CURRENT_DIR = dirname(abspath(__file__))
CHECKPOINT_DIR = join(CURRENT_DIR, "..", "trained_policies", "checkpoints")
CHECKPOINT_FILE = "checkpoint_LATEST"
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
    # theta = np.arccos(0.5 * (np.trace(car_R_ground_frame) - 1.0))

    OFFSET = np.pi #0.0 # CHANGE THIS TO FIT WITH THE ORIENATION OF THE CAR APRIL-TAG
    MAX_ABS_VEL = np.array([1.0, 1.0, 1.0])

    x_dir = np.array((1.0, 0.0))
    x_dir_rotated = car_R_ground_frame[0:2, 0:2] @ x_dir # approximate rotation about z-axis
    _theta = -np.arctan2(x_dir_rotated[1]*x_dir[0] - x_dir_rotated[0]*x_dir[1], x_dir_rotated[0]*x_dir[0] + x_dir_rotated[1]*x_dir[1])
    theta = np.mod(_theta + OFFSET, 2*np.pi)
    # theta = np.arccos(np.dot(x_dir, x_dir_rotated) / (np.linalg.norm(x_dir, ord=2)*np.linalg.norm(x_dir_rotated, ord=2)))

    q_car = np.array((car_t_ground_frame[0][0], -car_t_ground_frame[1][0], theta)) # car_x == cam_z, car_y == -cam_x

    qd_car = (finite_difference(q_car, prev_q_car, dt) + prev_qd_car) / 2.0 # Average of previous computed velocity and positional finite differences
    qd_car[2] = qd_car[2] if abs(qd_car[2]) <= 5.0 else -np.sign(qd_car[2])*min(abs(finite_difference(qd_car[2]-2*np.pi, prev_qd_car[2], dt)), abs(finite_difference(qd_car[2]+2*np.pi, prev_qd_car[2], dt)))
    qd_car = np.clip(np.round(qd_car, 2), -MAX_ABS_VEL, MAX_ABS_VEL)

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

    p_goal = np.array([1.4428220, -0.6173251])
    dc_goal = np.array([np.linalg.norm(p_goal[0:2] - q_car[0:2], ord=2)])
     
    obs = np.concatenate((
        q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, pd_ball, p_goal, dc_goal, #dcc_0, dcc_1, dcc_2, dcc_3, dgc_0, dgc_1, dgc_2, dgc_3
    ), axis=0)

    return obs, aux

# Zeus Modes
STANDBY = 0
ACT = 1 
CONTINUE = 2 # don't need to send

async def websocket_client(queue, uri="ws://192.168.4.1:8765"): # ESP32-CAM IP address when it creates an Access Point
    done = {"A": 0, "B": 0, "C": 0, "D": STANDBY}
    async for websocket in websockets.connect(uri):
        try:
            print("\nConnected to Zeus Car.\n")
            while True:
                if not queue.empty():
                    message = await queue.get()

                    if message is not None:
                        for _ in range(2):
                            await websocket.send(dumps(message))

                    elif message is None:
                        print("Closing connection to Zeus.")
                        for _ in range(10):
                            await websocket.send(dumps(done))
                        await websocket.close()
                        return None

                await asyncio.sleep(0.001)

        except websockets.ConnectionClosed:
            if not queue.empty() and queue.get_nowait() is None:
                print("Closing connection to Zeus.")
                for _ in range(10):
                    await websocket.send(dumps(done))
                await websocket.close()
                return None
            print("\nConnection with Zeus Car lost.\n")

EPSILON = 0.005

def main():
    env = setup_env()

    ctrl_time = 0.04
    ctrl_time_ns = int(ctrl_time * 1.0e9)

    ground_R = np.eye(3)
    ground_t = np.zeros((3, 1))

    prev_q_car = np.zeros(3)
    prev_qd_car = np.zeros(3)

    rng = PRNGKey(reproducibility_globals.PRNG_SEED)
    obs_size = env.obs_space.sample().shape[0]
    act_sizes = (space.sample().shape[0] for space in env.act_spaces)
    num_envs = 1
    num_agents = 2
    rnn_hidden_size = 16 
    rnn_fc_size = 64 
    lr = 1e-3
    max_grad_norm = 0.5

    actors, actor_hidden_states = load_policies(rng, obs_size, act_sizes, num_envs, num_agents, rnn_hidden_size, rnn_fc_size, lr, max_grad_norm)

    detector: Detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.5,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    pipe, cam_params, K, car_R, car_t = setup_camera()
    frame = pipe.wait_for_frames().get_infrared_frame()

    car_command_queue = asyncio.Queue()
    print("Running Zeus websocket client ...")

    
    lowpass_obs = LowPassFilter(input_shape=(obs_size, ), history_length=10, bandlimit_hz=0.75, sample_rate_hz=1.0/(ctrl_time + EPSILON))

    dt = ctrl_time

    async def loop_body():
        nonlocal pipe, car_R, car_t, ground_R, ground_t, prev_q_car, prev_qd_car, actor_hidden_states, frame, dt 

        print("Running april-tag detection...\nPress 'q' to exit")
        done = False
        while not done:
            start_ns = clock_gettime_ns(CLOCK_MONOTONIC)
            observation, aux = observe(ground_R, ground_t, car_R, car_t, prev_q_car, prev_qd_car, dt) 
            observation = lowpass_obs(observation)

            (
                q_car, q_arm, q_gripper, p_ball, 
                qd_car, qd_arm, qd_gripper, pd_ball, 
                p_goal, 
                dc_goal,
                dcc_0, dcc_1, dcc_2, dcc_3,
                dgc_0, dgc_1, dgc_2, dgc_3,
                dbc_0, dbc_1, dbc_2, dbc_3,
                db_target
             ) = env.decode_observation(observation)

            if dc_goal <= 0.1:
                print("Goal Reached...")
                car_command_queue.put_nowait(None)
                done = True 
                return None

            prev_q_car = q_car
            prev_qd_car = qd_car

            terminal = np.array([False])
            actions, actor_hidden_states = jit(policy_inference, static_argnums=0)(num_agents, actors, actor_hidden_states, observation, terminal)
            action = np.concatenate(actions, axis=-1)

            car_orientation = q_car[2]

            print(f"Car x: {q_car[0]}, Car y: {q_car[1]}, Car theta: {q_car[2]}, Car vx: {qd_car[0]}, Car vy: {qd_car[1]}, Car omega: {qd_car[2]}, dt: {dt}, time-error: {dt - ctrl_time}")

            # ctrl = jit(env.compute_controls)(car_orientation, observation, action)
            action = env.scale_action(action[0:3], env.act_space_car.low, env.act_space_car.high)

            # TODO: add modifier based on omega as well
            def car_velocity_modifier(theta):
                return 0.5 + 0.5*( np.abs( ( np.mod(theta, (np.pi/2.0)) ) - (np.pi/4.0) ) / (np.pi/4.0) )

            def rotation_velocity_modifier(velocity, omega):
                return np.clip(velocity - np.abs(omega), 0.0, velocity)

            # action[0] = car_velocity_modifier(action[1])*action[0]
            action[0] = rotation_velocity_modifier(action[0], action[2])

            # NOTE: do I need to scale the velocity/magnitude based on angle? or is it better to just let the hardware do whatever it can?
            # action = env.
            
            car_command = {"A": round(float(action[0]), 4), "B": round(float(action[1]), 4), "C": round(float(action[2]), 4), "D": ACT}
            print(car_command)
            # car_command = {"A": round(float(action[0]), 3), "B": round(float(action[1]), 3), "C": round(float(0.0), 3), "D": ACT}
            # car_command = {"A": round(0.0, 2), "B": round(float(np.mod(0.0, 2*np.pi)), 2), "C": round(1.000000011234124, 2), "D": ACT}
            car_command_queue.put_nowait(car_command)

            while clock_gettime_ns(CLOCK_MONOTONIC) - start_ns <= ctrl_time_ns - EPSILON:
                await asyncio.sleep(EPSILON/2.0)

                _frame = pipe.wait_for_frames().get_infrared_frame()
                frame = _frame if _frame else frame

                image = np.asarray(frame.data, dtype=np.uint8)
                detections: list[Detection] = detector.detect(image, estimate_tag_pose=True, camera_params=cam_params, tag_size=0.1858975) # type: ignore[assignment]

                car_detection = list(filter(lambda d: d.tag_id == 0, detections))
                floor_detection = list(filter(lambda d: d.tag_id == 1, detections))

                if car_detection:
                    car_R = car_detection[0].pose_R
                    car_t = car_detection[0].pose_t

                if floor_detection:
                    ground_R = floor_detection[0].pose_R
                    ground_t = floor_detection[0].pose_t

                frame = draw_axes(image, car_R, car_t, K)
                frame = draw_axes(image, ground_R, ground_t, K)
                # print("\n\n::::: ", car_R, car_t, " :::::\n\n")

                cv2.imshow("image", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exiting...")
                    car_command_queue.put_nowait(None)
                    done = True 

            dt = (clock_gettime_ns(CLOCK_MONOTONIC) - start_ns) * 1e-9


        return None

    async def gather():
        res = await asyncio.gather(loop_body(), websocket_client(car_command_queue))

    asyncio.run(gather())
    pipe.stop()


if __name__ == "__main__":
    main()
