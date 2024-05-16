import threading
import numpy as np
import cv2
from websockets.sync.client import connect
from enum import IntEnum
from time import clock_gettime_ns, CLOCK_BOOTTIME
from json import dumps
from inference.setup_for_inference import setup_camera, setup_env, load_policies, observe, policy_inference
from inference.processing import LowPassFilter
from environments.options import EnvironmentOptions as _EnvOpts


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

class Visualization:
    data: np.ndarray
    mtx = threading.Lock()

    def __init__(self, w, h):
        with self.mtx:
            self.data = np.zeros((w, h), dtype=np.int_)

    def update_data(self, data) -> None:
        with self.mtx:
            self.data = data

def draw_axes(img, R, t, K, dist_coeffs):
    rotV, _ = cv2.Rodrigues(R)
    points = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1], [0, 0, 0]], dtype=np.float32).reshape(-1, 3) # inverted z axis
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, dist_coeffs)
    axisPoints = axisPoints.astype(int)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[2].ravel()), (0,0,255), 3)

    return img


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
                print("Sending:", dumps(m))
                for _ in range(num_act_cmds):
                    socket.send(dumps(m))
            msg_event.clear()

        for _ in range(num_stop_cmds):
            socket.send(dumps(ZeusMessage().msg))

    print("\nDisconnected from ZeusCar...")



# WARNING: UNFINISHED (need to run low level control here)
def inner_loop() -> tuple:
    sleep(0.01)
    return (1,2,3,4,5,6,7,8,9,10)

# WARNING: UNFINISHED
def visualize(
        vis: Visualization,
        vis_event: threading.Event
    ) -> None:

    while True:
        vis_event.wait()
        with vis.mtx:
            print("Visualizing")
            sleep(0.01)
        vis_event.clear()

# WARNING: UNFINISHED
def loop_body(
        msg: ZeusMessage,
        vis: Visualization,
        msg_event: threading.Event,
        exit_event: threading.Event,
        vis_event: threading.Event,
        ctrl_time_ns: float = 0.04,
        lowpass_filter: LowPassFilter = LowPassFilter(input_shape=(_EnvOpts(None).obs_min.shape, ), history_length=10, bandlimit_hz=0.75, sample_rate_hz=1.0/0.04)
    ) -> bool:

    obs, aux = observe(
        # env: A_to_B,  # partial()
        # p_goal: Array,
        # ground_R: Array,
        # ground_t: Array,
        # car_R: Array,
        # car_t: Array,
        # prev_q_car: Array,
        # prev_qd_car: Array,
        # robot_state: RobotState,
        # gripper_state: GripperState,
        # gripping: bool,
        # dt: float
    )
    obs = lowpass_filter(obs)

    msg.write(magnitude, angle, rot_vel, mode)
    msg_event.set()

    start_ns = clock_gettime_ns(CLOCK_BOOTTIME)
    while clock_gettime_ns(CLOCK_BOOTTIME) - start_ns < ctrl_time_ns:
        data = inner_loop()
        vis.update_data(data)
        vis_event.set()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit_event.set()
        return True

    return False

# TODO: jit
def rotation_velocity_modifier(velocity, omega):
    return np.clip(velocity - np.abs(omega), 0.0, velocity)


def main() -> None:
    rnn_hidden_size = 16
    rnn_fc_size = 64
    env = setup_env()
    actors, actor_hidden_states = load_policies(env, rnn_hidden_size, rnn_fc_size, "checkpoint_LATEST")
    pipe, cam_params, dist_coeffs, K, R, t, width, height, detector = setup_camera()

    msg = ZeusMessage()
    vis = Visualization(width, height)
    msg_event = threading.Event()
    exit_event = threading.Event()
    vis_event = threading.Event()

    websocket_client_thread = threading.Thread(target=websocket_client, args=(msg, msg_event, exit_event))
    vis_loop_thread = threading.Thread(target=visualize, args=(vis, vis_event))

    websocket_client_thread.start()
    vis_loop_thread.start()

    # TODO: add outer loop that handles resetting
    finished = False
    while not finished:
        finished = loop_body(msg, vis, msg_event, exit_event, vis_event)

    websocket_client_thread.join()
    vis_loop_thread.join()


if __name__ == "__main__":
    main()
