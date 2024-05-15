import threading
from websockets.sync.client import connect
from enum import IntEnum
from time import monotonic_ns, sleep


ZEUS_URI = "ws://192.168.4.1:8765"


ZeusMode = IntEnum("ZeusMode", ["STANDBY", "ACT", "CONTINUE"])

class ZeusMessage:
    new = False
    msg = {"A": 0.0, "B": 0.0, "C": 0.0, "D": ZeusMode.STANDBY}
    mtx = threading.Lock()

    def write(self, A: float, B: float, C: float, D: ZeusMode) -> None:
        with self.mtx:
            self.msg.update([("A", A), ("B", B), ("C", C), ("D", D)])
            self.new = True

    def read(self) -> dict | None:
        with self.mtx:
            if not self.new:
                return None
            self.new = False

            return self.msg

class Visualization:
    data = (1,2,3,4,5,6,7,9,10)
    mtx = threading.Lock()

    def update_data(self, data) -> None:
        with self.mtx:
            self.data = data

def websocket_client(
        msg: ZeusMessage,
        msg_event: threading.Event,
        exit_event: threading.Event,
        num_stop_cmds: int = 5
    ) -> None:

    with connect(ZEUS_URI) as socket:
        while not exit_event.is_set():
            msg_event.wait()
            if m := msg.read() is not None:
                socket.send(json.dumps(m))
            msg_event.clear()

        for _ in range(num_stop_cmds):
            socket.send(json.dumps(ZeusMessage().msg))

    print("\nDisconnected from ZeusCar...")


def process(direction: int) -> tuple[float, float, float, int]:
    sleep(0.01)
    return 0.0, 0.0, direction*1.0, ZeusMode.ACT

def inner_loop() -> None:
    sleep(0.01)
    return (1,2,3,4,5,6,7,8,9,10)

def visualize(
        vis: Visualization,
        vis_event: threading.Event
    ) -> None:

    while True:
        vis_event.wait()
        with vis.mtx:
            sleep(0.01)
        vis_event.clear()

def loop_body(
        msg: ZeusMessage,
        vis: Visualization,
        msg_event: threading.Event,
        exit_event: threading.Event,
        vis_event: threading.Event
    ) -> int:

    global counter, d # temporary
    ctrl_time_ns = 0.04

    d = -d
    magnitude, angle, rot_vel, mode = process(d)
    msg.write(magnitude, angle, rot_vel, mode)
    msg_event.set()

    start_ns = monotonic_ns()
    while monotonic_ns() - start_ns < ctrl_time_ns:
        data = inner_loop()
        vis.update_data(data)
        vis_event.set()

    if counter >= 1000:
        exit_event.set()
        return 0

    counter += 1



def main() -> None:
    # Setup a bunch of stuff
    msg = ZeusMessage()
    vis = Visualization()
    msg_event = threading.Event()
    exit_event = threading.Event()
    vis_event = threading.Event()

    websocket_client_thread = threading.Thread(target=websocket_client, args=(msg, msg_event, exit_event))
    vis_loop_thread = threading.Thread(target=visualize, args=(vis, vis_event))
    # loop_body_thread = threading.Thread(target=loop_body)

    websocket_client_thread.start()
    vis_loop_thread.start()

    # loop_body_thread.run()
    while True:
        loop_body(msg, vis, msg_event, exit_event, vis_event)

    websocket_client_thread.join()
    vis_loop_thread.join()


if __name__ == "__main__":
    counter = 0
    d = 1.0
    main()
