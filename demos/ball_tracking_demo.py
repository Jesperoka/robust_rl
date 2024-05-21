
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics.models import YOLO

def main():
    model = YOLO("yolov8s-world.pt", task="detect")
    model.set_classes([
        "stressball",
        "baseball",
        "softball",
        "sportsball",
        "white ball",
        "gray ball",
        "grey ball",
        "small white ball",
        "small ball",
        "tiny ball",
        "ball",
        "fast-moving ball",
        "fastball",
        "moving ball",
        "blurry ball",
        # "motion blurred ball",
        "large ball",
        "tiny ball",
        "toy ball",
        "rubber ball",
        "industrial ball",
        "flying ball",
        "bouncing ball",
        "rolling ball",
        "spinning ball",
        "rotating ball",
        "floating ball",
        "hovering ball",
        "spherical object",
        "round object",
        "circular object",
        "sphere",
        "round ball",
        "round thing",
        "orb",
    ])

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
    # cfg.enable_stream(rs.stream.color, 1, 848, 480, rs.format.bgr8, 60)
    profile = pipe.start(cfg)
    device = profile.get_device()
    stereo_sensor = device.query_sensors()[0]

    stereo_sensor.set_option(rs.option.emitter_enabled, 0)
    stereo_sensor.set_option(rs.option.laser_power, 0)
    stereo_sensor.set_option(rs.option.enable_auto_exposure, 0)
    stereo_sensor.set_option(rs.option.gain, 32) # default is 16 (had 32 before)
    stereo_sensor.set_option(rs.option.exposure, 3300)

    while True:
        frames = pipe.wait_for_frames()
        frame = frames.get_infrared_frame()
        # frame = frames.get_color_frame()
        if not frame: continue

        image = np.asarray(frame.data, dtype=np.uint8)
        # image = cv2.resize(image, (640, 640))
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)#.reshape((640, 640, 3))
        results = model.predict(source=image, stream=False, conf=0.001, imgsz=(480, 864), agnostic_nms=False, max_det=1, augment=True, iou=0.9)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = results[0].plot(img=image)
        print()

        cv2.imshow("Realsense Camera", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

if __name__ == "__main__":
    main()
