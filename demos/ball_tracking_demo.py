import os
from os.path import join, dirname, abspath
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics.models import YOLO
from torch import set_num_threads,  set_num_interop_threads
from torch.onnx import is_onnxrt_backend_supported

set_num_threads(1)
set_num_interop_threads(1)

def main():
    # cache_dir = join(dirname(abspath(__file__)), "../", "compiled_functions")

    model = YOLO("yolov8s-worldv2.pt", task="detect")

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
        "sphere",
        "round ball",
        "orb",
    ])
    model.compile(mode="max-autotune-no-cudagraphs", backend="inductor")

    os.sched_setaffinity(0, {3})

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
        if not frame: continue

        image = np.asarray(frame.data, dtype=np.uint8)

        crop_x_left = 152
        crop_x_right = 152
        crop_y_top = 0
        crop_y_bottom = 32
        image = image[crop_y_top:-crop_y_bottom, crop_x_left:-crop_x_right]

        reduction = 3
        imgsz = (480-crop_y_bottom-crop_y_top-32*reduction, 848-crop_x_left-crop_x_right-32*reduction)

        image = cv2.resize(image, imgsz[::-1], interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        def ball_pos_cam_frame(
                xywh,
                diameter=0.062,
                fx=426.0619812011719,
                fy=426.0619812011719,
                cx=425.89910888671875,
                cy=232.21575927734375
            ) -> np.ndarray:

            pixel_diameter = sum(xywh[2:])/2.0
            x, y = xywh[0], xywh[1]
            z = (diameter * fx) / pixel_diameter
            x = (x - cx) * z / fx
            y = (y - cy) * z / fy

            return np.array([x, y, z])

        results = model.predict(
            source=image,
            stream=False,
            verbose=False,
            conf=0.03,
            imgsz=imgsz,
            agnostic_nms=True,
            max_det=1,
            augment=True,
            iou=0.4
        )

        # print(results[0].boxes.xywh[0])
        res = results[0].boxes.xywh.squeeze()
        if not res.shape[0] == 0:
            print(ball_pos_cam_frame(res))
        else:
            print("No ball detected")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = results[0].plot(img=image)



        print()

        cv2.imshow("Realsense Camera", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn")
    process = mp.Process(target=main)
    process.start()
    # main()
    process.join()
