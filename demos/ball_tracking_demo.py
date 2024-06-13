import os
from os.path import join, dirname, abspath
os.sched_setaffinity(0, {1,2,3})
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics.models import YOLO
from torch import set_num_threads,  set_num_interop_threads
from torch.onnx import is_onnxrt_backend_supported

set_num_threads(4)
set_num_interop_threads(4)

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
        # "motion blurred ball", # finds larger objects
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
        "camoflauged ball",
        "snowball",
        "snow ball",
        "ice ball",
        # "ball in robot gripper",
        # "ball in robot claw",
        # "ball between robot fingers",
        # "ball in robot end effector",
    ])
    model.compile(mode="max-autotune-no-cudagraphs", backend="inductor")

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    # cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)

    formats = ["rgb8"]

    stream_index = 0

    for format in formats:
        try:
            print("Trying to enable stream with format: ", format)
            cfg.enable_stream(rs.stream.color, stream_index, 848, 480, getattr(rs.format, format), 60)
            profile = pipe.start(cfg)
            print("Stream enabled with format: ", format)
            break
        except Exception as e:
            print("Failed to enable stream with format: ", format)
            print(e)

    device = profile.get_device()
    sensor = device.query_sensors()[0]

    sensor.set_option(rs.option.emitter_enabled, 0)
    sensor.set_option(rs.option.laser_power, 0)
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.gain, 16) # default is 16 (had 32 before)
    sensor.set_option(rs.option.exposure, 3300)
    # sensor.set_option(rs.option.hue, )

    # cam_intrinsics = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
    cam_intrinsics = profile.get_stream(rs.stream.color, stream_index).as_video_stream_profile().get_intrinsics()

    FX: float = cam_intrinsics.fx
    FY: float = cam_intrinsics.fy
    CX: float = cam_intrinsics.ppx
    CY: float = cam_intrinsics.ppy

    while True:
        frames = pipe.wait_for_frames()
        # frame = frames.get_infrared_frame()
        frame = frames.get_color_frame()
        if not frame: continue

        image = np.asarray(frame.data, dtype=np.uint8)

        # For infrared
        # crop_x_left = 152
        # crop_x_right = 152
        # crop_y_top = 0
        # crop_y_bottom = 32

        # For color
        crop_x_left = 152
        crop_x_right = 88
        crop_y_top = 0
        crop_y_bottom = 0

        # image = image[crop_y_top:-crop_y_bottom, crop_x_left:-crop_x_right]
        image = image[crop_y_top:, crop_x_left:-crop_x_right]
        # image = image[crop_y_top:, crop_x_left:]

        reduction = 3
        imgsz = (480-crop_y_bottom-crop_y_top-32*reduction, 848-crop_x_left-crop_x_right-32*reduction)
        # imgsz = (480, 848)

        # image = cv2.convertScaleAbs(image, alpha=0.2, beta=0.5)
        image = cv2.resize(image, imgsz[::-1], interpolation=cv2.INTER_AREA)

        def ball_pos_cam_frame(
                xywh,
                diameter=0.062,
                fx=FX,
                fy=FY,
                cx=CX,
                cy=CY
            ) -> np.ndarray:

            # shift principal point based on cropping
            cx -= crop_x_left
            cy -= crop_y_top

            # scale principal point and focal lengths based on resize ratio
            # Note the swapped x and y
            scale_x = (imgsz[1] - 32.0*reduction) / float(imgsz[1])
            scale_y = (imgsz[0] - 32.0*reduction) / float(imgsz[0])
            fx *= scale_x
            fy *= scale_y
            cx *= scale_x
            cy *= scale_y

            # Scale the pixel diameter to correct for error in bouding box predictions
            scale_correction = 1.0 # DISABLED (2.6227 / 2.7532) #(2.2344/3.4709)

            pixel_diameter = sum(xywh[2:]) / 2.0
            pixel_diameter *= scale_correction

            x, y = xywh[0], xywh[1]
            z = (diameter * fx) / pixel_diameter
            x = (x - cx) * z / fx
            y = (y - cy) * z / fy

            return np.array([x, y, z])

        def observe_ball(
            p_ball_cam_frame: np.ndarray,
            ball_pos_offset_x: float = 0.0,
            ball_pos_offset_y: float = 0.0,
            ball_pos_offset_z: float = 0.0
                ) -> np.ndarray:

            floor_R = np.array([
                [-0.019886, -0.9998, -0.0032623],
                [0.63195, -0.015098, 0.77486],
                [-0.77475, 0.013347, 0.63212]
            ])

            floor_t = np.array([
                [0.15317],
                [0.17532],
                [2.7532]
            ])
            # Value we got for ball:  [     0.1724    0.097051      2.3837]
            # Value we got for ball:  [    0.12073     0.12233      2.2344]

            p_ball = floor_R.T @ (p_ball_cam_frame - floor_t.squeeze())
            p_ball[2] = -p_ball[2]

            return p_ball

        results = model.predict(
            source=image,
            stream=False,
            verbose=False,
            conf=0.02,
            imgsz=imgsz,
            agnostic_nms=True,
            max_det=1,
            augment=True,
            iou=0.4
        )

        # print(results[0].boxes.xywh[0])
        res = results[0].boxes.xywh.squeeze()
        if not res.shape[0] == 0:
            bpcf = ball_pos_cam_frame(res)
            print("\nCam frame: ", bpcf)
            print("\nWorld frame: ", observe_ball(bpcf))
        else:
            # print("No ball detected")
            pass

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = results[0].plot(img=image)

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
