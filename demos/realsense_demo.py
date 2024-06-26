import numpy as np
import cv2
import pyrealsense2 as rs
from pupil_apriltags import Detector, Detection

from pprint import pprint


def draw_axis(img, R, t, K):
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1], [0, 0, 0]]).reshape(-1, 3) # inverted z axis
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, DIST_COEFFS)
    axisPoints = axisPoints.astype(int)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img


if __name__ == "__main__":
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.disable_all_streams()
    # cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
    cfg.enable_stream(rs.stream.color, 0, 848, 480, rs.format.rgb8, 60)
    profile = pipe.start(cfg)
    device = profile.get_device()
    print(device.query_sensors())
    sensor = device.query_sensors()[0]

    print(device.query_sensors())
    # rgb_sensor = device.query_sensors()[1]

    print(dir(rs.option))
    sensor.set_option(rs.option.emitter_enabled, 0)
    sensor.set_option(rs.option.laser_power, 0)
    sensor.set_option(rs.option.enable_auto_exposure, 0)
    sensor.set_option(rs.option.gain, 16) # default is 16
    sensor.set_option(rs.option.exposure, 3000)

    print(rs.option.hue.value)
    print(rs.option.saturation.value)

    # cam_intrinsics = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()
    cam_intrinsics = profile.get_stream(rs.stream.color, 0).as_video_stream_profile().get_intrinsics()

    FX: float = cam_intrinsics.fx
    FY: float = cam_intrinsics.fy
    CX: float = cam_intrinsics.ppx
    CY: float = cam_intrinsics.ppy
    CAMERA_PARAMS: tuple[float, ...]= (FX, FY, CX, CY)
    DIST_COEFFS: np.ndarray = np.array(cam_intrinsics.coeffs, dtype=np.float32)

    K = np.float32([[FX, 0, CX],
                    [0, FY, CY],
                    [0, 0, 1]])

    car_R = np.eye(3)
    car_t = np.zeros((3, 1))
    floor_R = np.eye(3)
    floor_t = np.zeros((3, 1))

    detector: Detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=1.5,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    print("Running april-tag detection...\nPress 'q' to exit")
    while True:
        frames = pipe.wait_for_frames()
        # frame = frames.get_infrared_frame()
        frame = frames.get_color_frame()
        if not frame: continue

        image = np.asarray(frame.data, dtype=np.uint8)

        crop_left_x = 152
        crop_right_x = 88
        crop_top_y = 0
        # crop_bottom_y = 0

        image = image[crop_top_y:, crop_left_x:-crop_right_x]

        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.equalizeHist(image)
        image = cv2.convertScaleAbs(image, alpha=0.5, beta=0.0)

        detections: list[Detection] = detector.detect(
            image, estimate_tag_pose=True,
            camera_params=CAMERA_PARAMS,
            tag_size=0.1475# 0.1858975
        ) # type: ignore[assignment]

        car_detection = list(filter(lambda d: d.tag_id == 0, detections))
        floor_detection = list(filter(lambda d: d.tag_id == 1, detections))

        if car_detection:
                car_R = car_detection[0].pose_R      # type: ignore[assignment]
                car_t = car_detection[0].pose_t       # type: ignore[assignment]

        if floor_detection:
                floor_R=floor_detection[0].pose_R    # type: ignore[assignment]
                floor_t=floor_detection[0].pose_t     # type: ignore[assignment]

        frame = draw_axis(image, car_R, car_t, K)
        frame = draw_axis(image, floor_R, floor_t, K)

        cv2.imshow("Realsense Camera", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    pipe.stop()
