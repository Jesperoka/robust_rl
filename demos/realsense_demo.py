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
    cfg.enable_stream(rs.stream.infrared, 1, 848, 480, rs.format.y8, 90)
    profile = pipe.start(cfg)
    device = profile.get_device()
    stereo_sensor = device.query_sensors()[0]

    print(device.query_sensors())
    rgb_sensor = device.query_sensors()[1]

    stereo_sensor.set_option(rs.option.emitter_enabled, 0)
    stereo_sensor.set_option(rs.option.laser_power, 0)
    stereo_sensor.set_option(rs.option.enable_auto_exposure, 0)
    stereo_sensor.set_option(rs.option.gain, 32) # default is 16
    stereo_sensor.set_option(rs.option.exposure, 3300)

    # pprint(dir(rs.option))
    print(rs.option.contrast.value)
    print(rs.option.white_balance.value)

    cam_intrinsics = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics()

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
        quad_decimate=0.0,
        quad_sigma=0.0,
        refine_edges=0,
        decode_sharpening=0.0
    )

    print("Running april-tag detection...\nPress 'q' to exit")
    while True:
        frames = pipe.wait_for_frames()
        frame = frames.get_infrared_frame()
        if not frame: continue

        image = np.asarray(frame.data, dtype=np.uint8)
        detections: list[Detection] = detector.detect(
            image, estimate_tag_pose=True,
            camera_params=CAMERA_PARAMS,
            tag_size=0.1858975
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
