import numpy as np
import cv2
from pupil_apriltags import Detector, Detection
from mediapy import write_video


# from https://stackoverflow.com/questions/50402026/what-is-the-intrinsicmatrix-for-an-iphone-x-rear-camera
# and using swapped x and y axis since they mentioned 1920x1080 whereas I'm using 1080x1920
FX: float = 2.05341424e+03
FY: float = 1.97547873e+03
CX: float = 5.13500761e+02
CY: float = 1.06077279e+03

CAMERA_PARAMS: tuple[float, ...]= (FX, FY, CX, CY)

K = np.float32([[FX, 0, CX],
                [0, FY, CY],
                [0, 0, 1]])

INPUT_VIDEO_PATH: str = "demos/assets/apriltag_lab_demo.mp4"
OUTPUT_VIDEO_PATH: str = "demos/assets/apriltag_demo_output.mp4"


class IterableVideoCapture(cv2.VideoCapture):
    def __iter__(self):
        return self

    def __next__(self):
        more_left, frame = self.read()
        if more_left:
            return frame
        else:
            raise StopIteration

def draw_axis(img, R, t, K):
    rotV, _ = cv2.Rodrigues(R)
    points = np.float32([[0.1, 0, 0], [0, 0.1, 0], [0, 0, -0.1], [0, 0, 0]]).reshape(-1, 3) # inverted z axis
    axisPoints, _ = cv2.projectPoints(points, rotV, t, K, (0, 0, 0, 0))
    axisPoints = axisPoints.astype(int)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[0].ravel()), (255,0,0), 3)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[1].ravel()), (0,255,0), 3)
    img = cv2.line(img, axisPoints[3].ravel(), tuple(axisPoints[2].ravel()), (0,0,255), 3)
    return img

if __name__ == "__main__":
    detector: Detector = Detector(
        families="tag36h11",
        nthreads=1,
        quad_decimate=0.0,
        quad_sigma=1.0,
        refine_edges=1,
        decode_sharpening=0.25,
    )

    capture: IterableVideoCapture = IterableVideoCapture(INPUT_VIDEO_PATH)

    frames = []
    R = np.eye(3)
    t = np.zeros((3, 1))
    for frame in capture:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        detection: Detection = detector.detect(gray_frame, estimate_tag_pose=True, camera_params=CAMERA_PARAMS, tag_size=0.14)
        if len(detection) > 0:
            R = detection[0].pose_R
            t = detection[0].pose_t

        frame = draw_axis(frame, R, t, K)
        frames.append(frame)

    write_video(OUTPUT_VIDEO_PATH, frames, fps=30)
