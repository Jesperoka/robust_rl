import torch
import cv2
import onnxruntime as ort
import numpy as np
import time

import cv2.dnn

from ultralytics.models import YOLO
from pprint import pprint

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

VIDEO = "demos/assets/airplane_hack.mp4"

CLASSES = yaml_load(check_yaml("DOTAv1.5.yaml"))["names"]
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def draw_rotated_bounding_box(img, box_info):
    color = (0, 255, 0)
    vertices = box_info["box"].points().astype(int)
    cv2.line(img, vertices[0], vertices[1], color, 2)
    cv2.line(img, vertices[1], vertices[2], color, 2)
    cv2.line(img, vertices[2], vertices[3], color, 2)
    cv2.line(img, vertices[3], vertices[0], color, 2)
    label = f"{box_info['class_id']} ({box_info['score']:.2f})"
    text_pos = tuple(vertices[1] - np.array([0, 10]))
    cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def detect_rotated(model_output, scale, original_image):
    score_threshold = 0.25
    nms_threshold = 0.45
    nms_threshold_decay = 0.5

    print(model_output.shape)
    data = np.array([model_output[0]])[0]
    rows, cols = data.shape
    print(data.shape)

    x_factor = scale
    y_factor = scale 

    boxes = []
    BOXES = []
    scores = []

    for i in range(cols):
        class_scores = data[4:-1, i]                # first 4 are bounding box coordinates, last one is angle
        max_class_score = np.max(class_scores)
        class_id = np.argmax(class_scores)
        
        if max_class_score > score_threshold: 
            scores.append(max_class_score)
            x = data[0, i] * x_factor
            y = data[1, i] * y_factor
            w = data[2, i] * x_factor
            h = data[3, i] * y_factor
            angle = data[-1, i]

            if angle >= np.pi and angle <= 0.75 * np.pi:
                angle = angle - np.pi

            box = cv2.RotatedRect((x, y), (w, h), angle * 180 / np.pi)
            box_info = {
                "class_id": class_id,
                "score": max_class_score,
                "box": box, 
            }
            boxes.append(box)
            BOXES.append(box_info)


    indices = cv2.dnn.NMSBoxesRotated(boxes, scores, score_threshold, nms_threshold, nms_threshold_decay)

    remaining_boxes = []
    for i in indices:
        remaining_boxes.append(BOXES[i])
        draw_rotated_bounding_box(original_image, BOXES[i])

    cv2.imshow("image", original_image)
    print("Press 0 to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return remaining_boxes 


if __name__ == '__main__':
    # _model = YOLO('yolov8x-obb.pt', task="obb") 
    # _model.export(format="onnx", imgsz=640, simplify=True, half=True, opset=12)

    # model = torch.jit.load('yolov8x-obb.torchscript')
    # model = YOLO("yolov8x-obb.onnx", task="obb") 
    # _model = YOLO("yolov8x-obb.onnx", task="obb") 
    session = ort.InferenceSession('yolov8x-obb.onnx', providers=['CPUExecutionProvider'])

    img = cv2.imread('demos/assets/airplane_aerial_view.png')
    img = cv2.resize(img, (640, 640))
    # img = np.ascontiguousarray(np.expand_dims(cv2.resize(img, (640, 640)).astype('float32'), axis=0).transpose(0, 3, 1, 2))

    [height, width, _] = img.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = img 
    scale = length / 640

    # img_tensor = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    blob = np.ascontiguousarray(np.expand_dims(image.astype('float32') / 255.0, axis=0).transpose(0, 3, 1, 2))

    # results = model(img_tensor)
    results = session.run(None, {'images': blob})
    # results = _model.predict(source=img, save=False)

    # pprint(results[0].shape)

    # detections = detect("yolov8x-obb.onnx", "demos/assets/airplane_aerial_view.png")
    detections = detect_rotated(results[0], scale, img)

