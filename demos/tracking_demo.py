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

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def detect(onnx_model, input_image):
    model: cv2.dnn.Net = cv2.dnn.readNetFromONNX(onnx_model)
    original_image: np.ndarray = cv2.imread(input_image)
    [height, width, _] = original_image.shape
    length = max((height, width))
    image = np.zeros((length, length, 3), np.uint8)
    image[0:height, 0:width] = original_image
    scale = length / 640
    blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
    model.setInput(blob)

    outputs = model.forward()
    print(outputs.shape)

    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
    detections = []

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    cv2.imshow("image", original_image)
    print("Press 0 to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections

def detect_2(model_output, scale, original_image):
    outputs = np.array([cv2.transpose(model_output[0])])
    rows = outputs.shape[1]
    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
        if maxScore >= 0.25:
            box = [
                outputs[0][i][0] - (0.5 * outputs[0][i][2]),
                outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                outputs[0][i][2],
                outputs[0][i][3],
            ]
            boxes.append(box)
            scores.append(maxScore)
            class_ids.append(maxClassIndex)

    result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.7, 0.1, 0.1)
    detections = []

    for i in range(len(result_boxes)):
        index = result_boxes[i]
        box = boxes[index]
        detection = {
            "class_id": class_ids[index],
            "class_name": CLASSES[class_ids[index]],
            "confidence": scores[index],
            "box": box,
            "scale": scale,
        }
        detections.append(detection)
        draw_bounding_box(
            original_image,
            class_ids[index],
            scores[index],
            round(box[0] * scale),
            round(box[1] * scale),
            round((box[0] + box[2]) * scale),
            round((box[1] + box[3]) * scale),
        )

    cv2.imshow("image", original_image)
    print("Press 0 to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return detections

if __name__ == '__main__':
    _model = YOLO('yolov8x-obb.pt', task="obb") 
    torch.jit.save(torch.jit.script(_model), 'yolov8x-obb.torchscript_2')
    input()
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
    detections = detect_2(results[0], scale, img)

