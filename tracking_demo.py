from ultralytics.models import YOLO

model = YOLO('yolov8n-obb.pt') 

if __name__ == '__main__':
    model = YOLO('yolov8n-obb.pt')
    results = model.track(source="box_slide_pool.mp4", save=True, tracker='bytetrack.yaml')

