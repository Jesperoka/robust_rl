from ultralytics.models import YOLO

VIDEO = "demos/assets/airplane_hack.mp4"

if __name__ == '__main__':
    model = YOLO('yolov8x-obb.pt') 
    results = model.track(source=VIDEO, save=True, tracker='botsort.yaml', )

