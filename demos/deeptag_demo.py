import torch
import cv2 

MODEL_PATH = "demos/assets/scripted_deeptag_aruco.pt"
IMG_PATH = "demos/assets/apriltag_cube.png"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    img = cv2.imread(IMG_PATH)
    input = torch.as_tensor(img, dtype=torch.float32, device=DEVICE).unsqueeze(0).permute(0, 3, 2, 1)

    print("Running inference...")
    results = model(input)
    print("done!")
    print(results[1].shape) 
