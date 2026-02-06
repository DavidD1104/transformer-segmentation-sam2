# src/main.py
import os
from src.inference import run_inference
from src.visualization import save_and_show_masks

#IMAGE_DIR = "data/NEU-DET/train/images/scratches"
IMAGE_DIR = "data/val2017"
#OUTPUT_DIR = "outputs/scratches"
OUTPUT_DIR = "outputs/coco"

def main():
    images = [f for f in os.listdir(IMAGE_DIR) if f.endswith((".jpg", ".png"))]

    if not images:
        raise RuntimeError("No images found in data/images")

    for img_name in images:
        img_path = os.path.join(IMAGE_DIR, img_name)
        image, masks, scores = run_inference(img_path)
        output_path = os.path.join(OUTPUT_DIR, img_name)
        save_and_show_masks(image, masks, output_path, show=True)

if __name__ == "__main__":
    main()
