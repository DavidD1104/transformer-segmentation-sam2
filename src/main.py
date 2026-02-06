# src/main.py
import cv2
import os
import numpy as np
from src.interactive import mouse_callback, points, labels
from src.inference_with_points import run_inference_with_points
from src.visualization import save_and_show_masks

#IMAGE_DIR = "data/NEU-DET/train/images/scratches"
#IMAGE_DIR = "data/val2017"
#OUTPUT_DIR = "outputs/scratches"
#OUTPUT_DIR = "outputs/coco"
IMAGE_PATH = "data/example1.jpg"
OUTPUT_PATH = "outputs/results/interactive.png"

def main():
    image = cv2.imread(IMAGE_PATH)
    clone = image.copy()

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback)

    while True:
        display = clone.copy()


        for (x, y), lbl in zip(points, labels):
            color = (0, 255, 0) if lbl == 1 else (0, 0, 255)
            cv2.circle(display, (x, y), 5, color, -1)


        cv2.imshow("Image", display)
        key = cv2.waitKey(1) & 0xFF


        if key == ord("s"):
            cv2.destroyAllWindows()
            image_rgb, masks, scores = run_inference_with_points(
            IMAGE_PATH, points, labels
            )
            save_and_show_masks(image_rgb, masks, OUTPUT_PATH, show=True)
            break

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
