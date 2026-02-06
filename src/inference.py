# src/inference.py
import cv2
from src.sam_model import load_sam2

def run_inference(image_path):
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    predictor = load_sam2()
    predictor.set_image(image_rgb)

    masks, scores, _ = predictor.predict(multimask_output=True)

    return image_rgb, masks, scores
