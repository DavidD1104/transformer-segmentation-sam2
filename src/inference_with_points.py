import cv2
import numpy as np
from src.sam_model import load_sam2

def run_inference_with_points(image_path, points, labels):
    """
    points: lista de puntos [[x, y], ...]
    labels: lista de etiquetas [1 (foreground) | 0 (background)]
    """
    image_bgr = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    predictor = load_sam2()
    predictor.set_image(image_rgb)

    point_coords = np.array(points)
    point_labels = np.array(labels)

    masks, scores, _ = predictor.predict(
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True,
    )

    return image_rgb, masks, scores
