# src/sam_model.py
import torch
from pathlib import Path
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT = BASE_DIR / "checkpoints" / "sam2.1_hiera_small.pt"
CONFIG = BASE_DIR / "configs" / "sam2.1_hiera_s.yaml"


def load_sam2():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(str(CONFIG), str(CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor
