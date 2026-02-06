# Transformers - segmentation with sam2

## Overview
This project showcases a **Transformer-based computer vision pipeline** using **SAM 2 (Segment Anything Model)** to transform raw visual data into structured understanding.

The system performs **zero-shot image segmentation** and **prompt-guided segmentation** using spatial inputs (points), demonstrating how modern foundation models can be adapted to real-world perception tasks without task-specific training.

---

## Pipeline

1. Image ingestion and preprocessing (OpenCV)
2. Loading a pretrained **SAM 2 Transformer-based model**
3. Inference modes:
   - **Automatic segmentation (no prompts)** -> (main_no_points.py)
   - **Prompt-based segmentation using spatial cues** ->  (main.py)
4. Mask generation and scoring
5. Overlay visualization for qualitative analysis

---

## Technologies
- OpenCV
- Transformer-based architectures  
- Segment Anything Model (SAM 2)
- NumPy  
- Matplotlib  

---


## Expected output




<p align="center">
  <img src="assets/.png" width="45%" />
  <img src="assets/.png" width="45%" />
</p>

<p align="center">
   &nbsp;&nbsp; | &nbsp;&nbsp;
  
</p>

## Results


## Dataset


## Future Work


