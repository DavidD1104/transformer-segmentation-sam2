import cv2
import numpy as np

points = []
labels = []

def mouse_callback(event, x, y, flags, param):
    global points, labels

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        labels.append(1)
        print(f"Foreground point: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN:
        points.append([x, y])
        labels.append(0)
        print(f"Background point: ({x}, {y})")
