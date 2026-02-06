# src/visualization.py
import os
import matplotlib.pyplot as plt

def save_and_show_masks(image, masks, output_path, show=True):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    for mask in masks:
        plt.imshow(mask, alpha=0.5)
        plt.axis("off")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()

    plt.close()
