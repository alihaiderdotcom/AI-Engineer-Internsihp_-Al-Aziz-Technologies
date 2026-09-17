"""
Week 3 - Day 4: Computer Vision Fundamentals with OpenCV
Demonstrates:
  1. Image loading, representation (dimensions, channels, dtypes).
  2. Color space conversions (BGR to RGB, BGR to Grayscale).
  3. Image transformations: resizing and normalization.
  4. Spatial filtering: Gaussian blur and Canny edge detection.
  5. Image augmentation techniques: flipping, rotation, and brightness shifts.
  6. Saving transformed output figures.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_sample_scene():
    """Create a synthetic rich scene with shapes and gradients to demonstrate CV operations."""
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Background gradient
    for i in range(300):
        img[i, :, 0] = int(255 * (i / 300))  # Blue gradient
        img[:, i, 1] = int(200 * (1 - i / 300))  # Green gradient

    # Draw shapes
    cv2.circle(img, (150, 150), 60, (0, 0, 255), -1)  # Red circle
    cv2.rectangle(img, (40, 40), (100, 100), (255, 255, 0), -1)  # Cyan square
    pts = np.array([[200, 260], [280, 260], [240, 180]], np.int32)
    cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 255), thickness=4)  # Yellow triangle
    cv2.putText(img, "OpenCV & AI", (50, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img


def run_cv_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sample_bgr = generate_sample_scene()

    print("=== OpenCV Computer Vision Fundamentals ===")
    print(f"Sample Image Shape: {sample_bgr.shape} (Height, Width, Channels)")
    print(f"Data Type: {sample_bgr.dtype}")
    print(f"Pixel Value Range: Min={sample_bgr.min()}, Max={sample_bgr.max()}")

    # 1. Color Space Conversions
    sample_rgb = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2RGB)
    sample_gray = cv2.cvtColor(sample_bgr, cv2.COLOR_BGR2GRAY)

    # 2. Resizing
    resized_small = cv2.resize(sample_rgb, (150, 150), interpolation=cv2.INTER_AREA)
    resized_large = cv2.resize(sample_rgb, (450, 450), interpolation=cv2.INTER_CUBIC)

    # 3. Blurring & Smoothing
    blurred_gaussian = cv2.GaussianBlur(sample_rgb, (11, 11), sigmaX=3.0)

    # 4. Edge Detection (Canny)
    edges = cv2.Canny(sample_gray, threshold1=50, threshold2=150)

    # 5. Normalization
    # Min-Max Normalization to [0, 1]
    normalized_float = sample_rgb.astype(np.float32) / 255.0
    print(f"Normalized Range: Min={normalized_float.min():.2f}, Max={normalized_float.max():.2f}")

    # 6. Data Augmentations
    # Horizontal Flip
    flipped_h = cv2.flip(sample_rgb, 1)
    # Rotation (45 degrees around center)
    center = (sample_rgb.shape[1] // 2, sample_rgb.shape[0] // 2)
    rot_matrix = cv2.getRotationMatrix2D(center, angle=45, scale=1.0)
    rotated = cv2.warpAffine(sample_rgb, rot_matrix, (sample_rgb.shape[1], sample_rgb.shape[0]))
    # Brightness Adjustment
    brightened = cv2.convertScaleAbs(sample_rgb, alpha=1.2, beta=40)

    # 7. Visualization Grid
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    axes[0, 0].imshow(sample_rgb)
    axes[0, 0].set_title("1. Original RGB Image", fontsize=10, fontweight="bold")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sample_gray, cmap="gray")
    axes[0, 1].set_title("2. Grayscale Conversion", fontsize=10, fontweight="bold")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(blurred_gaussian)
    axes[0, 2].set_title("3. Gaussian Blur (11x11)", fontsize=10, fontweight="bold")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(edges, cmap="gray")
    axes[0, 3].set_title("4. Canny Edge Detection", fontsize=10, fontweight="bold")
    axes[0, 3].axis("off")

    axes[0, 1].imshow(sample_gray, cmap="gray")

    axes[1, 0].imshow(resized_small)
    axes[1, 0].set_title("5. Downscaled (150x150)", fontsize=10, fontweight="bold")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(flipped_h)
    axes[1, 1].set_title("6. Augmentation: H-Flip", fontsize=10, fontweight="bold")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(rotated)
    axes[1, 2].set_title("7. Augmentation: 45° Rotation", fontsize=10, fontweight="bold")
    axes[1, 2].axis("off")

    axes[1, 3].imshow(brightened)
    axes[1, 3].set_title("8. Augmentation: Brightness Shift", fontsize=10, fontweight="bold")
    axes[1, 3].axis("off")

    plt.tight_layout()
    output_plot = os.path.join(script_dir, "cv_transformations.png")
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"Visual transformation matrix saved to: {output_plot}")


if __name__ == "__main__":
    run_cv_pipeline()
