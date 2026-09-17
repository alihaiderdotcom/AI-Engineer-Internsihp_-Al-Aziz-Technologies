"""
Week 3 - Day 5 Weekly Project
Inference Module: Single and batch image prediction using OpenCV and PyTorch.
Annotates the prediction with confidence bars and saves visualization.
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import cv2

from model import get_model
from dataset import CLASSES, generate_single_sample


def load_classifier(checkpoint_path=None, device="cpu"):
    if checkpoint_path is None:
        checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_vision_model.pth")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    classes = checkpoint.get("classes", CLASSES)
    model = get_model(in_channels=1, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def preprocess_image(image_input, target_size=48):
    """
    Accepts an image path (str) or a numpy array.
    Converts to grayscale, resizes, and normalizes to (1, 1, H, W) float32 tensor.
    """
    if isinstance(image_input, str):
        if not os.path.exists(image_input):
            raise FileNotFoundError(f"Image not found at: {image_input}")
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Failed to decode image from {image_input}")
    else:
        img = image_input

    # Ensure 2D grayscale
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[0] == 1:
        gray = img[0]
    else:
        gray = img

    # Resize to model input dimensions
    resized = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Normalize to [0.0, 1.0]
    if resized.max() > 1.0:
        norm = resized.astype(np.float32) / 255.0
    else:
        norm = resized.astype(np.float32)

    tensor = torch.tensor(norm[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    return tensor, gray


def predict_single_image(model, image_tensor, classes, device="cpu"):
    """Perform forward inference and return top prediction, confidence, and all class probabilities."""
    model.eval()
    if not isinstance(image_tensor, torch.Tensor):
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32)
    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = int(np.argmax(probs))
    top_class = classes[top_idx]
    top_conf = float(probs[top_idx])

    prob_dict = {classes[i]: float(probs[i]) for i in range(len(classes))}
    return top_class, top_conf, prob_dict


def render_prediction_card(original_gray, top_class, top_conf, prob_dict, output_path):
    """Render a clean OpenCV visual card showing the image and prediction confidence bars."""
    card_h, card_w = 260, 480
    card = np.full((card_h, card_w, 3), 245, dtype=np.uint8)  # Light background

    # Insert resized original image (160x160)
    display_img = cv2.resize(original_gray, (160, 160), interpolation=cv2.INTER_NEAREST)
    if display_img.max() <= 1.0:
        display_img = (display_img * 255).astype(np.uint8)
    display_bgr = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)

    card[50:210, 20:180] = display_bgr
    cv2.rectangle(card, (20, 50), (180, 210), (100, 100, 100), 2)

    # Title
    cv2.putText(card, "VisionFlow AI Inference", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

    # Prediction text
    cv2.putText(card, f"Predicted: {top_class}", (200, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 0), 2)
    cv2.putText(card, f"Confidence: {top_conf*100:.1f}%", (200, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)

    # Draw probability bars for each class
    bar_x = 200
    start_y = 105
    for idx, (cls_name, prob) in enumerate(prob_dict.items()):
        y = start_y + idx * 28
        cv2.putText(card, f"{cls_name[:7]:<7}", (bar_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
        # Background bar
        cv2.rectangle(card, (bar_x + 75, y - 10), (bar_x + 245, y + 2), (210, 210, 210), -1)
        # Active bar
        bar_len = int(170 * prob)
        color = (0, 160, 0) if cls_name == top_class else (180, 120, 70)
        cv2.rectangle(card, (bar_x + 75, y - 10), (bar_x + 75 + bar_len, y + 2), color, -1)
        # Percentage text
        cv2.putText(card, f"{prob*100:4.1f}%", (bar_x + 250, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    cv2.imwrite(output_path, card)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="VisionFlow Inference Utility")
    parser.add_argument("--image", type=str, default=None, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--demo", action="store_true", help="Run self-contained demo with synthetic test image")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))

    ckpt_path = args.checkpoint if args.checkpoint else os.path.join(script_dir, "best_vision_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"Model checkpoint not found at: {ckpt_path}. Please run train.py first.")
        return

    model, classes = load_classifier(ckpt_path, device=device)

    if args.demo or args.image is None:
        print("Running demo inference on synthetic test sample (Star)...")
        # Generate sample Star image
        sample_tensor = generate_single_sample("Star", img_size=48)
        raw_gray = (sample_tensor[0, 0] * 255).astype(np.uint8)
        input_tensor = torch.tensor(sample_tensor, dtype=torch.float32)
    else:
        input_tensor, raw_gray = preprocess_image(args.image, target_size=48)

    top_class, top_conf, prob_dict = predict_single_image(model, input_tensor, classes, device=device)

    print("\n--- Inference Result ---")
    print(f"Top Prediction : {top_class} ({top_conf * 100:.2f}% confidence)")
    print("Class Probabilities:")
    for cls_name, prob in prob_dict.items():
        print(f"  {cls_name:<10}: {prob * 100:5.1f}%")

    out_card = os.path.join(script_dir, "prediction_result.png")
    render_prediction_card(raw_gray, top_class, top_conf, prob_dict, out_card)
    print(f"\nVisual prediction result saved to: {out_card}")


if __name__ == "__main__":
    main()
