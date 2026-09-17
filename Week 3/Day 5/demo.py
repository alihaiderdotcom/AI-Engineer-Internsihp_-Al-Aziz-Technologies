"""
Week 3 - Day 5 Weekly Project
Complete Project Demonstration Runner
Orchestrates training, evaluation, and live inference demonstration.
"""

import os
import sys
from train import train_pipeline
from evaluate import evaluate_pipeline
from predict import load_classifier, generate_single_sample, predict_single_image, render_prediction_card
import numpy as np
import torch


def run_full_project_demo():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("###################################################################")
    print("#  AL AZIZ TECHNOLOGIES - AI ENGINEERING INTERNSHIP               #")
    print("#  WEEK 3 PROJECT: Deep Learning & Computer Vision (VisionFlow)   #")
    print("###################################################################\n")

    # Step 1: Model Training
    print(">>> STAGE 1: Training Deep Convolutional Neural Network...")
    best_model_path, test_loader, classes = train_pipeline(
        epochs=15, batch_size=32, lr=0.001, output_dir=script_dir
    )

    # Step 2: Evaluation on Test Data
    print("\n>>> STAGE 2: Independent Test Evaluation & Metrics...")
    evaluate_pipeline(model_path=best_model_path, output_dir=script_dir)

    # Step 3: Live Inference Demonstration
    print("\n>>> STAGE 3: Live Real-time Inference Demonstration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_classifier(best_model_path, device=device)

    demo_shapes = ["Circle", "Square", "Triangle", "Star", "Hexagon"]
    print("Testing single-sample prediction across all classes:")
    for shape_name in demo_shapes:
        sample_tensor = generate_single_sample(shape_name, img_size=48)
        raw_gray = (sample_tensor[0, 0] * 255).astype(np.uint8)
        input_tensor = torch.tensor(sample_tensor, dtype=torch.float32)

        top_class, top_conf, prob_dict = predict_single_image(model, input_tensor, classes, device=device)
        status = "✓ Correct" if top_class == shape_name else "✗ Mismatch"
        print(f"  Target: {shape_name:<9} | Predicted: {top_class:<9} | Confidence: {top_conf*100:5.1f}% | [{status}]")

    # Save final visual card for the Hexagon test
    test_tensor = generate_single_sample("Hexagon", img_size=48)
    raw_gray = (test_tensor[0, 0] * 255).astype(np.uint8)
    top_class, top_conf, prob_dict = predict_single_image(model, torch.tensor(test_tensor), classes, device=device)
    out_card = os.path.join(script_dir, "prediction_result.png")
    render_prediction_card(raw_gray, top_class, top_conf, prob_dict, out_card)

    print(f"\nDemo card generated at: {out_card}")
    print("\n[SUCCESS] Week 3 Project pipeline executed successfully!")


if __name__ == "__main__":
    run_full_project_demo()
