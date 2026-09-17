"""
Week 3 - Day 5 Weekly Project
Evaluation Module: Loads saved model checkpoint and evaluates on independent test data.
Computes test metrics, per-class classification report, confusion matrix heatmap,
and saves sample predictions.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from model import get_model
from dataset import build_dataloaders, CLASSES


def evaluate_pipeline(model_path=None, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))
    if model_path is None:
        model_path = os.path.join(output_dir, "best_vision_model.pth")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("  VisionFlow: Test Set Evaluation & Metrics       ")
    print("==================================================")
    print(f"Loading checkpoint: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get("classes", CLASSES)

    model = get_model(in_channels=1, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Load fresh test split
    _, _, test_loader, _ = build_dataloaders(samples_per_class=300, img_size=48, batch_size=32, seed=42)

    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    all_preds = []
    all_targets = []
    all_images = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x_dev, batch_y_dev = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x_dev)
            loss = criterion(logits, batch_y_dev)

            test_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(logits, 1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(batch_y.numpy())
            all_images.append(batch_x.numpy())

    all_images = np.concatenate(all_images, axis=0)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    total_samples = len(all_targets)
    test_loss /= total_samples
    test_acc = (all_preds == all_targets).mean()

    print(f"\nTest Samples : {total_samples}")
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    # Classification Report
    report_dict = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
    report_text = classification_report(all_targets, all_preds, target_names=classes)
    print("--- Detailed Classification Report ---")
    print(report_text)

    # Save Evaluation Report Text
    report_file = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_file, "w") as f:
        f.write("VisionFlow Evaluation Report\n")
        f.write("=" * 35 + "\n")
        f.write(f"Checkpoint   : {model_path}\n")
        f.write(f"Test Loss    : {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc * 100:.2f}%\n\n")
        f.write(report_text)
    print(f"Saved text report to: {report_file}")

    # Save Metrics JSON
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({
            "test_loss": test_loss,
            "test_accuracy": float(test_acc),
            "report": report_dict
        }, f, indent=4)
    print(f"Saved metrics JSON to: {metrics_file}")

    # Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title="VisionFlow: Confusion Matrix",
           ylabel="True Label",
           xlabel="Predicted Label")

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontweight="bold")

    fig.tight_layout()
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix heatmap to: {cm_path}")

    # Visualizing Test Predictions Grid
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    sample_indices = np.random.choice(len(all_targets), 15, replace=False)

    for i, idx in enumerate(sample_indices):
        ax = axes[i // 5, i % 5]
        img = all_images[idx, 0]
        true_lbl = classes[all_targets[idx]]
        pred_lbl = classes[all_preds[idx]]
        color = "#2ca02c" if true_lbl == pred_lbl else "#d62728"
        ax.imshow(img, cmap="gray")
        ax.set_title(f"Pred: {pred_lbl}\nTrue: {true_lbl}", color=color, fontweight="bold", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    grid_path = os.path.join(output_dir, "test_predictions.png")
    plt.savefig(grid_path, dpi=200)
    plt.close()
    print(f"Saved test predictions grid to: {grid_path}")


if __name__ == "__main__":
    evaluate_pipeline()
