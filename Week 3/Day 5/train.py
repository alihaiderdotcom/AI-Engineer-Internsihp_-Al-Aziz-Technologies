"""
Week 3 - Day 5 Weekly Project
Training Module: Trains VisionClassifierNet with early stopping and checkpointing.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import get_model, count_parameters
from dataset import build_dataloaders


def train_pipeline(epochs=20, batch_size=32, lr=0.001, output_dir=None):
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("==================================================")
    print("  VisionFlow: Deep Learning Training Pipeline     ")
    print("==================================================")
    print(f"Hardware Device : {device}")

    # Build Dataloaders
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        samples_per_class=300, img_size=48, batch_size=batch_size, seed=42
    )
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train Batches   : {len(train_loader)} ({len(train_loader.dataset)} samples)")
    print(f"Val Batches     : {len(val_loader)} ({len(val_loader.dataset)} samples)")
    print(f"Test Batches    : {len(test_loader)} ({len(test_loader.dataset)} samples)")

    # Initialize Model
    model = get_model(in_channels=1, num_classes=len(classes), dropout_rate=0.4).to(device)
    total_p, train_p = count_parameters(model)
    print(f"Total Parameters: {total_p:,} | Trainable: {train_p:,}\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "learning_rates": []
    }

    best_val_loss = float("inf")
    best_model_path = os.path.join(output_dir, "best_vision_model.pth")

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(logits, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_y.size(0)

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, preds = torch.max(logits, 1)
                val_correct += (preds == batch_y).sum().item()
                val_total += batch_y.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        current_lr = optimizer.param_groups[0]['lr']

        scheduler.step(epoch_val_loss)

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["learning_rates"].append(current_lr)

        # Checkpoint Best Model
        saved_str = ""
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": epoch_val_acc,
                "classes": classes
            }, best_model_path)
            saved_str = "[★ Checkpoint Saved]"

        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs}] "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:5.1f}%  ||  "
                  f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:5.1f}%  "
                  f"(lr={current_lr:.5f}) {saved_str}")

    print(f"\nOptimization finished. Best model weights: {best_model_path}")

    # Plot & Save Learning Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(history["val_loss"], label="Validation Loss", color="#ff7f0e", lw=2)
    ax1.set_title("Training and Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot([a * 100 for a in history["train_acc"]], label="Train Accuracy (%)", color="#1f77b4", lw=2)
    ax2.plot([a * 100 for a in history["val_acc"]], label="Validation Accuracy (%)", color="#ff7f0e", lw=2)
    ax2.set_title("Training and Validation Accuracy (%)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Training curves saved to: {curves_path}")

    # Save History JSON
    history_path = os.path.join(output_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to: {history_path}")

    return best_model_path, test_loader, classes


if __name__ == "__main__":
    train_pipeline()
