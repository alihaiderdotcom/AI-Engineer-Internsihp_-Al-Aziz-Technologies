"""
Week 3 - Day 4: Convolutional Neural Network (CNN) Image Classification
Demonstrates:
  1. Synthetic image dataset generation with OpenCV (Shapes: Circle, Square, Triangle, Cross).
  2. PyTorch CNN architecture with Conv2D, BatchNorm2d, MaxPool2d, Dropout, and Linear layers.
  3. Image tensor formatting: (N, Channels, Height, Width) normalized to [0, 1].
  4. Training loop with CrossEntropyLoss and Adam optimizer.
  5. Performance evaluation: Accuracy, Confusion Matrix, and Classification Report.
  6. Visualizing sample predictions and training curves.
"""

import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


CLASSES = ["Circle", "Square", "Triangle", "Cross"]
CLASS_MAP = {name: idx for idx, name in enumerate(CLASSES)}


def generate_shape_dataset(num_samples_per_class=250, img_size=32):
    """
    Generate synthetic images of geometric shapes with noise and varied positions
    to evaluate CNN spatial feature learning.
    """
    np.random.seed(42)
    images = []
    labels = []

    for class_idx, class_name in enumerate(CLASSES):
        for _ in range(num_samples_per_class):
            img = np.zeros((img_size, img_size), dtype=np.uint8)
            cx = np.random.randint(img_size // 4, 3 * img_size // 4)
            cy = np.random.randint(img_size // 4, 3 * img_size // 4)
            size = np.random.randint(img_size // 6, img_size // 3)

            if class_name == "Circle":
                cv2.circle(img, (cx, cy), size, 255, -1)
            elif class_name == "Square":
                cv2.rectangle(img, (cx - size, cy - size), (cx + size, cy + size), 255, -1)
            elif class_name == "Triangle":
                pts = np.array([[cx, cy - size], [cx - size, cy + size], [cx + size, cy + size]], np.int32)
                cv2.fillPoly(img, [pts], 255)
            elif class_name == "Cross":
                thick = max(1, size // 3)
                cv2.rectangle(img, (cx - size, cy - thick), (cx + size, cy + thick), 255, -1)
                cv2.rectangle(img, (cx - thick, cy - size), (cx + thick, cy + size), 255, -1)

            # Add subtle Gaussian noise
            noise = np.random.normal(0, 15, (img_size, img_size)).astype(np.int16)
            noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Normalize to [0.0, 1.0] and add channel dimension (1, H, W)
            norm_img = noisy_img.astype(np.float32) / 255.0
            images.append(norm_img[np.newaxis, :, :])
            labels.append(class_idx)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Shuffle dataset
    indices = np.random.permutation(len(labels))
    images, labels = images[indices], labels[indices]

    split = int(0.8 * len(labels))
    return (images[:split], labels[:split]), (images[split:], labels[split:])


# 2. Convolutional Neural Network Architecture
class SimpleVisionCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleVisionCNN, self).__init__()
        # Conv Block 1: Input (1, 32, 32) -> Output (16, 16, 16)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Conv Block 2: Input (16, 16, 16) -> Output (32, 8, 8)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Fully Connected Head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        logits = self.classifier(x)
        return logits


def train_and_evaluate_cnn():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== PyTorch CNN Image Classifier ===")
    print(f"Device: {device}\n")

    # Prepare Data
    (X_train, y_train), (X_val, y_val) = generate_shape_dataset()
    print(f"Training Samples: {X_train.shape[0]} | Validation Samples: {X_val.shape[0]}")
    print(f"Tensor Shape: {X_train.shape} (N, C, H, W)")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=32, shuffle=False
    )

    model = SimpleVisionCNN(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)

    epochs = 15
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    print("\nStarting CNN Training Loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.size(0)

        t_loss = total_loss / total
        t_acc = correct / total
        train_losses.append(t_loss)
        train_accs.append(t_acc)

        # Validation
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                v_loss += loss.item() * batch_x.size(0)
                _, preds = torch.max(outputs, 1)
                v_correct += (preds == batch_y).sum().item()
                v_total += batch_y.size(0)

        val_loss = v_loss / v_total
        val_acc = v_correct / v_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] "
                  f"Train Loss: {t_loss:.4f}, Train Acc: {t_acc*100:.1f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.1f}%")

    # Evaluation on Validation Set
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report_str = classification_report(all_labels, all_preds, target_names=CLASSES)
    print("\n=== Classification Report ===")
    print(report_str)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save Checkpoint
    model_path = os.path.join(script_dir, "cnn_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    # Plot Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_losses, label="Train Loss", color="#1f77b4")
    ax1.plot(val_losses, label="Val Loss", color="#ff7f0e")
    ax1.set_title("CNN Loss Curves", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label="Train Accuracy", color="#1f77b4")
    ax2.plot(val_accs, label="Val Accuracy", color="#ff7f0e")
    ax2.set_title("CNN Accuracy Curves", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = os.path.join(script_dir, "cnn_training_curves.png")
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Training curves saved to {curves_path}")

    # Visualize Sample Predictions
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    sample_indices = np.random.choice(len(y_val), 10, replace=False)

    for i, idx in enumerate(sample_indices):
        ax = axes[i // 5, i % 5]
        img = X_val[idx, 0]
        true_lbl = CLASSES[y_val[idx]]
        pred_lbl = CLASSES[all_preds[idx]]
        color = "green" if true_lbl == pred_lbl else "red"
        ax.imshow(img, cmap="gray")
        ax.set_title(f"P: {pred_lbl}\nT: {true_lbl}", color=color, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    pred_path = os.path.join(script_dir, "sample_predictions.png")
    plt.savefig(pred_path, dpi=200)
    plt.close()
    print(f"Sample predictions grid saved to {pred_path}")

    # Save summary results
    results_path = os.path.join(script_dir, "cnn_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "final_train_loss": train_losses[-1],
            "final_train_acc": train_accs[-1],
            "final_val_loss": val_losses[-1],
            "final_val_acc": val_accs[-1],
            "classes": CLASSES
        }, f, indent=4)


if __name__ == "__main__":
    train_and_evaluate_cnn()
