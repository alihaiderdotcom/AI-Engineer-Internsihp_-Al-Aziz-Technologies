"""
Week 3 - Day 3: Neural Network Training Experiments
Demonstrates:
  1. Full training and validation workflow in PyTorch.
  2. Overfitting diagnosis and mitigation using Dropout & L2 Regularization (Weight Decay).
  3. Optimizer comparison: Adam vs SGD with Momentum.
  4. Model checkpointing (saving best model state based on validation loss).
  5. Training curve visualization (loss and accuracy) saved to disk.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 1. Reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 2. Dataset Generation
def prepare_data(n_samples=1200, n_features=20, n_informative=10, test_size=0.25, batch_size=32):
    """
    Generate synthetic classification data with some noise to demonstrate
    generalization vs overfitting.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=4,
        n_clusters_per_class=2,
        flip_y=0.08,  # Add noise to induce potential overfitting
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, n_features


# 3. Model Definition with Optional Dropout
class ConfigurableMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_classes=2, dropout_rate=0.0):
        super(ConfigurableMLP, self).__init__()
        self.use_dropout = dropout_rate > 0.0

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        if self.use_dropout:
            x = self.drop1(x)
        x = self.relu2(self.fc2(x))
        if self.use_dropout:
            x = self.drop2(x)
        x = self.fc3(x)
        return x


# 4. Training Loop with Metrics and Checkpoint Saving
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=50,
                checkpoint_path=None, device="cpu"):
    model.to(device)
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        # Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)

        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                running_val_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)

        epoch_val_loss = running_val_loss / total_val
        epoch_val_acc = correct_val / total_val

        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Save Best Checkpoint
        if checkpoint_path and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": epoch_val_acc
            }, checkpoint_path)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] "
                  f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% "
                  f"|| Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}%")

    if checkpoint_path:
        print(f"-> Best checkpoint saved at Epoch {best_epoch} with Val Loss: {best_val_loss:.4f}")

    return history


# 5. Main Experiment Runner
def run_experiments():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Week 3 Day 3: Neural Network Training Experiments ===")
    print(f"Device: {device}\n")

    train_loader, val_loader, input_dim = prepare_data()
    criterion = nn.CrossEntropyLoss()
    epochs = 40
    results = {}

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # -------------------------------------------------------------
    # Experiment 1: Baseline (High Overfitting - No Dropout, No Weight Decay)
    # -------------------------------------------------------------
    print(">>> Experiment 1: Baseline Model (No Regularization, Adam)")
    model_baseline = ConfigurableMLP(input_dim=input_dim, dropout_rate=0.0)
    optimizer_baseline = optim.Adam(model_baseline.parameters(), lr=0.005, weight_decay=0.0)
    hist_baseline = train_model(
        model_baseline, train_loader, val_loader, criterion, optimizer_baseline,
        epochs=epochs, checkpoint_path=os.path.join(script_dir, "baseline_checkpoint.pth"),
        device=device
    )
    results["baseline"] = {
        "final_train_loss": hist_baseline["train_loss"][-1],
        "final_train_acc": hist_baseline["train_acc"][-1],
        "final_val_loss": hist_baseline["val_loss"][-1],
        "final_val_acc": hist_baseline["val_acc"][-1],
        "best_val_loss": min(hist_baseline["val_loss"]),
        "best_val_acc": max(hist_baseline["val_acc"])
    }

    # -------------------------------------------------------------
    # Experiment 2: Regularized (Dropout = 0.35 + L2 Weight Decay = 1e-3)
    # -------------------------------------------------------------
    print("\n>>> Experiment 2: Regularized Model (Dropout=0.35, Weight Decay=1e-3, Adam)")
    set_seed(42)
    model_reg = ConfigurableMLP(input_dim=input_dim, dropout_rate=0.35)
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=0.005, weight_decay=1e-3)
    hist_reg = train_model(
        model_reg, train_loader, val_loader, criterion, optimizer_reg,
        epochs=epochs, checkpoint_path=os.path.join(script_dir, "best_checkpoint.pth"),
        device=device
    )
    results["regularized"] = {
        "final_train_loss": hist_reg["train_loss"][-1],
        "final_train_acc": hist_reg["train_acc"][-1],
        "final_val_loss": hist_reg["val_loss"][-1],
        "final_val_acc": hist_reg["val_acc"][-1],
        "best_val_loss": min(hist_reg["val_loss"]),
        "best_val_acc": max(hist_reg["val_acc"])
    }

    # -------------------------------------------------------------
    # Experiment 3: Optimizer Comparison (SGD with Momentum vs Adam)
    # -------------------------------------------------------------
    print("\n>>> Experiment 3: SGD with Momentum (lr=0.02, momentum=0.9)")
    set_seed(42)
    model_sgd = ConfigurableMLP(input_dim=input_dim, dropout_rate=0.35)
    optimizer_sgd = optim.SGD(model_sgd.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-3)
    hist_sgd = train_model(
        model_sgd, train_loader, val_loader, criterion, optimizer_sgd,
        epochs=epochs, checkpoint_path=os.path.join(script_dir, "sgd_checkpoint.pth"),
        device=device
    )
    results["sgd_momentum"] = {
        "final_train_loss": hist_sgd["train_loss"][-1],
        "final_train_acc": hist_sgd["train_acc"][-1],
        "final_val_loss": hist_sgd["val_loss"][-1],
        "final_val_acc": hist_sgd["val_acc"][-1],
        "best_val_loss": min(hist_sgd["val_loss"]),
        "best_val_acc": max(hist_sgd["val_acc"])
    }

    # -------------------------------------------------------------
    # 6. Plotting Training Curves
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Baseline vs Regularized Loss
    axes[0, 0].plot(hist_baseline["train_loss"], label="Train Loss (Baseline)", color="#d9534f", linestyle="--")
    axes[0, 0].plot(hist_baseline["val_loss"], label="Val Loss (Baseline)", color="#d9534f")
    axes[0, 0].plot(hist_reg["train_loss"], label="Train Loss (Regularized)", color="#2b5797", linestyle="--")
    axes[0, 0].plot(hist_reg["val_loss"], label="Val Loss (Regularized)", color="#2b5797")
    axes[0, 0].set_title("Loss: Baseline vs Regularized (Mitigating Overfitting)", fontsize=11, fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Subplot 2: Baseline vs Regularized Accuracy
    axes[0, 1].plot(hist_baseline["train_acc"], label="Train Acc (Baseline)", color="#d9534f", linestyle="--")
    axes[0, 1].plot(hist_baseline["val_acc"], label="Val Acc (Baseline)", color="#d9534f")
    axes[0, 1].plot(hist_reg["train_acc"], label="Train Acc (Regularized)", color="#2b5797", linestyle="--")
    axes[0, 1].plot(hist_reg["val_acc"], label="Val Acc (Regularized)", color="#2b5797")
    axes[0, 1].set_title("Accuracy: Baseline vs Regularized", fontsize=11, fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Subplot 3: Optimizer Comparison Loss (Adam vs SGD)
    axes[1, 0].plot(hist_reg["train_loss"], label="Train Loss (Adam)", color="#2b5797")
    axes[1, 0].plot(hist_sgd["train_loss"], label="Train Loss (SGD+Mom)", color="#5cb85c")
    axes[1, 0].plot(hist_reg["val_loss"], label="Val Loss (Adam)", color="#1b3767", linestyle=":")
    axes[1, 0].plot(hist_sgd["val_loss"], label="Val Loss (SGD+Mom)", color="#3e8e41", linestyle=":")
    axes[1, 0].set_title("Loss: Adam vs SGD with Momentum", fontsize=11, fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Subplot 4: Optimizer Comparison Accuracy (Adam vs SGD)
    axes[1, 1].plot(hist_reg["val_acc"], label="Val Acc (Adam)", color="#2b5797")
    axes[1, 1].plot(hist_sgd["val_acc"], label="Val Acc (SGD+Mom)", color="#5cb85c")
    axes[1, 1].set_title("Validation Accuracy: Adam vs SGD with Momentum", fontsize=11, fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Validation Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(script_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"\nVisualization saved to {plot_path}")

    # -------------------------------------------------------------
    # 7. Save Summary Metrics JSON
    # -------------------------------------------------------------
    metrics_path = os.path.join(script_dir, "training_results.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Summary metrics saved to {metrics_path}")

    print("\n=== Experiment Summary ===")
    print(f"Baseline (No Reg)       -> Best Val Loss: {results['baseline']['best_val_loss']:.4f} | Best Val Acc: {results['baseline']['best_val_acc']*100:.2f}%")
    print(f"Regularized (Dropout+L2)-> Best Val Loss: {results['regularized']['best_val_loss']:.4f} | Best Val Acc: {results['regularized']['best_val_acc']*100:.2f}%")
    print(f"SGD with Momentum       -> Best Val Loss: {results['sgd_momentum']['best_val_loss']:.4f} | Best Val Acc: {results['sgd_momentum']['best_val_acc']*100:.2f}%")


if __name__ == "__main__":
    run_experiments()
