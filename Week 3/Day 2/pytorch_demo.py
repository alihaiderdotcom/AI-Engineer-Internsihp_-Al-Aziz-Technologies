import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

def generate_synthetic_data():
    """Generate a simple synthetic binary classification dataset."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 100 samples, 2 features
    X = np.random.randn(100, 2).astype(np.float32)
    # Label is 1 if sum of features > 0, else 0
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32).reshape(-1, 1)
    
    # Convert to PyTorch Tensors
    X_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)
    
    # Split into Train (80) and Validation (20)
    train_dataset = TensorDataset(X_tensor[:80], y_tensor[:80])
    val_dataset = TensorDataset(X_tensor[80:], y_tensor[80:])
    
    return train_dataset, val_dataset


# Define a simple Neural Network using nn.Module
class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        # Input layer (2 features) -> Hidden layer (4 neurons)
        self.hidden = nn.Linear(2, 4)
        self.relu = nn.ReLU()
        # Hidden layer (4 neurons) -> Output layer (1 neuron)
        self.output = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x


def main():
    # 1. Device Configuration (GPU vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    # 2. Prepare Datasets and DataLoaders
    train_dataset, val_dataset = generate_synthetic_data()
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # 3. Initialize Model, Loss Function, and Optimizer
    model = SimpleClassifier().to(device)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # 4. Training Loop
    epochs = 20
    print("--- Starting Training Loop ---")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)

        # 5. Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                # Calculate accuracy
                predicted_classes = (predictions >= 0.5).float()
                correct += (predicted_classes == batch_y).sum().item()
                total += batch_y.size(0)
                
        val_loss /= len(val_loader.dataset)
        accuracy = correct / total

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {accuracy:.4f}")

    # 6. Saving the Model
    import os
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "simple_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # 7. Loading the Model
    loaded_model = SimpleClassifier().to(device)
    loaded_model.load_state_dict(torch.load(model_path))
    loaded_model.eval()
    print("Model loaded successfully and set to evaluation mode.")

    # Test prediction with loaded model
    test_input = torch.tensor([[1.0, 1.0]], device=device)
    with torch.no_grad():
        test_pred = loaded_model(test_input)
    print(f"Prediction for input [1.0, 1.0]: {test_pred.item():.4f}")


if __name__ == "__main__":
    main()
