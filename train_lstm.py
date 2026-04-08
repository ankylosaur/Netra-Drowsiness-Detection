"""
train_lstm.py
=============
Trains a PyTorch LSTM Sequence Classifier using the features extracted by MediaPipe.
Splits the data 70/15/15, creates a learning curve, evaluates the test set,
and saves the trained model weights.

Usage:
    python train_lstm.py --features-dir extracted_features
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


class DrowsinessLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, num_classes=2):
        super(DrowsinessLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Take the output from the last timestep
        out = self.fc(out[:, -1, :])
        return out


def train_model(
    x_train: np.ndarray, y_train: np.ndarray,
    x_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 30, batch_size: int = 32, lr: float = 0.001
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Convert to standard PyTorch format
    train_data = TensorDataset(torch.tensor(x_train), torch.tensor(y_train, dtype=torch.long))
    val_data   = TensorDataset(torch.tensor(x_val),   torch.tensor(y_val, dtype=torch.long))

    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    val_loader   = DataLoader(val_data, shuffle=False, batch_size=batch_size)

    model = DrowsinessLSTM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    os.makedirs("models", exist_ok=True)
    best_model_path = os.path.join("models", "drowsiness_lstm.pt")

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        v_losses = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                v_loss = criterion(outputs, targets)
                v_losses.append(v_loss.item())

        val_loss = np.mean(v_losses)
        val_losses.append(val_loss)

        print(f"Epoch [{epoch:2d}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

    print(f"\nTraining completed. Best model saved to {best_model_path}")
    return train_losses, val_losses, best_model_path


def evaluate_model(model_path: str, x_test: np.ndarray, y_test: np.ndarray):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DrowsinessLSTM().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    inputs = torch.tensor(x_test).to(device)
    targets = torch.tensor(y_test, dtype=torch.long).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

    preds_np = preds.cpu().numpy()
    targets_np = targets.cpu().numpy()

    acc = accuracy_score(targets_np, preds_np)
    prec = precision_score(targets_np, preds_np, zero_division=0)
    rec = recall_score(targets_np, preds_np, zero_division=0)
    f1 = f1_score(targets_np, preds_np, zero_division=0)

    cm = confusion_matrix(targets_np, preds_np)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        spec = 0.0

    print("\n" + "=" * 50)
    print("LSTM Evaluation on Test Set (15%)")
    print("=" * 50)
    print(f"Accuracy    : {acc:.4f}")
    print(f"Precision   : {prec:.4f}")
    print(f"Recall      : {rec:.4f}")
    print(f"Specificity : {spec:.4f}")
    print(f"F1-Score    : {f1:.4f}")
    print("=" * 50)

    # Save Confusion Matrix Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Awake', 'Drowsy'], yticklabels=['Awake', 'Drowsy'])
    plt.title('LSTM Test Set Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('lstm_confusion_matrix.png')
    plt.close()
    print("Saved -> lstm_confusion_matrix.png")


def plot_learning_curve(train_losses, val_losses, output_path="lstm_learning_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('CrossEntropy Loss')
    plt.title('LSTM Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser("Train PyTorch LSTM on extracted temporal frames")
    parser.add_argument("--features-dir", default="extracted_features", help="Directory with .npy arrays")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    feats_path = os.path.join(args.features_dir, "features.npy")
    labs_path = os.path.join(args.features_dir, "labels.npy")

    if not os.path.exists(feats_path) or not os.path.exists(labs_path):
        print(f"Could not find features inside '{args.features_dir}'. Run extract_features.py first.")
        return

    # Load arrays
    X = np.load(feats_path)
    y = np.load(labs_path)

    n_total = len(X)
    print(f"Loaded dataset: {X.shape} (N, Timesteps, 2)")
    print(f"Total samples : {n_total}")

    # Standardize Features (Z-Score scaling across all temporal frames)
    # Mean and Std across dimensions 0 (samples) and 1 (timesteps)
    mean = np.mean(X, axis=(0, 1), keepdims=True)
    std = np.std(X, axis=(0, 1), keepdims=True)
    std[std == 0] = 1.0  # Prevent divide by zero
    X_scaled = (X - mean) / std

    # Save the scaler values for real-time inference
    os.makedirs("models", exist_ok=True)
    np.save(os.path.join("models", "scaler_mean.npy"), mean)
    np.save(os.path.join("models", "scaler_std.npy"), std)

    # Manual split 70 / 15 / 15
    indices = np.arange(n_total)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    n_train = int(0.70 * n_total)
    n_val   = int(0.15 * n_total)
    
    train_idx = indices[:n_train]
    val_idx   = indices[n_train: n_train + n_val]
    test_idx  = indices[n_train + n_val:]

    x_train, y_train = X_scaled[train_idx], y[train_idx]
    x_val,   y_val   = X_scaled[val_idx],   y[val_idx]
    x_test,  y_test  = X_scaled[test_idx],  y[test_idx]

    print(f"Train split : {len(x_train)}")
    print(f"Val split   : {len(x_val)}")
    print(f"Test split  : {len(x_test)}\n")

    train_losses, val_losses, best_model_path = train_model(
        x_train, y_train,
        x_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr
    )

    plot_learning_curve(train_losses, val_losses)
    evaluate_model(best_model_path, x_test, y_test)


if __name__ == "__main__":
    main()
