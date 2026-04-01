"""
CNN on sklearn digits (8x8) — compares Adam, SGD+Nesterov, and AdamW.
Uses sklearn.datasets.load_digits; small conv net; validation leaderboard by accuracy.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUTPUT = os.path.join(_TASK_DIR, "output")


def get_task_metadata():
    return {
        "task_name": "cnn_sklearn_digits_optimizer_compare",
        "task_type": "classification",
        "num_classes": 10,
        "input_shape": [1, 8, 8],
        "description": (
            "Small CNN on sklearn load_digits; compares Adam, SGD+Nesterov, "
            "and AdamW with the same epoch budget."
        ),
    }


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DigitsCNN(nn.Module):
    """Lightweight CNN for 8x8 single-channel digits."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def _numpy_digits_to_tensors(stratify_seed=42):
    bunch = load_digits()
    x = bunch.data.astype(np.float32) / 16.0
    y = bunch.target.astype(np.int64)
    x = x.reshape(-1, 1, 8, 8)

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.30,
        stratify=y,
        random_state=stratify_seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=stratify_seed,
    )
    return (
        torch.from_numpy(x_train),
        torch.from_numpy(y_train),
        torch.from_numpy(x_val),
        torch.from_numpy(y_val),
        torch.from_numpy(x_test),
        torch.from_numpy(y_test),
    )


def make_dataloaders(batch_size=64, num_workers=0):
    x_tr, y_tr, x_va, y_va, x_te, y_te = _numpy_digits_to_tensors()

    train_ds = TensorDataset(x_tr, y_tr)
    val_ds = TensorDataset(x_va, y_va)
    test_ds = TensorDataset(x_te, y_te)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_eval_loader, test_loader


def build_model(num_classes=10):
    model = DigitsCNN(num_classes=num_classes)
    return model.to(device)


def _make_optimizer(model, optimizer_name, lr, weight_decay):
    name = optimizer_name.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd_nesterov":
        return optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer_name: {optimizer_name}")


def train(
    model,
    train_loader,
    val_loader,
    epochs=40,
    optimizer_name="adam",
    lr=None,
    weight_decay=1e-4,
):
    """Train with one optimizer; return per-epoch history."""
    criterion = nn.CrossEntropyLoss()
    if lr is None:
        lr = 0.08 if optimizer_name.lower() == "sgd_nesterov" else 1e-3
    optimizer = _make_optimizer(model, optimizer_name, lr, weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            running += loss.item()

        avg_train = running / max(len(train_loader), 1)
        train_losses.append(avg_train)

        val_m = evaluate(model, val_loader, return_predictions=False)
        val_losses.append(val_m["loss"])
        val_accuracies.append(val_m["accuracy"])
        scheduler.step(val_m["loss"])

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "optimizer_name": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
    }


def evaluate(model, data_loader, return_predictions=True):
    """Return loss, accuracy, MSE/R2 (one-hot vs softmax), and optional preds."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            loss = criterion(logits, target)
            total_loss += loss.item()
            all_logits.append(logits.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    probs = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    pred = np.argmax(logits_np, axis=1)

    n_classes = logits_np.shape[1]
    y_onehot = np.eye(n_classes, dtype=np.float64)[targets_np]
    mse = float(mean_squared_error(y_onehot, probs))
    r2 = float(r2_score(y_onehot, probs, multioutput="uniform_average"))

    avg_loss = total_loss / max(len(data_loader), 1)
    accuracy = float(accuracy_score(targets_np, pred))

    out = {
        "loss": avg_loss,
        "accuracy": accuracy,
        "mse": mse,
        "r2": r2,
    }
    if return_predictions:
        out["predictions"] = pred
        out["targets"] = targets_np
        out["probabilities"] = probs
    return out


def predict(model, data_loader):
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for data, _ in data_loader:
            data = data.to(device)
            logits = model(data)
            pr = torch.softmax(logits, dim=1)
            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(pr.cpu().numpy())
    return np.array(all_preds), np.array(all_probs)


def save_artifacts(model, metrics, output_dir=None):
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if "val_predictions" in metrics and "val_targets" in metrics:
        cm = confusion_matrix(metrics["val_targets"], metrics["val_predictions"])
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title("Confusion matrix — sklearn digits (best optimizer)")
        plt.colorbar()
        ticks = np.arange(10)
        plt.xticks(ticks, ticks)
        plt.yticks(ticks, ticks)
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
        plt.close()

    if "optimizer_val_accuracies" in metrics:
        plt.figure(figsize=(9, 5))
        for name, series in metrics["optimizer_val_accuracies"].items():
            plt.plot(range(1, len(series) + 1), series, label=name, linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Validation accuracy")
        plt.title("Optimizer comparison — validation accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "optimizer_comparison_val_acc.png"), dpi=150)
        plt.close()

    print(f"Artifacts saved to {output_dir}")


def run_optimizer_comparison(train_loader, val_loader, epochs, num_classes):
    specs = [
        ("adam", None, 1e-4),
        ("sgd_nesterov", None, 1e-4),
        ("adamw", None, 1e-2),
    ]
    results = []
    histories = {}
    best_model = None
    best_name = None
    best_val_acc = -1.0

    for opt_name, lr_override, wd in specs:
        set_seed(42)
        m = build_model(num_classes=num_classes)
        hist = train(
            m,
            train_loader,
            val_loader,
            epochs=epochs,
            optimizer_name=opt_name,
            lr=lr_override,
            weight_decay=wd,
        )
        val_snapshot = evaluate(m, val_loader, return_predictions=False)
        best_epoch_acc = max(hist["val_accuracies"])
        results.append(
            {
                "optimizer": opt_name,
                "val_accuracy": val_snapshot["accuracy"],
                "best_epoch_val_accuracy": best_epoch_acc,
                "val_loss": val_snapshot["loss"],
                "val_mse": val_snapshot["mse"],
                "val_r2": val_snapshot["r2"],
                "lr": hist["lr"],
                "weight_decay": hist["weight_decay"],
            }
        )
        histories[opt_name] = hist["val_accuracies"]
        if val_snapshot["accuracy"] > best_val_acc:
            best_val_acc = val_snapshot["accuracy"]
            best_name = opt_name
            best_model = m

    return best_model, results, histories, best_name


def main():
    print("=" * 60)
    print("CNN on sklearn digits — optimizer comparison")
    print("=" * 60)

    meta = get_task_metadata()
    epochs = 45

    train_loader, val_loader, train_eval_loader, test_loader = make_dataloaders(
        batch_size=64
    )
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    best_model, leaderboard, opt_histories, best_opt = run_optimizer_comparison(
        train_loader, val_loader, epochs, meta["num_classes"]
    )

    print("\nOptimizer leaderboard (by final validation accuracy):")
    for row in sorted(leaderboard, key=lambda r: -r["val_accuracy"]):
        print(
            f"  {row['optimizer']}: val_acc={row['val_accuracy']:.4f}, "
            f"best_epoch_val_acc={row['best_epoch_val_accuracy']:.4f}, "
            f"val_mse={row['val_mse']:.6f}, val_r2={row['val_r2']:.4f}"
        )
    print(f"\nSelected best optimizer: {best_opt}")

    train_metrics = evaluate(best_model, train_eval_loader)
    val_metrics = evaluate(best_model, val_loader, return_predictions=True)
    test_metrics = evaluate(best_model, test_loader)

    print("\n--- Best model metrics ---")
    print(
        f"Train — loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.4f}, "
        f"MSE: {train_metrics['mse']:.6f}, R2: {train_metrics['r2']:.4f}"
    )
    print(
        f"Val   — loss: {val_metrics['loss']:.4f}, acc: {val_metrics['accuracy']:.4f}, "
        f"MSE: {val_metrics['mse']:.6f}, R2: {val_metrics['r2']:.4f}"
    )
    print(
        f"Test  — loss: {test_metrics['loss']:.4f}, acc: {test_metrics['accuracy']:.4f}, "
        f"MSE: {test_metrics['mse']:.6f}, R2: {test_metrics['r2']:.4f}"
    )

    all_metrics = {
        "train_loss": train_metrics["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_mse": train_metrics["mse"],
        "train_r2": train_metrics["r2"],
        "val_loss": val_metrics["loss"],
        "val_accuracy": val_metrics["accuracy"],
        "val_mse": val_metrics["mse"],
        "val_r2": val_metrics["r2"],
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "test_mse": test_metrics["mse"],
        "test_r2": test_metrics["r2"],
        "best_optimizer": best_opt,
        "optimizer_comparison": leaderboard,
        "optimizer_val_accuracies": opt_histories,
        "val_predictions": val_metrics["predictions"].tolist(),
        "val_targets": val_metrics["targets"].tolist(),
    }

    save_artifacts(best_model, all_metrics, output_dir=_DEFAULT_OUTPUT)

    # Quality checks
    ok = True
    val_acc = val_metrics["accuracy"]
    if val_acc < 0.92:
        print(f"FAIL: val accuracy {val_acc:.4f} < 0.92")
        ok = False
    else:
        print(f"PASS: val accuracy >= 0.92 ({val_acc:.4f})")

    worst = min(r["val_accuracy"] for r in leaderboard)
    if worst < 0.75:
        print(f"FAIL: an optimizer fell below 0.75 val acc (worst={worst:.4f})")
        ok = False
    else:
        print(f"PASS: all optimizers val acc >= 0.75 (worst={worst:.4f})")

    adam_row = next(r for r in leaderboard if r["optimizer"] == "adam")
    if val_acc + 1e-6 < adam_row["val_accuracy"]:
        print("FAIL: selected best model underperforms standalone Adam run")
        ok = False
    else:
        print("PASS: selected best model matches or beats Adam run")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
