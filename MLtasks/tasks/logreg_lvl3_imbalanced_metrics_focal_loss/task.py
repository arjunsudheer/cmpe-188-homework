import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_kddcup99
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, classification_report
)
import matplotlib.pyplot as plt

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

OUTPUT_DIR = "./tasks/logreg_lvl3_imbalanced_metrics_focal_loss/output"

def get_task_metadata():
    """Return task metadata."""
    return {
        'task_name': 'logreg_imbalanced_metrics_focal_loss',
        'task_type': 'multiclass_classification',
        'dataset': 'KDD Cup 99 Network Intrusion Detection',
        'n_classes': 5,
        'description': 'Multiclass imbalanced classification using KDD Cup 99 dataset with severe class imbalance, comparing Weighted CrossEntropyLoss vs Focal Loss'
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dataloaders(batch_size=64, test_size=0.2, random_state=42):
    """Load KDD Cup 99 dataset and create dataloaders.
    
    KDD Cup 99 (network intrusion detection) is highly imbalanced with real-world complexity.
    This makes it suitable for demonstrating focal loss benefits over weighted CE loss.
    """
    print("   Loading KDD Cup 99 dataset (this may take a moment)...")
    
    # Fetch the dataset
    data = fetch_kddcup99(data_home='/tmp/kddcup99', download_if_missing=True)
    X, y = data.data, data.target
    
    # Decode target labels (they are bytes)
    y = np.array([label.decode('utf-8') if isinstance(label, bytes) else label for label in y])
    
    print(f"   Raw dataset shape: {X.shape}")
    print(f"   Unique classes: {np.unique(y)[:10]}...")  # Show first 10 classes
    
    # Get the top 5 most common classes to ensure we have data for all classes
    unique, counts = np.unique(y, return_counts=True)
    top_5_classes = unique[np.argsort(counts)[-5:]]
    
    print(f"   Top 5 classes: {top_5_classes}")
    print(f"   Their counts: {sorted(counts)[-5:]}")
    
    # Filter to top 5 classes
    mask = np.isin(y, top_5_classes)
    X = X[mask]
    y = y[mask]
    
    # Relabel to 0-4 for easier processing
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    print(f"   Filtered dataset shape: {X.shape}")
    
    # Encode categorical features (columns that contain strings)
    # KDD Cup 99 has categorical features: protocol_type (col 1), service (col 2), flag (col 3)
    X_processed = []
    
    for i, row in enumerate(X):
        processed_row = []
        for j, val in enumerate(row):
            # Try to convert to float
            try:
                processed_row.append(float(val))
            except (ValueError, TypeError):
                # If it's categorical, encode it as a hash for reproducibility
                # Use a simple hash-based encoding
                hash_val = hash(str(val)) % 1000 / 1000.0  # Normalize to [0, 1)
                processed_row.append(hash_val)
        X_processed.append(processed_row)
    
    X = np.array(X_processed, dtype=np.float32)
    
    print(f"   Processed dataset shape: {X.shape}")
    print(f"   Class distribution: {np.bincount(y, minlength=5)}")
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=5)
    # Avoid division by zero by using a safe weight calculation
    class_weights = np.zeros(5)
    for i in range(5):
        if class_counts[i] > 0:
            class_weights[i] = len(y_train) / (5 * class_counts[i])
        else:
            class_weights[i] = 1.0  # Default weight for empty classes
    
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    return train_loader, val_loader, class_weights, X_train, X_val, y_train, y_val

class MulticlassLogisticRegression(nn.Module):
    """Logistic Regression model for multiclass classification."""
    
    def __init__(self, input_dim, num_classes=5):
        super(MulticlassLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.linear(x)

def build_model(input_dim, num_classes=5):
    """Build the multiclass logistic regression model."""
    model = MulticlassLogisticRegression(input_dim, num_classes)
    return model.to(device)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection"
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Args:
            inputs: logits of shape (batch_size, num_classes)
            targets: target labels of shape (batch_size,)
        """
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def train(model, train_loader, val_loader, class_weights, loss_type='weighted_ce', 
          max_epochs=100, patience=10, lr=0.01):
    """Train the model with specified loss function and early stopping.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        class_weights: Class weights for imbalanced data
        loss_type: 'weighted_ce' for Weighted CrossEntropyLoss or 'focal' for Focal Loss
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        lr: Learning rate
    """
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Set up loss function
    if loss_type == 'weighted_ce':
        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
        loss_name = 'Weighted CrossEntropyLoss'
    elif loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, reduction='mean')
        loss_name = 'Focal Loss'
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    print(f"\n   Using {loss_name}")
    
    best_val_f1 = 0
    best_model_state = None
    patience_counter = 0
    train_losses = []
    val_f1_scores = []
    
    # Track detailed history
    history = {
        'epochs': [],
        'train_loss': [],
        'train_accuracy': [],
        'train_weighted_f1': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_weighted_f1': []
    }
    
    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_X)
            
            # Compute loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Evaluate on training set
        train_metrics = evaluate(model, train_loader)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader)
        val_f1 = val_metrics['weighted_f1']
        val_f1_scores.append(val_f1)
        
        # Store in history (every epoch for detailed tracking)
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(avg_loss)
        history['train_accuracy'].append(train_metrics['accuracy'])
        history['train_weighted_f1'].append(train_metrics['weighted_f1'])
        history['val_loss'].append(0.0)  # We don't compute val loss separately
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_weighted_f1'].append(val_metrics['weighted_f1'])
        
        # Early stopping based on F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"   Early stopping at epoch {epoch + 1}")
            break
        
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch [{epoch+1}/{max_epochs}], Loss: {avg_loss:.4f}, Val Weighted F1: {val_f1:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_f1_scores, history

def evaluate(model, data_loader):
    """Evaluate the model and compute metrics for multiclass classification.
    
    Returns metrics including accuracy, precision, recall, f1, weighted f1, MSE, and R2.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_logits = []
    
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(batch_y.cpu().numpy().flatten())
            all_logits.extend(outputs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_logits = np.array(all_logits)
    
    # Calculate multiclass metrics
    accuracy = accuracy_score(all_targets, all_preds)
    
    # Macro-averaged precision, recall, f1
    precision_macro = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall_macro = recall_score(all_targets, all_preds, average='macro', zero_division=0)
    f1_macro = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    # Weighted-averaged metrics (accounts for class imbalance)
    precision_weighted = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall_weighted = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    # For multiclass, compute one-hot encoded predictions and targets for MSE/R2
    num_classes = all_logits.shape[1]
    all_probs = torch.nn.functional.softmax(torch.FloatTensor(all_logits), dim=1).numpy()
    
    # One-hot encode targets for MSE and R2 computation
    num_classes = all_logits.shape[1]
    all_targets_onehot = np.eye(num_classes)[all_targets]
    
    # Calculate MSE (mean squared error between softmax probs and one-hot targets)
    mse = np.mean((all_probs - all_targets_onehot) ** 2)
    
    # Calculate R2 (for each class independently, then average)
    r2_scores = []
    for class_idx in range(num_classes):
        ss_res = np.sum((all_targets_onehot[:, class_idx] - all_probs[:, class_idx]) ** 2)
        ss_tot = np.sum((all_targets_onehot[:, class_idx] - np.mean(all_targets_onehot[:, class_idx])) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        r2_scores.append(r2)
    
    r2_mean = np.mean(r2_scores)
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision_weighted': precision_weighted,
        'recall_weighted': recall_weighted,
        'weighted_f1': f1_weighted,
        'mse': mse,
        'r2': r2_mean
    }

def predict(model, X):
    """Make predictions on input data."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        preds = torch.argmax(outputs, dim=1)
        probs = torch.nn.functional.softmax(outputs, dim=1)
    return preds.cpu().numpy().flatten(), probs.cpu().numpy()

def save_artifacts(model_ce, model_focal, train_losses_ce, val_f1_scores_ce, 
                   train_losses_focal, val_f1_scores_focal,
                   history_ce, history_focal,
                   metrics_ce, metrics_focal, X_train, y_train, X_val, y_val, 
                   output_dir):
    """Save models, plots, and metrics for both loss functions."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save models
    ce_model_path = os.path.join(output_dir, 'model_weighted_ce.pth')
    focal_model_path = os.path.join(output_dir, 'model_focal.pth')
    torch.save(model_ce.state_dict(), ce_model_path)
    torch.save(model_focal.state_dict(), focal_model_path)
    
    # Save metrics for Weighted CrossEntropyLoss
    ce_metrics_path = os.path.join(output_dir, 'metrics_weighted_ce.json')
    with open(ce_metrics_path, 'w') as f:
        json.dump(metrics_ce, f, indent=2)
    
    # Save metrics for Focal Loss
    focal_metrics_path = os.path.join(output_dir, 'metrics_focal.json')
    with open(focal_metrics_path, 'w') as f:
        json.dump(metrics_focal, f, indent=2)
    
    # Save training history for Weighted CrossEntropyLoss
    ce_history_path = os.path.join(output_dir, 'training_history_weighted_ce.json')
    with open(ce_history_path, 'w') as f:
        json.dump(history_ce, f, indent=2)
    
    # Save training history for Focal Loss
    focal_history_path = os.path.join(output_dir, 'training_history_focal.json')
    with open(focal_history_path, 'w') as f:
        json.dump(history_focal, f, indent=2)
    
    # Save comparison metrics
    comparison_path = os.path.join(output_dir, 'comparison.txt')
    with open(comparison_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("COMPARISON: Weighted CrossEntropyLoss vs Focal Loss\n")
        f.write("="*60 + "\n\n")
        
        f.write("WEIGHTED CROSSENTROPYLOSS METRICS:\n")
        f.write("-"*60 + "\n")
        for key, value in metrics_ce.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nFOCAL LOSS METRICS:\n")
        f.write("-"*60 + "\n")
        for key, value in metrics_focal.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "="*60 + "\n")
        f.write("PERFORMANCE COMPARISON:\n")
        f.write("="*60 + "\n")
        
        # Determine which loss performs better
        ce_f1 = metrics_ce['weighted_f1']
        focal_f1 = metrics_focal['weighted_f1']
        
        if ce_f1 > focal_f1:
            f.write(f"✓ Weighted CrossEntropyLoss achieves better Weighted F1: {ce_f1:.4f} vs {focal_f1:.4f}\n")
        elif focal_f1 > ce_f1:
            f.write(f"✓ Focal Loss achieves better Weighted F1: {focal_f1:.4f} vs {ce_f1:.4f}\n")
        else:
            f.write(f"✓ Both losses achieve equal Weighted F1: {ce_f1:.4f}\n")
        
        f.write(f"\nWeighted CrossEntropyLoss - Accuracy: {metrics_ce['accuracy']:.4f}, MSE: {metrics_ce['mse']:.4f}, R2: {metrics_ce['r2']:.4f}\n")
        f.write(f"Focal Loss - Accuracy: {metrics_focal['accuracy']:.4f}, MSE: {metrics_focal['mse']:.4f}, R2: {metrics_focal['r2']:.4f}\n")
        
        f.write(f"\n" + "="*60 + "\n")
        f.write("TRAINING HISTORY SUMMARY:\n")
        f.write("="*60 + "\n")
        f.write(f"Weighted CrossEntropyLoss - Trained for {len(history_ce['epochs'])} epochs\n")
        f.write(f"  Initial train accuracy: {history_ce['train_accuracy'][0]:.4f}, Final: {history_ce['train_accuracy'][-1]:.4f}\n")
        f.write(f"  Initial val accuracy:   {history_ce['val_accuracy'][0]:.4f}, Final: {history_ce['val_accuracy'][-1]:.4f}\n")
        f.write(f"  Initial train loss: {history_ce['train_loss'][0]:.4f}, Final: {history_ce['train_loss'][-1]:.4f}\n")
        
        f.write(f"\nFocal Loss - Trained for {len(history_focal['epochs'])} epochs\n")
        f.write(f"  Initial train accuracy: {history_focal['train_accuracy'][0]:.4f}, Final: {history_focal['train_accuracy'][-1]:.4f}\n")
        f.write(f"  Initial val accuracy:   {history_focal['val_accuracy'][0]:.4f}, Final: {history_focal['val_accuracy'][-1]:.4f}\n")
        f.write(f"  Initial train loss: {history_focal['train_loss'][0]:.4f}, Final: {history_focal['train_loss'][-1]:.4f}\n")
    
    # Plot training curves for both loss functions
    plt.figure(figsize=(15, 10))
    
    # CE Loss - Training Loss
    plt.subplot(2, 3, 1)
    plt.plot(history_ce['epochs'], history_ce['train_loss'], label='Training Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Weighted CrossEntropyLoss - Training Loss')
    plt.grid(True)
    plt.legend()
    
    # Focal Loss - Training Loss
    plt.subplot(2, 3, 2)
    plt.plot(history_focal['epochs'], history_focal['train_loss'], label='Training Loss', color='orange', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Focal Loss - Training Loss')
    plt.grid(True)
    plt.legend()
    
    # Training Accuracy Comparison
    plt.subplot(2, 3, 4)
    plt.plot(history_ce['epochs'], history_ce['train_accuracy'], label='Weighted CE', linewidth=2)
    plt.plot(history_focal['epochs'], history_focal['train_accuracy'], label='Focal Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Train Accuracy')
    plt.title('Training Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    # Validation Accuracy Comparison
    plt.subplot(2, 3, 5)
    plt.plot(history_ce['epochs'], history_ce['val_accuracy'], label='Weighted CE', linewidth=2)
    plt.plot(history_focal['epochs'], history_focal['val_accuracy'], label='Focal Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Validation Accuracy Comparison')
    plt.legend()
    plt.grid(True)
    
    # Validation F1 comparison
    plt.subplot(2, 3, 3)
    plt.plot(history_ce['epochs'], history_ce['train_weighted_f1'], label='Weighted CE Train', linewidth=2)
    plt.plot(history_focal['epochs'], history_focal['train_weighted_f1'], label='Focal Loss Train', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1 Score')
    plt.title('Training Weighted F1 Score Comparison')
    plt.legend()
    plt.grid(True)
    
    # Validation Weighted F1
    plt.subplot(2, 3, 6)
    plt.plot(history_ce['epochs'], history_ce['val_weighted_f1'], label='Weighted CE Val', linewidth=2)
    plt.plot(history_focal['epochs'], history_focal['val_weighted_f1'], label='Focal Loss Val', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1 Score')
    plt.title('Validation Weighted F1 Score Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_comparison.png'), dpi=100)
    plt.close()
    
    print(f"Artifacts saved to {output_dir}")

def main():
    """Main function to run the complete training and evaluation pipeline with both loss functions."""
    print("=" * 80)
    print("KDD Cup 99 Network Intrusion Detection - Multiclass Logistic Regression")
    print("Comparing Weighted CrossEntropyLoss vs Focal Loss")
    print("=" * 80)
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Description: {metadata['description']}")
    print(f"Number of classes: {metadata['n_classes']}")
    
    # Create dataloaders
    print("\n[1] Creating dataloaders...")
    train_loader, val_loader, class_weights, X_train, X_val, y_train, y_val = make_dataloaders()
    print(f"   Train samples: {len(y_train)}, Val samples: {len(y_val)}")
    print(f"   Feature dimension: {X_train.shape[1]}")
    print(f"   Class distribution in train: {np.bincount(y_train, minlength=5)}")
    print(f"   Class weights: {class_weights.cpu().numpy()}")
    
    # ===================================================================
    # TRAIN WITH WEIGHTED CROSSENTROPYLOSS
    # ===================================================================
    print("\n" + "="*80)
    print("[2a] Training with Weighted CrossEntropyLoss...")
    print("="*80)
    
    # Build model for CE loss
    model_ce = build_model(input_dim=X_train.shape[1], num_classes=5)
    print(f"   Model: MulticlassLogisticRegression({X_train.shape[1]}, 5)")
    
    # Train model with CE loss
    model_ce, train_losses_ce, val_f1_scores_ce, history_ce = train(
        model_ce, train_loader, val_loader, class_weights,
        loss_type='weighted_ce',
        max_epochs=100, patience=15, lr=0.01
    )
    
    # Evaluate on training set
    print("\n   Evaluating on training set...")
    train_metrics_ce = evaluate(model_ce, train_loader)
    print(f"   Training Metrics (CE):")
    print(f"     Accuracy:           {train_metrics_ce['accuracy']:.4f}")
    print(f"     Precision (macro):  {train_metrics_ce['precision_macro']:.4f}")
    print(f"     Recall (macro):     {train_metrics_ce['recall_macro']:.4f}")
    print(f"     F1 (macro):         {train_metrics_ce['f1_macro']:.4f}")
    print(f"     F1 (weighted):      {train_metrics_ce['weighted_f1']:.4f}")
    print(f"     MSE:                {train_metrics_ce['mse']:.4f}")
    print(f"     R2:                 {train_metrics_ce['r2']:.4f}")
    
    # Evaluate on validation set
    print("\n   Evaluating on validation set...")
    val_metrics_ce = evaluate(model_ce, val_loader)
    print(f"   Validation Metrics (CE):")
    print(f"     Accuracy:           {val_metrics_ce['accuracy']:.4f}")
    print(f"     Precision (macro):  {val_metrics_ce['precision_macro']:.4f}")
    print(f"     Recall (macro):     {val_metrics_ce['recall_macro']:.4f}")
    print(f"     F1 (macro):         {val_metrics_ce['f1_macro']:.4f}")
    print(f"     F1 (weighted):      {val_metrics_ce['weighted_f1']:.4f}")
    print(f"     MSE:                {val_metrics_ce['mse']:.4f}")
    print(f"     R2:                 {val_metrics_ce['r2']:.4f}")
    
    # ===================================================================
    # TRAIN WITH FOCAL LOSS
    # ===================================================================
    print("\n" + "="*80)
    print("[2b] Training with Focal Loss...")
    print("="*80)
    
    # Reset seeds for fair comparison
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Build model for Focal loss
    model_focal = build_model(input_dim=X_train.shape[1], num_classes=5)
    print(f"   Model: MulticlassLogisticRegression({X_train.shape[1]}, 5)")
    
    # Train model with Focal loss
    model_focal, train_losses_focal, val_f1_scores_focal, history_focal = train(
        model_focal, train_loader, val_loader, class_weights,
        loss_type='focal',
        max_epochs=100, patience=15, lr=0.01
    )
    
    # Evaluate on training set
    print("\n   Evaluating on training set...")
    train_metrics_focal = evaluate(model_focal, train_loader)
    print(f"   Training Metrics (Focal):")
    print(f"     Accuracy:           {train_metrics_focal['accuracy']:.4f}")
    print(f"     Precision (macro):  {train_metrics_focal['precision_macro']:.4f}")
    print(f"     Recall (macro):     {train_metrics_focal['recall_macro']:.4f}")
    print(f"     F1 (macro):         {train_metrics_focal['f1_macro']:.4f}")
    print(f"     F1 (weighted):      {train_metrics_focal['weighted_f1']:.4f}")
    print(f"     MSE:                {train_metrics_focal['mse']:.4f}")
    print(f"     R2:                 {train_metrics_focal['r2']:.4f}")
    
    # Evaluate on validation set
    print("\n   Evaluating on validation set...")
    val_metrics_focal = evaluate(model_focal, val_loader)
    print(f"   Validation Metrics (Focal):")
    print(f"     Accuracy:           {val_metrics_focal['accuracy']:.4f}")
    print(f"     Precision (macro):  {val_metrics_focal['precision_macro']:.4f}")
    print(f"     Recall (macro):     {val_metrics_focal['recall_macro']:.4f}")
    print(f"     F1 (macro):         {val_metrics_focal['f1_macro']:.4f}")
    print(f"     F1 (weighted):      {val_metrics_focal['weighted_f1']:.4f}")
    print(f"     MSE:                {val_metrics_focal['mse']:.4f}")
    print(f"     R2:                 {val_metrics_focal['r2']:.4f}")
    
    # ===================================================================
    # COMPARISON
    # ===================================================================
    print("\n" + "="*80)
    print("[3] Comparing Loss Functions")
    print("="*80)
    
    ce_f1 = val_metrics_ce['weighted_f1']
    focal_f1 = val_metrics_focal['weighted_f1']
    
    print(f"\n   Weighted CrossEntropyLoss - Val Weighted F1:  {ce_f1:.4f}")
    print(f"   Focal Loss - Val Weighted F1:                {focal_f1:.4f}")
    
    if ce_f1 > focal_f1:
        print(f"\n   ★ Weighted CrossEntropyLoss performs better! (Δ = {ce_f1 - focal_f1:.4f})")
        better_loss = 'Weighted CrossEntropyLoss'
    elif focal_f1 > ce_f1:
        print(f"\n   ★ Focal Loss performs better! (Δ = {focal_f1 - ce_f1:.4f})")
        better_loss = 'Focal Loss'
    else:
        print(f"\n   ★ Both losses achieve equal performance!")
        better_loss = 'Both equal'
    
    print(f"\n   Other metrics comparison:")
    print(f"   CrossEntropyLoss  - Accuracy: {val_metrics_ce['accuracy']:.4f}, MSE: {val_metrics_ce['mse']:.4f}, R2: {val_metrics_ce['r2']:.4f}")
    print(f"   Focal Loss        - Accuracy: {val_metrics_focal['accuracy']:.4f}, MSE: {val_metrics_focal['mse']:.4f}, R2: {val_metrics_focal['r2']:.4f}")
    
    # Save artifacts
    print(f"\n[4] Saving artifacts...")
    output_dir = OUTPUT_DIR
    save_artifacts(
        model_ce, model_focal,
        train_losses_ce, val_f1_scores_ce,
        train_losses_focal, val_f1_scores_focal,
        history_ce, history_focal,
        val_metrics_ce, val_metrics_focal,
        X_train, y_train, X_val, y_val,
        output_dir
    )
    
    # Quality threshold assertions
    print("\n" + "=" * 80)
    print("Quality Threshold Assertions")
    print("=" * 80)
    
    all_passed = True
    
    # Check accuracy threshold for CE loss model
    try:
        assert val_metrics_ce['accuracy'] > 0.50, f"CE Accuracy {val_metrics_ce['accuracy']:.4f} <= 0.50"
        print("✓ Weighted CE - Accuracy > 0.50: PASSED")
    except AssertionError as e:
        print(f"✗ Weighted CE - Accuracy > 0.50: FAILED - {e}")
        all_passed = False
    
    # Check accuracy threshold for Focal loss model
    try:
        assert val_metrics_focal['accuracy'] > 0.50, f"Focal Accuracy {val_metrics_focal['accuracy']:.4f} <= 0.50"
        print("✓ Focal Loss - Accuracy > 0.50: PASSED")
    except AssertionError as e:
        print(f"✗ Focal Loss - Accuracy > 0.50: FAILED - {e}")
        all_passed = False
    
    # Check weighted F1 threshold for CE loss model
    try:
        assert val_metrics_ce['weighted_f1'] > 0.30, f"CE F1 {val_metrics_ce['weighted_f1']:.4f} <= 0.30"
        print("✓ Weighted CE - Weighted F1 > 0.30: PASSED")
    except AssertionError as e:
        print(f"✗ Weighted CE - Weighted F1 > 0.30: FAILED - {e}")
        all_passed = False
    
    # Check weighted F1 threshold for Focal loss model
    try:
        assert val_metrics_focal['weighted_f1'] > 0.30, f"Focal F1 {val_metrics_focal['weighted_f1']:.4f} <= 0.30"
        print("✓ Focal Loss - Weighted F1 > 0.30: PASSED")
    except AssertionError as e:
        print(f"✗ Focal Loss - Weighted F1 > 0.30: FAILED - {e}")
        all_passed = False
    
    # Check MSE threshold for CE loss model
    try:
        assert val_metrics_ce['mse'] < 0.50, f"CE MSE {val_metrics_ce['mse']:.4f} >= 0.50"
        print("✓ Weighted CE - MSE < 0.50: PASSED")
    except AssertionError as e:
        print(f"✗ Weighted CE - MSE < 0.50: FAILED - {e}")
        all_passed = False
    
    # Check MSE threshold for Focal loss model
    try:
        assert val_metrics_focal['mse'] < 0.50, f"Focal MSE {val_metrics_focal['mse']:.4f} >= 0.50"
        print("✓ Focal Loss - MSE < 0.50: PASSED")
    except AssertionError as e:
        print(f"✗ Focal Loss - MSE < 0.50: FAILED - {e}")
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 80)
    print(f"BETTER PERFORMING LOSS FUNCTION: {better_loss}")
    if all_passed:
        print("PASS: All quality thresholds met!")
    else:
        print("FAIL: Some quality thresholds not met!")
    print("=" * 80)
    
    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)