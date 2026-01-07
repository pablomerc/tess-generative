"""
Downstream Task Evaluation for Flow V5 Model

This script evaluates the quality of learned representations by:
1. Loading a pretrained flow_v5 model
2. Extracting latent representations (z_number, z_filter) from data
3. Training MLPs to predict labels from latents (classification or regression)
4. Evaluating classification accuracy or R2 scores

Usage:
    python -m flow_v5.downstream_tasks --checkpoint <path> --num_samples <N> --task <classification|regression>
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

# Add repo root to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
de_path = os.path.join(repo_root, 'double-encoder-model')
if de_path not in sys.path:
    sys.path.insert(0, de_path)

from flow_v5.data import make_triplet_creator, make_multi_triplet_creator
from flow_v5.model import build_model
from flow_v5.utils import normalize_to_flow_range
from flow_v5 import config as cfg


class LatentClassifier(nn.Module):
    """
    MLP classifier for downstream classification tasks.
    
    Outputs raw logits (no softmax) - CrossEntropyLoss applies log_softmax internally.
    """
    def __init__(self, input_dim, num_classes, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final layer outputs raw logits (no activation)
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass returning raw logits.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Logits [batch_size, num_classes] (no softmax applied)
        """
        return self.classifier(x)


class LatentRegressor(nn.Module):
    """
    MLP regressor for downstream regression tasks.
    
    Outputs continuous values for regression (no activation on final layer).
    """
    def __init__(self, input_dim, output_dim=1, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Final layer outputs raw values (no activation for regression)
        layers.append(nn.Linear(prev_dim, output_dim))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass returning continuous predictions.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Predictions [batch_size, output_dim] (continuous values)
        """
        return self.regressor(x)


def extract_latent_representations(
    model,
    triplet_creator,
    num_samples: int = 10000,
    dataset: str = 'test',
    multi_samples: bool = False,
    num_filter_augs: int = 2,
    num_number_augs: int = 2
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract latent representations from the pretrained model

    Args:
        model: Pretrained DoubleEncoderFlowMatching model
        triplet_creator: TripletCreator instance
        num_samples: Number of samples to extract
        dataset: Dataset to use ('train' or 'test')
        multi_samples: Whether to use multi-sample encoding
        num_filter_augs: Number of filter augmentations (for multi-sample)
        num_number_augs: Number of number augmentations (for multi-sample)

    Returns:
        tuple: (z_number, z_filter, digit_labels, rotation_labels, [optional: combined_z])
    """
    model.eval()
    device = next(model.parameters()).device

    z_number_list = []
    z_filter_list = []
    combined_z_list = []
    digit_labels_list = []
    rotation_labels_list = []

    batch_size = 256
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Extracting latent representations from {num_samples} samples...")
    print(f"Using multi-sample encoding: {multi_samples}")

    with torch.no_grad():
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{num_batches}")

            if multi_samples:
                batch = triplet_creator.create_batch_multi_triplets(
                    batch_size=batch_size,
                    dataset=dataset,
                    num_filter_augs=num_filter_augs,
                    num_number_augs=num_number_augs
                )

                anchor = normalize_to_flow_range(batch["anchor"]).to(device)
                same_number_augments = normalize_to_flow_range(batch["same_number_augments"]).to(device)
                same_filter_augments = normalize_to_flow_range(batch["same_filter_augments"]).to(device)
                anchor_labels = batch["anchor_labels"].to(device)
                filter_labels = batch["filter_labels"].to(device)

                # Use multi-sample encoding
                combined_z, pooled_number_z, pooled_filter_z = model.multi_sample_encoding(
                    same_number_augments,
                    same_filter_augments
                )

                z_number_list.append(pooled_number_z.cpu())
                z_filter_list.append(pooled_filter_z.cpu())
                combined_z_list.append(combined_z.cpu())
                digit_labels_list.append(anchor_labels.cpu())
                rotation_labels_list.append(filter_labels[:, 0].cpu())  # Use first filter label
            else:
                # Single-sample encoding
                (ground_truth, different_digit, same_digit, original_labels, different_labels,
                 ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                    triplet_creator.create_batch_triplets(batch_size, dataset=dataset)

                ground_truth = normalize_to_flow_range(ground_truth.to(device))
                different_digit = normalize_to_flow_range(different_digit.to(device))
                same_digit = normalize_to_flow_range(same_digit.to(device))
                original_labels = original_labels.to(device)
                ground_truth_rotations = ground_truth_rotations.to(device)

                # Extract latent representations
                number_z, filter_z, _, _, _, _ = model.encode_only(same_digit, different_digit)
                combined_z = torch.cat([number_z, filter_z], dim=1)

                z_number_list.append(number_z.cpu())
                z_filter_list.append(filter_z.cpu())
                combined_z_list.append(combined_z.cpu())
                digit_labels_list.append(original_labels.cpu())
                rotation_labels_list.append(ground_truth_rotations.cpu())

    # Concatenate all batches
    z_number = torch.cat(z_number_list, dim=0)[:num_samples]
    z_filter = torch.cat(z_filter_list, dim=0)[:num_samples]
    combined_z = torch.cat(combined_z_list, dim=0)[:num_samples]
    digit_labels = torch.cat(digit_labels_list, dim=0)[:num_samples]
    rotation_labels = torch.cat(rotation_labels_list, dim=0)[:num_samples]

    print(f"Extracted {z_number.shape[0]} samples")
    print(f"z_number shape: {z_number.shape}")
    print(f"z_filter shape: {z_filter.shape}")
    print(f"combined_z shape: {combined_z.shape}")

    return z_number, z_filter, combined_z, digit_labels, rotation_labels


def train_classifier(
    z_data: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    model_name: str,
    learning_rate: float = 0.001,
    num_epochs: int = 50,
    batch_size: int = 64,
    device: str = None
) -> Dict[str, Any]:
    """
    Train a downstream classifier on latent representations

    Returns:
        dict: Results containing model, accuracies, predictions, and labels
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    z_data = z_data.float()
    labels = labels.long()

    # Split data: 70% train, 15% val, 15% test
    z_train, z_temp, labels_train, labels_temp = train_test_split(
        z_data, labels, test_size=0.3, random_state=42, stratify=labels
    )
    z_val, z_test, labels_val, labels_test = train_test_split(
        z_temp, labels_temp, test_size=0.5, random_state=42, stratify=labels_temp
    )

    # Create datasets and dataloaders
    train_dataset = TensorDataset(z_train, labels_train)
    val_dataset = TensorDataset(z_val, labels_val)
    test_dataset = TensorDataset(z_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = LatentClassifier(z_data.shape[1], num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nTraining {model_name} classifier...")
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_correct = 0
        train_total = 0

        for batch_z, batch_labels in train_loader:
            batch_z, batch_labels = batch_z.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_z)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()

        train_acc = train_correct / train_total
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_z, batch_labels in val_loader:
                batch_z, batch_labels = batch_z.to(device), batch_labels.to(device)
                outputs = model(batch_z)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()

        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # Test evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_z, batch_labels in test_loader:
            batch_z, batch_labels = batch_z.to(device), batch_labels.to(device)
            outputs = model(batch_z)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())

    test_acc = test_correct / test_total

    print(f"{model_name} Final Results:")
    print(f"Train Accuracy: {train_accuracies[-1]:.4f}")
    print(f"Validation Accuracy: {val_accuracies[-1]:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return {
        'model': model,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies,
        'test_acc': test_acc,
        'predictions': all_predictions,
        'true_labels': all_labels
    }


def train_regressor(
    z_data: torch.Tensor,
    labels: torch.Tensor,
    model_name: str,
    learning_rate: float = 0.001,
    num_epochs: int = 50,
    batch_size: int = 64,
    device: str = None
) -> Dict[str, Any]:
    """
    Train a downstream regressor on latent representations

    Returns:
        dict: Results containing model, R2 scores, MSE, MAE, predictions, and labels
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    z_data = z_data.float()
    labels = labels.float()

    # Split data: 70% train, 15% val, 15% test
    z_train, z_temp, labels_train, labels_temp = train_test_split(
        z_data, labels, test_size=0.3, random_state=42
    )
    z_val, z_test, labels_val, labels_test = train_test_split(
        z_temp, labels_temp, test_size=0.5, random_state=42
    )

    # Create datasets and dataloaders
    train_dataset = TensorDataset(z_train, labels_train)
    val_dataset = TensorDataset(z_val, labels_val)
    test_dataset = TensorDataset(z_test, labels_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = LatentRegressor(z_data.shape[1], output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    print(f"\nTraining {model_name} regressor...")
    train_r2_scores = []
    val_r2_scores = []

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_preds = []
        train_targets = []

        for batch_z, batch_labels in train_loader:
            batch_z, batch_labels = batch_z.to(device), batch_labels.to(device)

            optimizer.zero_grad()
            outputs = model(batch_z).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_preds.extend(outputs.detach().cpu().numpy())
            train_targets.extend(batch_labels.cpu().numpy())

        train_r2 = r2_score(train_targets, train_preds)
        train_r2_scores.append(train_r2)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []

        with torch.no_grad():
            for batch_z, batch_labels in val_loader:
                batch_z, batch_labels = batch_z.to(device), batch_labels.to(device)
                outputs = model(batch_z).squeeze()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_labels.cpu().numpy())

        val_r2 = r2_score(val_targets, val_preds)
        val_r2_scores.append(val_r2)

        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: Train R2: {train_r2:.4f}, Val R2: {val_r2:.4f}")

    # Test evaluation
    model.eval()
    test_preds = []
    test_targets = []

    with torch.no_grad():
        for batch_z, batch_labels in test_loader:
            batch_z, batch_labels = batch_z.to(device), batch_labels.to(device)
            outputs = model(batch_z).squeeze()
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(batch_labels.cpu().numpy())

    test_r2 = r2_score(test_targets, test_preds)
    test_mse = mean_squared_error(test_targets, test_preds)
    test_mae = mean_absolute_error(test_targets, test_preds)

    print(f"{model_name} Final Results:")
    print(f"Train R2: {train_r2_scores[-1]:.4f}")
    print(f"Validation R2: {val_r2_scores[-1]:.4f}")
    print(f"Test R2: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")

    return {
        'model': model,
        'train_r2': train_r2_scores,
        'val_r2': val_r2_scores,
        'test_r2': test_r2,
        'test_mse': test_mse,
        'test_mae': test_mae,
        'predictions': test_preds,
        'true_labels': test_targets
    }


def plot_results(all_results: Dict[str, Dict[str, Any]], output_dir: str, task_type: str = 'classification'):
    """Plot training curves and summary visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    if task_type == 'classification':
        # Plot training curves
        for name, results in all_results.items():
            epochs = range(1, len(results['train_acc']) + 1)
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, results['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
            plt.plot(epochs, results['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
            plt.title(f'{name} Training Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            filename = f"{name.lower().replace(' ', '_')}_training_curves.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()

        # Summary bar plot
        test_names = list(all_results.keys())
        test_accuracies = [all_results[name]['test_acc'] for name in test_names]

        plt.figure(figsize=(12, 8))
        bars = plt.bar(test_names, test_accuracies, alpha=0.7)
        plt.xlabel('Test Configuration')
        plt.ylabel('Test Accuracy')
        plt.title('Downstream Classification Task Results')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3)

        for bar, acc in zip(bars, test_accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'classification_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()

    else:  # regression
        # Plot training curves
        for name, results in all_results.items():
            epochs = range(1, len(results['train_r2']) + 1)
            plt.figure(figsize=(8, 6))
            plt.plot(epochs, results['train_r2'], 'b-', label='Train R2', linewidth=2)
            plt.plot(epochs, results['val_r2'], 'r-', label='Validation R2', linewidth=2)
            plt.title(f'{name} Training Curves')
            plt.xlabel('Epoch')
            plt.ylabel('R2 Score')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            filename = f"{name.lower().replace(' ', '_')}_training_curves.png"
            plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()

        # Summary bar plot
        test_names = list(all_results.keys())
        test_r2_scores = [all_results[name]['test_r2'] for name in test_names]

        plt.figure(figsize=(12, 8))
        bars = plt.bar(test_names, test_r2_scores, alpha=0.7)
        plt.xlabel('Test Configuration')
        plt.ylabel('Test R2 Score')
        plt.title('Downstream Regression Task Results')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)

        for bar, r2 in zip(bars, test_r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{r2:.3f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'regression_summary.png'), dpi=150, bbox_inches='tight')
        plt.close()


def save_results(all_results: Dict[str, Dict[str, Any]], output_dir: str, task_type: str = 'classification'):
    """Save results to text file and CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    # Save text file
    results_file = os.path.join(output_dir, f"downstream_results_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write(f"Downstream Task Evaluation Results ({task_type.upper()})\n")
        f.write("=" * 60 + "\n\n")

        f.write("SUMMARY TABLE\n")
        f.write("-" * 60 + "\n")
        if task_type == 'classification':
            f.write(f"{'Test':<25} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}\n")
            f.write("-" * 60 + "\n")
            for test_name, results in all_results.items():
                f.write(f"{test_name:<25} {results['train_acc'][-1]:<12.4f} {results['val_acc'][-1]:<12.4f} {results['test_acc']:<12.4f}\n")
        else:
            f.write(f"{'Test':<25} {'Train R2':<12} {'Val R2':<12} {'Test R2':<12} {'Test MSE':<12} {'Test MAE':<12}\n")
            f.write("-" * 60 + "\n")
            for test_name, results in all_results.items():
                f.write(f"{test_name:<25} {results['train_r2'][-1]:<12.4f} {results['val_r2'][-1]:<12.4f} "
                       f"{results['test_r2']:<12.4f} {results['test_mse']:<12.4f} {results['test_mae']:<12.4f}\n")

        f.write("\n" + "="*60 + "\n\n")

        # Detailed results
        for test_name, results in all_results.items():
            f.write(f"{test_name.upper()} RESULTS\n")
            f.write("-" * 40 + "\n")
            if task_type == 'classification':
                f.write(f"Train Accuracy: {results['train_acc'][-1]:.4f}\n")
                f.write(f"Validation Accuracy: {results['val_acc'][-1]:.4f}\n")
                f.write(f"Test Accuracy: {results['test_acc']:.4f}\n\n")
                f.write("Classification Report:\n")
                f.write(classification_report(results['true_labels'], results['predictions']))
            else:
                f.write(f"Train R2: {results['train_r2'][-1]:.4f}\n")
                f.write(f"Validation R2: {results['val_r2'][-1]:.4f}\n")
                f.write(f"Test R2: {results['test_r2']:.4f}\n")
                f.write(f"Test MSE: {results['test_mse']:.4f}\n")
                f.write(f"Test MAE: {results['test_mae']:.4f}\n")
            f.write("\n" + "="*60 + "\n\n")

    print(f"Results saved to: {results_file}")

    # Save CSV
    csv_file = os.path.join(output_dir, f"downstream_results_{timestamp}.csv")
    with open(csv_file, 'w') as f:
        if task_type == 'classification':
            f.write("Test,Train_Accuracy,Val_Accuracy,Test_Accuracy\n")
            for test_name, results in all_results.items():
                f.write(f"{test_name},{results['train_acc'][-1]:.6f},{results['val_acc'][-1]:.6f},{results['test_acc']:.6f}\n")
        else:
            f.write("Test,Train_R2,Val_R2,Test_R2,Test_MSE,Test_MAE\n")
            for test_name, results in all_results.items():
                f.write(f"{test_name},{results['train_r2'][-1]:.6f},{results['val_r2'][-1]:.6f},"
                       f"{results['test_r2']:.6f},{results['test_mse']:.6f},{results['test_mae']:.6f}\n")

    print(f"CSV saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Downstream task evaluation for Flow V5")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--num_samples", type=int, default=10000,
                       help="Number of samples to encode")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification",
                       help="Task type: classification or regression")
    parser.add_argument("--dataset", choices=["mnist", "fashion_mnist"], default=None,
                       help="Dataset type (defaults to config)")
    parser.add_argument("--multi_samples", action="store_true",
                       help="Use multi-sample encoding")
    parser.add_argument("--num_filter_augs", type=int, default=2,
                       help="Number of filter augmentations (for multi-sample)")
    parser.add_argument("--num_number_augs", type=int, default=2,
                       help="Number of number augmentations (for multi-sample)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    parser.add_argument("--epochs", type=int, default=50,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for downstream training")

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("\nLoading pretrained model...")
    dataset_type = args.dataset or cfg.DATASET_TYPE
    multi_samples = args.multi_samples or getattr(cfg, 'USE_MULTI_SAMPLES', False)
    
    model = build_model(device=device, multi_samples=multi_samples)
    
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    else:
        print(f"ERROR: Checkpoint not found at {args.checkpoint}")
        return

    model.eval()

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"flow_models/downstream_{dataset_type}_{args.task}_{timestamp}"
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Create triplet creator
    print("\nInitializing triplet creator...")
    if multi_samples:
        triplet_creator = make_multi_triplet_creator(dataset_type=dataset_type)
    else:
        triplet_creator = make_triplet_creator(dataset_type=dataset_type)
    triplet_creator.get_dataset_info()

    # Extract latent representations
    print("\nExtracting latent representations...")
    z_number, z_filter, combined_z, digit_labels, rotation_labels = extract_latent_representations(
        model=model,
        triplet_creator=triplet_creator,
        num_samples=args.num_samples,
        dataset='test',
        multi_samples=multi_samples,
        num_filter_augs=args.num_filter_augs,
        num_number_augs=args.num_number_augs
    )

    # Prepare labels
    if args.task == 'classification':
        # For classification: predict digit from z_number, rotation class from z_filter
        num_digit_classes = len(triplet_creator.class_names)
        
        # Convert rotation angles to discrete classes
        rotation_range = getattr(cfg, 'ROTATION_DEGREES', 30)
        rotation_step = getattr(cfg, 'ROTATION_STEP', 5)
        angle_bins = torch.arange(-rotation_range, rotation_range + 1, rotation_step)
        rotation_class_labels = []
        for angle in rotation_labels:
            distances = torch.abs(angle_bins - angle)
            closest_bin = torch.argmin(distances)
            rotation_class_labels.append(closest_bin)
        rotation_class_labels = torch.tensor(rotation_class_labels)

        num_rotation_classes = len(torch.unique(rotation_class_labels))

        print(f"\nClassification task:")
        print(f"Number of digit classes: {num_digit_classes}")
        print(f"Number of rotation classes: {num_rotation_classes}")

        # Train classifiers
        all_results = {}

        # Test 1: Digit classification from z_number
        print("\n" + "="*50)
        print("TEST 1: Digit classification from z_number")
        print("="*50)
        results = train_classifier(
            z_number, digit_labels, num_digit_classes,
            "Digit_from_z_number",
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        all_results['digit_from_z_number'] = results

        # Test 2: Rotation classification from z_filter
        print("\n" + "="*50)
        print("TEST 2: Rotation classification from z_filter")
        print("="*50)
        results = train_classifier(
            z_filter, rotation_class_labels, num_rotation_classes,
            "Rotation_from_z_filter",
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        all_results['rotation_from_z_filter'] = results

        # Test 3: Digit classification from z_filter (cross-test)
        print("\n" + "="*50)
        print("TEST 3: Digit classification from z_filter (cross-test)")
        print("="*50)
        results = train_classifier(
            z_filter, digit_labels, num_digit_classes,
            "Digit_from_z_filter",
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        all_results['digit_from_z_filter'] = results

        # Test 4: Rotation classification from z_number (cross-test)
        print("\n" + "="*50)
        print("TEST 4: Rotation classification from z_number (cross-test)")
        print("="*50)
        results = train_classifier(
            z_number, rotation_class_labels, num_rotation_classes,
            "Rotation_from_z_number",
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        all_results['rotation_from_z_number'] = results

    else:  # regression
        print(f"\nRegression task:")
        print(f"Predicting rotation angles (continuous)")

        # Train regressors
        all_results = {}

        # Test 1: Rotation regression from z_filter
        print("\n" + "="*50)
        print("TEST 1: Rotation regression from z_filter")
        print("="*50)
        results = train_regressor(
            z_filter, rotation_labels,
            "Rotation_from_z_filter",
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        all_results['rotation_from_z_filter'] = results

        # Test 2: Rotation regression from z_number (cross-test)
        print("\n" + "="*50)
        print("TEST 2: Rotation regression from z_number (cross-test)")
        print("="*50)
        results = train_regressor(
            z_number, rotation_labels,
            "Rotation_from_z_number",
            learning_rate=args.lr,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            device=device
        )
        all_results['rotation_from_z_number'] = results

    # Plot and save results
    print("\nCreating visualizations...")
    plot_results(all_results, output_dir, task_type=args.task)

    print("\nSaving results...")
    save_results(all_results, output_dir, task_type=args.task)

    # Print summary
    print("\n" + "="*80)
    print(f"DOWNSTREAM TASK EVALUATION SUMMARY ({args.task.upper()})")
    print("="*80)
    if args.task == 'classification':
        print(f"{'Test':<25} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
        print("-" * 80)
        for test_name, results in all_results.items():
            print(f"{test_name:<25} {results['train_acc'][-1]:<12.4f} {results['val_acc'][-1]:<12.4f} {results['test_acc']:<12.4f}")
    else:
        print(f"{'Test':<25} {'Train R2':<12} {'Val R2':<12} {'Test R2':<12} {'Test MSE':<12} {'Test MAE':<12}")
        print("-" * 80)
        for test_name, results in all_results.items():
            print(f"{test_name:<25} {results['train_r2'][-1]:<12.4f} {results['val_r2'][-1]:<12.4f} "
                  f"{results['test_r2']:<12.4f} {results['test_mse']:<12.4f} {results['test_mae']:<12.4f}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()


'''
python -m flow_v5.downstream_tasks \
    --checkpoint flow_models/mnist/<your_run>/double_encoder_flow_model_mnist_epoch_250_<timestamp>.pth \
    --multi_samples \
    --num_filter_augs 5 \
    --num_number_augs 5 \
    --task classification
'''

