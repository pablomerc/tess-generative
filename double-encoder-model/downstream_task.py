"""
Downstream Task Evaluation for Double Encoder Model

This script evaluates the quality of learned representations by:
1. Loading a pretrained double encoder model
2. Extracting latent representations (z_number, z_filter) from test data
3. Training MLPs to predict digit labels from z_number and rotation angles from z_filter
4. Evaluating and visualizing the classification accuracy

DISENTANGLEMENT EVALUATION LOGIC:
- z_number comes from 'same_digit_different_rotation' (input to number encoder)
- z_filter comes from 'different_digit_same_rotation' (input to filter encoder)
- To test disentanglement:
  * z_number should capture digit identity (high accuracy on digit classification)
  * z_filter should capture rotation style (high accuracy on rotation classification)
  * z_number should NOT capture rotation style (low accuracy on rotation classification)
  * z_filter should NOT capture digit identity (low accuracy on digit classification)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Import our modules
from config import *
from triplet_creation import TripletCreator
from decoder import DoubleEncoderDecoder
from utils import load_model


class LatentClassifier(nn.Module):
    """
    Simple MLP classifier for downstream tasks
    """
    def __init__(self, input_dim, num_classes, hidden_dims=[256,128, 64]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


def extract_latent_representations(model, triplet_creator, num_samples=10000, dataset='test'):
    """
    Extract latent representations from the pretrained model

    Args:
        model: Pretrained DoubleEncoderDecoder model
        triplet_creator: TripletCreator instance
        num_samples: Number of samples to extract
        dataset: Dataset to use ('train' or 'test')

    Returns:
        tuple: (z_number, z_filter, digit_labels, rotation_labels)
    """
    model.eval()
    device = next(model.parameters()).device

    z_number_list = []
    z_filter_list = []
    digit_labels_list = []
    rotation_labels_list = []

    batch_size = 256  # Process in batches for efficiency
    num_batches = (num_samples + batch_size - 1) // batch_size

    print(f"Extracting latent representations from {num_samples} samples...")

    with torch.no_grad():
        for batch_idx in range(num_batches):
            if batch_idx % 10 == 0:
                print(f"Processing batch {batch_idx + 1}/{num_batches}")

            # Create triplet batch
            (ground_truth, different_digit, same_digit, original_labels, different_labels,
             ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
                triplet_creator.create_batch_triplets(batch_size, dataset=dataset)

            # Move to device
            ground_truth = ground_truth.to(device)
            different_digit = different_digit.to(device)
            same_digit = same_digit.to(device)
            original_labels = original_labels.to(device)
            ground_truth_rotations = ground_truth_rotations.to(device)

            # Extract latent representations
            number_z, filter_z, _, _, _, _ = model.encode_only(same_digit, different_digit)

            # Store results
            z_number_list.append(number_z.cpu())
            z_filter_list.append(filter_z.cpu())
            digit_labels_list.append(original_labels.cpu())
            rotation_labels_list.append(ground_truth_rotations.cpu())

    # Concatenate all batches
    z_number = torch.cat(z_number_list, dim=0)[:num_samples]
    z_filter = torch.cat(z_filter_list, dim=0)[:num_samples]
    digit_labels = torch.cat(digit_labels_list, dim=0)[:num_samples]
    rotation_labels = torch.cat(rotation_labels_list, dim=0)[:num_samples]

    print(f"Extracted {z_number.shape[0]} samples")
    print(f"z_number shape: {z_number.shape}")
    print(f"z_filter shape: {z_filter.shape}")
    print(f"Note: z_number comes from 'same_digit_different_rotation' (number encoder input)")
    print(f"Note: z_filter comes from 'different_digit_same_rotation' (filter encoder input)")
    print(f"Note: rotation_labels correspond to the rotation of 'different_digit_same_rotation'")

    return z_number, z_filter, digit_labels, rotation_labels


def prepare_rotation_labels(rotation_angles):
    """
    Convert rotation angles to discrete class labels

    Args:
        rotation_angles: Tensor of rotation angles

    Returns:
        torch.Tensor: Discrete class labels
    """
    # Convert angles to discrete bins
    # Use ROTATION_DEGREES from config to determine the range
    rotation_range = ROTATION_DEGREES
    rotation_step = ROTATION_STEP
    angle_bins = torch.arange(-rotation_range, rotation_range + 1, rotation_step)

    # Find the closest bin for each angle
    rotation_labels = []
    for angle in rotation_angles:
        distances = torch.abs(angle_bins - angle)
        closest_bin = torch.argmin(distances)
        rotation_labels.append(closest_bin)

    return torch.tensor(rotation_labels)


def train_downstream_classifier(z_data, labels, num_classes, model_name,
                               learning_rate=0.001, num_epochs=50, batch_size=64):
    """
    Train a downstream classifier on latent representations

    Args:
        z_data: Latent representations
        labels: Target labels
        num_classes: Number of classes
        model_name: Name for the model (for logging)
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        tuple: (trained_model, train_acc, val_acc, test_acc)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Prepare data
    z_data = z_data.float()
    labels = labels.long()

    # Split data
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

    return model, train_accuracies, val_accuracies, test_acc, all_predictions, all_labels


def plot_training_curves(train_acc, val_acc, model_name, save_dir):
    """
    Plot training curves for the downstream classifier
    """
    epochs = range(1, len(train_acc) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title(f'{model_name} Training Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    filename = f"{model_name.lower().replace(' ', '_')}_training_curves.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training curves saved to: {filepath}")


def create_summary_visualization(all_results, save_dir, digit_baseline, rotation_baseline):
    """
    Create a summary visualization showing all test results

    Args:
        all_results: Dictionary containing all test results
        save_dir: Directory to save the visualization
        digit_baseline: Expected random baseline for digit classification
        rotation_baseline: Expected random baseline for rotation classification
    """
    # Extract test accuracies
    test_names = list(all_results.keys())
    test_accuracies = [all_results[name]['test_acc'] for name in test_names]

    # Create bar plot
    plt.figure(figsize=(12, 8))

    # Color coding: green for expected high, red for expected low
    colors = []
    for name in test_names:
        if 'number_on_z_number' in name or 'filter_on_z_filter' in name:
            colors.append('green')  # Expected high
        elif 'random' in name:
            colors.append('red')    # Expected low (baseline)
        else:
            colors.append('orange') # Expected low (cross-test)

    bars = plt.bar(test_names, test_accuracies, color=colors, alpha=0.7)
    plt.xlabel('Test Configuration')
    plt.ylabel('Test Accuracy')
    plt.title('Downstream Task Evaluation Results')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, acc in zip(bars, test_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

    # Add horizontal lines for reference
    # plt.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Good performance threshold')
    # plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='Poor performance threshold')
    plt.axhline(y=digit_baseline, color='red', linestyle='--', alpha=0.5, label=f'Digit random baseline ({digit_baseline:.3f})')
    plt.axhline(y=rotation_baseline, color='purple', linestyle='--', alpha=0.5, label=f'Rotation random baseline ({rotation_baseline:.3f})')

    plt.legend()
    plt.tight_layout()

    # Save plot
    filename = "downstream_evaluation_summary.png"
    filepath = os.path.join(save_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Summary visualization saved to: {filepath}")


def save_comprehensive_results(all_results, save_dir, class_names, digit_baseline, rotation_baseline):
    """
    Save comprehensive results to a text file

    Args:
        all_results: Dictionary containing all test results
        save_dir: Directory to save results
        class_names: List of class names for digits
        digit_baseline: Expected random baseline for digit classification
        rotation_baseline: Expected random baseline for rotation classification
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f"comprehensive_downstream_results_{timestamp}.txt")

    with open(results_file, 'w') as f:
        f.write("Comprehensive Downstream Task Evaluation Results\n")
        f.write("=" * 60 + "\n\n")

        # Summary table
        f.write("SUMMARY TABLE\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Test':<25} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}\n")
        f.write("-" * 60 + "\n")
        for test_name, results in all_results.items():
            f.write(f"{test_name:<25} {results['train_acc'][-1]:<12.4f} {results['val_acc'][-1]:<12.4f} {results['test_acc']:<12.4f}\n")
        f.write("\n")

        # Detailed results for each test
        for test_name, results in all_results.items():
            f.write(f"{test_name.upper()} RESULTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Train Accuracy: {results['train_acc'][-1]:.4f}\n")
            f.write(f"Validation Accuracy: {results['val_acc'][-1]:.4f}\n")
            f.write(f"Test Accuracy: {results['test_acc']:.4f}\n\n")

            # Classification report
            f.write("Classification Report:\n")
            f.write(classification_report(results['true_labels'], results['predictions']))
            f.write("\n" + "="*60 + "\n\n")

        # Disentanglement analysis
        f.write("DISENTANGLEMENT ANALYSIS\n")
        f.write("-" * 40 + "\n")
        number_on_number = all_results['number_on_z_number']['test_acc']
        filter_on_filter = all_results['filter_on_z_filter']['test_acc']
        number_on_filter = all_results['number_on_z_filter']['test_acc']
        filter_on_number = all_results['filter_on_z_number']['test_acc']
        number_on_random = all_results['number_on_z_random']['test_acc']
        filter_on_random = all_results['filter_on_z_random']['test_acc']

        f.write(f"✓ z_number captures digit identity: {number_on_number:.4f} (should be > 0.8)\n")
        f.write(f"✓ z_filter captures rotation style: {filter_on_filter:.4f} (should be > 0.8)\n")
        f.write(f"✗ z_filter does NOT capture digit identity: {number_on_filter:.4f} (should be < 0.3)\n")
        f.write(f"✗ z_number does NOT capture rotation style: {filter_on_number:.4f} (should be < 0.3)\n")
        f.write(f"✗ Random baseline for digit: {number_on_random:.4f} (should be ~{digit_baseline:.3f})\n")
        f.write(f"✗ Random baseline for rotation: {filter_on_random:.4f} (should be ~{rotation_baseline:.3f})\n\n")

        # Overall score
        disentanglement_score = (number_on_number + filter_on_filter - number_on_filter - filter_on_number) / 2
        f.write(f"Overall Disentanglement Score: {disentanglement_score:.4f} (higher is better)\n")

        if disentanglement_score > 0.5:
            f.write("🎉 EXCELLENT disentanglement achieved!\n")
        elif disentanglement_score > 0.3:
            f.write("✅ GOOD disentanglement achieved!\n")
        elif disentanglement_score > 0.1:
            f.write("⚠️  MODERATE disentanglement achieved\n")
        else:
            f.write("❌ POOR disentanglement - model needs improvement\n")

    print(f"Comprehensive results saved to: {results_file}")

    # Also save a simple CSV for easy analysis
    csv_file = os.path.join(save_dir, f"downstream_results_{timestamp}.csv")
    with open(csv_file, 'w') as f:
        f.write("Test,Train_Accuracy,Val_Accuracy,Test_Accuracy\n")
        for test_name, results in all_results.items():
            f.write(f"{test_name},{results['train_acc'][-1]:.6f},{results['val_acc'][-1]:.6f},{results['test_acc']:.6f}\n")

    print(f"CSV results saved to: {csv_file}")


def save_labels_to_csv(digit_labels, rotation_labels, save_dir, class_names):
    """
    Save digit and rotation labels to a CSV file for analysis.

    Args:
        digit_labels: Tensor of digit labels
        rotation_labels: Tensor of rotation labels
        save_dir: Directory to save the CSV file
        class_names: List of class names for digits
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels_file = os.path.join(save_dir, f"downstream_labels_{timestamp}.csv")

    # Convert tensors to numpy for easier handling
    digit_labels_np = digit_labels.numpy()
    rotation_labels_np = rotation_labels.numpy()

    with open(labels_file, 'w') as f:
        f.write("Sample_Index,Digit_Label,Digit_Class_Name,Rotation_Label\n")
        for i in range(len(digit_labels_np)):
            digit_label = int(digit_labels_np[i])
            digit_class_name = class_names[digit_label] if digit_label < len(class_names) else f"Unknown_{digit_label}"
            rotation_label = int(rotation_labels_np[i])
            f.write(f"{i},{digit_label},{digit_class_name},{rotation_label}\n")

    print(f"Labels saved to: {labels_file}")

    # Print unique labels information
    print("\n" + "="*50)
    print("LABEL ANALYSIS")
    print("="*50)

    print("\nDigit Labels:")
    unique_digits = torch.unique(digit_labels)
    print(f"Number of unique digit labels: {len(unique_digits)}")
    print(f"Unique digit labels: {unique_digits.tolist()}")
    print(f"Digit label distribution:")
    for label in unique_digits:
        count = (digit_labels == label).sum().item()
        class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
        print(f"  Label {label} ({class_name}): {count} samples")

    print("\nRotation Labels:")
    unique_rotations = torch.unique(rotation_labels)
    print(f"Number of unique rotation labels: {len(unique_rotations)}")
    print(f"Unique rotation labels: {unique_rotations.tolist()}")
    print(f"Rotation label distribution:")
    for label in unique_rotations:
        count = (rotation_labels == label).sum().item()
        # Convert label back to angle (using ROTATION_DEGREES and ROTATION_STEP from config)
        angle = -ROTATION_DEGREES + label * ROTATION_STEP
        print(f"  Label {label} (angle {angle}°): {count} samples")

    # Save summary statistics to a separate file
    summary_file = os.path.join(save_dir, f"label_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write("Downstream Task Label Summary\n")
        f.write("=" * 40 + "\n\n")

        f.write("Digit Labels:\n")
        f.write(f"Number of unique digit labels: {len(unique_digits)}\n")
        f.write(f"Unique digit labels: {unique_digits.tolist()}\n")
        f.write("Digit label distribution:\n")
        for label in unique_digits:
            count = (digit_labels == label).sum().item()
            class_name = class_names[label] if label < len(class_names) else f"Unknown_{label}"
            f.write(f"  Label {label} ({class_name}): {count} samples\n")

        f.write("\nRotation Labels:\n")
        f.write(f"Number of unique rotation labels: {len(unique_rotations)}\n")
        f.write(f"Unique rotation labels: {unique_rotations.tolist()}\n")
        f.write("Rotation label distribution:\n")
        for label in unique_rotations:
            count = (rotation_labels == label).sum().item()
            angle = -ROTATION_DEGREES + label * ROTATION_STEP
            f.write(f"  Label {label} (angle {angle}°): {count} samples\n")

    print(f"\nLabel summary saved to: {summary_file}")


def save_results(number_results, filter_results, save_dir):
    """
    Save results to a text file (legacy function for backward compatibility)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(save_dir, f"downstream_results_{timestamp}.txt")

    with open(results_file, 'w') as f:
        f.write("Downstream Task Evaluation Results\n")
        f.write("=" * 50 + "\n\n")

        f.write("Number Classifier Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Train Accuracy: {number_results['train_acc'][-1]:.4f}\n")
        f.write(f"Validation Accuracy: {number_results['val_acc'][-1]:.4f}\n")
        f.write(f"Test Accuracy: {number_results['test_acc']:.4f}\n\n")

        f.write("Filter Classifier Results:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Train Accuracy: {filter_results['train_acc'][-1]:.4f}\n")
        f.write(f"Validation Accuracy: {filter_results['val_acc'][-1]:.4f}\n")
        f.write(f"Test Accuracy: {filter_results['test_acc']:.4f}\n\n")

        f.write("Classification Reports:\n")
        f.write("-" * 30 + "\n")
        f.write("Number Classifier:\n")
        f.write(classification_report(number_results['true_labels'], number_results['predictions']))
        f.write("\nFilter Classifier:\n")
        f.write(classification_report(filter_results['true_labels'], filter_results['predictions']))

    print(f"Results saved to: {results_file}")


def main():
    """
    Main function to run downstream task evaluation
    """
    print("Downstream Task Evaluation for Double Encoder Model")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pretrained model
    print("\nLoading pretrained model...")
    model = DoubleEncoderDecoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Specify the path to your pretrained model
    # pretrained_path = "../models/double_encoder_model_20250729_134537/double_encoder_epoch_40.pth"
    # pretrained_path="../models/double_encoder_model_mnist_20250801_170446/double_encoder_epoch_20.pth"
    pretrained_path="../models/double_encoder_model_mnist_20250801_171627/double_encoder_epoch_40.pth"

    if os.path.exists(pretrained_path):
        start_epoch, _ = load_model(model, optimizer, pretrained_path)
        print(f"Loaded pretrained model from epoch {start_epoch}")
    else:
        print(f"Error: Pretrained model not found at {pretrained_path}")
        return

    # Create output directory with epoch information
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"../figures/downstream/downstream_evaluation_{DATASET_TYPE}_{start_epoch}epochs_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Create triplet creator
    print("\nInitializing triplet creator...")
    triplet_creator = TripletCreator(dataset_type=DATASET_TYPE)

    # Extract latent representations
    print("\nExtracting latent representations...")
    z_number, z_filter, digit_labels, rotation_angles = extract_latent_representations(
        model, triplet_creator, num_samples=10000, dataset='test'
    )

    # Prepare rotation labels (convert to discrete classes)
    rotation_labels = prepare_rotation_labels(rotation_angles)

    # Create proper baselines based on number of unique classes
    print("\nCreating proper baselines...")
    num_digit_classes = len(triplet_creator.class_names)
    num_rotation_classes = len(torch.unique(rotation_labels))

    # Baseline 1: Random chance for digit classification (1/num_digit_classes)
    digit_baseline = 1.0 / num_digit_classes

    # Baseline 2: Random chance for rotation classification (1/num_rotation_classes)
    rotation_baseline = 1.0 / num_rotation_classes

    print(f"Digit baseline (random chance): {digit_baseline:.4f} (1/{num_digit_classes})")
    print(f"Rotation baseline (random chance): {rotation_baseline:.4f} (1/{num_rotation_classes})")

    # Create random data for baseline tests (keeping the same shape for compatibility)
    z_random = torch.randn_like(z_number)  # Same shape as z_number

    print(f"\nData summary:")
    print(f"Number of samples: {z_number.shape[0]}")
    print(f"Number of digit classes: {len(triplet_creator.class_names)}")
    print(f"Number of rotation classes: {len(torch.unique(rotation_labels))}")
    print(f"Digit labels distribution: {torch.bincount(digit_labels)}")
    print(f"Rotation labels distribution: {torch.bincount(rotation_labels)}")
    print(f"z_number shape: {z_number.shape}")
    print(f"z_filter shape: {z_filter.shape}")
    print(f"z_random shape: {z_random.shape}")

    # Save all labels to CSV for analysis
    print("\nSaving labels to CSV for analysis...")
    save_labels_to_csv(digit_labels, rotation_labels, output_dir, triplet_creator.class_names)

    # Dictionary to store all results
    all_results = {}

    # Test 1: Train number classifier on z_number
    print("\n" + "="*50)
    print("TEST 1: Number classifier on z_number")
    print("="*50)
    print("Testing if z_number (from 'same_digit_different_rotation') captures digit identity")
    model_1, train_acc_1, val_acc_1, test_acc_1, pred_1, true_1 = train_downstream_classifier(
        z_number, digit_labels, num_classes=len(triplet_creator.class_names),
        model_name="Number_on_z_number", learning_rate=0.001, num_epochs=50
    )
    all_results['number_on_z_number'] = {
        'train_acc': train_acc_1, 'val_acc': val_acc_1, 'test_acc': test_acc_1,
        'predictions': pred_1, 'true_labels': true_1
    }

    # Test 2: Train filter classifier on z_number
    print("\n" + "="*50)
    print("TEST 2: Filter classifier on z_number")
    print("="*50)
    print("Testing if z_number (from 'same_digit_different_rotation') captures rotation style")
    model_2, train_acc_2, val_acc_2, test_acc_2, pred_2, true_2 = train_downstream_classifier(
        z_number, rotation_labels, num_classes=len(torch.unique(rotation_labels)),
        model_name="Filter_on_z_number", learning_rate=0.001, num_epochs=50
    )
    all_results['filter_on_z_number'] = {
        'train_acc': train_acc_2, 'val_acc': val_acc_2, 'test_acc': test_acc_2,
        'predictions': pred_2, 'true_labels': true_2
    }

    # Test 3: Train number classifier on z_filter
    print("\n" + "="*50)
    print("TEST 3: Number classifier on z_filter")
    print("="*50)
    print("Testing if z_filter (from 'different_digit_same_rotation') captures digit identity")
    model_3, train_acc_3, val_acc_3, test_acc_3, pred_3, true_3 = train_downstream_classifier(
        z_filter, digit_labels, num_classes=len(triplet_creator.class_names),
        model_name="Number_on_z_filter", learning_rate=0.001, num_epochs=50
    )
    all_results['number_on_z_filter'] = {
        'train_acc': train_acc_3, 'val_acc': val_acc_3, 'test_acc': test_acc_3,
        'predictions': pred_3, 'true_labels': true_3
    }

    # Test 4: Train filter classifier on z_filter
    print("\n" + "="*50)
    print("TEST 4: Filter classifier on z_filter")
    print("="*50)
    print("Testing if z_filter (from 'different_digit_same_rotation') captures rotation style")
    model_4, train_acc_4, val_acc_4, test_acc_4, pred_4, true_4 = train_downstream_classifier(
        z_filter, rotation_labels, num_classes=len(torch.unique(rotation_labels)),
        model_name="Filter_on_z_filter", learning_rate=0.001, num_epochs=50
    )
    all_results['filter_on_z_filter'] = {
        'train_acc': train_acc_4, 'val_acc': val_acc_4, 'test_acc': test_acc_4,
        'predictions': pred_4, 'true_labels': true_4
    }

    # Test 5: Train number classifier on z_random (baseline)
    print("\n" + "="*50)
    print("TEST 5: Number classifier on z_random (BASELINE)")
    print("="*50)
    model_5, train_acc_5, val_acc_5, test_acc_5, pred_5, true_5 = train_downstream_classifier(
        z_random, digit_labels, num_classes=len(triplet_creator.class_names),
        model_name="Number_on_z_random", learning_rate=0.001, num_epochs=50
    )
    all_results['number_on_z_random'] = {
        'train_acc': train_acc_5, 'val_acc': val_acc_5, 'test_acc': test_acc_5,
        'predictions': pred_5, 'true_labels': true_5
    }

    # Test 6: Train filter classifier on z_random (baseline)
    print("\n" + "="*50)
    print("TEST 6: Filter classifier on z_random (BASELINE)")
    print("="*50)
    model_6, train_acc_6, val_acc_6, test_acc_6, pred_6, true_6 = train_downstream_classifier(
        z_random, rotation_labels, num_classes=len(torch.unique(rotation_labels)),
        model_name="Filter_on_z_random", learning_rate=0.001, num_epochs=50
    )
    all_results['filter_on_z_random'] = {
        'train_acc': train_acc_6, 'val_acc': val_acc_6, 'test_acc': test_acc_6,
        'predictions': pred_6, 'true_labels': true_6
    }

    # Plot training curves for all tests
    print("\nCreating visualizations...")
    for test_name, results in all_results.items():
        plot_training_curves(results['train_acc'], results['val_acc'], test_name.replace('_', ' ').title(), output_dir)

    # Create summary visualization
    create_summary_visualization(all_results, output_dir, digit_baseline, rotation_baseline)

    # Save comprehensive results
    save_comprehensive_results(all_results, output_dir, triplet_creator.class_names, digit_baseline, rotation_baseline)

    # Print summary
    print("\n" + "="*80)
    print("COMPREHENSIVE DOWNSTREAM TASK EVALUATION SUMMARY")
    print("="*80)
    print(f"{'Test':<25} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12}")
    print("-" * 80)
    for test_name, results in all_results.items():
        print(f"{test_name:<25} {results['train_acc'][-1]:<12.4f} {results['val_acc'][-1]:<12.4f} {results['test_acc']:<12.4f}")

    print(f"\nResults saved to: {output_dir}")

    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("Expected good disentanglement pattern:")
    print("✓ number_on_z_number: HIGH (z_number from 'same_digit_different_rotation' captures digit identity)")
    print("✓ filter_on_z_filter: HIGH (z_filter from 'different_digit_same_rotation' captures rotation style)")
    print("✗ number_on_z_filter: LOW (z_filter from 'different_digit_same_rotation' should NOT capture digit identity)")
    print("✗ filter_on_z_number: LOW (z_number from 'same_digit_different_rotation' should NOT capture rotation style)")
    print(f"✗ number_on_z_random: LOW (random baseline)")
    print(f"✗ filter_on_z_random: LOW (random baseline)")

    # Check if disentanglement is achieved
    number_on_number = all_results['number_on_z_number']['test_acc']
    filter_on_filter = all_results['filter_on_z_filter']['test_acc']
    number_on_filter = all_results['number_on_z_filter']['test_acc']
    filter_on_number = all_results['filter_on_z_number']['test_acc']
    number_on_random = all_results['number_on_z_random']['test_acc']
    filter_on_random = all_results['filter_on_z_random']['test_acc']

    print(f"\nDisentanglement Analysis:")
    print(f"✓ z_number captures digit identity: {number_on_number:.4f} (should be > 0.8)")
    print(f"✓ z_filter captures rotation style: {filter_on_filter:.4f} (should be > 0.8)")
    print(f"✗ z_filter does NOT capture digit identity: {number_on_filter:.4f} (should be < 0.3)")
    print(f"✗ z_number does NOT capture rotation style: {filter_on_number:.4f} (should be < 0.3)")
    print(f"✗ Random baseline for digit: {number_on_random:.4f} (should be ~{digit_baseline:.3f})")
    print(f"✗ Random baseline for rotation: {filter_on_random:.4f} (should be ~{rotation_baseline:.3f})")

    # Overall assessment
    disentanglement_score = (number_on_number + filter_on_filter - number_on_filter - filter_on_number) / 2
    print(f"\nOverall Disentanglement Score: {disentanglement_score:.4f} (higher is better)")

    if disentanglement_score > 0.5:
        print("🎉 EXCELLENT disentanglement achieved!")
    elif disentanglement_score > 0.3:
        print("✅ GOOD disentanglement achieved!")
    elif disentanglement_score > 0.1:
        print("⚠️  MODERATE disentanglement achieved")
    else:
        print("❌ POOR disentanglement - model needs improvement")


if __name__ == "__main__":
    main()
