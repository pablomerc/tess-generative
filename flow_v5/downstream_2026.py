"""
Downstream Task Evaluation for Flow V5 Model

This script evaluates the quality of learned representations by:
1. Loading a pretrained flow_v5 model
2. Extracting latent representations (z_number, z_filter) from data
3. Training MLPs to predict labels from latents (classification or regression)
4. Evaluating classification accuracy or R2 scores

In this case, instead of using augmentations to predict the target image/ground truth
we will take 1 example, run it through both encoders and try to predict that same example
to see how the encoders captured different information

"""

from sklearn.metrics import accuracy_score, classification_report, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, TensorDataset


import os 
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim


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


def get_n_samples(triplet_creator, n=500):
    '''
    Use the triplet creator function to extract a set of 'n' examples with their label and rotation.
    
    Returns:
        tuple: (images, labels, rotations) where:
            images: torch.Tensor of shape [n, C, H, W]
            labels: torch.Tensor of shape [n] with digit labels
            rotations: torch.Tensor of shape [n] with rotation angles
    '''
    images_list = []
    labels_list = []
    rotations_list = []
    
    batch_size = 256
    num_batches = (n + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        current_batch_size = min(batch_size, n - len(images_list))
        if current_batch_size <= 0:
            break
            
        (ground_truth, different_digit, same_digit, original_labels, different_labels,
         ground_truth_rotations, ground_truth_scales, same_digit_rotations, same_digit_scales) = \
            triplet_creator.create_batch_triplets(current_batch_size, dataset='test')
        
        images_list.append(ground_truth)
        labels_list.append(original_labels)
        rotations_list.append(ground_truth_rotations)
    
    # Concatenate all batches and take exactly n samples
    images = torch.cat(images_list, dim=0)[:n]
    labels = torch.cat(labels_list, dim=0)[:n]
    rotations = torch.cat(rotations_list, dim=0)[:n]
    
    return images, labels, rotations

class LatentClassifier(nn.Module):
    """
    MLP classifier for downstream tasks
    """
    def __init__(self, input_dim, num_classes, hidden_dims=[256,128,64], dropout=0.2):
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
        
        # Final layer outputs raw logits
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    
    def forward(self, x):
        """
        Forward pass - returns raw logits

        Args:
            x: Input features [batch_size, input_dim]
        Returns: 
            Logits: [batch_size, num_classes] (no softmax applied)
        """
        return self.classifier(x)
    
def train_classifier(
    z_data: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    model_name: str,
    learning_rate: float = 1e-3,
    num_epochs: int = 50,
    batch_size: int = 64,
    device: str = None):
    """
    Train a downstream classifier on latent representations

    Returns:
    TODO: define what it returns
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Split the data
    # Convert to numpy for sklearn, then back to tensors
    z_np = z_data.detach().cpu().numpy() if isinstance(z_data, torch.Tensor) else z_data
    labels_np = labels.detach().cpu().numpy() if isinstance(labels, torch.Tensor) else labels
    
    z_train, z_temp, labels_train, labels_temp = train_test_split(
        z_np, labels_np, test_size=0.2, random_state=42, stratify=labels_np
    )
    z_val, z_test, labels_val, labels_test = train_test_split(
        z_temp, labels_temp, test_size=0.5, random_state=42, stratify=labels_temp
    )
    
    # Convert back to tensors - labels must be long dtype for CrossEntropyLoss
    z_train = torch.tensor(z_train, dtype=torch.float32)
    z_val = torch.tensor(z_val, dtype=torch.float32)
    z_test = torch.tensor(z_test, dtype=torch.float32)
    labels_train = torch.tensor(labels_train, dtype=torch.long)
    labels_val = torch.tensor(labels_val, dtype=torch.long)
    labels_test = torch.tensor(labels_test, dtype=torch.long)

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
    # Training loop
    print(f"\nTraining {model_name} classifier...")
    train_accuracies = []
    val_accuracies = []

    for epoch in range(num_epochs):
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
        
        # Validation evaluation
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
        model.train()
        
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


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Using device:{device}')

    # check_point_path = 'path_to_checkpoint'
    # check_point_path = '/Users/pablomercaderperez/Desktop/tess-generative/flow_models/mnist/double-encoder-flow-mnist-v5-20260107_125957/double_encoder_flow_model_mnist_epoch_1_20260107_131442.pth'
    check_point_path = '/Users/pablomercaderperez/Desktop/tess-generative/pdo_models/double_encoder_flow_model_mnist_epoch_250_20260108_045316.pth'
    if check_point_path == 'path_to_checkpoint':
        print('NEED TO UPDATE PATH TO CHECKPOINT')
        # return

    checkpoint = torch.load(check_point_path, map_location=device)
    model = build_model(
        device=device
    )

    # Load state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
    if missing_keys:
        print(f"  Missing keys (first 5): {list(missing_keys)[:5]}")
    if unexpected_keys:
        print(f"  Unexpected keys (first 5): {list(unexpected_keys)[:5]}")

    model.eval()

    # Data loader
    dataset_type = cfg.DATASET_TYPE

    triplet_creator = make_triplet_creator(dataset_type=dataset_type)

    print('Creating images')
    images, labels, rotations = get_n_samples(triplet_creator, n=1000)
    print(f'Images shape {images.shape}')
    print(f'Range of images', images.min(), images.max())

    # Move images to device and normalize to flow range [-1, 1]
    images = normalize_to_flow_range(images.to(device))
    
    number_z, filter_z, number_mu, number_logvar, filter_mu, filter_logvar = model.encode_only(images, images)


    # Test one, use number_z to predict label (number)
    print('Training classifier to predict number from number_z')
    result_dict_number = train_classifier(z_data=number_z, labels=labels,
        num_classes=10,
        model_name='pred-num_from_num-z',
        learning_rate=1e-3,
        num_epochs=50,
        batch_size=64,
        device=device)

    print('Training classifier to predict number from filter_z')
    result_dict_filter = train_classifier(z_data=filter_z, labels=labels,
        num_classes=10,
        model_name='pred-num_from_filter-z',
        learning_rate=1e-3,
        num_epochs=50,
        batch_size=64,
        device=device)

    print('Comparison of results')
    print(f'Number from number_z: {result_dict_number["test_acc"]}')
    print(f'Number from filter_z: {result_dict_filter["test_acc"]}')
    


    # Unique rotatioins
    unique_rotations = torch.unique(rotations)

    # Convert rotations to discrete class labels
    print('\nConverting rotations to discrete class labels...')
    # Create mapping from rotation value to class index
    unique_rotations_sorted = torch.sort(unique_rotations)[0]
    rotation_to_class = {rot.item(): idx for idx, rot in enumerate(unique_rotations_sorted)}
    rotation_labels = torch.tensor([rotation_to_class[rot.item()] for rot in rotations], dtype=torch.long)
    num_rotation_classes = len(unique_rotations)
    print(f'Rotation classes: {num_rotation_classes}')
    print(f'Rotation label mapping: {dict(zip([r.item() for r in unique_rotations_sorted], range(num_rotation_classes)))}')
    
    # Train rotation classifiers
    print('\nTraining classifier to predict rotation from number_z')
    result_dict_rotation_from_number = train_classifier(z_data=number_z, labels=rotation_labels,
        num_classes=num_rotation_classes,
        model_name='pred-rotation_from_num-z',
        learning_rate=1e-3,
        num_epochs=50,
        batch_size=64,
        device=device)
    
    print('Training classifier to predict rotation from filter_z')
    result_dict_rotation_from_filter = train_classifier(z_data=filter_z, labels=rotation_labels,
        num_classes=num_rotation_classes,
        model_name='pred-rotation_from_filter-z',
        learning_rate=1e-3,
        num_epochs=50,
        batch_size=64,
        device=device)
    
    print('\nRotation prediction comparison:')
    print(f'Rotation from number_z: {result_dict_rotation_from_number["test_acc"]:.4f}')
    print(f'Rotation from filter_z: {result_dict_rotation_from_filter["test_acc"]:.4f}')



    print('Comparison of all results')
    print(f'Number from number_z: {result_dict_number["test_acc"]}')
    print(f'Number from filter_z: {result_dict_filter["test_acc"]}')
    print(f'Rotation from number_z: {result_dict_rotation_from_number["test_acc"]}')
    print(f'Rotation from filter_z: {result_dict_rotation_from_filter["test_acc"]}')


if __name__ == '__main__':
    main()