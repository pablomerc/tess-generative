import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta

# augmentation_name = 'rotation_v2'
augmentation_name = 'rotation_zoom'

# Set device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(f"Using device: {device}")

# ====== Data Loading ======
print("Loading augmented MNIST dataset...")

# Load augmented data (same as VAE script)
train_images = np.load(f'../data/augmented/{augmentation_name}_augmented_mnist_train_images.npy')
train_labels = np.load(f'../data/augmented/{augmentation_name}_augmented_mnist_train_labels.npy')
test_images = np.load(f'../data/augmented/{augmentation_name}_augmented_mnist_test_images.npy')
test_labels = np.load(f'../data/augmented/{augmentation_name}_augmented_mnist_test_labels.npy')

# Convert to torch tensors
train_images = torch.from_numpy(train_images).unsqueeze(1).float()  # shape: (N, 1, 28, 28)
train_labels = torch.from_numpy(train_labels).long()
test_images = torch.from_numpy(test_images).unsqueeze(1).float()
test_labels = torch.from_numpy(test_labels).long()

# Create TensorDatasets
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

# Create DataLoaders
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset loaded successfully!")
print(f"Train set: {len(train_dataset)} samples")
print(f"Test set: {len(test_dataset)} samples")
print(f"Batch size: {batch_size}")

# Visualize a few examples
print("\nVisualizing a few examples from the dataset...")
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i in range(10):
    row, col = i // 5, i % 5
    img, label = train_dataset[i]
    axes[row, col].imshow(img.squeeze(), cmap='gray')
    axes[row, col].set_title(f"Label: {label}")
    axes[row, col].axis('off')
plt.tight_layout()
plt.show()

print("Data loading complete! Ready for model definition and training.")

# ====== Model Definition ======
class AugmentedMNISTClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(AugmentedMNISTClassifier, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28 -> 28x28
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 28x28 -> 28x28

        # Max pooling layers
        self.pool = nn.MaxPool2d(2, 2)  # 28x28 -> 14x14 -> 7x7

        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

    def forward(self, x):
        # Convolutional layers with batch norm, ReLU, and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 28x28 -> 14x14
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 14x14 -> 7x7
        x = F.relu(self.bn3(self.conv3(x)))  # 7x7 -> 7x7 (no pooling after last conv)

        # Flatten for fully connected layers
        x = x.view(-1, 128 * 7 * 7)

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def predict(self, x):
        """Returns predicted labels directly"""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            return torch.argmax(outputs, dim=1)

    def predict_proba(self, x):
        """Returns prediction probabilities"""
        self.eval()
        with torch.no_grad():
            outputs = self(x)
            return F.softmax(outputs, dim=1)

# Create model instance
model = AugmentedMNISTClassifier(num_classes=10).to(device)
print(f"Model created and moved to {device}")

# Print model summary
print("\nModel architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

print(f"Loss function: {criterion}")
print(f"Optimizer: {optimizer}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

# ====== Training Functions ======
def evaluate_model(model, data_loader, criterion):
    """Evaluate model on given data loader"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in data_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss / len(data_loader)
    return avg_loss, accuracy

# ====== Training Loop ======
print("\n" + "="*60)
print("Starting Training")
print("="*60)

num_epochs = 10
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Timing setup
start_time = time.time()
epoch_times = []

for epoch in range(num_epochs):
    epoch_start = time.time()

    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        # Progress update every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}')

    # Calculate epoch statistics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

        # Validation phase
    val_loss, val_accuracy = evaluate_model(model, test_loader, criterion)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Calculate timing
    epoch_time = time.time() - epoch_start
    epoch_times.append(epoch_time)
    total_time = time.time() - start_time

    # Estimate remaining time
    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    remaining_epochs = num_epochs - (epoch + 1)
    estimated_remaining = avg_epoch_time * remaining_epochs

    # Format time strings
    elapsed_str = str(timedelta(seconds=int(total_time)))
    remaining_str = str(timedelta(seconds=int(estimated_remaining)))

    # Print epoch summary
    print(f'\nEpoch [{epoch+1}/{num_epochs}] Summary:')
    print(f'  Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%')
    print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    print(f'  Time: {epoch_time:.1f}s, Elapsed: {elapsed_str}, ETA: {remaining_str}')
    print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print('-' * 60)

# Training completed
total_training_time = time.time() - start_time
print(f"\nTraining completed in {str(timedelta(seconds=int(total_training_time)))}")
print(f"Average epoch time: {sum(epoch_times)/len(epoch_times):.1f} seconds")

# Final evaluation
print("\n" + "="*60)
print("Final Model Evaluation")
print("="*60)

final_train_loss, final_train_acc = evaluate_model(model, train_loader, criterion)
final_val_loss, final_val_acc = evaluate_model(model, test_loader, criterion)

print(f"Final Training Accuracy: {final_train_acc:.2f}%")
print(f"Final Validation Accuracy: {final_val_acc:.2f}%")
print(f"Final Training Loss: {final_train_loss:.4f}")
print(f"Final Validation Loss: {final_val_loss:.4f}")

# Plot training curves
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(val_losses, label='Val Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', color='blue')
plt.plot(val_accuracies, label='Val Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print("Training and evaluation complete!")

# ====== Save the Trained Model ======
print("\n" + "="*60)
print("Saving Trained Model")
print("="*60)

import os

# Create models directory if it doesn't exist
model_dir = "../models"
os.makedirs(model_dir, exist_ok=True)

# Save the model
model_path = os.path.join(model_dir, f"{augmentation_name}_augmented_mnist_classifier.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to: {model_path}")

# Also save model info for easy loading
model_info = {
    'model_class': 'AugmentedMNISTClassifier',
    'num_classes': 10,
    'architecture': str(model),
    'total_params': total_params,
    'trainable_params': trainable_params,
    'final_train_acc': final_train_acc,
    'final_val_acc': final_val_acc,
    'training_epochs': num_epochs,
    'batch_size': batch_size,
    'learning_rate': optimizer.param_groups[0]['lr']
}

model_info_path = os.path.join(model_dir, f"{augmentation_name}_augmented_mnist_classifier_info.pth")
torch.save(model_info, model_info_path)
print(f"Model info saved to: {model_info_path}")

print("Model saving complete!")

# ====== Show Examples with Predictions ======
print("\n" + "="*60)
print("Example Predictions")
print("="*60)

# Get a few examples from test set
model.eval()
with torch.no_grad():
    # Get 2 examples
    test_examples, test_labels = next(iter(test_loader))
    examples = test_examples[:2].to(device)
    true_labels = test_labels[:2]

    # Get predictions using the new method
    predicted_labels = model.predict(examples)
    probabilities = model.predict_proba(examples)
    confidence_scores = torch.max(probabilities, dim=1)[0]

# Convert to numpy for plotting
examples_np = examples.cpu().numpy()
probabilities_np = probabilities.cpu().numpy()
predicted_labels_np = predicted_labels.cpu().numpy()
true_labels_np = true_labels.numpy()
confidence_scores_np = confidence_scores.cpu().numpy()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for i in range(2):
    # Show image
    axes[i, 0].imshow(examples_np[i, 0], cmap='gray')
    axes[i, 0].set_title(f'Example {i+1}\nTrue: {true_labels_np[i]}, Predicted: {predicted_labels_np[i]}\nConfidence: {confidence_scores_np[i]:.3f}')
    axes[i, 0].axis('off')

    # Show prediction probabilities
    axes[i, 1].bar(range(10), probabilities_np[i])
    axes[i, 1].set_title(f'Prediction Probabilities for Example {i+1}')
    axes[i, 1].set_xlabel('Digit')
    axes[i, 1].set_ylabel('Probability')
    axes[i, 1].set_xticks(range(10))
    axes[i, 1].set_ylim(0, 1)

    # Highlight the predicted class
    axes[i, 1].bar(predicted_labels_np[i], probabilities_np[i, predicted_labels_np[i]],
                   color='red', alpha=0.7, label=f'Predicted: {predicted_labels_np[i]}')
    axes[i, 1].bar(true_labels_np[i], probabilities_np[i, true_labels_np[i]],
                   color='green', alpha=0.7, label=f'True: {true_labels_np[i]}')
    axes[i, 1].legend()

plt.tight_layout()
plt.show()

# Print detailed results
print("\nDetailed Results:")
for i in range(2):
    print(f"\nExample {i+1}:")
    print(f"  True Label: {true_labels_np[i]}")
    print(f"  Predicted Label: {predicted_labels_np[i]}")
    print(f"  Confidence: {confidence_scores_np[i]:.3f}")
    print(f"  Correct: {'✓' if predicted_labels_np[i] == true_labels_np[i] else '✗'}")
    print(f"  Top 3 predictions:")
    top_3_indices = np.argsort(probabilities_np[i])[-3:][::-1]
    for j, idx in enumerate(top_3_indices):
        print(f"    {j+1}. Digit {idx}: {probabilities_np[i, idx]:.3f}")

print(f"\nOverall: {sum(predicted_labels_np == true_labels_np)}/2 correct predictions")
