import torch
import torch.nn as nn
import torch.nn.functional as F

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
