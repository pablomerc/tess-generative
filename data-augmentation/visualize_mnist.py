import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Set the path to your MNIST data directory
data_dir = '../data'

# Define a transform (if you want to augment, you can add more here)
transform = transforms.ToTensor()

import os
print('Current working directory:', os.getcwd())

# Load the training set
mnist_train = datasets.MNIST(
    root=data_dir,
    train=True,
    transform=transform,
    download=False  # Don't re-download, just use existing files
)

# Load the test set
mnist_test = datasets.MNIST(
    root=data_dir,
    train=False,
    transform=transform,
    download=False
)

# Example: get the first image and label
img, label = mnist_train[0]
print(img.shape, label)  # torch.Size([1, 28, 28]), label is int

plt.imshow(img.squeeze(), cmap='gray')
plt.show()
