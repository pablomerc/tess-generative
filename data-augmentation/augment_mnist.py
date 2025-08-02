import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import numpy as np

# Control variables
save_dataset = False
visualize_augmentations = True

# Augmentation parameters
rotation_degrees = 90
scale_range = (0.5, 1.1)  # Increased scale range for more variation
translate_range = (0, 0)

augment = transforms.Compose([
    transforms.Resize(56, interpolation=InterpolationMode.BILINEAR),
    transforms.RandomRotation(rotation_degrees, interpolation=InterpolationMode.BILINEAR, fill=0),
    transforms.Resize(28, interpolation=InterpolationMode.LANCZOS),
    transforms.RandomAffine(degrees=0, scale=scale_range), # Zoom filter
    transforms.RandomAffine(degrees=0, translate=translate_range), # Translate filter
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))  # Optional
])


# Load MNIST with this augmentation
data_dir = '../data'
mnist_train = datasets.MNIST(
    root=data_dir,
    train=True,
    transform=augment,
    download=False
)

mnist_test = datasets.MNIST(
    root=data_dir,
    train=False,
    transform=augment,
    download=False
)


if visualize_augmentations:
    # Visualize 8 examples with random augmentations
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img, lbl = mnist_train[torch.randint(len(mnist_train), (1,)).item()]
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f"Label: {lbl}")
        ax.axis('off')
    fig.suptitle('Augmented MNIST - Random Scales and Rotations')
    plt.tight_layout()
    plt.show()

# Only save if save_dataset is True
if save_dataset:
    # Number of samples to save
    n_train = len(mnist_train)
    n_test = len(mnist_test)

    # Preallocate arrays
    augmented_mnist_train_images = np.zeros((n_train, 28, 28), dtype=np.float32)
    augmented_mnist_train_labels = np.zeros((n_train,), dtype=np.int64)
    augmented_mnist_test_images = np.zeros((n_test, 28, 28), dtype=np.float32)
    augmented_mnist_test_labels = np.zeros((n_test,), dtype=np.int64)

    # Save train set
    for i in range(n_train):
        img, lbl = mnist_train[i]
        augmented_mnist_train_images[i] = img.squeeze().numpy()
        augmented_mnist_train_labels[i] = lbl

    # Save test set
    for i in range(n_test):
        img, lbl = mnist_test[i]
        augmented_mnist_test_images[i] = img.squeeze().numpy()
        augmented_mnist_test_labels[i] = lbl

    # Save to disk
    import os
    os.makedirs('../data/augmented', exist_ok=True)

    augmentation_name = 'rotation_zoom'
    np.save(f'../data/augmented/{augmentation_name}_augmented_mnist_train_images.npy', augmented_mnist_train_images)
    np.save(f'../data/augmented/{augmentation_name}_augmented_mnist_train_labels.npy', augmented_mnist_train_labels)
    np.save(f'../data/augmented/{augmentation_name}_augmented_mnist_test_images.npy', augmented_mnist_test_images)
    np.save(f'../data/augmented/{augmentation_name}_augmented_mnist_test_labels.npy', augmented_mnist_test_labels)
    print(f"Augmented MNIST saved to ../data/augmented/")

    print(f"Augmented MNIST train images shape: {augmented_mnist_train_images.shape}")
    print(f"Augmented MNIST train labels shape: {augmented_mnist_train_labels.shape}")
    print(f"Augmented MNIST test images shape: {augmented_mnist_test_images.shape}")
    print(f"Augmented MNIST test labels shape: {augmented_mnist_test_labels.shape}")

print(mnist_train[0][1])
