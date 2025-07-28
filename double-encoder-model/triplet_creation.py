"""
Triplet Creation Module for Double Encoder Model

This module creates triplets of images for training:
1. Ground truth: original digit with specific augmentation
2. Different digit: different digit with same augmentation
3. Same digit: same digit with different augmentation
"""

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import numpy as np
import random
from config import *


class TripletCreator:
    def __init__(self, data_dir=DATA_DIR):
        """
        Initialize the triplet creator with MNIST dataset
        """
        self.data_dir = data_dir
        self.device = device

        # Load original MNIST (no augmentation)
        self.mnist_train = datasets.MNIST(
            root=data_dir,
            train=True,
            transform=None,  # We'll apply transforms manually
            download=False
        )

        self.mnist_test = datasets.MNIST(
            root=data_dir,
            train=False,
            transform=None,  # We'll apply transforms manually
            download=False
        )

        # Create augmentation transforms
        self.create_augmentation_transforms()

    def create_augmentation_transforms(self):
        """
        Create different augmentation transforms for rotation
        Based on the working augment_mnist.py approach
        """
        self.augmentation_transforms = {}

        # Create transforms for different rotation angles
        for angle in range(-90, 91, 15):  # -90 to 90 degrees in steps of 15
            transform = transforms.Compose([
                transforms.Resize(56, interpolation=InterpolationMode.BILINEAR),
                transforms.RandomRotation(
                    degrees=(angle, angle),  # Fixed angle
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0
                ),
                transforms.Resize(28, interpolation=InterpolationMode.LANCZOS),
                transforms.ToTensor()
            ])
            self.augmentation_transforms[angle] = transform

    def get_random_rotation_angle(self):
        """Get a random rotation angle between -90 and 90 degrees"""
        return random.choice(list(self.augmentation_transforms.keys()))

    def get_different_rotation_angle(self, original_angle, min_diff=MIN_ROTATION_DIFF, max_diff=MAX_ROTATION_DIFF):
        """Get a different rotation angle with minimum difference from original"""
        available_angles = list(self.augmentation_transforms.keys())
        valid_angles = []

        for angle in available_angles:
            diff = abs(angle - original_angle)
            if min_diff <= diff <= max_diff:
                valid_angles.append(angle)

        if not valid_angles:
            # If no valid angles, just pick any different angle
            valid_angles = [angle for angle in available_angles if angle != original_angle]

        return random.choice(valid_angles)

    def create_triplet(self, dataset='train'):
        """
        Create a triplet of images:
        1. ground_truth: original digit with specific augmentation
        2. different_digit: different digit with same augmentation
        3. same_digit: same digit with different augmentation

        Returns:
            tuple: (ground_truth, different_digit, same_digit, original_digit, ground_truth_label, different_digit_label)
        """
        # Select dataset
        mnist_dataset = self.mnist_train if dataset == 'train' else self.mnist_test

        # Sample original digit
        original_idx = random.randint(0, len(mnist_dataset) - 1)
        original_image, original_label = mnist_dataset[original_idx]

        # Sample different digit (different label)
        different_idx = random.randint(0, len(mnist_dataset) - 1)
        different_image, different_label = mnist_dataset[different_idx]

        # Keep sampling until we get a different label
        while different_label == original_label:
            different_idx = random.randint(0, len(mnist_dataset) - 1)
            different_image, different_label = mnist_dataset[different_idx]

        # Choose rotation angles
        ground_truth_angle = self.get_random_rotation_angle()
        different_rotation_angle = self.get_different_rotation_angle(ground_truth_angle)

        # Apply augmentations
        ground_truth_transform = self.augmentation_transforms[ground_truth_angle]
        different_rotation_transform = self.augmentation_transforms[different_rotation_angle]

        # Create the three images
        ground_truth = ground_truth_transform(original_image)
        different_digit = ground_truth_transform(different_image)
        same_digit = different_rotation_transform(original_image)

        return (
            ground_truth,           # Target for reconstruction
            different_digit,        # Input for filter encoder
            same_digit,             # Input for number encoder
            original_image,         # Original image (for reference)
            original_label,         # Original label
            different_label         # Different digit label
        )

    def create_batch_triplets(self, batch_size=BATCH_SIZE, dataset='train'):
        """
        Create a batch of triplets

        Returns:
            tuple: (ground_truth_batch, different_digit_batch, same_digit_batch,
                   original_labels, different_labels)
        """
        ground_truth_batch = []
        different_digit_batch = []
        same_digit_batch = []
        original_labels = []
        different_labels = []

        for _ in range(batch_size):
            ground_truth, different_digit, same_digit, _, orig_label, diff_label = self.create_triplet(dataset)

            ground_truth_batch.append(ground_truth)
            different_digit_batch.append(different_digit)
            same_digit_batch.append(same_digit)
            original_labels.append(orig_label)
            different_labels.append(diff_label)

        # Stack into tensors
        ground_truth_batch = torch.stack(ground_truth_batch)
        different_digit_batch = torch.stack(different_digit_batch)
        same_digit_batch = torch.stack(same_digit_batch)
        original_labels = torch.tensor(original_labels)
        different_labels = torch.tensor(different_labels)

        return ground_truth_batch, different_digit_batch, same_digit_batch, original_labels, different_labels


def test_triplet_creation():
    """Test function to visualize triplets"""
    import matplotlib.pyplot as plt

    creator = TripletCreator()

    # Create a few triplets and visualize them
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))

    for i in range(5):
        ground_truth, different_digit, same_digit, original, orig_label, diff_label = creator.create_triplet()

        # Plot original
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {orig_label}')
        axes[0, i].axis('off')

        # Plot ground truth
        axes[1, i].imshow(ground_truth.squeeze(), cmap='gray')
        axes[1, i].set_title(f'Ground Truth\nTarget for reconstruction')
        axes[1, i].axis('off')

        # Plot different digit (filter encoder input)
        axes[2, i].imshow(different_digit.squeeze(), cmap='gray')
        axes[2, i].set_title(f'Different Digit\nFilter encoder input\nLabel: {diff_label}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Also show same digit with different rotation
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    ground_truth, different_digit, same_digit, original, orig_label, diff_label = creator.create_triplet()

    # Plot original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original\nLabel: {orig_label}')
    axes[0].axis('off')

    # Plot ground truth (same digit, rotation 1)
    axes[1].imshow(ground_truth.squeeze(), cmap='gray')
    axes[1].set_title(f'Ground Truth\nSame digit, rotation 1')
    axes[1].axis('off')

    # Plot same digit with different rotation
    axes[2].imshow(same_digit.squeeze(), cmap='gray')
    axes[2].set_title(f'Same digit, rotation 2\nNumber encoder input')
    axes[2].axis('off')

    # Plot different digit with same rotation
    axes[3].imshow(different_digit.squeeze(), cmap='gray')
    axes[3].set_title(f'Different digit, same rotation\nFilter encoder input\nLabel: {diff_label}')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_triplet_creation()
