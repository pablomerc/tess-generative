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
    def __init__(self, data_dir=DATA_DIR, dataset_type=DATASET_TYPE):
        """
        Initialize the triplet creator with MNIST or Fashion MNIST dataset

        Args:
            data_dir: Directory to store/load the dataset
            dataset_type: Type of dataset ('mnist' or 'fashion_mnist')
        """
        self.data_dir = data_dir
        self.device = device
        self.dataset_type = dataset_type

        # Load original dataset (no augmentation)
        if dataset_type == 'fashion_mnist':
            self.train_dataset = datasets.FashionMNIST(
                root=data_dir,
                train=True,
                transform=None,  # We'll apply transforms manually
                download=True
            )

            self.test_dataset = datasets.FashionMNIST(
                root=data_dir,
                train=False,
                transform=None,  # We'll apply transforms manually
                download=True
            )

            # Fashion MNIST class names
            self.class_names = [
                'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
            ]
        else:  # Default to MNIST
            self.train_dataset = datasets.MNIST(
                root=data_dir,
                train=True,
                transform=None,  # We'll apply transforms manually
                download=True
            )

            self.test_dataset = datasets.MNIST(
                root=data_dir,
                train=False,
                transform=None,  # We'll apply transforms manually
                download=True
            )

            # MNIST class names (digits 0-9)
            self.class_names = [str(i) for i in range(10)]

        # Create augmentation transforms
        self.create_augmentation_transforms()

    def create_augmentation_transforms(self):
        """
        Create different augmentation transforms for rotation and zoom
        Based on the working augment_mnist.py approach
        """
        self.augmentation_transforms = {}

        # Check if we're using scale transformations
        use_scale = SCALE_RANGE[0] != SCALE_RANGE[1] or SCALE_RANGE[0] != 1.0

        # Create transforms for different rotation angles
        # Use ROTATION_DEGREES from config to determine the range
        rotation_range = ROTATION_DEGREES
        rotation_step = ROTATION_STEP
        for angle in range(-rotation_range, rotation_range + 1, rotation_step):  # Use config values instead of hardcoded values
            if use_scale:
                # Create transforms for different scale factors
                for scale in np.arange(SCALE_RANGE[0], SCALE_RANGE[1] + 0.1, 0.1):  # Scale from 0.5 to 1.0 in steps of 0.1
                    scale = round(scale, 1)  # Round to avoid floating point issues

                    transform = transforms.Compose([
                        transforms.Resize(56, interpolation=InterpolationMode.BILINEAR),
                        transforms.RandomRotation(
                            degrees=(angle, angle),  # Fixed angle
                            interpolation=InterpolationMode.BILINEAR,
                            fill=0
                        ),
                        transforms.RandomAffine(
                            degrees=0,
                            scale=(scale, scale),  # Fixed scale
                            interpolation=InterpolationMode.BILINEAR,
                            fill=0
                        ),
                        transforms.Resize(28, interpolation=InterpolationMode.LANCZOS),
                        transforms.ToTensor()
                    ])

                    # Use tuple as key for rotation and scale
                    self.augmentation_transforms[(angle, scale)] = transform
            else:
                # No scale transformation, just rotation
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

                # Use tuple as key for rotation and scale (scale is always 1.0)
                self.augmentation_transforms[(angle, 1.0)] = transform

    def get_random_rotation_angle(self):
        """Get a random rotation angle between -ROTATION_DEGREES and +ROTATION_DEGREES"""
        angles = list(set(key[0] for key in self.augmentation_transforms.keys()))
        return random.choice(angles)

    def get_random_scale_factor(self):
        """Get a random scale factor within the configured range"""
        scales = list(set(key[1] for key in self.augmentation_transforms.keys()))
        return random.choice(scales)

    def get_random_transform_params(self):
        """Get random rotation angle and scale factor"""
        angle = self.get_random_rotation_angle()
        scale = self.get_random_scale_factor()
        return angle, scale

    def get_different_rotation_angle(self, original_angle, min_diff=MIN_ROTATION_DIFF, max_diff=MAX_ROTATION_DIFF):
        """Get a different rotation angle with minimum difference from original"""
        available_angles = list(set(key[0] for key in self.augmentation_transforms.keys()))
        valid_angles = []

        for angle in available_angles:
            diff = abs(angle - original_angle)
            if min_diff <= diff <= max_diff:
                valid_angles.append(angle)

        if not valid_angles:
            # If no valid angles, just pick any different angle
            valid_angles = [angle for angle in available_angles if angle != original_angle]

        return random.choice(valid_angles)

    def get_different_scale_factor(self, original_scale, min_diff=MIN_SCALE_DIFF):
        """Get a different scale factor with minimum difference from original"""
        available_scales = list(set(key[1] for key in self.augmentation_transforms.keys()))

        # If there's only one scale factor (e.g., always 1.0), return the same
        if len(available_scales) == 1:
            return original_scale

        valid_scales = []

        for scale in available_scales:
            diff = abs(scale - original_scale)
            if diff >= min_diff:
                valid_scales.append(scale)

        if not valid_scales:
            # If no valid scales, just pick any different scale
            valid_scales = [scale for scale in available_scales if scale != original_scale]

        return random.choice(valid_scales)

    def create_triplet(self, dataset='train'):
        """
        Create a triplet of images:
        1. ground_truth: original digit with specific augmentation (rotation + scale)
        2. different_digit: different digit with same augmentation (rotation + scale)
        3. same_digit: same digit with different augmentation (different rotation + scale)

        Returns:
            tuple: (ground_truth, different_digit, same_digit, original_digit,
                   ground_truth_label, different_digit_label, ground_truth_rotation,
                   ground_truth_scale, same_digit_rotation, same_digit_scale)
        """
        # Select dataset
        mnist_dataset = self.train_dataset if dataset == 'train' else self.test_dataset

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

        # Choose rotation angles and scale factors
        ground_truth_angle, ground_truth_scale = self.get_random_transform_params()
        different_rotation_angle = self.get_different_rotation_angle(ground_truth_angle)
        different_scale_factor = self.get_different_scale_factor(ground_truth_scale)

        # Apply augmentations
        ground_truth_transform = self.augmentation_transforms[(ground_truth_angle, ground_truth_scale)]
        different_rotation_transform = self.augmentation_transforms[(different_rotation_angle, different_scale_factor)]

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
            different_label,        # Different digit label
            ground_truth_angle,     # Ground truth rotation angle
            ground_truth_scale,     # Ground truth scale factor
            different_rotation_angle, # Same digit rotation angle
            different_scale_factor   # Same digit scale factor
        )

    def create_batch_triplets(self, batch_size=BATCH_SIZE, dataset='train'):
        """
        Create a batch of triplets

        Returns:
            tuple: (ground_truth_batch, different_digit_batch, same_digit_batch,
                   original_labels, different_labels, ground_truth_rotations,
                   ground_truth_scales, same_digit_rotations, same_digit_scales)
        """
        ground_truth_batch = []
        different_digit_batch = []
        same_digit_batch = []
        original_labels = []
        different_labels = []
        ground_truth_rotations = []
        ground_truth_scales = []
        same_digit_rotations = []
        same_digit_scales = []

        for _ in range(batch_size):
            (ground_truth, different_digit, same_digit, _, orig_label, diff_label,
             gt_rotation, gt_scale, same_rotation, same_scale) = self.create_triplet(dataset)

            ground_truth_batch.append(ground_truth)
            different_digit_batch.append(different_digit)
            same_digit_batch.append(same_digit)
            original_labels.append(orig_label)
            different_labels.append(diff_label)
            ground_truth_rotations.append(gt_rotation)
            ground_truth_scales.append(gt_scale)
            same_digit_rotations.append(same_rotation)
            same_digit_scales.append(same_scale)

        # Stack into tensors
        ground_truth_batch = torch.stack(ground_truth_batch)
        different_digit_batch = torch.stack(different_digit_batch)
        same_digit_batch = torch.stack(same_digit_batch)
        original_labels = torch.tensor(original_labels)
        different_labels = torch.tensor(different_labels)
        ground_truth_rotations = torch.tensor(ground_truth_rotations, dtype=torch.float32)
        ground_truth_scales = torch.tensor(ground_truth_scales, dtype=torch.float32)
        same_digit_rotations = torch.tensor(same_digit_rotations, dtype=torch.float32)
        same_digit_scales = torch.tensor(same_digit_scales, dtype=torch.float32)

        return (ground_truth_batch, different_digit_batch, same_digit_batch,
                original_labels, different_labels, ground_truth_rotations,
                ground_truth_scales, same_digit_rotations, same_digit_scales)

    def get_dataset_info(self):
        """Get information about the loaded dataset"""
        print(f"Dataset type: {self.dataset_type}")
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Class names: {self.class_names}")


def test_triplet_creation():
    """Test function to visualize triplets with rotation and zoom"""
    import matplotlib.pyplot as plt

    # Test with MNIST
    print("Testing with MNIST dataset:")
    creator_mnist = TripletCreator(dataset_type='mnist')
    creator_mnist.get_dataset_info()

    # Test with Fashion MNIST
    print("\nTesting with Fashion MNIST dataset:")
    creator_fashion = TripletCreator(dataset_type='fashion_mnist')
    creator_fashion.get_dataset_info()

    # Create a few triplets and visualize them for MNIST
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('MNIST Triplets', fontsize=16)

    for i in range(5):
        (ground_truth, different_digit, same_digit, original, orig_label, diff_label,
         gt_rotation, gt_scale, same_rotation, same_scale) = creator_mnist.create_triplet()

        # Plot original
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {creator_mnist.class_names[orig_label]}')
        axes[0, i].axis('off')

        # Plot ground truth
        axes[1, i].imshow(ground_truth.squeeze(), cmap='gray')
        axes[1, i].set_title(f'Ground Truth\nTarget for reconstruction\nRotation: {gt_rotation}°, Scale: {gt_scale}')
        axes[1, i].axis('off')

        # Plot different digit (filter encoder input)
        axes[2, i].imshow(different_digit.squeeze(), cmap='gray')
        axes[2, i].set_title(f'Different Digit\nFilter encoder input\nLabel: {creator_mnist.class_names[diff_label]}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Create a few triplets and visualize them for Fashion MNIST
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Fashion MNIST Triplets', fontsize=16)

    for i in range(5):
        (ground_truth, different_digit, same_digit, original, orig_label, diff_label,
         gt_rotation, gt_scale, same_rotation, same_scale) = creator_fashion.create_triplet()

        # Plot original
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {creator_fashion.class_names[orig_label]}')
        axes[0, i].axis('off')

        # Plot ground truth
        axes[1, i].imshow(ground_truth.squeeze(), cmap='gray')
        axes[1, i].set_title(f'Ground Truth\nTarget for reconstruction\nRotation: {gt_rotation}°, Scale: {gt_scale}')
        axes[1, i].axis('off')

        # Plot different digit (filter encoder input)
        axes[2, i].imshow(different_digit.squeeze(), cmap='gray')
        axes[2, i].set_title(f'Different Digit\nFilter encoder input\nLabel: {creator_fashion.class_names[diff_label]}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Show examples with different transformations for Fashion MNIST
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Fashion MNIST with Different Transformations', fontsize=16)

    for i in range(4):
        (ground_truth, different_digit, same_digit, original, orig_label, diff_label,
         gt_rotation, gt_scale, same_rotation, same_scale) = creator_fashion.create_triplet()

        # Plot original
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {creator_fashion.class_names[orig_label]}')
        axes[0, i].axis('off')

        # Plot ground truth (same digit, rotation + scale 1)
        axes[1, i].imshow(ground_truth.squeeze(), cmap='gray')
        axes[1, i].set_title(f'Ground Truth\nSame digit, rotation + scale 1\nRotation: {gt_rotation}°, Scale: {gt_scale}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Show the triplet structure more clearly for Fashion MNIST
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('Fashion MNIST Triplet Structure', fontsize=16)

    (ground_truth, different_digit, same_digit, original, orig_label, diff_label,
     gt_rotation, gt_scale, same_rotation, same_scale) = creator_fashion.create_triplet()

    # Plot original
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Original\nLabel: {creator_fashion.class_names[orig_label]}')
    axes[0].axis('off')

    # Plot ground truth (same digit, rotation + scale 1)
    axes[1].imshow(ground_truth.squeeze(), cmap='gray')
    axes[1].set_title(f'Ground Truth\nSame digit, rotation + scale 1\nRotation: {gt_rotation}°, Scale: {gt_scale}')
    axes[1].axis('off')

    # Plot same digit with different rotation + scale
    axes[2].imshow(same_digit.squeeze(), cmap='gray')
    axes[2].set_title(f'Same digit, rotation + scale 2\nNumber encoder input\nRotation: {same_rotation}°, Scale: {same_scale}')
    axes[2].axis('off')

    # Plot different digit with same rotation + scale
    axes[3].imshow(different_digit.squeeze(), cmap='gray')
    axes[3].set_title(f'Different digit, same rotation + scale\nFilter encoder input\nLabel: {creator_fashion.class_names[diff_label]}')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    # Print some statistics about the transforms
    print(f"\nNumber of available transforms: {len(creator_fashion.augmentation_transforms)}")
    angles = list(set(key[0] for key in creator_fashion.augmentation_transforms.keys()))
    scales = list(set(key[1] for key in creator_fashion.augmentation_transforms.keys()))
    print(f"Available rotation angles: {angles}")
    print(f"Available scale factors: {scales}")


def test_scale_configurations():
    """Test function to verify scale configurations work properly"""
    import matplotlib.pyplot as plt

    print("Testing scale configurations...")

    # Test with scale enabled
    print("\n1. Testing with SCALE_RANGE = (0.5, 1.0):")
    creator_with_scale = TripletCreator(dataset_type='fashion_mnist')
    print(f"Number of transforms: {len(creator_with_scale.augmentation_transforms)}")
    angles = list(set(key[0] for key in creator_with_scale.augmentation_transforms.keys()))
    scales = list(set(key[1] for key in creator_with_scale.augmentation_transforms.keys()))
    print(f"Available rotation angles: {len(angles)} angles")
    print(f"Available scale factors: {scales}")

    # Test a few triplets
    for i in range(3):
        (gt, diff, same, orig, orig_label, diff_label, _, _, _, _) = creator_with_scale.create_triplet()
        print(f"Triplet {i+1}: Created successfully")

    # Test with scale disabled (temporarily modify config)
    print("\n2. Testing with SCALE_RANGE = (1.0, 1.0):")

    # Temporarily save original config and declare global
    global SCALE_RANGE
    original_scale_range = SCALE_RANGE

    # Create a test creator with scale disabled
    SCALE_RANGE = (1.0, 1.0)

    creator_no_scale = TripletCreator(dataset_type='fashion_mnist')
    print(f"Number of transforms: {len(creator_no_scale.augmentation_transforms)}")
    angles = list(set(key[0] for key in creator_no_scale.augmentation_transforms.keys()))
    scales = list(set(key[1] for key in creator_no_scale.augmentation_transforms.keys()))
    print(f"Available rotation angles: {len(angles)} angles")
    print(f"Available scale factors: {scales}")

    # Test a few triplets
    for i in range(3):
        (gt, diff, same, orig, orig_label, diff_label, _, _, _, _) = creator_no_scale.create_triplet()
        print(f"Triplet {i+1}: Created successfully")

    # Restore original config
    SCALE_RANGE = original_scale_range

    print("\nScale configuration tests completed successfully!")


if __name__ == "__main__":
    test_triplet_creation()
    test_scale_configurations()
