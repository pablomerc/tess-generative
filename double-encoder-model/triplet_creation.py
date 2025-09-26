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
#
    def create_multi_triplet(
            self,
            dataset: str = 'train',
            num_filter_augs: int = 2,
            num_number_augs: int = 2,
                             ):
        """ Produce one anchor plus multiple augmentations for the filter and number encoders respectively.
        Returns a dict so downstream callers can inspect metadata without
        positional juggling.

        Current version will be limited to 1<=N<=5
        TODO: Make it return a random number of augmentations for each encoder, between 1 and N
        """

        if not (1 <= num_filter_augs <= 5 and 1 <= num_number_augs <= 5):
            raise ValueError('num_filter_augs and num_number_augs mut be in [1,5]')

        # Select dataset
        mnist_dataset = self.train_dataset if dataset == 'train' else self.test_dataset

        # Sample original digit
        original_idx = random.randint(0, len(mnist_dataset) - 1)
        original_image, original_label = mnist_dataset[original_idx]



        # Choose rotation angles and scale factors
        ground_truth_angle, ground_truth_scale = self.get_random_transform_params()


        # Create the augmentations
        ground_truth_transform = self.augmentation_transforms[(ground_truth_angle, ground_truth_scale)]
        # Create the images

        anchor = ground_truth_transform(original_image)

        # Augmentations with same filter
        same_filter_augs, same_filter_params, filter_labels = [],[],[]
        for _ in range(num_filter_augs):
            # Sample different digit (different label)
            different_idx = random.randint(0, len(mnist_dataset) - 1)
            different_image, different_label = mnist_dataset[different_idx]

            # Keep sampling until we get a different label
            while different_label == original_label:
                different_idx = random.randint(0, len(mnist_dataset) - 1)
                different_image, different_label = mnist_dataset[different_idx]

            same_filter_augs.append(ground_truth_transform(different_image))
            same_filter_params.append((ground_truth_angle, ground_truth_scale))
            filter_labels.append(different_label)

        # Augmentations wtih same digit
        same_num_augs, same_num_params = [],[]
        for _ in range(num_number_augs):
            #Sample a new angle and scale
            different_rotation_angle = self.get_different_rotation_angle(ground_truth_angle)
            different_scale_factor = self.get_different_scale_factor(ground_truth_scale)
            different_rotation_transform = self.augmentation_transforms[(different_rotation_angle, different_scale_factor)]

            same_num_augs.append(different_rotation_transform(original_image))
            same_num_params.append((different_rotation_angle, different_scale_factor))


        return {
            "anchor": anchor,
            "same_filter_augments": torch.stack(same_filter_augs),
            "same_number_augments": torch.stack(same_num_augs),
            "anchor_label": original_label,
            "filter_labels": filter_labels,
            "anchor_params": (ground_truth_angle, ground_truth_scale),
            "filter_params": same_filter_params,
            "number_params": same_num_params,
        }

    def create_batch_multi_triplets(self, batch_size=BATCH_SIZE, dataset='train', num_filter_augs=2, num_number_augs=2):
    """Create a batch of multi-triplets
    Returns
    Dictionary
    """
    batch_anchors = []
    batch_same_filter_augments = []
    batch_same_number_augments = []
    batch_anchor_labels = []
    batch_same_filter_labels = []
    batch_anchor_params = []
    batch_same_filter_params = []
    batch_same_number_params = []

    for _ in range(batch_size):
        sample = self.create_multi_triplet(
            dataset=dataset,
            num_filter_augs=num_filter_augs,
            num_number_augs=num_number_augs
        )

        batch_anchors.append(sample["anchor"])
        batch_same_filter_augments.append(sample["same_filter_augments"])
        batch_same_number_augments.append(sample["same_number_augments"])
        batch_anchor_labels.append(sample["anchor_label"])
        batch_filter_labels.append(sample["filter_labels"])
        batch_anchor_params.append(sample["anchor_params"])
        batch_filter_params.append(sample["filter_params"])
        batch_number_params.append(sample["number_params"])

    return {
        "batch_anchors": torch.stack(batch_anchors), # [B, C, H, W]
        "batch_same_filter_augments": torch.stack(batch_same_filter_augments), # [B, F, C, H, W]
        "batch_same_number_augments": torch.stack(batch_same_number_augments), # [B, N, C, H, W]
        "batch_anchor_labels": torch.tensor(batch_anchor_labels), # [B]
        "batch_filter_labels": torch.tensor(batch_filter_labels), # [B, F]
        "batch_anchor_params": torch.tensor(batch_anchor_params), # [B, 2]
        "batch_filter_params": torch.tensor(batch_filter_params), # [B, N, 2]
        "batch_number_params": torch.tensor(batch_number_params), # [B, N, 2]
    }


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


def test_multi_triplet_creation():
    """Test function for create_multi_triplet with N=5 and visualization"""
    import matplotlib.pyplot as plt

    print("Testing create_multi_triplet function with N=5...")

    # Create triplet creator
    creator = TripletCreator(dataset_type='mnist')
    creator.get_dataset_info()

    # Test with N=2 for both filter and number augmentations
    num_filter_augs = 5
    num_number_augs = 5

    print(f"\nCreating multi-triplet with {num_filter_augs} filter augmentations and {num_number_augs} number augmentations...")

    # Create the multi-triplet
    result = creator.create_multi_triplet(
        dataset='train',
        num_filter_augs=num_filter_augs,
        num_number_augs=num_number_augs
    )

    # Print result structure
    print("\nResult structure:")
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: tensor with shape {value.shape}")
        else:
            print(f"  {key}: {value}")

    # Verify shapes
    print(f"\nShape verification:")
    print(f"  Anchor shape: {result['anchor'].shape}")
    print(f"  Same filter augments shape: {result['same_filter_augments'].shape}")
    print(f"  Same number augments shape: {result['same_number_augments'].shape}")
    print(f"  Number of filter labels: {len(result['filter_labels'])}")
    print(f"  Number of filter params: {len(result['filter_params'])}")
    print(f"  Number of number params: {len(result['number_params'])}")

    # Create visualization
    # Use max + 1 to accommodate the anchor in the first row
    max_cols = max(num_filter_augs, num_number_augs)
    fig, axes = plt.subplots(3, max_cols, figsize=(15, 9))
    fig.suptitle(f'Multi-Triplet Creation Test (N={num_filter_augs})', fontsize=16)

    # Plot anchor (only in first column of first row)
    axes[0, 0].imshow(result['anchor'].squeeze(), cmap='gray')
    axes[0, 0].set_title(f'Anchor\nLabel: {creator.class_names[result["anchor_label"]]}\n'
                        f'Rotation: {result["anchor_params"][0]}°, Scale: {result["anchor_params"][1]}')
    axes[0, 0].axis('off')

    # Hide unused columns in the first row (anchor row)
    for i in range(1, max_cols):
        axes[0, i].axis('off')

    # Plot same filter augmentations (different digits, same transformation)
    for i in range(num_filter_augs):
        axes[1, i].imshow(result['same_filter_augments'][i].squeeze(), cmap='gray')
        filter_label = result['filter_labels'][i]
        filter_params = result['filter_params'][i]
        axes[1, i].set_title(f'Same Filter Aug {i+1}\nLabel: {creator.class_names[filter_label]}\n'
                            f'Rotation: {filter_params[0]}°, Scale: {filter_params[1]}')
        axes[1, i].axis('off')

    # Hide unused columns in the second row (filter augmentations)
    for i in range(num_filter_augs, max_cols):
        axes[1, i].axis('off')

    # Plot same number augmentations (same digit, different transformations)
    for i in range(num_number_augs):
        axes[2, i].imshow(result['same_number_augments'][i].squeeze(), cmap='gray')
        number_params = result['number_params'][i]
        axes[2, i].set_title(f'Same Number Aug {i+1}\nLabel: {creator.class_names[result["anchor_label"]]}\n'
                            f'Rotation: {number_params[0]}°, Scale: {number_params[1]}')
        axes[2, i].axis('off')

    # Hide unused columns in the third row (number augmentations)
    for i in range(num_number_augs, max_cols):
        axes[2, i].axis('off')

    # Add row labels
    axes[0, 0].text(-0.1, 0.5, 'Anchor', transform=axes[0, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
    axes[1, 0].text(-0.1, 0.5, 'Same Filter\n(Different Digits)', transform=axes[1, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')
    axes[2, 0].text(-0.1, 0.5, 'Same Number\n(Different Transforms)', transform=axes[2, 0].transAxes,
                    rotation=90, va='center', ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Test multiple samples
    print(f"\nTesting multiple samples...")
    for i in range(3):
        result = creator.create_multi_triplet(
            dataset='train',
            num_filter_augs=num_filter_augs,
            num_number_augs=num_number_augs
        )
        print(f"Sample {i+1}: Anchor label {result['anchor_label']}, "
              f"Filter labels {result['filter_labels']}, "
              f"Anchor params {result['anchor_params']}")

    # Test edge cases
    print(f"\nTesting edge cases...")

    # Test with N=1
    result_n1 = creator.create_multi_triplet(
        dataset='train',
        num_filter_augs=1,
        num_number_augs=1
    )
    print(f"N=1: Filter augs shape {result_n1['same_filter_augments'].shape}, "
          f"Number augs shape {result_n1['same_number_augments'].shape}")

    # Test with N=5 (maximum)
    result_n5 = creator.create_multi_triplet(
        dataset='train',
        num_filter_augs=5,
        num_number_augs=5
    )
    print(f"N=5: Filter augs shape {result_n5['same_filter_augments'].shape}, "
          f"Number augs shape {result_n5['same_number_augments'].shape}")

    print("\nMulti-triplet creation test completed successfully!")


if __name__ == "__main__":
    test_triplet_creation()
    test_scale_configurations()
    test_multi_triplet_creation()
