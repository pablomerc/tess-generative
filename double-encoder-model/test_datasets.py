"""
Test script to verify both MNIST and Fashion MNIST datasets work correctly
"""

import matplotlib.pyplot as plt
from config import DATASET_TYPE
from triplet_creation import TripletCreator


def test_both_datasets():
    """Test both MNIST and Fashion MNIST datasets"""

    print("Testing both datasets...")
    print("="*50)

    # Test MNIST
    print("\n1. Testing MNIST dataset:")
    mnist_creator = TripletCreator(dataset_type='mnist')
    mnist_creator.get_dataset_info()

    # Test Fashion MNIST
    print("\n2. Testing Fashion MNIST dataset:")
    fashion_creator = TripletCreator(dataset_type='fashion_mnist')
    fashion_creator.get_dataset_info()

    # Create visualizations for both datasets
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle('MNIST vs Fashion MNIST Comparison', fontsize=16)

    # MNIST examples
    for i in range(5):
        ground_truth, different_digit, same_digit, original, orig_label, diff_label = mnist_creator.create_triplet()

        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f'MNIST\nLabel: {mnist_creator.class_names[orig_label]}')
        axes[0, i].axis('off')

    # Fashion MNIST examples
    for i in range(5):
        ground_truth, different_digit, same_digit, original, orig_label, diff_label = fashion_creator.create_triplet()

        axes[1, i].imshow(original, cmap='gray')
        axes[1, i].set_title(f'Fashion MNIST\nLabel: {fashion_creator.class_names[orig_label]}')
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Test current configuration
    print(f"\n3. Current configuration uses: {DATASET_TYPE}")
    current_creator = TripletCreator(dataset_type=DATASET_TYPE)
    current_creator.get_dataset_info()

    print("\nAll tests completed successfully!")


def test_triplet_examples():
    """Show triplet examples for both MNIST and Fashion MNIST"""

    print("Testing triplet examples for both datasets...")
    print("="*60)

    # Create creators for both datasets
    mnist_creator = TripletCreator(dataset_type='mnist')
    fashion_creator = TripletCreator(dataset_type='fashion_mnist')

    # Show triplet examples for MNIST
    print("\n1. MNIST Triplet Examples:")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('MNIST Triplet Examples', fontsize=16)

    for i in range(4):
        ground_truth, different_digit, same_digit, original, orig_label, diff_label = mnist_creator.create_triplet()

        # Original image
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {mnist_creator.class_names[orig_label]}')
        axes[0, i].axis('off')

        # Ground truth (same digit, rotation + scale 1)
        axes[1, i].imshow(ground_truth.squeeze(), cmap='gray')
        axes[1, i].set_title(f'Ground Truth\nSame digit, rotation + scale 1')
        axes[1, i].axis('off')

        # Different digit with same rotation + scale
        axes[2, i].imshow(different_digit.squeeze(), cmap='gray')
        axes[2, i].set_title(f'Different digit, same rotation + scale\nLabel: {mnist_creator.class_names[diff_label]}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Show triplet examples for Fashion MNIST
    print("\n2. Fashion MNIST Triplet Examples:")
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('Fashion MNIST Triplet Examples', fontsize=16)

    for i in range(4):
        ground_truth, different_digit, same_digit, original, orig_label, diff_label = fashion_creator.create_triplet()

        # Original image
        axes[0, i].imshow(original, cmap='gray')
        axes[0, i].set_title(f'Original\nLabel: {fashion_creator.class_names[orig_label]}')
        axes[0, i].axis('off')

        # Ground truth (same digit, rotation + scale 1)
        axes[1, i].imshow(ground_truth.squeeze(), cmap='gray')
        axes[1, i].set_title(f'Ground Truth\nSame digit, rotation + scale 1')
        axes[1, i].axis('off')

        # Different digit with same rotation + scale
        axes[2, i].imshow(different_digit.squeeze(), cmap='gray')
        axes[2, i].set_title(f'Different digit, same rotation + scale\nLabel: {fashion_creator.class_names[diff_label]}')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    # Show complete triplet structure for both datasets
    print("\n3. Complete Triplet Structure Comparison:")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Complete Triplet Structure: MNIST vs Fashion MNIST', fontsize=16)

    # MNIST triplet
    gt_mnist, diff_mnist, same_mnist, orig_mnist, orig_label_mnist, diff_label_mnist = mnist_creator.create_triplet()

    # Fashion MNIST triplet
    gt_fashion, diff_fashion, same_fashion, orig_fashion, orig_label_fashion, diff_label_fashion = fashion_creator.create_triplet()

    # MNIST triplet structure
    axes[0, 0].imshow(orig_mnist, cmap='gray')
    axes[0, 0].set_title(f'MNIST Original\nLabel: {mnist_creator.class_names[orig_label_mnist]}')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_mnist.squeeze(), cmap='gray')
    axes[0, 1].set_title('MNIST Ground Truth\nTarget for reconstruction')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(same_mnist.squeeze(), cmap='gray')
    axes[0, 2].set_title('MNIST Same Digit\nNumber encoder input')
    axes[0, 2].axis('off')

    axes[0, 3].imshow(diff_mnist.squeeze(), cmap='gray')
    axes[0, 3].set_title(f'MNIST Different Digit\nFilter encoder input\nLabel: {mnist_creator.class_names[diff_label_mnist]}')
    axes[0, 3].axis('off')

    # Fashion MNIST triplet structure
    axes[1, 0].imshow(orig_fashion, cmap='gray')
    axes[1, 0].set_title(f'Fashion MNIST Original\nLabel: {fashion_creator.class_names[orig_label_fashion]}')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(gt_fashion.squeeze(), cmap='gray')
    axes[1, 1].set_title('Fashion MNIST Ground Truth\nTarget for reconstruction')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(same_fashion.squeeze(), cmap='gray')
    axes[1, 2].set_title('Fashion MNIST Same Digit\nNumber encoder input')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(diff_fashion.squeeze(), cmap='gray')
    axes[1, 3].set_title(f'Fashion MNIST Different Digit\nFilter encoder input\nLabel: {fashion_creator.class_names[diff_label_fashion]}')
    axes[1, 3].axis('off')

    plt.tight_layout()
    plt.show()

    print("Triplet examples completed successfully!")


def test_triplet_creation():
    """Test triplet creation with current dataset"""

    print(f"Testing triplet creation with {DATASET_TYPE}...")
    print("="*50)

    creator = TripletCreator(dataset_type=DATASET_TYPE)

    # Create a few triplets
    for i in range(3):
        ground_truth, different_digit, same_digit, original, orig_label, diff_label = creator.create_triplet()
        print(f"Triplet {i+1}: Original={creator.class_names[orig_label]}, Different={creator.class_names[diff_label]}")

    # Test batch creation
    print("\nTesting batch creation...")
    ground_truth_batch, different_digit_batch, same_digit_batch, original_labels, different_labels = \
        creator.create_batch_triplets(batch_size=4)

    print(f"Batch shapes: Ground truth={ground_truth_batch.shape}, Different={different_digit_batch.shape}, Same={same_digit_batch.shape}")
    print(f"Labels: Original={original_labels}, Different={different_labels}")

    # Visualize a batch
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle(f'{DATASET_TYPE.upper()} Triplet Batch', fontsize=16)

    for i in range(4):
        # Original images
        axes[0, i].imshow(ground_truth_batch[i, 0], cmap='gray')
        axes[0, i].set_title(f'Ground Truth\nLabel: {creator.class_names[original_labels[i]]}')
        axes[0, i].axis('off')

        # Different digit
        axes[1, i].imshow(different_digit_batch[i, 0], cmap='gray')
        axes[1, i].set_title(f'Different Digit\nLabel: {creator.class_names[different_labels[i]]}')
        axes[1, i].axis('off')

        # Same digit with different transformation
        axes[2, i].imshow(same_digit_batch[i, 0], cmap='gray')
        axes[2, i].set_title(f'Same Digit\nDifferent Transform')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.show()

    print("Triplet creation test completed successfully!")


if __name__ == "__main__":
    test_both_datasets()
    test_triplet_examples()
    test_triplet_creation()
