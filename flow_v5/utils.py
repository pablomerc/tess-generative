import torch


def normalize_to_flow_range(x):
    """Convert from [0,1] to [-1,1] range for flow matching"""
    return 2.0 * x - 1.0


def to_visualization_range(x):
    """Convert from [-1,1] to [0,1] range for visualization"""
    return (x + 1.0) / 2.0


def create_batch_multi_triplets_diff_seq_length(triplet_creator, batch_size, dataset='train',
                                               max_num_filter_augs=None, max_num_number_augs=None):
    """
    Create a batch of multi-triplets with variable sequence lengths between batches.

    This function randomly selects the number of augmentations for the entire batch,
    allowing for variable sequence lengths between different batches during training.
    This is particularly useful for attention-based models that can handle variable-length sequences.

    Args:
        triplet_creator: TripletCreator instance for generating multi-triplets
        batch_size (int): Number of samples in the batch
        dataset (str): Dataset split to use ('train' or 'test')
        max_num_filter_augs (int): Maximum number of filter augmentations per sample (inclusive)
        max_num_number_augs (int): Maximum number of number augmentations per sample (inclusive)

    Returns:
        dict: Batch dictionary containing:
            - anchor: [B, C, H, W] - Anchor images
            - same_filter_augments: [B, F, C, H, W] - Filter augmentations (F is same for all samples in batch)
            - same_number_augments: [B, N, C, H, W] - Number augmentations (N is same for all samples in batch)
            - anchor_labels: [B] - Anchor labels
            - filter_labels: [B, F] - Filter labels
            - And other metadata fields

    Raises:
        ValueError: If max_num_filter_augs or max_num_number_augs is None

    Note:
        All samples in the batch will have the same number of augmentations, but different
        batches will have different numbers of augmentations. This creates variable-length
        sequences between batches that are suitable for attention-based pooling mechanisms.
    """
    if max_num_filter_augs is None or max_num_number_augs is None:
        raise ValueError("Need to pass in max_number of filter and numbers augs to use different seq lengths")

    # Randomly select number of augmentations for this batch
    # Use +1 because torch.randint is exclusive of the upper bound
    num_filter_augs = torch.randint(1, max_num_filter_augs + 1, (1,)).item()
    num_number_augs = torch.randint(1, max_num_number_augs + 1, (1,)).item()

    batch = triplet_creator.create_batch_multi_triplets(
        batch_size=batch_size,
        dataset=dataset,
        num_filter_augs=num_filter_augs,
        num_number_augs=num_number_augs
    )

    return batch


def test_variable_length_batch():
    """
    Test function to verify the variable-length batch creation works correctly.
    """
    # Import here to avoid circular imports
    from flow_v5.data import make_multi_triplet_creator

    print("Testing variable-length batch creation...")

    # Create triplet creator
    triplet_creator = make_multi_triplet_creator(dataset_type='mnist')

    # Test parameters
    batch_size = 128
    max_filter_augs = 5
    max_number_augs = 5

    # Create multiple batches to show variability
    for i in range(4):
        batch = create_batch_multi_triplets_diff_seq_length(
            triplet_creator=triplet_creator,
            batch_size=batch_size,
            dataset='train',
            max_num_filter_augs=max_filter_augs,
            max_num_number_augs=max_number_augs
        )

        print(f"Batch {i+1}:")
        print(f"  Anchor shape: {batch['anchor'].shape}")
        print(f"  Same filter augments shape: {batch['same_filter_augments'].shape}")
        print(f"  Same number augments shape: {batch['same_number_augments'].shape}")
        print(f"  Filter augs count: {batch['same_filter_augments'].shape[1]}")
        print(f"  Number augs count: {batch['same_number_augments'].shape[1]}")
        print()

    print("✓ All tests passed!")


if __name__ == "__main__":
    # Run test when script is executed directly
    test_variable_length_batch()
