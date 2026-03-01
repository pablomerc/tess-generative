import h5py
import torch
import torch.nn.functional as F


def center_crop(image, crop_size: int = 30):
    """Center crop an image tensor."""
    _, _, height, width = image.shape
    start_x = (width - crop_size) // 2
    start_y = (height - crop_size) // 2
    return image[
        :, :, start_y : start_y + crop_size, start_x : start_x + crop_size
    ]


def zoom_legacy_image(leg_im: torch.Tensor, factor: float = 0.64) -> torch.Tensor:
    """
    Zoom in the legacy images to have the same FoV as HSC.
    This matches the function in data.py.

    Args:
        leg_im: torch.Tensor, shape (C, H, W) or (N, C, H, W)
        factor: zoom factor (default: 0.64)

    Returns:
        torch.Tensor with same shape as input, zoomed to match HSC FoV
    """
    # Handle both 3D (C, H, W) and 4D (N, C, H, W) inputs
    is_3d = len(leg_im.shape) == 3
    if is_3d:
        leg_im = leg_im.unsqueeze(0)  # Add batch dimension

    im_size = leg_im.shape[-1]
    new_size = round(factor * im_size)

    cropped_im = center_crop(leg_im, new_size)

    y = F.interpolate(
        cropped_im, size=(im_size, im_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,   # great for downsampling
    )

    if is_3d:
        y = y.squeeze(0)  # Remove batch dimension

    return y


file_path = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/data/preprocessed_hsc_legacy_laptop.h5'
# file_path = '/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5'

with h5py.File(file_path, 'r') as f:
    print("Keys in the file:", list(f.keys()))
    indices = list(f.keys())[1]
    dataset_length = len(f[indices])

    print(f"The dataset '{indices}' has {dataset_length} entries.")


    # Let's load it into RAM
    hsc_numpy = f['hsc_images'][:]
    legacy_numpy = f['legacy_images'][:]

    # 2. Convert to PyTorch Tensor
    # .from_numpy shares the memory, so it's very fast
    hsc_tensor = torch.from_numpy(hsc_numpy)
    legacy_tensor = torch.from_numpy(legacy_numpy)

    print(f"Tensor shape: {hsc_tensor.shape}")
    print(f"Tensor dtype: {hsc_tensor.dtype}")

    # Calculate mean and std per channel (across dims 0, 2, 3: batch, height, width)
    # This gives per-channel values for normalization
    mean_hsc = hsc_tensor.mean(dim=(0, 2, 3))  # Shape: (C,)
    std_hsc = hsc_tensor.std(dim=(0, 2, 3))  # Uses biased estimator (divides by n), which is standard for normalization

    mean_legacy = legacy_tensor.mean(dim=(0, 2, 3))  # Shape: (C,)
    std_legacy = legacy_tensor.std(dim=(0, 2, 3))  # Uses biased estimator (divides by n), which is standard for normalization

    # Apply zoom to legacy images (same as in data.py)
    print("\nApplying zoom to legacy images (factor=0.64)...")
    legacy_zoomed = zoom_legacy_image(legacy_tensor, factor=0.64)

    # Calculate mean and std AFTER zoom
    mean_legacy_zoom = legacy_zoomed.mean(dim=(0, 2, 3))  # Shape: (C,)
    std_legacy_zoom = legacy_zoomed.std(dim=(0, 2, 3))  # Shape: (C,)


print('\n=== BEFORE ZOOM ===')
print('HSC Images 96x96 Mean (per channel):', mean_hsc.tolist())
print('HSC Images 96x96 Std (per channel):', std_hsc.tolist())
print('Legacy Images 96x96 Mean (per channel):', mean_legacy.tolist())
print('Legacy Images 96x96 Std (per channel):', std_legacy.tolist())

print('\n=== AFTER ZOOM ===')
print('Legacy Images 96x96 (zoomed) Mean (per channel):', mean_legacy_zoom.tolist())
print('Legacy Images 96x96 (zoomed) Std (per channel):', std_legacy_zoom.tolist())

print('\n=== SUMMARY ===')
mean_change = mean_legacy_zoom - mean_legacy
std_change = std_legacy_zoom - std_legacy
print(f'Legacy mean change (per channel): {mean_change.tolist()}')
print(f'Legacy std change (per channel):  {std_change.tolist()}')
