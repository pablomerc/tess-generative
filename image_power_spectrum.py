import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import torch
import torchvision
import torchvision.transforms as transforms
from astropy.io import fits
from astropy.wcs import WCS
import requests
import tempfile
import os

def compute_power_spectrum(image):
    # Step 1: Fourier transform and shift
    F = fft2(image)
    F_shifted = fftshift(F)

    # Step 2: Compute power spectrum
    power = np.abs(F_shifted)**2

    return power

def radial_profile(power_spectrum):
    y, x = np.indices(power_spectrum.shape)
    center = np.array([(x.max() - x.min())/2.0, (y.max() - y.min())/2.0])
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)

    # Radial average
    tbin = np.bincount(r.ravel(), power_spectrum.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def load_tess_ffi(url, crop_size=None, full_resolution=False):
    """Load a TESS FFI and optionally crop or use full resolution"""
    with fits.open(url, mode="readonly") as hdulist:
        cal_image = hdulist[1].data

        if full_resolution:
            # Use the full image (might be very large!)
            full_image = cal_image
            print(f"Loaded full TESS image: {full_image.shape}")

            # Normalize to [0, 1] range
            full_image = (full_image - np.min(full_image)) / (np.max(full_image) - np.min(full_image))
            return full_image
        else:
            # Crop a random region to get a manageable size
            h, w = cal_image.shape
            if crop_size is None:
                crop_size = 256
            start_h = np.random.randint(0, h - crop_size)
            start_w = np.random.randint(0, w - crop_size)

            cropped_image = cal_image[start_h:start_h+crop_size, start_w:start_w+crop_size]

            # Normalize to [0, 1] range
            cropped_image = (cropped_image - np.min(cropped_image)) / (np.max(cropped_image) - np.min(cropped_image))

            return cropped_image

# Load STL10 dataset (96x96 pixels - much higher resolution than CIFAR10)
transform_color = transforms.Compose([
    transforms.Resize((128, 128)),  # Optionally resize to even larger size
    transforms.ToTensor(),
])

transform_grayscale = transforms.Compose([
    transforms.Resize((128, 128)),  # Optionally resize to even larger size
    transforms.ToTensor(),
    transforms.Grayscale()  # Convert to grayscale for power spectrum analysis
])

# Load STL10 dataset (we'll load both color and grayscale versions)
stl10_dataset_color = torchvision.datasets.STL10(
    root='./data',
    split='train',  # or 'test'
    download=True,
    transform=transform_color
)

stl10_dataset_gray = torchvision.datasets.STL10(
    root='./data',
    split='train',  # or 'test'
    download=True,
    transform=transform_grayscale
)

# Get 4 random images from STL10
stl10_indices = np.random.choice(len(stl10_dataset_color), 4, replace=False)
color_images = []
gray_images = []

for idx in stl10_indices:
    # Get color image (3 channels: R, G, B)
    img_tensor_color, label = stl10_dataset_color[idx]
    color_images.append(img_tensor_color.numpy())  # Shape: (3, H, W)

    # Get grayscale image (1 channel)
    img_tensor_gray, _ = stl10_dataset_gray[idx]
    gray_images.append(img_tensor_gray.squeeze().numpy())  # Shape: (H, W)

# Choose whether to use full resolution or cropped images
USE_FULL_RESOLUTION = True  # Set to True for full 2000x2000 analysis

# Load 4 TESS FFI images
tess_urls = [
    "https://archive.stsci.edu/missions/tess/ffi/s0001/2018/206/4-2/tess2018206192942-s0001-4-2-0120-s_ffic.fits",
    "https://archive.stsci.edu/missions/tess/ffi/s0023/2020/088/2-3/tess2020088005920-s0023-2-3-0177-s_ffic.fits",
    "https://archive.stsci.edu/missions/tess/ffi/s0063/2023/074/2-4/tess2023074014518-s0063-2-4-0255-s_ffir.fits",
    "https://archive.stsci.edu/missions/tess/ffi/s0090/2025/074/4-2/tess2025074001249-s0090-4-2-0287-s_ffic.fits"
]

tess_images = []
for url in tess_urls:
    if USE_FULL_RESOLUTION:
        print(f"Loading FULL RESOLUTION image from: {url}")
        tess_img = load_tess_ffi(url, full_resolution=True)
    else:
        print(f"Loading CROPPED image from: {url}")
        tess_img = load_tess_ffi(url, crop_size=256)
    print(f"Final image shape: {tess_img.shape}")
    tess_images.append(tess_img)

# STL10 class names
class_names = ['airplane', 'bird', 'car', 'cat', 'deer',
               'dog', 'horse', 'monkey', 'ship', 'truck']

# Create subplots: 4 rows (STL10 color, STL10 grayscale, TESS images, all spectra), 4 columns
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
fig.suptitle('Natural Images vs TESS FFI Images and Their Power Spectra', fontsize=16)

# Plot STL10 images (rows 1-2)
for i, (color_img, gray_img, idx) in enumerate(zip(color_images, gray_images, stl10_indices)):
    # Get the class name for the image
    _, label = stl10_dataset_color[idx]
    class_name = class_names[label]

    # Plot color image
    axes[0, i].imshow(np.transpose(color_img, (1, 2, 0)))  # Convert from (C, H, W) to (H, W, C)
    axes[0, i].set_title(f'STL10: {class_name.capitalize()} (Color)')
    axes[0, i].axis('off')

    # Plot grayscale image
    axes[1, i].imshow(gray_img, cmap='gray')
    axes[1, i].set_title(f'STL10: {class_name.capitalize()} (Grayscale)')
    axes[1, i].axis('off')

# Plot TESS images (row 3)
for i, tess_img in enumerate(tess_images):
    print(f"TESS image {i+1} shape: {tess_img.shape}")
    axes[2, i].imshow(tess_img, cmap='viridis')
    axes[2, i].set_title(f'TESS FFI {i+1} ({tess_img.shape[0]}x{tess_img.shape[1]})')
    axes[2, i].axis('off')

# Plot power spectra (row 4)
for i in range(4):
    # STL10 grayscale power spectrum
    gray_power_spectrum = compute_power_spectrum(gray_images[i])
    gray_radial_ps = radial_profile(gray_power_spectrum)
    axes[3, i].loglog(gray_radial_ps, label='STL10 Grayscale', color='blue', linewidth=2)

    # TESS power spectrum
    tess_power_spectrum = compute_power_spectrum(tess_images[i])
    tess_radial_ps = radial_profile(tess_power_spectrum)
    axes[3, i].loglog(tess_radial_ps, label='TESS FFI', color='red', linewidth=2)

    axes[3, i].set_xlabel("Wavenumber (pixels⁻¹)")
    axes[3, i].set_ylabel("Power")
    axes[3, i].set_title(f'Power Spectra Comparison')
    axes[3, i].legend()
    axes[3, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Also create a separate comparison plot showing just the power spectra
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
for i, tess_img in enumerate(tess_images):
    tess_power_spectrum = compute_power_spectrum(tess_img)
    tess_radial_ps = radial_profile(tess_power_spectrum)
    plt.loglog(tess_radial_ps, label=f'TESS FFI {i+1}', alpha=0.7)
plt.xlabel("Wavenumber (pixels⁻¹)")
plt.ylabel("Power")
plt.title("TESS FFI Power Spectra")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
for i, (gray_img, idx) in enumerate(zip(gray_images, stl10_indices)):
    gray_power_spectrum = compute_power_spectrum(gray_img)
    gray_radial_ps = radial_profile(gray_power_spectrum)
    plt.loglog(gray_radial_ps, label=f'STL10 {class_names[idx]}', alpha=0.7)
plt.xlabel("Wavenumber (pixels⁻¹)")
plt.ylabel("Power")
plt.title("STL10 Power Spectra")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
# Average power spectra
tess_avg_ps = np.mean([radial_profile(compute_power_spectrum(tess_img)) for tess_img in tess_images], axis=0)
stl10_avg_ps = np.mean([radial_profile(compute_power_spectrum(gray_img)) for gray_img in gray_images], axis=0)
plt.loglog(tess_avg_ps, label='TESS FFI (avg)', color='red', linewidth=3)
plt.loglog(stl10_avg_ps, label='STL10 (avg)', color='blue', linewidth=3)
plt.xlabel("Wavenumber (pixels⁻¹)")
plt.ylabel("Power")
plt.title("Average Power Spectra Comparison")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
# Show one example of each type
plt.imshow(gray_images[0], cmap='gray')
plt.title("STL10 Example")
plt.axis('off')

plt.tight_layout()
plt.show()

# Sanity check: Separate detailed plot of the first TESS FFI
print("\n=== SANITY CHECK: TESS FFI 1 DETAILED ANALYSIS ===")
print(f"TESS FFI 1 shape: {tess_images[0].shape}")
print(f"TESS FFI 1 data range: {np.min(tess_images[0]):.6f} to {np.max(tess_images[0]):.6f}")
print(f"TESS FFI 1 mean: {np.mean(tess_images[0]):.6f}")
print(f"TESS FFI 1 std: {np.std(tess_images[0]):.6f}")

# Create a detailed plot of the first TESS image
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'TESS FFI 1 Detailed Analysis - Shape: {tess_images[0].shape}', fontsize=16)

# Full image
axes[0, 0].imshow(tess_images[0], cmap='viridis')
axes[0, 0].set_title('Full TESS FFI 1')
axes[0, 0].axis('off')

# Zoom into center region (if image is large enough)
if tess_images[0].shape[0] > 500 and tess_images[0].shape[1] > 500:
    center_h, center_w = tess_images[0].shape[0]//2, tess_images[0].shape[1]//2
    crop_size = 500
    start_h = max(0, center_h - crop_size//2)
    end_h = min(tess_images[0].shape[0], center_h + crop_size//2)
    start_w = max(0, center_w - crop_size//2)
    end_w = min(tess_images[0].shape[1], center_w + crop_size//2)

    cropped = tess_images[0][start_h:end_h, start_w:end_w]
    axes[0, 1].imshow(cropped, cmap='viridis')
    axes[0, 1].set_title(f'Center Region ({crop_size}x{crop_size})')
    axes[0, 1].axis('off')
else:
    axes[0, 1].imshow(tess_images[0], cmap='viridis')
    axes[0, 1].set_title('Full Image (small)')
    axes[0, 1].axis('off')

# Histogram of pixel values
axes[0, 2].hist(tess_images[0].flatten(), bins=100, alpha=0.7, color='blue')
axes[0, 2].set_title('Pixel Value Distribution')
axes[0, 2].set_xlabel('Pixel Value')
axes[0, 2].set_ylabel('Frequency')
axes[0, 2].grid(True, alpha=0.3)

# Power spectrum
tess_power_spectrum = compute_power_spectrum(tess_images[0])
tess_radial_ps = radial_profile(tess_power_spectrum)

axes[1, 0].loglog(tess_radial_ps, 'r-', linewidth=2)
axes[1, 0].set_xlabel("Wavenumber (pixels⁻¹)")
axes[1, 0].set_ylabel("Power")
axes[1, 0].set_title("Power Spectrum (Log-Log)")
axes[1, 0].grid(True, alpha=0.3)

# Power spectrum (linear scale)
axes[1, 1].plot(tess_radial_ps, 'r-', linewidth=2)
axes[1, 1].set_xlabel("Wavenumber (pixels⁻¹)")
axes[1, 1].set_ylabel("Power")
axes[1, 1].set_title("Power Spectrum (Linear)")
axes[1, 1].grid(True, alpha=0.3)

# 2D power spectrum (log scale)
log_power = np.log10(tess_power_spectrum + 1e-10)  # Add small constant to avoid log(0)
im = axes[1, 2].imshow(log_power, cmap='viridis')
axes[1, 2].set_title("2D Power Spectrum (Log Scale)")
axes[1, 2].axis('off')
plt.colorbar(im, ax=axes[1, 2], shrink=0.8)

plt.tight_layout()
plt.show()

print("=== END SANITY CHECK ===\n")
