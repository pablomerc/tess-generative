"""
Load a pretrained model, encode 4048 images, compute UMAP, and create
an interactive Holoviews visualization with image thumbnails saved to HTML.
Designed to work on remote machines - saves HTML file that can be opened locally.
"""

import torch
from double_train_fm import ConditionalFlowMatchingModule
from torch.utils.data import DataLoader
from data import HSCLegacyDataset
import time
from pathlib import Path
import numpy as np
import umap
import base64
from io import BytesIO
from PIL import Image
import pandas as pd
from bokeh.models import HoverTool

# Holoviews for interactive visualization
import holoviews as hv
hv.extension("bokeh")

# ===== Configuration =====
# checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/33mo9r3n/checkpoints/epoch=201-step=75000.ckpt'  # z_dim = 512
checkpoint_path = '/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/galaxy-flow-matching/s39qr0v8/checkpoints/epoch=201-step=75000.ckpt' # model with latent space of 128

dim = 128  # Dimension setting (used in output file names)

# Number of images to load (4048 total = 2024 pairs)
num_images = 4048
start_idx = 95_000  # Starting index in the dataset
end_idx = start_idx + num_images

# Output directory
figures_dir = Path('/data/vision/billf/scratch/pablomer/projects/tess-generative/galaxy_images/galaxy_model/figures')
figures_dir.mkdir(parents=True, exist_ok=True)

# ===== Helper Functions =====
def _row_scale_rgb(x_chw: torch.Tensor, vmin, vmax) -> np.ndarray:
    """
    Scale a (C,H,W) tensor to (H,W,3) in [0,1] using fixed per-channel vmin/vmax.
    vmin/vmax: tensor-like shape (C,) or (3,))
    Returns numpy array in (H,W,3) format.
    """
    x = x_chw[:3]  # Take first 3 channels
    vmin_t = torch.as_tensor(vmin, device=x.device, dtype=x.dtype).view(3, 1, 1)
    vmax_t = torch.as_tensor(vmax, device=x.device, dtype=x.dtype).view(3, 1, 1)
    y = (x - vmin_t) / (vmax_t - vmin_t + 1e-8)
    y = y.clamp(0, 1)
    y = y.permute(1, 2, 0)  # (3,H,W) -> (H,W,3)
    return y.detach().cpu().numpy()

def tensor_to_base64(tensor: torch.Tensor, vmin=None, vmax=None) -> str:
    """
    Converts a Torch tensor (C, H, W) to a base64 encoded PNG string for HTML embedding.
    Uses per-channel scaling similar to load_pretrained_model.py
    """
    # Compute per-channel vmin/vmax if not provided
    if vmin is None or vmax is None:
        tensor_chw = tensor[:3]  # (3,H,W)
        vmin = tensor_chw.amin(dim=(1, 2)).cpu().numpy()  # (3,)
        vmax = tensor_chw.amax(dim=(1, 2)).cpu().numpy()  # (3,)

    # Scale to RGB
    img_rgb = _row_scale_rgb(tensor, vmin, vmax)  # (H,W,3) in [0,1]

    # Convert to uint8
    img_uint8 = (img_rgb * 255).astype(np.uint8)

    # Convert to PIL Image and encode
    pil_img = Image.fromarray(img_uint8, mode='RGB')
    buff = BytesIO()
    pil_img.save(buff, format="PNG")
    b64_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64_str}"

# ===== Load Model =====
print("="*60)
print("Loading pretrained model...")
print("="*60)
model = ConditionalFlowMatchingModule.load_from_checkpoint(checkpoint_path)

# Set the model to evaluation mode and disable gradient calculation for inference
model.eval()
torch.set_grad_enabled(False)

# Move model to appropriate device (GPU if available, else CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(f"Model loaded on device: {device}")

# ===== Load Dataset =====
print("\n" + "="*60)
print(f"Loading {num_images} images (indices {start_idx} to {end_idx-1})...")
print("="*60)

dataset_start = time.perf_counter()
dataset = HSCLegacyDataset(
    hdf5_path='/data/vision/billf/scratch/pablomer/legacysurvey_hsc/preprocessed_hsc_legacy_48x48_all.h5',
    idx_list=list(range(start_idx, end_idx)),
)
dataset_time = time.perf_counter() - dataset_start
print(f"Dataset loaded in {dataset_time:.4f} s")

# Create DataLoader
train_loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=4)

# Load all batches
batch_start = time.perf_counter()
all_hsc_images = []
all_legacy_images = []

for batch in train_loader:
    hsc_batch, legacy_batch = batch
    all_hsc_images.append(hsc_batch)
    all_legacy_images.append(legacy_batch)

# Concatenate all batches
hsc_images = torch.cat(all_hsc_images, dim=0).to(device)
legacy_images = torch.cat(all_legacy_images, dim=0).to(device)
batch_time = time.perf_counter() - batch_start

print(f"All images loaded in {batch_time:.4f} s")
print(f"  HSC images shape: {hsc_images.shape}")
print(f"  Legacy images shape: {legacy_images.shape}")

# ===== Encode Images =====
print("\n" + "="*60)
print("Encoding images with both encoders...")
print("="*60)

encode_start = time.perf_counter()
with torch.no_grad():
    hsc_embeddings_1 = model.encoder_1(hsc_images)
    legacy_embeddings_1 = model.encoder_1(legacy_images)
    hsc_embeddings_2 = model.encoder_2(hsc_images)
    legacy_embeddings_2 = model.encoder_2(legacy_images)
encode_time = time.perf_counter() - encode_start

print(f"Encoding completed in {encode_time:.4f} s")
print(f"  HSC embeddings 1 shape: {hsc_embeddings_1.shape}")
print(f"  HSC embeddings 2 shape: {hsc_embeddings_2.shape}")
print(f"  Legacy embeddings 1 shape: {legacy_embeddings_1.shape}")
print(f"  Legacy embeddings 2 shape: {legacy_embeddings_2.shape}")

# Prepare embeddings for UMAP (flatten spatial dimensions)
all_embeddings_1 = torch.concat([hsc_embeddings_1, legacy_embeddings_1], dim=0)
all_embeddings_1 = all_embeddings_1.flatten(start_dim=1)

all_embeddings_2 = torch.concat([hsc_embeddings_2, legacy_embeddings_2], dim=0)
all_embeddings_2 = all_embeddings_2.flatten(start_dim=1)

print(f"\nFlattened embeddings:")
print(f"  Encoder 1 shape: {all_embeddings_1.shape}")
print(f"  Encoder 2 shape: {all_embeddings_2.shape}")

num_hsc = hsc_embeddings_1.shape[0]

# ===== Compute UMAP =====
print("\n" + "="*60)
print("Computing UMAP embeddings...")
print("="*60)

umap_params = {
    'n_neighbors': 15,
    'min_dist': 0.1,
    'n_components': 2,
    'metric': 'euclidean',
    'random_state': 42,
}

# Encoder 1 UMAP
print("Computing UMAP for Encoder 1 (Physics Latent Space)...")
umap_start = time.perf_counter()
reducer_1 = umap.UMAP(**umap_params)
embedding_1 = reducer_1.fit_transform(all_embeddings_1.cpu().numpy())
umap_time_1 = time.perf_counter() - umap_start
print(f"  Completed in {umap_time_1:.4f} s")

hsc_embedding_1 = embedding_1[:num_hsc]
legacy_embedding_1 = embedding_1[num_hsc:]

# Encoder 2 UMAP
print("Computing UMAP for Encoder 2 (Instrument Latent Space)...")
umap_start = time.perf_counter()
reducer_2 = umap.UMAP(**umap_params)
embedding_2 = reducer_2.fit_transform(all_embeddings_2.cpu().numpy())
umap_time_2 = time.perf_counter() - umap_start
print(f"  Completed in {umap_time_2:.4f} s")

hsc_embedding_2 = embedding_2[:num_hsc]
legacy_embedding_2 = embedding_2[num_hsc:]

# ===== Prepare Images for Display =====
print("\n" + "="*60)
print("Converting images to HTML-ready Base64 strings...")
print("="*60)

print("Processing HSC images...")
hsc_b64 = [tensor_to_base64(img) for img in hsc_images]

print("Processing Legacy images...")
legacy_b64 = [tensor_to_base64(img) for img in legacy_images]

# Combine all data
all_images_b64 = hsc_b64 + legacy_b64
labels = ['HSC'] * num_hsc + ['Legacy'] * num_hsc

# Combine UMAP coordinates
umap_1_coords = np.vstack([hsc_embedding_1, legacy_embedding_1])
umap_2_coords = np.vstack([hsc_embedding_2, legacy_embedding_2])

# ===== Create Interactive Visualization =====
print("\n" + "="*60)
print("Creating interactive visualization with image tooltips...")
print("="*60)

# Create DataFrames for Holoviews
df_1 = pd.DataFrame({
    'x': umap_1_coords[:, 0],
    'y': umap_1_coords[:, 1],
    'label': labels,
    'image': all_images_b64
})

df_2 = pd.DataFrame({
    'x': umap_2_coords[:, 0],
    'y': umap_2_coords[:, 1],
    'label': labels,
    'image': all_images_b64
})

# Define custom HTML tooltip with image
image_hover = HoverTool(
    tooltips="""
    <div style="background: white; padding: 8px; border-radius: 5px; box-shadow: 2px 2px 4px rgba(0,0,0,0.2);">
        <div style="margin-bottom: 5px; font-weight: bold; color: #333;">@label Galaxy</div>
        <img src="@image" width="96" height="96" style="border: 2px solid #ccc; border-radius: 3px;">
        <div style="font-size: 10px; color: #666; margin-top: 5px;">UMAP: (@x{0.2f}, @y{0.2f})</div>
    </div>
    """
)

def create_plot(df, title):
    """Create an interactive plot with image tooltips."""
    points = hv.Points(
        df,
        kdims=['x', 'y'],
        vdims=['label', 'image']
    ).opts(
        color='label',
        cmap={'HSC': 'dodgerblue', 'Legacy': 'darkorange'},
        size=5,
        alpha=0.7,
        tools=[image_hover, 'box_select', 'lasso_select', 'tap', 'pan', 'wheel_zoom', 'reset'],
        width=800,
        height=600,
        title=title,
        legend_position='top_right',
        nonselection_alpha=0.1
    )
    return points

plot_1 = create_plot(df_1, "Encoder 1: Physics Latent Space (UMAP)")
plot_2 = create_plot(df_2, "Encoder 2: Instrument Latent Space (UMAP)")

combined_plot = (plot_1 + plot_2).cols(2)

# Save to HTML
output_path = figures_dir / f'umap_interactive_images_zdim{dim}.html'
print(f"\nSaving interactive visualization to: {output_path}")
hv.save(combined_plot, output_path)

print(f"\n{'='*60}")
print("Visualization saved successfully!")
print(f"{'='*60}")
print(f"Output file: {output_path}")
print(f"  Total points: {2 * num_hsc} ({num_hsc} HSC + {num_hsc} Legacy)")
print(f"\nTo view:")
print(f"  1. Transfer the HTML file to your local machine:")
print(f"     scp {output_path} <local_path>")
print(f"  2. Open the HTML file in your web browser")
print(f"  3. Hover over any point to see the galaxy image!")
print(f"{'='*60}")
