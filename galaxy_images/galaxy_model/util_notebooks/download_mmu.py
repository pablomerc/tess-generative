"""
Download Multimodal Universe parquet files to local scratch.

Usage:
    python util_notebooks/download_mmu.py
"""
import os
from huggingface_hub import snapshot_download

# MMU dataset: HSC x Legacy Survey crossmatch (Multimodal Universe)
REPO_ID = "Smith42/legacysurvey_hsc_crossmatched"

DATA_DIR = "/work1/jeroenaudenaert/pablomer/data/raw_mmu"
HF_HOME = "/work1/jeroenaudenaert/pablomer/data/.cache/huggingface"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)

os.environ["HF_HOME"] = HF_HOME

snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=DATA_DIR,
    resume_download=True,
    allow_patterns=["data/train-*.parquet"],
)

print("Download complete:", DATA_DIR)
