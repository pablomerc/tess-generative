import os
from huggingface_hub import snapshot_download

repo_id = "Smith42/legacysurvey_hsc_crossmatched"

DATA_DIR = "/data/vision/billf/scratch/pablomer/legacysurvey_hsc"
HF_HOME = "/data/vision/billf/scratch/pablomer/.cache/huggingface"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HF_HOME, exist_ok=True)

# Make sure cache is in scratch, not ~/.cache
os.environ["HF_HOME"] = HF_HOME

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    local_dir=DATA_DIR,
    resume_download=True,
    allow_patterns=["data/train-*.parquet"],
)

print("Download complete:", DATA_DIR)
