from datasets import load_dataset, Dataset
from functools import partial
import sys

dataset_name = sys.argv[1]
output_path = sys.argv[2]

print(f"dataset_name: {dataset_name}")
print(f"output_path: {output_path}")

print("Loading dataset...")

from huggingface_hub import snapshot_download

repo_id = dataset_name

snapshot_download(repo_id=repo_id, repo_type="dataset", 
                  local_dir=f"{output_path}/download",
                  cache_dir=f"{output_path}/cache")
