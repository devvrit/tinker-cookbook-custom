"""
Download OpenThoughts3-1.2M dataset from Hugging Face.

The dataset has 120 parquet shards (1.2M rows total, ~10k per shard).
You can download individual shards or all of them.
"""

import argparse
from huggingface_hub import hf_hub_download, snapshot_download
import os

REPO_ID = "open-thoughts/OpenThoughts3-1.2M"
NUM_SHARDS = 120


def download_shard(shard_id: int) -> str:
    """Download a single shard by ID (0-119)."""
    filename = f"data/train-{shard_id:05d}-of-00120.parquet"
    print(f"Downloading {filename} from {REPO_ID}...")
    return hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )


def main():
    parser = argparse.ArgumentParser(description="Download OpenThoughts3-1.2M data files")
    parser.add_argument(
        "--shard",
        type=int,
        default=0,
        help="Shard ID to download (0-119). Each shard has ~10k samples. Default: 0"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all 120 shards (warning: ~28GB total)"
    )
    args = parser.parse_args()
    
    if args.all:
        print(f"Downloading ALL 120 shards from {REPO_ID}...")
        print("Warning: This is ~28GB total!")
        local_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            allow_patterns="data/*.parquet",
        )
        print(f"\nDownloaded to: {local_dir}")
        print(f"\nTo analyze all files:")
        print(f"  for f in {local_dir}/data/*.parquet; do python analyze_nemotron_math.py \"$f\" --skip-tir-filter; done")
    else:
        if args.shard < 0 or args.shard >= NUM_SHARDS:
            print(f"Error: shard must be 0-{NUM_SHARDS-1}")
            return
        
        local_path = download_shard(args.shard)
        print(f"\nDownloaded to: {local_path}")
        print(f"\nTo analyze, run:")
        print(f"  python analyze_nemotron_math.py {local_path} --skip-tir-filter")


if __name__ == "__main__":
    main()
