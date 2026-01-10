"""
Download specific data files from nvidia/Nemotron-Math-v2 dataset.
"""

import argparse
from huggingface_hub import hf_hub_download

REPO_ID = "nvidia/Nemotron-Math-v2"

# Available files
AVAILABLE_FILES = {
    "medium": "data/medium.jsonl",
    "high_00": "data/high.part_00.jsonl",
    "high_01": "data/high.part_01.jsonl",
    "high_02": "data/high.part_02.jsonl",
    "low": "data/low.jsonl",
}


def main():
    parser = argparse.ArgumentParser(description="Download Nemotron-Math-v2 data files")
    parser.add_argument(
        "split",
        choices=list(AVAILABLE_FILES.keys()),
        help="Which split to download"
    )
    args = parser.parse_args()
    
    filename = AVAILABLE_FILES[args.split]
    print(f"Downloading {filename} from {REPO_ID}...")
    
    local_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
    )
    
    print(f"\nDownloaded to: {local_path}")
    print(f"\nTo analyze, run:")
    print(f"  python analyze_nemotron_math.py {local_path}")


if __name__ == "__main__":
    main()
