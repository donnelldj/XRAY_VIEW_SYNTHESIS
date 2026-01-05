#!/usr/bin/env python
"""
download_luna16_and_place.py

Downloads the LUNA16 dataset from Kaggle using kagglehub and
copies it into the expected repository location:

    xray_synth/data/luna16/

Prerequisites:
- Kaggle account
- kaggle.json configured locally
  https://www.kaggle.com/docs/api

Usage:
  python download_luna16_and_place.py
"""

from pathlib import Path
import shutil
import kagglehub


def main():
    # Download (cached by kagglehub)
    dataset_path = Path(kagglehub.dataset_download("avc0706/luna16")).resolve()
    print(f"Downloaded LUNA16 to: {dataset_path}")

    # Target directory relative to repo root
    repo_root = Path(__file__).resolve().parent
    target_root = repo_root / "xray_synth" / "data" / "luna16"
    target_root.mkdir(parents=True, exist_ok=True)

    # Copy subset folders
    copied = 0
    for item in dataset_path.iterdir():
        if item.is_dir() and item.name.startswith("subset"):
            dest = target_root / item.name
            if dest.exists():
                print(f"Skipping existing folder: {dest}")
            else:
                print(f"Copying {item.name} -> {dest}")
                shutil.copytree(item, dest)
                copied += 1

    print("\nDone.")
    print(f"LUNA16 is now available under: {target_root}")
    if copied == 0:
        print("(No new folders were copied; data may already exist.)")


if __name__ == "__main__":
    main()
