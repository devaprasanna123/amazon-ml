"""
colab/sync_project.py
=====================
Intelligent bi-directional staging and synchronization between persistent
Google Drive storage and fast ephemeral Colab local NVMe storage.

Prevents re-copying unchanged multi-gigabyte files and guarantees that all
checkpoints, models, reports, and submission TSVs are immediately persisted.

Usage in Colab:
    python colab/sync_project.py --action stage_dataset
    python colab/sync_project.py --action push_artifacts
    python colab/sync_project.py --action status
"""

import argparse
import hashlib
import os
import shutil
import sys
import time
from pathlib import Path


def get_file_stats(path: Path):
    if not path.exists():
        return None
    st = path.stat()
    return {
        "size": st.st_size,
        "mtime": st.st_mtime,
    }


def copy_if_different(src: Path, dst: Path, log_prefix="  "):
    if not src.exists():
        print(f"{log_prefix}Source does not exist: {src}")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        src_stat = src.stat()
        dst_stat = dst.stat()
        # Fast check: identical size
        if src_stat.st_size == dst_stat.st_size and src_stat.st_size > 0:
            print(f"{log_prefix}Skipped (identical size {src_stat.st_size / (1024*1024):.1f} MB): {dst.name}")
            return True

    print(f"{log_prefix}Copying {src.name} ({src.stat().st_size / (1024*1024):.1f} MB) -> {dst} ...")
    t0 = time.time()
    shutil.copy2(src, dst)
    elapsed = time.time() - t0
    rate = (src.stat().st_size / (1024 * 1024)) / max(0.01, elapsed)
    print(f"{log_prefix}Done in {elapsed:.1f}s ({rate:.1f} MB/s)")
    return True


def stage_dataset(drive_root: Path, local_root: Path):
    """
    Stages raw TSV files from Google Drive to fast local Colab NVMe.
    """
    print("\n--- [STAGE DATASET] Drive -> Colab Local NVMe ---")
    drive_dataset = drive_root / "dataset"
    local_raw = local_root / "data" / "raw"

    if not drive_dataset.exists():
        print(f"Warning: Drive dataset folder {drive_dataset} not found.")
        return

    # Check for train and test folders
    for split in ["train", "test"]:
        src_split = drive_dataset / split
        dst_split = local_raw / split
        if src_split.exists():
            for tsv_file in src_split.glob("*.tsv"):
                copy_if_different(tsv_file, dst_split / tsv_file.name)
        else:
            # Check if tsv files are directly in drive_dataset
            for tsv_file in drive_dataset.glob(f"{split}_*.tsv"):
                copy_if_different(tsv_file, dst_split / tsv_file.name)


def push_artifacts(local_root: Path, drive_root: Path):
    """
    Persists all generated checkpoints, indexes, models, reports, and outputs
    from Colab local NVMe back to persistent Google Drive.
    """
    print("\n--- [PUSH ARTIFACTS] Colab Local -> Google Drive Persistent ---")
    sync_pairs = [
        (local_root / "models", drive_root / "artifacts" / "models"),
        (local_root / "reports", drive_root / "reports"),
        (local_root / "output", drive_root / "outputs"),
        (local_root / "checkpoints", drive_root / "checkpoints"),
        (local_root / "indexes", drive_root / "indexes"),
    ]

    for local_dir, drive_dir in sync_pairs:
        if not local_dir.exists():
            continue
        drive_dir.mkdir(parents=True, exist_ok=True)
        for item in local_dir.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(local_dir)
                target_file = drive_dir / rel_path
                copy_if_different(item, target_file)


def show_status(drive_root: Path, local_root: Path):
    """Prints current file inventory across both environments."""
    print("\n--- STORAGE STATUS REPORT ---")
    print(f"Google Drive Base: {drive_root}")
    if drive_root.exists():
        d_files = list(drive_root.rglob("*"))
        d_file_count = sum(1 for f in d_files if f.is_file())
        d_total_bytes = sum(f.stat().st_size for f in d_files if f.is_file())
        print(f"  Drive Files: {d_file_count} files ({d_total_bytes / (1024**3):.2f} GB)")
    else:
        print("  Drive Base does not exist!")

    print(f"\nLocal Colab Base: {local_root}")
    if local_root.exists():
        l_files = list(local_root.rglob("*"))
        l_file_count = sum(1 for f in l_files if f.is_file())
        l_total_bytes = sum(f.stat().st_size for f in l_files if f.is_file())
        print(f"  Local Files: {l_file_count} files ({l_total_bytes / (1024**3):.2f} GB)")
    else:
        print("  Local Base does not exist!")


def main():
    parser = argparse.ArgumentParser(description="Synchronize project artifacts between Colab and Drive")
    parser.add_argument("--action", type=str, required=True,
                        choices=["stage_dataset", "push_artifacts", "status"],
                        help="Action to perform")
    parser.add_argument("--drive-mount", type=str, default="/content/drive/MyDrive/amazon_ml_challenge_2026",
                        help="Path to Drive project root")
    parser.add_argument("--local-root", type=str, default="/content/amazon_ml_challenge",
                        help="Path to local Colab compute root")
    args = parser.parse_args()

    drive_root = Path(args.drive_mount)
    local_root = Path(args.local_root)

    if args.action == "stage_dataset":
        stage_dataset(drive_root, local_root)
    elif args.action == "push_artifacts":
        push_artifacts(local_root, drive_root)
    elif args.action == "status":
        show_status(drive_root, local_root)


if __name__ == "__main__":
    main()
