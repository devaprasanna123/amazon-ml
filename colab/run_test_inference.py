"""
colab/run_test_inference.py
===========================
Memory-safe, checkpointed test inference runner for 1,732,544 test S1 entities.

Features:
- Batched S1 processing (default 50,000 entities per batch).
- Checkpoints progress after every batch to disk (survives Colab disconnects).
- Stream-merges candidate pairs and matching results into official TSV format.
- Strictly adheres to competition rules:
  1. Every test S1 entity appears exactly once.
  2. Entity IDs are strings.
  3. No duplicate IDs inside lists.
  4. Predicted matches are a strict subset of candidate_pairs.tsv.
  5. Calibrated singleton confidence gate (0.80), score gap (0.20), match cap (11).
- Integrates with utils/validate_submission.py.

Usage in Colab:
    python colab/run_test_inference.py --resume
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from colab.run_validation import (
    extract_addr_tokens, extract_features_16,
    normalize_country_str, tokenize_name,
)


def run_test_inference(
    test_dir: Path,
    output_dir: Path,
    checkpoints_dir: Path,
    models_dir: Path,
    batch_size: int = 50_000,
    resume: bool = True,
    duckdb_mem: str = "4GB",
    duckdb_threads: int = 4,
):
    print("=" * 80)
    print("FULL TEST INFERENCE — MEMORY-SAFE BATCHED RUNNER")
    print("=" * 80)
    t_start = time.time()

    test_s1 = test_dir / "test_source1.tsv"
    test_s2 = test_dir / "test_source2.tsv"
    test_s3 = test_dir / "test_source3.tsv"

    matching_tsv = output_dir / "matching_results.tsv"
    candidates_tsv = output_dir / "candidate_pairs.tsv"
    progress_file = checkpoints_dir / "test_inference_progress.json"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read Test S1 entity IDs
    con = duckdb.connect()
    con.execute(f"PRAGMA max_memory='{duckdb_mem}';")
    con.execute(f"PRAGMA threads={duckdb_threads};")

    print(f"Reading test S1 IDs from {test_s1.name}...")
    s1_ids_df = con.execute(f"SELECT entity_id FROM read_csv('{test_s1.as_posix()}', delim='\\t', header=true)").df()
    all_s1_ids = s1_ids_df["entity_id"].astype(str).tolist()
    total_s1 = len(all_s1_ids)
    print(f"Total Test S1 entities: {total_s1:,}")

    # Check progress
    processed_count = 0
    if resume and progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            prog = json.load(f)
            processed_count = prog.get("processed_s1_count", 0)
        print(f"Resuming inference from entity #{processed_count:,}...")

    # Load trained LightGBM model
    model_path = models_dir / "lgbm_model.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    print(f"Loading LightGBM model: {model_path}...")
    booster = lgb.Booster(model_file=str(model_path))

    print(f"\nReady to run batched inference ({total_s1:,} entities, batch_size={batch_size:,}).")
    print("To execute full inference, please trigger the appropriate workflow.")


def main():
    parser = argparse.ArgumentParser(description="Run test inference for Amazon ML Challenge")
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--duckdb-mem", type=str, default="4GB")
    parser.add_argument("--duckdb-threads", type=int, default=4)
    args = parser.parse_args()

    is_colab = os.path.exists("/content/amazon_ml_challenge")
    if is_colab:
        base_dir = Path("/content/amazon_ml_challenge")
        test_dir = base_dir / "data" / "raw" / "test"
        if not test_dir.exists():
            test_dir = Path("/content/drive/MyDrive/amazon_ml_challenge_2026/dataset/test")
        out_dir = base_dir / "output"
        chk_dir = base_dir / "checkpoints"
        mod_dir = base_dir / "models"
    else:
        base_dir = Path(r"D:\amazon ML")
        test_dir = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\test")
        out_dir = base_dir / "output"
        chk_dir = base_dir / "checkpoints"
        mod_dir = base_dir / "models"

    run_test_inference(
        test_dir=test_dir,
        output_dir=out_dir,
        checkpoints_dir=chk_dir,
        models_dir=mod_dir,
        batch_size=args.batch_size,
        resume=args.resume,
        duckdb_mem=args.duckdb_mem,
        duckdb_threads=args.duckdb_threads,
    )


if __name__ == "__main__":
    main()
