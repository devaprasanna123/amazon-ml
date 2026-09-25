"""
scripts/run_normalization.py
============================
Task 3: Normalize all training and test source files.
Benchmarks on a sample, then processes full files.
Saves normalized parquet/CSV files to data/normalized/.
Saves normalization examples to reports/normalization_examples.txt.
"""

import io
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    PROJECT_ROOT, REPORTS_DIR,
)
from src.preprocessing import (
    normalize_name, normalize_address, normalize_country,
    name_tokens, address_tokens, address_numeric_tokens,
    extract_postal_code, normalize_dataframe,
)

NORM_DIR = PROJECT_ROOT / "data" / "normalized"
NORM_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train_source1": TRAIN_SOURCE1,
    "train_source2": TRAIN_SOURCE2,
    "train_source3": TRAIN_SOURCE3,
    "test_source1": TEST_SOURCE1,
    "test_source2": TEST_SOURCE2,
    "test_source3": TEST_SOURCE3,
}


def benchmark_sample():
    """Run normalization on 10k rows and measure throughput."""
    print("=== BENCHMARK (10k rows from train_source1) ===")
    sample = pd.read_csv(TRAIN_SOURCE1, sep="\t", nrows=10000,
                          dtype=str, keep_default_na=False)
    t0 = time.time()
    normed = normalize_dataframe(sample)
    elapsed = time.time() - t0
    rows_per_sec = len(sample) / elapsed
    print(f"  Rows: {len(sample):,}  Time: {elapsed:.2f}s  "
          f"Throughput: {rows_per_sec:,.0f} rows/s")
    return rows_per_sec, normed


def collect_normalization_examples(normed: pd.DataFrame) -> str:
    """Build examples text from normalized sample DataFrame."""
    lines = []
    lines.append("=" * 72)
    lines.append("NORMALIZATION EXAMPLES (from train_source1 sample)")
    lines.append("=" * 72)
    lines.append("")

    # Show 30 random examples
    sample = normed.sample(min(30, len(normed)), random_state=42)
    for _, row in sample.iterrows():
        lines.append(f"  entity_id : {row['entity_id']}")
        lines.append(f"  country   : {row['country']!r} -> {row['norm_country']!r}")
        lines.append(f"  orig name : {row['business_name']!r}")
        lines.append(f"  norm name : {row['norm_name']!r}")
        lines.append(f"  name toks : {row['name_toks']!r}")
        lines.append(f"  orig addr : {row['business_address'][:80]!r}")
        lines.append(f"  norm addr : {row['norm_address'][:80]!r}")
        lines.append(f"  addr toks : {row['addr_toks'][:60]!r}")
        lines.append(f"  numerics  : {row['addr_numerics']!r}")
        lines.append(f"  postal    : {row['postal_code']!r}")
        lines.append("")

    return "\n".join(lines)


def main():
    print("TASK 3 — NORMALIZATION")
    print("=" * 60)

    # Benchmark
    rows_per_sec, sample_normed = benchmark_sample()

    # Save normalization examples
    examples_txt = collect_normalization_examples(sample_normed)
    ex_path = REPORTS_DIR / "normalization_examples.txt"
    with open(ex_path, "w", encoding="utf-8") as f:
        f.write(examples_txt)
    print(f"\nNormalization examples saved: {ex_path}")

    # Estimate full processing time
    total_train_rows = 2_206_821 + 5_034_616 + 5_285_603
    total_test_rows = 1_732_544 + 4_887_273 + 5_082_316
    est_time = (total_train_rows + total_test_rows) / rows_per_sec
    print(f"\nEstimated full normalization time: {est_time/60:.1f} min")

    # Process all files
    print("\nProcessing all source files...")
    t_global = time.time()
    for label, src_path in FILES.items():
        out_path = NORM_DIR / f"{label}_normalized.csv"
        if out_path.exists():
            print(f"  [{label}] already exists, skipping.")
            continue
        print(f"  [{label}] {src_path.stat().st_size / (1024**2):.1f} MB -> {out_path.name}")
        t0 = time.time()
        chunks = []
        for chunk in pd.read_csv(src_path, sep="\t", dtype=str,
                                  keep_default_na=False, chunksize=50_000):
            chunks.append(normalize_dataframe(chunk))
        df_full = pd.concat(chunks, ignore_index=True)
        df_full.to_csv(str(out_path), index=False)
        elapsed = time.time() - t0
        print(f"       rows={len(df_full):,}  time={elapsed:.1f}s  -> {out_path}")

    print(f"\nAll files normalized in {time.time()-t_global:.1f}s")
    print(f"Normalized files in: {NORM_DIR}")


if __name__ == "__main__":
    main()
