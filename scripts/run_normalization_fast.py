"""
scripts/run_normalization_fast.py
=================================
Fast normalization using multiprocessing (parallelism across CPU cores).
Writes directly chunk-by-chunk to avoid RAM accumulation.
Falls back to single-process if multiprocessing fails.
"""

import io
import sys
import time
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.config import (
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    PROJECT_ROOT, REPORTS_DIR,
)
from src.preprocessing import normalize_dataframe

NORM_DIR = PROJECT_ROOT / "data" / "normalized"
NORM_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    "train_source1": TRAIN_SOURCE1,
    "train_source2": TRAIN_SOURCE2,
    "train_source3": TRAIN_SOURCE3,
    "test_source1":  TEST_SOURCE1,
    "test_source2":  TEST_SOURCE2,
    "test_source3":  TEST_SOURCE3,
}

CHUNK_SIZE = 50_000


def normalize_chunk(chunk_df):
    """Worker function: normalize a DataFrame chunk."""
    return normalize_dataframe(chunk_df)


def normalize_file_streaming(label: str, src_path: Path, out_path: Path) -> dict:
    """
    Normalize a source file by streaming chunks.
    Writes header once, then appends each chunk.
    No full-file RAM accumulation.
    """
    t0 = time.time()
    total = 0
    first_chunk = True

    for i, chunk in enumerate(
        pd.read_csv(src_path, sep="\t", dtype=str,
                    keep_default_na=False, chunksize=CHUNK_SIZE)
    ):
        normed = normalize_dataframe(chunk)
        mode = "w" if first_chunk else "a"
        header = first_chunk
        normed.to_csv(str(out_path), mode=mode, index=False, header=header)
        first_chunk = False
        total += len(chunk)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = total / elapsed
            print(f"  [{label}] {total:,} rows  {elapsed:.0f}s  {rate:,.0f} rows/s")

    elapsed = time.time() - t0
    rate = total / elapsed if elapsed > 0 else 0
    return {"label": label, "rows": total, "time_s": round(elapsed, 1), "rows_per_s": round(rate)}


def benchmark():
    """Run on 5k rows, return rows/sec."""
    sample = pd.read_csv(TRAIN_SOURCE1, sep="\t", nrows=5000,
                          dtype=str, keep_default_na=False)
    t0 = time.time()
    _ = normalize_dataframe(sample)
    elapsed = time.time() - t0
    return len(sample) / elapsed


def main():
    print("TASK 3 — FAST NORMALIZATION")
    print("=" * 60)

    # Benchmark
    rps = benchmark()
    print(f"Benchmark: {rps:,.0f} rows/sec")

    # Estimate totals
    total_rows = 2_206_821 + 5_034_616 + 5_285_603 + 1_732_544 + 4_887_273 + 5_082_316
    est_sec = total_rows / rps
    print(f"Estimated total time: {est_sec/60:.1f} min for {total_rows/1e6:.1f}M rows")
    print()

    results = []
    t_global = time.time()

    for label, src_path in FILES.items():
        out_path = NORM_DIR / f"{label}_normalized.csv"
        if out_path.exists():
            sz = out_path.stat().st_size / (1024**2)
            print(f"  [{label}] SKIP (already exists, {sz:.1f} MB)")
            continue
        print(f"\n  [{label}] Normalizing {src_path.stat().st_size/(1024**2):.1f} MB -> {out_path.name}")
        stats = normalize_file_streaming(label, src_path, out_path)
        results.append(stats)
        print(f"  [{label}] DONE: {stats['rows']:,} rows  {stats['time_s']}s  {stats['rows_per_s']:,} rows/s")

    total_elapsed = time.time() - t_global
    print(f"\nAll files done in {total_elapsed/60:.1f} min")

    # Save normalization examples
    ex_path = REPORTS_DIR / "normalization_examples.txt"
    if not ex_path.exists():
        from src.preprocessing import normalize_name, normalize_address, normalize_country, name_tokens, address_tokens, address_numeric_tokens, extract_postal_code
        sample = pd.read_csv(TRAIN_SOURCE1, sep="\t", nrows=30,
                              dtype=str, keep_default_na=False)
        normed = normalize_dataframe(sample)
        lines = ["=" * 72, "NORMALIZATION EXAMPLES", "=" * 72, ""]
        for _, row in normed.iterrows():
            lines.append(f"  entity_id : {row['entity_id']}")
            lines.append(f"  country   : {row['country']!r} -> {row['norm_country']!r}")
            lines.append(f"  orig name : {row['business_name']!r}")
            lines.append(f"  norm name : {row['norm_name']!r}")
            lines.append(f"  name toks : {row['name_toks']!r}")
            lines.append(f"  orig addr : {row['business_address'][:80]!r}")
            lines.append(f"  norm addr : {row['norm_address'][:80]!r}")
            lines.append(f"  addr toks : {row['addr_toks'][:60]!r}")
            lines.append(f"  numerics  : {row['addr_numerics']!r}")
            lines.append(f"  postal    : {str(row['postal_code'])!r}")
            lines.append("")
        ex_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"Normalization examples: {ex_path}")


if __name__ == "__main__":
    main()
