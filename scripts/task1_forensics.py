"""
Task 1 — Dataset Forensics
Memory-efficient audit of the Amazon ML Challenge dataset.
Uses chunked pandas reading (no Polars dependency needed; Polars used if available).
"""

import csv
import json
import os
import sys
import io
import time
from collections import Counter, defaultdict
from pathlib import Path

# Force UTF-8 on Windows stdout (avoids cp1252 UnicodeEncodeError)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ------------------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------------------
DATASET_ROOT = Path(
    r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset"
)
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"
PROJECT_ROOT = Path(r"D:\amazon ML")
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = {
    "train_source1": TRAIN_DIR / "train_source1.tsv",
    "train_source2": TRAIN_DIR / "train_source2.tsv",
    "train_source3": TRAIN_DIR / "train_source3.tsv",
    "train_ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}
TEST_FILES = {
    "test_source1": TEST_DIR / "test_source1.tsv",
    "test_source2": TEST_DIR / "test_source2.tsv",
    "test_source3": TEST_DIR / "test_source3.tsv",
}
ALL_FILES = {**TRAIN_FILES, **TEST_FILES}

CHUNK_SIZE = 50_000  # rows per chunk


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 ** 2)


def streaming_source_audit(path: Path, label: str) -> dict:
    """Audit a source TSV file in a memory-efficient streaming fashion."""
    import pandas as pd

    row_count = 0
    columns = None
    missing: Counter = Counter()
    country_set: set = set()
    id_counter: Counter = Counter()
    name_lengths: list = []   # we'll store per-chunk stats, merge at end
    addr_lengths: list = []

    # Running stats for name / address lengths (Welford online)
    def welford_update(state, x):
        state["n"] += 1
        delta = x - state["mean"]
        state["mean"] += delta / state["n"]
        delta2 = x - state["mean"]
        state["M2"] += delta * delta2

    def make_state():
        return {"n": 0, "mean": 0.0, "M2": 0.0, "min": float("inf"), "max": float("-inf")}

    name_st = make_state()
    addr_st = make_state()

    for chunk in pd.read_csv(path, sep="\t", chunksize=CHUNK_SIZE,
                              dtype=str, keep_default_na=False):
        if columns is None:
            columns = list(chunk.columns)

        row_count += len(chunk)

        for col in columns:
            missing[col] += (chunk[col] == "").sum() + chunk[col].isna().sum()

        if "country" in columns:
            country_set.update(chunk["country"].dropna().unique().tolist())

        if "entity_id" in columns:
            for eid in chunk["entity_id"]:
                id_counter[eid] += 1

        if "business_name" in columns:
            lens = chunk["business_name"].str.len().dropna()
            for v in lens:
                welford_update(name_st, v)
                if v < name_st["min"]:
                    name_st["min"] = v
                if v > name_st["max"]:
                    name_st["max"] = v

        if "business_address" in columns:
            lens = chunk["business_address"].str.len().dropna()
            for v in lens:
                welford_update(addr_st, v)
                if v < addr_st["min"]:
                    addr_st["min"] = v
                if v > addr_st["max"]:
                    addr_st["max"] = v

    import math
    def finalise(st):
        variance = st["M2"] / st["n"] if st["n"] > 1 else 0
        return {
            "count": st["n"],
            "mean": round(st["mean"], 2),
            "std": round(math.sqrt(variance), 2),
            "min": st["min"] if st["n"] > 0 else None,
            "max": st["max"] if st["n"] > 0 else None,
        }

    duplicate_ids = {k: v for k, v in id_counter.items() if v > 1}

    result = {
        "file": str(path),
        "size_mb": round(file_size_mb(path), 2),
        "row_count": row_count,
        "columns": columns,
        "missing_values": {k: int(v) for k, v in missing.items()},
        "distinct_countries": sorted(country_set) if country_set else None,
        "unique_id_count": len(id_counter),
        "duplicate_id_count": len(duplicate_ids),
        "duplicate_ids_sample": list(duplicate_ids.keys())[:5],
    }

    if name_st["n"] > 0:
        result["name_length_stats"] = finalise(name_st)
    if addr_st["n"] > 0:
        result["address_length_stats"] = finalise(addr_st)

    print(f"  [{label}] rows={row_count:,}  unique_ids={len(id_counter):,}  "
          f"dups={len(duplicate_ids)}  countries={sorted(country_set)}")
    return result


def audit_ground_truth(path: Path, s1_ids: set, s2_ids: set, s3_ids: set) -> dict:
    """Parse ground truth, compute distribution, verify integrity."""
    import pandas as pd

    gt_columns = None
    row_count = 0
    match_count_dist: Counter = Counter()   # 0,1,2,3+
    max_matches = 0
    total_positive_links = 0
    singleton_s1 = 0

    # integrity checks
    missing_s1: list = []
    missing_s2: list = []
    missing_s3: list = []
    s1_appears_in_matched: list = []

    for chunk in pd.read_csv(path, sep="\t", chunksize=CHUNK_SIZE,
                              dtype=str, keep_default_na=False):
        if gt_columns is None:
            gt_columns = list(chunk.columns)

        row_count += len(chunk)

        for _, row in chunk.iterrows():
            s1_id = row.get("source1_entity_id", "")
            matched_raw = row.get("matched_entity_ids", "")

            # verify S1 exists in train_source1
            if s1_id not in s1_ids:
                missing_s1.append(s1_id)

            if matched_raw.strip() == "":
                matches = []
            else:
                matches = [m.strip() for m in matched_raw.split(",") if m.strip()]

            n = len(matches)
            total_positive_links += n
            max_matches = max(max_matches, n)

            if n == 0:
                match_count_dist["0"] += 1
                singleton_s1 += 1
            elif n == 1:
                match_count_dist["1"] += 1
            elif n == 2:
                match_count_dist["2"] += 1
            else:
                match_count_dist["3+"] += 1

            for mid in matches:
                if mid.startswith("S2-"):
                    if mid not in s2_ids:
                        missing_s2.append(mid)
                    if mid in s1_ids:
                        s1_appears_in_matched.append(mid)
                elif mid.startswith("S3-"):
                    if mid not in s3_ids:
                        missing_s3.append(mid)
                    if mid in s1_ids:
                        s1_appears_in_matched.append(mid)

    proportion_singletons = singleton_s1 / row_count if row_count > 0 else 0.0

    integrity = {
        "s1_ids_not_in_source1": len(missing_s1),
        "s2_ids_not_in_source2": len(missing_s2),
        "s3_ids_not_in_source3": len(missing_s3),
        "s1_ids_in_matched_column": len(s1_appears_in_matched),
        "passed": (len(missing_s1) == 0 and len(missing_s2) == 0
                   and len(missing_s3) == 0 and len(s1_appears_in_matched) == 0),
    }
    if missing_s1[:5]:
        integrity["missing_s1_sample"] = missing_s1[:5]
    if missing_s2[:5]:
        integrity["missing_s2_sample"] = missing_s2[:5]
    if missing_s3[:5]:
        integrity["missing_s3_sample"] = missing_s3[:5]

    return {
        "file": str(path),
        "size_mb": round(file_size_mb(path), 2),
        "row_count": row_count,
        "columns": gt_columns,
        "match_count_distribution": {k: int(v) for k, v in sorted(match_count_dist.items())},
        "max_matches_for_one_s1": max_matches,
        "singleton_s1_count": singleton_s1,
        "proportion_singleton_s1": round(proportion_singletons, 4),
        "total_positive_links": total_positive_links,
        "integrity": integrity,
    }


def collect_ids(path: Path, col: str = "entity_id") -> set:
    """Stream-collect a set of IDs from a column."""
    import pandas as pd
    ids: set = set()
    for chunk in pd.read_csv(path, sep="\t", chunksize=CHUNK_SIZE,
                              dtype=str, keep_default_na=False,
                              usecols=[col]):
        ids.update(chunk[col].tolist())
    return ids


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("=" * 70)
    print("TASK 1 — DATASET FORENSICS")
    print("=" * 70)

    report = {
        "dataset_root": str(DATASET_ROOT),
        "audit_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": {},
        "ground_truth": None,
    }

    # -- Audit source files --------------------------------------------------
    source_labels = [
        ("train_source1", TRAIN_FILES["train_source1"]),
        ("train_source2", TRAIN_FILES["train_source2"]),
        ("train_source3", TRAIN_FILES["train_source3"]),
        ("test_source1",  TEST_FILES["test_source1"]),
        ("test_source2",  TEST_FILES["test_source2"]),
        ("test_source3",  TEST_FILES["test_source3"]),
    ]

    print("\n-- Source file audits ----------------------------------------------")
    all_countries: set = set()

    for label, path in source_labels:
        print(f"\nAuditing {label} ({file_size_mb(path):.1f} MB) …")
        stats = streaming_source_audit(path, label)
        report["files"][label] = stats
        if stats.get("distinct_countries"):
            all_countries.update(stats["distinct_countries"])

    report["all_distinct_countries"] = sorted(all_countries)

    # -- Collect ID sets for integrity checks --------------------------------
    print("\n-- Collecting ID sets for integrity check …")
    s1_ids = collect_ids(TRAIN_FILES["train_source1"])
    s2_ids = collect_ids(TRAIN_FILES["train_source2"])
    s3_ids = collect_ids(TRAIN_FILES["train_source3"])
    print(f"   S1={len(s1_ids):,}  S2={len(s2_ids):,}  S3={len(s3_ids):,}")

    # -- Audit ground truth --------------------------------------------------
    print(f"\nAuditing train_ground_truth ({file_size_mb(TRAIN_FILES['train_ground_truth']):.1f} MB) …")
    gt_stats = audit_ground_truth(
        TRAIN_FILES["train_ground_truth"], s1_ids, s2_ids, s3_ids
    )
    report["ground_truth"] = gt_stats

    elapsed = time.time() - t0

    # -- Save JSON report ----------------------------------------------------
    json_path = REPORTS_DIR / "dataset_inventory.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # -- Build TXT report ----------------------------------------------------
    txt_lines = []
    def w(s=""):
        txt_lines.append(s)

    w("=" * 70)
    w("DATASET INVENTORY REPORT")
    w(f"Generated: {report['audit_timestamp']}")
    w("=" * 70)
    w()
    w(f"Dataset root: {report['dataset_root']}")
    w()

    w("-- FILES ---------------------------------------------------------------")
    for lbl, stats in report["files"].items():
        w(f"  {lbl}")
        w(f"    Path     : {stats['file']}")
        w(f"    Size     : {stats['size_mb']} MB")
        w(f"    Rows     : {stats['row_count']:,}")
        w(f"    Columns  : {stats['columns']}")
        mv = stats['missing_values']
        nonzero_mv = {k: v for k, v in mv.items() if v > 0}
        w(f"    Missing  : {nonzero_mv if nonzero_mv else 'none'}")
        if stats.get("distinct_countries"):
            w(f"    Countries: {stats['distinct_countries']}")
        w(f"    Unique IDs: {stats['unique_id_count']:,}  Dup IDs: {stats['duplicate_id_count']}")
        if stats.get("name_length_stats"):
            ns = stats["name_length_stats"]
            w(f"    Name len : mean={ns['mean']} std={ns['std']} min={ns['min']} max={ns['max']}")
        if stats.get("address_length_stats"):
            addr = stats["address_length_stats"]
            w(f"    Addr len : mean={addr['mean']} std={addr['std']} min={addr['min']} max={addr['max']}")
        w()

    w("-- GROUND-TRUTH --------------------------------------------------------")
    gt = report["ground_truth"]
    w(f"  File      : {gt['file']}")
    w(f"  Size      : {gt['size_mb']} MB")
    w(f"  Rows      : {gt['row_count']:,}")
    w(f"  Columns   : {gt['columns']}")
    w()
    w("  Match count distribution:")
    for k, v in gt["match_count_distribution"].items():
        pct = 100 * v / gt["row_count"]
        w(f"    {k} matches : {v:>8,}  ({pct:.2f}%)")
    w()
    w(f"  Max matches for one S1  : {gt['max_matches_for_one_s1']}")
    w(f"  Singleton S1 count      : {gt['singleton_s1_count']:,}")
    w(f"  Proportion singleton S1 : {gt['proportion_singleton_s1']:.4f} "
      f"({100*gt['proportion_singleton_s1']:.2f}%)")
    w(f"  Total positive links    : {gt['total_positive_links']:,}")
    w()

    w("-- INTEGRITY CHECK -----------------------------------------------------")
    ig = gt["integrity"]
    w(f"  S1 IDs not in train_source1   : {ig['s1_ids_not_in_source1']}")
    w(f"  S2 IDs not in train_source2   : {ig['s2_ids_not_in_source2']}")
    w(f"  S3 IDs not in train_source3   : {ig['s3_ids_not_in_source3']}")
    w(f"  S1 IDs appearing in matched   : {ig['s1_ids_in_matched_column']}")
    w(f"  PASSED: {ig['passed']}")
    w()

    w("-- ALL DISTINCT COUNTRIES ----------------------------------------------")
    w(f"  Train: {sorted(all_countries)}")
    w(f"  (Test may include France)")
    w()
    w(f"Audit completed in {elapsed:.1f}s")
    w("=" * 70)

    txt_path = REPORTS_DIR / "dataset_inventory.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))

    # -- Print concise summary -----------------------------------------------
    print()
    print("=" * 70)
    print("CONCISE SUMMARY")
    print("=" * 70)

    print("\nFILES")
    for lbl, stats in report["files"].items():
        print(f"  {lbl:<25} {stats['size_mb']:>8.1f} MB  {stats['row_count']:>10,} rows")

    print("\nROWS")
    for lbl, stats in report["files"].items():
        print(f"  {lbl}: {stats['row_count']:,}")

    print("\nCOUNTRIES")
    print(f"  Train: {report['all_distinct_countries']}")

    print("\nGROUND-TRUTH DISTRIBUTION")
    for k, v in gt["match_count_distribution"].items():
        pct = 100 * v / gt["row_count"]
        print(f"  {k} matches: {v:,} ({pct:.2f}%)")

    print("\nSINGLETONS")
    print(f"  Count     : {gt['singleton_s1_count']:,}")
    print(f"  Proportion: {gt['proportion_singleton_s1']:.4f} ({100*gt['proportion_singleton_s1']:.2f}%)")

    print("\nMAX MATCHES")
    print(f"  Max matches for one S1 entity: {gt['max_matches_for_one_s1']}")

    print("\nINTEGRITY CHECK")
    ig = gt["integrity"]
    print(f"  S1 IDs missing from source1 : {ig['s1_ids_not_in_source1']}")
    print(f"  S2 IDs missing from source2 : {ig['s2_ids_not_in_source2']}")
    print(f"  S3 IDs missing from source3 : {ig['s3_ids_not_in_source3']}")
    print(f"  S1 IDs leaked into matched  : {ig['s1_ids_in_matched_column']}")
    print(f"  INTEGRITY PASSED            : {ig['passed']}")

    print(f"\nReports saved:")
    print(f"  {json_path}")
    print(f"  {txt_path}")
    print(f"\nTotal time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
