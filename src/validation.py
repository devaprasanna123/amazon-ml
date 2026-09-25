"""
src/validation.py
=================
Leakage-safe train/validation split and ground-truth parsing utilities.

Split strategy
--------------
- Groups are defined by source1_entity_id (S1 entity).
- No S1 entity can appear in both train and validation sets.
- Stratified by match cardinality bucket to preserve the distribution
  of singletons vs. multi-match S1s in both splits.
- Deterministic via a fixed random seed.

Usage
-----
    from src.validation import build_split, load_ground_truth

    gt = load_ground_truth()           # full dict: {s1_id: [matched_ids]}
    train_gt, val_gt = build_split(gt) # leakage-safe split
"""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Ground-truth loading
# ──────────────────────────────────────────────────────────────────────────────

def load_ground_truth(
    path: Path | None = None,
    chunk_size: int = 50_000,
) -> Dict[str, List[str]]:
    """
    Stream-parse train_ground_truth.tsv and return
    {source1_entity_id: [matched_entity_id, ...]}

    Empty matched_entity_ids → empty list (singleton).

    Parameters
    ----------
    path       : defaults to config.TRAIN_GROUND_TRUTH
    chunk_size : rows per pandas chunk

    Returns
    -------
    dict mapping each S1 entity_id to its list of matched IDs.
    """
    import pandas as pd

    if path is None:
        from src.config import TRAIN_GROUND_TRUTH
        path = TRAIN_GROUND_TRUTH

    gt: Dict[str, List[str]] = {}

    for chunk in pd.read_csv(
        path, sep="\t", chunksize=chunk_size,
        dtype=str, keep_default_na=False,
    ):
        for _, row in chunk.iterrows():
            s1_id = row["source1_entity_id"]
            raw = row["matched_entity_ids"].strip()
            if raw:
                matches = [m.strip() for m in raw.split(",") if m.strip()]
            else:
                matches = []
            gt[s1_id] = matches

    return gt


# ──────────────────────────────────────────────────────────────────────────────
# Split
# ──────────────────────────────────────────────────────────────────────────────

def _cardinality_bucket(n_matches: int) -> str:
    """Map match count to a stratification bucket."""
    if n_matches == 0:
        return "0"
    elif n_matches == 1:
        return "1"
    elif n_matches <= 3:
        return "2-3"
    else:
        return "4+"


def build_split(
    ground_truth: Dict[str, List[str]],
    val_fraction: float | None = None,
    seed: int | None = None,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Create a leakage-safe train/validation split grouped by source1_entity_id.

    No S1 entity can appear in both splits.  Stratified by match-cardinality
    bucket to preserve singleton / multi-match proportions.

    Parameters
    ----------
    ground_truth  : full {s1_id: [matched_ids]} dict
    val_fraction  : fraction of S1 entities for validation (default from config)
    seed          : random seed (default from config)

    Returns
    -------
    (train_gt, val_gt) — both same dict format as ground_truth
    """
    from src.config import VAL_FRACTION, RANDOM_SEED

    if val_fraction is None:
        val_fraction = VAL_FRACTION
    if seed is None:
        seed = RANDOM_SEED

    rng = random.Random(seed)

    # Group S1 IDs by cardinality bucket for stratification
    buckets: Dict[str, List[str]] = defaultdict(list)
    for s1_id, matches in ground_truth.items():
        bucket = _cardinality_bucket(len(matches))
        buckets[bucket].append(s1_id)

    val_ids: set = set()
    for bucket, ids in buckets.items():
        ids_shuffled = ids[:]
        rng.shuffle(ids_shuffled)
        n_val = max(1, round(len(ids_shuffled) * val_fraction))
        val_ids.update(ids_shuffled[:n_val])

    train_gt: Dict[str, List[str]] = {}
    val_gt: Dict[str, List[str]] = {}

    for s1_id, matches in ground_truth.items():
        if s1_id in val_ids:
            val_gt[s1_id] = matches
        else:
            train_gt[s1_id] = matches

    # Sanity: no overlap
    assert not (set(train_gt) & set(val_gt)), "LEAK: overlap between train and val sets!"

    return train_gt, val_gt


# ──────────────────────────────────────────────────────────────────────────────
# Source data loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_source(
    path: Path | None = None,
    source: str = "1",
    chunk_size: int = 50_000,
) -> "pd.DataFrame":
    """
    Load a source TSV file into a pandas DataFrame.
    `source` must be '1', '2', or '3'.
    """
    import pandas as pd
    from src.config import TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3

    if path is None:
        mapping = {"1": TRAIN_SOURCE1, "2": TRAIN_SOURCE2, "3": TRAIN_SOURCE3}
        if source not in mapping:
            raise ValueError(f"source must be '1', '2', or '3', got {source!r}")
        path = mapping[source]

    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def load_source_index(
    path: Path | None = None,
    source: str = "1",
) -> Dict[str, dict]:
    """
    Stream-load a source file and return an id→record dict for fast lookup.
    Returns {entity_id: {col: val, ...}}.
    """
    import pandas as pd
    from src.config import (TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
                             CHUNK_SIZE)

    if path is None:
        mapping = {"1": TRAIN_SOURCE1, "2": TRAIN_SOURCE2, "3": TRAIN_SOURCE3}
        path = mapping[source]

    index: Dict[str, dict] = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=CHUNK_SIZE):
        for row in chunk.itertuples(index=False):
            index[row.entity_id] = row._asdict()
    return index


# ──────────────────────────────────────────────────────────────────────────────
# Metadata persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_split_metadata(
    train_gt: Dict[str, List[str]],
    val_gt: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> None:
    """
    Persist:
    - reports/validation_split_ids.json   — S1 IDs in each split
    - reports/validation_metadata.json    — summary statistics

    Also saves lightweight ground truth references (S1 → match list) for both
    splits to parquet (if pyarrow available) else JSON.
    """
    import json as _json
    from src.config import (VAL_SPLIT_IDS_JSON, VAL_METADATA_JSON,
                             VAL_GT_PARQUET, TRAIN_GT_PARQUET, REPORTS_DIR)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── split IDs ──
    split_ids = {
        "train_s1_ids": sorted(train_gt.keys()),
        "val_s1_ids": sorted(val_gt.keys()),
    }
    with open(VAL_SPLIT_IDS_JSON, "w", encoding="utf-8") as f:
        _json.dump(split_ids, f, indent=2)

    # ── metadata ──
    def dist(gt_part):
        from collections import Counter
        c = Counter(_cardinality_bucket(len(ms)) for ms in gt_part.values())
        return dict(sorted(c.items()))

    total_links_train = sum(len(ms) for ms in train_gt.values())
    total_links_val = sum(len(ms) for ms in val_gt.values())
    singleton_train = sum(1 for ms in train_gt.values() if len(ms) == 0)
    singleton_val = sum(1 for ms in val_gt.values() if len(ms) == 0)

    meta = {
        "total_s1": len(ground_truth),
        "train_s1": len(train_gt),
        "val_s1": len(val_gt),
        "val_fraction_actual": round(len(val_gt) / len(ground_truth), 4),
        "train_total_positive_links": total_links_train,
        "val_total_positive_links": total_links_val,
        "train_singleton_count": singleton_train,
        "val_singleton_count": singleton_val,
        "train_singleton_pct": round(100 * singleton_train / len(train_gt), 2),
        "val_singleton_pct": round(100 * singleton_val / len(val_gt), 2),
        "train_distribution": dist(train_gt),
        "val_distribution": dist(val_gt),
        "leakage_check": "PASSED" if not (set(train_gt) & set(val_gt)) else "FAILED",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(VAL_METADATA_JSON, "w", encoding="utf-8") as f:
        _json.dump(meta, f, indent=2)

    # ── try parquet, fall back to JSON ──
    def save_gt_frame(gt_dict: Dict[str, List[str]], out_path: Path) -> None:
        import pandas as pd
        rows = [
            {"source1_entity_id": k,
             "matched_entity_ids": ",".join(v)}
            for k, v in gt_dict.items()
        ]
        df = pd.DataFrame(rows)
        try:
            df.to_parquet(str(out_path), index=False)
        except Exception:
            df.to_csv(str(out_path.with_suffix(".csv")), index=False)

    save_gt_frame(train_gt, TRAIN_GT_PARQUET)
    save_gt_frame(val_gt, VAL_GT_PARQUET)

    print(f"Split metadata saved:")
    print(f"  {VAL_SPLIT_IDS_JSON}")
    print(f"  {VAL_METADATA_JSON}")
    print(f"  {TRAIN_GT_PARQUET}")
    print(f"  {VAL_GT_PARQUET}")

    return meta


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Build and save the train/validation split."""
    print("Loading ground truth …")
    t0 = time.time()
    gt = load_ground_truth()
    print(f"  Loaded {len(gt):,} S1 entities in {time.time()-t0:.1f}s")

    print("\nBuilding leakage-safe split …")
    train_gt, val_gt = build_split(gt)

    n_train, n_val = len(train_gt), len(val_gt)
    print(f"  Train: {n_train:,} entities  Val: {n_val:,} entities")
    print(f"  Overlap: {len(set(train_gt) & set(val_gt))} (should be 0)")

    meta = save_split_metadata(train_gt, val_gt, gt)

    print("\n── Validation metadata ─────────────────────────────────────────")
    for k, v in meta.items():
        print(f"  {k:<35} : {v}")


if __name__ == "__main__":
    main()
