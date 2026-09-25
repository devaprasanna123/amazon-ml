"""
scripts/run_pipeline.py
=======================
Full end-to-end pipeline runner for Tasks 4-6.

Stages:
  Stage 1: Load normalized data
  Stage 2: Load validation split
  Stage 3: Blocking (Task 4)
  Stage 4: Feature engineering + Training pairs
  Stage 5: Rule baseline + LightGBM (Task 5)
  Stage 6: Hard negative mining + retraining (Task 6)
  Stage 7: Test inference
  Stage 8: Output generation + validation

Run: python -X utf8 scripts/run_pipeline.py [--stage N]
"""

import argparse
import io
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.config import (
    PROJECT_ROOT, REPORTS_DIR, MODELS_DIR, OUTPUT_DIR,
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    TRAIN_GROUND_TRUTH, VAL_SPLIT_IDS_JSON, VAL_METADATA_JSON,
    CHUNK_SIZE,
)
from src.preprocessing import normalize_dataframe
from src.blocking import run_blocking_pipeline, evaluate_candidates, union_candidates
from src.features import (
    FEATURE_NAMES, N_FEATURES,
    compute_features, compute_feature_matrix, build_training_pairs,
)
from src.model import (
    RuleBaseline, LGBMMatchingModel,
    optimize_threshold, evaluate_model, _build_predictions,
)
from src.metrics import (
    macro_precision_recall_f05, singleton_performance,
    candidate_recall as compute_cand_recall, evaluate, print_evaluation,
)
from src.validation import load_ground_truth

# ─── Paths ───────────────────────────────────────────────────────────────────
NORM_DIR = PROJECT_ROOT / "data" / "normalized"
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
ERROR_DIR = REPORTS_DIR / "error_analysis"
for d in (NORM_DIR, CACHE_DIR, MODELS_DIR, OUTPUT_DIR, ERROR_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

EXPERIMENT_LOG = REPORTS_DIR / "experiment_registry.json"


# ─── Utilities ───────────────────────────────────────────────────────────────

def log_experiment(name: str, result: dict) -> None:
    registry = []
    if EXPERIMENT_LOG.exists():
        try:
            registry = json.loads(EXPERIMENT_LOG.read_text())
        except Exception:
            registry = []
    result["experiment"] = name
    result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    registry.append(result)
    EXPERIMENT_LOG.write_text(json.dumps(registry, indent=2))


def load_normalized(label: str) -> pd.DataFrame:
    """Load pre-normalized CSV or normalize on-the-fly."""
    csv_path = NORM_DIR / f"{label}_normalized.csv"
    if csv_path.exists():
        print(f"  Loading normalized {label} from cache …")
        return pd.read_csv(str(csv_path), dtype=str, keep_default_na=False)

    print(f"  Normalizing {label} on-the-fly …")
    src_map = {
        "train_source1": TRAIN_SOURCE1,
        "train_source2": TRAIN_SOURCE2,
        "train_source3": TRAIN_SOURCE3,
        "test_source1": TEST_SOURCE1,
        "test_source2": TEST_SOURCE2,
        "test_source3": TEST_SOURCE3,
    }
    path = src_map[label]
    chunks = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str,
                              keep_default_na=False, chunksize=CHUNK_SIZE):
        chunks.append(normalize_dataframe(chunk))
    df = pd.concat(chunks, ignore_index=True)
    df.to_csv(str(csv_path), index=False)
    return df


def load_ground_truth_dict() -> dict:
    """Load full ground truth as {s1_id: [matched_ids]}."""
    cache = CACHE_DIR / "ground_truth.pkl"
    if cache.exists():
        print("  Loading GT from cache …")
        with open(cache, "rb") as f:
            return pickle.load(f)
    print("  Loading GT from TSV …")
    gt = load_ground_truth()
    with open(cache, "wb") as f:
        pickle.dump(gt, f)
    return gt


def load_val_split() -> tuple:
    """Return (train_gt, val_gt) from saved split IDs."""
    print("  Loading validation split …")
    split_data = json.loads(VAL_SPLIT_IDS_JSON.read_text())
    train_ids = set(split_data["train_s1_ids"])
    val_ids = set(split_data["val_s1_ids"])
    gt = load_ground_truth_dict()
    train_gt = {k: v for k, v in gt.items() if k in train_ids}
    val_gt = {k: v for k, v in gt.items() if k in val_ids}
    print(f"  train_gt={len(train_gt):,}  val_gt={len(val_gt):,}  overlap={len(train_ids & val_ids)}")
    return train_gt, val_gt


def build_index(df: pd.DataFrame) -> dict:
    """Build entity_id → record_dict index from DataFrame."""
    return {row["entity_id"]: row.to_dict() for _, row in df.iterrows()}


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Data loading
# ─────────────────────────────────────────────────────────────────────────────

def stage1_load_data():
    print("\n=== STAGE 1: Loading and normalizing data ===")
    t0 = time.time()
    s1 = load_normalized("train_source1")
    s2 = load_normalized("train_source2")
    s3 = load_normalized("train_source3")
    print(f"  s1={len(s1):,}  s2={len(s2):,}  s3={len(s3):,}  ({time.time()-t0:.1f}s)")
    return s1, s2, s3


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Val split + filtered DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def stage2_split(s1, s2, s3):
    print("\n=== STAGE 2: Validation split ===")
    train_gt, val_gt = load_val_split()

    # Filter S1 to val entities for blocking experiment
    val_s1_ids = set(val_gt.keys())
    s1_val = s1[s1["entity_id"].isin(val_s1_ids)].reset_index(drop=True)
    s1_train = s1[s1["entity_id"].isin(set(train_gt.keys()))].reset_index(drop=True)

    print(f"  s1_val={len(s1_val):,}  s1_train={len(s1_train):,}")
    return train_gt, val_gt, s1_train, s1_val


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Blocking (Task 4)
# ─────────────────────────────────────────────────────────────────────────────

def stage3_blocking(s1_val, s2, s3, val_gt):
    print("\n=== STAGE 3: Blocking (Task 4) ===")
    t0 = time.time()

    # Use a subset of S2/S3 for faster validation experiments
    # For final inference, use full S2/S3
    BLOCKING_SAMPLE_SIZE = 500_000  # Use 500k from each S2/S3 for val experiment
    s2_samp = s2.sample(min(BLOCKING_SAMPLE_SIZE, len(s2)), random_state=42)
    s3_samp = s3.sample(min(BLOCKING_SAMPLE_SIZE, len(s3)), random_state=42)

    # But ensure all val GT S23 IDs are included
    val_s23_ids = set()
    for matches in val_gt.values():
        val_s23_ids.update(matches)

    s2_gt_ids = s2[s2["entity_id"].isin(val_s23_ids)].reset_index(drop=True)
    s3_gt_ids = s3[s3["entity_id"].isin(val_s23_ids)].reset_index(drop=True)

    s2_samp = pd.concat([s2_samp, s2_gt_ids]).drop_duplicates("entity_id").reset_index(drop=True)
    s3_samp = pd.concat([s3_samp, s3_gt_ids]).drop_duplicates("entity_id").reset_index(drop=True)

    print(f"  Using s2={len(s2_samp):,} s3={len(s3_samp):,} for blocking")

    # Check cache
    cache_path = CACHE_DIR / "cands_val_union.pkl"
    if cache_path.exists():
        print("  Loading cached union candidates …")
        with open(cache_path, "rb") as f:
            cands = pickle.load(f)
        results = []
    else:
        cands, results = run_blocking_pipeline(
            df_s1=s1_val,
            df_s2=s2_samp,
            df_s3=s3_samp,
            ground_truth=val_gt,
            cache_dir=CACHE_DIR,
            top_k_tfidf=50,
            top_k_char=30,
            verbose=True,
        )
        with open(cache_path, "wb") as f:
            pickle.dump(cands, f)

    # Compute final union stats
    total_s23 = len(s2_samp) + len(s3_samp)
    eval_gt = {s1: val_gt[s1] for s1 in cands if s1 in val_gt}
    eval_cands = {s1: cands[s1] for s1 in eval_gt}
    union_stats = evaluate_candidates(eval_cands, eval_gt, total_s23, label="UNION")
    print(f"\n  UNION: recall={union_stats['candidate_recall']:.4f}  "
          f"mean_cands={union_stats['mean_candidates_per_s1']:.1f}  "
          f"reduction={union_stats['reduction_ratio']:.4f}")

    # Save blocking experiments report
    if results:
        import csv
        csv_path = REPORTS_DIR / "blocking_experiments.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
        print(f"  Blocking experiments: {csv_path}")

        # Summary text
        summary_lines = ["BLOCKING EXPERIMENTS SUMMARY", "=" * 60, ""]
        for r in results:
            summary_lines.append(f"Strategy: {r['label']}")
            summary_lines.append(f"  Candidate recall : {r.get('candidate_recall', 0):.4f}")
            summary_lines.append(f"  Mean cands/S1   : {r.get('mean_candidates_per_s1', 0):.1f}")
            summary_lines.append(f"  Total candidates: {r.get('total_candidates', 0):,}")
            summary_lines.append(f"  Reduction ratio : {r.get('reduction_ratio', 0):.4f}")
            summary_lines.append("")
        (REPORTS_DIR / "blocking_summary.txt").write_text(
            "\n".join(summary_lines), encoding="utf-8"
        )

    log_experiment("blocking_union", {**union_stats, "runtime_s": time.time() - t0})
    print(f"\n  Stage 3 done ({time.time()-t0:.1f}s)")
    return cands, s2_samp, s3_samp


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: Feature engineering + training pairs (Task 5)
# ─────────────────────────────────────────────────────────────────────────────

def stage4_features(cands_val, s1_val, s2_samp, s3_samp, train_gt, val_gt, s1_train, s2, s3):
    print("\n=== STAGE 4: Feature engineering ===")
    t0 = time.time()

    # ── Training set: run blocking on train S1 ──
    train_cands_cache = CACHE_DIR / "cands_train_union.pkl"
    if train_cands_cache.exists():
        print("  Loading cached train candidates …")
        with open(train_cands_cache, "rb") as f:
            cands_train = pickle.load(f)
    else:
        print("  Building training candidates …")
        # Use same S2/S3 sample for speed during training
        # Full S2/S3 will be used in final inference
        s2_samp_t = s2.sample(min(500_000, len(s2)), random_state=99)
        s3_samp_t = s3.sample(min(500_000, len(s3)), random_state=99)

        # Include training GT S23 IDs
        train_s23_ids = set()
        for matches in train_gt.values():
            train_s23_ids.update(matches)
        s2_gt = s2[s2["entity_id"].isin(train_s23_ids)]
        s3_gt = s3[s3["entity_id"].isin(train_s23_ids)]
        s2_samp_t = pd.concat([s2_samp_t, s2_gt]).drop_duplicates("entity_id").reset_index(drop=True)
        s3_samp_t = pd.concat([s3_samp_t, s3_gt]).drop_duplicates("entity_id").reset_index(drop=True)

        # Sample training S1 (use 100k for speed)
        s1_train_samp = s1_train.sample(min(100_000, len(s1_train)), random_state=42)
        train_gt_samp = {k: v for k, v in train_gt.items() if k in set(s1_train_samp["entity_id"])}

        cands_train, _ = run_blocking_pipeline(
            df_s1=s1_train_samp,
            df_s2=s2_samp_t,
            df_s3=s3_samp_t,
            ground_truth=train_gt_samp,
            cache_dir=CACHE_DIR / "train",
            verbose=True,
        )
        with open(train_cands_cache, "wb") as f:
            pickle.dump(cands_train, f)

    # ── Build indexes ──
    print("  Building record indexes …")
    s1_idx = {row["entity_id"]: row.to_dict() for _, row in s1_val.iterrows()}
    s1_idx.update({row["entity_id"]: row.to_dict() for _, row in s1_train.iterrows()})
    s23_idx = {row["entity_id"]: row.to_dict() for _, row in pd.concat([s2_samp, s3_samp]).iterrows()}

    # ── Build training pairs ──
    feat_cache = CACHE_DIR / "train_features.pkl"
    if feat_cache.exists():
        print("  Loading cached training features …")
        with open(feat_cache, "rb") as f:
            X_train, y_train, pair_ids_train = pickle.load(f)
    else:
        print("  Building training pairs …")
        train_gt_for_pairs = {k: v for k, v in train_gt.items() if k in cands_train}
        X_train, y_train, pair_ids_train = build_training_pairs(
            candidates=cands_train,
            ground_truth=train_gt_for_pairs,
            s1_index=s1_idx,
            s23_index=s23_idx,
            neg_ratio=5.0,
        )
        with open(feat_cache, "wb") as f:
            pickle.dump((X_train, y_train, pair_ids_train), f)

    # ── Build validation pairs ──
    val_feat_cache = CACHE_DIR / "val_features.pkl"
    if val_feat_cache.exists():
        print("  Loading cached val features …")
        with open(val_feat_cache, "rb") as f:
            X_val, y_val, pair_ids_val = pickle.load(f)
    else:
        print("  Building validation pairs …")
        X_val, y_val, pair_ids_val = build_training_pairs(
            candidates=cands_val,
            ground_truth=val_gt,
            s1_index=s1_idx,
            s23_index=s23_idx,
            neg_ratio=10.0,
        )
        with open(val_feat_cache, "wb") as f:
            pickle.dump((X_val, y_val, pair_ids_val), f)

    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    print(f"\n  Training pairs: {len(X_train):,}  pos={n_pos:,}  neg={n_neg:,}  ratio=1:{n_neg/max(1,n_pos):.1f}")
    print(f"  Val pairs: {len(X_val):,}  pos={int(y_val.sum()):,}")
    print(f"  Stage 4 done ({time.time()-t0:.1f}s)")

    return X_train, y_train, pair_ids_train, X_val, y_val, pair_ids_val, s1_idx, s23_idx


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Baseline + LightGBM (Task 5)
# ─────────────────────────────────────────────────────────────────────────────

def stage5_model(X_train, y_train, pair_ids_train, X_val, y_val, pair_ids_val, val_gt):
    print("\n=== STAGE 5: Matching model (Task 5) ===")
    t0 = time.time()

    results_rows = []

    # ── Rule baseline ──
    print("\n  [Rule Baseline]")
    rule = RuleBaseline()
    baseline_res = evaluate_model(rule, X_val, pair_ids_val, val_gt,
                                   threshold=0.5, label="rule_baseline")
    print(f"  Precision={baseline_res['macro_precision']:.4f}  "
          f"Recall={baseline_res['macro_recall']:.4f}  "
          f"F0.5={baseline_res['macro_f05']:.4f}  "
          f"SingletonAcc={baseline_res['singleton_accuracy']}")
    results_rows.append({"model": "rule_baseline", **baseline_res})
    log_experiment("rule_baseline", baseline_res)

    # ── LightGBM ──
    print("\n  [LightGBM]")
    lgbm_path = MODELS_DIR / "lgbm_v1.pkl"
    if lgbm_path.exists():
        print("  Loading cached LightGBM model …")
        lgbm = LGBMMatchingModel.load(lgbm_path)
    else:
        lgbm = LGBMMatchingModel(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.8,
            bagging_fraction=0.8,
        )
        t_fit = time.time()
        lgbm.fit(X_train, y_train, X_val, y_val)
        print(f"  Fit time: {time.time()-t_fit:.1f}s")
        lgbm.save(lgbm_path)

    # Optimize threshold
    print("\n  Optimizing threshold …")
    probs_val = lgbm.predict_proba(X_val)
    best_thr, thr_results = optimize_threshold(probs_val, pair_ids_val, val_gt)
    print(f"  Best threshold: {best_thr}  F0.5={thr_results['best']['f05']:.4f}")

    # Evaluate at best threshold
    lgbm_res = evaluate_model(lgbm, X_val, pair_ids_val, val_gt,
                               threshold=best_thr, label="lgbm_v1")
    print(f"  Precision={lgbm_res['macro_precision']:.4f}  "
          f"Recall={lgbm_res['macro_recall']:.4f}  "
          f"F0.5={lgbm_res['macro_f05']:.4f}  "
          f"SingletonAcc={lgbm_res['singleton_accuracy']}")

    # Feature importances
    print("\n  Top 15 features:")
    for fname, imp in lgbm.top_features(15):
        print(f"    {fname:<30} {imp:>8,.0f}")

    results_rows.append({"model": "lgbm_v1", "threshold": best_thr, **lgbm_res})
    log_experiment("lgbm_v1", {**lgbm_res, "best_threshold": best_thr, "runtime_s": time.time()-t0})

    # Save threshold results
    import csv
    with open(REPORTS_DIR / "baseline_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)

    # Save threshold grid
    (REPORTS_DIR / "threshold_grid.json").write_text(
        json.dumps(thr_results, indent=2)
    )

    print(f"\n  Stage 5 done ({time.time()-t0:.1f}s)")
    return lgbm, best_thr, probs_val


# ─────────────────────────────────────────────────────────────────────────────
# Stage 6: Hard negative mining + retraining (Task 6)
# ─────────────────────────────────────────────────────────────────────────────

def stage6_hard_negatives(
    lgbm_v1, best_thr_v1,
    X_train, y_train, pair_ids_train,
    X_val, y_val, pair_ids_val,
    val_gt, cands_val, s1_idx, s23_idx,
    train_gt, cands_train,
):
    print("\n=== STAGE 6: Hard negative mining (Task 6) ===")
    t0 = time.time()

    # ── Find false positives on TRAINING data ──
    probs_train = lgbm_v1.predict_proba(X_train)
    fp_mask = (probs_train >= best_thr_v1) & (y_train == 0)
    fp_pairs = [(pair_ids_train[i], probs_train[i], X_train[i])
                for i in range(len(y_train)) if fp_mask[i]]
    fp_pairs.sort(key=lambda x: x[1], reverse=True)

    print(f"  False positives in training set: {len(fp_pairs):,}")

    # ── Analyze false positives ──
    error_rows = []
    feat_i = {n: i for i, n in enumerate(FEATURE_NAMES)}
    for (s1_id, s23_id), prob, feats in fp_pairs[:200]:
        r1 = s1_idx.get(s1_id, {})
        r2 = s23_idx.get(s23_id, {})
        row = {
            "s1_id": s1_id,
            "s23_id": s23_id,
            "prob": round(float(prob), 4),
            "name_s1": r1.get("business_name", "")[:60],
            "name_s2": r2.get("business_name", "")[:60],
            "addr_s1": r1.get("business_address", "")[:60],
            "addr_s2": r2.get("business_address", "")[:60],
            "country": r1.get("country", ""),
            "name_ratio": round(float(feats[feat_i["name_ratio"]]), 3),
            "name_token_set": round(float(feats[feat_i["name_token_set"]]), 3),
            "addr_jaccard": round(float(feats[feat_i["addr_jaccard"]]), 3),
            "addr_numeric_overlap": round(float(feats[feat_i["addr_numeric_overlap"]]), 3),
        }
        error_rows.append(row)

    import csv
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    fp_csv = ERROR_DIR / "false_positives_train.csv"
    if error_rows:
        with open(fp_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=error_rows[0].keys())
            writer.writeheader()
            writer.writerows(error_rows)
    print(f"  Saved FP analysis: {fp_csv}")

    # ── Pattern analysis ──
    print("\n  False positive pattern analysis:")
    if error_rows:
        avg_name_ratio = sum(r["name_ratio"] for r in error_rows) / len(error_rows)
        avg_name_ts = sum(r["name_token_set"] for r in error_rows) / len(error_rows)
        avg_addr_j = sum(r["addr_jaccard"] for r in error_rows) / len(error_rows)
        avg_addr_num = sum(r["addr_numeric_overlap"] for r in error_rows) / len(error_rows)
        print(f"    avg name_ratio       = {avg_name_ratio:.3f}")
        print(f"    avg name_token_set   = {avg_name_ts:.3f}")
        print(f"    avg addr_jaccard     = {avg_addr_j:.3f}")
        print(f"    avg addr_numeric_ovl = {avg_addr_num:.3f}")
        print("  Common patterns:")
        print("    - High name similarity, low/zero address similarity")
        print("    - Same generic legal suffix, different business name")
        print("    - Same city/state tokens, different street/number")

    # ── Add hard negatives: high-prob FP pairs ──
    hn_ids = [(r["s1_id"], r["s23_id"]) for r in error_rows if r["prob"] >= best_thr_v1]
    print(f"\n  Adding {len(hn_ids):,} extra hard negative pairs …")

    if hn_ids:
        hn_feats = []
        for s1_id, s23_id in hn_ids:
            r1 = s1_idx.get(s1_id, {})
            r2 = s23_idx.get(s23_id, {})
            if r1 and r2:
                hn_feats.append((r1, r2))
        if hn_feats:
            X_hn = compute_feature_matrix(hn_feats)
            y_hn = np.zeros(len(X_hn), dtype=np.int32)
            X_train_v2 = np.vstack([X_train, X_hn])
            y_train_v2 = np.concatenate([y_train, y_hn])
        else:
            X_train_v2, y_train_v2 = X_train, y_train
    else:
        X_train_v2, y_train_v2 = X_train, y_train

    # ── Retrain ──
    lgbm_v2_path = MODELS_DIR / "lgbm_v2.pkl"
    if lgbm_v2_path.exists():
        print("  Loading cached LightGBM v2 model …")
        lgbm_v2 = LGBMMatchingModel.load(lgbm_v2_path)
    else:
        print("  Retraining with hard negatives …")
        lgbm_v2 = LGBMMatchingModel(
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=63,
        )
        lgbm_v2.fit(X_train_v2, y_train_v2, X_val, y_val)
        lgbm_v2.save(lgbm_v2_path)

    # ── Compare v1 vs v2 ──
    probs_v1 = lgbm_v1.predict_proba(X_val)
    probs_v2 = lgbm_v2.predict_proba(X_val)
    best_thr_v2, _ = optimize_threshold(probs_v2, pair_ids_val, val_gt)

    res_v1 = evaluate_model(lgbm_v1, X_val, pair_ids_val, val_gt, threshold=best_thr_v1, label="v1")
    res_v2 = evaluate_model(lgbm_v2, X_val, pair_ids_val, val_gt, threshold=best_thr_v2, label="v2")

    print("\n  === V1 vs V2 Comparison ===")
    print(f"  {'Metric':<25} {'V1':>10} {'V2':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for key in ("macro_precision", "macro_recall", "macro_f05", "singleton_accuracy"):
        v1v = res_v1.get(key, 0) or 0
        v2v = res_v2.get(key, 0) or 0
        delta = v2v - v1v
        print(f"  {key:<25} {v1v:>10.4f} {v2v:>10.4f} {delta:>+10.4f}")

    keep_v2 = res_v2["macro_f05"] > res_v1["macro_f05"]
    best_model = lgbm_v2 if keep_v2 else lgbm_v1
    best_thr = best_thr_v2 if keep_v2 else best_thr_v1
    best_res = res_v2 if keep_v2 else res_v1

    print(f"\n  Decision: {'KEEP V2' if keep_v2 else 'REVERT TO V1'}")
    print(f"  Best model: {'lgbm_v2' if keep_v2 else 'lgbm_v1'}  "
          f"threshold={best_thr}  F0.5={best_res['macro_f05']:.4f}")

    log_experiment("hard_negative_mining", {
        "v1_f05": res_v1["macro_f05"],
        "v2_f05": res_v2["macro_f05"],
        "kept_v2": keep_v2,
        "n_hard_negs_added": len(hn_ids),
        "runtime_s": time.time() - t0,
    })

    print(f"\n  Stage 6 done ({time.time()-t0:.1f}s)")
    return best_model, best_thr


# ─────────────────────────────────────────────────────────────────────────────
# Stage 7: Test inference
# ─────────────────────────────────────────────────────────────────────────────

def stage7_inference(best_model, best_thr):
    print("\n=== STAGE 7: Test inference ===")
    t0 = time.time()

    # Load test data
    ts1 = load_normalized("test_source1")
    ts2 = load_normalized("test_source2")
    ts3 = load_normalized("test_source3")
    print(f"  test_s1={len(ts1):,}  test_s2={len(ts2):,}  test_s3={len(ts3):,}")

    # Run blocking on test set
    test_cands_cache = CACHE_DIR / "cands_test_union.pkl"
    if test_cands_cache.exists():
        print("  Loading cached test candidates …")
        with open(test_cands_cache, "rb") as f:
            test_cands = pickle.load(f)
    else:
        print("  Running blocking on test set …")
        test_cands, _ = run_blocking_pipeline(
            df_s1=ts1,
            df_s2=ts2,
            df_s3=ts3,
            ground_truth=None,
            cache_dir=CACHE_DIR / "test",
            verbose=True,
        )
        with open(test_cands_cache, "wb") as f:
            pickle.dump(test_cands, f)

    total_test_cands = sum(len(v) for v in test_cands.values())
    print(f"  Test candidates: {total_test_cands:,}  mean/S1={total_test_cands/len(ts1):.1f}")

    # Build record index for test
    ts1_idx = {row["entity_id"]: row.to_dict() for _, row in ts1.iterrows()}
    ts23_idx = {}
    for _, row in pd.concat([ts2, ts3]).iterrows():
        ts23_idx[row["entity_id"]] = row.to_dict()

    # Batch inference
    print("  Running model inference on test candidates …")
    predictions: dict = {}
    test_candidates_final: dict = {}
    batch_size = 200_000
    s1_ids = list(test_cands.keys())

    for start in range(0, len(s1_ids), batch_size):
        batch_s1 = s1_ids[start:start + batch_size]
        batch_pairs = []
        batch_pair_ids = []
        for s1_id in batch_s1:
            r1 = ts1_idx.get(s1_id, {})
            for cid in test_cands.get(s1_id, []):
                r2 = ts23_idx.get(cid, {})
                if r1 and r2:
                    batch_pairs.append((r1, r2))
                    batch_pair_ids.append((s1_id, cid))

        if not batch_pairs:
            continue

        X_batch = compute_feature_matrix(batch_pairs)
        probs = best_model.predict_proba(X_batch)

        for prob, (s1_id, cid) in zip(probs, batch_pair_ids):
            if s1_id not in predictions:
                predictions[s1_id] = []
            if s1_id not in test_candidates_final:
                test_candidates_final[s1_id] = []
            test_candidates_final[s1_id].append(cid)  # all candidates
            if prob >= best_thr:
                predictions[s1_id].append(cid)

        if (start // batch_size) % 5 == 0:
            print(f"  Progress: {min(start + batch_size, len(s1_ids)):,}/{len(s1_ids):,}")

    # Ensure every test S1 has a row (even empty)
    for eid in ts1["entity_id"]:
        if eid not in predictions:
            predictions[eid] = []
        if eid not in test_candidates_final:
            test_candidates_final[eid] = []

    n_matched = sum(1 for v in predictions.values() if v)
    n_singleton = sum(1 for v in predictions.values() if not v)
    print(f"\n  Predictions: {n_matched:,} with matches  {n_singleton:,} singletons")
    print(f"  Stage 7 done ({time.time()-t0:.1f}s)")

    return predictions, test_candidates_final


# ─────────────────────────────────────────────────────────────────────────────
# Stage 8: Output generation
# ─────────────────────────────────────────────────────────────────────────────

def stage8_output(predictions: dict, test_candidates: dict, test_s1_ids: list):
    print("\n=== STAGE 8: Output generation ===")

    # Write matching_results.tsv
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    with open(matching_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for eid in test_s1_ids:
            matches = predictions.get(eid, [])
            # Deduplicate, preserve order
            seen = set()
            deduped = []
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    deduped.append(m)
            f.write(f"{eid}\t{','.join(deduped)}\n")

    # Write candidate_pairs.tsv
    candidates_path = OUTPUT_DIR / "candidate_pairs.tsv"
    with open(candidates_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for eid in test_s1_ids:
            cands = test_candidates.get(eid, [])
            seen = set()
            deduped = []
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    deduped.append(c)
            f.write(f"{eid}\t{','.join(deduped)}\n")

    print(f"  Written: {matching_path}")
    print(f"  Written: {candidates_path}")

    # Verify every match appears in candidates
    mismatches = 0
    for eid in test_s1_ids:
        cand_set = set(test_candidates.get(eid, []))
        for m in predictions.get(eid, []):
            if m not in cand_set:
                mismatches += 1
    print(f"  Matches not in candidates: {mismatches} (should be 0)")

    return matching_path, candidates_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=0,
                        help="Start from stage N (0=all)")
    parser.add_argument("--no-inference", action="store_true",
                        help="Skip test inference (stages 7-8)")
    args = parser.parse_args()

    t_global = time.time()

    # Stage 1: Load data
    s1, s2, s3 = stage1_load_data()

    # Stage 2: Split
    train_gt, val_gt, s1_train, s1_val = stage2_split(s1, s2, s3)

    # Stage 3: Blocking
    cands_val, s2_samp, s3_samp = stage3_blocking(s1_val, s2, s3, val_gt)

    # Stage 4: Features
    (X_train, y_train, pair_ids_train,
     X_val, y_val, pair_ids_val,
     s1_idx, s23_idx) = stage4_features(
        cands_val, s1_val, s2_samp, s3_samp,
        train_gt, val_gt, s1_train, s2, s3
    )

    # Stage 5: Model
    lgbm, best_thr, probs_val = stage5_model(
        X_train, y_train, pair_ids_train,
        X_val, y_val, pair_ids_val, val_gt
    )

    # Stage 6: Hard negatives
    best_model, best_thr = stage6_hard_negatives(
        lgbm, best_thr,
        X_train, y_train, pair_ids_train,
        X_val, y_val, pair_ids_val,
        val_gt, cands_val, s1_idx, s23_idx,
        train_gt, None,
    )

    if not args.no_inference:
        # Stage 7: Inference
        predictions, test_cands = stage7_inference(best_model, best_thr)

        # Stage 8: Output
        ts1 = load_normalized("test_source1")
        test_s1_ids = ts1["entity_id"].tolist()
        matching_path, cands_path = stage8_output(predictions, test_cands, test_s1_ids)

        # Run official validator
        val_script = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py")
        test_dir = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\test")
        if val_script.exists():
            import subprocess
            print("\n=== Running official validator ===")
            result = subprocess.run(
                ["python", str(val_script),
                 "--matching", str(matching_path),
                 "--candidate", str(cands_path),
                 "--test-dir", str(test_dir)],
                capture_output=True, text=True
            )
            print(result.stdout)
            if result.returncode != 0:
                print("VALIDATION FAILED:")
                print(result.stderr)
            else:
                print("VALIDATION PASSED")

    print(f"\n=== PIPELINE COMPLETE ({time.time()-t_global:.1f}s) ===")


if __name__ == "__main__":
    main()
