"""
scripts/run_tasks_7_and_8.py
============================
End-to-End Pipeline for:
- Task 4: Candidate Generation (Blocking)
- Task 5: Matching Model Baseline (LightGBM)
- Task 6: Hard Negative Mining & Feature Importance
- Task 7: Threshold Search & Entity-Level Decision Experiments
- Task 8: Final Test Inference & Submission Validation
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from collections import defaultdict

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
from rapidfuzz import fuzz

from src.config import (
    DATASET_ROOT, TRAIN_DIR, TEST_DIR, PROJECT_ROOT,
    TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3, TRAIN_GROUND_TRUTH,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    OUTPUT_DIR, REPORTS_DIR, MODELS_DIR,
    VAL_SPLIT_IDS_JSON, VAL_METADATA_JSON
)
from src.preprocessing import (
    normalize_name, normalize_address, normalize_country,
    tokenize, extract_postal_code, _ADDR_STOPWORDS
)
from src.metrics import macro_precision_recall_f05, singleton_performance

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LEGAL_TOKENS = {
    "llc", "inc", "ltd", "pvt", "limited", "private",
    "corp", "corporation", "co", "company", "llp", "pc", "plc", "lp"
}

def clean_tokens(norm_name_str):
    if not norm_name_str:
        return []
    return [t for t in norm_name_str.split() if t not in LEGAL_TOKENS]

def get_stem_name(norm_name_str):
    toks = clean_tokens(norm_name_str)
    return " ".join(toks) if toks else norm_name_str

def extract_numerics(norm_addr_str):
    if not norm_addr_str:
        return []
    return [t for t in norm_addr_str.split() if t.isdigit() and len(t) >= 3]

FEATURE_NAMES = [
    "name_ratio",
    "name_token_set_ratio",
    "name_token_sort_ratio",
    "name_partial_ratio",
    "name_exact",
    "name_stem_exact",
    "name_jaccard",
    "name_len_diff",
    "addr_ratio",
    "addr_token_set_ratio",
    "addr_exact",
    "addr_jaccard",
    "addr_num_overlap",
    "addr_num_exact",
    "addr_missing_s2",
    "source_is_s2",
]

def extract_pair_features(r1, r2_tuple, r2_eid):
    """
    Compute 16 pairwise features for candidate pair.
    r1: dict of S1 features
    r2_tuple: (country, norm_name, norm_addr)
    r2_eid: target entity id
    """
    feat = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    
    country2, n2, a2 = r2_tuple
    n1 = r1["norm_name"]
    s1 = r1["stem_name"]
    s2 = get_stem_name(n2)
    
    # Name features
    feat[0] = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[1] = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[2] = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[3] = fuzz.partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[4] = 1.0 if n1 and n1 == n2 else 0.0
    feat[5] = 1.0 if s1 and s1 == s2 else 0.0
    
    t1 = set(r1["tokens"])
    t2 = set(clean_tokens(n2))
    u_name = len(t1 | t2)
    feat[6] = len(t1 & t2) / u_name if u_name > 0 else 0.0
    
    max_len = max(len(n1), len(n2))
    feat[7] = abs(len(n1) - len(n2)) / max_len if max_len > 0 else 0.0
    
    # Address features
    a1 = r1["norm_addr"]
    has_a1 = bool(a1)
    has_a2 = bool(a2)
    
    feat[14] = 0.0 if has_a2 else 1.0  # addr_missing_s2
    
    if has_a1 and has_a2:
        feat[8] = fuzz.ratio(a1, a2) / 100.0
        feat[9] = fuzz.token_set_ratio(a1, a2) / 100.0
        feat[10] = 1.0 if a1 == a2 else 0.0
        
        at1 = set(a1.split())
        at2 = set(a2.split())
        u_addr = len(at1 | at2)
        feat[11] = len(at1 & at2) / u_addr if u_addr > 0 else 0.0
        
        num1 = set(r1["nums"])
        num2 = set(extract_numerics(a2))
        if num1 and num2:
            feat[12] = len(num1 & num2) / len(num1 | num2)
            feat[13] = 1.0 if num1 == num2 else 0.0
    
    feat[15] = 1.0 if r2_eid.startswith("S2-") else 0.0
    return feat


class TargetInvertedIndex:
    """Memory-efficient inverted index over S2 and S3 target records partitioned by country."""
    def __init__(self):
        self.idx_norm_name = defaultdict(list)
        self.idx_stem_name = defaultdict(list)
        self.idx_token_pair = defaultdict(list)
        self.idx_single_tok = defaultdict(list)
        self.idx_norm_addr = defaultdict(list)
        self.idx_addr_num = defaultdict(list)
        self.records = {}  # entity_id -> (country, norm_name, norm_addr)
        
    def add_records(self, df):
        """Add records from dataframe in a memory-compact manner using fast itertuples."""
        for r in df.itertuples(index=False):
            eid = r.entity_id
            country = normalize_country(str(r.country or ""))
            raw_name = str(r.business_name or "")
            raw_addr = str(r.business_address or "")
            
            nn = normalize_name(raw_name)
            na = normalize_address(raw_addr)
            sn = get_stem_name(nn)
            toks = clean_tokens(nn)
            nums = extract_numerics(na)
            
            # Store compact tuple of strings: takes only ~60 bytes in RAM
            self.records[eid] = (country, nn, na)
            
            if nn:
                self.idx_norm_name[(country, nn)].append(eid)
            if sn:
                self.idx_stem_name[(country, sn)].append(eid)
            if len(toks) >= 2:
                self.idx_token_pair[(country, toks[0], toks[1])].append(eid)
            elif len(toks) == 1 and len(toks[0]) >= 4:
                self.idx_single_tok[(country, toks[0])].append(eid)
            if na:
                self.idx_norm_addr[(country, na)].append(eid)
                if nums and toks:
                    self.idx_addr_num[(country, nums[0], toks[0])].append(eid)

    def query(self, s1_rec, max_cands=60):
        """Retrieve candidate IDs for an S1 entity."""
        country = s1_rec["country"]
        nn = s1_rec["norm_name"]
        sn = s1_rec["stem_name"]
        toks = s1_rec["tokens"]
        na = s1_rec["norm_addr"]
        nums = s1_rec["nums"]
        
        cands = set()
        
        # 1. Exact norm name
        if nn:
            for eid in self.idx_norm_name.get((country, nn), []):
                cands.add(eid)
                if len(cands) >= max_cands:
                    return list(cands)
                    
        # 2. Stem name
        if sn:
            for eid in self.idx_stem_name.get((country, sn), []):
                cands.add(eid)
                if len(cands) >= max_cands:
                    return list(cands)
                    
        # 3. Token pair
        if len(toks) >= 2:
            for eid in self.idx_token_pair.get((country, toks[0], toks[1]), [])[:max_cands]:
                cands.add(eid)
                if len(cands) >= max_cands:
                    return list(cands)
        elif len(toks) == 1 and len(toks[0]) >= 4:
            for eid in self.idx_single_tok.get((country, toks[0]), [])[:max_cands]:
                cands.add(eid)
                if len(cands) >= max_cands:
                    return list(cands)
                    
        # 4. Exact address
        if na:
            for eid in self.idx_norm_addr.get((country, na), [])[:30]:
                cands.add(eid)
                if len(cands) >= max_cands:
                    return list(cands)
                    
        # 5. Address number + first name token
        if nums and toks:
            for eid in self.idx_addr_num.get((country, nums[0], toks[0]), [])[:30]:
                cands.add(eid)
                if len(cands) >= max_cands:
                    return list(cands)
                    
        return list(cands)


def main():
    t_start = time.time()
    
    print("\n" + "=" * 80)
    print("STEP 1: PREPARING TRAINING & VALIDATION DATASETS", flush=True)
    print("=" * 80)
    
    # Load split IDs
    with open(VAL_SPLIT_IDS_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    train_s1_pool = split_data["train_s1_ids"]
    val_s1_pool = split_data["val_s1_ids"]
    
    # Sample 40,000 train S1 and 20,000 val S1
    np.random.seed(42)
    train_sample_ids = set(np.random.choice(train_s1_pool, size=min(40000, len(train_s1_pool)), replace=False))
    val_sample_ids = set(np.random.choice(val_s1_pool, size=min(20000, len(val_s1_pool)), replace=False))
    
    print(f"Sampled {len(train_sample_ids):,} train S1 and {len(val_sample_ids):,} val S1 entities.", flush=True)
    
    # Fast load ground truth with DuckDB
    t0 = time.time()
    gt_df = duckdb.query(f"""
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('{TRAIN_GROUND_TRUTH.as_posix()}', delim='\\t', header=true)
    """).df()
    
    train_gt = {}
    val_gt = {}
    train_targets_needed = set()
    val_targets_needed = set()
    
    for s1_id, val in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if pd.isna(val) or val is None or str(val).strip() in ("", "nan", "None"):
            matched = []
        else:
            matched = [m.strip() for m in str(val).split(",") if m.strip() and m.strip().lower() != "nan"]
            
        if s1_id in train_sample_ids:
            train_gt[s1_id] = matched
            train_targets_needed.update(matched)
        elif s1_id in val_sample_ids:
            val_gt[s1_id] = matched
            val_targets_needed.update(matched)
            
    n_train_singletons = sum(1 for v in train_gt.values() if len(v) == 0)
    n_val_singletons = sum(1 for v in val_gt.values() if len(v) == 0)
    print(f"Ground truth loaded ({time.time()-t0:.2f}s):", flush=True)
    print(f"  Train: {len(train_gt):,} entities, {sum(len(v) for v in train_gt.values()):,} positives, {n_train_singletons:,} singletons", flush=True)
    print(f"  Val  : {len(val_gt):,} entities, {sum(len(v) for v in val_gt.values()):,} positives, {n_val_singletons:,} singletons", flush=True)
    
    # Load S1 records
    all_needed_s1 = pd.DataFrame({"entity_id": list(train_sample_ids | val_sample_ids)})
    s1_df = duckdb.query(f"""
        SELECT s.entity_id, s.business_name, s.business_address, s.country
        FROM read_csv('{TRAIN_SOURCE1.as_posix()}', delim='\\t', header=true) s
        JOIN all_needed_s1 n ON s.entity_id = n.entity_id
    """).df()
    
    s1_dict = {}
    for r in s1_df.itertuples(index=False):
        eid = r.entity_id
        country = normalize_country(str(r.country or ""))
        raw_name = str(r.business_name or "")
        raw_addr = str(r.business_address or "")
        nn = normalize_name(raw_name)
        na = normalize_address(raw_addr)
        sn = get_stem_name(nn)
        toks = clean_tokens(nn)
        nums = extract_numerics(na)
        s1_dict[eid] = {
            "entity_id": eid,
            "country": country,
            "norm_name": nn,
            "norm_addr": na,
            "stem_name": sn,
            "tokens": toks,
            "nums": nums,
        }
    print(f"Loaded {len(s1_dict):,} S1 entity records.", flush=True)
    
    # Load target S2/S3 pool: all true matches + 200,000 background entities from S2 & S3
    all_needed_targets = pd.DataFrame({"entity_id": list(train_targets_needed | val_targets_needed)})
    print(f"Loading target pools ({len(all_needed_targets):,} true matches + background samples)...", flush=True)
    
    targets_s2 = duckdb.query(f"""
        SELECT s.entity_id, s.business_name, s.business_address, s.country
        FROM read_csv('{TRAIN_SOURCE2.as_posix()}', delim='\\t', header=true) s
        JOIN all_needed_targets n ON s.entity_id = n.entity_id
        UNION ALL
        (SELECT entity_id, business_name, business_address, country
         FROM read_csv('{TRAIN_SOURCE2.as_posix()}', delim='\\t', header=true)
         LIMIT 150000)
    """).df()
    
    targets_s3 = duckdb.query(f"""
        SELECT s.entity_id, s.business_name, s.business_address, s.country
        FROM read_csv('{TRAIN_SOURCE3.as_posix()}', delim='\\t', header=true) s
        JOIN all_needed_targets n ON s.entity_id = n.entity_id
        UNION ALL
        (SELECT entity_id, business_name, business_address, country
         FROM read_csv('{TRAIN_SOURCE3.as_posix()}', delim='\\t', header=true)
         LIMIT 150000)
    """).df()
    
    all_targets_df = pd.concat([targets_s2, targets_s3]).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    print(f"Total target records pool: {len(all_targets_df):,}", flush=True)
    
    # Build Inverted Index
    print("Building target inverted index...", flush=True)
    t0 = time.time()
    target_index = TargetInvertedIndex()
    target_index.add_records(all_targets_df)
    print(f"Inverted index built in {time.time()-t0:.2f}s.", flush=True)
    
    # Candidate Generation & Pair Extraction for Training
    print("\n" + "=" * 80)
    print("STEP 2: CANDIDATE GENERATION & FEATURE EXTRACTION (TRAIN & VAL)", flush=True)
    print("=" * 80)
    
    def build_pair_dataset(s1_ids, gt_dict, max_cands_per_s1=60, neg_sample_ratio=5):
        X_list = []
        y_list = []
        pair_ids = []
        cands_dict = {}
        
        covered = 0
        total_links = 0
        
        for s1_id in s1_ids:
            s1_rec = s1_dict[s1_id]
            true_matches = set(gt_dict.get(s1_id, []))
            total_links += len(true_matches)
            
            cands = target_index.query(s1_rec, max_cands=max_cands_per_s1)
            cands_set = set(cands)
            cands_dict[s1_id] = cands
            
            # Count covered links
            covered += len(true_matches & cands_set)
            
            # Positives
            pos_in_cands = [m for m in true_matches if m in cands_set]
            for m in pos_in_cands:
                r2 = target_index.records.get(m)
                if r2:
                    feat = extract_pair_features(s1_rec, r2, m)
                    X_list.append(feat)
                    y_list.append(1)
                    pair_ids.append((s1_id, m))
                    
            # Negatives
            negs = [c for c in cands if c not in true_matches]
            n_neg_keep = min(len(negs), max(1, len(pos_in_cands) * neg_sample_ratio))
            for neg_id in negs[:n_neg_keep]:
                r2 = target_index.records.get(neg_id)
                if r2:
                    feat = extract_pair_features(s1_rec, r2, neg_id)
                    X_list.append(feat)
                    y_list.append(0)
                    pair_ids.append((s1_id, neg_id))
                    
        recall = covered / total_links if total_links > 0 else 1.0
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        return X, y, pair_ids, cands_dict, recall
        
    print("Generating train candidate pairs...", flush=True)
    t0 = time.time()
    X_train, y_train, train_pairs, train_cands, train_recall = build_pair_dataset(train_sample_ids, train_gt)
    print(f"Train pairs: {len(X_train):,} (Pos={y_train.sum():,}, Neg={len(y_train)-y_train.sum():,}), Recall={train_recall:.4f} ({time.time()-t0:.2f}s)", flush=True)
    
    print("Generating val candidate pairs...", flush=True)
    t0 = time.time()
    X_val, y_val, val_pairs, val_cands, val_recall = build_pair_dataset(val_sample_ids, val_gt, neg_sample_ratio=20)
    print(f"Val pairs: {len(X_val):,} (Pos={y_val.sum():,}, Neg={len(y_val)-y_val.sum():,}), Recall={val_recall:.4f} ({time.time()-t0:.2f}s)", flush=True)
    
    # Save Task 4 blocking report
    blocking_summary = f"""BLOCKING EXPERIMENTS SUMMARY
============================
Evaluated on {len(val_sample_ids):,} validation S1 entities.
Target Pool: {len(all_targets_df):,} records
Candidate Recall: {val_recall:.4f}
Average Candidates per S1: {len(X_val)/len(val_sample_ids):.2f}
Total Candidates Evaluated: {len(X_val):,}
Reduction Ratio: {1.0 - len(X_val)/(len(val_sample_ids)*len(all_targets_df)):.6f}
"""
    (REPORTS_DIR / "blocking_summary.txt").write_text(blocking_summary, encoding="utf-8")
    print(f"Saved: {REPORTS_DIR / 'blocking_summary.txt'}", flush=True)
    
    # Model Training (Task 5 & 6)
    print("\n" + "=" * 80)
    print("STEP 3: TRAINING LIGHTGBM MATCHING MODEL", flush=True)
    print("=" * 80)
    
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": -1,
        "min_child_samples": 20,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 5,
        "scale_pos_weight": 1.0,  # Unweighted logloss for calibrated probabilities
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }
    
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=FEATURE_NAMES)
    
    t0 = time.time()
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=600,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=40), lgb.log_evaluation(period=100)],
    )
    print(f"Model trained in {time.time()-t0:.2f}s.", flush=True)
    
    # Save model
    model.save_model(str(MODELS_DIR / "lgbm_model.txt"))
    print(f"Saved model to {MODELS_DIR / 'lgbm_model.txt'}", flush=True)
    
    # Feature Importance (Task 6)
    importance = dict(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain")))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 Features by Gain:")
    for fname, gain in sorted_imp[:10]:
        print(f"  {fname:<25}: {gain:>12,.1f}")
        
    # Predict on Validation Pairs
    val_probs = model.predict(X_val)
    
    # Organize validation predictions by S1 ID
    s1_val_pair_scores = defaultdict(list)
    for (s1_id, s23_id), prob in zip(val_pairs, val_probs):
        s1_val_pair_scores[s1_id].append((s23_id, prob))
        
    for s1_id in val_sample_ids:
        if s1_id not in s1_val_pair_scores:
            s1_val_pair_scores[s1_id] = []
            
    # TASK 7 — THRESHOLD AND ENTITY-LEVEL DECISION
    print("\n" + "=" * 80)
    print("STEP 4: TASK 7 — THRESHOLD EVALUATION (0.50 to 0.99)", flush=True)
    print("=" * 80)
    
    threshold_grid = [round(t, 2) for t in np.arange(0.50, 1.00, 0.02)]
    threshold_records = []
    
    val_gt_eval = {s1_id: val_gt.get(s1_id, []) for s1_id in val_sample_ids}
    
    best_f05 = -1.0
    best_thr = 0.50
    
    print(f"{'Thr':<6} {'F0.5':<8} {'Precision':<10} {'Recall':<8} {'SingletonAcc':<13} {'AvgMatches':<11} {'FalseMerges':<11}")
    print("-" * 75)
    
    for thr in threshold_grid:
        preds = {}
        total_pred_matches = 0
        false_merges = 0
        
        for s1_id in val_sample_ids:
            matched = [cid for cid, score in s1_val_pair_scores[s1_id] if score >= thr]
            preds[s1_id] = matched
            total_pred_matches += len(matched)
            
            # False merge check on singletons
            true_matches = val_gt_eval[s1_id]
            if len(true_matches) == 0 and len(matched) > 0:
                false_merges += 1
                
        metrics = macro_precision_recall_f05(preds, val_gt_eval)
        sg_metrics = singleton_performance(preds, val_gt_eval)
        
        f05 = metrics["f05"]
        p = metrics["precision"]
        r = metrics["recall"]
        sg_acc_raw = sg_metrics.get("singleton_accuracy")
        sg_acc = float(sg_acc_raw) if sg_acc_raw is not None else 1.0
        avg_matches = total_pred_matches / len(val_sample_ids)
        
        threshold_records.append({
            "threshold": thr,
            "macro_f05": f05,
            "precision": p,
            "recall": r,
            "singleton_accuracy": sg_acc,
            "average_predicted_matches_per_s1": round(avg_matches, 3),
            "false_merges": false_merges,
        })
        
        print(f"{thr:<6.2f} {f05:<8.4f} {p:<10.4f} {r:<8.4f} {sg_acc:<13.4f} {avg_matches:<11.3f} {false_merges:<11}")
        
        if f05 > best_f05:
            best_f05 = f05
            best_thr = thr
            
    df_thresh = pd.DataFrame(threshold_records)
    thresh_csv = REPORTS_DIR / "threshold_search.csv"
    df_thresh.to_csv(thresh_csv, index=False)
    print(f"\nSaved threshold search results to: {thresh_csv}")
    print(f"Optimal Threshold: {best_thr} (Macro F0.5 = {best_f05:.4f})")
    
    # TASK 7 — ENTITY-LEVEL DECISION LOGIC
    print("\n" + "=" * 80)
    print("STEP 5: TASK 7 — INVESTIGATING ENTITY-LEVEL DECISION LOGIC", flush=True)
    print("=" * 80)
    
    entity_decision_experiments = []
    
    def get_sg_acc(sg_dict):
        raw = sg_dict.get("singleton_accuracy")
        return float(raw) if raw is not None else 1.0
        
    # Baseline: Pure optimal threshold
    preds_base = {s1_id: [cid for cid, s in s1_val_pair_scores[s1_id] if s >= best_thr] for s1_id in val_sample_ids}
    m_base = macro_precision_recall_f05(preds_base, val_gt_eval)
    sg_base = singleton_performance(preds_base, val_gt_eval)
    entity_decision_experiments.append({
        "strategy": f"Pure Threshold ({best_thr})",
        "description": "Base model with calibrated optimal threshold",
        "macro_f05": m_base["f05"],
        "precision": m_base["precision"],
        "recall": m_base["recall"],
        "singleton_accuracy": get_sg_acc(sg_base),
        "params": {"type": "pure_threshold", "threshold": best_thr}
    })
    
    # Experiment 1: Score Gap / Margin Requirement
    for delta in [0.10, 0.15, 0.20]:
        preds_gap = {}
        for s1_id in val_sample_ids:
            candidates = sorted(s1_val_pair_scores[s1_id], key=lambda x: x[1], reverse=True)
            if not candidates or candidates[0][1] < best_thr:
                preds_gap[s1_id] = []
            else:
                top_score = candidates[0][1]
                preds_gap[s1_id] = [cid for cid, s in candidates if s >= best_thr and (top_score - s) <= delta]
        m = macro_precision_recall_f05(preds_gap, val_gt_eval)
        sg = singleton_performance(preds_gap, val_gt_eval)
        entity_decision_experiments.append({
            "strategy": f"Score Gap (thr={best_thr}, delta={delta})",
            "description": f"Allow secondary matches only if within {delta} of top candidate",
            "macro_f05": m["f05"],
            "precision": m["precision"],
            "recall": m["recall"],
            "singleton_accuracy": get_sg_acc(sg),
            "params": {"type": "score_gap", "threshold": best_thr, "delta": delta}
        })
        
    # Experiment 2: Maximum matches cap
    for max_k in [4, 6, 8]:
        preds_cap = {}
        for s1_id in val_sample_ids:
            candidates = sorted(s1_val_pair_scores[s1_id], key=lambda x: x[1], reverse=True)
            preds_cap[s1_id] = [cid for cid, s in candidates[:max_k] if s >= best_thr]
        m = macro_precision_recall_f05(preds_cap, val_gt_eval)
        sg = singleton_performance(preds_cap, val_gt_eval)
        entity_decision_experiments.append({
            "strategy": f"Match Count Cap (max={max_k}, thr={best_thr})",
            "description": f"Cap maximum matches per S1 to top {max_k}",
            "macro_f05": m["f05"],
            "precision": m["precision"],
            "recall": m["recall"],
            "singleton_accuracy": get_sg_acc(sg),
            "params": {"type": "match_cap", "threshold": best_thr, "max_k": max_k}
        })
        
    # Experiment 3: Singleton Confidence Rule
    for bonus in [0.02, 0.05]:
        preds_sing = {}
        for s1_id in val_sample_ids:
            candidates = sorted(s1_val_pair_scores[s1_id], key=lambda x: x[1], reverse=True)
            if not candidates or candidates[0][1] < (best_thr + bonus):
                preds_sing[s1_id] = []
            else:
                preds_sing[s1_id] = [cid for cid, s in candidates if s >= best_thr]
        m = macro_precision_recall_f05(preds_sing, val_gt_eval)
        sg = singleton_performance(preds_sing, val_gt_eval)
        entity_decision_experiments.append({
            "strategy": f"High Singleton Confidence (gate={best_thr+bonus:.2f})",
            "description": f"Require top candidate >= {best_thr+bonus:.2f} to break singleton",
            "macro_f05": m["f05"],
            "precision": m["precision"],
            "recall": m["recall"],
            "singleton_accuracy": get_sg_acc(sg),
            "params": {"type": "singleton_confidence", "threshold": best_thr, "bonus": bonus}
        })
        
    # Print experiments summary
    print(f"\n{'Strategy':<40} {'F0.5':<8} {'Precision':<10} {'Recall':<8} {'SingletonAcc':<13}")
    print("-" * 80)
    for exp in entity_decision_experiments:
        print(f"{exp['strategy']:<40} {exp['macro_f05']:<8.4f} {exp['precision']:<10.4f} {exp['recall']:<8.4f} {exp['singleton_accuracy']:<13.4f}")
        
    df_entity_exp = pd.DataFrame([
        {k: v for k, v in exp.items() if k != "params"} for exp in entity_decision_experiments
    ])
    entity_csv = REPORTS_DIR / "entity_decision_experiments.csv"
    df_entity_exp.to_csv(entity_csv, index=False)
    print(f"\nSaved entity decision experiments to: {entity_csv}")
    
    best_strategy = max(entity_decision_experiments, key=lambda x: x["macro_f05"])
    print(f"Selected Best Configuration: {best_strategy['strategy']} (Macro F0.5 = {best_strategy['macro_f05']:.4f})")
    selected_params = best_strategy["params"]
    
    # TASK 8 — FINAL TEST INFERENCE
    print("\n" + "=" * 80)
    print("STEP 6: TASK 8 — FINAL TEST INFERENCE", flush=True)
    print("=" * 80)
    
    print("Loading test sources...", flush=True)
    t0 = time.time()
    
    # Build Inverted Index on Test S2 & Test S3
    test_index = TargetInvertedIndex()
    
    print("Indexing test_source2.tsv...", flush=True)
    for chunk in pd.read_csv(TEST_SOURCE2, sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
        test_index.add_records(chunk)
        
    print("Indexing test_source3.tsv...", flush=True)
    for chunk in pd.read_csv(TEST_SOURCE3, sep="\t", chunksize=250000, dtype=str, keep_default_na=False):
        test_index.add_records(chunk)
        
    print(f"Indexed {len(test_index.records):,} test S2+S3 records in {time.time()-t0:.2f}s.", flush=True)
    
    # Run Inference on test_source1.tsv
    matching_tsv = OUTPUT_DIR / "matching_results.tsv"
    candidates_tsv = OUTPUT_DIR / "candidate_pairs.tsv"
    
    f_match = open(matching_tsv, "w", encoding="utf-8", newline="")
    f_cand = open(candidates_tsv, "w", encoding="utf-8", newline="")
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    total_test_s1 = 0
    total_with_matches = 0
    total_singletons = 0
    total_links_pred = 0
    total_cands_count = 0
    
    print("Streaming test_source1.tsv and predicting...", flush=True)
    t_inf_start = time.time()
    
    st_type = selected_params["type"]
    thr_val = selected_params["threshold"]
    delta_val = selected_params.get("delta", 0.15)
    max_k_val = selected_params.get("max_k", 8)
    bonus_val = selected_params.get("bonus", 0.05)
    
    for chunk_idx, s1_chunk in enumerate(pd.read_csv(TEST_SOURCE1, sep="\t", chunksize=50000, dtype=str, keep_default_na=False)):
        batch_pairs = []
        batch_pair_indices = []  # (s1_row_idx, s23_id)
        
        # 1. Preprocess chunk and generate candidates
        chunk_s1_records = []
        for r in s1_chunk.itertuples(index=False):
            eid = r.entity_id
            country = normalize_country(str(r.country or ""))
            raw_name = str(r.business_name or "")
            raw_addr = str(r.business_address or "")
            nn = normalize_name(raw_name)
            na = normalize_address(raw_addr)
            sn = get_stem_name(nn)
            toks = clean_tokens(nn)
            nums = extract_numerics(na)
            
            s1_rec = {
                "entity_id": eid,
                "country": country,
                "norm_name": nn,
                "norm_addr": na,
                "stem_name": sn,
                "tokens": toks,
                "nums": nums,
            }
            chunk_s1_records.append(s1_rec)
            
            cands = test_index.query(s1_rec, max_cands=60)
            total_cands_count += len(cands)
            f_cand.write(f"{eid}\t{','.join(cands)}\n")
            
            for cid in cands:
                r2 = test_index.records.get(cid)
                if r2:
                    feat = extract_pair_features(s1_rec, r2, cid)
                    batch_pairs.append(feat)
                    batch_pair_indices.append((len(chunk_s1_records) - 1, cid))
                    
        # 2. Score batch
        s1_scores = defaultdict(list)
        if batch_pairs:
            X_batch = np.array(batch_pairs, dtype=np.float32)
            probs = model.predict(X_batch)
            for (idx, cid), prob in zip(batch_pair_indices, probs):
                s1_scores[idx].append((cid, prob))
                
        # 3. Apply selected entity-level decision logic and write matching_results.tsv
        for idx, s1_rec in enumerate(chunk_s1_records):
            eid = s1_rec["entity_id"]
            candidates = sorted(s1_scores[idx], key=lambda x: x[1], reverse=True)
            
            if st_type == "pure_threshold":
                matched = [cid for cid, s in candidates if s >= thr_val]
            elif st_type == "score_gap":
                if not candidates or candidates[0][1] < thr_val:
                    matched = []
                else:
                    top_score = candidates[0][1]
                    matched = [cid for cid, s in candidates if s >= thr_val and (top_score - s) <= delta_val]
            elif st_type == "match_cap":
                matched = [cid for cid, s in candidates[:max_k_val] if s >= thr_val]
            elif st_type == "singleton_confidence":
                if not candidates or candidates[0][1] < (thr_val + bonus_val):
                    matched = []
                else:
                    matched = [cid for cid, s in candidates if s >= thr_val]
            else:
                matched = [cid for cid, s in candidates if s >= thr_val]
                
            total_test_s1 += 1
            if matched:
                total_with_matches += 1
                total_links_pred += len(matched)
                f_match.write(f"{eid}\t{','.join(matched)}\n")
            else:
                total_singletons += 1
                f_match.write(f"{eid}\t\n")
                
        if (chunk_idx + 1) % 5 == 0:
            elapsed = time.time() - t_inf_start
            rate = total_test_s1 / elapsed
            print(f"  Processed {total_test_s1:,} test S1 entities ({rate:,.0f} rows/s)...", flush=True)
            
    f_match.close()
    f_cand.close()
    
    print(f"\nInference complete in {time.time()-t_inf_start:.2f}s!", flush=True)
    
    # Official Submission Validation
    print("\n" + "=" * 80)
    print("STEP 7: RUNNING OFFICIAL SUBMISSION VALIDATOR", flush=True)
    print("=" * 80)
    
    val_script = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py")
    test_data_dir = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\test")
    
    val_proc = subprocess.run([
        sys.executable,
        str(val_script),
        "--matching", str(matching_tsv),
        "--candidate", str(candidates_tsv),
        "--test-dir", str(test_data_dir)
    ], capture_output=True, text=True)
    
    val_output = val_proc.stdout + "\n" + val_proc.stderr
    print(val_output.strip())
    
    val_status = "PASS" if val_proc.returncode == 0 else "FAIL"
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY REPORT", flush=True)
    print("=" * 80)
    print(f"number of test S1               : {total_test_s1:,}")
    print(f"number with predicted matches   : {total_with_matches:,}")
    print(f"number predicted singleton      : {total_singletons:,}")
    print(f"total predicted links           : {total_links_pred:,}")
    print(f"candidate count                 : {total_cands_count:,}")
    print(f"validator result                : {val_status}")
    print("=" * 80)
    print(f"Total Pipeline Runtime: {(time.time()-t_start)/60:.2f} minutes")

if __name__ == "__main__":
    main()
