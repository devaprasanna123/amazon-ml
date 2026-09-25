"""
experiments/v2_recovery/run_all_v2_experiments.py
Comprehensive experimental execution of Task 12:
- Evaluates V1 (baseline)
- Evaluates V2A (Candidate generation fix only)
- Evaluates V2B (Entity decision fix only)
- Evaluates V2C (Normalization fix only)
- Evaluates V2D (All fixes together)
- Produces comparison table, decision grid, and final report data.
"""

import sys
import os
import time
import json
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

# Ensure unbuffered UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.path.insert(0, r"D:\amazon ML")

import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
from rapidfuzz import fuzz

TRAIN_S1 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source1.tsv")
TRAIN_S2 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv")
TRAIN_S3 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv")
TRAIN_GT = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv")

MODEL_PATH = Path(r"D:\amazon ML\models\lgbm_model.txt")
SPLIT_JSON = Path(r"D:\amazon ML\reports\validation_split_ids.json")
EXP_DIR = Path(r"D:\amazon ML\experiments\v2_recovery")
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. LEGAL TOKENS & NORMALIZATION
# ─────────────────────────────────────────────────────────────────────────────

V1_LEGAL_TOKENS = {
    "llc", "inc", "ltd", "pvt", "limited", "private",
    "corp", "corporation", "co", "company", "llp", "pc", "plc", "lp"
}

FRENCH_LEGAL_TOKENS = {
    "sarl", "sas", "sasu", "sa", "eurl", "snc", "sci", "gie",
    "scop", "selarl", "ei", "micro", "entreprise"
}

INTL_LEGAL_TOKENS = {
    "gmbh", "ag", "bv", "nv", "spa", "srl", "sl", "opc"
}

V2_LEGAL_TOKENS = V1_LEGAL_TOKENS | FRENCH_LEGAL_TOKENS | INTL_LEGAL_TOKENS

from src.preprocessing import normalize_name as v1_normalize_name
from src.preprocessing import normalize_address as v1_normalize_address
from src.preprocessing import normalize_country as v1_normalize_country

def strip_accents(text: str) -> str:
    if not text:
        return ""
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

def clean_tokens(norm_name_str, legal_set):
    if not norm_name_str:
        return []
    return [t for t in norm_name_str.split() if t not in legal_set]

def extract_numerics(norm_addr_str):
    if not norm_addr_str:
        return []
    return [t for t in norm_addr_str.split() if t.isdigit() and len(t) >= 2]

def normalize_entity(raw_name: str, raw_addr: str, country: str, use_v2_norm=True):
    country_norm = v1_normalize_country(str(country or ""))
    legal_set = V2_LEGAL_TOKENS if use_v2_norm else V1_LEGAL_TOKENS
    
    if use_v2_norm:
        name_clean = strip_accents(str(raw_name or "")).lower()
        addr_clean = strip_accents(str(raw_addr or "")).lower()
        nn = v1_normalize_name(name_clean)
        na = v1_normalize_address(addr_clean)
    else:
        nn = v1_normalize_name(str(raw_name or ""))
        na = v1_normalize_address(str(raw_addr or ""))
        
    toks = clean_tokens(nn, legal_set)
    sorted_toks = sorted(toks)
    stem_name = " ".join(toks) if toks else nn
    sorted_stem = " ".join(sorted_toks) if sorted_toks else nn
    nums = extract_numerics(na)
    
    return {
        "country": country_norm,
        "norm_name": nn,
        "norm_addr": na,
        "stem_name": stem_name,
        "sorted_stem": sorted_stem,
        "tokens": toks,
        "sorted_tokens": sorted_toks,
        "nums": nums,
    }

# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "name_ratio", "name_token_set_ratio", "name_token_sort_ratio", "name_partial_ratio",
    "name_exact", "name_stem_exact", "name_jaccard", "name_len_diff",
    "addr_ratio", "addr_token_set_ratio", "addr_exact", "addr_jaccard",
    "addr_num_overlap", "addr_num_exact", "addr_missing_s2", "source_is_s2",
]

def extract_pair_features(r1, r2, r2_eid):
    feat = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    n1, s1, a1 = r1["norm_name"], r1["stem_name"], r1["norm_addr"]
    n2, s2, a2 = r2["norm_name"], r2["stem_name"], r2["norm_addr"]
    
    feat[0] = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[1] = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[2] = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[3] = fuzz.partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[4] = 1.0 if n1 and n1 == n2 else 0.0
    feat[5] = 1.0 if s1 and s1 == s2 else 0.0
    
    t1, t2 = set(r1["tokens"]), set(r2["tokens"])
    u_name = len(t1 | t2)
    feat[6] = len(t1 & t2) / u_name if u_name > 0 else 0.0
    
    max_len = max(len(n1), len(n2))
    feat[7] = abs(len(n1) - len(n2)) / max_len if max_len > 0 else 0.0
    
    has_a1, has_a2 = bool(a1), bool(a2)
    feat[14] = 0.0 if has_a2 else 1.0
    
    if has_a1 and has_a2:
        feat[8] = fuzz.ratio(a1, a2) / 100.0
        feat[9] = fuzz.token_set_ratio(a1, a2) / 100.0
        feat[10] = 1.0 if a1 == a2 else 0.0
        
        at1, at2 = set(a1.split()), set(a2.split())
        u_addr = len(at1 | at2)
        feat[11] = len(at1 & at2) / u_addr if u_addr > 0 else 0.0
        
        num1, num2 = set(r1["nums"]), set(r2["nums"])
        if num1 and num2:
            feat[12] = len(num1 & num2) / len(num1 | num2)
            feat[13] = 1.0 if num1 == num2 else 0.0
            
    feat[15] = 1.0 if r2_eid.startswith("S2-") else 0.0
    return feat

# ─────────────────────────────────────────────────────────────────────────────
# 3. FAST STREAMING INVERTED INDEX BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class FastInvertedIndex:
    def __init__(self, mode="V2"):
        """
        mode:
          'V1': order-sensitive, no frequency subdivision
          'V2': token-order-invariant + sorted stem + frequency-aware subdivision
        """
        self.mode = mode
        self.idx_norm_name = defaultdict(list)
        self.idx_stem_name = defaultdict(list)
        self.idx_sorted_stem = defaultdict(list)
        self.idx_token_pair = defaultdict(list)
        self.idx_token_num = defaultdict(list)
        self.idx_token_3 = defaultdict(list)
        self.idx_single_tok = defaultdict(list)
        self.idx_norm_addr = defaultdict(list)
        self.idx_addr_num = defaultdict(list)
        
    def add_batch(self, batch_records, use_v2_norm):
        for eid, cty, raw_name, raw_addr in batch_records:
            rec = normalize_entity(raw_name, raw_addr, cty, use_v2_norm=use_v2_norm)
            country = rec["country"]
            nn = rec["norm_name"]
            sn = rec["stem_name"]
            ssn = rec["sorted_stem"]
            toks = rec["tokens"]
            stoks = rec["sorted_tokens"]
            na = rec["norm_addr"]
            nums = rec["nums"]
            
            if nn:
                self.idx_norm_name[(country, nn)].append(eid)
            if sn:
                self.idx_stem_name[(country, sn)].append(eid)
                
            if self.mode == "V2":
                if ssn and ssn != sn:
                    self.idx_sorted_stem[(country, ssn)].append(eid)
                    
                if len(stoks) >= 2:
                    k_pair = (country, stoks[0], stoks[1])
                    self.idx_token_pair[k_pair].append(eid)
                    if nums:
                        self.idx_token_num[(country, stoks[0], stoks[1], nums[0])].append(eid)
                    if len(stoks) >= 3:
                        self.idx_token_3[(country, stoks[0], stoks[1], stoks[2])].append(eid)
                        self.idx_token_pair[(country, stoks[0], stoks[2])].append(eid)
                        self.idx_token_pair[(country, stoks[1], stoks[2])].append(eid)
                elif len(stoks) == 1 and len(stoks[0]) >= 4:
                    self.idx_single_tok[(country, stoks[0])].append(eid)
            else:
                if len(toks) >= 2:
                    self.idx_token_pair[(country, toks[0], toks[1])].append(eid)
                elif len(toks) == 1 and len(toks[0]) >= 4:
                    self.idx_single_tok[(country, toks[0])].append(eid)
                    
            if na:
                self.idx_norm_addr[(country, na)].append(eid)
                if nums and (stoks if self.mode == "V2" else toks):
                    first_t = stoks[0] if self.mode == "V2" else toks[0]
                    self.idx_addr_num[(country, nums[0], first_t)].append(eid)

    def query(self, s1_rec, max_cands=60, freq_threshold=50):
        country = s1_rec["country"]
        nn = s1_rec["norm_name"]
        sn = s1_rec["stem_name"]
        ssn = s1_rec["sorted_stem"]
        toks = s1_rec["tokens"]
        stoks = s1_rec["sorted_tokens"]
        na = s1_rec["norm_addr"]
        nums = s1_rec["nums"]
        
        cands = set()
        cand_sources = defaultdict(set)
        
        # 1. Exact norm name
        if nn:
            for eid in self.idx_norm_name.get((country, nn), []):
                cands.add(eid)
                cand_sources[eid].add("exact_name")
                if len(cands) >= max_cands:
                    return list(cands), cand_sources
                    
        # 2. Exact sorted stem (Order-invariant full match)
        if self.mode == "V2" and ssn:
            for eid in self.idx_sorted_stem.get((country, ssn), []):
                cands.add(eid)
                cand_sources[eid].add("sorted_stem")
                if len(cands) >= max_cands:
                    return list(cands), cand_sources
                    
        # 3. Stem name
        if sn:
            for eid in self.idx_stem_name.get((country, sn), []):
                cands.add(eid)
                cand_sources[eid].add("stem_name")
                if len(cands) >= max_cands:
                    return list(cands), cand_sources

        # 4. Token pairs / Shingles
        if self.mode == "V2":
            if len(stoks) >= 2:
                k_pair = (country, stoks[0], stoks[1])
                bucket = self.idx_token_pair.get(k_pair, [])
                
                if len(bucket) <= freq_threshold:
                    for eid in bucket:
                        cands.add(eid)
                        cand_sources[eid].add("sorted_shingle")
                        if len(cands) >= max_cands:
                            return list(cands), cand_sources
                else:
                    # Frequency-aware subdivision!
                    if nums:
                        k_num = (country, stoks[0], stoks[1], nums[0])
                        sub_bucket = self.idx_token_num.get(k_num, [])
                        for eid in sub_bucket[:30]:
                            cands.add(eid)
                            cand_sources[eid].add("shingle_addr_num")
                            if len(cands) >= max_cands:
                                return list(cands), cand_sources
                    if len(stoks) >= 3:
                        k_3 = (country, stoks[0], stoks[1], stoks[2])
                        sub_bucket_3 = self.idx_token_3.get(k_3, [])
                        for eid in sub_bucket_3[:30]:
                            cands.add(eid)
                            cand_sources[eid].add("shingle_3tok")
                            if len(cands) >= max_cands:
                                return list(cands), cand_sources
                    # Fallback to top 25
                    for eid in bucket[:25]:
                        cands.add(eid)
                        cand_sources[eid].add("shingle_frequent")
                        if len(cands) >= max_cands:
                            return list(cands), cand_sources
                            
                # Secondary sorted shingles (t0, t2) and (t1, t2)
                if len(stoks) >= 3 and len(cands) < max_cands:
                    for k_sec in [(country, stoks[0], stoks[2]), (country, stoks[1], stoks[2])]:
                        sec_bucket = self.idx_token_pair.get(k_sec, [])
                        if len(sec_bucket) <= freq_threshold:
                            for eid in sec_bucket[:20]:
                                cands.add(eid)
                                cand_sources[eid].add("sec_shingle")
                                if len(cands) >= max_cands:
                                    return list(cands), cand_sources
            elif len(stoks) == 1 and len(stoks[0]) >= 4:
                for eid in self.idx_single_tok.get((country, stoks[0]), [])[:max_cands]:
                    cands.add(eid)
                    cand_sources[eid].add("single_token")
                    if len(cands) >= max_cands:
                        return list(cands), cand_sources
        else:
            # V1 logic
            if len(toks) >= 2:
                for eid in self.idx_token_pair.get((country, toks[0], toks[1]), [])[:max_cands]:
                    cands.add(eid)
                    cand_sources[eid].add("v1_token_pair")
                    if len(cands) >= max_cands:
                        return list(cands), cand_sources
            elif len(toks) == 1 and len(toks[0]) >= 4:
                for eid in self.idx_single_tok.get((country, toks[0]), [])[:max_cands]:
                    cands.add(eid)
                    cand_sources[eid].add("v1_single_tok")
                    if len(cands) >= max_cands:
                        return list(cands), cand_sources
                        
        # 5. Exact address
        if na:
            for eid in self.idx_norm_addr.get((country, na), [])[:30]:
                cands.add(eid)
                cand_sources[eid].add("exact_address")
                if len(cands) >= max_cands:
                    return list(cands), cand_sources
                    
        # 6. Address number + first name token
        first_t = stoks[0] if (self.mode == "V2" and stoks) else (toks[0] if toks else "")
        if nums and first_t:
            for eid in self.idx_addr_num.get((country, nums[0], first_t), [])[:30]:
                cands.add(eid)
                cand_sources[eid].add("addr_num_token")
                if len(cands) >= max_cands:
                    return list(cands), cand_sources
                    
        return list(cands), cand_sources


# ─────────────────────────────────────────────────────────────────────────────
# 4. EVALUATION METRIC
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_predictions(preds, gt):
    per_entity = {}
    for s1_id, true_list in gt.items():
        true_set = set(true_list)
        pred_set = set(preds.get(s1_id, []))
        if len(true_set) == 0:
            if len(pred_set) == 0:
                p, r, f = 1.0, 1.0, 1.0
            else:
                p, r, f = 0.0, 1.0, 0.0
        else:
            if len(pred_set) == 0:
                p, r, f = 0.0, 0.0, 0.0
            else:
                tp = len(pred_set & true_set)
                p = tp / len(pred_set)
                r = tp / len(true_set)
                denom = 0.25 * p + r
                f = (1.25 * p * r / denom) if denom > 0 else 0.0
        per_entity[s1_id] = (p, r, f)
        
    macro_f05 = sum(v[2] for v in per_entity.values()) / len(per_entity)
    macro_p = sum(v[0] for v in per_entity.values()) / len(per_entity)
    macro_r = sum(v[1] for v in per_entity.values()) / len(per_entity)
    
    sing_ids = [s1 for s1, ms in gt.items() if len(ms) == 0]
    sing_correct = sum(1 for s1 in sing_ids if len(preds.get(s1, [])) == 0)
    sing_acc = sing_correct / len(sing_ids) if sing_ids else 1.0
    false_merges = len(sing_ids) - sing_correct
    
    n_links = [len(preds.get(s1, [])) for s1 in gt.keys()]
    avg_links = np.mean(n_links)
    max_links = max(n_links) if n_links else 0
    entities_over_11 = sum(1 for nl in n_links if nl > 11)
    
    return {
        "macro_f05": round(macro_f05, 4),
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "singleton_accuracy": round(sing_acc, 4),
        "false_merges": false_merges,
        "avg_links_per_s1": round(float(avg_links), 4),
        "max_links_per_s1": max_links,
        "entities_over_11": entities_over_11,
    }

def apply_decision_layer(s1_scores, sample_ids, base_thr=0.64, singleton_gate=0.80, score_gap=0.20, match_cap=11):
    preds = {}
    for s1_id in sample_ids:
        cands = sorted(s1_scores.get(s1_id, []), key=lambda x: x[1], reverse=True)
        if not cands or cands[0][1] < singleton_gate:
            preds[s1_id] = []
        else:
            top_score = cands[0][1]
            matched = [cid for cid, s in cands if s >= base_thr and (top_score - s) <= score_gap]
            if len(matched) > match_cap:
                matched = matched[:match_cap]
            preds[s1_id] = matched
    return preds


def main():
    print("=" * 80)
    print("TASK 12: LEADERBOARD RECOVERY V2 EXPERIMENTAL RUNNER")
    print("=" * 80)
    t_start = time.time()
    
    # Load 5k deterministic validation sample
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    val_s1_pool = split_data["val_s1_ids"]
    rng = np.random.RandomState(42)
    sample_5k_ids = sorted(list(rng.choice(val_s1_pool, size=5000, replace=False)))
    sample_5k_set = set(sample_5k_ids)
    
    # Ground truth
    gt_df = duckdb.query(f"""
        SELECT source1_entity_id, matched_entity_ids
        FROM read_csv('{TRAIN_GT.as_posix()}', delim='\\t', header=true)
    """).df()
    gt_5k = {}
    for s1_id, val in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if s1_id in sample_5k_set:
            if pd.isna(val) or val is None or str(val).strip() in ("", "nan", "None"):
                gt_5k[s1_id] = []
            else:
                gt_5k[s1_id] = [m.strip() for m in str(val).split(",") if m.strip() and m.strip().lower() != "nan"]
    total_true_links = sum(len(v) for v in gt_5k.values())
    print(f"Loaded ground truth for 5,000 S1 sample: {total_true_links:,} true links, {sum(1 for v in gt_5k.values() if len(v)==0)} singletons.")

    # Load S1 records
    s1_raw_df = duckdb.query(f"""
        SELECT entity_id, business_name, business_address, country
        FROM read_csv('{TRAIN_S1.as_posix()}', delim='\\t', header=true)
        WHERE entity_id IN (SELECT unnest($1))
    """, [sample_5k_ids]).df()
    
    # S1 records preprocessed under V1 and V2
    s1_v1 = {}
    s1_v2 = {}
    for r in s1_raw_df.itertuples(index=False):
        s1_v1[r.entity_id] = normalize_entity(r.business_name, r.business_address, r.country, use_v2_norm=False)
        s1_v2[r.entity_id] = normalize_entity(r.business_name, r.business_address, r.country, use_v2_norm=True)
        
    print(f"Preprocessed {len(s1_v1)} S1 entities for V1 and V2.")
    
    # Load production model
    bst = lgb.Booster(model_file=str(MODEL_PATH))
    
    # ─────────────────────────────────────────────────────────────────────────
    # BUILD FAST TARGET INVERTED INDEXES ON FULL 10.3M TARGETS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BUILDING TARGET INVERTED INDEXES (V1 & V2) ON FULL 10.3M TARGETS...")
    print("=" * 80)
    t0 = time.time()
    
    index_v1 = FastInvertedIndex(mode="V1")
    index_v2 = FastInvertedIndex(mode="V2")
    
    con = duckdb.connect()
    total_target_rows = 0
    chunk_size = 1000000
    
    for filepath in [TRAIN_S2, TRAIN_S3]:
        print(f"Streaming {filepath.name}...")
        rel = con.execute(f"SELECT entity_id, country, business_name, business_address FROM read_csv('{filepath.as_posix()}', delim='\\t', header=true)")
        while True:
            chunk = rel.fetch_df_chunk(chunk_size)
            if chunk is None or len(chunk) == 0:
                break
            records = list(chunk.itertuples(index=False))
            index_v1.add_batch(records, use_v2_norm=False)
            index_v2.add_batch(records, use_v2_norm=True)
            total_target_rows += len(chunk)
            print(f"  Indexed {total_target_rows:,} / 10,320,219 records ({time.time()-t0:.1f}s)...")
            
    print(f"Both indexes built in {(time.time()-t0)/60:.2f} minutes.")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: EVALUATE CANDIDATE GENERATION (V1 vs V2)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 1: BENCHMARKING CANDIDATE GENERATION (V1 vs V2)")
    print("=" * 80)
    
    def run_candidate_gen(index, s1_dict, mode_label):
        t_c = time.time()
        cands_dict = {}
        sources_dict = {}
        total_c = 0
        covered = 0
        c_counts = []
        
        for s1_id in sample_5k_ids:
            s1_rec = s1_dict[s1_id]
            cands, sources = index.query(s1_rec, max_cands=60)
            cands_dict[s1_id] = cands
            sources_dict[s1_id] = sources
            total_c += len(cands)
            c_counts.append(len(cands))
            
            true_set = set(gt_5k[s1_id])
            covered += len(true_set & set(cands))
            
        elapsed = time.time() - t_c
        rec = covered / total_true_links if total_true_links > 0 else 1.0
        avg_c = total_c / len(sample_5k_ids)
        p50_c = np.percentile(c_counts, 50)
        p95_c = np.percentile(c_counts, 95)
        red_ratio = 1.0 - (total_c / (len(sample_5k_ids) * total_target_rows))
        
        print(f"[{mode_label}] Results ({elapsed:.2f}s):")
        print(f"  Candidate Recall    : {rec:.4f} ({covered:,} / {total_true_links:,})")
        print(f"  Total Candidates    : {total_c:,}")
        print(f"  Average Cands/S1    : {avg_c:.2f}")
        print(f"  p50 Cands/S1        : {p50_c:.1f}")
        print(f"  p95 Cands/S1        : {p95_c:.1f}")
        print(f"  Reduction Ratio     : {red_ratio:.8f}")
        
        return {
            "label": mode_label,
            "candidate_recall": rec,
            "total_candidates": total_c,
            "avg_cands": avg_c,
            "p50_cands": p50_c,
            "p95_cands": p95_c,
            "reduction_ratio": red_ratio,
            "runtime_s": elapsed,
            "cands_dict": cands_dict,
            "sources_dict": sources_dict,
        }
        
    cands_v1 = run_candidate_gen(index_v1, s1_v1, "V1 Blocking (Order-Sensitive)")
    cands_v2 = run_candidate_gen(index_v2, s1_v2, "V2 Blocking (Order-Invariant + Freq-Aware)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: EXTRACT TARGET DETAILS & COMPUTE MODEL SCORES FOR RETRIEVED PAIRS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 2: RETRIEVING TARGET TEXT & SCORING CANDIDATE PAIRS")
    print("=" * 80)
    
    # Gather all candidate target IDs from both V1 and V2
    all_needed_eids = set()
    for cs in [cands_v1["cands_dict"], cands_v2["cands_dict"]]:
        for c_list in cs.values():
            all_needed_eids.update(c_list)
            
    print(f"Total unique target records needed across all experiments: {len(all_needed_eids):,} (out of 10.3M)")
    t0 = time.time()
    eid_df = pd.DataFrame({"entity_id": list(all_needed_eids)})
    
    target_records_v1 = {}
    target_records_v2 = {}
    
    for filepath in [TRAIN_S2, TRAIN_S3]:
        df_sub = con.execute(f"""
            SELECT s.entity_id, s.country, s.business_name, s.business_address
            FROM read_csv('{filepath.as_posix()}', delim='\\t', header=true) s
            JOIN eid_df n ON s.entity_id = n.entity_id
        """).df()
        for r in df_sub.itertuples(index=False):
            target_records_v1[r.entity_id] = normalize_entity(r.business_name, r.business_address, r.country, use_v2_norm=False)
            target_records_v2[r.entity_id] = normalize_entity(r.business_name, r.business_address, r.country, use_v2_norm=True)
            
    print(f"Retrieved and normalized {len(target_records_v2):,} target records in {time.time()-t0:.2f}s.")
    
    def score_candidate_dict(cands_dict, s1_dict, target_dict):
        t0 = time.time()
        pairs = []
        keys = []
        for s1_id in sample_5k_ids:
            r1 = s1_dict[s1_id]
            for cid in cands_dict[s1_id]:
                r2 = target_dict.get(cid)
                if r2:
                    feat = extract_pair_features(r1, r2, cid)
                    pairs.append(feat)
                    keys.append((s1_id, cid))
        if pairs:
            X = np.array(pairs, dtype=np.float32)
            scores = bst.predict(X)
        else:
            scores = np.array([])
            
        s1_scores = defaultdict(list)
        for (s1_id, cid), sc in zip(keys, scores):
            s1_scores[s1_id].append((cid, float(sc)))
            
        print(f"  Scored {len(pairs):,} pairs in {time.time()-t0:.2f}s.")
        return s1_scores
        
    print("\nScoring V1 candidate pairs with V1 normalization...")
    scores_v1 = score_candidate_dict(cands_v1["cands_dict"], s1_v1, target_records_v1)
    
    print("\nScoring V2 candidate pairs with V2 normalization...")
    scores_v2 = score_candidate_dict(cands_v2["cands_dict"], s1_v2, target_records_v2)
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: ENTITY DECISION GRID SEARCH (FIX 2) ON V1 CANDIDATES
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 3: FIX 2 — ENTITY DECISION GRID SEARCH ON V1 CANDIDATES")
    print("=" * 80)
    
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    singleton_gates = [0.64, 0.70, 0.75, 0.80, 0.85, 0.90]
    score_gaps = [0.10, 0.15, 0.20, 0.25]
    match_caps = [8, 10, 11]
    
    grid_results = []
    
    for gate in singleton_gates:
        for thr in thresholds:
            if thr > gate:
                continue
            for gap in score_gaps:
                for cap in match_caps:
                    preds = apply_decision_layer(scores_v1, sample_5k_ids, base_thr=thr, singleton_gate=gate, score_gap=gap, match_cap=cap)
                    m = evaluate_predictions(preds, gt_5k)
                    grid_results.append({
                        "singleton_gate": gate,
                        "base_threshold": thr,
                        "score_gap": gap,
                        "match_cap": cap,
                        **m
                    })
                    
    df_grid = pd.DataFrame(grid_results)
    grid_csv = EXP_DIR / "v2_entity_decision_grid.csv"
    df_grid.to_csv(grid_csv, index=False)
    print(f"Evaluated {len(df_grid)} decision configurations. Saved to {grid_csv}")
    
    # Sort by F0.5
    top_decisions = df_grid.sort_values(by="macro_f05", ascending=False).head(10)
    print("\nTop 10 Decision Configurations on V1 Candidates:")
    print(top_decisions[["singleton_gate", "base_threshold", "score_gap", "match_cap", "macro_f05", "precision", "recall", "singleton_accuracy", "avg_links_per_s1", "max_links_per_s1"]].to_string(index=False))
    
    best_decision_row = top_decisions.iloc[0]
    best_gate = best_decision_row["singleton_gate"]
    best_thr = best_decision_row["base_threshold"]
    best_gap = best_decision_row["score_gap"]
    best_cap = int(best_decision_row["match_cap"])
    print(f"\nSelected Optimal Decision Config: Gate={best_gate}, Thr={best_thr}, Gap={best_gap}, Cap={best_cap} (F0.5 = {best_decision_row['macro_f05']:.4f})")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: COMBINED EVALUATION (V1, V2A, V2B, V2C, V2D)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("STEP 4: COMBINED EVALUATION (V1, V2A, V2B, V2C, V2D)")
    print("=" * 80)
    
    comparison_rows = []
    
    # V1: Baseline (V1 blocking, V1 norm, V1 decision: gate=0.64, thr=0.64, gap=0.20, cap=60)
    preds_v1 = apply_decision_layer(scores_v1, sample_5k_ids, base_thr=0.64, singleton_gate=0.64, score_gap=0.20, match_cap=60)
    m_v1 = evaluate_predictions(preds_v1, gt_5k)
    comparison_rows.append({
        "Version": "V1 (Current Production)",
        "Candidate Recall": round(cands_v1["candidate_recall"], 4),
        "Precision": m_v1["precision"],
        "Recall": m_v1["recall"],
        "F0.5": m_v1["macro_f05"],
        "Avg Cands/S1": round(cands_v1["avg_cands"], 2),
        "Singleton Acc": m_v1["singleton_accuracy"],
        "False Merges": m_v1["false_merges"],
        "Max Matches/S1": m_v1["max_links_per_s1"],
    })
    
    # V2A: Candidate generation fix only (V2 blocking, V1 norm, V1 decision)
    # Note: cands_v2 is generated with V2 blocking. Using V1 decision rules.
    preds_v2a = apply_decision_layer(scores_v2, sample_5k_ids, base_thr=0.64, singleton_gate=0.64, score_gap=0.20, match_cap=60)
    m_v2a = evaluate_predictions(preds_v2a, gt_5k)
    comparison_rows.append({
        "Version": "V2A (Candidate Gen Fix Only)",
        "Candidate Recall": round(cands_v2["candidate_recall"], 4),
        "Precision": m_v2a["precision"],
        "Recall": m_v2a["recall"],
        "F0.5": m_v2a["macro_f05"],
        "Avg Cands/S1": round(cands_v2["avg_cands"], 2),
        "Singleton Acc": m_v2a["singleton_accuracy"],
        "False Merges": m_v2a["false_merges"],
        "Max Matches/S1": m_v2a["max_links_per_s1"],
    })
    
    # V2B: Entity decision fix only (V1 blocking, V1 norm, V2 decision: best_gate, best_thr, best_gap, best_cap)
    preds_v2b = apply_decision_layer(scores_v1, sample_5k_ids, base_thr=best_thr, singleton_gate=best_gate, score_gap=best_gap, match_cap=best_cap)
    m_v2b = evaluate_predictions(preds_v2b, gt_5k)
    comparison_rows.append({
        "Version": "V2B (Entity Decision Fix Only)",
        "Candidate Recall": round(cands_v1["candidate_recall"], 4),
        "Precision": m_v2b["precision"],
        "Recall": m_v2b["recall"],
        "F0.5": m_v2b["macro_f05"],
        "Avg Cands/S1": round(cands_v1["avg_cands"], 2),
        "Singleton Acc": m_v2b["singleton_accuracy"],
        "False Merges": m_v2b["false_merges"],
        "Max Matches/S1": m_v2b["max_links_per_s1"],
    })
    
    # V2C: Normalization fix only (V1 blocking with V2 normalization, V1 decision)
    # Using V1 blocking with V2 normalized records:
    cands_v2c = run_candidate_gen(index_v1, s1_v2, "V2C (Norm Fix Only)")
    scores_v2c = score_candidate_dict(cands_v2c["cands_dict"], s1_v2, target_records_v2)
    preds_v2c = apply_decision_layer(scores_v2c, sample_5k_ids, base_thr=0.64, singleton_gate=0.64, score_gap=0.20, match_cap=60)
    m_v2c = evaluate_predictions(preds_v2c, gt_5k)
    comparison_rows.append({
        "Version": "V2C (Normalization Fix Only)",
        "Candidate Recall": round(cands_v2c["candidate_recall"], 4),
        "Precision": m_v2c["precision"],
        "Recall": m_v2c["recall"],
        "F0.5": m_v2c["macro_f05"],
        "Avg Cands/S1": round(cands_v2c["avg_cands"], 2),
        "Singleton Acc": m_v2c["singleton_accuracy"],
        "False Merges": m_v2c["false_merges"],
        "Max Matches/S1": m_v2c["max_links_per_s1"],
    })
    
    # V2D: All fixes together (V2 blocking, V2 norm, V2 decision)
    # Search optimal decision on scores_v2
    best_v2d_f05 = -1
    best_v2d_m = None
    best_v2d_params = None
    
    for gate in [0.75, 0.80, 0.85, 0.90]:
        for thr in [0.60, 0.65, 0.70, 0.75]:
            if thr > gate:
                continue
            for gap in [0.15, 0.20, 0.25]:
                for cap in [8, 10, 11]:
                    preds_tmp = apply_decision_layer(scores_v2, sample_5k_ids, base_thr=thr, singleton_gate=gate, score_gap=gap, match_cap=cap)
                    m_tmp = evaluate_predictions(preds_tmp, gt_5k)
                    if m_tmp["macro_f05"] > best_v2d_f05:
                        best_v2d_f05 = m_tmp["macro_f05"]
                        best_v2d_m = m_tmp
                        best_v2d_params = (gate, thr, gap, cap)
                        
    comparison_rows.append({
        "Version": "V2D (All Fixes Together)",
        "Candidate Recall": round(cands_v2["candidate_recall"], 4),
        "Precision": best_v2d_m["precision"],
        "Recall": best_v2d_m["recall"],
        "F0.5": best_v2d_m["macro_f05"],
        "Avg Cands/S1": round(cands_v2["avg_cands"], 2),
        "Singleton Acc": best_v2d_m["singleton_accuracy"],
        "False Merges": best_v2d_m["false_merges"],
        "Max Matches/S1": best_v2d_m["max_links_per_s1"],
    })
    
    df_comp = pd.DataFrame(comparison_rows)
    comp_csv = EXP_DIR / "v2_comparison_table.csv"
    df_comp.to_csv(comp_csv, index=False)
    print("\n" + "=" * 80)
    print("V2 RECOVERY COMPARISON TABLE:")
    print("=" * 80)
    print(df_comp.to_string(index=False))
    print(f"\nBest V2D Parameters: Singleton Gate={best_v2d_params[0]}, Base Thr={best_v2d_params[1]}, Score Gap={best_v2d_params[2]}, Match Cap={best_v2d_params[3]}")
    
    # Save experiment summary JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sample_size": len(sample_5k_ids),
        "target_pool_size": total_target_rows,
        "v1_baseline": comparison_rows[0],
        "v2a_cand_gen": comparison_rows[1],
        "v2b_entity_decision": comparison_rows[2],
        "v2c_normalization": comparison_rows[3],
        "v2d_all_fixes": comparison_rows[4],
        "best_v2d_params": {
            "singleton_gate": best_v2d_params[0],
            "base_threshold": best_v2d_params[1],
            "score_gap": best_v2d_params[2],
            "match_cap": best_v2d_params[3],
        },
        "delta_f05": round(best_v2d_m["macro_f05"] - m_v1["macro_f05"], 4),
        "delta_candidate_recall": round(cands_v2["candidate_recall"] - cands_v1["candidate_recall"], 4),
    }
    with open(EXP_DIR / "v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print(f"\nSaved summary to: {EXP_DIR / 'v2_summary.json'}")
    print(f"Total Experiment Runtime: {(time.time()-t_start)/60:.2f} minutes.")

if __name__ == "__main__":
    main()
