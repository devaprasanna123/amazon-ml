"""
scratch/pipeline_v2.py
Comprehensive implementation for Task 12:
- FIX 1: Token-Order-Invariant, Frequency-Aware Candidate Generation
- FIX 2: Entity-Level Precision & Singleton Control
- FIX 3: Robust International Text Normalization (Unicode NFKD + French Legal Suffixes)
- Evaluation comparing V1, V2A, V2B, V2C, V2D on the 5k sample against full 10.3M targets.
"""

import sys
import os
import time
import json
import unicodedata
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import duckdb
import lightgbm as lgb
from rapidfuzz import fuzz

sys.path.insert(0, r"D:\amazon ML")

TRAIN_S1 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source1.tsv")
TRAIN_S2 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv")
TRAIN_S3 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv")
TRAIN_GT = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv")

MODEL_PATH = Path(r"D:\amazon ML\models\lgbm_model.txt")
SPLIT_JSON = Path(r"D:\amazon ML\reports\validation_split_ids.json")
EXP_DIR = Path(r"D:\amazon ML\experiments\v2_recovery")
EXP_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LEGAL TOKENS (V1 vs V2)
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


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZATION FUNCTIONS (V1 vs V2)
# ─────────────────────────────────────────────────────────────────────────────

def strip_accents(text: str) -> str:
    """Unicode NFKD decomposition to strip diacritics and ligatures."""
    if not text:
        return ""
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def clean_tokens(norm_name_str, legal_set):
    if not norm_name_str:
        return []
    return [t for t in norm_name_str.split() if t not in legal_set]


def get_stem_name(norm_name_str, legal_set):
    toks = clean_tokens(norm_name_str, legal_set)
    return " ".join(toks) if toks else norm_name_str


def get_sorted_stem(norm_name_str, legal_set):
    toks = sorted(clean_tokens(norm_name_str, legal_set))
    return " ".join(toks) if toks else norm_name_str


def extract_numerics(norm_addr_str):
    if not norm_addr_str:
        return []
    return [t for t in norm_addr_str.split() if t.isdigit() and len(t) >= 2]


from src.preprocessing import normalize_name as v1_normalize_name
from src.preprocessing import normalize_address as v1_normalize_address
from src.preprocessing import normalize_country as v1_normalize_country


def normalize_record_v2(raw_name: str, raw_addr: str, country: str):
    """
    FIX 3: Robust International Normalization.
    Strips accents/diacritics, applies extended legal tokens, produces sorted tokens.
    """
    country_norm = v1_normalize_country(str(country or ""))
    
    # Strip diacritics
    name_clean = strip_accents(str(raw_name or "")).lower()
    addr_clean = strip_accents(str(raw_addr or "")).lower()
    
    nn = v1_normalize_name(name_clean)
    na = v1_normalize_address(addr_clean)
    
    toks = clean_tokens(nn, V2_LEGAL_TOKENS)
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


def normalize_record_v1(raw_name: str, raw_addr: str, country: str):
    """V1 Baseline Normalization."""
    country_norm = v1_normalize_country(str(country or ""))
    nn = v1_normalize_name(str(raw_name or ""))
    na = v1_normalize_address(str(raw_addr or ""))
    toks = clean_tokens(nn, V1_LEGAL_TOKENS)
    sorted_toks = sorted(toks)
    stem_name = " ".join(toks) if toks else nn
    sorted_stem = " ".join(sorted_toks) if sorted_toks else nn
    nums = [t for t in na.split() if t.isdigit() and len(t) >= 3]
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
# FEATURES & SCORING
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "name_ratio", "name_token_set_ratio", "name_token_sort_ratio", "name_partial_ratio",
    "name_exact", "name_stem_exact", "name_jaccard", "name_len_diff",
    "addr_ratio", "addr_token_set_ratio", "addr_exact", "addr_jaccard",
    "addr_num_overlap", "addr_num_exact", "addr_missing_s2", "source_is_s2",
]


def extract_pair_features(r1, r2, r2_eid, legal_set=V1_LEGAL_TOKENS):
    feat = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
    n1 = r1["norm_name"]
    s1 = r1["stem_name"]
    a1 = r1["norm_addr"]
    
    n2 = r2["norm_name"]
    s2 = r2["stem_name"]
    a2 = r2["norm_addr"]
    
    feat[0] = fuzz.ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[1] = fuzz.token_set_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[2] = fuzz.token_sort_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[3] = fuzz.partial_ratio(n1, n2) / 100.0 if n1 and n2 else 0.0
    feat[4] = 1.0 if n1 and n1 == n2 else 0.0
    feat[5] = 1.0 if s1 and s1 == s2 else 0.0
    
    t1 = set(r1["tokens"])
    t2 = set(r2["tokens"])
    u_name = len(t1 | t2)
    feat[6] = len(t1 & t2) / u_name if u_name > 0 else 0.0
    
    max_len = max(len(n1), len(n2))
    feat[7] = abs(len(n1) - len(n2)) / max_len if max_len > 0 else 0.0
    
    has_a1 = bool(a1)
    has_a2 = bool(a2)
    feat[14] = 0.0 if has_a2 else 1.0
    
    if has_a1 and has_a2:
        feat[8] = fuzz.ratio(a1, a2) / 100.0
        feat[9] = fuzz.token_set_ratio(a1, a2) / 100.0
        feat[10] = 1.0 if a1 == a2 else 0.0
        
        at1 = set(a1.split())
        at2 = set(a2.split())
        u_addr = len(at1 | at2)
        feat[11] = len(at1 & at2) / u_addr if u_addr > 0 else 0.0
        
        num1 = set(r1["nums"])
        num2 = set(r2["nums"])
        if num1 and num2:
            feat[12] = len(num1 & num2) / len(num1 | num2)
            feat[13] = 1.0 if num1 == num2 else 0.0
            
    feat[15] = 1.0 if r2_eid.startswith("S2-") else 0.0
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# TARGET INVERTED INDEX V2 (TOKEN-ORDER-INVARIANT & FREQUENCY-AWARE)
# ─────────────────────────────────────────────────────────────────────────────

class TargetInvertedIndexV2:
    def __init__(self, mode="V2"):
        """
        mode:
          'V1': Baseline order-sensitive (toks[0], toks[1]), arbitrary [:60]
          'V2': Order-invariant sorted shingles + frequency-aware subdivision + source tracking
        """
        self.mode = mode
        self.idx_norm_name = defaultdict(list)
        self.idx_stem_name = defaultdict(list)
        self.idx_sorted_stem = defaultdict(list)
        self.idx_token_pair = defaultdict(list)
        self.idx_token_3 = defaultdict(list)
        self.idx_token_num = defaultdict(list)
        self.idx_single_tok = defaultdict(list)
        self.idx_norm_addr = defaultdict(list)
        self.idx_addr_num = defaultdict(list)
        self.records = {}  # eid -> dict of normalized features
        
    def add_record(self, eid, rec):
        self.records[eid] = rec
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
            # Sorted full stem
            if ssn and ssn != sn:
                self.idx_sorted_stem[(country, ssn)].append(eid)
                
            # Token-order-invariant sorted 2-token shingles
            if len(stoks) >= 2:
                # Primary sorted pair
                self.idx_token_pair[(country, stoks[0], stoks[1])].append(eid)
                
                # Numeric subdivision for frequent blocks
                if nums:
                    self.idx_token_num[(country, stoks[0], stoks[1], nums[0])].append(eid)
                    
                # 3-token shingle if available
                if len(stoks) >= 3:
                    self.idx_token_3[(country, stoks[0], stoks[1], stoks[2])].append(eid)
                    self.idx_token_pair[(country, stoks[0], stoks[2])].append(eid)
                    self.idx_token_pair[(country, stoks[1], stoks[2])].append(eid)
            elif len(stoks) == 1 and len(stoks[0]) >= 4:
                self.idx_single_tok[(country, stoks[0])].append(eid)
        else:
            # V1 order-sensitive
            if len(toks) >= 2:
                self.idx_token_pair[(country, toks[0], toks[1])].append(eid)
            elif len(toks) == 1 and len(toks[0]) >= 4:
                self.idx_single_tok[(country, toks[0])].append(eid)
                
        if na:
            self.idx_norm_addr[(country, na)].append(eid)
            if nums and (stoks or toks):
                first_tok = stoks[0] if self.mode == "V2" else toks[0]
                self.idx_addr_num[(country, nums[0], first_tok)].append(eid)

    def query(self, s1_rec, max_cands=60, freq_threshold=60):
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

        # 4. Token pairs / Shingles with Frequency-Aware Control
        if self.mode == "V2":
            if len(stoks) >= 2:
                # Primary sorted pair
                k_pair = (country, stoks[0], stoks[1])
                bucket = self.idx_token_pair.get(k_pair, [])
                
                if len(bucket) <= freq_threshold:
                    # Clean bucket: keep all
                    for eid in bucket:
                        cands.add(eid)
                        cand_sources[eid].add("sorted_shingle")
                        if len(cands) >= max_cands:
                            return list(cands), cand_sources
                else:
                    # FREQUENT BUCKET: Subdivide instead of blindly truncating!
                    # A. Subdivide with address number
                    if nums:
                        k_num = (country, stoks[0], stoks[1], nums[0])
                        sub_bucket = self.idx_token_num.get(k_num, [])
                        for eid in sub_bucket[:30]:
                            cands.add(eid)
                            cand_sources[eid].add("shingle_addr_num")
                            if len(cands) >= max_cands:
                                return list(cands), cand_sources
                                
                    # B. Subdivide with 3rd token
                    if len(stoks) >= 3:
                        k_3 = (country, stoks[0], stoks[1], stoks[2])
                        sub_bucket_3 = self.idx_token_3.get(k_3, [])
                        for eid in sub_bucket_3[:30]:
                            cands.add(eid)
                            cand_sources[eid].add("shingle_3tok")
                            if len(cands) >= max_cands:
                                return list(cands), cand_sources
                                
                    # C. Fallback: take top 25 from the frequent bucket
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
                                cand_sources[eid].add("secondary_shingle")
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
# DECISION LAYER (FIX 2)
# ─────────────────────────────────────────────────────────────────────────────

def apply_decision_layer(s1_scores, sample_ids, base_thr=0.64, singleton_gate=0.80, score_gap=0.20, match_cap=11):
    """
    FIX 2: Configurable entity decision layer.
    Protects singletons via singleton_gate, enforces score gap, and caps maximum matches.
    """
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
        "macro_f05": macro_f05,
        "precision": macro_p,
        "recall": macro_r,
        "singleton_accuracy": sing_acc,
        "false_merges": false_merges,
        "avg_links_per_s1": avg_links,
        "max_links_per_s1": max_links,
        "entities_over_11": entities_over_11,
    }
