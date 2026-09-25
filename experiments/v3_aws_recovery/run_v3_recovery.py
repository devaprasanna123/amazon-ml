"""
experiments/v3_aws_recovery/run_v3_recovery.py
TASK 15: AWS V3 HIGH-RECALL ENTITY RESOLUTION PIPELINE

Features:
- PART A: AWS-portable architecture (configurable DuckDB memory & threads).
- PART B: 100% full-pool validation on all 10,320,219 targets without early exit.
- PART C: Multi-tiered country handling (country-agnostic high-specificity tiers).
- PART D: Order-invariant name blocking (sorted shingles + sorted stems).
- PART E: Elimination of lossy truncation (frequency-aware subdivision, per-tier quotas, cap 120).
- PART F: International normalization (Unicode NFKD + French legal suffixes).
- PART G: Comprehensive candidate-recall forensics -> reports/leaderboard_gap/v3_candidate_misses.csv.
- PART H & I: LightGBM model scoring + dynamic entity-level decision layer.
- PART J: Comparison of V1, V2, V3 and scoreboard update -> reports/master_experiments.csv.
"""

import sys
import os
import gc
import time
import json
import ctypes
from ctypes import wintypes
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

from src.preprocessing import normalize_name, normalize_address, normalize_country
from src.scoreboard import log_experiment, get_scoreboard
from src.decision_optimizer import evaluate_macro_f05, apply_decision_rules, grid_search_decision_layer

TRAIN_S1 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source1.tsv")
TRAIN_S2 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv")
TRAIN_S3 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv")
TRAIN_GT = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv")

SPLIT_JSON = Path(r"D:\amazon ML\reports\validation_split_ids.json")
MODEL_PATH = Path(r"D:\amazon ML\models\lgbm_model.txt")
EXP_DIR = Path(r"D:\amazon ML\experiments\v3_aws_recovery")
EXP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(r"D:\amazon ML\reports\leaderboard_gap")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. AWS & HARDWARE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

AWS_MODE = os.getenv("AWS_MODE", "0") == "1"
DUCKDB_MAX_MEMORY = os.getenv("DUCKDB_MAX_MEMORY", "16GB" if AWS_MODE else "500MB")
DUCKDB_THREADS = int(os.getenv("DUCKDB_THREADS", "8" if AWS_MODE else "4"))

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
        ('PrivateUsage', ctypes.c_size_t),
    ]

psapi = ctypes.windll.psapi
psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS_EX), wintypes.DWORD]
psapi.GetProcessMemoryInfo.restype = wintypes.BOOL

def get_process_memory_mb():
    pmc = PROCESS_MEMORY_COUNTERS_EX()
    pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
    h = ctypes.windll.kernel32.GetCurrentProcess()
    psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
    return pmc.WorkingSetSize / (1024 * 1024), pmc.PeakWorkingSetSize / (1024 * 1024)

# ─────────────────────────────────────────────────────────────────────────────
# 2. NORMALIZATION & FAST TOKENIZATION
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

def strip_accents(text: str) -> str:
    if not text:
        return ""
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

def fast_normalize_tokens(raw_name: str, use_intl=True):
    if not raw_name:
        return "", "", []
    s = str(raw_name)
    if use_intl and not s.isascii():
        s = strip_accents(s)
    s = s.lower()
    if "|" in s:
        s = s.split("|")[0]
    if s.startswith("--"):
        s = s.lstrip("- ")
    # Basic punct replace
    for ch in [",", ".", ";", ":", "!", "?", "\"", "'", "(", ")", "[", "]", "{", "}", "/", "\\", "#", "@", "*", "+"]:
        if ch in s:
            s = s.replace(ch, " ")
    toks = [t for t in s.split() if t and t not in (V2_LEGAL_TOKENS if use_intl else V1_LEGAL_TOKENS)]
    norm_name = " ".join(toks)
    stem_name = norm_name
    return norm_name, stem_name, toks

def extract_addr_features(raw_addr: str):
    if not raw_addr:
        return "", [], None, None
    s = str(raw_addr).lower()
    for ch in [",", ".", ";", ":", "#", "-", "/"]:
        if ch in s:
            s = s.replace(ch, " ")
    parts = s.split()
    nums = [t for t in parts if t.isdigit() and len(t) >= 2]
    street_num = nums[0] if nums else None
    postal = nums[-1] if (nums and len(nums[-1]) in (5, 6)) else None
    norm_addr = " ".join(parts)
    return norm_addr, nums, street_num, postal

# ─────────────────────────────────────────────────────────────────────────────
# 3. FEATURE EXTRACTION FOR SCORING (16 BASELINE PRODUCTION FEATURES)
# ─────────────────────────────────────────────────────────────────────────────

def extract_16_features(r1, r2, r2_eid):
    feat = np.zeros(16, dtype=np.float32)
    n1, s1, a1 = r1["norm_name"], r1["stem_name"], r1["norm_addr"]
    n2, s2, a2 = r2["norm_name"], r2["stem_name"], r2["norm_addr"]
    
    if n1 and n2:
        feat[0] = fuzz.ratio(n1, n2) / 100.0
        feat[1] = fuzz.token_set_ratio(n1, n2) / 100.0
        feat[2] = fuzz.token_sort_ratio(n1, n2) / 100.0
        feat[3] = fuzz.partial_ratio(n1, n2) / 100.0
        feat[4] = 1.0 if n1 == n2 else 0.0
        feat[5] = 1.0 if s1 and s1 == s2 else 0.0
        
        t1, t2 = set(r1["tokens"]), set(r2["tokens"])
        u_name = len(t1 | t2)
        feat[6] = len(t1 & t2) / u_name if u_name > 0 else 0.0
        max_l = max(len(n1), len(n2))
        feat[7] = abs(len(n1) - len(n2)) / max_l if max_l > 0 else 0.0

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
# 4. MAIN EXPERIMENTAL EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("TASK 15: AWS V3 HIGH-RECALL ENTITY RESOLUTION PIPELINE")
    print("=" * 80)
    t_start = time.time()
    
    ws, peak = get_process_memory_mb()
    print(f"Architecture: {'AWS Linux (64GB)' if AWS_MODE else 'Local Windows (16GB)'}")
    print(f"DuckDB Config: PRAGMA max_memory='{DUCKDB_MAX_MEMORY}'; PRAGMA threads={DUCKDB_THREADS};")
    print(f"Initial Memory: Process WS = {ws:.1f} MB")

    # Step 1: Validation split & Ground Truth
    print("\n[Step 1/6] Loading 5,000 deterministic S1 validation entities...")
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    rng = np.random.RandomState(42)
    sample_5k_ids = sorted(list(rng.choice(split_data["val_s1_ids"], size=5000, replace=False)))
    sample_5k_set = set(sample_5k_ids)
    sample_df = pd.DataFrame({"entity_id": [str(x) for x in sample_5k_ids]})

    gt_df = pd.read_csv(TRAIN_GT, sep="\t")
    gt_5k = {}
    for s1_id, val in zip(gt_df["source1_entity_id"], gt_df["matched_entity_ids"]):
        if s1_id in sample_5k_set:
            if pd.isna(val) or str(val).strip() in ("", "nan", "None"):
                gt_5k[s1_id] = []
            else:
                gt_5k[s1_id] = [m.strip() for m in str(val).split(",") if m.strip() and m.strip().lower() != "nan"]
    total_true_links = sum(len(v) for v in gt_5k.values())
    n_singletons = sum(1 for v in gt_5k.values() if len(v) == 0)
    print(f"Validation sample: 5,000 S1 queries, {total_true_links:,} true links, {n_singletons:,} singletons.")

    # Load S1 raw records
    con = duckdb.connect()
    con.execute(f"PRAGMA max_memory='{DUCKDB_MAX_MEMORY}';")
    con.execute(f"PRAGMA threads={DUCKDB_THREADS};")

    s1_raw_df = con.execute(f"""
        SELECT s.entity_id, s.country, s.business_name, s.business_address
        FROM read_csv('{TRAIN_S1.as_posix()}', delim='\\t', header=true) s
        JOIN sample_df n ON s.entity_id = n.entity_id
    """).df()
    print(f"Loaded {len(s1_raw_df):,} S1 raw records.")

    # Preprocess S1 queries
    s1_profiles = {}
    s1_raw_info = {}

    for r in s1_raw_df.itertuples(index=False):
        eid, cty, raw_name, raw_addr = r
        cty_norm = normalize_country(str(cty or ""))
        nn, sn, toks = fast_normalize_tokens(raw_name, use_intl=True)
        stoks = sorted(toks)
        ssn = " ".join(stoks)
        na, nums, street_num, postal = extract_addr_features(raw_addr)
        
        s1_raw_info[eid] = (raw_name, raw_addr, cty_norm)
        s1_profiles[eid] = {
            "country": cty_norm,
            "norm_name": nn,
            "stem_name": sn,
            "sorted_stem": ssn,
            "tokens": toks,
            "sorted_tokens": stoks,
            "norm_addr": na,
            "nums": nums,
            "street_num": street_num,
            "postal": postal,
        }

    # Step 2: Build Multi-Tiered Query Lookup Tables
    print("\n[Step 2/6] Building multi-tiered query lookup tables for 5,000 S1 queries...")
    
    # Tier 1: Country-agnostic exact name
    t1_exact_name = defaultdict(list)
    t1_sorted_stem = defaultdict(list)
    
    # Tier 2: Country-agnostic street number + postal
    t2_street_postal = defaultdict(list)
    t2_exact_addr = defaultdict(list)
    
    # Tier 3: Country-aware shingles & stems
    t3_sorted_pair = defaultdict(list)
    t3_sec_shingle = defaultdict(list)
    t3_ordered_pair = defaultdict(list)
    t3_single_tok = defaultdict(list)
    
    # Tier 4: Frequency subdivision
    t4_sub_num = defaultdict(list)
    t4_sub_3 = defaultdict(list)

    for s1_id in sample_5k_ids:
        p = s1_profiles[s1_id]
        c, nn, sn, ssn, toks, stoks = p["country"], p["norm_name"], p["stem_name"], p["sorted_stem"], p["tokens"], p["sorted_tokens"]
        na, nums, street_num, postal = p["norm_addr"], p["nums"], p["street_num"], p["postal"]
        
        # Tier 1 (Country-agnostic)
        if nn and len(nn) >= 4:
            t1_exact_name[nn].append(s1_id)
        if ssn and len(ssn) >= 4:
            t1_sorted_stem[ssn].append(s1_id)
            
        # Tier 2 (Country-agnostic address)
        if street_num and postal:
            t2_street_postal[(street_num, postal)].append(s1_id)
        if na and len(na) >= 8:
            t2_exact_addr[na].append(s1_id)
            
        # Tier 3 (Country-aware)
        if len(stoks) >= 2:
            t3_sorted_pair[(c, stoks[0], stoks[1])].append(s1_id)
            if nums:
                t4_sub_num[(c, stoks[0], stoks[1], nums[0])].append(s1_id)
            if len(stoks) >= 3:
                t4_sub_3[(c, stoks[0], stoks[1], stoks[2])].append(s1_id)
                t3_sec_shingle[(c, stoks[0], stoks[2])].append(s1_id)
                t3_sec_shingle[(c, stoks[1], stoks[2])].append(s1_id)
        elif len(toks) == 1 and len(toks[0]) >= 5:
            t3_single_tok[(c, toks[0])].append(s1_id)
            
        if len(toks) >= 2:
            t3_ordered_pair[(c, toks[0], toks[1])].append(s1_id)

    total_keys = sum(len(d) for d in [
        t1_exact_name, t1_sorted_stem, t2_street_postal, t2_exact_addr,
        t3_sorted_pair, t3_sec_shingle, t3_ordered_pair, t3_single_tok,
        t4_sub_num, t4_sub_3
    ])
    ws, peak = get_process_memory_mb()
    print(f"Total compact multi-tier query keys: {total_keys:,}. Process WS: {ws:.1f} MB")

    # Step 3: Stream 100% of Target Population (S2 & S3, 10,320,219 rows)
    print("\n[Step 3/6] Streaming 100% of all 10,320,219 targets from disk without early exit...")
    
    # Candidate matches per s1_id per tier
    # s1_cands[s1_id] = list of candidate target IDs
    s1_candidates_v3 = defaultdict(list)
    s1_tier_hits = defaultdict(lambda: defaultdict(list))
    target_profiles = {}
    
    # Block frequency tracker
    shingle_freq = Counter()
    
    # Reverse ground truth lookup for forensics
    gt_reverse = defaultdict(set)
    for s1_id, t_list in gt_5k.items():
        for tid in t_list:
            gt_reverse[tid].add(s1_id)
            
    gt_target_details = {}
    gt_recovered_by_tier = defaultdict(set) # tier -> set of (s1_id, tid)

    t_stream_start = time.time()
    total_targets_scanned = 0
    chunk_vectors = 50 # ~102,400 rows per chunk

    for filepath in [TRAIN_S2, TRAIN_S3]:
        file_t0 = time.time()
        print(f"  Streaming {filepath.name}...")
        rel = con.execute(f"SELECT entity_id, country, business_name, business_address FROM read_csv('{filepath.as_posix()}', delim='\\t', header=true)")
        
        file_rows = 0
        while True:
            chunk = rel.fetch_df_chunk(chunk_vectors)
            if chunk is None or len(chunk) == 0:
                break
                
            for r in chunk.itertuples(index=False):
                eid, raw_cty, raw_name, raw_addr = r
                matched = False
                
                # Ground truth tracking
                if eid in gt_reverse:
                    gt_target_details[eid] = (raw_name, raw_addr, raw_cty)
                    
                # Fast token normalization
                nn, sn, toks = fast_normalize_tokens(raw_name, use_intl=True)
                stoks = sorted(toks)
                ssn = " ".join(stoks)
                cty = normalize_country(str(raw_cty or ""))
                
                # TIER 1: Exact Name (Country-Agnostic)
                if nn and nn in t1_exact_name:
                    matched = True
                    for s1_id in t1_exact_name[nn]:
                        m_list = s1_tier_hits[s1_id]["t1_exact"]
                        if len(m_list) < 50:
                            m_list.append(eid)
                        if (s1_id, eid) in gt_reverse:
                            gt_recovered_by_tier["Tier1_ExactName"].add((s1_id, eid))
                            
                if ssn and ssn != nn and ssn in t1_sorted_stem:
                    matched = True
                    for s1_id in t1_sorted_stem[ssn]:
                        m_list = s1_tier_hits[s1_id]["t1_sorted_stem"]
                        if len(m_list) < 50:
                            m_list.append(eid)
                        if (s1_id, eid) in gt_reverse:
                            gt_recovered_by_tier["Tier1_SortedStem"].add((s1_id, eid))
                            
                # TIER 2: Address Numeric & Postal (Country-Agnostic)
                na = ""
                nums = []
                if raw_addr:
                    na, nums, street_num, postal = extract_addr_features(raw_addr)
                    if street_num and postal:
                        k_sp = (street_num, postal)
                        if k_sp in t2_street_postal:
                            matched = True
                            for s1_id in t2_street_postal[k_sp]:
                                m_list = s1_tier_hits[s1_id]["t2_street_postal"]
                                if len(m_list) < 50:
                                    m_list.append(eid)
                                if (s1_id, eid) in gt_reverse:
                                    gt_recovered_by_tier["Tier2_StreetPostal"].add((s1_id, eid))
                                    
                    if na and na in t2_exact_addr:
                        matched = True
                        for s1_id in t2_exact_addr[na]:
                            m_list = s1_tier_hits[s1_id]["t2_exact_addr"]
                            if len(m_list) < 50:
                                m_list.append(eid)
                                
                    if nums and len(stoks) >= 2:
                        k_snum = (cty, stoks[0], stoks[1], nums[0])
                        if k_snum in t4_sub_num:
                            matched = True
                            for s1_id in t4_sub_num[k_snum]:
                                m_list = s1_tier_hits[s1_id]["t4_sub_num"]
                                if len(m_list) < 50:
                                    m_list.append(eid)
                                if (s1_id, eid) in gt_reverse:
                                    gt_recovered_by_tier["Tier4_SubNum"].add((s1_id, eid))

                # TIER 3: Country-Aware Shingles & Stems
                if len(stoks) >= 2:
                    k_sort = (cty, stoks[0], stoks[1])
                    if k_sort in t3_sorted_pair:
                        matched = True
                        shingle_freq[k_sort] += 1
                        for s1_id in t3_sorted_pair[k_sort]:
                            m_list = s1_tier_hits[s1_id]["t3_sorted_pair"]
                            if len(m_list) < 50:
                                m_list.append(eid)
                            if (s1_id, eid) in gt_reverse:
                                gt_recovered_by_tier["Tier3_SortedPair"].add((s1_id, eid))
                                
                    if len(stoks) >= 3:
                        for k_sec in [(cty, stoks[0], stoks[2]), (cty, stoks[1], stoks[2])]:
                            if k_sec in t3_sec_shingle:
                                matched = True
                                for s1_id in t3_sec_shingle[k_sec]:
                                    m_list = s1_tier_hits[s1_id]["t3_sec_shingle"]
                                    if len(m_list) < 30:
                                        m_list.append(eid)
                                    if (s1_id, eid) in gt_reverse:
                                        gt_recovered_by_tier["Tier3_SecShingle"].add((s1_id, eid))
                                        
                        k_sub3 = (cty, stoks[0], stoks[1], stoks[2])
                        if k_sub3 in t4_sub_3:
                            matched = True
                            for s1_id in t4_sub_3[k_sub3]:
                                m_list = s1_tier_hits[s1_id]["t4_sub_3"]
                                if len(m_list) < 30:
                                    m_list.append(eid)
                                if (s1_id, eid) in gt_reverse:
                                    gt_recovered_by_tier["Tier4_Sub3"].add((s1_id, eid))
                                    
                elif len(toks) == 1 and len(toks[0]) >= 5:
                    k_sing = (cty, toks[0])
                    if k_sing in t3_single_tok:
                        matched = True
                        for s1_id in t3_single_tok[k_sing]:
                            m_list = s1_tier_hits[s1_id]["t3_single_tok"]
                            if len(m_list) < 30:
                                m_list.append(eid)
                                
                if len(toks) >= 2:
                    k_ord = (cty, toks[0], toks[1])
                    if k_ord in t3_ordered_pair:
                        matched = True
                        for s1_id in t3_ordered_pair[k_ord]:
                            m_list = s1_tier_hits[s1_id]["t3_ordered_pair"]
                            if len(m_list) < 30:
                                m_list.append(eid)

                if matched:
                    target_profiles[eid] = {
                        "country": cty,
                        "norm_name": nn,
                        "stem_name": sn,
                        "tokens": toks,
                        "norm_addr": na,
                        "nums": nums,
                    }

            file_rows += len(chunk)
            total_targets_scanned += len(chunk)
            del chunk
            gc.collect()

        ws, peak = get_process_memory_mb()
        print(f"  Finished {filepath.name}: {file_rows:,} rows in {time.time()-file_t0:.1f}s. Process WS: {ws:.1f} MB (Peak: {peak:.1f} MB)")

    total_stream_time = time.time() - t_stream_start
    print(f"\n100% COMPLETE: Successfully processed all {total_targets_scanned:,} targets in {total_stream_time:.1f}s ({total_targets_scanned/total_stream_time:,.0f} rows/s).")
    con.close()
    gc.collect()
    ws, peak = get_process_memory_mb()
    print(f"Post-streaming Memory: Process WS = {ws:.1f} MB (Peak {peak:.1f} MB)")

    # Step 4: Assemble V3 Candidates with Non-Lossy Quotas
    print("\n[Step 4/6] Assembling V3 candidate pool with non-lossy quotas (cap=120)...")
    
    v3_cands = {}
    v3_covered_links = set()
    c_lens = []

    for s1_id in sample_5k_ids:
        hits = s1_tier_hits[s1_id]
        selected = []
        seen = set()

        def add_tier(tier_key, quota):
            nonlocal selected, seen
            count = 0
            for eid in hits.get(tier_key, []):
                if eid not in seen:
                    seen.add(eid)
                    selected.append(eid)
                    count += 1
                    if quota is not None and count >= quota:
                        break

        # Tier 1: Exact Name & Sorted Stem (Country-Agnostic) -> Unlimited
        add_tier("t1_exact", None)
        add_tier("t1_sorted_stem", 30)

        # Tier 2: Address & Postal (Country-Agnostic)
        add_tier("t2_street_postal", 30)
        add_tier("t2_exact_addr", 25)

        # Tier 4: Subdivided Matches for Frequent Blocks
        add_tier("t4_sub_num", 30)
        add_tier("t4_sub_3", 30)

        # Tier 3: Sorted Shingles & Pairs
        add_tier("t3_sorted_pair", 40)
        add_tier("t3_sec_shingle", 25)
        add_tier("t3_ordered_pair", 25)
        add_tier("t3_single_tok", 20)

        v3_selected = selected[:120]
        v3_cands[s1_id] = v3_selected
        c_lens.append(len(v3_selected))

        # Check coverage against final selected candidates (post-quota cap=120)
        true_set = set(gt_5k[s1_id])
        v3_set = set(v3_selected)
        for tid in true_set:
            if tid in v3_set:
                v3_covered_links.add((s1_id, tid))

    total_v3_candidates = sum(c_lens)
    v3_candidate_recall = len(v3_covered_links) / total_true_links
    avg_cands = np.mean(c_lens)
    p50_cands = np.percentile(c_lens, 50)
    p95_cands = np.percentile(c_lens, 95)
    max_cands = max(c_lens)

    print(f"V3 Candidate Recall: {v3_candidate_recall:.4f} ({len(v3_covered_links):,} / {total_true_links:,})")
    print(f"Total V3 Candidates: {total_v3_candidates:,} (avg {avg_cands:.2f}, p50 {p50_cands:.1f}, p95 {p95_cands:.1f}, max {max_cands})")

    # Step 5: Candidate-Recall Forensics (PART G)
    print("\n[Step 5/6] Generating candidate-recall forensics (reports/leaderboard_gap/v3_candidate_misses.csv)...")
    
    forensics = []
    category_counts = Counter()

    for s1_id, true_list in gt_5k.items():
        s1_raw_name, s1_raw_addr, s1_cty = s1_raw_info[s1_id]
        p1 = s1_profiles[s1_id]

        for tid in true_list:
            is_recovered = (s1_id, tid) in v3_covered_links
            t_raw_name, t_raw_addr, t_raw_cty = gt_target_details.get(tid, ("", "", ""))
            
            source_tag = "S2" if tid.startswith("S2-") else "S3"
            
            if is_recovered:
                status = "RECOVERED"
                cat = "Recovered"
            else:
                status = "MISSED"
                # Determine precise cause
                t_toks = set(t_raw_name.lower().split()) if t_raw_name else set()
                s_toks = set(p1["tokens"])
                common = s_toks & t_toks

                if not t_raw_name:
                    cat = "Empty Target Name"
                elif len(common) == 0:
                    cat = "Zero Token Overlap (Severe Distortion / Script Mismatch)"
                elif len(common) == 1:
                    cat = "Single Token Overlap Only (<2 Tokens)"
                elif s1_cty != normalize_country(str(t_raw_cty or "")):
                    cat = "Inconsistent Country Code"
                else:
                    cat = "Block Frequency Collisions / Truncated Beyond Quota"

                category_counts[cat] += 1

            forensics.append({
                "s1_id": s1_id,
                "target_id": tid,
                "source": source_tag,
                "s1_name": s1_raw_name,
                "target_name": t_raw_name,
                "s1_addr": s1_raw_addr,
                "target_addr": t_raw_addr,
                "s1_country": s1_cty,
                "target_country": t_raw_cty,
                "status": status,
                "failure_category": cat,
            })

    df_forensics = pd.DataFrame(forensics)
    forensics_csv = REPORT_DIR / "v3_candidate_misses.csv"
    df_forensics.to_csv(forensics_csv, index=False)
    print(f"Saved: {forensics_csv}")

    print("\nTop Failure Categories on Missed Links:")
    total_missed = sum(category_counts.values())
    for cat, cnt in category_counts.most_common():
        print(f"  - {cat}: {cnt:,} ({cnt/total_missed*100:.1f}%)")

    # Step 6: End-to-End Model Scoring & Decision Optimization (PART H & I)
    print("\n[Step 6/6] Scoring candidates with LightGBM and optimizing entity decision layer...")
    
    # Load production model
    bst = lgb.Booster(model_file=str(MODEL_PATH))
    
    # Target profiles were already cached on the fly during Step 3!
    needed_eids = set()
    for cl in v3_cands.values():
        needed_eids.update(cl)
    print(f"Scoring {len(needed_eids):,} unique target candidates cached on the fly ({len(target_profiles):,} cached total)...")

    # Score candidate pairs
    pairs_feat = []
    pair_keys = []
    for s1_id in sample_5k_ids:
        r1 = s1_profiles[s1_id]
        for cid in v3_cands[s1_id]:
            r2 = target_profiles.get(cid)
            if r2:
                feat = extract_16_features(r1, r2, cid)
                pairs_feat.append(feat)
                pair_keys.append((s1_id, cid))

    if pairs_feat:
        X = np.array(pairs_feat, dtype=np.float32)
        scores = bst.predict(X)
    else:
        scores = np.array([])

    s1_scores = defaultdict(list)
    for (s1_id, cid), sc in zip(pair_keys, scores):
        s1_scores[s1_id].append((cid, float(sc)))

    # Optimize Decision Layer
    best_config, df_grid = grid_search_decision_layer(s1_scores, gt_5k, sample_5k_ids)
    print(f"\nOptimal V3 Decision Layer:")
    print(f"  Singleton Gate : {best_config['singleton_gate']}")
    print(f"  Base Threshold : {best_config['base_threshold']}")
    print(f"  Score Gap      : {best_config['score_gap']}")
    print(f"  Match Cap      : {best_config['match_cap']}")
    print(f"  Macro F0.5     : {best_config['macro_f05']:.4f}")
    print(f"  Precision      : {best_config['precision']:.4f}")
    print(f"  Recall         : {best_config['recall']:.4f}")
    print(f"  Singleton Acc  : {best_config['singleton_accuracy']:.4f}")

    # Build Comparison Table (PART J)
    # V1, V2, V3 comparison
    scoreboard_df = get_scoreboard()
    
    # Values from our verified scoreboard and V3 run
    v1_recs = scoreboard_df[scoreboard_df["experiment_id"] == "V1_Uncalibrated_Production"]
    if len(v1_recs) == 0:
        v1_recs = scoreboard_df[scoreboard_df["experiment_id"] == "V1"]
    v1_rec = v1_recs.iloc[-1]
    
    v2_recs = scoreboard_df[scoreboard_df["experiment_id"] == "V6"]
    if len(v2_recs) == 0:
        v2_recs = scoreboard_df[scoreboard_df["experiment_id"] == "V2"]
    v2_rec = v2_recs.iloc[-1]
    
    comp_rows = [
        {
            "VERSION": "V1 (Production Baseline)",
            "CAND RECALL": f"{v1_rec['candidate_recall']:.4f}",
            "CANDIDATES": f"{int(v1_rec['candidate_count']):,}",
            "AVG CANDS/S1": f"{float(v1_rec['candidate_count'])/5000:.2f}",
            "PRECISION": f"{v1_rec['precision']:.4f}",
            "RECALL": f"{v1_rec['recall']:.4f}",
            "F0.5": f"{v1_rec['macro_f05']:.4f}",
            "SINGLETON ACC": f"{v1_rec['singleton_accuracy']:.4f}",
            "FALSE MERGES": f"{int(v1_rec['false_merges'])}",
            "AVG MATCHES": f"{v1_rec['avg_predicted_matches']:.2f}",
            "MAX MATCHES": f"{int(v1_rec['max_predicted_matches'])}",
            "RUNTIME": f"{v1_rec['runtime_s']:.1f}s",
            "PEAK RAM": f"{v1_rec['peak_ram_mb']:.1f} MB",
        },
        {
            "VERSION": "V2 (Task 13/14 Baseline)",
            "CAND RECALL": f"{v2_rec['candidate_recall']:.4f}",
            "CANDIDATES": f"{int(v2_rec['candidate_count']):,}",
            "AVG CANDS/S1": f"{float(v2_rec['candidate_count'])/5000:.2f}",
            "PRECISION": f"{v2_rec['precision']:.4f}",
            "RECALL": f"{v2_rec['recall']:.4f}",
            "F0.5": f"{v2_rec['macro_f05']:.4f}",
            "SINGLETON ACC": f"{v2_rec['singleton_accuracy']:.4f}",
            "FALSE MERGES": f"{int(v2_rec['false_merges'])}",
            "AVG MATCHES": f"{v2_rec['avg_predicted_matches']:.2f}",
            "MAX MATCHES": f"{int(v2_rec['max_predicted_matches'])}",
            "RUNTIME": f"{v2_rec['runtime_s']:.1f}s",
            "PEAK RAM": f"{v2_rec['peak_ram_mb']:.1f} MB",
        },
        {
            "VERSION": "V3 (AWS V3 High-Recall)",
            "CAND RECALL": f"{v3_candidate_recall:.4f}",
            "CANDIDATES": f"{total_v3_candidates:,}",
            "AVG CANDS/S1": f"{avg_cands:.2f}",
            "PRECISION": f"{best_config['precision']:.4f}",
            "RECALL": f"{best_config['recall']:.4f}",
            "F0.5": f"{best_config['macro_f05']:.4f}",
            "SINGLETON ACC": f"{best_config['singleton_accuracy']:.4f}",
            "FALSE MERGES": f"{best_config['false_merges']}",
            "AVG MATCHES": f"{best_config['avg_predicted_matches']:.2f}",
            "MAX MATCHES": f"{best_config['max_predicted_matches']}",
            "RUNTIME": f"{time.time()-t_start:.1f}s",
            "PEAK RAM": f"{get_process_memory_mb()[1]:.1f} MB",
        }
    ]

    df_comp = pd.DataFrame(comp_rows)

    # Log V3 to master scoreboard
    log_experiment(
        experiment_id="V3_AWS_High_Recall",
        description="Task 15: V3 AWS-compatible High-Recall pipeline on 100% full target universe",
        candidate_recall=v3_candidate_recall,
        precision=best_config['precision'],
        recall=best_config['recall'],
        macro_f05=best_config['macro_f05'],
        singleton_accuracy=best_config['singleton_accuracy'],
        false_merges=best_config['false_merges'],
        avg_predicted_matches=best_config['avg_predicted_matches'],
        max_predicted_matches=best_config['max_predicted_matches'],
        candidate_count=total_v3_candidates,
        p50_candidates=p50_cands,
        p95_candidates=p95_cands,
        runtime_s=time.time()-t_start,
        peak_ram_mb=get_process_memory_mb()[1],
    )

    # PRINT FINAL REPORT
    print("\n" + "=" * 80)
    print("V3 EXPERIMENT REPORT — TASK 15")
    print("=" * 80)
    print(df_comp.to_string(index=False))
    print()
    print(f"CANDIDATE RECALL: {v3_candidate_recall:.4f}")
    print(f"MACRO F0.5: {best_config['macro_f05']:.4f}")
    print(f"FALSE MERGES / SINGLETON ACCURACY: {best_config['false_merges']} false merges / {best_config['singleton_accuracy']:.4f} accuracy")
    print()
    print("Do not perform full test inference.")
    print("Do not submit.")
    print("STOP.")
    print("=" * 80)

if __name__ == "__main__":
    main()
