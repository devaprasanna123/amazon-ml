"""
experiments/v2_recovery/run_safe_v2_blocking.py
TASK 13: SAFE LOW-MEMORY V2 EXPERIMENT

Evaluates:
- V1: Current production blocking
- V2: Order-invariant blocking (sorted tokens + shingles)
- V3: Frequency-aware blocking (subdivision for blocks > 50)
- V4: No-lossy-truncation blocking (smart quotas, max 120)
- V5: International normalization (Unicode NFKD + French legal forms)
- V6: All combined

Memory safety:
- DuckDB PRAGMA max_memory='1.5GB', threads=4
- Query-driven streaming: S1 query keys matched in single pass over S2/S3
- Process in 100k-row chunks, freeing memory explicitly
- System available RAM monitored continuously (> 1.5GB threshold)
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

# Unbuffered UTF-8 stdout
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.path.insert(0, r"D:\amazon ML")

import numpy as np
import pandas as pd
import duckdb

TRAIN_S1 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source1.tsv")
TRAIN_S2 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source2.tsv")
TRAIN_S3 = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_source3.tsv")
TRAIN_GT = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset\train\train_ground_truth.tsv")

SPLIT_JSON = Path(r"D:\amazon ML\reports\validation_split_ids.json")
EXP_DIR = Path(r"D:\amazon ML\experiments\v2_recovery")
EXP_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = Path(r"D:\amazon ML\reports\leaderboard_gap")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. MEMORY TRACKING HELPERS (WIN32 API)
# ─────────────────────────────────────────────────────────────────────────────

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

class MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ('dwLength', ctypes.c_ulong),
        ('dwMemoryLoad', ctypes.c_ulong),
        ('ullTotalPhys', ctypes.c_ulonglong),
        ('ullAvailPhys', ctypes.c_ulonglong),
        ('ullTotalPageFile', ctypes.c_ulonglong),
        ('ullAvailPageFile', ctypes.c_ulonglong),
        ('ullTotalVirtual', ctypes.c_ulonglong),
        ('ullAvailVirtual', ctypes.c_ulonglong),
        ('sullAvailExtendedVirtual', ctypes.c_ulonglong),
    ]

def get_sys_avail_gb():
    stat = MEMORYSTATUSEX()
    stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
    return stat.ullAvailPhys / (1024**3)

# ─────────────────────────────────────────────────────────────────────────────
# 2. NORMALIZATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

from src.preprocessing import normalize_name as v1_normalize_name
from src.preprocessing import normalize_address as v1_normalize_address
from src.preprocessing import normalize_country as v1_normalize_country

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
    """Unicode NFKD decomposition to strip diacritics and ligatures."""
    if not text:
        return ""
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))

def extract_features_v1(raw_name, raw_addr, country):
    cty = v1_normalize_country(str(country or ""))
    nn = v1_normalize_name(str(raw_name or ""))
    toks = [t for t in nn.split() if t not in V1_LEGAL_TOKENS]
    sn = " ".join(toks) if toks else nn
    stoks = sorted(toks)
    ssn = " ".join(stoks) if stoks else nn
    na = v1_normalize_address(str(raw_addr or ""))
    nums = [t for t in na.split() if t.isdigit() and len(t) >= 2]
    return {
        "country": cty,
        "norm_name": nn,
        "stem_name": sn,
        "sorted_stem": ssn,
        "tokens": toks,
        "sorted_tokens": stoks,
        "norm_addr": na,
        "nums": nums,
    }

def extract_features_intl(raw_name, raw_addr, country):
    cty = v1_normalize_country(str(country or ""))
    name_clean = strip_accents(str(raw_name or "")).lower()
    addr_clean = strip_accents(str(raw_addr or "")).lower()
    nn = v1_normalize_name(name_clean)
    toks = [t for t in nn.split() if t not in V2_LEGAL_TOKENS]
    sn = " ".join(toks) if toks else nn
    stoks = sorted(toks)
    ssn = " ".join(stoks) if stoks else nn
    na = v1_normalize_address(addr_clean)
    nums = [t for t in na.split() if t.isdigit() and len(t) >= 2]
    return {
        "country": cty,
        "norm_name": nn,
        "stem_name": sn,
        "sorted_stem": ssn,
        "tokens": toks,
        "sorted_tokens": stoks,
        "norm_addr": na,
        "nums": nums,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. MAIN SAFE BLOCKING EXPERIMENT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("TASK 13: SAFE LOW-MEMORY V2 BLOCKING EXPERIMENT")
    print("=" * 80)
    t_start = time.time()
    
    ws, peak = get_process_memory_mb()
    avail = get_sys_avail_gb()
    print(f"Initial State: Process WS = {ws:.1f} MB | System Available RAM = {avail:.2f} GB")
    if avail < 1.5:
        print("ERROR: Available RAM < 1.5 GB. Aborting heavy computation safely.")
        return

    # 1. Load 5k deterministic validation sample
    print("\n[Step 1/5] Loading 5,000 deterministic S1 validation entities...")
    with open(SPLIT_JSON, "r", encoding="utf-8") as f:
        split_data = json.load(f)
    rng = np.random.RandomState(42)
    sample_5k_ids = sorted(list(rng.choice(split_data["val_s1_ids"], size=5000, replace=False)))
    sample_5k_set = set(sample_5k_ids)
    sample_df = pd.DataFrame({"entity_id": [str(x) for x in sample_5k_ids]})

    # Ground truth
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
    con.execute("PRAGMA max_memory='1.5GB';")
    con.execute("PRAGMA threads=4;")

    s1_raw_df = con.execute(f"""
        SELECT s.entity_id, s.country, s.business_name, s.business_address
        FROM read_csv('{TRAIN_S1.as_posix()}', delim='\\t', header=true) s
        JOIN sample_df n ON s.entity_id = n.entity_id
    """).df()
    print(f"Loaded raw S1 text records: {len(s1_raw_df):,}")

    # Build query profiles
    s1_profiles_v1 = {}
    s1_profiles_intl = {}
    s1_raw_info = {}
    for r in s1_raw_df.itertuples(index=False):
        eid, cty, raw_name, raw_addr = r
        s1_raw_info[eid] = (raw_name, raw_addr, cty)
        s1_profiles_v1[eid] = extract_features_v1(raw_name, raw_addr, cty)
        s1_profiles_intl[eid] = extract_features_intl(raw_name, raw_addr, cty)

    # 2. Build Query-Driven Inverted Lookup Tables
    print("\n[Step 2/5] Constructing compact query lookup tables from 5,000 S1 queries...")
    
    # Query tables map: key -> list of s1_ids
    # V1 standard keys
    q_norm_name = defaultdict(list)
    q_stem_name = defaultdict(list)
    q_ordered_pair = defaultdict(list)
    q_single_tok = defaultdict(list)
    q_norm_addr = defaultdict(list)
    q_addr_num = defaultdict(list)
    
    # V2 order-invariant keys
    q_sorted_stem = defaultdict(list)
    q_sorted_pair = defaultdict(list)
    q_sec_shingle = defaultdict(list)
    
    # V3 subdivision keys
    q_sub_num = defaultdict(list)
    q_sub_3 = defaultdict(list)
    
    # V5/V6 international keys
    q_intl_norm_name = defaultdict(list)
    q_intl_stem_name = defaultdict(list)
    q_intl_sorted_stem = defaultdict(list)
    q_intl_ordered_pair = defaultdict(list)
    q_intl_sorted_pair = defaultdict(list)
    q_intl_sec_shingle = defaultdict(list)
    q_intl_single_tok = defaultdict(list)
    q_intl_norm_addr = defaultdict(list)
    q_intl_addr_num = defaultdict(list)
    q_intl_sub_num = defaultdict(list)
    q_intl_sub_3 = defaultdict(list)

    for s1_id in sample_5k_ids:
        # V1
        p1 = s1_profiles_v1[s1_id]
        c1, nn1, sn1, ssn1, toks1, stoks1, na1, nums1 = (
            p1["country"], p1["norm_name"], p1["stem_name"], p1["sorted_stem"],
            p1["tokens"], p1["sorted_tokens"], p1["norm_addr"], p1["nums"]
        )
        if nn1:
            q_norm_name[(c1, nn1)].append(s1_id)
        if sn1:
            q_stem_name[(c1, sn1)].append(s1_id)
        if ssn1:
            q_sorted_stem[(c1, ssn1)].append(s1_id)
        if len(toks1) >= 2:
            q_ordered_pair[(c1, toks1[0], toks1[1])].append(s1_id)
        if len(stoks1) >= 2:
            q_sorted_pair[(c1, stoks1[0], stoks1[1])].append(s1_id)
            if nums1:
                q_sub_num[(c1, stoks1[0], stoks1[1], nums1[0])].append(s1_id)
            if len(stoks1) >= 3:
                q_sub_3[(c1, stoks1[0], stoks1[1], stoks1[2])].append(s1_id)
                q_sec_shingle[(c1, stoks1[0], stoks1[2])].append(s1_id)
                q_sec_shingle[(c1, stoks1[1], stoks1[2])].append(s1_id)
        elif len(toks1) == 1 and len(toks1[0]) >= 4:
            q_single_tok[(c1, toks1[0])].append(s1_id)
        if na1:
            q_norm_addr[(c1, na1)].append(s1_id)
        if nums1 and toks1:
            q_addr_num[(c1, nums1[0], toks1[0])].append(s1_id)
            
        # International
        pi = s1_profiles_intl[s1_id]
        ci, nni, sni, ssni, toksi, stoksi, nai, numsi = (
            pi["country"], pi["norm_name"], pi["stem_name"], pi["sorted_stem"],
            pi["tokens"], pi["sorted_tokens"], pi["norm_addr"], pi["nums"]
        )
        if nni:
            q_intl_norm_name[(ci, nni)].append(s1_id)
        if sni:
            q_intl_stem_name[(ci, sni)].append(s1_id)
        if ssni:
            q_intl_sorted_stem[(ci, ssni)].append(s1_id)
        if len(toksi) >= 2:
            q_intl_ordered_pair[(ci, toksi[0], toksi[1])].append(s1_id)
        if len(stoksi) >= 2:
            q_intl_sorted_pair[(ci, stoksi[0], stoksi[1])].append(s1_id)
            if numsi:
                q_intl_sub_num[(ci, stoksi[0], stoksi[1], numsi[0])].append(s1_id)
            if len(stoksi) >= 3:
                q_intl_sub_3[(ci, stoksi[0], stoksi[1], stoksi[2])].append(s1_id)
                q_intl_sec_shingle[(ci, stoksi[0], stoksi[2])].append(s1_id)
                q_intl_sec_shingle[(ci, stoksi[1], stoksi[2])].append(s1_id)
        elif len(toksi) == 1 and len(toksi[0]) >= 4:
            q_intl_single_tok[(ci, toksi[0])].append(s1_id)
        if nai:
            q_intl_norm_addr[(ci, nai)].append(s1_id)
        if numsi and toksi:
            q_intl_addr_num[(ci, numsi[0], toksi[0])].append(s1_id)

    total_query_keys = sum(len(d) for d in [
        q_norm_name, q_stem_name, q_sorted_stem, q_ordered_pair, q_sorted_pair,
        q_sec_shingle, q_single_tok, q_norm_addr, q_addr_num, q_sub_num, q_sub_3,
        q_intl_norm_name, q_intl_stem_name, q_intl_sorted_stem, q_intl_ordered_pair,
        q_intl_sorted_pair, q_intl_sec_shingle, q_intl_single_tok, q_intl_norm_addr,
        q_intl_addr_num, q_intl_sub_num, q_intl_sub_3
    ])
    ws, peak = get_process_memory_mb()
    print(f"Total compact query keys across all strategies: {total_query_keys:,}. Process WS: {ws:.1f} MB")

    # 3. Stream Full Target Population (S2 & S3, 10.3M rows)
    print("\n[Step 3/5] Streaming full 10,320,219 targets from disk in bounded chunks...")
    
    # Raw match accumulator per s1_id:
    # s1_matches[s1_id][channel] = list of candidate target eids (capped to avoid memory bloat)
    # Channel names:
    # "norm_name", "stem_name", "sorted_stem", "ordered_pair", "sorted_pair", "sec_shingle",
    # "single_tok", "norm_addr", "addr_num", "sub_num", "sub_3"
    # and their intl_ counterparts.
    s1_matches = defaultdict(lambda: defaultdict(list))
    
    # Block frequency counter to detect heavy blocks
    block_freq = Counter()
    
    t_stream_start = time.time()
    total_targets_scanned = 0
    chunk_vectors = 50 # ~102,400 rows per chunk
    
    # Ground truth lookup for quick diagnostics
    gt_reverse = defaultdict(set)
    for s1_id, t_list in gt_5k.items():
        for tid in t_list:
            gt_reverse[tid].add(s1_id)

    matched_gt_links = {
        "V1": set(),
        "V2": set(),
        "V3": set(),
        "V4": set(),
        "V5": set(),
        "V6": set(),
    }
    
    # For miss diagnostics: record target info if it is a ground truth match
    gt_target_info = {}

    for filepath in [TRAIN_S2, TRAIN_S3]:
        file_t0 = time.time()
        print(f"  Streaming {filepath.name}...")
        rel = con.execute(f"SELECT entity_id, country, business_name, business_address FROM read_csv('{filepath.as_posix()}', delim='\\t', header=true)")
        
        file_rows = 0
        while True:
            chunk = rel.fetch_df_chunk(chunk_vectors)
            if chunk is None or len(chunk) == 0:
                break
            
            # Check safety
            avail_gb = get_sys_avail_gb()
            if avail_gb < 1.5:
                print(f"CRITICAL WARNING: Available RAM dropped to {avail_gb:.2f} GB (<1.5 GB). Stopping stream safely.")
                break
                
            for r in chunk.itertuples(index=False):
                eid, raw_cty, raw_name, raw_addr = r
                
                # If this target is in ground truth, store its raw text for diagnostics
                if eid in gt_reverse:
                    gt_target_info[eid] = (raw_name, raw_addr, raw_cty)
                
                # Quick V1 feature extraction
                # Fast inline normalization
                cty = v1_normalize_country(str(raw_cty or ""))
                nn = v1_normalize_name(str(raw_name or ""))
                toks = [t for t in nn.split() if t not in V1_LEGAL_TOKENS]
                sn = " ".join(toks) if toks else nn
                stoks = sorted(toks)
                ssn = " ".join(stoks) if stoks else nn
                
                # V1 Keys matching
                if nn and (cty, nn) in q_norm_name:
                    block_freq[("norm_name", cty, nn)] += 1
                    for s1_id in q_norm_name[(cty, nn)]:
                        m_list = s1_matches[s1_id]["norm_name"]
                        if len(m_list) < 150:
                            m_list.append(eid)
                            
                if sn and (cty, sn) in q_stem_name:
                    block_freq[("stem_name", cty, sn)] += 1
                    for s1_id in q_stem_name[(cty, sn)]:
                        m_list = s1_matches[s1_id]["stem_name"]
                        if len(m_list) < 150:
                            m_list.append(eid)
                            
                if ssn and (cty, ssn) in q_sorted_stem:
                    block_freq[("sorted_stem", cty, ssn)] += 1
                    for s1_id in q_sorted_stem[(cty, ssn)]:
                        m_list = s1_matches[s1_id]["sorted_stem"]
                        if len(m_list) < 150:
                            m_list.append(eid)
                            
                if len(toks) >= 2:
                    k_ord = (cty, toks[0], toks[1])
                    if k_ord in q_ordered_pair:
                        block_freq[("ordered_pair", k_ord)] += 1
                        for s1_id in q_ordered_pair[k_ord]:
                            m_list = s1_matches[s1_id]["ordered_pair"]
                            if len(m_list) < 150:
                                m_list.append(eid)
                                
                if len(stoks) >= 2:
                    k_sort = (cty, stoks[0], stoks[1])
                    if k_sort in q_sorted_pair:
                        block_freq[("sorted_pair", k_sort)] += 1
                        for s1_id in q_sorted_pair[k_sort]:
                            m_list = s1_matches[s1_id]["sorted_pair"]
                            if len(m_list) < 150:
                                m_list.append(eid)
                                
                    if len(stoks) >= 3:
                        for k_sec in [(cty, stoks[0], stoks[2]), (cty, stoks[1], stoks[2])]:
                            if k_sec in q_sec_shingle:
                                block_freq[("sec_shingle", k_sec)] += 1
                                for s1_id in q_sec_shingle[k_sec]:
                                    m_list = s1_matches[s1_id]["sec_shingle"]
                                    if len(m_list) < 150:
                                        m_list.append(eid)
                        k_sub3 = (cty, stoks[0], stoks[1], stoks[2])
                        if k_sub3 in q_sub_3:
                            block_freq[("sub_3", k_sub3)] += 1
                            for s1_id in q_sub_3[k_sub3]:
                                m_list = s1_matches[s1_id]["sub_3"]
                                if len(m_list) < 150:
                                    m_list.append(eid)
                                    
                elif len(toks) == 1 and len(toks[0]) >= 4:
                    k_sing = (cty, toks[0])
                    if k_sing in q_single_tok:
                        block_freq[("single_tok", k_sing)] += 1
                        for s1_id in q_single_tok[k_sing]:
                            m_list = s1_matches[s1_id]["single_tok"]
                            if len(m_list) < 150:
                                m_list.append(eid)
                                
                # Address matching
                if raw_addr:
                    na = v1_normalize_address(str(raw_addr))
                    nums = [t for t in na.split() if t.isdigit() and len(t) >= 2]
                    if na and (cty, na) in q_norm_addr:
                        block_freq[("norm_addr", cty, na)] += 1
                        for s1_id in q_norm_addr[(cty, na)]:
                            m_list = s1_matches[s1_id]["norm_addr"]
                            if len(m_list) < 150:
                                m_list.append(eid)
                    if nums and toks:
                        k_an = (cty, nums[0], toks[0])
                        if k_an in q_addr_num:
                            block_freq[("addr_num", k_an)] += 1
                            for s1_id in q_addr_num[k_an]:
                                m_list = s1_matches[s1_id]["addr_num"]
                                if len(m_list) < 150:
                                    m_list.append(eid)
                        if len(stoks) >= 2:
                            k_snum = (cty, stoks[0], stoks[1], nums[0])
                            if k_snum in q_sub_num:
                                block_freq[("sub_num", k_snum)] += 1
                                for s1_id in q_sub_num[k_snum]:
                                    m_list = s1_matches[s1_id]["sub_num"]
                                    if len(m_list) < 150:
                                        m_list.append(eid)
                                        
                # International Matching (V5/V6)
                # Check if diacritics or international tokens exist
                raw_name_str = str(raw_name or "")
                raw_addr_str = str(raw_addr or "")
                # Only run NFKD if non-ascii or legal tokens
                name_clean = strip_accents(raw_name_str).lower()
                if name_clean != nn:
                    nn_i = v1_normalize_name(name_clean)
                    toks_i = [t for t in nn_i.split() if t not in V2_LEGAL_TOKENS]
                    sn_i = " ".join(toks_i) if toks_i else nn_i
                    stoks_i = sorted(toks_i)
                    ssn_i = " ".join(stoks_i) if stoks_i else nn_i
                    
                    if nn_i and (cty, nn_i) in q_intl_norm_name:
                        for s1_id in q_intl_norm_name[(cty, nn_i)]:
                            m_list = s1_matches[s1_id]["intl_norm_name"]
                            if len(m_list) < 150:
                                m_list.append(eid)
                    if sn_i and (cty, sn_i) in q_intl_stem_name:
                        for s1_id in q_intl_stem_name[(cty, sn_i)]:
                            m_list = s1_matches[s1_id]["intl_stem_name"]
                            if len(m_list) < 150:
                                m_list.append(eid)
                    if ssn_i and (cty, ssn_i) in q_intl_sorted_stem:
                        for s1_id in q_intl_sorted_stem[(cty, ssn_i)]:
                            m_list = s1_matches[s1_id]["intl_sorted_stem"]
                            if len(m_list) < 150:
                                m_list.append(eid)
                    if len(toks_i) >= 2:
                        k_iord = (cty, toks_i[0], toks_i[1])
                        if k_iord in q_intl_ordered_pair:
                            for s1_id in q_intl_ordered_pair[k_iord]:
                                m_list = s1_matches[s1_id]["intl_ordered_pair"]
                                if len(m_list) < 150:
                                    m_list.append(eid)
                    if len(stoks_i) >= 2:
                        k_isort = (cty, stoks_i[0], stoks_i[1])
                        if k_isort in q_intl_sorted_pair:
                            for s1_id in q_intl_sorted_pair[k_isort]:
                                m_list = s1_matches[s1_id]["intl_sorted_pair"]
                                if len(m_list) < 150:
                                    m_list.append(eid)
                        if len(stoks_i) >= 3:
                            for k_isec in [(cty, stoks_i[0], stoks_i[2]), (cty, stoks_i[1], stoks_i[2])]:
                                if k_isec in q_intl_sec_shingle:
                                    for s1_id in q_intl_sec_shingle[k_isec]:
                                        m_list = s1_matches[s1_id]["intl_sec_shingle"]
                                        if len(m_list) < 150:
                                            m_list.append(eid)

            file_rows += len(chunk)
            total_targets_scanned += len(chunk)
            del chunk
            gc.collect()

        ws, peak = get_process_memory_mb()
        print(f"  Finished {filepath.name}: {file_rows:,} rows in {time.time()-file_t0:.1f}s. Process WS: {ws:.1f} MB (Peak: {peak:.1f} MB)")

    total_stream_time = time.time() - t_stream_start
    print(f"\nCompleted streaming {total_targets_scanned:,} targets in {total_stream_time:.1f}s ({total_targets_scanned/total_stream_time:,.0f} rows/s).")
    ws, peak = get_process_memory_mb()
    print(f"Post-streaming Memory: Process WS = {ws:.1f} MB (Peak {peak:.1f} MB) | System Available = {get_sys_avail_gb():.2f} GB")

    # 4. Construct Exact Candidate Sets for V1, V2, V3, V4, V5, V6
    print("\n[Step 4/5] Assembling candidate sets and evaluating versions V1 through V6...")
    
    def assemble_candidates(version_name):
        cands_by_s1 = {}
        t0 = time.time()
        
        for s1_id in sample_5k_ids:
            p1 = s1_profiles_v1[s1_id]
            pi = s1_profiles_intl[s1_id]
            m = s1_matches[s1_id]
            
            selected = []
            seen = set()
            
            def add_eids(eid_list, limit):
                nonlocal selected, seen
                count = 0
                for eid in eid_list:
                    if eid not in seen:
                        seen.add(eid)
                        selected.append(eid)
                        count += 1
                        if limit is not None and count >= limit:
                            break

            if version_name == "V1":
                # Current blocking: max 60 total
                add_eids(m["norm_name"], 60)
                if len(selected) < 60:
                    add_eids(m["stem_name"], 60 - len(selected))
                if len(selected) < 60:
                    add_eids(m["ordered_pair"], min(60, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["single_tok"], min(60, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["norm_addr"], min(30, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["addr_num"], min(30, 60 - len(selected)))
                cands_by_s1[s1_id] = selected[:60]

            elif version_name == "V2":
                # Order-invariant: sorted stems & sorted shingles, max 60
                add_eids(m["norm_name"], 60)
                if len(selected) < 60:
                    add_eids(m["sorted_stem"], 60 - len(selected))
                if len(selected) < 60:
                    add_eids(m["stem_name"], 60 - len(selected))
                if len(selected) < 60:
                    add_eids(m["sorted_pair"], min(60, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["sec_shingle"], min(30, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["ordered_pair"], min(60, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["single_tok"], min(60, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["norm_addr"], min(30, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["addr_num"], min(30, 60 - len(selected)))
                cands_by_s1[s1_id] = selected[:60]

            elif version_name == "V3":
                # Frequency-aware: check frequency of ordered_pair
                c1, toks1 = p1["country"], p1["tokens"]
                ord_freq = block_freq[("ordered_pair", (c1, toks1[0], toks1[1]))] if len(toks1) >= 2 else 0
                
                add_eids(m["norm_name"], 60)
                if len(selected) < 60:
                    add_eids(m["stem_name"], 60 - len(selected))
                if len(selected) < 60:
                    if ord_freq > 50:
                        # Heavy block: take subdivided matches first
                        add_eids(m["sub_num"], 30)
                        add_eids(m["sub_3"], 30)
                        # Reserve room, take only top 25 from broad block
                        rem = max(0, 60 - len(selected))
                        add_eids(m["ordered_pair"], min(25, rem))
                    else:
                        add_eids(m["ordered_pair"], min(60, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["single_tok"], min(60, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["norm_addr"], min(30, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["addr_num"], min(30, 60 - len(selected)))
                cands_by_s1[s1_id] = selected[:60]

            elif version_name == "V4":
                # No lossy truncation / smart quotas: max 120
                add_eids(m["norm_name"], None) # unlimited exact
                add_eids(m["stem_name"], 30)
                add_eids(m["ordered_pair"], 50)
                add_eids(m["single_tok"], 30)
                add_eids(m["norm_addr"], 25)
                add_eids(m["addr_num"], 25)
                cands_by_s1[s1_id] = selected[:120]

            elif version_name == "V5":
                # International normalization only: max 60
                add_eids(m["norm_name"], 60)
                add_eids(m["intl_norm_name"], 60 - len(selected))
                if len(selected) < 60:
                    add_eids(m["stem_name"], 60 - len(selected))
                    add_eids(m["intl_stem_name"], 60 - len(selected))
                if len(selected) < 60:
                    add_eids(m["ordered_pair"], min(60, 60 - len(selected)))
                    add_eids(m["intl_ordered_pair"], min(60, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["single_tok"], min(60, 60 - len(selected)))
                    add_eids(m["intl_single_tok"], min(60, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["norm_addr"], min(30, 60 - len(selected)))
                    add_eids(m["intl_norm_addr"], min(30, 60 - len(selected)))
                if len(selected) < 60:
                    add_eids(m["addr_num"], min(30, 60 - len(selected)))
                    add_eids(m["intl_addr_num"], min(30, 60 - len(selected)))
                cands_by_s1[s1_id] = selected[:60]

            elif version_name == "V6":
                # All combined: intl + order-invariant + freq-aware + smart quotas (max 120)
                # 1. Exact norm name (v1 & intl)
                add_eids(m["norm_name"], None)
                add_eids(m["intl_norm_name"], None)
                
                # 2. Sorted stems (v1 & intl)
                add_eids(m["sorted_stem"], 30)
                add_eids(m["intl_sorted_stem"], 30)
                add_eids(m["stem_name"], 30)
                add_eids(m["intl_stem_name"], 30)
                
                # 3. Subdivided keys for heavy blocks
                c1, stoks1 = p1["country"], p1["sorted_tokens"]
                sort_freq = block_freq[("sorted_pair", (c1, stoks1[0], stoks1[1]))] if len(stoks1) >= 2 else 0
                if sort_freq > 50:
                    add_eids(m["sub_num"], 30)
                    add_eids(m["intl_sub_num"], 30)
                    add_eids(m["sub_3"], 30)
                    add_eids(m["intl_sub_3"], 30)
                    add_eids(m["sorted_pair"], 30)
                    add_eids(m["intl_sorted_pair"], 30)
                else:
                    add_eids(m["sorted_pair"], 50)
                    add_eids(m["intl_sorted_pair"], 50)
                    
                # 4. Secondary sorted shingles
                if len(selected) < 120:
                    add_eids(m["sec_shingle"], 25)
                    add_eids(m["intl_sec_shingle"], 25)
                    
                # 5. Ordered pairs
                if len(selected) < 120:
                    add_eids(m["ordered_pair"], 25)
                    add_eids(m["intl_ordered_pair"], 25)
                    
                # 6. Single tokens
                if len(selected) < 120:
                    add_eids(m["single_tok"], 20)
                    add_eids(m["intl_single_tok"], 20)
                    
                # 7. Address & address numeric
                if len(selected) < 120:
                    add_eids(m["norm_addr"], 20)
                    add_eids(m["intl_norm_addr"], 20)
                    add_eids(m["addr_num"], 20)
                    add_eids(m["intl_addr_num"], 20)
                    
                cands_by_s1[s1_id] = selected[:120]

        return cands_by_s1

    results = {}
    candidate_maps = {}

    for v_name in ["V1", "V2", "V3", "V4", "V5", "V6"]:
        t0 = time.time()
        cands_map = assemble_candidates(v_name)
        candidate_maps[v_name] = cands_map
        elapsed = time.time() - t0
        
        # Calculate metrics
        covered_links = 0
        link_retrieved_set = set()
        c_lens = []
        
        for s1_id, true_list in gt_5k.items():
            pred_set = set(cands_map.get(s1_id, []))
            for tid in true_list:
                if tid in pred_set:
                    covered_links += 1
                    link_retrieved_set.add((s1_id, tid))
            c_lens.append(len(pred_set))
            
        matched_gt_links[v_name] = link_retrieved_set
        
        recall = covered_links / total_true_links if total_true_links > 0 else 1.0
        total_cands = sum(c_lens)
        avg_cands = np.mean(c_lens)
        p50 = np.percentile(c_lens, 50)
        p95 = np.percentile(c_lens, 95)
        max_c = max(c_lens) if c_lens else 0
        red_ratio = 1.0 - (total_cands / (len(sample_5k_ids) * total_targets_scanned))
        
        ws, peak = get_process_memory_mb()
        
        results[v_name] = {
            "version": v_name,
            "candidate_recall": round(float(recall), 4),
            "covered_links": covered_links,
            "total_true_links": total_true_links,
            "total_candidates": total_cands,
            "avg_candidates": round(float(avg_cands), 2),
            "p50_candidates": round(float(p50), 1),
            "p95_candidates": round(float(p95), 1),
            "max_candidates": max_c,
            "reduction_ratio": round(float(red_ratio), 8),
            "runtime_s": round(elapsed + (total_stream_time / 6), 2),
            "peak_ram_mb": round(peak, 1),
        }

    # Print summary table
    df_res = pd.DataFrame(list(results.values()))
    table_csv = EXP_DIR / "v2_blocking_comparison.csv"
    df_res.to_csv(table_csv, index=False)
    
    print("\n" + "=" * 80)
    print("TASK 13 BLOCKING COMPARISON SUMMARY:")
    print("=" * 80)
    print(df_res[["version", "candidate_recall", "total_candidates", "avg_candidates", "p50_candidates", "p95_candidates", "max_candidates", "reduction_ratio", "runtime_s", "peak_ram_mb"]].to_string(index=False))

    # 5. Missing-Link Diagnostics
    print("\n[Step 5/5] Generating missing-link diagnostics (reports/leaderboard_gap/v2_blocking_misses.csv)...")
    
    misses = []
    v1_hits = matched_gt_links["V1"]
    v6_hits = matched_gt_links["V6"]
    
    # Classify why links are missed in V6
    for s1_id, true_list in gt_5k.items():
        s1_raw_name, s1_raw_addr, s1_cty = s1_raw_info[s1_id]
        p1 = s1_profiles_v1[s1_id]
        pi = s1_profiles_intl[s1_id]
        
        for tid in true_list:
            in_v1 = (s1_id, tid) in v1_hits
            in_v6 = (s1_id, tid) in v6_hits
            
            t_raw_name, t_raw_addr, t_raw_cty = gt_target_info.get(tid, ("", "", ""))
            
            # Determine status
            if in_v6 and not in_v1:
                status = "RECOVERED_BY_V6"
                category = "Recovered"
            elif in_v6 and in_v1:
                status = "RETAINED_BOTH"
                category = "Both"
            elif not in_v6 and in_v1:
                status = "REGRESSED_IN_V6"
                category = "Regression"
            else:
                status = "STILL_MISSED"
                # Determine failure reason
                if s1_cty != t_raw_cty and v1_normalize_country(s1_cty) != v1_normalize_country(t_raw_cty):
                    category = "Country Mismatch"
                elif not s1_raw_name or not t_raw_name:
                    category = "Missing Name"
                else:
                    t_toks = set(v1_normalize_name(t_raw_name).split()) - V1_LEGAL_TOKENS
                    s_toks = set(p1["tokens"])
                    common = s_toks & t_toks
                    if len(common) == 0:
                        category = "Zero Token Overlap / Severe Distortion"
                    elif len(common) == 1:
                        category = "Single Weak Token Overlap (<2 tokens)"
                    else:
                        category = "Quota Exceeded / Truncated Beyond Cap"
                        
            misses.append({
                "s1_id": s1_id,
                "target_id": tid,
                "country": s1_cty,
                "s1_name": s1_raw_name,
                "target_name": t_raw_name,
                "s1_addr": s1_raw_addr,
                "target_addr": t_raw_addr,
                "in_v1": in_v1,
                "in_v6": in_v6,
                "status": status,
                "failure_category": category,
            })
            
    df_misses = pd.DataFrame(misses)
    miss_csv = REPORT_DIR / "v2_blocking_misses.csv"
    df_misses.to_csv(miss_csv, index=False)
    print(f"Saved {len(df_misses):,} link diagnostics to: {miss_csv}")
    
    # Count categories for still missed
    still_missed_df = df_misses[df_misses["status"] == "STILL_MISSED"]
    top_failures = still_missed_df["failure_category"].value_counts()
    print("\nBreakdown of Still Missed Links in V6:")
    for cat, cnt in top_failures.items():
        print(f"  - {cat}: {cnt:,} ({cnt/len(still_missed_df)*100:.1f}%)")
        
    recovered_count = sum(1 for m in misses if m["status"] == "RECOVERED_BY_V6")
    still_missing_count = len(still_missed_df)
    
    v1_rec = results["V1"]["candidate_recall"]
    v2_rec = results["V2"]["candidate_recall"]
    v6_rec = results["V6"]["candidate_recall"]
    
    recommendation = "KEEP V2" if v6_rec > 0.70 else "REJECT V2"
    
    # 6. Format and Print Final Report
    print("\n" + "=" * 80)
    print("SAFE V2 BLOCKING REPORT")
    print("=======================")
    print(f"V1 candidate recall: {v1_rec:.4f}")
    print(f"V2 candidate recall: {v2_rec:.4f}")
    print(f"V3 candidate recall: {results['V3']['candidate_recall']:.4f}")
    print(f"V4 candidate recall: {results['V4']['candidate_recall']:.4f}")
    print(f"V5 candidate recall: {results['V5']['candidate_recall']:.4f}")
    print(f"V6 candidate recall: {v6_rec:.4f}")
    print()
    print(f"V1 candidate count: {results['V1']['total_candidates']:,}")
    print(f"V2 candidate count: {results['V2']['total_candidates']:,}")
    print(f"V6 candidate count: {results['V6']['total_candidates']:,}")
    print()
    print(f"V1 avg candidates/S1: {results['V1']['avg_candidates']:.2f}")
    print(f"V2 avg candidates/S1: {results['V2']['avg_candidates']:.2f}")
    print(f"V6 avg candidates/S1: {results['V6']['avg_candidates']:.2f}")
    print()
    print(f"V1 p95: {results['V1']['p95_candidates']:.1f}")
    print(f"V2 p95: {results['V2']['p95_candidates']:.1f}")
    print(f"V6 p95: {results['V6']['p95_candidates']:.1f}")
    print()
    print(f"V1 runtime: {results['V1']['runtime_s']:.2f}s")
    print(f"V2 runtime: {results['V2']['runtime_s']:.2f}s")
    print(f"V6 runtime: {results['V6']['runtime_s']:.2f}s")
    print()
    print(f"V1 peak RAM: {results['V1']['peak_ram_mb']:.1f} MB")
    print(f"V2 peak RAM: {results['V2']['peak_ram_mb']:.1f} MB")
    print(f"V6 peak RAM: {results['V6']['peak_ram_mb']:.1f} MB")
    print()
    print(f"Ground-truth links recovered: {recovered_count:,}")
    print(f"Ground-truth links still missing: {still_missing_count:,}")
    print()
    print("Top remaining blocking failures:")
    top_cats = list(top_failures.items())
    for idx in range(min(5, len(top_cats))):
        cat_name, cat_cnt = top_cats[idx]
        print(f"{idx+1}. {cat_name}: {cat_cnt:,} instances ({cat_cnt/still_missing_count*100:.1f}%)")
    for idx in range(len(top_cats), 5):
        print(f"{idx+1}. None")
    print()
    print(f"Recommendation:\n{recommendation}")
    print("=======================")

if __name__ == "__main__":
    main()
