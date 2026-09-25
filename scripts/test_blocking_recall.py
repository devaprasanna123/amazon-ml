import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb
import pandas as pd
from collections import defaultdict
from src.preprocessing import normalize_name, normalize_address, tokenize
from src.config import (
    TRAIN_GROUND_TRUTH, TRAIN_SOURCE1, TRAIN_SOURCE2, TRAIN_SOURCE3,
    VAL_SPLIT_IDS_JSON, REPORTS_DIR,
)

print("Evaluating blocking strategies on training/validation ground truth...")

# Load 20,000 S1 entities from val split
import json
split = json.loads(open(VAL_SPLIT_IDS_JSON).read())
val_s1_ids = set(split["val_s1_ids"][:20000])

# Load ground truth for these S1 entities
print("Loading GT...")
gt_df = duckdb.query(f"""
    SELECT source1_entity_id, matched_entity_ids
    FROM read_csv('{TRAIN_GROUND_TRUTH.as_posix()}', delim='\\t', header=true)
    WHERE matched_entity_ids != ''
""").df()

val_gt = {}
target_s23_needed = set()
for _, r in gt_df.iterrows():
    s1_id = r["source1_entity_id"]
    if s1_id in val_s1_ids:
        matches = [m.strip() for m in r["matched_entity_ids"].split(",") if m.strip()]
        val_gt[s1_id] = matches
        target_s23_needed.update(matches)

print(f"Loaded {len(val_gt):,} non-empty S1 entities in val sample with {len(target_s23_needed):,} true matches.")

# Load S1 records
s1_df = duckdb.query(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{TRAIN_SOURCE1.as_posix()}', delim='\\t', header=true)
""").df()
s1_records = s1_df[s1_df["entity_id"].isin(val_gt.keys())].to_dict(orient="records")
print(f"Loaded {len(s1_records):,} S1 records.")

# For targets, load the needed true matches PLUS a random background pool of 200,000 S2 and 200,000 S3 records
print("Loading target S2 and S3 pool...")
s2_bg = duckdb.query(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{TRAIN_SOURCE2.as_posix()}', delim='\\t', header=true)
    LIMIT 200000
""").df()
s3_bg = duckdb.query(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{TRAIN_SOURCE3.as_posix()}', delim='\\t', header=true)
    LIMIT 200000
""").df()

# Also ensure all true matches are loaded
needed_list = list(target_s23_needed)
s2_true = duckdb.query(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{TRAIN_SOURCE2.as_posix()}', delim='\\t', header=true)
    WHERE entity_id IN (SELECT unnest({needed_list}))
""").df()
s3_true = duckdb.query(f"""
    SELECT entity_id, business_name, business_address, country
    FROM read_csv('{TRAIN_SOURCE3.as_posix()}', delim='\\t', header=true)
    WHERE entity_id IN (SELECT unnest({needed_list}))
""").df()

targets_df = pd.concat([s2_bg, s3_bg, s2_true, s3_true]).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
print(f"Total target records pool: {len(targets_df):,}")

# Build Inverted Indexes on targets
print("Building inverted indexes on targets...")
t0 = time.time()
idx_norm_name = defaultdict(list)
idx_stem_name = defaultdict(list)
idx_norm_addr = defaultdict(list)
idx_country_name_token = defaultdict(list)
idx_country_addr_num = defaultdict(list)

# Legal suffix tokens to strip for stem
LEGAL_TOKENS = {"llc", "inc", "ltd", "pvt", "limited", "private", "corp", "corporation", "co", "company", "llp", "pc", "plc"}

for _, r in targets_df.iterrows():
    eid = r["entity_id"]
    country = str(r["country"]).strip().upper()
    name = str(r["business_name"] or "")
    addr = str(r["business_address"] or "")
    
    nn = normalize_name(name)
    na = normalize_address(addr)
    
    if nn:
        idx_norm_name[(country, nn)].append(eid)
        toks = [t for t in nn.split() if t not in LEGAL_TOKENS]
        if toks:
            stem = " ".join(toks)
            idx_stem_name[(country, stem)].append(eid)
            # Index first 2 tokens if distinct
            if len(toks) >= 2:
                idx_country_name_token[(country, toks[0], toks[1])].append(eid)
            elif len(toks) == 1 and len(toks[0]) >= 4:
                idx_country_name_token[(country, toks[0], "")].append(eid)
    
    if na:
        idx_norm_addr[(country, na)].append(eid)
        nums = [t for t in na.split() if t.isdigit() and len(t) >= 3]
        if nums and nn:
            first_tok = nn.split()[0] if nn.split() else ""
            if first_tok:
                idx_country_addr_num[(country, nums[0], first_tok)].append(eid)

print(f"Indexes built in {time.time()-t0:.2f}s.")

# Now query each S1 entity and measure candidate recall
print("Querying candidate sets...")
t0 = time.time()
total_true_links = sum(len(v) for v in val_gt.values())
covered_links = 0
total_candidates = 0

for r in s1_records:
    s1_id = r["entity_id"]
    country = str(r["country"]).strip().upper()
    name = str(r["business_name"] or "")
    addr = str(r["business_address"] or "")
    
    nn = normalize_name(name)
    na = normalize_address(addr)
    
    cands = set()
    # Strategy 1: exact norm name
    if nn and (country, nn) in idx_norm_name:
        cands.update(idx_norm_name[(country, nn)])
    
    # Strategy 2: stem name
    toks = [t for t in nn.split() if t not in LEGAL_TOKENS]
    if toks:
        stem = " ".join(toks)
        if (country, stem) in idx_stem_name:
            cands.update(idx_stem_name[(country, stem)])
        if len(toks) >= 2 and (country, toks[0], toks[1]) in idx_country_name_token:
            cands.update(idx_country_name_token[(country, toks[0], toks[1])][:100])
        elif len(toks) == 1 and len(toks[0]) >= 4 and (country, toks[0], "") in idx_country_name_token:
            cands.update(idx_country_name_token[(country, toks[0], "")][:100])
            
    # Strategy 3: exact norm address
    if na and (country, na) in idx_norm_addr:
        cands.update(idx_norm_addr[(country, na)])
        
    # Strategy 4: number + first name token
    nums = [t for t in na.split() if t.isdigit() and len(t) >= 3]
    if nums and nn:
        first_tok = nn.split()[0] if nn.split() else ""
        if first_tok and (country, nums[0], first_tok) in idx_country_addr_num:
            cands.update(idx_country_addr_num[(country, nums[0], first_tok)][:50])
            
    total_candidates += len(cands)
    true_set = set(val_gt[s1_id])
    covered_links += len(true_set & cands)

el = time.time() - t0
recall = covered_links / total_true_links
avg_cands = total_candidates / len(s1_records)
print(f"Results on {len(s1_records):,} S1 entities ({el:.2f}s):")
print(f"  Candidate Recall: {recall:.4f} ({covered_links:,} / {total_true_links:,})")
print(f"  Average candidates per S1: {avg_cands:.2f}")
print(f"  Total candidate pairs: {total_candidates:,}")
