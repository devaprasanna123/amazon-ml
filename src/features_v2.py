"""
src/features_v2.py
TASK 14 PHASE D: Advanced Feature Engineering for Business Entity Resolution.

Feature groups:
1. NAME SIMILARITY:
   - name_ratio (Levenshtein)
   - name_partial_ratio
   - name_token_sort_ratio
   - name_token_set_ratio
   - name_exact
   - name_stem_exact
   - name_sorted_stem_exact
   - name_char_3gram_jaccard
   - name_char_4gram_jaccard
   - name_token_jaccard
   - name_len_diff_ratio
   - name_prefix_match (first 4 chars)
   - name_acronym_match

2. ADDRESS SIMILARITY:
   - addr_ratio
   - addr_token_sort_ratio
   - addr_token_set_ratio
   - addr_token_jaccard
   - addr_exact
   - addr_num_overlap_ratio
   - addr_num_exact
   - addr_num_conflict (distinct non-overlapping numbers)
   - addr_street_num_match (first numeric match)
   - addr_postal_match
   - addr_empty_s1
   - addr_empty_s2

3. CROSS-FIELD & INTERACTION SIGNALS:
   - name_strong_addr_conflict (name >= 0.85, addr_conflict == 1)
   - name_moderate_addr_strong (name >= 0.60, addr >= 0.85)
   - name_weak_addr_strong (name < 0.50, addr >= 0.85)
   - country_match
   - is_source2 (1 for S2, 0 for S3)
"""

from __future__ import annotations
import re
from typing import List, Set, Tuple, Optional
import numpy as np

try:
    from rapidfuzz import fuzz as rfuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

FEATURE_NAMES_V2 = [
    # ── NAME (13 features) ──────────────────────────────────────────────────
    "name_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_exact",
    "name_stem_exact",
    "name_sorted_stem_exact",
    "name_char_3gram_jaccard",
    "name_char_4gram_jaccard",
    "name_token_jaccard",
    "name_len_diff_ratio",
    "name_prefix_match",
    "name_acronym_match",
    # ── ADDRESS (12 features) ───────────────────────────────────────────────
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_exact",
    "addr_num_overlap_ratio",
    "addr_num_exact",
    "addr_num_conflict",
    "addr_street_num_match",
    "addr_postal_match",
    "addr_empty_s1",
    "addr_empty_s2",
    # ── CROSS-FIELD & INTERACTION (5 features) ──────────────────────────────
    "name_strong_addr_conflict",
    "name_moderate_addr_strong",
    "name_weak_addr_strong",
    "country_match",
    "is_source2",
]

N_FEATURES_V2 = len(FEATURE_NAMES_V2)


def get_char_ngrams(text: str, n: int) -> Set[str]:
    clean = text.strip()
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def ngram_jaccard(ngrams1: Set[str], ngrams2: Set[str]) -> float:
    if not ngrams1 and not ngrams2:
        return 1.0
    if not ngrams1 or not ngrams2:
        return 0.0
    inter = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)
    return inter / union if union > 0 else 0.0


def extract_acronym(tokens: List[str]) -> str:
    if not tokens:
        return ""
    return "".join(t[0] for t in tokens if t).lower()


def extract_features_v2_pair(rec1: dict, rec2: dict, r2_eid: str) -> np.ndarray:
    """
    Computes 30 rich features for candidate pair (rec1, rec2).
    rec1 and rec2 must have:
      - norm_name, stem_name, sorted_stem, tokens, sorted_tokens
      - norm_addr, nums
      - country
    """
    feat = np.zeros(N_FEATURES_V2, dtype=np.float32)

    n1, n2 = rec1["norm_name"], rec2["norm_name"]
    s1, s2 = rec1["stem_name"], rec2["stem_name"]
    ss1, ss2 = rec1["sorted_stem"], rec2["sorted_stem"]
    t1, t2 = rec1["tokens"], rec2["tokens"]
    a1, a2 = rec1["norm_addr"], rec2["norm_addr"]
    nums1, nums2 = rec1["nums"], rec2["nums"]
    c1, c2 = rec1["country"], rec2["country"]

    # 1. NAME FEATURES
    if n1 and n2:
        feat[0] = rfuzz.ratio(n1, n2) / 100.0 if HAS_RAPIDFUZZ else (1.0 if n1 == n2 else 0.0)
        feat[1] = rfuzz.partial_ratio(n1, n2) / 100.0 if HAS_RAPIDFUZZ else 0.0
        feat[2] = rfuzz.token_sort_ratio(n1, n2) / 100.0 if HAS_RAPIDFUZZ else 0.0
        feat[3] = rfuzz.token_set_ratio(n1, n2) / 100.0 if HAS_RAPIDFUZZ else 0.0
        feat[4] = 1.0 if n1 == n2 else 0.0
        feat[5] = 1.0 if s1 and s1 == s2 else 0.0
        feat[6] = 1.0 if ss1 and ss1 == ss2 else 0.0

        # Character n-grams
        ng3_1, ng3_2 = get_char_ngrams(n1, 3), get_char_ngrams(n2, 3)
        feat[7] = ngram_jaccard(ng3_1, ng3_2)
        ng4_1, ng4_2 = get_char_ngrams(n1, 4), get_char_ngrams(n2, 4)
        feat[8] = ngram_jaccard(ng4_1, ng4_2)

        # Token Jaccard
        st1, st2 = set(t1), set(t2)
        union_t = len(st1 | st2)
        feat[9] = len(st1 & st2) / union_t if union_t > 0 else 0.0

        # Length difference ratio
        max_l = max(len(n1), len(n2))
        feat[10] = abs(len(n1) - len(n2)) / max_l if max_l > 0 else 0.0

        # Prefix match (first 4 chars)
        feat[11] = 1.0 if len(n1) >= 4 and len(n2) >= 4 and n1[:4] == n2[:4] else 0.0

        # Acronym match
        acr1, acr2 = extract_acronym(t1), extract_acronym(t2)
        if acr1 and acr2 and len(acr1) >= 2 and len(acr2) >= 2:
            feat[12] = 1.0 if (acr1 == acr2 or acr1 in n2 or acr2 in n1) else 0.0

    # 2. ADDRESS FEATURES
    feat[23] = 1.0 if not a1 else 0.0
    feat[24] = 1.0 if not a2 else 0.0

    if a1 and a2:
        feat[13] = rfuzz.ratio(a1, a2) / 100.0 if HAS_RAPIDFUZZ else (1.0 if a1 == a2 else 0.0)
        feat[14] = rfuzz.token_sort_ratio(a1, a2) / 100.0 if HAS_RAPIDFUZZ else 0.0
        feat[15] = rfuzz.token_set_ratio(a1, a2) / 100.0 if HAS_RAPIDFUZZ else 0.0

        at1, at2 = set(a1.split()), set(a2.split())
        union_at = len(at1 | at2)
        feat[16] = len(at1 & at2) / union_at if union_at > 0 else 0.0
        feat[17] = 1.0 if a1 == a2 else 0.0

        # Numeric overlaps
        s_nums1, s_nums2 = set(nums1), set(nums2)
        if s_nums1 and s_nums2:
            inter_nums = len(s_nums1 & s_nums2)
            feat[18] = inter_nums / len(s_nums1)
            feat[19] = 1.0 if s_nums1 == s_nums2 else 0.0
            feat[20] = 1.0 if inter_nums == 0 else 0.0  # CONFLICT
            # Street number match (first numeric token)
            feat[21] = 1.0 if nums1[0] == nums2[0] else 0.0
            # Postal match (last numeric token if length 5 or 6)
            p1 = nums1[-1] if len(nums1[-1]) in (5, 6) else None
            p2 = nums2[-1] if len(nums2[-1]) in (5, 6) else None
            if p1 and p2:
                feat[22] = 1.0 if p1 == p2 else 0.0

    # 3. CROSS-FIELD INTERACTIONS
    name_score = feat[0]
    addr_score = feat[13]
    num_conflict = feat[20]

    # Name strong + address conflict (classic distractor!)
    feat[25] = 1.0 if (name_score >= 0.85 and num_conflict == 1.0) else 0.0
    # Name moderate + address strong (same location, different legal/short name)
    feat[26] = 1.0 if (name_score >= 0.60 and addr_score >= 0.85) else 0.0
    # Name weak + address strong
    feat[27] = 1.0 if (name_score < 0.50 and addr_score >= 0.85) else 0.0
    # Country match
    feat[28] = 1.0 if c1 and c2 and c1 == c2 else 0.0
    # Source indicator
    feat[29] = 1.0 if r2_eid.startswith("S2-") else 0.0

    return feat
