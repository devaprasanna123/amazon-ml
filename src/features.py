"""
src/features.py
===============
Pairwise feature computation for the matching model.

Feature groups:
  NAME    — exact match, fuzzy similarity (RapidFuzz), token similarity, TF-IDF cosine
  ADDRESS — exact match, fuzzy, token overlap, numeric overlap, postal match
  COUNTRY — exact match
  NUMERIC — numeric overlap ratios
  BLOCKING — which strategies generated this candidate (evidence)
  META    — length differences, missingness indicators, source indicator
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np

# Optional fast fuzzy — gracefully degrade if not installed
try:
    from rapidfuzz import fuzz as rfuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

from src.preprocessing import (
    normalize_name, normalize_address, normalize_country,
    name_tokens, address_tokens, address_numeric_tokens, extract_postal_code,
)

# ─────────────────────────────────────────────────────────────────────────────
# Low-level similarity utilities
# ─────────────────────────────────────────────────────────────────────────────

def _jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity on two token lists."""
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def _token_set_ratio(a: str, b: str) -> float:
    """
    Token-set ratio: sort tokens alphabetically and compare.
    Handles "LLC Foo" vs "Foo LLC".
    Uses RapidFuzz if available, else simple ratio.
    """
    a_sorted = " ".join(sorted(a.split()))
    b_sorted = " ".join(sorted(b.split()))
    if HAS_RAPIDFUZZ:
        return rfuzz.ratio(a_sorted, b_sorted) / 100.0
    else:
        return _simple_ratio(a_sorted, b_sorted)


def _simple_ratio(a: str, b: str) -> float:
    """Simple edit-distance-like ratio without RapidFuzz."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    la, lb = len(a), len(b)
    common = sum(1 for x, y in zip(a, b) if x == y)
    return 2 * common / (la + lb)


def _fuzzy_ratio(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return rfuzz.ratio(a, b) / 100.0
    return _simple_ratio(a, b)


def _fuzzy_partial(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return rfuzz.partial_ratio(a, b) / 100.0
    # Fallback: longer contains shorter
    if not a or not b:
        return 0.0
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    if shorter in longer:
        return 1.0
    return _simple_ratio(a, b)


def _fuzzy_wratio(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return rfuzz.WRatio(a, b) / 100.0
    return _fuzzy_ratio(a, b)


def _fuzzy_token_sort(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return rfuzz.token_sort_ratio(a, b) / 100.0
    return _token_set_ratio(a, b)


def _fuzzy_token_set(a: str, b: str) -> float:
    if HAS_RAPIDFUZZ:
        return rfuzz.token_set_ratio(a, b) / 100.0
    return _token_set_ratio(a, b)


def _numeric_overlap(nums_a: List[str], nums_b: List[str]) -> float:
    """Proportion of numeric tokens in a that also appear in b."""
    if not nums_a:
        return 0.0
    sa, sb = set(nums_a), set(nums_b)
    inter = len(sa & sb)
    return inter / len(sa)


def _len_diff_ratio(a: str, b: str) -> float:
    """Absolute length difference normalized by max length."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 0.0
    return abs(la - lb) / max(la, lb)


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction for a single pair
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Name features
    "name_exact",
    "name_norm_exact",
    "name_ratio",
    "name_partial",
    "name_wratio",
    "name_token_sort",
    "name_token_set",
    "name_jaccard",
    "name_len_diff",
    # Address features
    "addr_empty_s1",
    "addr_empty_s2",
    "addr_exact",
    "addr_norm_exact",
    "addr_ratio",
    "addr_partial",
    "addr_token_sort",
    "addr_token_set",
    "addr_jaccard",
    "addr_len_diff",
    # Numeric address
    "addr_numeric_overlap",
    "addr_numeric_exact",
    # Postal
    "postal_match",
    "postal_both_present",
    # Country
    "country_match",
    # Length features
    "name_len_s1",
    "name_len_s2",
    "addr_len_s1",
    "addr_len_s2",
    # Missingness
    "name_missing_s1",
    "name_missing_s2",
    "addr_missing_s1",
    "addr_missing_s2",
    # Source indicator (S2 vs S3)
    "is_s2",
    "is_s3",
]

N_FEATURES = len(FEATURE_NAMES)


def compute_features(
    row1: dict,
    row2: dict,
    blocking_sources: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Compute feature vector for a (s1_record, s23_record) pair.

    Parameters
    ----------
    row1 : s1 record dict (must have norm_name, norm_address, norm_country,
           name_toks, addr_toks, addr_numerics, postal_code)
    row2 : s23 record dict (same keys)
    blocking_sources : list of strategy names that generated this pair (for evidence)

    Returns
    -------
    np.ndarray of shape (N_FEATURES,), dtype float32
    """
    feat = np.zeros(N_FEATURES, dtype=np.float32)

    # ── NAME ──────────────────────────────────────────────────────────────
    n1 = str(row1.get("business_name", "") or "")
    n2 = str(row2.get("business_name", "") or "")
    nn1 = str(row1.get("norm_name", "") or normalize_name(n1))
    nn2 = str(row2.get("norm_name", "") or normalize_name(n2))
    nt1 = str(row1.get("name_toks", "") or "").split()
    nt2 = str(row2.get("name_toks", "") or "").split()

    feat[0] = float(n1.lower() == n2.lower())
    feat[1] = float(nn1 == nn2 and nn1 != "")
    feat[2] = _fuzzy_ratio(nn1, nn2)
    feat[3] = _fuzzy_partial(nn1, nn2)
    feat[4] = _fuzzy_wratio(nn1, nn2)
    feat[5] = _fuzzy_token_sort(nn1, nn2)
    feat[6] = _fuzzy_token_set(nn1, nn2)
    feat[7] = _jaccard(nt1, nt2)
    feat[8] = _len_diff_ratio(nn1, nn2)

    # ── ADDRESS ───────────────────────────────────────────────────────────
    a1 = str(row1.get("business_address", "") or "")
    a2 = str(row2.get("business_address", "") or "")
    na1 = str(row1.get("norm_address", "") or normalize_address(a1))
    na2 = str(row2.get("norm_address", "") or normalize_address(a2))
    at1 = str(row1.get("addr_toks", "") or "").split()
    at2 = str(row2.get("addr_toks", "") or "").split()
    an1 = str(row1.get("addr_numerics", "") or "").split()
    an2 = str(row2.get("addr_numerics", "") or "").split()
    p1 = str(row1.get("postal_code", "") or "")
    p2 = str(row2.get("postal_code", "") or "")

    addr_empty_s1 = float(not a1.strip())
    addr_empty_s2 = float(not a2.strip())

    feat[9] = addr_empty_s1
    feat[10] = addr_empty_s2
    feat[11] = float(a1.lower() == a2.lower() and a1 != "")
    feat[12] = float(na1 == na2 and na1 != "")
    feat[13] = _fuzzy_ratio(na1, na2) if not (addr_empty_s1 or addr_empty_s2) else 0.0
    feat[14] = _fuzzy_partial(na1, na2) if not (addr_empty_s1 or addr_empty_s2) else 0.0
    feat[15] = _fuzzy_token_sort(na1, na2) if not (addr_empty_s1 or addr_empty_s2) else 0.0
    feat[16] = _fuzzy_token_set(na1, na2) if not (addr_empty_s1 or addr_empty_s2) else 0.0
    feat[17] = _jaccard(at1, at2)
    feat[18] = _len_diff_ratio(na1, na2)

    feat[19] = _numeric_overlap(an1, an2)
    feat[20] = float(bool(an1) and bool(an2) and set(an1) == set(an2))

    feat[21] = float(p1 == p2 and p1 != "")
    feat[22] = float(p1 != "" and p2 != "")

    # ── COUNTRY ───────────────────────────────────────────────────────────
    c1 = str(row1.get("norm_country", "") or "")
    c2 = str(row2.get("norm_country", "") or "")
    feat[23] = float(c1 == c2 and c1 != "")

    # ── LENGTH ────────────────────────────────────────────────────────────
    feat[24] = float(len(nn1))
    feat[25] = float(len(nn2))
    feat[26] = float(len(na1))
    feat[27] = float(len(na2))

    # ── MISSINGNESS ───────────────────────────────────────────────────────
    feat[28] = float(not n1.strip())
    feat[29] = float(not n2.strip())
    feat[30] = addr_empty_s1
    feat[31] = addr_empty_s2

    # ── SOURCE ────────────────────────────────────────────────────────────
    eid2 = str(row2.get("entity_id", ""))
    feat[32] = float(eid2.startswith("S2-"))
    feat[33] = float(eid2.startswith("S3-"))

    return feat


# ─────────────────────────────────────────────────────────────────────────────
# Batch feature computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_matrix(
    pairs: List[Tuple[dict, dict]],
    show_progress: bool = False,
) -> np.ndarray:
    """
    Compute feature matrix for a list of (row1, row2) pairs.
    Returns np.ndarray of shape (n_pairs, N_FEATURES), dtype float32.
    """
    n = len(pairs)
    X = np.zeros((n, N_FEATURES), dtype=np.float32)
    for i, (r1, r2) in enumerate(pairs):
        X[i] = compute_features(r1, r2)
        if show_progress and (i + 1) % 100_000 == 0:
            print(f"  features: {i+1:,}/{n:,}")
    return X


def build_training_pairs(
    candidates: dict,           # {s1_id: set of s23_ids}
    ground_truth: dict,         # {s1_id: list of matched_ids}
    s1_index: dict,             # {entity_id: record_dict}
    s23_index: dict,            # {entity_id: record_dict}
    neg_ratio: float = 5.0,
    max_neg_per_s1: int = 50,   # cap negatives per S1 entity to limit skew
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, str]]]:
    """
    Build (X, y, pair_ids) for supervised training.

    Positives: all (s1, matched_id) pairs where matched_id appears in candidates.
    Hard negatives: per-entity-capped sample from non-matching candidates.

    neg_ratio: target global positives:negatives ratio
    max_neg_per_s1: hard cap on negatives from any single S1 entity
                    (prevents high-cardinality entities from dominating)

    Returns:
      X         : float32 array (n_pairs, N_FEATURES)
      y         : int array (n_pairs,) — 1=match, 0=no-match
      pair_ids  : list of (s1_id, s23_id) tuples in same order as X, y
    """
    import random
    rng = random.Random(rng_seed)

    # --- First pass: collect all pairs with labels ---
    all_raw = []   # list of (r1_dict, r2_dict, s1_id, s23_id, label)

    for s1_id, cand_set in candidates.items():
        true_set = set(ground_truth.get(s1_id, []))
        r1 = s1_index.get(s1_id)
        if r1 is None:
            continue

        # Positives: true matches in candidate set
        local_pos = []
        for mid in true_set:
            if mid in cand_set:
                r2 = s23_index.get(mid)
                if r2 is not None:
                    local_pos.append((r1, r2, s1_id, mid, 1))

        # Hard negatives: non-matching candidates
        hard_negs = [cid for cid in cand_set if cid not in true_set]
        rng.shuffle(hard_negs)
        # Per-entity cap: neg_ratio × local positives, but capped at max_neg_per_s1
        n_neg_local = min(
            max(1, round(len(local_pos) * neg_ratio)),
            max_neg_per_s1,
        )
        local_neg = []
        for cid in hard_negs[:n_neg_local]:
            r2 = s23_index.get(cid)
            if r2 is not None:
                local_neg.append((r1, r2, s1_id, cid, 0))

        all_raw.extend(local_pos)
        all_raw.extend(local_neg)

    # --- Global trim to maintain neg_ratio ---
    pos_raw = [p for p in all_raw if p[4] == 1]
    neg_raw = [p for p in all_raw if p[4] == 0]
    n_pos = len(pos_raw)
    max_total_neg = round(n_pos * neg_ratio)
    if len(neg_raw) > max_total_neg:
        rng.shuffle(neg_raw)
        neg_raw = neg_raw[:max_total_neg]

    all_raw = pos_raw + neg_raw
    rng.shuffle(all_raw)

    if not all_raw:
        return (
            np.zeros((0, N_FEATURES), dtype=np.float32),
            np.zeros(0, dtype=np.int32),
            [],
        )

    # --- Build feature matrix ---
    pair_ids = [(p[2], p[3]) for p in all_raw]
    y = np.array([p[4] for p in all_raw], dtype=np.int32)
    X = compute_feature_matrix([(p[0], p[1]) for p in all_raw], show_progress=True)

    return X, y, pair_ids

