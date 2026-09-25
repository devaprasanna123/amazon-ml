"""
src/blocking.py
===============
High-recall candidate generation for Business Entity Resolution.

Strategies implemented (measured independently and as union):
  1. exact_norm_name   — exact normalized name match
  2. name_token        — inverted index on name tokens
  3. name_ngram_tfidf  — TF-IDF BM25-like on name characters/tokens
  4. addr_token        — inverted index on address tokens
  5. addr_numeric      — shared numeric address tokens
  6. country_name      — (country, first_name_token) block
  7. tfidf_name        — sparse TF-IDF cosine retrieval on name
  8. tfidf_addr        — sparse TF-IDF cosine retrieval on address

Each strategy is implemented as a function:
  candidates: Dict[str, Set[str]]  {s1_id: {s2_id, s3_id, ...}}

The union is computed and returned for the matching model.
"""

from __future__ import annotations

import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing import (
    normalize_name, name_tokens, address_tokens,
    address_numeric_tokens, normalize_country, normalize_address,
)

# ─────────────────────────────────────────────────────────────────────────────
# Types
# ─────────────────────────────────────────────────────────────────────────────
Candidates = Dict[str, Set[str]]   # {s1_id: {s23_id, ...}}


# ─────────────────────────────────────────────────────────────────────────────
# Inverted index builder
# ─────────────────────────────────────────────────────────────────────────────

def build_inverted_index(
    df: pd.DataFrame,
    key_col: str,
    id_col: str = "entity_id",
    split_tokens: bool = True,
) -> Dict[str, Set[str]]:
    """
    Build inverted index: key → set of entity_ids.
    If split_tokens=True, key_col is treated as space-separated tokens.
    If split_tokens=False, key_col is used verbatim.
    """
    index: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df.iterrows():
        eid = row[id_col]
        val = str(row[key_col]) if row[key_col] else ""
        if not val:
            continue
        if split_tokens:
            for tok in val.split():
                if tok:
                    index[tok].add(eid)
        else:
            index[val].add(eid)
    return dict(index)


# ─────────────────────────────────────────────────────────────────────────────
# Individual blocking functions
# ─────────────────────────────────────────────────────────────────────────────

def block_exact_norm_name(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
) -> Candidates:
    """Exact match on normalized business name."""
    # Build target index: norm_name → set of s23 ids
    target: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df_s23.iterrows():
        if row.get("norm_name"):
            target[row["norm_name"]].add(row["entity_id"])

    cands: Candidates = {}
    for _, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        nn = row.get("norm_name", "")
        if nn and nn in target:
            cands[s1_id] = set(target[nn])
        else:
            cands[s1_id] = set()
    return cands


def block_name_tokens(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    min_token_len: int = 3,
    max_candidates_per_s1: int = 200,
) -> Candidates:
    """
    Token-based name blocking.
    S1 entity shares a token with a S23 entity → candidate.
    Only tokens of length >= min_token_len are used.
    """
    # Build target inverted index
    target_idx: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df_s23.iterrows():
        eid = row["entity_id"]
        for tok in str(row.get("name_toks", "")).split():
            if len(tok) >= min_token_len:
                target_idx[tok].add(eid)

    cands: Candidates = {}
    for _, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        found: Set[str] = set()
        for tok in str(row.get("name_toks", "")).split():
            if len(tok) >= min_token_len and tok in target_idx:
                found.update(target_idx[tok])
                if len(found) >= max_candidates_per_s1:
                    break
        cands[s1_id] = found
    return cands


def block_addr_tokens(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    min_token_len: int = 4,
    max_candidates_per_s1: int = 200,
) -> Candidates:
    """Token-based address blocking (meaningful tokens only)."""
    target_idx: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df_s23.iterrows():
        eid = row["entity_id"]
        for tok in str(row.get("addr_toks", "")).split():
            if len(tok) >= min_token_len:
                target_idx[tok].add(eid)

    cands: Candidates = {}
    for _, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        found: Set[str] = set()
        for tok in str(row.get("addr_toks", "")).split():
            if len(tok) >= min_token_len and tok in target_idx:
                found.update(target_idx[tok])
                if len(found) >= max_candidates_per_s1:
                    break
        cands[s1_id] = found
    return cands


def block_addr_numerics(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    min_digit_len: int = 3,
) -> Candidates:
    """
    Numeric address token blocking.
    Entities sharing a numeric component (house number, PIN, ZIP) are candidates.
    Only numeric tokens with >= min_digit_len digits to avoid single-digit noise.
    """
    target_idx: Dict[str, Set[str]] = defaultdict(set)
    for _, row in df_s23.iterrows():
        eid = row["entity_id"]
        for num in str(row.get("addr_numerics", "")).split():
            if len(num) >= min_digit_len:
                target_idx[num].add(eid)

    cands: Candidates = {}
    for _, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        found: Set[str] = set()
        for num in str(row.get("addr_numerics", "")).split():
            if len(num) >= min_digit_len and num in target_idx:
                found.update(target_idx[num])
        cands[s1_id] = found
    return cands


def block_country_first_token(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    max_candidates_per_s1: int = 300,
) -> Candidates:
    """
    (country, first_name_token) blocking.
    Reduces search space while keeping country-aware blocks.
    Only uses first token of name (highest-entropy blocker).
    """
    target_idx: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for _, row in df_s23.iterrows():
        eid = row["entity_id"]
        country = row.get("norm_country", "")
        toks = str(row.get("name_toks", "")).split()
        if toks and country:
            key = (country, toks[0])
            target_idx[key].add(eid)

    cands: Candidates = {}
    for _, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        country = row.get("norm_country", "")
        toks = str(row.get("name_toks", "")).split()
        found: Set[str] = set()
        if toks and country:
            key = (country, toks[0])
            if key in target_idx:
                found = set(list(target_idx[key])[:max_candidates_per_s1])
        cands[s1_id] = found
    return cands


def block_tfidf(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    text_col: str = "norm_name",
    top_k: int = 50,
    analyzer: str = "word",
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 200_000,
    batch_size: int = 1000,
    label: str = "tfidf",
) -> Candidates:
    """
    Sparse TF-IDF cosine retrieval.
    Builds index on S23 records, queries S1 in batches.
    Returns top_k candidates per S1 entity.
    """
    s23_texts = df_s23[text_col].fillna("").tolist()
    s23_ids = df_s23["entity_id"].tolist()
    s1_texts = df_s1[text_col].fillna("").tolist()
    s1_ids = df_s1["entity_id"].tolist()

    # Filter out empty
    s23_valid = [(i, t, eid) for i, (t, eid) in enumerate(zip(s23_texts, s23_ids)) if t.strip()]
    s1_valid = [(i, t, eid) for i, (t, eid) in enumerate(zip(s1_texts, s1_ids)) if t.strip()]

    if not s23_valid or not s1_valid:
        return {eid: set() for eid in s1_ids}

    # Build TF-IDF on S23
    _, s23_texts_clean, s23_ids_clean = zip(*s23_valid)
    _, s1_texts_clean, s1_ids_clean = zip(*s1_valid)

    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
    )
    s23_matrix = vectorizer.fit_transform(list(s23_texts_clean))
    s1_matrix = vectorizer.transform(list(s1_texts_clean))

    cands: Candidates = {eid: set() for eid in s1_ids}

    # Batch cosine similarity
    n_s1 = s1_matrix.shape[0]
    for start in range(0, n_s1, batch_size):
        end = min(start + batch_size, n_s1)
        batch = s1_matrix[start:end]
        sims = cosine_similarity(batch, s23_matrix)  # shape: (batch, n_s23)
        for i, sim_row in enumerate(sims):
            s1_id = s1_ids_clean[start + i]
            # Get top_k indices
            top_idx = np.argpartition(sim_row, -min(top_k, len(sim_row)))[-min(top_k, len(sim_row)):]
            top_idx = top_idx[sim_row[top_idx] > 0]
            for idx in top_idx:
                cands[s1_id].add(s23_ids_clean[idx])

    return cands


def block_tfidf_char_ngram(
    df_s1: pd.DataFrame,
    df_s23: pd.DataFrame,
    text_col: str = "norm_name",
    top_k: int = 30,
    ngram_range: Tuple[int, int] = (3, 4),
    max_features: int = 150_000,
    batch_size: int = 1000,
) -> Candidates:
    """Character n-gram TF-IDF retrieval — catches abbreviations and typos."""
    return block_tfidf(
        df_s1, df_s23,
        text_col=text_col,
        top_k=top_k,
        analyzer="char_wb",
        ngram_range=ngram_range,
        max_features=max_features,
        batch_size=batch_size,
        label="char_ngram",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Union of all strategies
# ─────────────────────────────────────────────────────────────────────────────

def union_candidates(*candidate_dicts: Candidates) -> Candidates:
    """Compute the union of multiple candidate dictionaries."""
    if not candidate_dicts:
        return {}
    all_s1 = set()
    for d in candidate_dicts:
        all_s1.update(d.keys())
    result: Candidates = {}
    for s1_id in all_s1:
        merged: Set[str] = set()
        for d in candidate_dicts:
            merged.update(d.get(s1_id, set()))
        result[s1_id] = merged
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_candidates(
    candidates: Candidates,
    ground_truth: Dict[str, list],
    total_s23: int,
    label: str = "blocking",
) -> dict:
    """
    Compute blocking statistics.

    Returns:
      candidate_recall    - fraction of true links covered
      mean_candidates     - average candidates per S1 entity
      total_candidates    - total candidate pairs
      reduction_ratio     - 1 - (candidates / (n_s1 * n_s23))
      n_s1                - number of S1 entities evaluated
    """
    n_s1 = len(candidates)
    total_cands = sum(len(v) for v in candidates.values())

    covered = 0
    total_true = 0
    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        total_true += len(true_matches)
        cand_set = candidates.get(s1_id, set())
        covered += sum(1 for m in true_matches if m in cand_set)

    recall = covered / total_true if total_true > 0 else 1.0
    mean_cands = total_cands / n_s1 if n_s1 > 0 else 0
    max_pairs = n_s1 * total_s23
    reduction = 1 - (total_cands / max_pairs) if max_pairs > 0 else 0

    return {
        "label": label,
        "candidate_recall": round(recall, 6),
        "mean_candidates_per_s1": round(mean_cands, 2),
        "total_candidates": total_cands,
        "reduction_ratio": round(reduction, 6),
        "n_s1": n_s1,
        "covered_true_links": covered,
        "total_true_links": total_true,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full blocking pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_blocking_pipeline(
    df_s1: pd.DataFrame,
    df_s2: pd.DataFrame,
    df_s3: pd.DataFrame,
    ground_truth: Optional[Dict[str, list]] = None,
    cache_dir: Optional[Path] = None,
    top_k_tfidf: int = 50,
    top_k_char: int = 30,
    verbose: bool = True,
) -> Tuple[Candidates, list]:
    """
    Run all blocking strategies and return the union candidates.

    Parameters
    ----------
    df_s1, df_s2, df_s3 : normalized DataFrames (with norm_name, name_toks, etc.)
    ground_truth : {s1_id: [matched_ids]} for recall evaluation (optional)
    cache_dir    : if set, cache/restore per-strategy candidates to disk
    top_k_tfidf  : TF-IDF word top-k per query
    top_k_char   : char-ngram TF-IDF top-k per query

    Returns
    -------
    union_cands : Candidates dict
    results     : list of per-strategy metric dicts
    """
    df_s23 = pd.concat([df_s2, df_s3], ignore_index=True)
    total_s23 = len(df_s23)

    if verbose:
        print(f"Blocking: {len(df_s1):,} S1 × {total_s23:,} S23")

    strategies = [
        ("exact_norm_name",   lambda: block_exact_norm_name(df_s1, df_s23)),
        ("name_tokens",       lambda: block_name_tokens(df_s1, df_s23)),
        ("addr_tokens",       lambda: block_addr_tokens(df_s1, df_s23)),
        ("addr_numerics",     lambda: block_addr_numerics(df_s1, df_s23)),
        ("country_first_tok", lambda: block_country_first_token(df_s1, df_s23)),
        ("tfidf_name_word",   lambda: block_tfidf(
            df_s1, df_s23, text_col="norm_name", top_k=top_k_tfidf,
            analyzer="word", ngram_range=(1, 2), label="tfidf_name_word")),
        ("tfidf_name_char",   lambda: block_tfidf_char_ngram(
            df_s1, df_s23, text_col="norm_name", top_k=top_k_char)),
        ("tfidf_addr_word",   lambda: block_tfidf(
            df_s1, df_s23, text_col="norm_address", top_k=top_k_tfidf,
            analyzer="word", ngram_range=(1, 2), label="tfidf_addr_word")),
    ]

    per_strategy: Dict[str, Candidates] = {}
    results = []

    for name, fn in strategies:
        cache_path = (cache_dir / f"cands_{name}.pkl") if cache_dir else None

        if cache_path and cache_path.exists():
            if verbose:
                print(f"  [{name}] loading from cache …")
            with open(cache_path, "rb") as f:
                cands = pickle.load(f)
        else:
            if verbose:
                print(f"  [{name}] running …", end="", flush=True)
            t0 = time.time()
            cands = fn()
            elapsed = time.time() - t0
            if verbose:
                print(f" {elapsed:.1f}s")
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump(cands, f)

        per_strategy[name] = cands

        if ground_truth is not None:
            # Only evaluate on S1 IDs that appear in ground_truth
            eval_gt = {s1: ground_truth[s1] for s1 in cands if s1 in ground_truth}
            eval_cands = {s1: cands[s1] for s1 in eval_gt}
            stats = evaluate_candidates(eval_cands, eval_gt, total_s23, label=name)
            stats["runtime_s"] = elapsed if "elapsed" in dir() else None
            results.append(stats)
            if verbose:
                print(f"       recall={stats['candidate_recall']:.4f}  "
                      f"mean_cands={stats['mean_candidates_per_s1']:.1f}  "
                      f"reduction={stats['reduction_ratio']:.4f}")

    # Union
    if verbose:
        print("  [union] computing …", end="", flush=True)
    t0 = time.time()
    union_cands = union_candidates(*per_strategy.values())
    elapsed_union = time.time() - t0
    if verbose:
        print(f" {elapsed_union:.1f}s")

    if cache_dir:
        with open(cache_dir / "cands_union.pkl", "wb") as f:
            pickle.dump(union_cands, f)

    if ground_truth is not None:
        eval_gt = {s1: ground_truth[s1] for s1 in union_cands if s1 in ground_truth}
        eval_cands = {s1: union_cands[s1] for s1 in eval_gt}
        union_stats = evaluate_candidates(eval_cands, eval_gt, total_s23, label="UNION")
        union_stats["runtime_s"] = elapsed_union
        results.append(union_stats)
        if verbose:
            print(f"  [UNION] recall={union_stats['candidate_recall']:.4f}  "
                  f"mean_cands={union_stats['mean_candidates_per_s1']:.1f}  "
                  f"reduction={union_stats['reduction_ratio']:.4f}")

    return union_cands, results
