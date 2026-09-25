"""
src/metrics.py
==============
All evaluation metrics for the Amazon Business Entity Resolution challenge.

Metric: macro-averaged F_β (β = 0.5), precision-heavy.

Formula
-------
    F_0.5 = (1 + β²) × P × R / (β² × P + R)
    With β = 0.5 → β² = 0.25 → F_0.5 = 1.25 × P × R / (0.25 × P + R)

Per-entity rules (from the problem statement)
---------------------------------------------
- Singleton truth + empty prediction  → F = 1.0
- Singleton truth + non-empty prediction → F = 0.0  (precision = 0)
- Non-singleton truth + empty prediction → F = 0.0  (recall = 0, unless empty truth)

Macro average: mean of per-entity F_0.5 across ALL S1 entities.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Set, Tuple

# β for F_β
BETA: float = 0.5
BETA_SQ: float = BETA ** 2


# ──────────────────────────────────────────────────────────────────────────────
# Low-level: single-entity F_β
# ──────────────────────────────────────────────────────────────────────────────

def entity_precision_recall_f(
    predicted: Iterable[str],
    truth: Iterable[str],
    beta_sq: float = BETA_SQ,
) -> Tuple[float, float, float]:
    """
    Compute (precision, recall, F_β) for a single Source 1 entity.

    Parameters
    ----------
    predicted : iterable of matched IDs from the model
    truth     : iterable of ground-truth matched IDs
    beta_sq   : β² (default 0.25 for F_0.5)

    Returns
    -------
    (precision, recall, f_beta) — all in [0, 1]

    Edge cases
    ----------
    - Both empty          → (1.0, 1.0, 1.0)   # correct singleton prediction
    - truth non-empty, predicted empty → (1.0, 0.0, 0.0)
    - truth empty, predicted non-empty → (0.0, 1.0, 0.0)
    """
    pred_set: Set[str] = set(predicted)
    true_set: Set[str] = set(truth)

    if not pred_set and not true_set:
        # Correct singleton prediction
        return 1.0, 1.0, 1.0

    tp = len(pred_set & true_set)

    # Precision: how many predictions are correct
    precision = tp / len(pred_set) if pred_set else 1.0
    # Recall: how many ground-truth links we found
    recall = tp / len(true_set) if true_set else 1.0

    denom = beta_sq * precision + recall
    f_beta = ((1 + beta_sq) * precision * recall / denom) if denom > 0 else 0.0

    return precision, recall, f_beta


def entity_f05(predicted: Iterable[str], truth: Iterable[str]) -> float:
    """Convenience wrapper — returns only F_0.5 for a single entity."""
    _, _, f = entity_precision_recall_f(predicted, truth)
    return f


# ──────────────────────────────────────────────────────────────────────────────
# Macro metrics over a full prediction dict
# ──────────────────────────────────────────────────────────────────────────────

Predictions = Dict[str, List[str]]  # {s1_id: [matched_ids]}
GroundTruth = Dict[str, List[str]]  # {s1_id: [matched_ids]}


def macro_precision_recall_f05(
    predictions: Predictions,
    ground_truth: GroundTruth,
) -> Dict[str, float]:
    """
    Compute macro-averaged Precision, Recall, and F_0.5.

    All S1 entities in `ground_truth` are included in the average.
    Missing S1 IDs in `predictions` count as empty predictions.

    Returns
    -------
    dict with keys: precision, recall, f05, n_entities
    """
    per_entity = per_entity_f05(predictions, ground_truth)
    n = len(per_entity)
    if n == 0:
        return {"precision": 0.0, "recall": 0.0, "f05": 0.0, "n_entities": 0}

    prec_sum = rec_sum = f_sum = 0.0
    for s1_id, (p, r, f) in per_entity.items():
        prec_sum += p
        rec_sum += r
        f_sum += f

    return {
        "precision": round(prec_sum / n, 6),
        "recall": round(rec_sum / n, 6),
        "f05": round(f_sum / n, 6),
        "n_entities": n,
    }


def per_entity_f05(
    predictions: Predictions,
    ground_truth: GroundTruth,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Return per-entity (precision, recall, f05) for every S1 ID in ground_truth.

    Parameters
    ----------
    predictions  : {s1_id: [predicted_matched_ids]}
    ground_truth : {s1_id: [true_matched_ids]}

    Returns
    -------
    {s1_id: (precision, recall, f05)}
    """
    result: Dict[str, Tuple[float, float, float]] = {}
    for s1_id, true_matches in ground_truth.items():
        pred_matches = predictions.get(s1_id, [])
        p, r, f = entity_precision_recall_f(pred_matches, true_matches)
        result[s1_id] = (p, r, f)
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Singleton performance
# ──────────────────────────────────────────────────────────────────────────────

def singleton_performance(
    predictions: Predictions,
    ground_truth: GroundTruth,
) -> Dict[str, float]:
    """
    Compute performance specifically on singleton S1 entities
    (those with no ground-truth matches).

    Returns accuracy (fraction of singletons correctly predicted as empty).
    """
    singleton_ids = [s1 for s1, ms in ground_truth.items() if len(ms) == 0]
    if not singleton_ids:
        return {"singleton_count": 0, "singleton_accuracy": None}

    correct = sum(
        1 for s1 in singleton_ids
        if len(predictions.get(s1, [])) == 0
    )
    false_merge = len(singleton_ids) - correct

    return {
        "singleton_count": len(singleton_ids),
        "singleton_correct": correct,
        "singleton_false_merge": false_merge,
        "singleton_accuracy": round(correct / len(singleton_ids), 6),
        # Avg F0.5 on singletons (each correctly predicted singleton → 1.0)
        "singleton_mean_f05": round(correct / len(singleton_ids), 6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Candidate recall
# ──────────────────────────────────────────────────────────────────────────────

def candidate_recall(
    candidates: Dict[str, List[str]],
    ground_truth: GroundTruth,
) -> Dict[str, float]:
    """
    Candidate recall = fraction of true positive links that appear
    anywhere in the candidate set.

    Parameters
    ----------
    candidates   : {s1_id: [candidate_matched_ids]} — blocking output
    ground_truth : {s1_id: [true_matched_ids]}

    Returns
    -------
    dict with keys: candidate_recall, total_true_links,
                    covered_links, missing_links, n_entities_with_matches
    """
    total_true = 0
    covered = 0
    entities_with_matches = 0

    for s1_id, true_matches in ground_truth.items():
        if not true_matches:
            continue
        entities_with_matches += 1
        total_true += len(true_matches)
        cand_set = set(candidates.get(s1_id, []))
        covered += sum(1 for m in true_matches if m in cand_set)

    recall = covered / total_true if total_true > 0 else 1.0
    return {
        "candidate_recall": round(recall, 6),
        "total_true_links": total_true,
        "covered_links": covered,
        "missing_links": total_true - covered,
        "n_entities_with_matches": entities_with_matches,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: full evaluation report
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    predictions: Predictions,
    ground_truth: GroundTruth,
    candidates: Dict[str, List[str]] | None = None,
    label: str = "eval",
) -> Dict:
    """
    Full evaluation: macro metrics + singleton perf + optional candidate recall.
    """
    macro = macro_precision_recall_f05(predictions, ground_truth)
    singletons = singleton_performance(predictions, ground_truth)

    report = {
        "label": label,
        "macro": macro,
        "singletons": singletons,
    }

    if candidates is not None:
        cand_rec = candidate_recall(candidates, ground_truth)
        report["candidate_recall"] = cand_rec

    return report


def print_evaluation(report: Dict) -> None:
    """Pretty-print an evaluation report."""
    lbl = report.get("label", "eval")
    print(f"\n{'='*60}")
    print(f"Evaluation: {lbl}")
    print(f"{'='*60}")
    macro = report.get("macro", {})
    print(f"  Macro Precision : {macro.get('precision', 0):.4f}")
    print(f"  Macro Recall    : {macro.get('recall', 0):.4f}")
    print(f"  Macro F_0.5     : {macro.get('f05', 0):.4f}  (n={macro.get('n_entities', 0):,})")

    sg = report.get("singletons", {})
    print(f"\n  Singletons      : {sg.get('singleton_count', 0):,}")
    print(f"  Singleton Acc.  : {sg.get('singleton_accuracy', 'N/A')}")

    if "candidate_recall" in report:
        cr = report["candidate_recall"]
        print(f"\n  Candidate Recall: {cr.get('candidate_recall', 0):.4f}  "
              f"(covered {cr.get('covered_links', 0):,}/{cr.get('total_true_links', 0):,})")
    print(f"{'='*60}")
