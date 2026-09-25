"""
src/decision_optimizer.py
TASK 14 PHASE F: Entity-Level Decision Layer & Distribution-Aware Multi-Match Controller.

Features:
- Singleton confidence gating: suppresses false merges on singletons when top candidate confidence is low.
- Score gap gating: prevents distant distractor candidates from being linked when a dominant true link exists.
- Ground-truth calibrated match cap: respects the empirical training cap (max 11 matches per S1).
- Official competition macro F0.5 evaluation.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd


def evaluate_macro_f05(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, List[str]],
) -> Dict[str, Any]:
    """
    Evaluates exact competition macro F0.5 per S1 entity.
    """
    per_entity = {}
    for s1_id, true_list in ground_truth.items():
        true_set = set(true_list)
        pred_set = set(predictions.get(s1_id, []))

        if len(true_set) == 0:
            # Singleton entity
            if len(pred_set) == 0:
                p, r, f = 1.0, 1.0, 1.0
            else:
                p, r, f = 0.0, 1.0, 0.0  # False merge penalty
        else:
            if len(pred_set) == 0:
                p, r, f = 0.0, 0.0, 0.0  # Completely missed
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

    # Singleton diagnostics
    sing_ids = [s1 for s1, ms in ground_truth.items() if len(ms) == 0]
    sing_correct = sum(1 for s1 in sing_ids if len(predictions.get(s1, [])) == 0)
    sing_acc = sing_correct / len(sing_ids) if sing_ids else 1.0
    false_merges = len(sing_ids) - sing_correct

    # Match distribution diagnostics
    n_preds = [len(predictions.get(s1, [])) for s1 in ground_truth.keys()]
    avg_matches = np.mean(n_preds)
    max_matches = max(n_preds) if n_preds else 0
    p50_matches = np.percentile(n_preds, 50)
    p95_matches = np.percentile(n_preds, 95)
    entities_over_11 = sum(1 for np_ in n_preds if np_ > 11)

    return {
        "macro_f05": round(macro_f05, 4),
        "precision": round(macro_p, 4),
        "recall": round(macro_r, 4),
        "singleton_accuracy": round(sing_acc, 4),
        "false_merges": int(false_merges),
        "avg_predicted_matches": round(float(avg_matches), 2),
        "max_predicted_matches": int(max_matches),
        "p50_predicted_matches": round(float(p50_matches), 1),
        "p95_predicted_matches": round(float(p95_matches), 1),
        "entities_over_11": int(entities_over_11),
    }


def apply_decision_rules(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    sample_ids: List[str],
    base_threshold: float = 0.64,
    singleton_gate: float = 0.80,
    score_gap: float = 0.20,
    match_cap: int = 11,
) -> Dict[str, List[str]]:
    """
    Applies calibrated entity-level decision logic:
    1. Sorts candidates by score descending.
    2. If top_score < singleton_gate -> predict [] (singleton).
    3. For candidates with score >= base_threshold AND (top_score - score) <= score_gap:
       include up to match_cap (max 11).
    """
    preds = {}
    for s1_id in sample_ids:
        cands = sorted(candidate_scores.get(s1_id, []), key=lambda x: x[1], reverse=True)
        if not cands or cands[0][1] < singleton_gate:
            preds[s1_id] = []
        else:
            top_score = cands[0][1]
            selected = [
                cid for cid, s in cands
                if s >= base_threshold and (top_score - s) <= score_gap
            ]
            if len(selected) > match_cap:
                selected = selected[:match_cap]
            preds[s1_id] = selected
    return preds


def grid_search_decision_layer(
    candidate_scores: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, List[str]],
    sample_ids: List[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Grid searches over decision hyperparameters to maximize macro F0.5.
    """
    singleton_gates = [0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90]
    base_thresholds = [0.55, 0.60, 0.64, 0.70, 0.75]
    score_gaps = [0.10, 0.15, 0.20, 0.25]
    match_caps = [8, 10, 11]

    records = []
    best_f05 = -1.0
    best_config = None

    for gate in singleton_gates:
        for thr in base_thresholds:
            if thr > gate:
                continue
            for gap in score_gaps:
                for cap in match_caps:
                    preds = apply_decision_rules(
                        candidate_scores, sample_ids,
                        base_threshold=thr, singleton_gate=gate,
                        score_gap=gap, match_cap=cap,
                    )
                    metrics = evaluate_macro_f05(preds, ground_truth)
                    rec = {
                        "singleton_gate": gate,
                        "base_threshold": thr,
                        "score_gap": gap,
                        "match_cap": cap,
                        **metrics,
                    }
                    records.append(rec)
                    if metrics["macro_f05"] > best_f05:
                        best_f05 = metrics["macro_f05"]
                        best_config = rec

    df_grid = pd.DataFrame(records)
    return best_config, df_grid
