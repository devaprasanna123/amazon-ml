"""
tests/test_metrics.py
=====================
Unit tests for src/metrics.py

Tests include:
- The exact worked example from the problem statement (F_0.5 = 0.714)
- Singleton edge cases
- Macro averaging
- Candidate recall
- Singleton performance utility
"""

import sys
from pathlib import Path

# Make src importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.metrics import (
    entity_precision_recall_f,
    entity_f05,
    macro_precision_recall_f05,
    per_entity_f05,
    singleton_performance,
    candidate_recall,
    evaluate,
    BETA_SQ,
)


# ──────────────────────────────────────────────────────────────────────────────
# Problem-statement example
# ──────────────────────────────────────────────────────────────────────────────

class TestProblemStatementExample:
    """
    From the README:
        Predicted : [S2-00047, S2-00193, S3-00812]
        Truth     : [S2-00047, S3-00812]
        Precision = 2/3
        Recall    = 2/2 = 1.0
        F_0.5     = (1.25 × 0.667 × 1.0) / (0.25 × 0.667 + 1.0) ≈ 0.714
    """

    def test_precision(self):
        pred = ["S2-00047", "S2-00193", "S3-00812"]
        truth = ["S2-00047", "S3-00812"]
        p, r, f = entity_precision_recall_f(pred, truth)
        assert abs(p - 2 / 3) < 1e-6, f"Expected precision 2/3, got {p}"

    def test_recall(self):
        pred = ["S2-00047", "S2-00193", "S3-00812"]
        truth = ["S2-00047", "S3-00812"]
        p, r, f = entity_precision_recall_f(pred, truth)
        assert abs(r - 1.0) < 1e-6, f"Expected recall 1.0, got {r}"

    def test_f05_value(self):
        """F_0.5 ≈ 0.714 per problem statement example."""
        pred = ["S2-00047", "S2-00193", "S3-00812"]
        truth = ["S2-00047", "S3-00812"]
        p, r, f = entity_precision_recall_f(pred, truth)

        # Manual calculation: (1 + 0.25) × (2/3) × 1.0 / (0.25 × 2/3 + 1.0)
        precision = 2 / 3
        recall = 1.0
        expected = (1 + BETA_SQ) * precision * recall / (BETA_SQ * precision + recall)
        assert abs(expected - 0.71428571) < 1e-6, f"Formula sanity: {expected}"
        assert abs(f - expected) < 1e-6, f"F_0.5={f} expected≈{expected}"

    def test_f05_approximately_714(self):
        """Match README's stated value of 0.714."""
        pred = ["S2-00047", "S2-00193", "S3-00812"]
        truth = ["S2-00047", "S3-00812"]
        f = entity_f05(pred, truth)
        assert abs(f - 0.714) < 0.001, f"Expected ~0.714, got {f:.4f}"


# ──────────────────────────────────────────────────────────────────────────────
# Singleton edge cases
# ──────────────────────────────────────────────────────────────────────────────

class TestSingletonEdgeCases:
    def test_both_empty_is_perfect(self):
        """Correctly predicting 'no match' → F=1.0"""
        p, r, f = entity_precision_recall_f([], [])
        assert p == 1.0
        assert r == 1.0
        assert f == 1.0

    def test_empty_pred_nonempty_truth_is_zero(self):
        """Missing all matches → F=0.0 (recall=0)"""
        p, r, f = entity_precision_recall_f([], ["S2-001"])
        assert r == 0.0
        assert f == 0.0

    def test_nonempty_pred_empty_truth_is_zero(self):
        """False merge on singleton entity → F=0.0 (precision=0)"""
        p, r, f = entity_precision_recall_f(["S2-001"], [])
        assert p == 0.0
        assert f == 0.0

    def test_nonempty_pred_empty_truth_recall_is_one(self):
        """When truth is empty and pred is non-empty, recall=1 by convention but f=0."""
        p, r, f = entity_precision_recall_f(["S2-001"], [])
        # recall = tp / |truth| → truth empty, convention: precision=0 because |pred|>0, |tp|=0
        assert p == 0.0
        assert f == 0.0

    def test_perfect_prediction(self):
        pred = ["S2-001", "S3-002"]
        truth = ["S2-001", "S3-002"]
        p, r, f = entity_precision_recall_f(pred, truth)
        assert p == 1.0
        assert r == 1.0
        assert f == 1.0

    def test_order_invariant(self):
        """Prediction order should not matter."""
        f1 = entity_f05(["S2-001", "S3-002"], ["S3-002", "S2-001"])
        f2 = entity_f05(["S3-002", "S2-001"], ["S2-001", "S3-002"])
        assert abs(f1 - f2) < 1e-9

    def test_duplicate_in_prediction_treated_as_set(self):
        """Duplicates in predictions should be deduplicated."""
        f_dup = entity_f05(["S2-001", "S2-001"], ["S2-001"])
        f_clean = entity_f05(["S2-001"], ["S2-001"])
        assert abs(f_dup - f_clean) < 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Precision / Recall formulas
# ──────────────────────────────────────────────────────────────────────────────

class TestPrecisionRecall:
    def test_all_correct(self):
        p, r, f = entity_precision_recall_f(["A", "B"], ["A", "B"])
        assert p == 1.0 and r == 1.0 and f == 1.0

    def test_half_precision(self):
        p, r, f = entity_precision_recall_f(["A", "B"], ["A"])
        assert abs(p - 0.5) < 1e-9
        assert abs(r - 1.0) < 1e-9

    def test_half_recall(self):
        p, r, f = entity_precision_recall_f(["A"], ["A", "B"])
        assert abs(p - 1.0) < 1e-9
        assert abs(r - 0.5) < 1e-9

    def test_no_overlap(self):
        p, r, f = entity_precision_recall_f(["C", "D"], ["A", "B"])
        assert p == 0.0 and r == 0.0 and f == 0.0

    def test_f05_weights_precision_more_than_recall(self):
        """
        F_0.5 should give a higher score when precision is high and recall is low,
        compared to the reverse (since β<1 weights precision more).
        """
        # High precision, low recall
        p1, r1, f_high_p = entity_precision_recall_f(["A"], ["A", "B", "C", "D"])
        # Low precision, high recall
        p2, r2, f_high_r = entity_precision_recall_f(["A", "B", "C", "D"], ["A"])
        # With β=0.5, precision is weighted 4× more than recall.
        # High-precision / low-recall case (pred=["A"], truth=4 items) should score
        # higher than low-precision / high-recall case (pred=4 items, truth=["A"]).
        assert f_high_p > f_high_r, (
            f"High-precision case ({f_high_p:.4f}) should outperform "
            f"high-recall case ({f_high_r:.4f}) under F_0.5"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Macro averaging
# ──────────────────────────────────────────────────────────────────────────────

class TestMacroAveraging:
    def test_single_entity(self):
        pred = {"S1-001": ["S2-001"]}
        truth = {"S1-001": ["S2-001"]}
        result = macro_precision_recall_f05(pred, truth)
        assert abs(result["f05"] - 1.0) < 1e-9

    def test_two_entities_average(self):
        pred = {
            "S1-001": ["S2-001", "S2-002", "S3-003"],  # from problem example → ~0.714
            "S1-002": [],                               # correct singleton → 1.0
        }
        truth = {
            "S1-001": ["S2-001", "S3-003"],
            "S1-002": [],
        }
        result = macro_precision_recall_f05(pred, truth)
        f1 = entity_f05(["S2-001", "S2-002", "S3-003"], ["S2-001", "S3-003"])
        f2 = entity_f05([], [])
        expected_macro = (f1 + f2) / 2
        assert abs(result["f05"] - expected_macro) < 1e-6

    def test_missing_prediction_counts_as_empty(self):
        """A missing S1 in predictions should be treated as predicting []."""
        pred = {}   # no predictions at all
        truth = {"S1-001": ["S2-001"]}
        result = macro_precision_recall_f05(pred, truth)
        # recall = 0, f = 0
        assert result["f05"] == 0.0

    def test_empty_ground_truth(self):
        result = macro_precision_recall_f05({}, {})
        assert result["n_entities"] == 0
        assert result["f05"] == 0.0

    def test_n_entities_correct(self):
        truth = {f"S1-{i:03d}": [] for i in range(10)}
        pred = {f"S1-{i:03d}": [] for i in range(10)}
        result = macro_precision_recall_f05(pred, truth)
        assert result["n_entities"] == 10


# ──────────────────────────────────────────────────────────────────────────────
# Per-entity output
# ──────────────────────────────────────────────────────────────────────────────

class TestPerEntityF05:
    def test_keys_match_ground_truth(self):
        pred = {"S1-001": ["S2-001"], "S1-002": []}
        truth = {"S1-001": ["S2-001"], "S1-002": []}
        per = per_entity_f05(pred, truth)
        assert set(per.keys()) == {"S1-001", "S1-002"}

    def test_values_are_tuples_of_three(self):
        pred = {"S1-001": ["S2-001"]}
        truth = {"S1-001": ["S2-001"]}
        per = per_entity_f05(pred, truth)
        for v in per.values():
            assert len(v) == 3
            assert all(isinstance(x, float) for x in v)


# ──────────────────────────────────────────────────────────────────────────────
# Singleton performance
# ──────────────────────────────────────────────────────────────────────────────

class TestSingletonPerformance:
    def test_all_singletons_correct(self):
        truth = {"S1-001": [], "S1-002": []}
        pred = {"S1-001": [], "S1-002": []}
        result = singleton_performance(pred, truth)
        assert result["singleton_count"] == 2
        assert result["singleton_correct"] == 2
        assert result["singleton_accuracy"] == 1.0
        assert result["singleton_mean_f05"] == 1.0

    def test_all_singletons_wrong(self):
        truth = {"S1-001": [], "S1-002": []}
        pred = {"S1-001": ["S2-bad"], "S1-002": ["S3-bad"]}
        result = singleton_performance(pred, truth)
        assert result["singleton_correct"] == 0
        assert result["singleton_accuracy"] == 0.0

    def test_mixed(self):
        truth = {"S1-001": [], "S1-002": [], "S1-003": ["S2-001"]}
        pred = {"S1-001": [], "S1-002": ["S2-bad"], "S1-003": ["S2-001"]}
        result = singleton_performance(pred, truth)
        # Only S1-001 and S1-002 are singletons; S1-001 correct, S1-002 wrong
        assert result["singleton_count"] == 2
        assert result["singleton_correct"] == 1
        assert abs(result["singleton_accuracy"] - 0.5) < 1e-9

    def test_no_singletons(self):
        truth = {"S1-001": ["S2-001"]}
        pred = {"S1-001": ["S2-001"]}
        result = singleton_performance(pred, truth)
        assert result["singleton_count"] == 0
        assert result["singleton_accuracy"] is None


# ──────────────────────────────────────────────────────────────────────────────
# Candidate recall
# ──────────────────────────────────────────────────────────────────────────────

class TestCandidateRecall:
    def test_all_covered(self):
        truth = {"S1-001": ["S2-001", "S3-002"], "S1-002": ["S2-003"]}
        cands = {"S1-001": ["S2-001", "S3-002", "S2-999"], "S1-002": ["S2-003"]}
        result = candidate_recall(cands, truth)
        assert result["candidate_recall"] == 1.0
        assert result["total_true_links"] == 3
        assert result["covered_links"] == 3

    def test_partial_coverage(self):
        truth = {"S1-001": ["S2-001", "S3-002"]}
        cands = {"S1-001": ["S2-001"]}  # misses S3-002
        result = candidate_recall(cands, truth)
        assert result["candidate_recall"] == 0.5
        assert result["missing_links"] == 1

    def test_singletons_excluded(self):
        """Singleton entities (no truth links) should not contribute to denominator."""
        truth = {"S1-001": ["S2-001"], "S1-002": []}
        cands = {"S1-001": ["S2-001"], "S1-002": []}
        result = candidate_recall(cands, truth)
        assert result["total_true_links"] == 1
        assert result["candidate_recall"] == 1.0

    def test_no_candidates_no_truth(self):
        """All singletons → candidate recall = 1.0 (nothing to cover)."""
        truth = {"S1-001": [], "S1-002": []}
        cands = {}
        result = candidate_recall(cands, truth)
        assert result["candidate_recall"] == 1.0
        assert result["total_true_links"] == 0

    def test_empty_candidates_misses_all(self):
        truth = {"S1-001": ["S2-001", "S3-002"]}
        cands = {}
        result = candidate_recall(cands, truth)
        assert result["candidate_recall"] == 0.0
        assert result["missing_links"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# Evaluate wrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestEvaluate:
    def test_returns_expected_keys(self):
        pred = {"S1-001": ["S2-001"]}
        truth = {"S1-001": ["S2-001"]}
        cands = {"S1-001": ["S2-001"]}
        report = evaluate(pred, truth, candidates=cands, label="test")
        assert "macro" in report
        assert "singletons" in report
        assert "candidate_recall" in report
        assert report["label"] == "test"

    def test_without_candidates(self):
        pred = {"S1-001": []}
        truth = {"S1-001": []}
        report = evaluate(pred, truth)
        assert "candidate_recall" not in report

    def test_full_correct_prediction(self):
        truth = {
            "S1-001": ["S2-001", "S3-002"],
            "S1-002": [],
            "S1-003": ["S2-005"],
        }
        pred = truth.copy()
        report = evaluate(pred, truth, candidates=pred)
        assert abs(report["macro"]["f05"] - 1.0) < 1e-9
        assert report["candidate_recall"]["candidate_recall"] == 1.0


if __name__ == "__main__":
    # Quick smoke-test without pytest
    import traceback
    passed = failed = 0
    test_classes = [
        TestProblemStatementExample,
        TestSingletonEdgeCases,
        TestPrecisionRecall,
        TestMacroAveraging,
        TestPerEntityF05,
        TestSingletonPerformance,
        TestCandidateRecall,
        TestEvaluate,
    ]
    for cls in test_classes:
        obj = cls()
        for name in [n for n in dir(cls) if n.startswith("test_")]:
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
