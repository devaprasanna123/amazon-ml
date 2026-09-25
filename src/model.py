"""
src/model.py
============
Matching model: Rule baseline + LightGBM binary classifier.

Usage
-----
    from src.model import RuleBaseline, LGBMMatchingModel

    baseline = RuleBaseline(name_threshold=0.85, addr_threshold=0.75)
    lgbm = LGBMMatchingModel()
    lgbm.fit(X_train, y_train, X_val, y_val)
    probs = lgbm.predict_proba(X_test)
"""

from __future__ import annotations

import json
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.features import FEATURE_NAMES, N_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Rule Baseline
# ─────────────────────────────────────────────────────────────────────────────

class RuleBaseline:
    """
    Simple threshold-based matching rules.
    Uses pre-computed feature vectors (same as the ML model).

    Decision: MATCH if any of:
      1. name_exact == 1
      2. name_norm_exact == 1
      3. name_wratio >= name_threshold AND country_match == 1
      4. name_token_set >= name_token_threshold AND addr_jaccard >= addr_token_threshold
      5. name_ratio >= name_threshold AND addr_numeric_overlap >= numeric_threshold
    """

    def __init__(
        self,
        name_threshold: float = 0.85,
        name_token_threshold: float = 0.80,
        addr_token_threshold: float = 0.40,
        numeric_threshold: float = 0.50,
    ):
        self.name_threshold = name_threshold
        self.name_token_threshold = name_token_threshold
        self.addr_token_threshold = addr_token_threshold
        self.numeric_threshold = numeric_threshold

        # Feature indices for fast access
        self._idx = {name: i for i, name in enumerate(FEATURE_NAMES)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions (0/1) for each row."""
        i = self._idx
        name_exact = X[:, i["name_exact"]]
        name_norm_exact = X[:, i["name_norm_exact"]]
        name_wratio = X[:, i["name_wratio"]]
        name_token_set = X[:, i["name_token_set"]]
        name_ratio = X[:, i["name_ratio"]]
        addr_jaccard = X[:, i["addr_jaccard"]]
        addr_numeric = X[:, i["addr_numeric_overlap"]]
        country_match = X[:, i["country_match"]]

        rule1 = name_exact >= 1.0
        rule2 = name_norm_exact >= 1.0
        rule3 = (name_wratio >= self.name_threshold) & (country_match >= 1.0)
        rule4 = (name_token_set >= self.name_token_threshold) & \
                (addr_jaccard >= self.addr_token_threshold)
        rule5 = (name_ratio >= self.name_threshold) & \
                (addr_numeric >= self.numeric_threshold)

        return (rule1 | rule2 | rule3 | rule4 | rule5).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return continuous score (max of normalized rule confidences) in [0, 1]."""
        i = self._idx
        name_wratio = X[:, i["name_wratio"]]
        name_token_set = X[:, i["name_token_set"]]
        addr_jaccard = X[:, i["addr_jaccard"]]
        country_match = X[:, i["country_match"]]

        score = (
            0.5 * name_wratio
            + 0.2 * name_token_set
            + 0.2 * addr_jaccard
            + 0.1 * country_match
        )
        return score


# ─────────────────────────────────────────────────────────────────────────────
# LightGBM Model
# ─────────────────────────────────────────────────────────────────────────────

class LGBMMatchingModel:
    """
    LightGBM binary classifier for pairwise entity matching.

    Trained on pre-computed feature vectors.
    Uses early stopping on validation F0.5-calibrated loss.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        max_depth: int = -1,
        min_child_samples: int = 20,
        feature_fraction: float = 0.8,
        bagging_fraction: float = 0.8,
        bagging_freq: int = 5,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = -1,
    ):
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=bagging_freq,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            objective="binary",
            metric="binary_logloss",
        )
        self.model = None
        self.feature_importances_ = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 30,
    ) -> "LGBMMatchingModel":
        import lightgbm as lgb

        # Auto-compute scale_pos_weight if not set
        if self.params["scale_pos_weight"] is None:
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            if n_pos > 0:
                self.params["scale_pos_weight"] = n_neg / n_pos
            else:
                self.params["scale_pos_weight"] = 1.0

        callbacks = [lgb.log_evaluation(period=50)]

        fit_kwargs = dict(
            X=X_train,
            y=y_train,
            feature_name=FEATURE_NAMES,
        )

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True))

        self.model = lgb.LGBMClassifier(**self.params)
        self.model.fit(**fit_kwargs, callbacks=callbacks)
        self.feature_importances_ = dict(
            zip(FEATURE_NAMES, self.model.feature_importances_)
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return P(match) for each row."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "LGBMMatchingModel":
        with open(Path(path), "rb") as f:
            return pickle.load(f)

    def top_features(self, n: int = 20) -> List[Tuple[str, float]]:
        if self.feature_importances_ is None:
            return []
        return sorted(
            self.feature_importances_.items(),
            key=lambda x: x[1], reverse=True
        )[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Threshold optimizer
# ─────────────────────────────────────────────────────────────────────────────

def optimize_threshold(
    probs: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, List[str]],
    threshold_grid: Optional[List[float]] = None,
) -> Tuple[float, dict]:
    """
    Find the threshold maximizing macro F0.5 on the validation set.

    Parameters
    ----------
    probs       : predicted probabilities (n_pairs,)
    pair_ids    : list of (s1_id, s23_id) for each pair
    ground_truth: {s1_id: [true_matches]}
    threshold_grid: thresholds to evaluate (default: 0.1 to 0.9 step 0.05)

    Returns
    -------
    best_threshold, full_results_dict
    """
    from src.metrics import macro_precision_recall_f05

    if threshold_grid is None:
        threshold_grid = [round(t, 2) for t in np.arange(0.1, 0.95, 0.05)]

    results = []
    for thr in threshold_grid:
        predictions = _build_predictions(probs, pair_ids, thr)
        metrics = macro_precision_recall_f05(predictions, ground_truth)
        metrics["threshold"] = thr
        results.append(metrics)

    best = max(results, key=lambda x: x["f05"])
    return best["threshold"], {"grid": results, "best": best}


def _build_predictions(
    probs: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    threshold: float,
) -> Dict[str, List[str]]:
    """Convert (probs, pair_ids, threshold) → predictions dict."""
    from collections import defaultdict
    preds: Dict[str, List[str]] = defaultdict(list)
    for prob, (s1_id, s23_id) in zip(probs, pair_ids):
        if prob >= threshold:
            preds[s1_id].append(s23_id)
    return dict(preds)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_val: np.ndarray,
    pair_ids_val: List[Tuple[str, str]],
    ground_truth_val: Dict[str, List[str]],
    threshold: float = 0.5,
    label: str = "val",
) -> dict:
    """Full evaluation of a model on validation pairs."""
    from src.metrics import macro_precision_recall_f05, singleton_performance

    probs = model.predict_proba(X_val)
    predictions = _build_predictions(probs, pair_ids_val, threshold)

    # Include all S1s from ground truth (not just those with candidates)
    for s1_id in ground_truth_val:
        if s1_id not in predictions:
            predictions[s1_id] = []

    macro = macro_precision_recall_f05(predictions, ground_truth_val)
    sg = singleton_performance(predictions, ground_truth_val)

    return {
        "label": label,
        "threshold": threshold,
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f05": macro["f05"],
        "n_entities": macro["n_entities"],
        "singleton_accuracy": sg.get("singleton_accuracy"),
        "singleton_count": sg.get("singleton_count"),
        "n_val_pairs": len(pair_ids_val),
        "n_positive_pred": int((probs >= threshold).sum()),
    }
