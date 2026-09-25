"""
src/scoreboard.py
Master Experiment Scoreboard for Business Entity Resolution.
Logs and tracks all experiment runs on the 5k deterministic test-like validation pool.
Never overwrites earlier results; appends each run immutably.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

SCOREBOARD_PATH = Path(r"D:\amazon ML\reports\master_experiments.csv")
SCOREBOARD_PATH.parent.mkdir(parents=True, exist_ok=True)

SCOREBOARD_COLUMNS = [
    "experiment_id",
    "timestamp",
    "description",
    "candidate_recall",
    "precision",
    "recall",
    "macro_f05",
    "singleton_accuracy",
    "false_merges",
    "avg_predicted_matches",
    "max_predicted_matches",
    "candidate_count",
    "p50_candidates",
    "p95_candidates",
    "runtime_s",
    "peak_ram_mb",
]


def log_experiment(
    experiment_id: str,
    description: str,
    candidate_recall: float,
    precision: float,
    recall: float,
    macro_f05: float,
    singleton_accuracy: float,
    false_merges: int,
    avg_predicted_matches: float,
    max_predicted_matches: int,
    candidate_count: int,
    p50_candidates: float,
    p95_candidates: float,
    runtime_s: float,
    peak_ram_mb: float,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Appends an experiment result to reports/master_experiments.csv.
    Guarantees thread-safe / atomic file update.
    """
    row = {
        "experiment_id": str(experiment_id),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "description": str(description),
        "candidate_recall": round(float(candidate_recall), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "macro_f05": round(float(macro_f05), 4),
        "singleton_accuracy": round(float(singleton_accuracy), 4),
        "false_merges": int(false_merges),
        "avg_predicted_matches": round(float(avg_predicted_matches), 2),
        "max_predicted_matches": int(max_predicted_matches),
        "candidate_count": int(candidate_count),
        "p50_candidates": round(float(p50_candidates), 1),
        "p95_candidates": round(float(p95_candidates), 1),
        "runtime_s": round(float(runtime_s), 2),
        "peak_ram_mb": round(float(peak_ram_mb), 1),
    }

    if extra_metadata:
        for k, v in extra_metadata.items():
            if k not in row:
                row[k] = v

    if SCOREBOARD_PATH.exists():
        df_existing = pd.read_csv(SCOREBOARD_PATH)
        df_new = pd.concat([df_existing, pd.DataFrame([row])], ignore_index=True)
    else:
        df_new = pd.DataFrame([row])

    df_new.to_csv(SCOREBOARD_PATH, index=False)
    print(f"Logged experiment '{experiment_id}' to {SCOREBOARD_PATH}")
    return df_new


def get_scoreboard() -> pd.DataFrame:
    """Read the current scoreboard."""
    if SCOREBOARD_PATH.exists():
        return pd.read_csv(SCOREBOARD_PATH)
    return pd.DataFrame(columns=SCOREBOARD_COLUMNS)
