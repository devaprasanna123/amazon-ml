"""
src/config.py
=============
Central configuration for the Amazon Business Entity Resolution pipeline.
All paths, column names, constants, and hyperparameters live here.
"""

from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
DATASET_ROOT = Path(
    r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\dataset"
)
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "test"

PROJECT_ROOT = Path(r"D:\amazon ML")
SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Create dirs at import time so downstream code doesn't have to
for _d in (REPORTS_DIR, MODELS_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ── Training data files ──────────────────────────────────────────────────────
TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

# ── Test data files ──────────────────────────────────────────────────────────
TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

# ── Column names ─────────────────────────────────────────────────────────────
COL_ENTITY_ID = "entity_id"
COL_BUSINESS_NAME = "business_name"
COL_BUSINESS_ADDRESS = "business_address"
COL_COUNTRY = "country"

COL_GT_S1 = "source1_entity_id"
COL_GT_MATCHED = "matched_entity_ids"

SOURCE_COLS = [COL_ENTITY_ID, COL_BUSINESS_NAME, COL_BUSINESS_ADDRESS, COL_COUNTRY]
GT_COLS = [COL_GT_S1, COL_GT_MATCHED]

# ── ID prefixes ───────────────────────────────────────────────────────────────
S1_PREFIX = "S1-"
S2_PREFIX = "S2-"
S3_PREFIX = "S3-"

# ── Metric settings ───────────────────────────────────────────────────────────
BETA = 0.5           # F_beta
BETA_SQ = BETA ** 2  # pre-computed β² for F_β formula

# ── Validation split settings ────────────────────────────────────────────────
VAL_FRACTION = 0.15   # 15% of S1 entities go to validation
RANDOM_SEED = 42

# ── I/O settings ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 50_000   # rows per pandas read_csv chunk

# ── Output files ──────────────────────────────────────────────────────────────
MATCHING_RESULTS_TSV = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_TSV = OUTPUT_DIR / "candidate_pairs.tsv"

# ── Validation metadata outputs ───────────────────────────────────────────────
VAL_SPLIT_IDS_JSON = REPORTS_DIR / "validation_split_ids.json"
VAL_METADATA_JSON = REPORTS_DIR / "validation_metadata.json"
VAL_GT_PARQUET = REPORTS_DIR / "val_ground_truth.parquet"
TRAIN_GT_PARQUET = REPORTS_DIR / "train_ground_truth_split.parquet"
