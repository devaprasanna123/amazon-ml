# Amazon ML Challenge 2026 — Business Entity Resolution

High-performance, memory-bounded solution for Business Entity Resolution in the Amazon ML Challenge 2026.

## 1. Challenge Architecture & Objective
- **Problem**: Link reference entities from `Source 1` to all matching entities in `Source 2` and `Source 3` (0, 1, or $N$ matches).
- **Metric**: Macro $F_{0.5}$ (precision-weighted harmonic mean, where false merges heavily penalize the score).
- **Target Scale**: ~10.3M training target entities and ~10.0M test target entities.

## 2. Repository Structure
```
amazon-ml/
├── colab/                          # Google Colab execution & synchronization runtime
│   ├── setup_colab.py              # Platform detection & dynamic resource configuration
│   ├── sync_project.py             # Staging & artifact persistence between Drive & local NVMe
│   ├── run_validation.py           # 5,000-S1 full-pool validation benchmark
│   ├── run_test_inference.py       # Batched test inference runner
│   └── Amazon_ML_Challenge_2026_Master.ipynb # Master 18-step execution notebook
├── src/                            # Core machine learning & decision logic
│   ├── config.py                   # Central paths, constants, and metric settings
│   ├── preprocessing.py            # Text normalization, legal suffix extraction, accent stripping
│   ├── blocking.py                 # Candidate generation & indexing rules
│   ├── features.py                 # Fast pairwise string & address feature extraction
│   ├── model.py                    # LightGBM binary classifier & rule baseline
│   ├── decision_optimizer.py       # Calibrated singleton gating, score gaps, and match capping
│   ├── metrics.py                  # Exact challenge macro F0.5 evaluation
│   └── validation.py               # Leakage-safe split loading
├── scripts/                        # Automation & audit scripts
│   ├── run_pipeline.py             # Pipeline runner
│   ├── final_qa_audit.py           # Automated schema & data integrity auditor
│   └── inspect_pairs.py            # Diagnostic inspection utilities
├── experiments/                    # Historical experimental pipelines & ablation baselines
│   ├── v2_recovery/                # Safe streaming V2-V6 candidate blocking experiments
│   └── v3_aws_recovery/            # AWS/Linux high-recall experiments
├── models/                         # Trained model artifacts
│   └── lgbm_model.txt              # Production LightGBM model weights
├── reports/                        # Master experiment logs, metadata, and tracking
│   ├── master_experiments.csv      # Experiment scoreboard (V1 - V6 baselines)
│   ├── dataset_inventory.txt       # Dataset profile & column distributions
│   └── threshold_search.csv        # Decision threshold search logs
├── tests/                          # Unit test suite
├── utils/                          # Official challenge validation tools
│   └── validate_submission.py      # Official challenge submission validator
├── Documentation_template.md       # Solution template
├── requirements.txt                # Pinned library dependencies
└── README.md
```

## 3. Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Full-Pool Validation
```bash
python colab/run_validation.py --version V1
python colab/run_validation.py --version V6
```

### Colab Execution
Open `colab/Amazon_ML_Challenge_2026_Master.ipynb` in Google Colab, connect a GPU/CPU runtime, and follow the step-by-step cells.

### Compliance
This project strictly utilizes only the official competition dataset and derived mathematical/textual features. No external APIs, commercial entity resolution services, geocoding lookups, or web scraping are employed.
