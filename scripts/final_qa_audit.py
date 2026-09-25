"""
scripts/final_qa_audit.py
=========================
Performs strict final submission audit and runs the official validator.
"""

import os
import sys
import hashlib
import subprocess
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (
    DATASET_ROOT, TEST_DIR, PROJECT_ROOT,
    TEST_SOURCE1, TEST_SOURCE2, TEST_SOURCE3,
    OUTPUT_DIR
)

MATCHING_TSV = OUTPUT_DIR / "matching_results.tsv"
CANDIDATES_TSV = OUTPUT_DIR / "candidate_pairs.tsv"

def sha256_file(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    if not MATCHING_TSV.exists() or not CANDIDATES_TSV.exists():
        print(f"ERROR: Output files not found in {OUTPUT_DIR}")
        return

    # 1. Count rows in test_source1.tsv
    with open(TEST_SOURCE1, "r", encoding="utf-8") as f:
        test_s1_header = next(f)
        test_s1_ids = [line.split("\t", 1)[0].strip() for line in f if line.strip()]
    test_s1_rows = len(test_s1_ids)
    test_s1_set = set(test_s1_ids)

    # 2. Count rows & parse matching_results.tsv
    with open(MATCHING_TSV, "r", encoding="utf-8") as f:
        m_header = next(f).strip()
        m_lines = [line.rstrip("\r\n") for line in f if line.strip()]
    matching_rows = len(m_lines)

    # 3. Count rows & parse candidate_pairs.tsv
    with open(CANDIDATES_TSV, "r", encoding="utf-8") as f:
        c_header = next(f).strip()
        c_lines = [line.rstrip("\r\n") for line in f if line.strip()]
    candidate_rows = len(c_lines)

    # Parse candidates: {s1_id: set_of_candidate_ids}
    candidates_dict = {}
    total_cands_count = 0
    for line in c_lines:
        parts = line.split("\t", 1)
        s1_id = parts[0].strip()
        cand_str = parts[1].strip() if len(parts) > 1 else ""
        cands = set(m.strip() for m in cand_str.split(",") if m.strip())
        candidates_dict[s1_id] = cands
        total_cands_count += len(cands)

    # Parse matches
    matched_s1_ids = []
    predicted_singleton = 0
    predicted_matched_s1 = 0
    total_predicted_links = 0
    max_links_for_one_s1 = 0

    valid_id_prefix_errors = 0
    candidate_subset_errors = 0
    duplicate_match_errors = 0

    for line in m_lines:
        parts = line.split("\t", 1)
        s1_id = parts[0].strip()
        matched_str = parts[1].strip() if len(parts) > 1 else ""
        matched_s1_ids.append(s1_id)

        if not matched_str:
            predicted_singleton += 1
        else:
            predicted_matched_s1 += 1
            m_list = [m.strip() for m in matched_str.split(",") if m.strip()]
            n_links = len(m_list)
            total_predicted_links += n_links
            if n_links > max_links_for_one_s1:
                max_links_for_one_s1 = n_links

            # Check duplicates within row
            if len(m_list) != len(set(m_list)):
                duplicate_match_errors += 1

            # Check prefixes
            for mid in m_list:
                if not (mid.startswith("S2-") or mid.startswith("S3-")):
                    valid_id_prefix_errors += 1

            # Check candidate subset
            cand_set = candidates_dict.get(s1_id, set())
            for mid in m_list:
                if mid not in cand_set:
                    candidate_subset_errors += 1

    # Verify matching rows == test_source1 rows
    row_count_match = (matching_rows == test_s1_rows)
    all_s1_unique = (len(matched_s1_ids) == len(set(matched_s1_ids)) == test_s1_rows)

    avg_links_per_s1 = total_predicted_links / test_s1_rows if test_s1_rows > 0 else 0.0
    avg_cands_per_s1 = total_cands_count / test_s1_rows if test_s1_rows > 0 else 0.0

    # Also sync copy to student_resource/output if needed
    student_res_output = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource\output")
    student_res_output.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(MATCHING_TSV, student_res_output / "matching_results.tsv")
    shutil.copy2(CANDIDATES_TSV, student_res_output / "candidate_pairs.tsv")

    # 8. Run official validator EXACTLY from student_resource directory
    student_res_dir = Path(r"D:\AMAZON DATASET\6ab10eb3b23ba_student_resource\student_resource")
    val_proc = subprocess.run([
        sys.executable,
        "utils/validate_submission.py",
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", "dataset/test"
    ], cwd=str(student_res_dir), capture_output=True, text=True, errors="replace")

    val_stdout = val_proc.stdout
    val_stderr = val_proc.stderr
    val_exit = val_proc.returncode
    val_status = "PASS" if val_exit == 0 else "FAIL"

    sha256_m = sha256_file(MATCHING_TSV)
    sha256_c = sha256_file(CANDIDATES_TSV)

    # Print ONLY requested format
    print("FINAL QA")
    print("---------")
    print(f"matching rows: {matching_rows}")
    print(f"candidate rows: {candidate_rows}")
    print(f"test S1 rows: {test_s1_rows}")
    print(f"predicted singleton: {predicted_singleton}")
    print(f"predicted matched S1: {predicted_matched_s1}")
    print(f"total predicted links: {total_predicted_links}")
    print(f"average links/S1: {avg_links_per_s1:.4f}")
    print(f"maximum links/S1: {max_links_for_one_s1}")
    print(f"validator: {val_status}")
    print(f"SHA256 matching: {sha256_m}")
    print(f"SHA256 candidates: {sha256_c}")

    if val_status == "PASS":
        print("\nREADY FOR LEADERBOARD")

if __name__ == "__main__":
    main()
