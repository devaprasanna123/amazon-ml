"""
colab/setup_colab.py
====================
Environment initialization, hardware detection, dynamic resource sizing,
and directory structure provisioning for Google Colab + Google Drive.

Usage in Colab:
    python colab/setup_colab.py [--drive-mount /content/drive/MyDrive]
"""

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path


def detect_hardware():
    """Detects CPU, RAM, GPU, and disk environment without hard-coding specs."""
    info = {
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count() or 1,
    }

    # Memory detection
    try:
        import psutil
        vm = psutil.virtual_memory()
        info["total_ram_gb"] = round(vm.total / (1024 ** 3), 2)
        info["available_ram_gb"] = round(vm.available / (1024 ** 3), 2)
    except ImportError:
        info["total_ram_gb"] = "unknown (psutil missing)"
        info["available_ram_gb"] = "unknown (psutil missing)"

    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu_available"] = True
            info["gpu_count"] = torch.cuda.device_count()
            devices = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                devices.append({
                    "id": i,
                    "name": props.name,
                    "total_memory_gb": round(props.total_memory / (1024 ** 3), 2),
                    "major": props.major,
                    "minor": props.minor,
                })
            info["gpu_devices"] = devices
        else:
            info["gpu_available"] = False
            info["gpu_devices"] = []
    except ImportError:
        info["gpu_available"] = False
        info["gpu_note"] = "PyTorch not installed"

    # Disk detection
    try:
        content_usage = shutil.disk_usage("/content" if os.path.exists("/content") else ".")
        info["disk_total_gb"] = round(content_usage.total / (1024 ** 3), 2)
        info["disk_free_gb"] = round(content_usage.free / (1024 ** 3), 2)
    except Exception as e:
        info["disk_error"] = str(e)

    return info


def compute_safe_resource_limits(hw_info):
    """Calculates safe memory limits for DuckDB, Polars, and batch processing."""
    avail_ram = hw_info.get("available_ram_gb")
    if isinstance(avail_ram, (int, float)):
        # Leave at least 2.5 GB for OS, Python runtime, and other processes
        safe_duckdb_gb = max(1.5, round(avail_ram * 0.70, 1))
        batch_size = 100_000 if avail_ram >= 12 else 50_000
    else:
        safe_duckdb_gb = 4.0
        batch_size = 50_000

    cpu_cores = hw_info.get("cpu_count", 4)
    safe_threads = max(2, min(cpu_cores, 8))

    return {
        "duckdb_memory_limit": f"{safe_duckdb_gb}GB",
        "duckdb_threads": safe_threads,
        "streaming_batch_size": batch_size,
    }


def provision_directories(drive_root: Path, local_root: Path):
    """
    Creates persistent Google Drive and fast local ephemeral directories.
    """
    drive_dirs = [
        drive_root / "dataset",
        drive_root / "project",
        drive_root / "artifacts",
        drive_root / "indexes",
        drive_root / "checkpoints",
        drive_root / "reports",
        drive_root / "experiments",
        drive_root / "outputs",
    ]
    for d in drive_dirs:
        d.mkdir(parents=True, exist_ok=True)

    local_dirs = [
        local_root / "data" / "raw",
        local_root / "data" / "normalized",
        local_root / "data" / "cache",
        local_root / "indexes",
        local_root / "checkpoints",
        local_root / "models",
        local_root / "reports",
        local_root / "output",
    ]
    for d in local_dirs:
        d.mkdir(parents=True, exist_ok=True)

    return drive_dirs, local_dirs


def main():
    parser = argparse.ArgumentParser(description="Setup Colab environment for Amazon ML Challenge 2026")
    parser.add_argument("--drive-mount", type=str, default="/content/drive/MyDrive",
                        help="Path to Google Drive MyDrive root")
    parser.add_argument("--local-root", type=str, default="/content/amazon_ml_challenge",
                        help="Local ephemeral compute root on Colab")
    args = parser.parse_args()

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 — COLAB ENVIRONMENT SETUP")
    print("=" * 80)

    hw = detect_hardware()
    limits = compute_safe_resource_limits(hw)

    print("\n[1] HARDWARE & PLATFORM PROFILE:")
    print(f"  OS:              {hw['os']}")
    print(f"  Python:          {hw['python_version']}")
    print(f"  CPU Cores:       {hw['cpu_count']}")
    print(f"  Total RAM:       {hw.get('total_ram_gb', 'N/A')} GB")
    print(f"  Available RAM:   {hw.get('available_ram_gb', 'N/A')} GB")
    print(f"  Disk Free:       {hw.get('disk_free_gb', 'N/A')} GB / {hw.get('disk_total_gb', 'N/A')} GB")

    if hw.get("gpu_available"):
        print(f"  GPU Detected:    {len(hw['gpu_devices'])} device(s)")
        for g in hw["gpu_devices"]:
            print(f"    - GPU {g['id']}: {g['name']} ({g['total_memory_gb']} GB VRAM)")
    else:
        print("  GPU Detected:    None / CPU-only execution")

    print("\n[2] DYNAMIC RUNTIME RESOURCE CONFIGURATION:")
    print(f"  DuckDB Max Memory:   {limits['duckdb_memory_limit']}")
    print(f"  DuckDB Threads:      {limits['duckdb_threads']}")
    print(f"  Batch Streaming:     {limits['streaming_batch_size']:,} rows/chunk")

    drive_base = Path(args.drive_mount) / "amazon_ml_challenge_2026"
    local_base = Path(args.local_root)

    print(f"\n[3] DIRECTORY PROVISIONING:")
    print(f"  Drive Persistent:    {drive_base}")
    print(f"  Local Fast Compute:  {local_base}")

    try:
        drive_dirs, local_dirs = provision_directories(drive_base, local_base)
        print("  ✓ Persistent Google Drive directories created.")
        print("  ✓ Local ephemeral compute directories created.")
    except Exception as e:
        print(f"  [Warning] Directory creation note: {e}")

    # Save environment config for downstream scripts
    config_file = local_base / "env_config.json"
    env_config = {
        "hardware": hw,
        "limits": limits,
        "drive_base": str(drive_base),
        "local_base": str(local_base),
    }
    local_base.mkdir(parents=True, exist_ok=True)
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(env_config, f, indent=2)
    print(f"\n[4] CONFIG PERSISTED: {config_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
