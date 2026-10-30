#!/usr/bin/env python3
"""Download the 4 LIBERO suites (LeRobot v2 format, OpenVLA 'no_noops' preprocessing) and drop in GR00T's modality.json.

Usage: python tools/download_libero.py --out data/libero --gr00t Isaac-GR00T [--suites spatial object goal 10]
"""
import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

REPOS = {
    "spatial": "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
    "object": "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
    "goal": "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
    "10": "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",  # = LIBERO-Long
}

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--gr00t", required=True)
ap.add_argument("--suites", nargs="+", default=list(REPOS))
args = ap.parse_args()

modality = Path(args.gr00t) / "examples/Libero/modality.json"
assert modality.exists(), modality
for s in args.suites:
    dst = Path(args.out) / f"libero_{s}"
    print(f"== {s}: {REPOS[s]} -> {dst}")
    snapshot_download(repo_id=REPOS[s], repo_type="dataset", local_dir=str(dst))
    shutil.copy(modality, dst / "meta" / "modality.json")
    print("   modality.json installed")

