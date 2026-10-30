#!/usr/bin/env python3
"""Download the RoboCasa-Kitchen (24 tasks, Panda-Omron) training data used by GR00T N1/N1.5 and keep only the first
N demos per task (paper: 100 MimicGen demos / task).

Source: nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim, folders `single_panda_gripper.<Task>` (3000 demos each,
LeRobot v2.0, 20 fps, 3 cameras, modality.json already included). Only episodes [0, N) are fetched, and meta/
(episodes.jsonl, info.json) is rewritten to match, so GR00T's loader sees an N-episode dataset.
(stats.json is left as shipped: normalisation stats from the full 3000 demos.)

Usage: python tools/download_robocasa.py --out data/robocasa --demos 100 [--tasks OpenDrawer CloseDrawer ...]
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
CHUNK = 1000  # episodes per chunk dir

# Paper Sec 5.1.1: 10 Relocation (PnP) + 6 Container (doors/drawers) + 5 Rotation + 3 Button = 24 tasks
TASKS = {
    "relocation": [
        "PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter", "PnPCounterToStove",
        "PnPStoveToCounter", "PnPCounterToMicrowave", "PnPMicrowaveToCounter", "CoffeeSetupMug", "CoffeeServeMug",
    ],
    "container": ["OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor", "OpenDrawer", "CloseDrawer"],
    "rotation": ["TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout", "TurnOnStove", "TurnOffStove"],
    "button": ["CoffeePressButton", "TurnOnMicrowave", "TurnOffMicrowave"],
}
ALL = [t for g in TASKS.values() for t in g]
assert len(ALL) == 24

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--demos", type=int, default=100)
ap.add_argument("--tasks", nargs="+", default=ALL)
args = ap.parse_args()
if not 1 <= args.demos <= CHUNK:
    ap.error("--demos must be between 1 and 1000")

from aac_duac.verify import verify_dataset

for task in args.tasks:
    name = f"single_panda_gripper.{task}"
    out = Path(args.out)
    done = out / name / ".subset_done"
    if done.exists():
        info = json.loads((out / name / "meta/info.json").read_text())
        if info["total_episodes"] != args.demos:
            raise ValueError(f"{name} already contains {info['total_episodes']} episodes, requested {args.demos}; use a new data directory")
        verify_dataset(out / name, expected_cameras=3)
        print(f"== {task}: already downloaded and checked")
        continue
    print(f"== {task}")
    eps = [f"episode_{i:06d}" for i in range(args.demos)]
    patterns = [f"{name}/meta/*"]
    patterns += [f"{name}/data/chunk-000/{e}.parquet" for e in eps]
    patterns += [f"{name}/videos/chunk-000/*/{e}.mp4" for e in eps]
    snapshot_download(repo_id=REPO, repo_type="dataset", local_dir=str(out), allow_patterns=patterns)

    meta = out / name / "meta"
    lines = (meta / "episodes.jsonl").read_text().splitlines()[: args.demos]
    if len(lines) != args.demos:
        raise ValueError(f"{task}: expected {args.demos} episodes, got {len(lines)}")
    (meta / "episodes.jsonl").write_text("\n".join(lines) + "\n")
    info = json.loads((meta / "info.json").read_text())
    info["total_episodes"] = len(lines)
    info["total_frames"] = sum(json.loads(l)["length"] for l in lines)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(lines)}"}
    (meta / "info.json").write_text(json.dumps(info, indent=4))
    verify_dataset(out / name, expected_cameras=3)
    done.write_text(json.dumps({"episodes": args.demos, "source": REPO}))
    print(f"   {len(lines)} episodes, {info['total_frames']} frames")

