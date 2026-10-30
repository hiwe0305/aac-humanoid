"""Run with PYTHONPATH=src python -m aac_duac.cli --help."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {"libero": "examples.Libero.custom_data_config:LiberoDataConfig", "robocasa": "single_panda_gripper"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download")
    dl.add_argument("target", choices=["model", "libero", "robocasa", "all"])
    verify = sub.add_parser("verify-data")
    verify.add_argument("dataset", choices=CONFIGS)
    verify.add_argument("--deep", action="store_true", help="read each parquet footer and decode a frame from each video")
    train = sub.add_parser("train")
    train.add_argument("dataset", choices=CONFIGS)
    train.add_argument("--steps", type=int, help="exact number of steps; also saves at the final step")
    ev = sub.add_parser("eval")
    ev.add_argument("dataset", choices=CONFIGS)
    ev.add_argument("--checkpoint")
    ev.add_argument("--suite", default="libero_spatial", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"])
    ev.add_argument("--horizon", type=int)
    ev.add_argument("--trials", type=int)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    def path(key):
        p = Path(cfg["paths"][key]).expanduser()
        return p if p.is_absolute() else ROOT / p
    data, output, gr00t = path("data"), path("output"), path("gr00t")
    env = os.environ.copy()
    env["HF_HOME"] = str(path("hf_cache"))
    env["PYTHONPATH"] = str(gr00t) + os.pathsep + env.get("PYTHONPATH", "")
    env["MUJOCO_GL"] = "egl"
    env["LIBERO_CONFIG_PATH"] = str(ROOT / ".libero")
    def run(command, **kwargs):
        print("+", " ".join(map(str, command)), flush=True)
        subprocess.run(list(map(str, command)), check=True, env=env, **kwargs)
    if args.command == "download":
        if args.target in ("libero", "all"):
            run([sys.executable, ROOT / "tools/download_libero.py", "--out", data / "libero", "--gr00t", gr00t])
        if args.target in ("robocasa", "all"):
            run([sys.executable, ROOT / "tools/download_robocasa.py", "--out", data / "robocasa", "--demos", cfg["data"]["robocasa_demos_per_task"]])
        if args.target in ("model", "all"):
            from huggingface_hub import snapshot_download
            snapshot_download(cfg["model"]["base"], cache_dir=str(path("hf_cache") / "hub"))
        return
    if args.command == "verify-data":
        from .verify import verify_dataset
        datasets = ([data / "libero" / ("libero_" + s) for s in ("spatial", "object", "goal", "10")] if args.dataset == "libero" else sorted((data / "robocasa").glob("single_panda_gripper.*")))
        if args.dataset == "robocasa":
            from .tasks import ROBOCASA_TASKS
            expected = {"single_panda_gripper." + name for name in ROBOCASA_TASKS}
            if {p.name for p in datasets} != expected:
                parser.error("RoboCasa task directories do not match the 24 expected task names")
        for dataset in datasets:
            try:
                episodes, frames, cameras = verify_dataset(dataset, deep=args.deep, expected_cameras=2 if args.dataset == "libero" else 3)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
                parser.error(f"{dataset}: {exc}")
            print(f"{dataset.name}: {episodes} episodes, {frames} frames, {cameras} cameras: files present")
        print("File checks passed; this does not validate episode provenance or simulator compatibility.")
        return
    if args.command == "train":
        t = cfg["training"]
        datasets = ([data / "libero" / ("libero_" + s) for s in ("spatial", "object", "goal", "10")] if args.dataset == "libero" else sorted((data / "robocasa").glob("single_panda_gripper.*")))
        if args.dataset == "robocasa":
            from .tasks import ROBOCASA_TASKS
            expected = {"single_panda_gripper." + name for name in ROBOCASA_TASKS}
            if {p.name for p in datasets} != expected:
                parser.error("RoboCasa task directories do not match the 24 expected task names")
        from .verify import verify_dataset
        for dataset in datasets:
            try:
                verify_dataset(dataset, expected_cameras=2 if args.dataset == "libero" else 3)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                parser.error(f"Dataset missing or incomplete: {dataset}: {exc}")
        if min(t["global_batch"], t["micro_batch"], t["num_gpus"], t["save_steps"]) <= 0 or t["global_batch"] % (t["micro_batch"] * t["num_gpus"]):
            parser.error("batch sizes and save_steps must be positive; global_batch must be divisible by micro_batch * num_gpus")
        if args.steps is not None and args.steps <= 0:
            parser.error("--steps must be positive")
        frames = sum(json.loads((d / "meta/info.json").read_text())["total_frames"] for d in datasets)
        steps = args.steps or math.ceil(t["epochs"] * frames / t["global_batch"])
        save = args.steps or t["save_steps"]
        if args.steps is None:
            steps = math.ceil(steps / save) * save
        cmd = [sys.executable, gr00t / "scripts/gr00t_finetune.py", "--dataset-path", *datasets, "--base-model-path", cfg["model"]["base"], "--data-config", CONFIGS[args.dataset], "--embodiment-tag", "new_embodiment", "--output-dir", output / args.dataset, "--num-gpus", t["num_gpus"], "--batch-size", t["micro_batch"], "--gradient-accumulation-steps", t["global_batch"] // (t["micro_batch"] * t["num_gpus"]), "--max-steps", steps, "--save-steps", save, "--learning-rate", t["learning_rate"], "--weight-decay", t["weight_decay"], "--warmup-ratio", t["warmup_ratio"], "--dataloader-num-workers", t["num_workers"], "--video-backend", t["video_backend"], "--report-to", "tensorboard", "--no-tune-llm", "--no-tune-visual", "--tune-projector", "--tune-diffusion-model"]
        run(cmd, cwd=gr00t)
        return
    e = cfg["evaluation"]
    checkpoints = [p for p in (output / args.dataset).glob("checkpoint-*") if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()]
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else max(checkpoints, key=lambda p: int(p.name.split("-")[-1]), default=None)
    if checkpoint is None or not checkpoint.is_dir():
        parser.error("checkpoint not found; supply --checkpoint")
    horizon = args.horizon if args.horizon is not None else e["horizon"]
    trials = args.trials if args.trials is not None else e[args.dataset + "_trials"]
    if horizon <= 0 or trials <= 0:
        parser.error("horizon and trials must be positive")
    sim_python = ROOT / (".venv-" + args.dataset) / "bin/python"
    if not sim_python.exists():
        parser.error(f"missing simulator environment: {sim_python}")
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    server_cmd = [sys.executable, str(gr00t / "scripts/inference_service.py"), "--server", "--model_path", str(checkpoint), "--data_config", CONFIGS[args.dataset], "--embodiment-tag", "new_embodiment", "--denoising-steps", str(cfg["model"]["denoising_steps"]), "--port", str(e["port"])]
    with (logs / f"server_{e['port']}.log").open("w") as log:
        server = subprocess.Popen(server_cmd, cwd=gr00t, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            run([sys.executable, ROOT / "eval/zmq_client.py", "--port", e["port"], "--wait", 900])
            result = output / "eval" / (f"libero_{args.suite}_h{horizon}.json" if args.dataset == "libero" else f"robocasa_kitchen_h{horizon}.json")
            cmd = [sim_python, ROOT / ("eval/eval_libero.py" if args.dataset == "libero" else "eval/eval_robocasa_kitchen.py"), "--port", e["port"], "--horizon", horizon, "--trials", trials, "--out", result]
            if args.dataset == "libero":
                cmd += ["--suite", args.suite, "--gr00t", gr00t]
            run(cmd)
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
