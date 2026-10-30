#!/usr/bin/env python3
"""LIBERO evaluation of a GR00T N1.5 checkpoint served over ZMQ, with a configurable *fixed* execution horizon.

Adapted from Isaac-GR00T/examples/Libero/eval/run_libero_eval.py. Differences:
  * executes the first `--horizon` actions of every predicted chunk (official script only uses action 0,
    i.e. h=1) -> reproduces paper's "GR00T (h=...)" rows
  * headless, no gr00t/torch import (talks to the server through eval/zmq_client.py)
  * 50 trials / task by default, JSON result file

Run in .venv-libero:
  python eval/eval_libero.py --suite libero_spatial --port 5555 --horizon 16 --out outputs/eval/x.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunking import select_horizon  # noqa: E402
from zmq_client import PolicyClient  # noqa: E402

os.environ.setdefault("MUJOCO_GL", "egl")

MAX_STEPS = {  # same as official script (longest training demo + margin)
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 600,
    "libero_10": 1000,
    "libero_90": 400,
}
NUM_STEPS_WAIT = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(MAX_STEPS))
    ap.add_argument("--gr00t", default=str(Path(__file__).resolve().parent.parent / "Isaac-GR00T"))
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--tasks", type=int, nargs="*", help="subset of task ids (debug)")
    ap.add_argument("--save-video-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sys.path.insert(0, args.gr00t)  # for examples.Libero.eval.utils (only needs libero + numpy + imageio)
    from libero.libero import benchmark
    from examples.Libero.eval.utils import (
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        normalize_gripper_action,
        quat2axisangle,
    )

    client = PolicyClient(args.host, args.port)
    client.ping()
    suite = benchmark.get_benchmark_dict()[args.suite]()
    max_steps = MAX_STEPS[args.suite]
    keys = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]

    def make_obs(obs, lang):
        img, wrist = get_libero_image(obs)
        rpy = quat2axisangle(obs["robot0_eef_quat"])
        xyz = obs["robot0_eef_pos"]
        return {
            "video.image": np.expand_dims(img, 0),
            "video.wrist_image": np.expand_dims(wrist, 0),
            "state.x": np.array([[xyz[0]]]),
            "state.y": np.array([[xyz[1]]]),
            "state.z": np.array([[xyz[2]]]),
            "state.roll": np.array([[rpy[0]]]),
            "state.pitch": np.array([[rpy[1]]]),
            "state.yaw": np.array([[rpy[2]]]),
            "state.gripper": np.expand_dims(obs["robot0_gripper_qpos"], 0),
            "annotation.human.action.task_description": [lang],
        }

    def to_action(chunk, i):
        a = np.array([np.atleast_1d(chunk[f"action.{k}"][i])[0] for k in keys], dtype=np.float32)
        return normalize_gripper_action(a, binarize=True)

    results = {"suite": args.suite, "horizon": args.horizon, "trials_per_task": args.trials, "tasks": []}
    t_start = time.time()
    task_ids = args.tasks if args.tasks else range(suite.n_tasks)
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, lang = get_libero_env(task, resolution=256)
        succ, n_queries = 0, 0
        for ep in range(args.trials):
            env.reset()
            obs = env.set_init_state(init_states[ep % len(init_states)])
            t, done, frames = 0, False, []
            while t < max_steps + NUM_STEPS_WAIT and not done:
                if t < NUM_STEPS_WAIT:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    t += 1
                    continue
                chunk = client.get_action(make_obs(obs, task.language))
                n_queries += 1
                h = select_horizon(chunk, args.horizon)
                for i in range(h):
                    obs, _, done, _ = env.step(to_action(chunk, i).tolist())
                    t += 1
                    if args.save_video_dir:
                        frames.append(get_libero_image(obs)[0])
                    if done or t >= max_steps + NUM_STEPS_WAIT:
                        break
            succ += int(done)
            if args.save_video_dir and frames:
                import imageio

                Path(args.save_video_dir).mkdir(parents=True, exist_ok=True)
                imageio.mimsave(f"{args.save_video_dir}/{args.suite}_t{task_id}_e{ep}_{'ok' if done else 'fail'}.mp4", frames, fps=20)
            print(f"[{args.suite}] task {task_id} ep {ep + 1}/{args.trials}  success={done}  running={succ}/{ep + 1}", flush=True)
        env.close()
        results["tasks"].append({"task_id": task_id, "language": lang, "successes": succ, "trials": args.trials,
                                 "success_rate": succ / args.trials, "avg_queries_per_ep": n_queries / args.trials})
        print(f"==> task {task_id} '{lang}': {succ}/{args.trials}", flush=True)

    tot = sum(t["successes"] for t in results["tasks"])
    n = sum(t["trials"] for t in results["tasks"])
    results["success_rate"] = tot / n
    results["minutes"] = (time.time() - t_start) / 60
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\n{args.suite} (h={args.horizon}): {tot}/{n} = {100 * tot / n:.1f}%  -> {args.out}")


if __name__ == "__main__":
    main()
