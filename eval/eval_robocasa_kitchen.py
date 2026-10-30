#!/usr/bin/env python3
"""RoboCasa-Kitchen (24 tasks, Panda-Omron) evaluation of a GR00T N1.5 checkpoint served over ZMQ, fixed horizon.

Why a custom client: Isaac-GR00T n1.5-release only ships a RoboCasa *GR1-tabletop* client. The 24 kitchen tasks the
paper uses are RoboCasa v0.2 (robosuite 1.5) tasks; observation/action key mapping below is ported from
robocasa(main)/robocasa/wrappers/gym_wrapper.py::PandaOmronKeyConverter, renamed to the keys of GR00T's
`single_panda_gripper` data config.

  obs  : cameras  robot0_agentview_left / robot0_agentview_right / robot0_eye_in_hand -> video.left_view / right_view / wrist_view
         proprio  robot0_base_to_eef_pos/quat, robot0_gripper_qpos, robot0_base_pos/quat -> state.*
  action: end_effector_position(3) + end_effector_rotation(3) -> arm ; gripper_close -> +-1 ;
          base_motion(4) -> base(3)+torso(1) ; control_mode -> base_mode +-1

Run in .venv-robocasa:
  python eval/eval_robocasa_kitchen.py --port 5555 --horizon 16 --trials 100 --out outputs/eval/rc.json
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

GROUPS = {  # paper Sec. 5.1.1
    "Relocation": ["PnPCounterToCab", "PnPCabToCounter", "PnPCounterToSink", "PnPSinkToCounter", "PnPCounterToStove",
                   "PnPStoveToCounter", "PnPCounterToMicrowave", "PnPMicrowaveToCounter", "CoffeeSetupMug", "CoffeeServeMug"],
    "Container": ["OpenSingleDoor", "CloseSingleDoor", "OpenDoubleDoor", "CloseDoubleDoor", "OpenDrawer", "CloseDrawer"],
    "Rotation": ["TurnOnSinkFaucet", "TurnOffSinkFaucet", "TurnSinkSpout", "TurnOnStove", "TurnOffStove"],
    "Button": ["CoffeePressButton", "TurnOnMicrowave", "TurnOffMicrowave"],
}
ALL_TASKS = [t for g in GROUPS.values() for t in g]
CAMS = [("robot0_agentview_left", "video.left_view"),
        ("robot0_agentview_right", "video.right_view"),
        ("robot0_eye_in_hand", "video.wrist_view")]
RES = 256


def make_env(task, seed):
    from robocasa.utils.env_utils import create_env

    return create_env(env_name=task, robots="PandaOmron", camera_names=[c for c, _ in CAMS],
                      camera_widths=RES, camera_heights=RES, seed=seed, render_onscreen=False)


def make_obs(env, raw):
    # robosuite images are upside-down -> flip vertically; add time dim T=1
    obs = {name: np.copy(raw[f"{cam}_image"][::-1])[None] for cam, name in CAMS}
    obs.update({
        "state.end_effector_position_relative": np.asarray(raw["robot0_base_to_eef_pos"], dtype=np.float64)[None],
        "state.end_effector_rotation_relative": np.asarray(raw["robot0_base_to_eef_quat"], dtype=np.float64)[None],
        "state.gripper_qpos": np.asarray(raw["robot0_gripper_qpos"], dtype=np.float64)[None],
        "state.base_position": np.asarray(raw["robot0_base_pos"], dtype=np.float64)[None],
        "state.base_rotation": np.asarray(raw["robot0_base_quat"], dtype=np.float64)[None],
        "annotation.human.action.task_description": [env.get_ep_meta().get("lang", "")],
    })
    return obs


def to_env_action(env, chunk, i):
    g = lambda k: np.atleast_1d(chunk[f"action.{k}"][i]).astype(np.float64)  # noqa: E731
    parts = {
        "right": np.concatenate([g("end_effector_position"), g("end_effector_rotation")]),
        "right_gripper": -1.0 if g("gripper_close")[0] < 0.5 else 1.0,
        "base": g("base_motion")[0:3],
        "torso": g("base_motion")[3:4],
    }
    base_mode = -1.0 if g("control_mode")[0] < 0.5 else 1.0
    vec = []
    for robot in env.robots:
        cc = robot.composite_controller
        a = np.zeros(cc.action_limits[0].shape)
        for part in cc.part_controllers:
            s, e = cc._action_split_indexes[part]
            a[s:e] = parts[part]
        if type(cc).__name__ == "HybridMobileBase":
            a[-1] = base_mode
        vec.append(a)
    return np.concatenate(vec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=720, help="env steps per episode (assumption, not given in paper)")
    ap.add_argument("--tasks", nargs="*", default=ALL_TASKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-video-dir", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    client = PolicyClient(args.host, args.port)
    client.ping()
    results = {"horizon": args.horizon, "trials_per_task": args.trials, "max_steps": args.max_steps, "tasks": {}}
    t_start = time.time()
    for task in args.tasks:
        env = make_env(task, args.seed)
        succ = 0
        for ep in range(args.trials):
            raw = env.reset()
            ok, t, frames = False, 0, []
            while t < args.max_steps and not ok:
                chunk = client.get_action(make_obs(env, raw))
                for i in range(select_horizon(chunk, args.horizon)):
                    raw, _, _, _ = env.step(to_env_action(env, chunk, i))
                    t += 1
                    if args.save_video_dir:
                        frames.append(raw[f"{CAMS[0][0]}_image"][::-1])
                    ok = bool(env._check_success())
                    if ok or t >= args.max_steps:
                        break
            succ += int(ok)
            if args.save_video_dir and frames:
                import imageio

                Path(args.save_video_dir).mkdir(parents=True, exist_ok=True)
                imageio.mimsave(f"{args.save_video_dir}/{task}_e{ep}_{'ok' if ok else 'fail'}.mp4", frames, fps=20)
            print(f"[{task}] ep {ep + 1}/{args.trials} success={ok} running={succ}/{ep + 1}", flush=True)
        env.close()
        results["tasks"][task] = succ / args.trials
        print(f"==> {task}: {100 * succ / args.trials:.1f}%", flush=True)

    sr = results["tasks"]
    results["groups"] = {g: float(np.mean([sr[t] for t in ts if t in sr])) for g, ts in GROUPS.items()
                         if any(t in sr for t in ts)}
    results["avg"] = float(np.mean(list(sr.values())))
    results["minutes"] = (time.time() - t_start) / 60
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps({"groups": results["groups"], "avg": results["avg"]}, indent=2))


if __name__ == "__main__":
    main()

