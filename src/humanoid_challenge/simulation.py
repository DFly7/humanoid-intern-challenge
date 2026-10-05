import json
import time
from pathlib import Path

import numpy as np

from .data import sha256
from .geometry import position_action


def list_tasks(suite):
    from libero.libero import benchmark

    tasks = benchmark.get_benchmark_dict()[suite]()
    return [
        {"id": i, "name": tasks.get_task(i).name, "instruction": tasks.get_task(i).language}
        for i in range(tasks.n_tasks)
    ]


def state(obs):
    # Unambiguous raw pose: xyz + quaternion xyzw + two finger joint positions.
    return np.concatenate(
        [obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]
    ).astype(np.float32)


def rollout(args):
    import imageio.v2 as imageio
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    trajectory_path = Path(args.trajectory)
    provenance = json.loads(trajectory_path.with_suffix(".json").read_text())
    with np.load(trajectory_path, allow_pickle=False) as file:
        targets, grips = file["position"].copy(), file["gripper"].copy()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    fps = provenance["fps"]
    suite = benchmark.get_benchmark_dict()[args.suite]()
    task = suite.get_task(args.task_id)
    # These initial-state pickles are from the pinned upstream checkout. Torch 2.6+
    # changed its default; avoid upstream's implicit torch.load for this trusted asset.
    states = torch.load(
        Path(get_libero_path("init_states")) / task.problem_folder / task.init_states_file,
        weights_only=False,
        map_location="cpu",
    )
    if not 0 <= args.init_id < len(states):
        raise ValueError(f"init-id must be between 0 and {len(states) - 1}")
    instruction = task.language
    if (
        provenance["manifest"]["source"] == "personally_collected"
        and provenance["manifest"]["instruction"] != instruction
    ):
        raise ValueError("Phone manifest instruction must match the selected LIBERO task")
    policy = None
    if args.policy:
        from .lerobot_bridge import SmolPolicy

        policy = SmolPolicy(args.policy, args.device, instruction)
    env = OffScreenRenderEnv(
        bddl_file_name=str(
            Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        ),
        camera_heights=args.resolution,
        camera_widths=args.resolution,
        control_freq=fps,
        horizon=len(targets) + 30,
    )
    observations, actions, positions, errors, latencies = [], [], [], [], []
    success = False
    start = time.perf_counter()
    try:
        env.seed(args.seed)
        env.reset()
        obs = env.set_init_state(states[args.init_id])
        for _ in range(10):
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        initial = {
            key: np.asarray(value).tolist()
            for key, value in obs.items()
            if key.endswith("_pos") or key == "robot0_eef_quat"
        }
        (output / "initial_state.json").write_text(json.dumps(initial, indent=2) + "\n")
        imageio.imwrite(output / "initial_scene.png", obs["agentview_image"][::-1])
        controller = env.robots[0].controller
        # Derive the translation gain from the live controller, not a guessed constant.
        scale = np.asarray(controller.output_max)[:3]
        if not np.allclose(np.asarray(controller.output_min)[:3], -scale):
            raise ValueError("Expected a symmetric OSC_POSE translation range")
        with imageio.get_writer(output / "rollout.mp4", fps=fps, macro_block_size=None) as video:
            for target, grip in zip(targets, grips):
                tick = time.perf_counter()
                if policy:
                    action = policy.act(obs)
                else:
                    action = np.r_[
                        position_action(target, obs["robot0_eef_pos"], scale), [0, 0, 0], grip
                    ].astype(np.float32)
                if action.shape != (7,) or not np.isfinite(action).all():
                    raise ValueError("Policy must produce seven finite OSC_POSE commands")
                action = np.clip(action, -1, 1)
                latencies.append(time.perf_counter() - tick)
                # Capture pre-action observations, not next-frame observations.
                observations.append(state(obs))
                for camera in ("agentview_image", "robot0_eye_in_hand_image"):
                    folder = output / camera
                    folder.mkdir(exist_ok=True)
                    # Keep raw simulator orientation for both training and VLA inference.
                    imageio.imwrite(folder / f"{len(actions):06d}.png", obs[camera])
                video.append_data(obs["agentview_image"][::-1])
                actions.append(action)
                obs, _, done, _ = env.step(action.tolist())
                positions.append(obs["robot0_eef_pos"].copy())
                errors.append(float(np.linalg.norm(target - obs["robot0_eef_pos"])))
                success = bool(env.check_success())
                if success or done:
                    break
        np.savez_compressed(
            output / "episode.npz", state=observations, action=actions, eef_position=positions
        )
        result = {
            "suite": args.suite,
            "task_id": args.task_id,
            "init_id": args.init_id,
            "seed": args.seed,
            "instruction": instruction,
            "success": success,
            "steps": len(actions),
            "fps": fps,
            "policy": args.policy or "phone_path_tracking",
            "tracking_error_mean_m": float(np.mean(errors)),
            "tracking_error_max_m": float(np.max(errors)),
            "action_latency_p50_ms": float(np.percentile(latencies, 50) * 1000),
            "action_latency_p95_ms": float(np.percentile(latencies, 95) * 1000),
            "elapsed_s": time.perf_counter() - start,
            "trajectory_sha256": sha256(trajectory_path),
            "provenance": provenance,
            "note": "Tracking error compares to the supplied path; success is LIBERO's predicate.",
        }
        (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        env.close()
