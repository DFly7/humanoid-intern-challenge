import json
from pathlib import Path

import numpy as np

from .data import sha256
from .simulation import state


class SmolPolicy:
    def __init__(self, checkpoint, device, instruction):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        self.torch, self.instruction = torch, instruction
        self.policy = SmolVLAPolicy.from_pretrained(checkpoint).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def act(self, obs):
        batch = {
            "observation.state": self.torch.from_numpy(state(obs)).unsqueeze(0),
            "task": [self.instruction],
        }
        for feature, camera in (
            ("image", "agentview_image"),
            ("image2", "robot0_eye_in_hand_image"),
        ):
            # Same raw images as the dataset; HWC uint8 -> BCHW float [0,1].
            batch[f"observation.images.{feature}"] = (
                self.torch.from_numpy(obs[camera].copy()).permute(2, 0, 1).unsqueeze(0).float()
                / 255
            )
        with self.torch.inference_mode():
            action = self.post(self.policy.select_action(self.pre(batch)))
        return action.squeeze(0).cpu().numpy()


def validate_runs(paths):
    """Do all provenance checks before creating a dataset directory."""
    runs, seen, tasks = [], set(), set()
    for path in map(Path, paths):
        metrics = json.loads((path / "metrics.json").read_text())
        manifest = metrics["provenance"]["manifest"]
        if manifest["source"] != "personally_collected":
            raise ValueError("Synthetic smoke tests cannot be used as applicant training data")
        if manifest["split"] != "train":
            raise ValueError("Export only train recordings; preserve held-out validation/test data")
        if not metrics["success"]:
            raise ValueError("Inspect and improve failed trajectories before imitation training")
        if metrics["policy"] != "phone_path_tracking":
            raise ValueError("Export retargeted demonstrations, not the VLA's own predictions")
        if not metrics["provenance"].get("video_sha256"):
            raise ValueError("Missing original video hash")
        fingerprint = sha256(path / "episode.npz")
        if fingerprint in seen:
            raise ValueError("Duplicate simulation episode")
        seen.add(fingerprint)
        tasks.add((metrics["suite"], metrics["task_id"]))
        runs.append((path, metrics, fingerprint))
    if len(tasks) != 1:
        raise ValueError("This first experiment deliberately trains one task at a time")
    if len({m["fps"] for _, m, _ in runs}) != 1:
        raise ValueError("All episodes must have the same control frequency")
    return runs


def export(args):
    runs = validate_runs(args.runs)
    import imageio.v2 as imageio
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    first = imageio.imread(runs[0][0] / "agentview_image/000000.png")
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": ["x", "y", "z", "qx", "qy", "qz", "qw", "finger1", "finger2"],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"],
        },
    }
    for feature in ("image", "image2"):
        features[f"observation.images.{feature}"] = {
            "dtype": "image",
            "shape": first.shape,
            "names": ["height", "width", "channels"],
        }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=Path(args.output),
        fps=runs[0][1]["fps"],
        features=features,
        robot_type="panda_osc_pose",
        use_videos=False,
        image_writer_threads=2,
    )
    try:
        for path, metrics, _ in runs:
            with np.load(path / "episode.npz", allow_pickle=False) as episode:
                for i, (observation, action) in enumerate(zip(episode["state"], episode["action"])):
                    frame = {
                        "observation.state": observation.astype(np.float32),
                        "action": action.astype(np.float32),
                        "task": metrics["instruction"],
                    }
                    for feature, camera in (
                        ("image", "agentview_image"),
                        ("image2", "robot0_eye_in_hand_image"),
                    ):
                        frame[f"observation.images.{feature}"] = imageio.imread(
                            path / camera / f"{i:06d}.png"
                        )
                    dataset.add_frame(frame)
                dataset.save_episode()
    finally:
        dataset.finalize()
    provenance = [{"episode_sha256": digest, "metrics": metrics} for _, metrics, digest in runs]
    (Path(args.output) / "phone_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    return {"episodes": len(runs), "output": args.output, "published_to_hub": False}
