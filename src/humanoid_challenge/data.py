import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import homography, project, retime


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(manifest_path, output):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["source"] not in ("personally_collected", "synthetic_fixture"):
        raise ValueError("source must be personally_collected or synthetic_fixture")
    if not manifest.get("collected_by") or not manifest.get("instruction"):
        raise ValueError("Specify collected_by and instruction")
    if manifest.get("split") not in ("train", "validation", "test", "smoke"):
        raise ValueError("Specify a recording-level split")
    provenance = {"manifest_sha256": sha256(manifest_path), "manifest": manifest}
    if manifest["source"] == "personally_collected":
        video = manifest_path.parent / manifest["video"]
        if video.suffix.lower() not in (".mp4", ".mov", ".m4v"):
            raise ValueError("Provide the original phone video, not a replacement artifact")
        provenance["video_sha256"] = sha256(video)
    track = manifest_path.parent / manifest["track"]
    provenance["track_sha256"] = sha256(track)
    with track.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 2:
        raise ValueError("Annotate at least two points")
    values = np.array(
        [
            [float(row[k]) for k in ("time_s", "u_px", "v_px", "height_m", "gripper", "confidence")]
            for row in rows
        ]
    )
    if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0):
        raise ValueError("Annotations must be finite with strictly increasing timestamps")
    if np.any(values[:, 0] < 0) or np.any(values[:, 3] < 0):
        raise ValueError("Time and manually estimated height must be non-negative")
    if not np.isin(values[:, 4], [-1, 1]).all():
        raise ValueError("Gripper convention is -1=open, +1=close")
    if np.any((values[:, 5] < 0) | (values[:, 5] > 1)):
        raise ValueError("Confidence must lie between zero and one")
    if np.any(values[:, 5] < manifest.get("min_confidence", 0.8)):
        raise ValueError("Low-confidence annotation: inspect the video; do not hide occlusion")
    if np.max(np.diff(values[:, 0])) > manifest.get("max_gap_s", 1.0):
        raise ValueError("Annotation gap too large; add points around the missing motion")
    calibration = manifest["calibration"]
    matrix = homography(calibration["pixels"], calibration["table_xy_m"])
    xy = project(values[:, 1:3], matrix)
    # Human table -> robot base: explicitly supplied rigid planar alignment.
    yaw = float(manifest["robot_mapping"].get("yaw_rad", 0))
    rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    mapping = manifest["robot_mapping"]
    xy = xy @ rotation.T + np.asarray(mapping["offset_xy_m"])
    xyz = np.column_stack([xy, values[:, 3] + mapping["table_z_m"]])
    low, high = np.asarray(mapping["workspace_min"]), np.asarray(mapping["workspace_max"])
    if low.shape != (3,) or high.shape != (3,) or np.any(low >= high):
        raise ValueError("Specify valid 3D workspace bounds")
    if not np.isfinite(xyz).all() or np.any(xyz < low) or np.any(xyz > high):
        raise ValueError("Target outside configured workspace: correct calibration/alignment")
    fps = int(manifest.get("fps", 20))
    time, position, gripper = retime(
        values[:, 0], xyz, values[:, 4], fps, manifest.get("max_speed_m_s", 0.15)
    )
    provenance.update(
        fps=fps,
        input_duration_s=float(values[-1, 0] - values[0, 0]),
        retimed_duration_s=float(time[-1]),
        frames=len(time),
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    with output.open("wb") as stream:
        np.savez_compressed(stream, time=time, position=position, gripper=gripper)
    output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance
