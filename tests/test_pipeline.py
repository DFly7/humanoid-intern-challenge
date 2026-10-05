import json
from pathlib import Path

import numpy as np
import pytest

from humanoid_challenge.data import prepare
from humanoid_challenge.geometry import homography, position_action, project, retime
from humanoid_challenge.lerobot_bridge import validate_runs

FIXTURE = Path(__file__).parents[1] / "examples/synthetic/manifest.json"


def test_perspective_calibration_maps_corners_and_rejects_degenerate_points():
    pixels = [[10, 20], [90, 10], [100, 95], [0, 80]]
    world = [[0, 0], [0.4, 0], [0.4, 0.3], [0, 0.3]]
    np.testing.assert_allclose(project(pixels, homography(pixels, world)), world, atol=1e-10)
    with pytest.raises(ValueError, match="Degenerate"):
        homography([[0, 0]] * 4, world)


def test_retiming_bounds_speed_and_preserves_contact():
    t, p, grip = retime(
        [0, 0.1, 0.2], [[0, 0, 0], [1, 0, 0], [1, 0, 0]], [-1, 1, -1], fps=20, max_speed=0.1
    )
    assert t[-1] >= 10.1
    assert np.max(np.linalg.norm(np.diff(p, axis=0), axis=1) * 20) <= 0.1 + 1e-9
    assert set(grip) == {-1, 1}
    assert grip[0] == -1 and grip[-1] == -1


def test_translation_uses_controller_units_and_saturates():
    np.testing.assert_allclose(
        position_action([0.025, -0.1, 0], [0, 0, 0], [0.05] * 3), [0.5, -1, 0]
    )


def test_brief_gripper_event_is_not_lost_between_control_ticks():
    _, _, grip = retime([0, 0.001, 0.002], np.zeros((3, 3)), [-1, 1, -1], 20, 0.1)
    np.testing.assert_array_equal(grip, [-1, 1, -1])


def test_prepare_keeps_synthetic_provenance_and_refuses_overwrite(tmp_path):
    output = tmp_path / "trajectory.npz"
    result = prepare(FIXTURE, output)
    assert result["manifest"]["source"] == "synthetic_fixture"
    assert len(result["track_sha256"]) == 64
    with np.load(output, allow_pickle=False) as data:
        assert data["position"].shape == (51, 3)
        assert np.isfinite(data["position"]).all()
    with pytest.raises(FileExistsError):
        prepare(FIXTURE, output)


def test_personal_recording_requires_the_original_video(tmp_path):
    manifest = json.loads(FIXTURE.read_text())
    manifest.update(source="personally_collected", video="missing.mov")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError):
        prepare(path, tmp_path / "trajectory.npz")


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("source", "synthetic_fixture", "Synthetic"),
        ("split", "test", "train"),
    ],
)
def test_export_rejects_synthetic_and_held_out_recordings(tmp_path, field, value, message):
    manifest = {"source": "personally_collected", "split": "train"}
    manifest[field] = value
    (tmp_path / "metrics.json").write_text(json.dumps({"provenance": {"manifest": manifest}}))
    with pytest.raises(ValueError, match=message):
        validate_runs([tmp_path])
