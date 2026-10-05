# Collect a small dataset you can explain

The challenge requires **your own recordings**. The synthetic example only checks the software.
Start with one manipulation task and finish an interpretable baseline before training a VLA.

## Choose the task in simulation first

Run `humanoid tasks --suite libero_goal` and choose a task with one short interaction.
Inspect its objects, table height, and initial state. Reproduce that arrangement on your table
with household objects. A phone-only video does not contain Panda actions: this project derives
end-effector targets from manually annotated hand motion, then records the controller actions
actually executed in the simulator.

The initial implementation keeps the Panda's initial wrist orientation fixed. Use a task that
can be performed with a downward grasp. Tasks requiring wrist rotation need an orientation
annotation and controller extension; drawer interactions may also need force/contact handling.

## Record

1. Mount a phone so it cannot move, with the whole work area visible. Use landscape 1080p/30fps.
   Use an overhead view where possible. The current homography assumes a fixed camera;
   an unstabilized moving egocentric camera is unsupported.
2. Put four visible reference marks around a rectangle on the table. Measure their coordinates
   in metres. Keep the same corner order in `pixels` and `table_xy_m`.
3. Record 10–20 short attempts to debug the pipeline. If the baseline works, add more variation
   and recordings before claiming policy generalization. Include reaches, pauses, grip changes,
   and releases. Keep your hand and object visible.
4. Keep complete source recordings. Store them under `data/raw/session_01/`; this directory is
   ignored by Git. Publish a short consented example and a data manifest when preparing results.
   Do not publish a hash as if it were the data itself.
5. Divide **recordings/sessions** into train, validation, and test before conversion. Never split
   adjacent frames of the same video into different sets. Keep an additional set of simulator
   initial-state IDs out of training.

## Annotate

Open `tools/annotate.html` in a browser and load your local video. Pause and click the centre of
the intended grasp/push contact. Use the video's native pixel coordinates for calibration.
Click all four reference corners at separate times to read their pixel coordinates in the table,
then undo those calibration clicks so they are not exported as hand motion. Download `track.csv`.

Each row has:

| Column | Meaning |
| --- | --- |
| `time_s` | Time in the original recording; strictly increasing |
| `u_px`, `v_px` | Image coordinates of the annotated grasp centre |
| `height_m` | Manually estimated grasp-centre height above the table |
| `gripper` | `-1` for open, `+1` for close |
| `confidence` | Annotation quality between 0 and 1; default threshold is 0.8 |

Add points at least every second, and more densely around contact. Height is **not inferred**
from monocular video. Planar homography is accurate on the table plane; mapping an elevated
hand through it introduces parallax. An overhead camera reduces this problem. Check target
paths against the simulator before treating them as demonstrations. Mark uncertain points and
re-annotate them; the converter rejects low confidence rather than silently filling large gaps.

## Create a manifest

Copy `examples/synthetic/manifest.json` beside your recording and edit:

```json
{
  "source": "personally_collected",
  "collected_by": "DFly7",
  "split": "train",
  "instruction": "<the exact selected LIBERO task instruction>",
  "video": "attempt_001.mov",
  "track": "track.csv",
  "fps": 20,
  "max_speed_m_s": 0.15,
  "min_confidence": 0.8,
  "max_gap_s": 1.0,
  "calibration": {
    "pixels": [[100,100],[500,100],[500,500],[100,500]],
    "table_xy_m": [[0,0],[0.4,0],[0.4,0.4],[0,0.4]]
  },
  "robot_mapping": {
    "yaw_rad": 0,
    "offset_xy_m": [-0.1,-0.1],
    "table_z_m": 0.8,
    "workspace_min": [-0.3,-0.3,0.81],
    "workspace_max": [0.3,0.3,1.3]
  }
}
```

**Those calibration/mapping numbers are examples, not measured values.** Measure your rectangle,
read the simulated workspace, and align the hand/object paths. `table_z_m` is the simulator's
world-frame table height; `yaw_rad` and `offset_xy_m` explicitly align the measured human table
to the robot base. The initial version uses rigid alignment, not a guessed body-scale factor.
Workspace limits reject targets; they do not perform collision checking.

If changing alignment to suit a randomized initial state, record that alignment as a new
configuration. Access to object pose for alignment is privileged simulator information. Report
it as part of the retargeting teacher; the trained VLA receives images, robot pose, and language.

Run `humanoid prepare` for each manifest. The sidecar records source hashes, parameters, and
retimed duration so another person can trace each demonstration back to its original video.
Hashes support traceability; they are not proof of who collected a recording.

