import numpy as np


def homography(pixels, table):
    """Fit a pixel -> tabletop metre homography from four non-collinear correspondences."""
    pixels, table = np.asarray(pixels, float), np.asarray(table, float)
    if pixels.shape != (4, 2) or table.shape != (4, 2):
        raise ValueError("Calibration needs exactly four corresponding 2D corners")
    if not np.isfinite(pixels).all() or not np.isfinite(table).all():
        raise ValueError("Calibration contains non-finite coordinates")
    rows, rhs = [], []
    for (u, v), (x, y) in zip(pixels, table):
        rows.extend([[u, v, 1, 0, 0, 0, -x * u, -x * v], [0, 0, 0, u, v, 1, -y * u, -y * v]])
        rhs.extend([x, y])
    if np.linalg.matrix_rank(rows) < 8:
        raise ValueError("Degenerate calibration: use four distinct table corners")
    return np.append(np.linalg.solve(rows, rhs), 1).reshape(3, 3)


def project(points, matrix):
    points = np.asarray(points, float)
    xyz = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    if np.any(np.abs(xyz[:, 2]) < 1e-9):
        raise ValueError("Point projects to infinity")
    return xyz[:, :2] / xyz[:, 2:3]


def retime(times, positions, grips, fps, max_speed):
    """Slow fast segments; preserve pauses and discrete contact transitions."""
    times, positions, grips = map(np.asarray, (times, positions, grips))
    if fps <= 0 or max_speed <= 0 or not np.isfinite([fps, max_speed]).all():
        raise ValueError("fps and max_speed must be finite and positive")
    dt = np.maximum(np.diff(times), np.linalg.norm(np.diff(positions, axis=0), axis=1) / max_speed)
    # Give every annotated phase at least one control frame so brief contact states
    # are not skipped when a phone timestamp falls between simulator ticks.
    dt = np.maximum(dt, 1 / fps)
    warped = np.concatenate([[0.0], np.cumsum(dt)])
    # ceil + fixed dt: the final observation may hold the last target briefly.
    grid = np.arange(int(np.ceil(warped[-1] * fps)) + 1) / fps
    xyz = np.column_stack([np.interp(grid, warped, positions[:, i]) for i in range(3)])
    # Zero-order hold: never interpolate through ambiguous gripper values.
    idx = np.searchsorted(warped, grid, side="right") - 1
    return grid, xyz, grips[np.clip(idx, 0, len(grips) - 1)]


def position_action(target, current, output_max):
    """OSC_POSE scales normalized translation by output_max, in metres/step."""
    scale = np.asarray(output_max, float)
    if scale.shape != (3,) or np.any(scale <= 0):
        raise ValueError("Expected three positive controller translation scales")
    return np.clip((np.asarray(target) - np.asarray(current)) / scale, -1, 1)
