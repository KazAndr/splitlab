from __future__ import annotations

import numpy as np


def safe_slice_x(arr2d: np.ndarray, x0: int, x1: int) -> np.ndarray:
    x0 = max(0, int(x0))
    x1 = min(int(x1), arr2d.shape[1])
    if x1 <= x0:
        return arr2d[:, 0:1]
    return arr2d[:, x0:x1]


def levels_by_percentile(arr2d: np.ndarray, qlow: float = 1, qhigh: float = 99) -> tuple[float, float]:
    a = np.asarray(arr2d)
    lo, hi = np.nanpercentile(a, [qlow, qhigh])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(a)) if np.isfinite(np.nanmin(a)) else 0.0
        hi = float(np.nanmax(a)) if np.isfinite(np.nanmax(a)) else 1.0
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return (0.0, 1.0)
    return (float(lo), float(hi))


def downsample_x(arr2d: np.ndarray, max_cols: int = 2048) -> tuple[np.ndarray, int]:
    """
    Return (downsampled, stride). stride=1 means no downsample.
    """
    w = int(arr2d.shape[1])
    if w <= max_cols:
        return arr2d, 1
    stride = int(np.ceil(w / max_cols))
    return arr2d[:, ::stride], stride