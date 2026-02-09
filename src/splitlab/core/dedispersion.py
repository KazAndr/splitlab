from __future__ import annotations

import numpy as np


def delays_pts_for_channels(freq_ghz: np.ndarray, dm: float, tsamp_sec: float) -> np.ndarray:
    """
    Robust delays: use f_ref = max(freq) regardless of bw sign.
    dt_ms = 4.148808 * DM * (f^-2 - f_ref^-2)
    pts = round(dt_ms / tsamp_ms)
    """
    if tsamp_sec <= 0:
        return np.zeros_like(freq_ghz, dtype=int)

    f_ref = float(np.nanmax(freq_ghz))
    t = 4.148808 * dm * (freq_ghz ** -2)
    t_ref = 4.148808 * dm * (f_ref ** -2)
    dt_ms = t - t_ref

    tsamp_ms = tsamp_sec * 1e3
    pts = np.rint(dt_ms / tsamp_ms).astype(int)
    return pts


def dedisperse_roll(data_f_t: np.ndarray, delays_pts: np.ndarray) -> np.ndarray:
    """
    data_f_t: (nchan, nt)
    delays_pts: (nchan,) shift in samples
    """
    out = np.empty_like(data_f_t)
    for i in range(data_f_t.shape[0]):
        out[i] = np.roll(data_f_t[i], -int(delays_pts[i]))
    return out

def dedisperse_shift(data_f_t: np.ndarray, delays_pts: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Shift channels left by delays_pts WITHOUT wrap-around (pad with fill_value).
    data_f_t: (nchan, nt)
    delays_pts: (nchan,), expected >= 0
    """
    nchan, nt = data_f_t.shape
    out = np.full_like(data_f_t, fill_value)
    for i in range(nchan):
        d = int(delays_pts[i])
        if d <= 0:
            out[i, :] = data_f_t[i, :]
        elif d < nt:
            out[i, : nt - d] = data_f_t[i, d:]
        # else: leave as fill_value
    return out