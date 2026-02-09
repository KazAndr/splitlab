from __future__ import annotations
import numpy as np

def t_func_dm_mu(dm: float, mu_ghz: float) -> float:
    return 4.148808 * dm * (mu_ghz ** -2)

def delays_dm_list(freq_ghz: np.ndarray, dm: float) -> np.ndarray:
    t2 = t_func_dm_mu(dm, float(freq_ghz[-1]))
    out = []
    for v in freq_ghz:
        t1 = t_func_dm_mu(dm, float(v))
        out.append(round(t1 - t2, 1))
    return np.array(out, dtype=float)

def dedisperse_roll(data_f_t: np.ndarray, delays_pts: np.ndarray) -> np.ndarray:
    out = np.empty_like(data_f_t)
    for i in range(data_f_t.shape[0]):
        out[i] = np.roll(data_f_t[i], -int(delays_pts[i]))
    return out
