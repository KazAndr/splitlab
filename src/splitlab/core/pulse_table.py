from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pandas as pd


COLUMNS_NAME = [
    "mjd", "name1", "phase", "fname", "name2", "epoch",
    "name3", "max_to_rms", "name4", "rms", "name5", "sum", "name6", "snr"
]


def load_pulse_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, delimiter=r"\s+", names=COLUMNS_NAME, dtype="string")
    df = df[["mjd", "phase", "fname", "epoch", "max_to_rms", "rms", "sum", "snr"]].copy()

    df["mjd"] = df["mjd"].astype(float)
    for c in ["phase", "epoch", "max_to_rms", "rms", "sum", "snr"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def apply_split_period(df: pd.DataFrame, period_sec: float) -> pd.DataFrame:
    """
    Split mode: after each pulse row add another with mjd + P/2.
    """
    if period_sec <= 0:
        return df.copy()

    half_day = (period_sec / 2.0) / 86400.0
    out_rows = []
    for _, r in df.iterrows():
        out_rows.append(r)
        r2 = r.copy()
        r2["mjd"] = float(r["mjd"]) + half_day
        out_rows.append(r2)

    out = pd.DataFrame(out_rows).reset_index(drop=True)
    return out


def get_position_in_filfile(filterbank_file, mjd_pulse: float) -> int:
    """
    Return position (sample) in filterbank by MJD.
    Uses Decimal (same logic as your web code).
    """
    mjd_pulse = Decimal(str(mjd_pulse))
    mjd_start = Decimal(str(filterbank_file.your_header.tstart))

    delta_t_mjd = mjd_pulse - mjd_start
    delta_t_seconds = delta_t_mjd * Decimal(86400)

    tsamp = Decimal(str(filterbank_file.your_header.tsamp))
    location_in_the_file = delta_t_seconds / tsamp

    return int(location_in_the_file.to_integral_value(rounding="ROUND_HALF_UP"))