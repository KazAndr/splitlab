from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .pulse_table import load_pulse_table, apply_split_period, get_position_in_filfile
from .utils import safe_slice_x, downsample_x
from .dedispersion import delays_pts_for_channels, dedisperse_shift

try:
    from your import Your  # type: ignore
except Exception:  # pragma: no cover
    import your  # type: ignore
    Your = your.Your  # type: ignore


@dataclass
class RenderedWindow:
    image: np.ndarray          # shown image (maybe downsampled, maybe flipped)
    x0_global: int             # global start sample of the non-downsampled window
    stride: int                # display_x * stride -> real sample offset
    width_real: int            # real width in samples (before downsample)


class DataManager:
    def __init__(self, segment_size: int = 256):
        self.SEG = int(segment_size)

        self.edmt: np.ndarray | None = None   # memmap npy (n_dm, n_time)
        self.fil = None                       # Your(...)
        self.df_base: pd.DataFrame | None = None
        self.df: pd.DataFrame | None = None   # derived (maybe split)

        self.period_sec: float = 1.0
        self.dm: float = 57.0
        self.split: bool = False

        self.labels_path: Path | None = None

    # ---------- loading ----------
    def load_edmt(self, path: str | Path) -> None:
        self.edmt = np.load(Path(path), mmap_mode="r")

    def load_fil(self, path: str | Path) -> None:
        self.fil = Your(str(path))

    def load_pulses(self, path: str | Path) -> None:
        self.df_base = load_pulse_table(path)
        self._rebuild_df()

    def set_period(self, value: float, unit: str) -> None:
        unit = unit.strip().lower()
        if unit == "s":
            self.period_sec = float(value)
        elif unit == "ms":
            self.period_sec = float(value) / 1e3
        elif unit == "us":
            self.period_sec = float(value) / 1e6
        else:
            self.period_sec = float(value)
        self._rebuild_df()

    def set_dm(self, dm: float) -> None:
        self.dm = float(dm)

    def set_split(self, split: bool) -> None:
        self.split = bool(split)
        self._rebuild_df()

    def ready_minimal(self) -> bool:
        return self.edmt is not None and self.fil is not None and self.df is not None

    # ---------- table / derived ----------
    def _rebuild_df(self) -> None:
        if self.df_base is None:
            self.df = None
            return
        if self.split:
            self.df = apply_split_period(self.df_base, self.period_sec)
        else:
            self.df = self.df_base.copy().reset_index(drop=True)

    def window_samples(self) -> int:
        if self.fil is None:
            return 0
        tsamp_sec = float(self.fil.your_header.tsamp)
        if tsamp_sec <= 0:
            return 0
        return int(round(self.period_sec / tsamp_sec))

    def start_pos_for_row(self, row_idx: int) -> int:
        if self.df is None or self.fil is None:
            return 0
        mjd = float(self.df.iloc[row_idx]["mjd"])
        return get_position_in_filfile(self.fil, mjd)

    def expected_seg_center_for_row(self, row_idx: int) -> int:
        # for resume heuristic (key is seg_center)
        start_pos = self.start_pos_for_row(row_idx)
        return int(start_pos // self.SEG)

    # ---------- rendering ----------
    def render_period_window(self, row_idx: int, max_cols: int = 2048) -> RenderedWindow:
        assert self.edmt is not None and self.df is not None and self.fil is not None

        win = self.window_samples()
        start = self.start_pos_for_row(row_idx)
        x0 = start
        x1 = start + win

        dm_slice = safe_slice_x(self.edmt, x0, x1)  # (n_dm, win)
        dm_ds, stride = downsample_x(dm_slice, max_cols=max_cols)

        
        img = np.asarray(dm_ds)
        return RenderedWindow(image=img, x0_global=x0, stride=stride, width_real=int(dm_slice.shape[1]))

    def render_segments_triplet(self, center_seg: int) -> np.ndarray:
        assert self.edmt is not None
        x0 = (center_seg - 1) * self.SEG
        x1 = x0 + 3 * self.SEG
        dm_slice = safe_slice_x(self.edmt, x0, x1)
        return np.asarray(dm_slice)

    def channel_freq_mhz(self) -> np.ndarray:
        freq, _ = self._channel_freq_mhz_and_foff()
        return freq

    def _channel_freq_mhz_and_foff(self) -> tuple[np.ndarray, float]:
        """
        Build per-channel frequency grid in MHz in the SAME order as data channels.

        Prefer hdr.foff if present.
        Otherwise use hdr.bw:
        - if |bw| > 5 MHz -> treat as TOTAL bandwidth and use foff = bw / nchans
        - else -> treat as per-channel offset directly
        """
        assert self.fil is not None
        hdr = self.fil.your_header
        nch = int(hdr.nchans)
        fch1 = float(hdr.fch1)

        if hasattr(hdr, "foff"):
            foff = float(getattr(hdr, "foff"))
        else:
            bw = float(getattr(hdr, "bw", 0.0))
            # Heuristic: large magnitude looks like total BW (e.g. -400 MHz)
            foff = (bw / nch) if abs(bw) > 5.0 else bw

        freq_mhz = fch1 + np.arange(nch, dtype=float) * foff
        return freq_mhz, float(foff)

    def read_filterbank_triplet(self, center_seg: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Read exactly enough samples on the right to compensate max DM delay,
        then dedisperse without wrap and return only 3*SEG for display.

        Returns (raw_3, dd_3, prof_3)
        raw_3: (nchan, 3*SEG)
        dd_3:  (nchan, 3*SEG)
        prof_3:(3*SEG,)
        """
        assert self.fil is not None
        hdr = self.fil.your_header

        fb_start = max(0, (center_seg - 1) * self.SEG)
        nsamp_show = 3 * self.SEG

        # compute per-channel delays in samples
        freq_mhz, _ = self._channel_freq_mhz_and_foff()
        freq_ghz = (freq_mhz / 1000.0).astype(float)

        tsamp_sec = float(hdr.tsamp)
        delays_pts = delays_pts_for_channels(freq_ghz, self.dm, tsamp_sec)
        max_delay = int(np.max(delays_pts)) if delays_pts.size else 0

        # read enough samples to avoid right-edge padding artefact
        nsamp_read = nsamp_show + max_delay + 1

        raw = self.fil.get_data(nstart=fb_start, nsamp=nsamp_read)

        # Normalize to (nchan, nt)
        if raw.shape[0] == nsamp_read:
            data_f_t_full = raw.T
        else:
            data_f_t_full = raw
        data_f_t_full = np.asarray(data_f_t_full, dtype=np.float32)

        # dedisperse WITHOUT wrap
        dd_full = dedisperse_shift(data_f_t_full, delays_pts)

        # crop: show only first 3 segments
        raw_3 = data_f_t_full[:, :nsamp_show]
        dd_3 = dd_full[:, :nsamp_show]
        prof_3 = np.sum(dd_3, axis=0)

        return raw_3, dd_3, prof_3