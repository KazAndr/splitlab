from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

@dataclass
class LabelRow:
    row_idx: int
    name_of_set: str
    mjd: float
    phase: float | None
    fname: str
    snr: float | None
    start_pos: int
    clicked_local_x: int | None
    clicked_global_x: int | None
    seg_left: int
    seg_center: int
    seg_right: int
    label_left: int
    label_center: int
    label_right: int
    comment: str = ""

class LabelStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.df = self._load_or_empty()

    def _load_or_empty(self) -> pd.DataFrame:
        if self.path.exists():
            try:
                return pd.read_csv(self.path)
            except Exception:
                return pd.DataFrame()
        return pd.DataFrame()

    def upsert(self, row: LabelRow) -> None:
        new = pd.DataFrame([asdict(row)])
        if self.df.empty:
            self.df = new
        else:
            if "seg_center" in self.df.columns:
                self.df = self.df[self.df["seg_center"] != row.seg_center]
            if "row_idx" in self.df.columns:
                self.df = self.df[self.df["row_idx"] != row.row_idx]
            self.df = pd.concat([self.df, new], ignore_index=True)
        self.df.to_csv(self.path, index=False)

    def is_labeled(self, seg_center: int) -> bool:
        if self.df.empty or "seg_center" not in self.df.columns:
            return False
        return bool((self.df["seg_center"] == seg_center).any())
