from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton,
    QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox, QComboBox, QSpinBox,
    QTextEdit
)
import pyqtgraph as pg

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        grid = QGridLayout(root)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(8)

        self.dm_period = pg.ImageView()
        self.dm_period.ui.roiBtn.hide()
        self.dm_period.ui.menuBtn.hide()
        box1 = self._boxed("1) DM-time (one period)", self.dm_period)

        self.dm_segments = pg.ImageView()
        self.dm_segments.ui.roiBtn.hide()
        self.dm_segments.ui.menuBtn.hide()
        box2 = self._boxed("2) 3 DM segments (center ±1)", self.dm_segments)

        self.cb_left = QCheckBox("Left (seg-1)")
        self.cb_center = QCheckBox("Center (seg)")
        self.cb_right = QCheckBox("Right (seg+1)")
        self.cb_center.setChecked(True)
        radios = QWidget()
        v = QVBoxLayout(radios)
        v.addWidget(self.cb_left)
        v.addWidget(self.cb_center)
        v.addWidget(self.cb_right)
        v.addStretch(1)
        box3 = self._boxed("3) Segment selection", radios)

        self.fb_view = pg.ImageView()
        self.fb_view.ui.roiBtn.hide()
        self.fb_view.ui.menuBtn.hide()
        self.profile_plot = pg.PlotWidget()
        fb = QWidget()
        vv = QVBoxLayout(fb)
        vv.addWidget(self.fb_view, 3)
        vv.addWidget(self.profile_plot, 1)
        box4 = self._boxed("4) Dedispersed filterbank + profile", fb)

        self.info = QTextEdit()
        self.info.setReadOnly(True)
        box5 = self._boxed("5) Info", self.info)

        self.btn_load_fil = QPushButton("Load .fil")
        self.btn_load_dm = QPushButton("Load DM-time .npy")
        self.btn_load_mjd = QPushButton("Load MJD table")
        loaders = QWidget()
        l = QVBoxLayout(loaders)
        l.addWidget(self.btn_load_fil)
        l.addWidget(self.btn_load_dm)
        l.addWidget(self.btn_load_mjd)
        l.addStretch(1)
        box6 = self._boxed("6) Data loading", loaders)

        self.cb_auto = QCheckBox("Using auto labeling")
        self.period_edit = QLineEdit("1.0")
        self.period_unit = QComboBox()
        self.period_unit.addItems(["s", "ms", "us"])
        self.dm_edit = QLineEdit("57.0")
        self.cb_split = QCheckBox("Split the period (P/2)")
        params = QWidget()
        p = QGridLayout(params)
        p.addWidget(self.cb_auto, 0, 0, 1, 2)
        p.addWidget(QLabel("Pulsar period:"), 1, 0)
        p.addWidget(self.period_edit, 1, 1)
        p.addWidget(self.period_unit, 1, 2)
        p.addWidget(QLabel("DM:"), 2, 0)
        p.addWidget(self.dm_edit, 2, 1)
        p.addWidget(self.cb_split, 3, 0, 1, 2)
        box7 = self._boxed("7) Parameters", params)

        self.seg_spin = QSpinBox()
        self.seg_spin.setRange(0, 2_000_000_000)
        self.btn_jump = QPushButton("Jump")
        jump = QWidget()
        jj = QHBoxLayout(jump)
        jj.addWidget(QLabel("seg_id:"))
        jj.addWidget(self.seg_spin, 1)
        jj.addWidget(self.btn_jump)
        box8 = self._boxed("8) Jump to segment index", jump)

        self.table_placeholder = QLabel("Labels table (QTableView) — TODO")
        self.table_placeholder.setAlignment(Qt.AlignCenter)
        box9 = self._boxed("9) Labels table", self.table_placeholder)

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_yes = QPushButton("Y: Positive")
        self.btn_no = QPushButton("N: Negative")
        nav = QWidget()
        nn = QGridLayout(nav)
        nn.addWidget(self.btn_prev, 0, 0)
        nn.addWidget(self.btn_next, 0, 1)
        nn.addWidget(self.btn_yes, 1, 0)
        nn.addWidget(self.btn_no, 1, 1)
        box10 = self._boxed("10) Navigation & labeling", nav)

        self.cb_autosave = QCheckBox("Autosave after each label")
        self.cb_autosave.setChecked(True)
        box11 = self._boxed("11) Saving", self.cb_autosave)

        self.mode = QComboBox()
        self.mode.addItems(["Viewing", "Labeling"])
        box12 = self._boxed("12) Mode", self.mode)

        grid.addWidget(box1, 0, 0, 2, 2)
        grid.addWidget(box2, 2, 0, 2, 2)
        grid.addWidget(box4, 4, 0, 2, 2)

        right = QVBoxLayout()
        for b in (box5, box6, box7, box8, box3, box9, box10, box11, box12):
            right.addWidget(b)
        right.addStretch(1)
        rightw = QWidget()
        rightw.setLayout(right)
        grid.addWidget(rightw, 0, 2, 6, 1)

        self.setCentralWidget(root)
        self.resize(1400, 900)

    def _boxed(self, title: str, widget):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.addWidget(widget)
        return box
