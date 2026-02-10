from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QDoubleValidator, QKeySequence
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QGridLayout, QLabel, QLineEdit, QPushButton,
    QGroupBox, QHBoxLayout, QVBoxLayout, QCheckBox, QComboBox, QSpinBox,
    QTextEdit, QFileDialog, QMessageBox, QAction
)

from splitlab.core.data_manager import DataManager
from splitlab.core.label_store import LabelRow, LabelStore
from splitlab.core.utils import levels_by_percentile


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.mgr = DataManager(segment_size=256)
        self._lut = self._plasma_lut()

        self.labels: LabelStore | None = None
        self.current_row_idx: int = 0

        # period-window render mapping
        self._period_x0_global: int = 0
        self._period_stride: int = 1
        self._period_width_real: int = 0

        # click state
        self.clicked_global_x: int | None = None
        self.clicked_local_x: int | None = None
        self.center_seg: int | None = None

        self._pending_comment: str = ""

        self._build_ui()
        self._wire()
        self._apply_mode()

    # ---------------- UI ----------------
    def _simplify_imageview(self, iv: pg.ImageView) -> None:
        # remove UI parts (histogram/levels pane, ROI/menu buttons)
        try:
            iv.ui.histogram.hide()
        except Exception:
            pass
        try:
            iv.ui.roiBtn.hide()
        except Exception:
            pass
        try:
            iv.ui.menuBtn.hide()
        except Exception:
            pass
        try:
            iv.ui.normGroup.hide()
        except Exception:
            pass

    def _tune_imageview(self, iv: pg.ImageView, invert_y: bool = False) -> None:
        """Make image views fill their box: no padding, unlocked aspect, optional Y invert."""
        vb = iv.getView()
        vb.setAspectLocked(False)
        vb.setDefaultPadding(0.0)
        vb.enableAutoRange(x=True, y=True)
        if invert_y:
            vb.invertY(True)

    def _plasma_lut(self, n: int = 256):
        # Prefer pyqtgraph colormap; fallback to matplotlib if available
        try:
            cmap = pg.colormap.get("plasma", source="matplotlib")
            return cmap.getLookupTable(nPts=n, alpha=False)
        except Exception:
            pass

        try:
            import numpy as np
            import matplotlib.cm as cm
            m = cm.get_cmap("plasma", n)
            lut = (m(np.linspace(0, 1, n))[:, :3] * 255).astype(np.ubyte)
            return lut
        except Exception:
            return None

    def _build_ui(self):
        root = QWidget()
        grid = QGridLayout(root)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setSpacing(6)

        # Block 1
        self.dm_period = pg.ImageView()
        self._simplify_imageview(self.dm_period)
        self.dm_period.ui.roiBtn.hide()
        self.dm_period.ui.menuBtn.hide()
        self._tune_imageview(self.dm_period, invert_y=True)
        self._period_vline = pg.InfiniteLine(angle=90, movable=False)
        self.dm_period.getView().addItem(self._period_vline)
        self._period_vline.hide()
        box1 = self._boxed("1) DM-time (one period) — click on pulse", self.dm_period)

        # Block 2
        self.dm_segments = pg.ImageView()
        self._simplify_imageview(self.dm_segments)
        self.dm_segments.ui.roiBtn.hide()
        self.dm_segments.ui.menuBtn.hide()
        self._tune_imageview(self.dm_segments, invert_y=True)
        self._seg_v1 = pg.InfiniteLine(pos=256, angle=90, movable=False)
        self._seg_v2 = pg.InfiniteLine(pos=512, angle=90, movable=False)
        self._seg_click = pg.InfiniteLine(angle=90, movable=False)
        v = self.dm_segments.getView()
        v.addItem(self._seg_v1)
        v.addItem(self._seg_v2)
        v.addItem(self._seg_click)
        self._seg_click.hide()
        box2 = self._boxed("2) 3 DM segments (center ±1)", self.dm_segments)

        # Block 3 (checkboxes, because segments are independent 0/1)
        self.cb_left = QCheckBox("Left (seg-1)")
        self.cb_center = QCheckBox("Center (seg)")
        self.cb_right = QCheckBox("Right (seg+1)")
        self.cb_center.setChecked(True)
        self.cb_center.setEnabled(False)  # center always active (per spec)
        radios = QWidget()
        v3 = QVBoxLayout(radios)
        v3.addWidget(self.cb_left)
        v3.addWidget(self.cb_center)
        v3.addWidget(self.cb_right)
        v3.addStretch(1)
        box3 = self._boxed("3) Segment labels (0/1)", radios)

        # Block 4
        self.fb_view = pg.ImageView()
        self._simplify_imageview(self.fb_view)
        self.fb_view.ui.roiBtn.hide()
        self.fb_view.ui.menuBtn.hide()
        self._tune_imageview(self.fb_view)
        self.profile_plot = pg.PlotWidget()
        self.profile_plot.showGrid(x=True, y=True)
        self.profile_plot.getPlotItem().getViewBox().setDefaultPadding(0.0)
        fb = QWidget()
        vv = QVBoxLayout(fb)
        vv.addWidget(self.fb_view, 3)
        vv.addWidget(self.profile_plot, 1)
        box4 = self._boxed("4) Dedispersed filterbank + profile (3*SEG)", fb)

        # Block 5 info
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        box5 = self._boxed("5) Info", self.info)

        # Block 6 loaders
        self.btn_load_fil = QPushButton("Load .fil")
        self.btn_load_dm = QPushButton("Load DM-time .npy")
        self.btn_load_mjd = QPushButton("Load MJD table")
        self.btn_labels_csv = QPushButton("Select labels CSV (resume)")

        self.st_fil = QLabel("✗")
        self.st_dm = QLabel("✗")
        self.st_mjd = QLabel("✗")
        self.st_lbl = QLabel("✗")

        self.path_fil = QLabel("")
        self.path_dm = QLabel("")
        self.path_mjd = QLabel("")
        self.path_lbl = QLabel("")

        for lab in (self.st_fil, self.st_dm, self.st_mjd, self.st_lbl):
            lab.setFixedWidth(20)

        status = QWidget()
        s = QGridLayout(status)
        s.addWidget(QLabel(".fil:"), 0, 0); s.addWidget(self.st_fil, 0, 1); s.addWidget(self.path_fil, 0, 2)
        s.addWidget(QLabel("DM .npy:"), 1, 0); s.addWidget(self.st_dm, 1, 1); s.addWidget(self.path_dm, 1, 2)
        s.addWidget(QLabel("MJD table:"), 2, 0); s.addWidget(self.st_mjd, 2, 1); s.addWidget(self.path_mjd, 2, 2)
        s.addWidget(QLabel("labels.csv:"), 3, 0); s.addWidget(self.st_lbl, 3, 1); s.addWidget(self.path_lbl, 3, 2)

        loaders = QWidget()
        l = QVBoxLayout(loaders)
        l.addWidget(status)
        l.addWidget(self.btn_load_fil)
        l.addWidget(self.btn_load_dm)
        l.addWidget(self.btn_load_mjd)
        l.addWidget(self.btn_labels_csv)
        l.addStretch(1)
        box6 = self._boxed("6) Data loading", loaders)

        # Block 7 params
        self.cb_auto = QCheckBox("Using auto labeling")
        self.edge_pct = QSpinBox()
        self.edge_pct.setRange(0, 50)
        self.edge_pct.setValue(10)
        self.edge_pct.setSuffix(" %")

        self.period_edit = QLineEdit("1.0")
        self.period_edit.setValidator(QDoubleValidator(0.0, 1e9, 12))
        self.period_unit = QComboBox()
        self.period_unit.addItems(["s", "ms", "us"])

        self.dm_edit = QLineEdit("57.0")
        self.dm_edit.setValidator(QDoubleValidator(0.0, 1e6, 6))

        self.cb_split = QCheckBox("Split the period (P/2)")
        params = QWidget()
        p = QGridLayout(params)
        p.addWidget(self.cb_auto, 0, 0, 1, 2)
        p.addWidget(QLabel("Edge threshold:"), 1, 0)
        p.addWidget(self.edge_pct, 1, 1)
        p.addWidget(QLabel("Pulsar period:"), 2, 0)
        p.addWidget(self.period_edit, 2, 1)
        p.addWidget(self.period_unit, 2, 2)
        p.addWidget(QLabel("DM:"), 3, 0)
        p.addWidget(self.dm_edit, 3, 1)
        p.addWidget(self.cb_split, 4, 0, 1, 2)
        box7 = self._boxed("7) Parameters", params)

        # Block 8 jump seg_id
        self.seg_spin = QSpinBox()
        self.seg_spin.setRange(0, 2_000_000_000)
        self.btn_jump = QPushButton("Jump")
        jump = QWidget()
        jj = QHBoxLayout(jump)
        jj.addWidget(QLabel("seg_id:"))
        jj.addWidget(self.seg_spin, 1)
        jj.addWidget(self.btn_jump)
        box8 = self._boxed("8) Jump to segment index", jump)

        # Block 9 (table later) + comment
        self.comment_edit = QLineEdit()
        self.comment_edit.setPlaceholderText("Comment for current seg_center (Enter saves, stays on row)")
        tbl = QWidget()
        t = QVBoxLayout(tbl)
        t.addWidget(QLabel("Labels table: TODO (will be QTableView)"))
        t.addWidget(self.comment_edit)
        box9 = self._boxed("9) Labels & comment", tbl)

        # Block 10 nav + label
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

        # Block 11 saving
        self.cb_autosave = QCheckBox("Autosave after each label")
        self.cb_autosave.setChecked(True)
        box11 = self._boxed("11) Saving", self.cb_autosave)

        # Block 12 mode
        self.mode = QComboBox()
        self.mode.addItems(["Viewing", "Labeling"])
        box12 = self._boxed("12) Mode", self.mode)

        # layout
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
        self.resize(1500, 980)

        # initial disable until minimal ready
        self._set_controls_enabled(False)
        self._apply_plasma()

    def _apply_plasma(self):
        # Plasma colormap for all images
        cmap = None
        try:
            cmap = pg.colormap.get("plasma", source="matplotlib")
        except Exception:
            try:
                cmap = pg.colormap.get("plasma")
            except Exception:
                cmap = None

        if cmap is None:
            return

        lut = cmap.getLookupTable(nPts=256, alpha=False)
        self.dm_period.getImageItem().setLookupTable(lut)
        self.dm_segments.getImageItem().setLookupTable(lut)
        self.fb_view.getImageItem().setLookupTable(lut)

    def _boxed(self, title: str, widget):
        box = QGroupBox(title)
        lay = QVBoxLayout(box)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        lay.addWidget(widget)
        box.setSizePolicy(widget.sizePolicy())
        return box

    def _wire(self):
        # loaders
        self.btn_load_dm.clicked.connect(self._on_load_dm)
        self.btn_load_fil.clicked.connect(self._on_load_fil)
        self.btn_load_mjd.clicked.connect(self._on_load_mjd)
        self.btn_labels_csv.clicked.connect(self._on_select_labels_csv)

        # params live-update
        self.period_edit.editingFinished.connect(self._on_params_changed)
        self.period_unit.currentTextChanged.connect(self._on_params_changed)
        self.dm_edit.editingFinished.connect(self._on_params_changed)

        # split lock after start: we’ll lock it once pulses loaded
        self.cb_split.stateChanged.connect(self._on_split_changed)

        # navigation
        self.btn_prev.clicked.connect(self._prev_row)
        self.btn_next.clicked.connect(self._next_row)

        # labeling
        self.btn_yes.clicked.connect(self._label_yes)
        self.btn_no.clicked.connect(self._label_no)

        # mode
        self.mode.currentTextChanged.connect(self._apply_mode)

        # jump
        self.btn_jump.clicked.connect(self._jump_to_seg)

        # comment
        self.comment_edit.returnPressed.connect(self._save_comment)
        self.comment_edit.editingFinished.connect(self._save_comment)

        # click on period view
        self.dm_period.getView().scene().sigMouseClicked.connect(self._on_period_click)

        # Hotkeys that work regardless of focus
        act_yes = QAction(self)
        act_yes.setShortcut(QKeySequence(Qt.Key_Y))
        act_yes.setShortcutContext(Qt.ApplicationShortcut)
        act_yes.triggered.connect(self._label_yes)
        self.addAction(act_yes)

        act_no = QAction(self)
        act_no.setShortcut(QKeySequence(Qt.Key_N))
        act_no.setShortcutContext(Qt.ApplicationShortcut)
        act_no.triggered.connect(self._label_no)
        self.addAction(act_no)

        act_prev = QAction(self)
        act_prev.setShortcut(QKeySequence(Qt.Key_Left))
        act_prev.setShortcutContext(Qt.ApplicationShortcut)
        act_prev.triggered.connect(self._prev_row)
        self.addAction(act_prev)

        act_next = QAction(self)
        act_next.setShortcut(QKeySequence(Qt.Key_Right))
        act_next.setShortcutContext(Qt.ApplicationShortcut)
        act_next.triggered.connect(self._next_row)
        self.addAction(act_next)

    # ---------------- helpers ----------------
    def _set_controls_enabled(self, enabled: bool) -> None:
        # viewing is still allowed to click & navigate, but only after minimal data ready
        self.btn_prev.setEnabled(enabled)
        self.btn_next.setEnabled(enabled)
        self.btn_jump.setEnabled(enabled)
        self.seg_spin.setEnabled(enabled)

        # label controls depend on mode + click
        self.btn_yes.setEnabled(False)
        self.btn_no.setEnabled(False)

        # params should be editable (period, dm) once fil loaded (tsamp needed) and pulses exist
        self.period_edit.setEnabled(enabled)
        self.period_unit.setEnabled(enabled)
        self.dm_edit.setEnabled(enabled)
        self.cb_auto.setEnabled(enabled)
        self.edge_pct.setEnabled(enabled)

    def _apply_mode(self):
        is_labeling = (self.mode.currentText() == "Labeling")
        # in Viewing you can inspect but cannot save
        if not is_labeling:
            self.btn_yes.setEnabled(False)
            self.btn_no.setEnabled(False)

    def _update_info(self, extra: str = ""):
        lines = []
        lines.append(f"mode: {self.mode.currentText()}")
        lines.append(f"period_sec: {self.mgr.period_sec}")
        lines.append(f"dm: {self.mgr.dm}")
        lines.append(f"split: {self.mgr.split}")
        if self.mgr.edmt is not None:
            lines.append(f"edmt: shape={self.mgr.edmt.shape}")
        if self.mgr.fil is not None:
            hdr = self.mgr.fil.your_header
            lines.append(f"fil: tsamp={hdr.tsamp} s  nchans={hdr.nchans}  fch1={hdr.fch1} MHz  bw={hdr.bw} MHz")
        if self.mgr.df is not None:
            lines.append(f"pulses rows: {len(self.mgr.df)}  current_row_idx={self.current_row_idx}")
            try:
                r = self.mgr.df.iloc[self.current_row_idx]
                lines.append(f"row mjd={r['mjd']}  phase={r['phase']}  snr={r['snr']}  fname={r['fname']}")
                start_pos = self.mgr.start_pos_for_row(self.current_row_idx) if self.mgr.fil is not None else None
                lines.append(f"start_pos={start_pos}")
            except Exception:
                pass
        lines.append(f"clicked_global_x={self.clicked_global_x}  center_seg={self.center_seg}")
        if self.mgr.labels_path is not None:
            lines.append(f"labels_csv: {self.mgr.labels_path}")
        if extra:
            lines.append("")
            lines.append(extra)
        self.info.setText("\n".join(lines))

    def _show_message(self, title: str, text: str):
        QMessageBox.information(self, title, text)

    def _show_message(self, title: str, text: str):
        QMessageBox.information(self, title, text)

    def _set_ok(self, lab: QLabel, ok: bool) -> None:
        lab.setText("✓" if ok else "✗")
        lab.setStyleSheet("color: #3fb950;" if ok else "color: #f85149;")

    # ---------------- load actions ----------------
    def _on_load_dm(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select DM-time .npy", "", "NumPy (*.npy)")
        if not path:
            return
        try:
            self.mgr.load_edmt(path)
        except Exception as e:
            QMessageBox.critical(self, "Load DM-time failed", str(e))
            self.mgr.edmt = None
            self._set_ok(self.st_dm, False)
            self.path_dm.setText("")
            self.path_dm.setToolTip("")
            return

        self._set_ok(self.st_dm, True)
        self.path_dm.setText(path)
        self.path_dm.setToolTip(path)
        self._update_info("Loaded DM-time dataset.")
        self._try_activate()

    def _on_load_fil(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select filterbank .fil", "", "Filterbank (*.fil)")
        if not path:
            return
        try:
            self.mgr.load_fil(path)
        except Exception as e:
            QMessageBox.critical(self, "Load .fil failed", str(e))
            self.mgr.fil = None
            self._set_ok(self.st_fil, False)
            self.path_fil.setText("")
            self.path_fil.setToolTip("")
            return

        self._set_ok(self.st_fil, True)
        self.path_fil.setText(path)
        self.path_fil.setToolTip(path)
        self._update_info("Loaded filterbank file.")
        self._try_activate()

    def _on_load_mjd(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select MJD pulse table", "", "Text/CSV (*)")
        if not path:
            return

        try:
            # apply current params before load (so split has correct P)
            self._on_params_changed(silent=True)
            self.mgr.load_pulses(path)
        except Exception as e:
            QMessageBox.critical(self, "Load MJD table failed", str(e))
            self.mgr.df_base = None
            self.mgr.df = None
            self._set_ok(self.st_mjd, False)
            self.path_mjd.setText("")
            self.path_mjd.setToolTip("")
            return

        # lock split once pulse table is loaded (per your rule)
        self.cb_split.setEnabled(False)

        self._set_ok(self.st_mjd, True)
        self.path_mjd.setText(path)
        self.path_mjd.setToolTip(path)

        self.current_row_idx = 0
        self._update_info("Loaded MJD pulse table.")
        self._try_activate()

    def _on_select_labels_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Select labels CSV (existing or new)", "", "CSV (*.csv)")
        if not path:
            return

        try:
            self.mgr.labels_path = Path(path)
            self.labels = LabelStore(path)
        except Exception as e:
            QMessageBox.critical(self, "Labels CSV failed", str(e))
            self.mgr.labels_path = None
            self.labels = None
            self._set_ok(self.st_lbl, False)
            self.path_lbl.setText("")
            self.path_lbl.setToolTip("")
            return

        self._set_ok(self.st_lbl, True)
        self.path_lbl.setText(path)
        self.path_lbl.setToolTip(path)

        self._update_info("Labels CSV selected. Resume will use seg_center = start_pos//SEG heuristic.")

        # resume: jump to first row whose expected seg_center is not in CSV
        if self.mgr.df is not None and self.mgr.fil is not None and self.labels is not None:
            for i in range(len(self.mgr.df)):
                seg = self.mgr.expected_seg_center_for_row(i)
                if not self.labels.is_labeled(seg):
                    self.current_row_idx = i
                    break
            self._load_current_row()


    def _on_split_changed(self):
        self.mgr.set_split(self.cb_split.isChecked())

    def _on_params_changed(self, silent: bool = False):
        try:
            p = float(self.period_edit.text())
        except Exception:
            p = self.mgr.period_sec
        unit = self.period_unit.currentText()
        self.mgr.set_period(p, unit)

        try:
            dm = float(self.dm_edit.text())
        except Exception:
            dm = self.mgr.dm
        self.mgr.set_dm(dm)

        if not silent:
            self._update_info("Params updated (period/DM). If data ready, window will redraw.")
        if self.mgr.ready_minimal():
            self._load_current_row()

        # note: split is locked after loading pulses; period changes still ok and will rebuild P/2 rows if split enabled.

    def _try_activate(self):
        if self.mgr.ready_minimal():
            self._set_controls_enabled(True)
            self._load_current_row()
        else:
            # not ready yet
            self._set_controls_enabled(False)

    # ---------------- row navigation ----------------
    def _load_current_row(self):
        # reset click
        self.clicked_global_x = None
        self.clicked_local_x = None
        self.center_seg = None
        self._period_vline.hide()
        self._seg_click.hide()
        self.cb_left.setChecked(False)
        self.cb_right.setChecked(False)

        if not self.mgr.ready_minimal():
            return
        if self.mgr.df is None:
            return
        self.current_row_idx = max(0, min(self.current_row_idx, len(self.mgr.df) - 1))

        rw = self.mgr.render_period_window(self.current_row_idx, max_cols=2048)
        self._period_x0_global = rw.x0_global
        self._period_stride = rw.stride
        self._period_width_real = rw.width_real

        lo, hi = levels_by_percentile(rw.image)
        self.dm_period.setImage(rw.image, autoLevels=False, levels=(lo, hi))
        if self._lut is not None:
            self.dm_period.getImageItem().setLookupTable(self._lut)

        self._update_info("Rendered period DM-time window (block 1). Click to show segments.")

        # in labeling mode, save buttons are disabled until click
        self.btn_yes.setEnabled(False)
        self.btn_no.setEnabled(False)
        self.comment_edit.setText("")
        self._pending_comment = ""

    def _prev_row(self):
        if self.mgr.df is None:
            return
        self.current_row_idx = max(0, self.current_row_idx - 1)
        self._load_current_row()

    def _next_row(self):
        if self.mgr.df is None:
            return
        self.current_row_idx = min(len(self.mgr.df) - 1, self.current_row_idx + 1)
        self._load_current_row()

    # ---------------- click handling ----------------
    def _on_period_click(self, ev):
        if not self.mgr.ready_minimal():
            return

        # only accept left click
        if ev.button() != Qt.LeftButton:
            return

        vb = self.dm_period.getView()
        pos = ev.scenePos()
        p = vb.mapSceneToView(pos)

        # p.x is in display pixels (downsampled)
        x_display = int(round(p.x()))
        x_display = max(0, x_display)

        # map display x -> real sample
        local_real = x_display * self._period_stride
        local_real = min(local_real, max(0, self._period_width_real - 1))

        global_x = self._period_x0_global + local_real

        self.clicked_global_x = int(global_x)
        self.clicked_local_x = int(local_real)
        self.center_seg = int(self.clicked_global_x // self.mgr.SEG)

        # show vline at display x
        self._period_vline.setPos(x_display)
        self._period_vline.show()

        # update segments + fb
        self._render_segments_and_fb()

        # enable Y/N only in labeling mode
        if self.mode.currentText() == "Labeling":
            self.btn_yes.setEnabled(True)
            self.btn_no.setEnabled(True)

        self._update_info("Click registered → rendered segments + filterbank.")

    def _render_segments_and_fb(self):
        if self.center_seg is None or self.clicked_global_x is None:
            return

        # block 2: 3 segments
        img = self.mgr.render_segments_triplet(self.center_seg)
        lo, hi = levels_by_percentile(img)
        self.dm_segments.setImage(img, autoLevels=False, levels=(lo, hi))
        if self._lut is not None:
            self.dm_segments.getImageItem().setLookupTable(self._lut)

        # click position inside 3*SEG cut:
        cut_start = (self.center_seg - 1) * self.mgr.SEG
        click_in_cut = self.clicked_global_x - cut_start
        self._seg_click.setPos(float(click_in_cut))
        self._seg_click.show()

        # block 3: reset + auto-labeling
        self.cb_left.setChecked(False)
        self.cb_right.setChecked(False)

        if self.cb_auto.isChecked():
            thr = int(round(self.mgr.SEG * (self.edge_pct.value() / 100.0)))
            offset_in_center = self.clicked_global_x - (self.center_seg * self.mgr.SEG)
            if offset_in_center <= thr:
                self.cb_left.setChecked(True)
            if offset_in_center >= (self.mgr.SEG - 1 - thr):
                self.cb_right.setChecked(True)

        # block 4: filterbank + dedisp + profile
        raw_f_t, dd_f_t, prof = self.mgr.read_filterbank_triplet(self.center_seg)

        freq_mhz = self.mgr.channel_freq_mhz()
        flip_freq = freq_mhz[0] < freq_mhz[-1]
        dd_img = np.flipud(dd_f_t) if flip_freq else dd_f_t
        
        lo2, hi2 = levels_by_percentile(dd_img)
        self.fb_view.setImage(dd_img, autoLevels=False, levels=(lo2, hi2))

        self.profile_plot.clear()
        self.profile_plot.plot(prof)

    # ---------------- saving ----------------
    def _ensure_labels_store(self) -> bool:
        if self.labels is not None and self.mgr.labels_path is not None:
            return True
        self._show_message("Labels CSV", "Select labels CSV first (button in block 6).")
        return False

    def _make_label_row(self, force_all_zero: bool) -> LabelRow:
        assert self.mgr.df is not None
        assert self.center_seg is not None
        assert self.clicked_global_x is not None

        r = self.mgr.df.iloc[self.current_row_idx]
        start_pos = self.mgr.start_pos_for_row(self.current_row_idx)

        seg_left = int(self.center_seg - 1)
        seg_center = int(self.center_seg)
        seg_right = int(self.center_seg + 1)

        if force_all_zero:
            l_left = l_center = l_right = 0
        else:
            l_left = 1 if self.cb_left.isChecked() else 0
            l_center = 1  # center always active
            l_right = 1 if self.cb_right.isChecked() else 0

        return LabelRow(
            row_idx=int(self.current_row_idx),
            name_of_set="",  # can be extended later
            mjd=float(r["mjd"]),
            phase=float(r["phase"]) if not np.isnan(r["phase"]) else None,
            fname=str(r["fname"]),
            snr=float(r["snr"]) if not np.isnan(r["snr"]) else None,
            start_pos=int(start_pos),
            clicked_local_x=int(self.clicked_local_x) if self.clicked_local_x is not None else None,
            clicked_global_x=int(self.clicked_global_x),
            seg_left=seg_left,
            seg_center=seg_center,
            seg_right=seg_right,
            label_left=int(l_left),
            label_center=int(l_center),
            label_right=int(l_right),
            comment=str(self._pending_comment or ""),
        )

    def _label_yes(self):
        if self.mode.currentText() != "Labeling":
            return
        if self.center_seg is None or self.clicked_global_x is None:
            return
        if not self._ensure_labels_store():
            return

        row = self._make_label_row(force_all_zero=False)
        assert self.labels is not None
        self.labels.upsert(row)
        self._pending_comment = ""
        self.comment_edit.setText("")
        self._next_row()

    def _label_no(self):
        if self.mode.currentText() != "Labeling":
            return
        if self.center_seg is None or self.clicked_global_x is None:
            return
        if not self._ensure_labels_store():
            return

        row = self._make_label_row(force_all_zero=True)
        assert self.labels is not None
        self.labels.upsert(row)
        self._pending_comment = ""
        self.comment_edit.setText("")
        self._next_row()

    def _save_comment(self):
        txt = self.comment_edit.text().strip()
        self._pending_comment = txt

        # If already labeled (seg_center exists), update immediately and stay on row
        if self.labels is None or self.center_seg is None:
            return
        if not self._ensure_labels_store():
            return
        if self.labels.is_labeled(int(self.center_seg)):
            # update by writing a new row with same labels as current UI (or zeros if none)
            # requires click to know seg_center reliably
            if self.clicked_global_x is None:
                return
            row = self._make_label_row(force_all_zero=False)
            self.labels.upsert(row)
            self._update_info("Comment saved (upsert). Staying on the same row.")
        else:
            self._update_info("Comment staged. Press Y/N to save labels (comment will be included).")

    # ---------------- jump ----------------
    def _jump_to_seg(self):
        if self.mgr.df is None or self.mgr.fil is None:
            return
        seg_id = int(self.seg_spin.value())
        target_global_x = seg_id * self.mgr.SEG + self.mgr.SEG // 2
        win = self.mgr.window_samples()

        best_idx = 0
        best_dist = None
        for i in range(len(self.mgr.df)):
            start = self.mgr.start_pos_for_row(i)
            if start <= target_global_x < start + win:
                best_idx = i
                best_dist = 0
                break
            dist = abs(start - target_global_x)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i

        self.current_row_idx = best_idx
        self._load_current_row()
