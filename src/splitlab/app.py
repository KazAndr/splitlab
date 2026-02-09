from __future__ import annotations

import sys
from PyQt5.QtWidgets import QApplication
from .ui.main_window import MainWindow

APP_TITLE = "SPLIT — Simple Pulse Labeling InTerface"

def run() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SPLIT")
    w = MainWindow()
    w.setWindowTitle(APP_TITLE)
    w.show()
    return app.exec_()
