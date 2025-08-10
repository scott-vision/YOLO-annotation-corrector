import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(os.getcwd())

from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QApplication, QCheckBox, QGraphicsScene
from PyQt6.QtTest import QTest

from graphics_items import GTBox, PredBox
from view_utils import ZoomableGraphicsView


class DummyWindow:
    """Minimal stub implementing callbacks expected by the items."""

    def __init__(self):
        self.final_checkbox = QCheckBox()

    def update_final_items(self):
        pass

    def flag_predictions(self):
        pass


def test_predbox_state_update():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = ZoomableGraphicsView(scene)
    win = DummyWindow()
    rect = QRectF(0, 0, 20, 20)
    state = {"line": "0 0.1 0.1 0.2 0.2", "conf": 0.9, "accepted": False}
    box = PredBox(rect, state, ["obj"], win, 100, 100)
    scene.addItem(box)
    view.show()
    QTest.qWait(10)  # exercise event loop
    box.accepted = True
    box._update_icon()
    assert box.accepted is True
    assert "✗" in box.icon.toHtml()


def test_gtbox_state_update():
    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = ZoomableGraphicsView(scene)
    win = DummyWindow()
    rect = QRectF(0, 0, 20, 20)
    state = {"line": "0 0.1 0.1 0.2 0.2", "kept": True}
    box = GTBox(rect, state, ["obj"], win, 100, 100)
    scene.addItem(box)
    view.show()
    QTest.qWait(10)
    box.kept = False
    box._update_icon()
    assert box.kept is False
    assert "✓" in box.icon.toHtml()

