import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(os.getcwd())

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QApplication, QGraphicsScene

from view_utils import ZoomableGraphicsView


def test_zoomable_graphics_view_zoom():
    """Wheel events should scale the view's transformation matrix."""

    app = QApplication.instance() or QApplication([])
    scene = QGraphicsScene()
    view = ZoomableGraphicsView(scene)
    view.show()  # required so the viewport exists
    initial = view.transform().m11()

    event = QWheelEvent(
        QPointF(),
        QPointF(),
        QPoint(),
        QPoint(0, 120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.ScrollUpdate,
        False,
    )
    view.wheelEvent(event)

    assert view.transform().m11() > initial

