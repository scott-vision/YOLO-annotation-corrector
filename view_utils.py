"""Utility classes for Qt graphics views.

This module currently provides :class:`ZoomableGraphicsView`, a thin wrapper
around :class:`PyQt6.QtWidgets.QGraphicsView` that adds convenient mouse-wheel
zooming behaviour.  The view is used throughout the application to display the
annotated images.
"""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QGraphicsView


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view supporting zooming and panning.

    The view performs a simple scaling transformation when the user rotates
    the mouse wheel.  Panning is implemented using the right mouse button so
    that left-click interactions on scene items remain available.
    The transformation anchor is set so that zooming is centred on the cursor
    position, which provides an intuitive user experience.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Start with no drag mode so items can receive mouse events.
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._last_pan_point: Optional[QPoint] = None
        # Ensure zooming occurs relative to the cursor position.
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        # Disable the context menu so right-click can be used for panning.
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        """Scale the view matrix in response to a wheel event."""

        # Positive delta values indicate that the wheel was rotated away from
        # the user (zoom in); negative values mean zoom out.
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self._last_pan_point = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # type: ignore[override]
        if self._last_pan_point is not None:
            delta = event.pos() - self._last_pan_point
            self._last_pan_point = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.RightButton:
            self._last_pan_point = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

