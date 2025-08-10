"""Utility classes for Qt graphics views.

This module currently provides :class:`ZoomableGraphicsView`, a thin wrapper
around :class:`PyQt6.QtWidgets.QGraphicsView` that adds convenient mouse-wheel
zooming behaviour.  The view is used throughout the application to display the
annotated images.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import QGraphicsView


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view supporting zooming and panning.

    The view enables scroll-hand dragging for panning and performs a simple
    scaling transformation when the user rotates the mouse wheel.  The
    transformation anchor is set so that zooming is centred on the cursor
    position, which provides an intuitive user experience.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Allow the user to pan the scene by dragging with the mouse.
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Ensure zooming occurs relative to the cursor position.
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        """Scale the view matrix in response to a wheel event."""

        # Positive delta values indicate that the wheel was rotated away from
        # the user (zoom in); negative values mean zoom out.
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

