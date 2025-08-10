"""Interactive graphics items representing bounding boxes.

This module provides two :class:`PyQt6.QtWidgets.QGraphicsRectItem` subclasses
used within the annotation tool:

``PredBox``
    Represents a predicted bounding box.  Clicking the box toggles whether the
    prediction should be accepted and saved.  Small handles in the top-left and
    bottom-right corners allow resizing.

``GTBox``
    Represents an existing ground-truth bounding box.  Clicking toggles whether
    the annotation is kept.  It shares the same resize behaviour as ``PredBox``.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from window import AnnotationWindow


def rect_to_yolo_line(rect: QRectF, cls_id: int, img_w: int, img_h: int) -> str:
    """Convert a ``QRectF`` to a YOLO-format label line."""

    xc = (rect.left() + rect.width() / 2) / img_w
    yc = (rect.top() + rect.height() / 2) / img_h
    w = rect.width() / img_w
    h = rect.height() / img_h
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


class PredBox(QGraphicsRectItem):
    """Graphics item for a predicted box."""

    HANDLE_SIZE = 10

    def __init__(
        self,
        rect: QRectF,
        state: dict,
        class_names: List[str],
        window: "AnnotationWindow",
        img_w: int,
        img_h: int,
    ) -> None:
        super().__init__(rect)
        self.window = window
        self.state = state
        self.line = state["line"]
        self.conf = state["conf"]
        self.accepted = state.get("accepted", False)
        self.setPen(QPen(QColor("red"), 2))
        self.img_w = img_w
        self.img_h = img_h
        self._resizing: Optional[str] = None

        cls_id = int(self.line.split()[0])
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        self.label = QGraphicsTextItem(self)
        self.label.setHtml(
            f"<div style='background-color:white;'>{cls_name}:{self.conf:.2f}</div>"
        )
        self.label.setPos(rect.left(), rect.top() - 20)

        self.tick = QGraphicsTextItem(self)
        self._update_tick()
        self.tick.setPos(rect.right() + 2, rect.top() - 20)

    def _update_tick(self) -> None:
        """Display a checkmark indicating whether the prediction is accepted."""

        color = "green" if self.accepted else "gray"
        self.tick.setHtml(f"<div style='color:{color};background-color:white;'>✓</div>")

    # ------------------------------------------------------------------
    # Resizing helpers
    # ------------------------------------------------------------------
    def _start_resize(self, event) -> bool:
        r = self.rect()
        pos = event.pos()
        if abs(pos.x() - r.left()) <= self.HANDLE_SIZE and abs(pos.y() - r.top()) <= self.HANDLE_SIZE:
            self._resizing = "topleft"
        elif abs(pos.x() - r.right()) <= self.HANDLE_SIZE and abs(pos.y() - r.bottom()) <= self.HANDLE_SIZE:
            self._resizing = "bottomright"
        else:
            self._resizing = None
        return self._resizing is not None

    def _resize(self, event) -> None:
        if not self._resizing:
            return
        r = self.rect()
        pos = event.pos()
        if self._resizing == "topleft":
            r.setTopLeft(pos)
        elif self._resizing == "bottomright":
            r.setBottomRight(pos)
        self.setRect(r)
        self._update_from_rect()

    def _update_from_rect(self) -> None:
        cls_id = int(self.line.split()[0])
        self.line = rect_to_yolo_line(self.rect(), cls_id, self.img_w, self.img_h)
        self.state["line"] = self.line
        self.label.setPos(self.rect().left(), self.rect().top() - 20)
        self.tick.setPos(self.rect().right() + 2, self.rect().top() - 20)
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()
        self.window.flag_predictions()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        """Toggle acceptance or start resizing on mouse press."""

        if self._start_resize(event):
            QGraphicsRectItem.mousePressEvent(self, event)
            return
        # Toggle accepted state when clicked.
        self.accepted = not self.accepted
        self.state["accepted"] = self.accepted
        self._update_tick()
        super().mousePressEvent(event)
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._resizing:
            self._resize(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._resizing:
            self._resize(event)
        self._resizing = None
        super().mouseReleaseEvent(event)


class GTBox(QGraphicsRectItem):
    """Graphics item for an existing ground truth box."""

    HANDLE_SIZE = 10

    def __init__(
        self,
        rect: QRectF,
        state: dict,
        class_names: List[str],
        window: "AnnotationWindow",
        img_w: int,
        img_h: int,
    ) -> None:
        super().__init__(rect)
        self.window = window
        self.state = state
        self.line = state["line"]
        self.kept = state.get("kept", True)
        self.setPen(QPen(QColor("green"), 2))
        self.img_w = img_w
        self.img_h = img_h
        self._resizing: Optional[str] = None

        cls_id = int(self.line.split()[0])
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        self.label = QGraphicsTextItem(self)
        self.label.setHtml(f"<div style='background-color:white;'>{cls_name}</div>")
        self.label.setPos(rect.left(), rect.top() - 20)

        self.cross = QGraphicsTextItem(self)
        self._update_cross()
        self.cross.setPos(rect.right() + 2, rect.top() - 20)

    def _update_cross(self) -> None:
        """Display a cross indicating whether the annotation is kept."""

        color = "red" if self.kept else "gray"
        self.cross.setHtml(f"<div style='color:{color};background-color:white;'>✗</div>")

    # ------------------------------------------------------------------
    # Resizing helpers
    # ------------------------------------------------------------------
    def _start_resize(self, event) -> bool:
        r = self.rect()
        pos = event.pos()
        if abs(pos.x() - r.left()) <= self.HANDLE_SIZE and abs(pos.y() - r.top()) <= self.HANDLE_SIZE:
            self._resizing = "topleft"
        elif abs(pos.x() - r.right()) <= self.HANDLE_SIZE and abs(pos.y() - r.bottom()) <= self.HANDLE_SIZE:
            self._resizing = "bottomright"
        else:
            self._resizing = None
        return self._resizing is not None

    def _resize(self, event) -> None:
        if not self._resizing:
            return
        r = self.rect()
        pos = event.pos()
        if self._resizing == "topleft":
            r.setTopLeft(pos)
        elif self._resizing == "bottomright":
            r.setBottomRight(pos)
        self.setRect(r)
        self._update_from_rect()

    def _update_from_rect(self) -> None:
        cls_id = int(self.line.split()[0])
        self.line = rect_to_yolo_line(self.rect(), cls_id, self.img_w, self.img_h)
        self.state["line"] = self.line
        self.label.setPos(self.rect().left(), self.rect().top() - 20)
        self.cross.setPos(self.rect().right() + 2, self.rect().top() - 20)
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()
        self.window.flag_predictions()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        """Toggle whether the ground truth annotation is kept."""

        if self._start_resize(event):
            QGraphicsRectItem.mousePressEvent(self, event)
            return
        self.kept = not self.kept
        self.state["kept"] = self.kept
        self._update_cross()
        super().mousePressEvent(event)
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()
        self.window.flag_predictions()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._resizing:
            self._resize(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._resizing:
            self._resize(event)
        self._resizing = None
        super().mouseReleaseEvent(event)

