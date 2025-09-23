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

from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import QGraphicsRectItem, QGraphicsTextItem, QMenu

if TYPE_CHECKING:  # pragma: no cover - used only for type hints
    from window import AnnotationWindow


def rect_to_yolo_line(rect: QRectF, cls_id: int, img_w: int, img_h: int) -> str:
    """Convert a ``QRectF`` to a YOLO-format label line."""

    xc = (rect.left() + rect.width() / 2) / img_w
    yc = (rect.top() + rect.height() / 2) / img_h
    w = rect.width() / img_w
    h = rect.height() / img_h
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


ICON_TEMPLATE = (
    "<div style=\"display:inline-block;color:{color};cursor:pointer;"
    "background-color:white;border:1px solid {color};"
    "font-weight:bold;width:18px;height:18px;line-height:18px;"
    "text-align:center;border-radius:3px;\">{symbol}</div>"
)


class ToggleIcon(QGraphicsTextItem):
    """Clickable icon that delegates the mouse press to a callback."""

    def __init__(self, parent, callback) -> None:
        super().__init__(parent)
        self._callback = callback
        # Only need to react to left clicks
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._callback()
        # Propagate the event in case other handlers rely on it
        super().mousePressEvent(event)


class ClickableTextItem(QGraphicsTextItem):
    """Label that invokes a callback when clicked."""

    def __init__(self, callback, parent=None) -> None:
        super().__init__(parent)
        self._callback = callback
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._callback(event)
            event.accept()
            return
        super().mousePressEvent(event)


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
        self.class_names = class_names

        self.label = QGraphicsTextItem(self)
        self._update_label()

        # A clickable + / - icon allowing users to toggle acceptance even when
        # the icon sits outside of the rectangle's bounds.
        self.icon = ToggleIcon(self, self._toggle)
        self._update_icon()
        self._position_items()

    def _update_icon(self) -> None:
        """Display a '+' to add or a '-' to remove a prediction."""

        if self.accepted:
            symbol = "-"
            color = "red"
        else:
            symbol = "+"
            color = "green"
        self.icon.setHtml(ICON_TEMPLATE.format(symbol=symbol, color=color))
        self._position_items()

    def _position_items(self) -> None:
        """Position the label and icon together beneath the box."""

        rect = self.rect()
        margin = 2
        label_x = rect.left() + margin
        label_y = rect.bottom() + margin
        self.label.setPos(label_x, label_y)

        label_width = self.label.boundingRect().width()
        icon_x = label_x + label_width + margin
        self.icon.setPos(icon_x, label_y)

    def _class_name(self) -> str:
        try:
            cls_id = int(self.line.split()[0])
        except (ValueError, IndexError):
            return "?"
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return str(cls_id)

    def _update_label(self) -> None:
        cls_name = self._class_name()
        self.label.setHtml(
            f"<div style='background-color:white;'>{cls_name}:{self.conf:.2f}</div>"
        )

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
        self._update_label()
        self._position_items()
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()
        self.window.flag_predictions()

    def set_class(self, cls_id: int) -> None:
        """Update the class identifier for the prediction."""

        parts = self.line.split()
        if len(parts) != 5:
            return
        parts[0] = str(cls_id)
        self.line = " ".join(parts)
        self.state["line"] = self.line
        self._update_label()
        self._position_items()
        if self.window.final_checkbox.isChecked() and not getattr(
            self.window, "_suspend_final_refresh", False
        ):
            self.window.update_final_items()
        self.window.flag_predictions()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def _toggle(self) -> None:
        """Toggle the accepted state and refresh related UI elements."""

        self.accepted = not self.accepted
        self.state["accepted"] = self.accepted
        self._update_icon()
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()
        self.window.flag_predictions()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        """Toggle acceptance or start resizing on left click."""

        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        if self._start_resize(event):
            QGraphicsRectItem.mousePressEvent(self, event)
            return
        # Toggle accepted state when clicked.
        self._toggle()
        super().mousePressEvent(event)

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
        self.class_names = class_names

        self.label = QGraphicsTextItem(self)
        self._update_label()

        # Clickable indicator mirroring the behaviour of PredBox
        self.icon = ToggleIcon(self, self._toggle)
        self._update_icon()
        self._position_items()

    def _update_icon(self) -> None:
        """Display a '-' to remove or a '+' to add the annotation."""

        if self.kept:
            symbol = "-"
            color = "red"
        else:
            symbol = "+"
            color = "green"
        self.icon.setHtml(ICON_TEMPLATE.format(symbol=symbol, color=color))
        self._position_items()

    def _position_items(self) -> None:
        """Place the label and icon near the top-left of the bounding box."""

        rect = self.rect()
        margin = 2
        label_x = rect.left() + margin
        label_height = self.label.boundingRect().height()
        label_y = rect.top() - label_height - margin
        self.label.setPos(label_x, label_y)

        label_width = self.label.boundingRect().width()
        icon_x = label_x + label_width + margin
        self.icon.setPos(icon_x, label_y)

    def _class_name(self) -> str:
        try:
            cls_id = int(self.line.split()[0])
        except (ValueError, IndexError):
            return "?"
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return str(cls_id)

    def _update_label(self) -> None:
        cls_name = self._class_name()
        self.label.setHtml(f"<div style='background-color:white;'>{cls_name}</div>")

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
        self._update_label()
        self._position_items()
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()
        self.window.flag_predictions()

    def set_class(self, cls_id: int) -> None:
        """Update the class identifier for this ground-truth annotation."""

        parts = self.line.split()
        if len(parts) != 5:
            return
        parts[0] = str(cls_id)
        self.line = " ".join(parts)
        self.state["line"] = self.line
        self._update_label()
        self._position_items()
        if self.window.final_checkbox.isChecked() and not getattr(
            self.window, "_suspend_final_refresh", False
        ):
            self.window.update_final_items()
        self.window.flag_predictions()

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def _toggle(self) -> None:
        """Toggle whether the annotation is kept and refresh UI."""

        self.kept = not self.kept
        self.state["kept"] = self.kept
        self._update_icon()
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()
        self.window.flag_predictions()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        """Toggle whether the ground truth annotation is kept on left click."""

        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        if self._start_resize(event):
            QGraphicsRectItem.mousePressEvent(self, event)
            return
        self._toggle()
        super().mousePressEvent(event)

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


class FinalBoxOverlay(QGraphicsRectItem):
    """Interactive overlay for final labels that proxies changes to a source box."""

    HANDLE_SIZE = 10

    def __init__(
        self,
        source: PredBox | GTBox,
        class_names: List[str],
        window: "AnnotationWindow",
    ) -> None:
        super().__init__(source.rect())
        self.source = source
        self.window = window
        self.class_names = class_names
        self._resizing: Optional[str] = None
        self._dragging = False
        self._drag_offset = QPointF()

        self.setPen(QPen(QColor("blue"), 2))
        self.setZValue(source.zValue() + 1)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        self.label = ClickableTextItem(self._show_class_menu, self)
        self._update_label()

    def _class_name(self) -> str:
        cls_id = self._class_id()
        if cls_id is None:
            return "?"
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return str(cls_id)

    def _class_id(self) -> Optional[int]:
        try:
            return int(self.source.line.split()[0])
        except (ValueError, IndexError):
            return None

    def _update_label(self) -> None:
        self.label.setHtml(
            f"<div style='color:blue;background-color:white;'>{self._class_name()}</div>"
        )
        rect = self.rect()
        margin = 2
        label_height = self.label.boundingRect().height()
        self.label.setPos(rect.left(), rect.top() - label_height - margin)

    def _show_class_menu(self, event) -> None:
        menu = QMenu()
        current_cls = self._class_id()
        for idx, name in enumerate(self.class_names):
            action = menu.addAction(name)
            action.setData(idx)
            action.setCheckable(True)
            action.setChecked(idx == current_cls)
        selected = menu.exec(event.screenPos().toPoint())
        if selected is None:
            return
        data = selected.data()
        if data is None:
            return
        self._set_class(int(data))

    def _set_class(self, cls_id: int) -> None:
        if self._class_id() == cls_id:
            return
        self.window._suspend_final_refresh = True
        try:
            self.source.set_class(cls_id)
        finally:
            self.window._suspend_final_refresh = False
        self._update_label()
        self.window.update_final_items()

    # ------------------------------------------------------------------
    # Resizing helpers
    # ------------------------------------------------------------------
    def _start_resize(self, event) -> bool:
        r = self.rect()
        pos = event.pos()
        if abs(pos.x() - r.left()) <= self.HANDLE_SIZE and abs(pos.y() - r.top()) <= self.HANDLE_SIZE:
            self._resizing = "topleft"
        elif (
            abs(pos.x() - r.right()) <= self.HANDLE_SIZE
            and abs(pos.y() - r.bottom()) <= self.HANDLE_SIZE
        ):
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
        self._apply_to_source()
        self._update_label()

    def _apply_to_source(self) -> None:
        self.window._suspend_final_refresh = True
        try:
            self.source.setRect(self.rect())
            self.source._update_from_rect()
        finally:
            self.window._suspend_final_refresh = False

    # ------------------------------------------------------------------
    # Mouse events
    # ------------------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() != Qt.MouseButton.LeftButton:
            event.ignore()
            return
        if not self._start_resize(event):
            if self.rect().contains(event.pos()):
                self._dragging = True
                self._drag_offset = event.pos() - self.rect().topLeft()
                event.accept()
                return
            event.ignore()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._resizing:
            self._resize(event)
        elif self._dragging:
            rect = self.rect()
            rect.moveTopLeft(event.pos() - self._drag_offset)
            self.setRect(rect)
            self._apply_to_source()
            self._update_label()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._resizing:
            self._resize(event)
            self._resizing = None
            self.window.update_final_items()
        elif self._dragging:
            self._dragging = False
            self.window.update_final_items()
        super().mouseReleaseEvent(event)

