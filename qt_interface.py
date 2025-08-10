from __future__ import annotations

import os
from typing import List, Optional

from PIL import ImageEnhance
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QImage, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class ZoomableGraphicsView(QGraphicsView):
    """Graphics view supporting zooming and panning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)


def yolo_line_to_rect(line: str, img_w: int, img_h: int) -> QRectF:
    """Convert a YOLO label line to a QRectF."""
    parts = line.split()
    if len(parts) != 5:
        return QRectF()
    _, xc, yc, w, h = parts
    xc, yc, w, h = map(float, (xc, yc, w, h))
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    return QRectF(x1, y1, w * img_w, h * img_h)


def rect_to_yolo_line(rect: QRectF, cls_id: int, img_w: int, img_h: int) -> str:
    """Convert a QRectF back to a YOLO label line."""
    xc = (rect.left() + rect.width() / 2) / img_w
    yc = (rect.top() + rect.height() / 2) / img_h
    w = rect.width() / img_w
    h = rect.height() / img_h
    return f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def iou(rect1: QRectF, rect2: QRectF) -> float:
    """Compute intersection-over-union of two rectangles."""
    inter = rect1.intersected(rect2)
    if inter.isNull():
        return 0.0
    inter_area = inter.width() * inter.height()
    union_area = rect1.width() * rect1.height() + rect2.width() * rect2.height() - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


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
    ):
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
        self.label.setHtml(f"<div style='background-color:white;'>{cls_name}:{self.conf:.2f}</div>")
        self.label.setPos(rect.left(), rect.top() - 20)

        self.tick = QGraphicsTextItem(self)
        self._update_tick()
        self.tick.setPos(rect.right() + 2, rect.top() - 20)

    def _update_tick(self):
        color = "green" if self.accepted else "gray"
        self.tick.setHtml(f"<div style='color:{color};background-color:white;'>✓</div>")

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

    def _resize(self, event):
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

    def _update_from_rect(self):
        cls_id = int(self.line.split()[0])
        self.line = rect_to_yolo_line(self.rect(), cls_id, self.img_w, self.img_h)
        self.state["line"] = self.line
        self.label.setPos(self.rect().left(), self.rect().top() - 20)
        self.tick.setPos(self.rect().right() + 2, self.rect().top() - 20)
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()
        self.window.flag_predictions()

    def mousePressEvent(self, event):
        if self._start_resize(event):
            QGraphicsRectItem.mousePressEvent(self, event)
            return
        self.accepted = not self.accepted
        self.state["accepted"] = self.accepted
        self._update_tick()
        super().mousePressEvent(event)
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()

    def mouseMoveEvent(self, event):
        if self._resizing:
            self._resize(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
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
    ):
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

    def _update_cross(self):
        color = "red" if self.kept else "gray"
        self.cross.setHtml(f"<div style='color:{color};background-color:white;'>✗</div>")

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

    def _resize(self, event):
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

    def _update_from_rect(self):
        cls_id = int(self.line.split()[0])
        self.line = rect_to_yolo_line(self.rect(), cls_id, self.img_w, self.img_h)
        self.state["line"] = self.line
        self.label.setPos(self.rect().left(), self.rect().top() - 20)
        self.cross.setPos(self.rect().right() + 2, self.rect().top() - 20)
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()
        self.window.flag_predictions()

    def mousePressEvent(self, event):
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

    def mouseMoveEvent(self, event):
        if self._resizing:
            self._resize(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resize(event)
        self._resizing = None
        super().mouseReleaseEvent(event)


class AnnotationWindow(QMainWindow):
    """Main window providing annotation controls with multi-image navigation."""

    def __init__(
        self,
        images: List,
        predictions: List[List[dict]],
        labels: List[List[dict]],
        label_files: List[str],
        class_names: List[str],
    ):
        super().__init__()
        self.setWindowTitle("YOLO Annotation Corrector")
        self.class_names = class_names
        self.images = images
        self.pred_states = predictions
        self.gt_states = labels
        self.label_files = label_files
        self.index = 0

        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene)

        self.pred_items: List[PredBox] = []
        self.gt_items: List[GTBox] = []
        self.final_items: List[QGraphicsRectItem] = []

        self.pred_checkbox = QCheckBox("Show predictions")
        self.pred_checkbox.setChecked(True)
        self.pred_checkbox.toggled.connect(self.toggle_predictions)

        self.gt_checkbox = QCheckBox("Show ground truth")
        self.gt_checkbox.setChecked(True)
        self.gt_checkbox.toggled.connect(self.toggle_gt)

        self.final_checkbox = QCheckBox("Show final labels")
        self.final_checkbox.setChecked(False)
        self.final_checkbox.toggled.connect(self.toggle_final)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_all)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_image)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_image)

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.update_image_display)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self.update_image_display)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.pred_checkbox)
        control_layout.addWidget(self.gt_checkbox)
        control_layout.addWidget(self.final_checkbox)
        control_layout.addWidget(self.preview_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.prev_btn)
        control_layout.addWidget(self.next_btn)
        control_layout.addWidget(self.exit_btn)
        controls = QWidget()
        controls.setLayout(control_layout)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("Brightness"))
        slider_layout.addWidget(self.brightness_slider)
        slider_layout.addWidget(QLabel("Contrast"))
        slider_layout.addWidget(self.contrast_slider)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addLayout(slider_layout)
        layout.addWidget(controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.load_image(0)

    # ------------------------------------------------------------------
    # Image and box management
    # ------------------------------------------------------------------
    def pil_to_pixmap(self, img):
        img = img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def adjust_image(self, img):
        b = self.brightness_slider.value() / 100.0
        c = self.contrast_slider.value() / 100.0
        out = img
        if b != 1.0:
            out = ImageEnhance.Brightness(out).enhance(b)
        if c != 1.0:
            out = ImageEnhance.Contrast(out).enhance(c)
        return out

    def load_image(self, index: int):
        self.scene.clear()
        self.pred_items = []
        self.gt_items = []
        self.final_items = []
        self.index = index

        img = self.images[index]
        pixmap = self.pil_to_pixmap(self.adjust_image(img))
        self.background_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.background_item)

        img_w = pixmap.width()
        img_h = pixmap.height()

        for state in self.pred_states[index]:
            rect = yolo_line_to_rect(state["line"], img_w, img_h)
            item = PredBox(rect, state, self.class_names, self, img_w, img_h)
            item.setVisible(self.pred_checkbox.isChecked())
            self.scene.addItem(item)
            self.pred_items.append(item)

        for state in self.gt_states[index]:
            rect = yolo_line_to_rect(state["line"], img_w, img_h)
            item = GTBox(rect, state, self.class_names, self, img_w, img_h)
            item.setVisible(self.gt_checkbox.isChecked())
            self.scene.addItem(item)
            self.gt_items.append(item)

        self.flag_predictions()
        self.update_final_items()

    def flag_predictions(self):
        for p in self.pred_items:
            best_iou = 0.0
            best_gt = None
            for g in self.gt_items:
                if not g.kept:
                    continue
                i = iou(p.rect(), g.rect())
                if i > best_iou:
                    best_iou = i
                    best_gt = g
            if best_iou == 0 or (
                best_gt and p.line.split()[0] != best_gt.line.split()[0]
            ):
                p.setPen(QPen(QColor(255, 191, 0), 2))
            else:
                p.setPen(QPen(QColor("red"), 2))

    def update_image_display(self):
        img = self.images[self.index]
        pixmap = self.pil_to_pixmap(self.adjust_image(img))
        self.background_item.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Navigation and saving
    # ------------------------------------------------------------------
    def next_image(self):
        if self.index < len(self.images) - 1:
            self.load_image(self.index + 1)

    def prev_image(self):
        if self.index > 0:
            self.load_image(self.index - 1)

    def collect_lines(self, idx: int) -> List[str]:
        lines = [s["line"] for s in self.gt_states[idx] if s.get("kept", True)]
        lines += [s["line"] for s in self.pred_states[idx] if s.get("accepted", False)]
        return lines

    def save_all(self):
        for idx, label_file in enumerate(self.label_files):
            lines = self.collect_lines(idx)
            os.makedirs(os.path.dirname(label_file), exist_ok=True)
            with open(label_file, "w") as f:
                for line in lines:
                    f.write(line + "\n")

    # ------------------------------------------------------------------
    # Visibility toggles and preview
    # ------------------------------------------------------------------
    def toggle_predictions(self, state: bool):
        for item in self.pred_items:
            item.setVisible(state)

    def toggle_gt(self, state: bool):
        for item in self.gt_items:
            item.setVisible(state)

    def toggle_final(self, state: bool):
        self.update_final_items()

    def update_final_items(self):
        for item in self.final_items:
            self.scene.removeItem(item)
        self.final_items = []
        if not self.final_checkbox.isChecked():
            return
        for item in self.gt_items:
            if item.kept:
                rect = QGraphicsRectItem(item.rect())
                rect.setPen(QPen(QColor("blue"), 2))
                self.scene.addItem(rect)
                self.final_items.append(rect)
        for item in self.pred_items:
            if item.accepted:
                rect = QGraphicsRectItem(item.rect())
                rect.setPen(QPen(QColor("blue"), 2))
                self.scene.addItem(rect)
                self.final_items.append(rect)

    def preview(self):
        lines = self.collect_lines(self.index)
        text = "\n".join(lines) if lines else "No labels selected"
        QMessageBox.information(self, "Final Labels", text)


def run_interface(
    images: List,
    predictions: List[List[dict]],
    labels: List[List[dict]],
    label_files: List[str],
    class_names: List[str],
):
    """Launch the PyQt6 interface for multiple images."""
    app = QApplication.instance() or QApplication([])
    window = AnnotationWindow(images, predictions, labels, label_files, class_names)
    window.show()
    app.exec()

