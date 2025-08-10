from __future__ import annotations

import os
from typing import List, Tuple

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
    QMainWindow,
    QMessageBox,
    QPushButton,
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


class PredBox(QGraphicsRectItem):
    """Graphics item for a predicted box."""

    def __init__(self, rect: QRectF, line: str, conf: float, class_names: List[str], window: "AnnotationWindow"):
        super().__init__(rect)
        self.window = window
        self.line = line
        self.conf = conf
        self.accepted = False
        self.setPen(QPen(QColor("red"), 2))

        cls_id = int(line.split()[0])
        cls_name = class_names[cls_id] if 0 <= cls_id < len(class_names) else str(cls_id)
        self.label = QGraphicsTextItem(self)
        self.label.setHtml(f"<div style='background-color:white;'>{cls_name}:{conf:.2f}</div>")
        self.label.setPos(rect.left(), rect.top() - 20)

        self.tick = QGraphicsTextItem(self)
        self._update_tick()
        self.tick.setPos(rect.right() + 2, rect.top() - 20)

    def _update_tick(self):
        color = "green" if self.accepted else "gray"
        self.tick.setHtml(f"<div style='color:{color};background-color:white;'>✓</div>")

    def mousePressEvent(self, event):
        self.accepted = not self.accepted
        self._update_tick()
        super().mousePressEvent(event)
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()


class GTBox(QGraphicsRectItem):
    """Graphics item for an existing ground truth box."""

    def __init__(self, rect: QRectF, line: str, class_names: List[str], window: "AnnotationWindow"):
        super().__init__(rect)
        self.window = window
        self.line = line
        self.kept = True
        self.setPen(QPen(QColor("green"), 2))

        cls_id = int(line.split()[0])
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

    def mousePressEvent(self, event):
        self.kept = not self.kept
        self._update_cross()
        super().mousePressEvent(event)
        if self.window.final_checkbox.isChecked():
            self.window.update_final_items()


class AnnotationWindow(QMainWindow):
    """Main window providing annotation controls."""

    def __init__(
        self,
        pixmap: QPixmap,
        predictions: List[Tuple[str, float]],
        labels: List[str],
        label_file: str,
        class_names: List[str],
    ):
        super().__init__()
        self.setWindowTitle("YOLO Annotation Corrector")
        self.label_file = label_file
        self.class_names = class_names

        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene)
        self.scene.addItem(QGraphicsPixmapItem(pixmap))

        img_w = pixmap.width()
        img_h = pixmap.height()

        self.pred_items: List[PredBox] = []
        for line, conf in predictions:
            rect = yolo_line_to_rect(line, img_w, img_h)
            item = PredBox(rect, line, conf, class_names, self)
            self.scene.addItem(item)
            self.pred_items.append(item)

        self.gt_items: List[GTBox] = []
        for line in labels:
            rect = yolo_line_to_rect(line, img_w, img_h)
            item = GTBox(rect, line, class_names, self)
            self.scene.addItem(item)
            self.gt_items.append(item)

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
        self.save_btn.clicked.connect(self.save_and_close)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.pred_checkbox)
        control_layout.addWidget(self.gt_checkbox)
        control_layout.addWidget(self.final_checkbox)
        control_layout.addWidget(self.preview_btn)
        control_layout.addWidget(self.save_btn)
        controls = QWidget()
        controls.setLayout(control_layout)

        layout = QVBoxLayout()
        layout.addWidget(self.view)
        layout.addWidget(controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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
        lines = [i.line for i in self.gt_items if i.kept]
        lines += [i.line for i in self.pred_items if i.accepted]
        text = "\n".join(lines) if lines else "No labels selected"
        QMessageBox.information(self, "Final Labels", text)

    def save_and_close(self):
        lines = [i.line for i in self.gt_items if i.kept]
        lines += [i.line for i in self.pred_items if i.accepted]
        os.makedirs(os.path.dirname(self.label_file), exist_ok=True)
        with open(self.label_file, "w") as f:
            for line in lines:
                f.write(line + "\n")
        self.close()


def run_interface(
    image,
    predictions: List[Tuple[str, float]],
    label_lines: List[str],
    label_file: str,
    class_names: List[str],
):
    """Launch the PyQt6 interface for a single image."""
    app = QApplication.instance() or QApplication([])

    img = image.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)

    window = AnnotationWindow(pixmap, predictions, label_lines, label_file, class_names)
    window.show()
    app.exec()
