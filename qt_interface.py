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

    def __init__(self, rect: QRectF, line: str, conf: float):
        super().__init__(rect)
        self.line = line
        self.conf = conf
        self.accepted = False
        self.setPen(QPen(QColor("red"), 2))

        cls = line.split()[0]
        self.label = QGraphicsTextItem(f"{cls}:{conf:.2f}", self)
        self.label.setDefaultTextColor(QColor("red"))
        self.label.setPos(rect.left(), rect.top() - 15)

        self.tick = QGraphicsTextItem("✓", self)
        self.tick.setDefaultTextColor(QColor("gray"))
        self.tick.setPos(rect.left(), rect.top())

    def mousePressEvent(self, event):
        self.accepted = not self.accepted
        color = QColor("green") if self.accepted else QColor("gray")
        self.tick.setDefaultTextColor(color)
        super().mousePressEvent(event)


class GTBox(QGraphicsRectItem):
    """Graphics item for an existing ground truth box."""

    def __init__(self, rect: QRectF, line: str):
        super().__init__(rect)
        self.line = line
        self.kept = True
        self.setPen(QPen(QColor("green"), 2))

        self.cross = QGraphicsTextItem("✗", self)
        self.cross.setDefaultTextColor(QColor("red"))
        self.cross.setPos(rect.left(), rect.top())

    def mousePressEvent(self, event):
        self.kept = not self.kept
        color = QColor("red") if self.kept else QColor("gray")
        self.cross.setDefaultTextColor(color)
        super().mousePressEvent(event)


class AnnotationWindow(QMainWindow):
    """Main window providing annotation controls."""

    def __init__(
        self,
        pixmap: QPixmap,
        predictions: List[Tuple[str, float]],
        labels: List[str],
        label_file: str,
    ):
        super().__init__()
        self.setWindowTitle("YOLO Annotation Corrector")
        self.label_file = label_file

        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.scene.addItem(QGraphicsPixmapItem(pixmap))

        img_w = pixmap.width()
        img_h = pixmap.height()

        self.pred_items: List[PredBox] = []
        for line, conf in predictions:
            rect = yolo_line_to_rect(line, img_w, img_h)
            item = PredBox(rect, line, conf)
            self.scene.addItem(item)
            self.pred_items.append(item)

        self.gt_items: List[GTBox] = []
        for line in labels:
            rect = yolo_line_to_rect(line, img_w, img_h)
            item = GTBox(rect, line)
            self.scene.addItem(item)
            self.gt_items.append(item)

        self.pred_checkbox = QCheckBox("Show predictions")
        self.pred_checkbox.setChecked(True)
        self.pred_checkbox.toggled.connect(self.toggle_predictions)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview)

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_and_close)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.pred_checkbox)
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


def run_interface(image, predictions: List[Tuple[str, float]], label_lines: List[str], label_file: str):
    """Launch the PyQt6 interface for a single image."""
    app = QApplication.instance() or QApplication([])

    img = image.convert("RGB")
    data = img.tobytes("raw", "RGB")
    qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)

    window = AnnotationWindow(pixmap, predictions, label_lines, label_file)
    window.show()
    app.exec()
