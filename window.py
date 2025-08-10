"""Main application window for correcting YOLO annotations."""

from __future__ import annotations

import os
from typing import List

from PIL import ImageEnhance
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QImage, QPixmap, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from graphics_items import GTBox, PredBox
from view_utils import ZoomableGraphicsView


def yolo_line_to_rect(line: str, img_w: int, img_h: int) -> QRectF:
    """Convert a YOLO label line to a ``QRectF``."""

    parts = line.split()
    if len(parts) != 5:
        return QRectF()
    _, xc, yc, w, h = parts
    xc, yc, w, h = map(float, (xc, yc, w, h))
    x1 = (xc - w / 2) * img_w
    y1 = (yc - h / 2) * img_h
    return QRectF(x1, y1, w * img_w, h * img_h)


def iou(rect1: QRectF, rect2: QRectF) -> float:
    """Compute intersection-over-union of two rectangles."""

    inter = rect1.intersected(rect2)
    if inter.isNull():
        return 0.0
    inter_area = inter.width() * inter.height()
    union_area = (
        rect1.width() * rect1.height()
        + rect2.width() * rect2.height()
        - inter_area
    )
    if union_area == 0:
        return 0.0
    return inter_area / union_area


class AnnotationWindow(QMainWindow):
    """Main window providing annotation controls with multi-image navigation."""

    def __init__(
        self,
        images: List,
        predictions: List[List[dict]],
        labels: List[List[dict]],
        label_files: List[str],
        class_names: List[str],
    ) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Annotation Corrector")
        self.class_names = class_names
        self.images = images
        self.pred_states = predictions
        self.gt_states = labels
        self.label_files = label_files
        self.index = 0

        # Ensure the window can receive key events for navigation.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene)

        self.pred_items: List[PredBox] = []
        self.gt_items: List[GTBox] = []
        self.final_items: List[QGraphicsItem] = []

        # Checkboxes controlling visibility of annotation layers
        self.pred_checkbox = QCheckBox("Show predictions")
        self.pred_checkbox.setChecked(True)
        self.pred_checkbox.toggled.connect(self.toggle_predictions)  # signal/slot

        self.gt_checkbox = QCheckBox("Show ground truth")
        self.gt_checkbox.setChecked(True)
        self.gt_checkbox.toggled.connect(self.toggle_gt)  # signal/slot

        self.final_checkbox = QCheckBox("Show final labels")
        self.final_checkbox.setChecked(False)
        self.final_checkbox.toggled.connect(self.toggle_final)  # signal/slot

        # Buttons for previewing, saving and navigation
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self.preview)  # signal/slot

        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save_all)  # signal/slot

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.clicked.connect(self.prev_image)  # signal/slot

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.next_image)  # signal/slot

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.clicked.connect(self.close)  # signal/slot

        # Sliders for simple brightness/contrast adjustments
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
        """Convert a PIL image to ``QPixmap``."""

        img = img.convert("RGB")
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def adjust_image(self, img):
        """Apply brightness and contrast adjustments using current slider values."""

        b = self.brightness_slider.value() / 100.0
        c = self.contrast_slider.value() / 100.0
        out = img
        if b != 1.0:
            out = ImageEnhance.Brightness(out).enhance(b)
        if c != 1.0:
            out = ImageEnhance.Contrast(out).enhance(c)
        return out

    def load_image(self, index: int) -> None:
        """Load the image and associated boxes at ``index`` into the scene."""

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

    def flag_predictions(self) -> None:
        """Highlight predictions that do not match any ground truth box."""

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
            if best_iou == 0 or (best_gt and p.line.split()[0] != best_gt.line.split()[0]):
                p.setPen(QPen(QColor(255, 191, 0), 2))
            else:
                p.setPen(QPen(QColor("red"), 2))

    def update_image_display(self) -> None:
        """Refresh the background image after adjustment changes."""

        img = self.images[self.index]
        pixmap = self.pil_to_pixmap(self.adjust_image(img))
        self.background_item.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Navigation and saving
    # ------------------------------------------------------------------
    def next_image(self) -> None:
        """Advance to the next image in the list."""

        if self.index < len(self.images) - 1:
            self.load_image(self.index + 1)

    def prev_image(self) -> None:
        """Return to the previous image in the list."""

        if self.index > 0:
            self.load_image(self.index - 1)

    # ------------------------------------------------------------------
    # Keyboard navigation
    # ------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        """Handle left/right arrow keys to navigate images."""

        if event.key() == Qt.Key.Key_Right:
            self.next_image()
        elif event.key() == Qt.Key.Key_Left:
            self.prev_image()
        else:
            super().keyPressEvent(event)

    def collect_lines(self, idx: int) -> List[str]:
        """Gather YOLO label lines for image ``idx``."""

        lines = [s["line"] for s in self.gt_states[idx] if s.get("kept", True)]
        lines += [s["line"] for s in self.pred_states[idx] if s.get("accepted", False)]
        return lines

    def save_all(self) -> None:
        """Write all accepted/kept label lines back to disk."""

        for idx, label_file in enumerate(self.label_files):
            lines = self.collect_lines(idx)
            os.makedirs(os.path.dirname(label_file), exist_ok=True)
            with open(label_file, "w") as f:
                for line in lines:
                    f.write(line + "\n")

    # ------------------------------------------------------------------
    # Visibility toggles and preview
    # ------------------------------------------------------------------
    def toggle_predictions(self, state: bool) -> None:
        """Show or hide predicted boxes."""

        for item in self.pred_items:
            item.setVisible(state)

    def toggle_gt(self, state: bool) -> None:
        """Show or hide ground-truth boxes."""

        for item in self.gt_items:
            item.setVisible(state)

    def toggle_final(self, state: bool) -> None:  # noqa: ARG002 - slot signature
        """Update final annotation overlay when checkbox changes."""

        self.update_final_items()

    def update_final_items(self) -> None:
        """Draw overlays for the final set of annotations."""

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
                cls_id = int(item.line.split()[0])
                cls_name = (
                    self.class_names[cls_id]
                    if 0 <= cls_id < len(self.class_names)
                    else str(cls_id)
                )
                label = QGraphicsTextItem()
                label.setHtml(
                    f"<div style='color:blue;background-color:white;'>{cls_name}</div>"
                )
                label.setPos(item.rect().left(), item.rect().top() - 20)
                self.scene.addItem(label)
                self.final_items.append(label)
        for item in self.pred_items:
            if item.accepted:
                rect = QGraphicsRectItem(item.rect())
                rect.setPen(QPen(QColor("blue"), 2))
                self.scene.addItem(rect)
                self.final_items.append(rect)
                cls_id = int(item.line.split()[0])
                cls_name = (
                    self.class_names[cls_id]
                    if 0 <= cls_id < len(self.class_names)
                    else str(cls_id)
                )
                label = QGraphicsTextItem()
                label.setHtml(
                    f"<div style='color:blue;background-color:white;'>{cls_name}</div>"
                )
                label.setPos(item.rect().left(), item.rect().top() - 20)
                self.scene.addItem(label)
                self.final_items.append(label)

    def preview(self) -> None:
        """Display a message box with the final labels for the current image."""

        lines = self.collect_lines(self.index)
        text = "\n".join(lines) if lines else "No labels selected"
        QMessageBox.information(self, "Final Labels", text)


def run_interface(
    images: List,
    predictions: List[List[dict]],
    labels: List[List[dict]],
    label_files: List[str],
    class_names: List[str],
) -> None:
    """Launch the PyQt6 interface for multiple images."""

    app = QApplication.instance() or QApplication([])
    window = AnnotationWindow(images, predictions, labels, label_files, class_names)
    window.show()
    app.exec()

