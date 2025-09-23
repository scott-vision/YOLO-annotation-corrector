"""Main application window for correcting YOLO annotations."""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

from PIL import ImageEnhance
from PyQt6.QtCore import QEvent, QPointF, Qt, QRectF
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
from progress import ProgressTracker
from view_utils import ZoomableGraphicsView

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from sam_integration import SamTightener


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
        image_ids: Optional[List[str]] = None,
        progress_tracker: Optional[ProgressTracker] = None,
        completed: Optional[List[bool]] = None,
        start_index: int = 0,
        sam_helper: Optional["SamTightener"] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Annotation Corrector")
        self.class_names = class_names
        self.images = images
        self.pred_states = predictions
        self.gt_states = labels
        self.label_files = label_files
        self.sam_helper = sam_helper
        if image_ids is None:
            image_ids = [
                os.path.splitext(os.path.basename(path))[0]
                for path in label_files
            ]
        if len(image_ids) != len(self.images):
            raise ValueError("The number of image identifiers must match the images provided.")
        self.image_ids = image_ids
        if completed is None:
            completed = [False] * len(self.images)
        if len(completed) != len(self.images):
            raise ValueError("The completed list must match the number of images.")
        self.completed_states = list(completed)
        self.progress_tracker = progress_tracker
        self.index = 0
        self._sam_history: Dict[int, List[Dict[str, List[str]]]] = defaultdict(list)
        self._waiting_for_single_sam = False

        # Ensure the window can receive key events for navigation.
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.scene = QGraphicsScene(self)
        self.view = ZoomableGraphicsView(self.scene)
        self.view.viewport().installEventFilter(self)

        self.pred_items: List[PredBox] = []
        self.gt_items: List[GTBox] = []
        self.final_items: List[QGraphicsItem] = []

        # Checkboxes controlling visibility of annotation layers
        self.pred_checkbox = QCheckBox("Show predictions")
        self.pred_checkbox.setChecked(True)
        self.pred_checkbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.pred_checkbox.toggled.connect(self.toggle_predictions)  # signal/slot

        self.gt_checkbox = QCheckBox("Show ground truth")
        self.gt_checkbox.setChecked(True)
        self.gt_checkbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.gt_checkbox.toggled.connect(self.toggle_gt)  # signal/slot

        self.final_checkbox = QCheckBox("Show final labels")
        self.final_checkbox.setChecked(False)
        self.final_checkbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.final_checkbox.toggled.connect(self.toggle_final)  # signal/slot

        self.complete_checkbox = QCheckBox("Mark image reviewed")
        self.complete_checkbox.setChecked(False)
        self.complete_checkbox.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.complete_checkbox.toggled.connect(self.completion_toggled)

        self.progress_label = QLabel("")
        self.progress_label.setMinimumWidth(220)
        self.progress_label.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # Buttons for previewing, saving and navigation
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.preview_btn.clicked.connect(self.preview)  # signal/slot

        self.save_btn = QPushButton("Save")
        self.save_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.save_btn.clicked.connect(self.save_all)  # signal/slot

        self.sam_tighten_btn = QPushButton("SAM2 tighten")
        self.sam_tighten_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sam_tighten_btn.clicked.connect(self.apply_sam_to_all)

        self.sam_revert_btn = QPushButton("Revert SAM2")
        self.sam_revert_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.sam_revert_btn.clicked.connect(self.revert_sam_changes)

        self.single_sam_btn = QPushButton("Single SAM2")
        self.single_sam_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.single_sam_btn.setCheckable(True)
        self.single_sam_btn.toggled.connect(self.queue_single_sam)

        self.prev_btn = QPushButton("Previous")
        self.prev_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.prev_btn.clicked.connect(self.prev_image)  # signal/slot

        self.next_btn = QPushButton("Next")
        self.next_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.next_btn.clicked.connect(self.next_image)  # signal/slot

        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.exit_btn.clicked.connect(self.close)  # signal/slot

        # Sliders for simple brightness/contrast adjustments
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(0, 200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.brightness_slider.valueChanged.connect(self.update_image_display)

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(0, 200)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.contrast_slider.valueChanged.connect(self.update_image_display)

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.pred_checkbox)
        control_layout.addWidget(self.gt_checkbox)
        control_layout.addWidget(self.final_checkbox)
        control_layout.addWidget(self.complete_checkbox)
        control_layout.addWidget(self.progress_label)
        control_layout.addWidget(self.preview_btn)
        control_layout.addWidget(self.save_btn)
        control_layout.addWidget(self.sam_tighten_btn)
        control_layout.addWidget(self.sam_revert_btn)
        control_layout.addWidget(self.single_sam_btn)
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

        target_index = min(max(start_index, 0), len(self.images) - 1) if self.images else 0
        self.load_image(target_index)
        # Ensure main window receives keyboard focus for arrow navigation
        self.setFocus()
        self._update_sam_buttons()

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
        # Preserve current scroll and zoom so navigation keeps the same view.
        h_val = self.view.horizontalScrollBar().value()
        v_val = self.view.verticalScrollBar().value()
        transform = self.view.transform()

        if not self.images:
            return

        index = min(max(index, 0), len(self.images) - 1)
        self.scene.clear()
        self.pred_items = []
        self.gt_items = []
        self.final_items = []
        self.index = index

        img = self.images[index]
        if self.single_sam_btn.isChecked():
            self.single_sam_btn.setChecked(False)
        if self.sam_helper is not None:
            self.sam_helper.set_image(img)
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
        self.update_completion_controls()
        self._update_sam_buttons()

        # Restore previous view parameters
        self.view.setTransform(transform)
        self.view.horizontalScrollBar().setValue(h_val)
        self.view.verticalScrollBar().setValue(v_val)

    def update_completion_controls(self) -> None:
        """Refresh completion checkbox and progress label for the current image."""

        self.complete_checkbox.blockSignals(True)
        self.complete_checkbox.setChecked(self.completed_states[self.index])
        self.complete_checkbox.blockSignals(False)
        self.update_progress_label()

    def update_progress_label(self) -> None:
        """Display a concise summary of review progress."""

        total = len(self.images)
        if total == 0:
            self.progress_label.setText("No images loaded")
            return
        reviewed = sum(1 for state in self.completed_states if state)
        current = self.index + 1
        name = self.image_ids[self.index]
        status = "Reviewed" if self.completed_states[self.index] else "Pending"
        self.progress_label.setText(
            f"Image {current}/{total} • Reviewed {reviewed}/{total} • {name} ({status})"
        )

    def _update_sam_buttons(self) -> None:
        """Enable or disable SAM controls based on availability and history."""

        has_helper = self.sam_helper is not None
        if not has_helper and self.single_sam_btn.isChecked():
            self.single_sam_btn.blockSignals(True)
            self.single_sam_btn.setChecked(False)
            self.single_sam_btn.blockSignals(False)
            self.view.setCursor(Qt.CursorShape.ArrowCursor)
        self.sam_tighten_btn.setEnabled(has_helper)
        self.single_sam_btn.setEnabled(has_helper)
        history = self._sam_history.get(self.index, [])
        self.sam_revert_btn.setEnabled(has_helper and bool(history))

    def completion_toggled(self, state: bool) -> None:
        """Handle the user toggling the completion checkbox."""

        self.completed_states[self.index] = state
        if self.progress_tracker is not None:
            self.progress_tracker.set_completed(self.image_ids[self.index], state)
        self.update_progress_label()

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

    # ------------------------------------------------------------------
    # SAM tightening helpers
    # ------------------------------------------------------------------
    def _snapshot_current_state(self) -> Dict[str, List[str]]:
        return {
            "pred": [state["line"] for state in self.pred_states[self.index]],
            "gt": [state["line"] for state in self.gt_states[self.index]],
        }

    def _apply_snapshot(self, snapshot: Dict[str, List[str]]) -> None:
        for state, line in zip(self.pred_states[self.index], snapshot.get("pred", [])):
            state["line"] = line
        for item, line in zip(self.pred_items, snapshot.get("pred", [])):
            rect = yolo_line_to_rect(line, item.img_w, item.img_h)
            item.setRect(rect)
            item._update_from_rect()
            item.line = line
            if isinstance(item.state, dict):
                item.state["line"] = line
        for state, line in zip(self.gt_states[self.index], snapshot.get("gt", [])):
            state["line"] = line
        for item, line in zip(self.gt_items, snapshot.get("gt", [])):
            rect = yolo_line_to_rect(line, item.img_w, item.img_h)
            item.setRect(rect)
            item._update_from_rect()
            item.line = line
            if isinstance(item.state, dict):
                item.state["line"] = line

    def _rect_from_bbox(
        self, bbox: Tuple[float, float, float, float], img_w: float, img_h: float
    ) -> Optional[QRectF]:
        x1, y1, x2, y2 = bbox
        x1 = max(0.0, min(x1, img_w))
        x2 = max(0.0, min(x2, img_w))
        y1 = max(0.0, min(y1, img_h))
        y2 = max(0.0, min(y2, img_h))
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None
        return QRectF(x1, y1, width, height)

    def _set_item_rect(self, item: PredBox | GTBox, rect: QRectF) -> bool:
        state = getattr(item, "state", {})
        old_line = state.get("line", "") if isinstance(state, dict) else getattr(item, "line", "")
        item.setRect(rect)
        item._update_from_rect()
        new_line = state.get("line", "") if isinstance(state, dict) else getattr(item, "line", "")
        return new_line != old_line

    def _apply_bbox_to_item(self, item: PredBox | GTBox, bbox: Tuple[float, float, float, float]) -> bool:
        rect = self._rect_from_bbox(bbox, item.img_w, item.img_h)
        if rect is None:
            return False
        return self._set_item_rect(item, rect)

    def _show_sam_error(self, message: str) -> None:
        QMessageBox.warning(self, "SAM tightening", message)

    def apply_sam_to_all(self) -> None:
        """Run SAM tightening for every box using its centre point."""

        if self.sam_helper is None:
            self._show_sam_error("SAM tightening requires a loaded SAM model.")
            return
        if not (self.pred_items or self.gt_items):
            return

        snapshot = self._snapshot_current_state()
        changed = False
        try:
            for item in [*self.gt_items, *self.pred_items]:
                rect = item.rect()
                if rect.width() <= 0 or rect.height() <= 0:
                    continue
                point = (rect.center().x(), rect.center().y())
                roi = (rect.left(), rect.top(), rect.right(), rect.bottom())
                bbox = self.sam_helper.predict_box(point, roi=roi)
                if bbox and self._apply_bbox_to_item(item, bbox):
                    changed = True
        except RuntimeError as exc:
            self._apply_snapshot(snapshot)
            self._update_sam_buttons()
            self._show_sam_error(str(exc))
            return

        if changed:
            self._sam_history[self.index].append(snapshot)
            self._update_sam_buttons()

    def revert_sam_changes(self) -> None:
        """Restore the previous state recorded before a SAM adjustment."""

        history = self._sam_history.get(self.index)
        if not history:
            return
        snapshot = history.pop()
        self._apply_snapshot(snapshot)
        self._update_sam_buttons()

    def queue_single_sam(self, checked: bool) -> None:
        """Toggle waiting for a single SAM prompt using the next click."""

        if not checked:
            self._waiting_for_single_sam = False
            self.view.setCursor(Qt.CursorShape.ArrowCursor)
            self.single_sam_btn.setText("Single SAM2")
            self._update_sam_buttons()
            return
        if self.sam_helper is None:
            self.single_sam_btn.blockSignals(True)
            self.single_sam_btn.setChecked(False)
            self.single_sam_btn.blockSignals(False)
            self._show_sam_error("SAM tightening requires a loaded SAM model.")
            self._update_sam_buttons()
            return
        self._waiting_for_single_sam = True
        self.view.setCursor(Qt.CursorShape.CrossCursor)
        self.single_sam_btn.setText("Single SAM2 (click target)")
        self._update_sam_buttons()

    def _rect_contains(self, outer: QRectF, inner: QRectF, tolerance: float = 1.0) -> bool:
        return (
            inner.left() >= outer.left() - tolerance
            and inner.right() <= outer.right() + tolerance
            and inner.top() >= outer.top() - tolerance
            and inner.bottom() <= outer.bottom() + tolerance
        )

    def apply_single_sam_at(self, point: Tuple[float, float]) -> bool:
        """Apply SAM tightening using a specific image-space point."""

        if self.sam_helper is None:
            self._show_sam_error("SAM tightening requires a loaded SAM model.")
            return False

        snapshot = self._snapshot_current_state()
        roi = None
        if self.gt_items or self.pred_items:
            qpoint = QPointF(point[0], point[1])
            rects: List[QRectF] = []
            for item in [*self.gt_items, *self.pred_items]:
                rect = item.rect()
                if rect.width() <= 0 or rect.height() <= 0:
                    continue
                if rect.contains(qpoint):
                    rects.append(rect)
            if rects:
                target = min(rects, key=lambda r: r.width() * r.height())
                roi = (target.left(), target.top(), target.right(), target.bottom())
        try:
            bbox = self.sam_helper.predict_box(point, roi=roi)
        except RuntimeError as exc:
            self._show_sam_error(str(exc))
            return False
        if not bbox:
            return False

        pixmap = self.background_item.pixmap()
        rect = self._rect_from_bbox(bbox, pixmap.width(), pixmap.height())
        if rect is None:
            return False

        changed = False
        for item in [*self.gt_items, *self.pred_items]:
            if self._rect_contains(item.rect(), rect) and self._set_item_rect(item, rect):
                changed = True

        if changed:
            self._sam_history[self.index].append(snapshot)
            self._update_sam_buttons()
        return changed

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self.view.viewport() and self._waiting_for_single_sam:
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                scene_pos = self.view.mapToScene(event.position().toPoint())
                image_pos = self.background_item.mapFromScene(scene_pos)
                if not self.background_item.contains(image_pos):
                    event.accept()
                    return True
                if self.apply_single_sam_at((image_pos.x(), image_pos.y())):
                    self.single_sam_btn.setChecked(False)
                event.accept()
                return True
        return super().eventFilter(obj, event)

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
    image_ids: Optional[List[str]] = None,
    progress_tracker: Optional[ProgressTracker] = None,
    completed: Optional[List[bool]] = None,
    start_index: int = 0,
    sam_helper: Optional["SamTightener"] = None,
) -> None:
    """Launch the PyQt6 interface for multiple images."""

    app = QApplication.instance() or QApplication([])
    window = AnnotationWindow(
        images,
        predictions,
        labels,
        label_files,
        class_names,
        image_ids=image_ids,
        progress_tracker=progress_tracker,
        completed=completed,
        start_index=start_index,
        sam_helper=sam_helper,
    )
    window.show()
    app.exec()

