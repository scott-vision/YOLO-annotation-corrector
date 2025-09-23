import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(os.getcwd())

from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QGraphicsTextItem
import pytest

from progress import ProgressTracker
from window import AnnotationWindow


class DummySamHelper:
    def __init__(self, width: float = 1.0) -> None:
        self.width = width
        self.image_size = (0, 0)
        self.points = []
        self.rois = []

    def set_image(self, image: Image.Image) -> None:
        self.image_size = image.size

    def predict_box(self, point, roi=None):
        self.points.append(point)
        self.rois.append(roi)
        if not self.image_size:
            return None
        w, h = self.image_size
        half = self.width / 2
        x, y = point
        x1 = max(0.0, x - half)
        y1 = max(0.0, y - half)
        x2 = min(float(w), x + half)
        y2 = min(float(h), y + half)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)


def test_annotation_window_smoke(tmp_path):
    """Smoke test ensuring the window can be constructed and basic methods run."""

    app = QApplication.instance() or QApplication([])
    img = Image.new("RGB", (10, 10))
    images = [img]
    preds = [[{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.9, "accepted": False}]]
    gts = [[{"line": "0 0.5 0.5 0.2 0.2", "kept": True}]]
    label_files = [str(tmp_path / "labels.txt")]
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    window = AnnotationWindow(images, preds, gts, label_files, ["obj"], progress_tracker=tracker)

    # Trigger a few simple operations
    window.flag_predictions()
    window.update_final_items()
    window.collect_lines(0)
    window.save_all()


def test_final_items_show_classes(tmp_path):
    app = QApplication.instance() or QApplication([])
    img = Image.new("RGB", (10, 10))
    images = [img]
    preds = [[{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.9, "accepted": True}]]
    gts = [[{"line": "0 0.5 0.5 0.2 0.2", "kept": True}]]
    label_files = [str(tmp_path / "labels.txt")]
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    window = AnnotationWindow(images, preds, gts, label_files, ["obj"], progress_tracker=tracker)
    window.final_checkbox.setChecked(True)
    window.update_final_items()
    texts = [
        item.toPlainText()
        for item in window.final_items
        if isinstance(item, QGraphicsTextItem)
    ]
    assert "obj" in texts


def test_navigation_preserves_view(tmp_path):
    app = QApplication.instance() or QApplication([])
    img = Image.new("RGB", (1000, 1000))
    images = [img, img]
    preds = [
        [{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.9, "accepted": False}],
        [{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.8, "accepted": False}],
    ]
    gts = [
        [{"line": "0 0.5 0.5 0.2 0.2", "kept": True}],
        [{"line": "0 0.5 0.5 0.2 0.2", "kept": True}],
    ]
    label_files = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    window = AnnotationWindow(images, preds, gts, label_files, ["obj"], progress_tracker=tracker)
    window.view.scale(2, 2)
    window.view.horizontalScrollBar().setValue(10)
    window.view.verticalScrollBar().setValue(20)
    window.next_image()
    assert window.view.transform().m11() == pytest.approx(2)
    assert window.view.horizontalScrollBar().value() == 10
    assert window.view.verticalScrollBar().value() == 20


def test_arrow_keys_navigate_after_control_interaction(tmp_path):
    app = QApplication.instance() or QApplication([])
    img = Image.new("RGB", (10, 10))
    images = [img, img]
    preds = [
        [{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.9, "accepted": False}],
        [{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.8, "accepted": False}],
    ]
    gts = [
        [{"line": "0 0.5 0.5 0.2 0.2", "kept": True}],
        [{"line": "0 0.5 0.5 0.2 0.2", "kept": True}],
    ]
    label_files = [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    window = AnnotationWindow(images, preds, gts, label_files, ["obj"], progress_tracker=tracker)

    # Interact with a control and ensure arrow keys still navigate images
    window.pred_checkbox.click()
    assert window.index == 0
    QTest.keyClick(window, Qt.Key.Key_Right)
    assert window.index == 1
    QTest.keyClick(window, Qt.Key.Key_Left)
    assert window.index == 0


def test_completion_toggle_updates_tracker(tmp_path):
    app = QApplication.instance() or QApplication([])
    img = Image.new("RGB", (10, 10))
    images = [img]
    preds = [[{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.9, "accepted": False}]]
    gts = [[{"line": "0 0.5 0.5 0.2 0.2", "kept": True}]]
    label_path = tmp_path / "labels.txt"
    label_files = [str(label_path)]
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    window = AnnotationWindow(images, preds, gts, label_files, ["obj"], progress_tracker=tracker)

    image_id = window.image_ids[0]
    assert not tracker.is_completed(image_id)
    assert "Reviewed 0/1" in window.progress_label.text()

    window.complete_checkbox.setChecked(True)
    assert tracker.is_completed(image_id)
    assert "Reviewed 1/1" in window.progress_label.text()

    window.complete_checkbox.setChecked(False)
    assert not tracker.is_completed(image_id)
    assert "Reviewed 0/1" in window.progress_label.text()


def test_sam_tighten_and_revert(tmp_path):
    app = QApplication.instance() or QApplication([])
    img = Image.new("RGB", (10, 10))
    images = [img]
    preds = [[{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.9, "accepted": False}]]
    gts = [[{"line": "0 0.5 0.5 0.2 0.2", "kept": True}]]
    label_files = [str(tmp_path / "labels.txt")]
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    helper = DummySamHelper(width=1.0)
    window = AnnotationWindow(
        images,
        preds,
        gts,
        label_files,
        ["obj"],
        progress_tracker=tracker,
        sam_helper=helper,
    )
    original_pred = window.pred_states[0][0]["line"]
    original_gt = window.gt_states[0][0]["line"]

    window.apply_sam_to_all()

    tightened_pred = window.pred_states[0][0]["line"]
    tightened_gt = window.gt_states[0][0]["line"]
    assert tightened_pred.endswith("0.100000 0.100000")
    assert tightened_gt.endswith("0.100000 0.100000")
    assert tightened_pred != original_pred
    assert helper.points  # ensure the helper was queried
    assert helper.rois[0] is not None

    window.revert_sam_changes()
    assert window.pred_states[0][0]["line"] == original_pred
    assert window.gt_states[0][0]["line"] == original_gt


def test_single_sam_restricts_containing_boxes(tmp_path):
    app = QApplication.instance() or QApplication([])
    img = Image.new("RGB", (10, 10))
    images = [img]
    preds = [[{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.9, "accepted": False}]]
    gts = [[{"line": "0 0.5 0.5 0.2 0.2", "kept": True}]]
    label_files = [str(tmp_path / "labels.txt")]
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    helper = DummySamHelper(width=1.0)
    window = AnnotationWindow(
        images,
        preds,
        gts,
        label_files,
        ["obj"],
        progress_tracker=tracker,
        sam_helper=helper,
    )

    original_line = window.gt_states[0][0]["line"]
    changed = window.apply_single_sam_at((5.0, 5.0))
    assert changed
    assert window.gt_states[0][0]["line"].endswith("0.100000 0.100000")
    assert window.pred_states[0][0]["line"].endswith("0.100000 0.100000")
    assert helper.rois[-1] is not None

    window.revert_sam_changes()
    assert window.gt_states[0][0]["line"] == original_line

