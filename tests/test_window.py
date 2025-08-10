import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(os.getcwd())

from PIL import Image
from PyQt6.QtWidgets import QApplication, QGraphicsTextItem
import pytest

from window import AnnotationWindow


def test_annotation_window_smoke(tmp_path):
    """Smoke test ensuring the window can be constructed and basic methods run."""

    app = QApplication.instance() or QApplication([])
    img = Image.new("RGB", (10, 10))
    images = [img]
    preds = [[{"line": "0 0.5 0.5 0.2 0.2", "conf": 0.9, "accepted": False}]]
    gts = [[{"line": "0 0.5 0.5 0.2 0.2", "kept": True}]]
    label_files = [str(tmp_path / "labels.txt")]
    window = AnnotationWindow(images, preds, gts, label_files, ["obj"])

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
    window = AnnotationWindow(images, preds, gts, label_files, ["obj"])
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
    window = AnnotationWindow(images, preds, gts, label_files, ["obj"])
    window.view.scale(2, 2)
    window.view.horizontalScrollBar().setValue(10)
    window.view.verticalScrollBar().setValue(20)
    window.next_image()
    assert window.view.transform().m11() == pytest.approx(2)
    assert window.view.horizontalScrollBar().value() == 10
    assert window.view.verticalScrollBar().value() == 20

