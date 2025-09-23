import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.append(os.getcwd())

from PIL import Image

import argparse

import annotation_corrector


def test_class_names_loaded_with_predictions(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    corrected_dir = tmp_path / "corrected"
    images_dir.mkdir()
    labels_dir.mkdir()
    corrected_dir.mkdir()

    # Create a dummy image
    img_path = images_dir / "img0.jpg"
    Image.new("RGB", (10, 10)).save(img_path)

    # Original labels (empty to force mismatch with predictions)
    (labels_dir / "img0.txt").write_text("")

    # Cached predictions
    pred_dir = corrected_dir / "predicted_labels"
    pred_dir.mkdir()
    (pred_dir / "img0.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")

    class DummyModel:
        def __init__(self):
            self.model = type("m", (), {"names": ["cls"]})()

    def fake_load_model(path):
        return DummyModel()

    captured = {}

    def fake_run_interface(images, predictions, labels, label_files, class_names, **kwargs):
        captured["class_names"] = class_names

    monkeypatch.setattr(annotation_corrector, "load_model", fake_load_model)
    monkeypatch.setattr(annotation_corrector, "run_interface", fake_run_interface)

    args = [
        "prog",
        "--images",
        str(images_dir),
        "--labels",
        str(labels_dir),
        "--corrected",
        str(corrected_dir),
        "--model",
        "dummy.pt",
        "--predictions",
    ]
    monkeypatch.setattr(sys, "argv", args)

    annotation_corrector.main()

    assert captured["class_names"] == ["cls"]


def test_configuration_file_applied(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    config_path = images_dir / annotation_corrector.CONFIG_FILENAME
    config_path.write_text(
        '{"labels": "L", "corrected": "C", "model": "M", "predictions": true}'
    )

    args = argparse.Namespace(
        images=str(images_dir),
        labels="",
        corrected="",
        model="",
        sam=None,
        predictions=False,
    )

    loaded = annotation_corrector._load_config_file(str(config_path))
    assert loaded is not None
    updated = annotation_corrector._apply_config_to_args(args, loaded)

    assert updated.labels == "L"
    assert updated.corrected == "C"
    assert updated.model == "M"
    assert updated.predictions is True


def test_progress_resume_start_index(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    corrected_dir = tmp_path / "corrected"
    images_dir.mkdir()
    labels_dir.mkdir()
    corrected_dir.mkdir()

    # Create dummy images
    for idx in range(2):
        Image.new("RGB", (10, 10)).save(images_dir / f"img{idx}.jpg")
        (labels_dir / f"img{idx}.txt").write_text("")

    pred_dir = corrected_dir / "predicted_labels"
    pred_dir.mkdir()
    for idx in range(2):
        (pred_dir / f"img{idx}.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")

    # Mark the first image as completed in the progress file
    progress_path = corrected_dir / "annotation_progress.json"
    progress_path.write_text('{"completed": {"img0": true}}')

    class DummyModel:
        def __init__(self):
            self.model = type("m", (), {"names": ["cls"]})()

    def fake_load_model(path):
        return DummyModel()

    captured = {}

    def fake_run_interface(images, predictions, labels, label_files, class_names, **kwargs):
        captured["start_index"] = kwargs.get("start_index")
        captured["completed"] = kwargs.get("completed")

    monkeypatch.setattr(annotation_corrector, "load_model", fake_load_model)
    monkeypatch.setattr(annotation_corrector, "run_interface", fake_run_interface)

    args = [
        "prog",
        "--images",
        str(images_dir),
        "--labels",
        str(labels_dir),
        "--corrected",
        str(corrected_dir),
        "--model",
        "dummy.pt",
        "--predictions",
    ]
    monkeypatch.setattr(sys, "argv", args)

    annotation_corrector.main()

    assert captured["start_index"] == 1
    assert captured["completed"] == [True, False]
