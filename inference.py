"""Inference utilities using SAHI for sliding window predictions.

This module provides helper functions to load a YOLO model and run
inference on large images by slicing them into 640x640 windows using
`sahi` (Slicing Aided Hyper Inference). Predictions are returned in the
same format as YOLO label files with an accompanying confidence score.
"""

from __future__ import annotations

from typing import List, Tuple

from PIL import Image
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


def load_model(model_path: str, device: str | None = None) -> AutoDetectionModel:
    """Load a YOLO model for sliced inference.

    Args:
        model_path: Path to the YOLO weights file.
        device: Device identifier (e.g., ``"cuda:0"`` or ``"cpu"``).
            If ``None``, ``sahi`` will select the available device.

    Returns:
        AutoDetectionModel: Wrapped detection model ready for SAHI inference.
    """

    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        device=device,
    )


def predict(
    model: AutoDetectionModel, image: Image.Image
) -> List[Tuple[str, float]]:
    """Run sliced prediction on an image and return YOLO-formatted boxes.

    The image is processed with a 640x640 sliding window and a small overlap
    to ensure objects at the borders are detected. Each prediction is
    formatted as a tuple containing the YOLO label line and its confidence
    score.

    Args:
        model: Loaded ``AutoDetectionModel`` from :func:`load_model`.
        image: Input image on which inference will be performed.

    Returns:
        List[Tuple[str, float]]: List of ``("cls xc yc w h", confidence)``
        pairs with normalized coordinates.
    """

    result = get_sliced_prediction(
        image,
        detection_model=model,
        slice_height=640,
        slice_width=640,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    img_w, img_h = image.size
    lines: List[Tuple[str, float]] = []
    for obj in result.object_prediction_list:
        bbox = obj.bbox
        x1, y1, x2, y2 = bbox.minx, bbox.miny, bbox.maxx, bbox.maxy
        w = x2 - x1
        h = y2 - y1
        xc = x1 + w / 2
        yc = y1 + h / 2

        # Normalize coordinates to image dimensions
        norm = (
            xc / img_w,
            yc / img_h,
            w / img_w,
            h / img_h,
        )

        cls = obj.category.id
        conf = obj.score.value
        line = f"{cls} {norm[0]:.6f} {norm[1]:.6f} {norm[2]:.6f} {norm[3]:.6f}"
        lines.append((line, conf))

    return lines

