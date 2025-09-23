"""Inference utilities using SAHI for sliding window predictions.

This module provides helper functions to load a YOLO model and run
inference on large images by slicing them into 640x640 windows using
`sahi` (Slicing Aided Hyper Inference). Predictions are returned in the
same format as YOLO label files with an accompanying confidence score.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Any

from PIL import Image

try:  # Optional dependency until inference is requested
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError as exc:  # pragma: no cover - handled at runtime if missing
    AutoDetectionModel = None  # type: ignore[assignment]
    get_sliced_prediction = None  # type: ignore[assignment]
    _SAHI_IMPORT_ERROR = exc
else:
    _SAHI_IMPORT_ERROR = None


LOGGER = logging.getLogger(__name__)


def _ensure_ultralytics_backports(block_module: Any | None = None) -> None:
    """Backport newer Ultralytics modules for older installations.

    Some YOLOv8 weights rely on layers such as :class:`C3k2` that were
    introduced in later releases of ``ultralytics``.  When these weights are
    loaded with an older package version the attribute lookup fails which in
    turn bubbles up as a ``TypeError`` when SAHI initialises the model.  To
    keep the application compatible with long-lived environments we
    dynamically add lightweight shims for the missing layers.  The
    implementation mirrors the upstream definitions closely enough for the
    state dict to load without modification.

    Args:
        block_module: Optional ``ultralytics.nn.modules.block`` module.  The
            parameter primarily exists for unit tests; at runtime the module
            is imported automatically when ``None``.
    """

    if block_module is None:
        try:
            from ultralytics.nn.modules import block as block_module  # type: ignore
        except (ImportError, AttributeError):
            return

    try:
        import torch.nn as nn
    except ImportError:
        LOGGER.debug("torch is not available; skipping Ultralytics backport shims")
        return

    bottleneck = getattr(block_module, "Bottleneck", None)
    c2f = getattr(block_module, "C2f", None)
    c3 = getattr(block_module, "C3", None)
    if not all((bottleneck, c2f, c3)):
        LOGGER.debug(
            "Required Ultralytics building blocks are missing; unable to backport C3k2"
        )
        return

    if not hasattr(block_module, "C3k"):
        class C3k(c3):  # type: ignore[misc, valid-type]
            """Approximation of the upstream C3k layer."""

            def __init__(
                self,
                c1: int,
                c2: int,
                n: int = 1,
                shortcut: bool = True,
                g: int = 1,
                e: float = 0.5,
                k: int = 3,
            ) -> None:
                super().__init__(c1, c2, n, shortcut, g, e)
                hidden = int(c2 * e)
                self.m = nn.Sequential(
                    *(bottleneck(hidden, hidden, shortcut, g, k=(k, k), e=1.0) for _ in range(n))
                )

        block_module.C3k = C3k  # type: ignore[attr-defined]
        LOGGER.info("Backported missing Ultralytics C3k layer")

    if not hasattr(block_module, "C3k2"):
        c3k_cls = getattr(block_module, "C3k", None)

        class C3k2(c2f):  # type: ignore[misc, valid-type]
            """Compatibility shim for the C3k2 layer used by new YOLO weights."""

            def __init__(
                self,
                c1: int,
                c2: int,
                n: int = 1,
                c3k: bool = False,
                e: float = 0.5,
                g: int = 1,
                shortcut: bool = True,
            ) -> None:
                super().__init__(c1, c2, n, shortcut, g, e)
                block_factory = (
                    (lambda: c3k_cls(self.c, self.c, 2, shortcut, g))
                    if c3k_cls and c3k
                    else (lambda: bottleneck(self.c, self.c, shortcut, g))
                )
                self.m = nn.ModuleList(block_factory() for _ in range(n))

        block_module.C3k2 = C3k2  # type: ignore[attr-defined]
        LOGGER.info("Backported missing Ultralytics C3k2 layer")


def load_model(model_path: str, device: str | None = None) -> AutoDetectionModel:
    """Load a YOLO model for sliced inference.

    Args:
        model_path: Path to the YOLO weights file.
        device: Device identifier (e.g., ``"cuda:0"`` or ``"cpu"``).
            If ``None``, ``sahi`` will select the available device.

    Returns:
        AutoDetectionModel: Wrapped detection model ready for SAHI inference.
    """

    if AutoDetectionModel is None:
        raise ImportError(
            "SAHI is required for inference but could not be imported."
        ) from _SAHI_IMPORT_ERROR

    _ensure_ultralytics_backports()

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

    if get_sliced_prediction is None:
        raise ImportError(
            "SAHI prediction utilities are unavailable; install optional dependencies to enable inference."
        ) from _SAHI_IMPORT_ERROR

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

