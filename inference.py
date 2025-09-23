"""Inference utilities for YOLO models.

This module offers helpers to load YOLOv8 weights either via the
standard Ultralytics interface or through :mod:`sahi` (Slicing Aided
Hyper Inference) for tiled predictions.  Predictions are returned in the
same format as YOLO label files together with a confidence score.
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

try:  # Ultralytics is required for direct inference
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover - handled at runtime if missing
    YOLO = None  # type: ignore[assignment]
    _ULTRALYTICS_IMPORT_ERROR = exc
else:
    _ULTRALYTICS_IMPORT_ERROR = None


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


def load_model(
    model_path: str,
    device: str | None = None,
    *,
    use_sahi: bool = False,
) -> Any:
    """Load a YOLO model for inference.

    Args:
        model_path: Path to the YOLO weights file.
        device: Optional device identifier (e.g. ``"cuda:0"`` or
            ``"cpu"``).  When using SAHI the value is forwarded to
            :func:`sahi.AutoDetectionModel.from_pretrained`.  Direct
            Ultralytics inference stores the preferred device on the
            returned model and passes it to :meth:`YOLO.predict`.
        use_sahi: When ``True`` the model is wrapped in SAHI for tiled
            predictions.  Otherwise direct Ultralytics inference is used.

    Returns:
        Loaded model instance that can be consumed by :func:`predict`.
    """

    _ensure_ultralytics_backports()

    if use_sahi:
        if AutoDetectionModel is None:
            raise ImportError(
                "SAHI is required for tiled inference but could not be imported."
            ) from _SAHI_IMPORT_ERROR

        return AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            device=device,
        )

    if YOLO is None:
        raise ImportError(
            "Ultralytics is required for direct inference but could not be imported."
        ) from _ULTRALYTICS_IMPORT_ERROR

    model = YOLO(model_path)
    if device:
        setattr(model, "_annotation_corrector_device", device)
    return model


def predict(
    model: Any,
    image: Image.Image,
    *,
    use_sahi: bool = False,
) -> List[Tuple[str, float]]:
    """Run prediction on an image and return YOLO-formatted boxes."""

    if use_sahi:
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

    if YOLO is None:
        raise ImportError(
            "Ultralytics is required for direct inference but could not be imported."
        ) from _ULTRALYTICS_IMPORT_ERROR

    device = getattr(model, "_annotation_corrector_device", None)
    results = model.predict(image, verbose=False, device=device)
    if not results:
        return []

    result = results[0]
    boxes = getattr(result, "boxes", None)
    if boxes is None or boxes.xywhn is None:
        return []

    lines = []
    xywhn = boxes.xywhn.cpu().tolist()
    confs = boxes.conf.cpu().tolist() if boxes.conf is not None else [0.0] * len(xywhn)
    clses = boxes.cls.cpu().tolist() if boxes.cls is not None else [0.0] * len(xywhn)
    for box, cls_val, conf in zip(xywhn, clses, confs):
        xc, yc, w, h = box
        line = f"{int(cls_val)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
        lines.append((line, float(conf)))

    return lines

