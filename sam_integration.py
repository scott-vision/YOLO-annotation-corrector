"""Utilities for running SAM2-based box tightening via Ultralytics."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import Image

try:  # pragma: no cover - optional dependency and environment specific
    from ultralytics import SAM as _UltralyticsSAM  # type: ignore
except Exception as exc:  # pragma: no cover - import error handled at runtime
    _UltralyticsSAM = None  # type: ignore[assignment]
    _SAM_IMPORT_ERROR = exc
else:  # pragma: no cover - exercised when dependency available at runtime
    _SAM_IMPORT_ERROR = None

SAM_IMPORT_ERROR = _SAM_IMPORT_ERROR


class SamTightener:
    """Thin wrapper around the Ultralytics SAM API for interactive tightening."""

    def __init__(self, weights_path: str) -> None:
        if not weights_path:
            raise ValueError("A path to SAM weights must be provided.")
        if _UltralyticsSAM is None:
            raise ImportError(
                "Ultralytics SAM models are not available. Install the optional dependencies and ensure "
                "that system packages such as libGL are present."
            ) from _SAM_IMPORT_ERROR
        self._model = _UltralyticsSAM(weights_path)
        self._image: Optional[np.ndarray] = None

    def set_image(self, image: Image.Image) -> None:
        """Store the current image for subsequent point prompts."""

        self._image = np.array(image.convert("RGB"), copy=True)

    def predict_box(
        self,
        point: Tuple[float, float],
        roi: Optional[Tuple[float, float, float, float]] = None,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Return a tightened bounding box around ``point`` using SAM.

        When ``roi`` is provided the model is executed on a cropped view that
        extends the region of interest by 50%% on each side. The prediction is
        then mapped back to the full-image coordinate system.
        """

        if self._image is None:
            raise RuntimeError("No image has been set on the SAM tightener.")

        full_height, full_width = self._image.shape[:2]
        crop_image = self._image
        offset_x = 0.0
        offset_y = 0.0

        if roi is not None:
            x1, y1, x2, y2 = roi
            left = float(min(x1, x2))
            right = float(max(x1, x2))
            top = float(min(y1, y2))
            bottom = float(max(y1, y2))
            roi_width = max(0.0, right - left)
            roi_height = max(0.0, bottom - top)
            if roi_width > 0.0 and roi_height > 0.0:
                pad_x = 0.5 * roi_width
                pad_y = 0.5 * roi_height
                crop_left = max(0.0, left - pad_x)
                crop_top = max(0.0, top - pad_y)
                crop_right = min(float(full_width), right + pad_x)
                crop_bottom = min(float(full_height), bottom + pad_y)
                crop_left_int = max(0, int(np.floor(crop_left)))
                crop_top_int = max(0, int(np.floor(crop_top)))
                crop_right_int = min(int(np.ceil(crop_right)), full_width)
                crop_bottom_int = min(int(np.ceil(crop_bottom)), full_height)
                if crop_right_int > crop_left_int and crop_bottom_int > crop_top_int:
                    candidate = self._image[crop_top_int:crop_bottom_int, crop_left_int:crop_right_int]
                    if candidate.size:
                        crop_image = candidate
                        offset_x = float(crop_left_int)
                        offset_y = float(crop_top_int)

        height, width = crop_image.shape[:2]
        if height == 0 or width == 0:
            return None

        x, y = point
        x = float(np.clip(x - offset_x, 0.0, max(width - 1, 0)))
        y = float(np.clip(y - offset_y, 0.0, max(height - 1, 0)))

        try:
            results = self._model.predict(  # type: ignore[call-arg]
                source=crop_image,
                points=[[x, y]],
                labels=[1],
                multimask_output=False,
                save=False,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - runtime failure depends on external libs
            raise RuntimeError(f"SAM prediction failed: {exc}") from exc

        if not results:
            return None
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        xyxy = boxes.xyxy
        scores = getattr(boxes, "conf", None)
        if hasattr(xyxy, "cpu"):
            xyxy_array = xyxy.cpu().numpy()
        else:
            xyxy_array = np.asarray(xyxy)

        index = 0
        if scores is not None:
            scores_array = scores.cpu().numpy() if hasattr(scores, "cpu") else np.asarray(scores)
            if scores_array.size:
                index = int(scores_array.argmax())

        x1, y1, x2, y2 = map(float, xyxy_array[index])
        x1 = float(np.clip(x1, 0.0, float(width))) + offset_x
        x2 = float(np.clip(x2, 0.0, float(width))) + offset_x
        y1 = float(np.clip(y1, 0.0, float(height))) + offset_y
        y2 = float(np.clip(y2, 0.0, float(height))) + offset_y
        x1 = float(np.clip(x1, 0.0, float(full_width)))
        x2 = float(np.clip(x2, 0.0, float(full_width)))
        y1 = float(np.clip(y1, 0.0, float(full_height)))
        y2 = float(np.clip(y2, 0.0, float(full_height)))
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2


__all__ = ["SamTightener", "SAM_IMPORT_ERROR"]
