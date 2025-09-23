"""Utilities for tracking annotation progress across sessions."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, Optional


class ProgressTracker:
    """Persist completion state for images being reviewed.

    The tracker stores a mapping from an image identifier (typically the
    filename without extension) to a boolean flag indicating whether the image
    has been reviewed.  The most recently viewed image identifier is also
    persisted so a later session can resume at the same position.  The data is
    written to a JSON file so progress can be restored on subsequent runs of
    the tool.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.completed: Dict[str, bool] = {}
        self.last_image_id: Optional[str] = None
        self._load()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _load(self) -> None:
        """Load existing progress information from ``self.path`` if present."""

        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError, TypeError):
            # Treat a missing or corrupt file as if no progress was saved.
            self.completed = {}
            return

        completed = data.get("completed") if isinstance(data, dict) else None
        if isinstance(completed, dict):
            self.completed = {str(k): bool(v) for k, v in completed.items()}
        else:
            self.completed = {}

        last_image = data.get("last_image") if isinstance(data, dict) else None
        if isinstance(last_image, str):
            self.last_image_id = last_image
        else:
            self.last_image_id = None

    def save(self) -> None:
        """Write the current progress mapping back to disk."""

        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = self.path + ".tmp"
        data = {"completed": self.completed, "last_image": self.last_image_id}
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self.path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_completed(self, image_id: str) -> bool:
        """Return ``True`` if ``image_id`` has been marked as completed."""

        return bool(self.completed.get(image_id, False))

    def set_completed(self, image_id: str, completed: bool) -> None:
        """Update the completion state for ``image_id`` and persist the change."""

        if completed:
            self.completed[image_id] = True
        else:
            self.completed.pop(image_id, None)
        self.save()

    def set_last_image(self, image_id: Optional[str]) -> None:
        """Persist the identifier of the most recently viewed image."""

        if image_id == self.last_image_id:
            return
        self.last_image_id = image_id
        self.save()

    def prune(self, valid_ids: Iterable[str]) -> None:
        """Remove entries for images that are no longer part of the session."""

        valid = set(valid_ids)
        removed = False
        last_removed = False
        for key in list(self.completed):
            if key not in valid:
                del self.completed[key]
                removed = True
        if self.last_image_id is not None and self.last_image_id not in valid:
            self.last_image_id = None
            last_removed = True
        if removed or last_removed:
            self.save()

