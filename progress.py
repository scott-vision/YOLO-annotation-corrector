"""Utilities for tracking annotation progress across sessions."""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable


class ProgressTracker:
    """Persist completion state for images being reviewed.

    The tracker stores a mapping from an image identifier (typically the
    filename without extension) to a boolean flag indicating whether the image
    has been reviewed.  The data is written to a JSON file so progress can be
    restored on subsequent runs of the tool.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.completed: Dict[str, bool] = {}
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

    def save(self) -> None:
        """Write the current progress mapping back to disk."""

        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = self.path + ".tmp"
        data = {"completed": self.completed}
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

    def prune(self, valid_ids: Iterable[str]) -> None:
        """Remove entries for images that are no longer part of the session."""

        valid = set(valid_ids)
        removed = False
        for key in list(self.completed):
            if key not in valid:
                del self.completed[key]
                removed = True
        if removed:
            self.save()

