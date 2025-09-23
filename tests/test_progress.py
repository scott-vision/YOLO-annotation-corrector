from __future__ import annotations

from progress import ProgressTracker


def test_progress_tracker_persists_last_image(tmp_path):
    path = tmp_path / "progress.json"
    tracker = ProgressTracker(str(path))

    tracker.set_last_image("first")
    tracker.set_completed("first", True)

    reloaded = ProgressTracker(str(path))
    assert reloaded.last_image_id == "first"
    assert reloaded.is_completed("first")


def test_prune_resets_unknown_last_image(tmp_path):
    tracker = ProgressTracker(str(tmp_path / "progress.json"))
    tracker.set_last_image("stale")
    tracker.prune(["fresh"])

    assert tracker.last_image_id is None
