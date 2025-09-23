import argparse
import glob
import json
import logging
import os
import shutil
from typing import List, Optional, Dict, Any

from PIL import Image
from tqdm import tqdm

try:  # Optional dependency until inference is requested
    from inference import load_model, predict
except ImportError as exc:  # pragma: no cover - handled at runtime if missing
    load_model = None  # type: ignore[assignment]
    predict = None  # type: ignore[assignment]
    _INFERENCE_IMPORT_ERROR = exc
else:
    _INFERENCE_IMPORT_ERROR = None

from preprocessing import preprocess
from progress import ProgressTracker
from sam_integration import SamTightener
from setup_dialog import prompt_for_configuration, CONFIG_FILENAME
from window import run_interface

try:  # Optional dependency until UI is requested
    from PyQt6.QtWidgets import QApplication
except ImportError:  # pragma: no cover - PyQt should be available for the UI
    QApplication = None  # type: ignore[assignment]



logging.basicConfig(level=logging.INFO)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}



def load_labels(label_file: str) -> List[str]:
    """Load YOLO label lines from a file.

    Args:
        label_file (str): Path to label file.

    Returns:
        List[str]: List of label lines.
    """
    if not os.path.exists(label_file):
        # Simply return an empty list when the label file is missing.
        # Previously a warning was logged here which interfered with the
        # tqdm progress bar output.  Suppressing the warning keeps the
        # progress bar intact while still handling the missing file
        # gracefully.
        return []
    try:
        with open(label_file) as f:
            return [line.strip() for line in f if line.strip()]
    except (OSError, UnicodeDecodeError) as e:
        logging.error("Failed to read label file %s: %s", label_file, e)
        return []


def _load_config_file(config_path: str) -> Optional[Dict[str, Any]]:
    """Load a JSON configuration file if it exists."""

    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logging.warning("Failed to load configuration %s: %s", config_path, exc)
        return None

    if not isinstance(data, dict):
        logging.warning("Configuration file %s must contain a JSON object", config_path)
        return None

    return data


def _apply_config_to_args(args: argparse.Namespace, config: Dict[str, Any]) -> argparse.Namespace:
    """Populate missing command line arguments from a configuration dictionary."""

    for key in ("images", "labels", "corrected", "model", "sam"):
        value = config.get(key)
        if value and not getattr(args, key, None):
            setattr(args, key, value)

    if not args.predictions and "predictions" in config:
        args.predictions = bool(config.get("predictions"))
    if not getattr(args, "sahi", False) and "sahi" in config:
        args.sahi = bool(config.get("sahi"))

    return args


def gather_configuration_from_ui(
    args: argparse.Namespace,
) -> Optional[argparse.Namespace]:
    """Open the setup dialog to collect configuration values."""

    if QApplication is None:
        logging.error(
            "PyQt6 is required to display the configuration window but is not installed."
        )
        return None

    _app = QApplication.instance() or QApplication([])
    config = prompt_for_configuration(
        {
            "images": args.images or "",
            "labels": args.labels or "",
            "corrected": args.corrected or "",
            "model": args.model or "",
            "sam": args.sam or "",
            "predictions": args.predictions,
            "sahi": bool(getattr(args, "sahi", False)),
        }
    )
    if not config:
        return None

    args.images = config["images"]
    args.labels = config["labels"]
    args.corrected = config["corrected"]
    args.model = config["model"]
    args.predictions = bool(config.get("predictions"))
    args.sam = config.get("sam") or None
    args.sahi = bool(config.get("sahi"))
    return args


def main():
    parser = argparse.ArgumentParser(description="YOLO Annotation Corrector")
    parser.add_argument("--images", help="Path to images directory")
    parser.add_argument("--labels", help="Path to original labels directory")
    parser.add_argument("--corrected", help="Directory to write corrected labels")
    parser.add_argument("--model", help="Path to YOLO model weights")
    parser.add_argument(
        "--predictions",
        action="store_true",
        help="Use cached predictions from a 'predicted_labels' folder",
    )
    parser.add_argument(
        "--sahi",
        action="store_true",
        help="Use SAHI sliding window inference instead of direct Ultralytics predictions",
    )
    parser.add_argument(
        "--sam",
        help="Path to a SAM or SAM2 model for tightening bounding boxes",
    )
    args = parser.parse_args()

    if args.images:
        config_path = os.path.join(args.images, CONFIG_FILENAME)
        loaded_config = _load_config_file(config_path)
        if loaded_config:
            args = _apply_config_to_args(args, loaded_config)

    if not all([args.images, args.labels, args.corrected, args.model]):
        gathered = gather_configuration_from_ui(args)
        if not gathered:
            logging.info("No configuration selected; exiting.")
            return
        args = gathered

    # Persist the final configuration alongside the images if desired
    if args.images:
        config_path = os.path.join(args.images, CONFIG_FILENAME)
        data = {
            "images": args.images,
            "labels": args.labels,
            "corrected": args.corrected,
            "model": args.model,
            "sam": args.sam or "",
            "predictions": args.predictions,
            "sahi": args.sahi,
        }
        try:
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except OSError as exc:
            logging.warning("Unable to update configuration file %s: %s", config_path, exc)

    os.makedirs(args.corrected, exist_ok=True)
    for src in glob.glob(os.path.join(args.labels, '*.txt')):
        dst = os.path.join(args.corrected, os.path.basename(src))
        if not os.path.exists(dst):
            try:
                shutil.copy(src, dst)
            except OSError as e:
                logging.error("Failed to copy %s to %s: %s", src, dst, e)

    pred_dir = os.path.join(args.corrected, "predicted_labels")
    os.makedirs(pred_dir, exist_ok=True)

    if load_model is None:
        raise ImportError(
            "Inference dependencies are not available; install optional requirements to run predictions."
        ) from _INFERENCE_IMPORT_ERROR

    model = load_model(args.model, use_sahi=args.sahi)
    class_names: List[str] = []
    if model is not None:
        names_source = getattr(model, "model", None)
        potential_names = None
        if names_source is not None:
            potential_names = getattr(names_source, "names", None)
        if not potential_names:
            potential_names = getattr(model, "names", None)

        if isinstance(potential_names, dict):
            class_names = [str(name) for name in potential_names.values()]
        elif isinstance(potential_names, (list, tuple)):
            class_names = [str(name) for name in potential_names]
        elif potential_names:
            class_names = [str(potential_names)]
    sam_helper = None
    if args.sam:
        if not os.path.isfile(args.sam):
            raise FileNotFoundError(f"SAM weights file '{args.sam}' does not exist.")
        sam_helper = SamTightener(args.sam)
    if args.predictions:
        # When using cached predictions we still need the class names but can
        # discard the model to avoid unnecessary memory usage.
        model = None
    image_paths = sorted(
        p
        for p in glob.glob(os.path.join(args.images, '*'))
        if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
    )

    images = []
    predictions = []
    labels = []
    label_files = []
    image_ids = []

    progress_file = os.path.join(args.corrected, "annotation_progress.json")
    progress_tracker = ProgressTracker(progress_file)


    for img_path in tqdm(image_paths, desc="Processing images"):

        try:
          image = Image.open(img_path).convert('RGB')
        except (OSError, ValueError) as e:
          logging.error("Failed to open image %s: %s", img_path, e)
          continue
        processed = preprocess(image)
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(args.corrected, base + '.txt')
        label_lines = load_labels(label_file)
        pred_file = os.path.join(pred_dir, base + '.txt')
    
        if args.predictions:
            pred_lines: List[tuple[str, float]] = []
            if os.path.exists(pred_file):
                with open(pred_file) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        conf = float(parts[5]) if len(parts) >= 6 else 0.0
                        pred_lines.append((" ".join(parts[:5]), conf))
        else:
            if predict is None:
                raise ImportError(
                    "Prediction function unavailable; install optional dependencies or use cached predictions."
                ) from _INFERENCE_IMPORT_ERROR
            pred_lines = predict(model, processed, use_sahi=args.sahi)
            with open(pred_file, "w") as f:
                for line, conf in pred_lines:
                    f.write(f"{line} {conf:.6f}\n")
    
        if set(line for line, _ in pred_lines) == set(label_lines):
            continue
    
        images.append(processed)
        predictions.append([{"line": line, "conf": conf, "accepted": False} for line, conf in pred_lines])
        labels.append([{"line": line, "kept": True} for line in label_lines])
        label_files.append(label_file)
        image_ids.append(base)

    if images:
        progress_tracker.prune(image_ids)
        completed_flags = [progress_tracker.is_completed(img_id) for img_id in image_ids]
        start_index = 0
        for idx, done in enumerate(completed_flags):
            if not done:
                start_index = idx
                break
        else:
            start_index = len(completed_flags) - 1 if completed_flags else 0

        run_interface(
            images,
            predictions,
            labels,
            label_files,
            class_names,
            image_ids=image_ids,
            progress_tracker=progress_tracker,
            completed=completed_flags,
            start_index=start_index,
            sam_helper=sam_helper,
        )


if __name__ == "__main__":
    main()
