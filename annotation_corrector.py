import argparse
import glob
import os
import shutil
from typing import List

from PIL import Image

from inference import load_model, predict
from preprocessing import preprocess
from window import run_interface


def load_labels(label_file: str) -> List[str]:
    """Load YOLO label lines from a file.

    Args:
        label_file (str): Path to label file.

    Returns:
        List[str]: List of label lines.
    """
    if not os.path.exists(label_file):
        return []
    with open(label_file) as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="YOLO Annotation Corrector")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--labels", required=True, help="Path to original labels directory")
    parser.add_argument("--corrected", required=True, help="Directory to write corrected labels")
    parser.add_argument("--model", required=True, help="Path to YOLO model weights")
    parser.add_argument(
        "--predictions",
        action="store_true",
        help="Use cached predictions from a 'predicted_labels' folder",
    )
    args = parser.parse_args()

    os.makedirs(args.corrected, exist_ok=True)
    for src in glob.glob(os.path.join(args.labels, '*.txt')):
        dst = os.path.join(args.corrected, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    pred_dir = os.path.join(args.corrected, "predicted_labels")
    os.makedirs(pred_dir, exist_ok=True)

    model = None
    class_names: List[str] = []
    if not args.predictions:
        model = load_model(args.model)
        class_names = getattr(getattr(model, "model", None), "names", [])
    image_paths = sorted(glob.glob(os.path.join(args.images, '*')))

    images = []
    predictions = []
    labels = []
    label_files = []

    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
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
            pred_lines = predict(model, processed)
            with open(pred_file, "w") as f:
                for line, conf in pred_lines:
                    f.write(f"{line} {conf:.6f}\n")

        if set(line for line, _ in pred_lines) == set(label_lines):
            continue

        images.append(processed)
        predictions.append([{"line": line, "conf": conf, "accepted": False} for line, conf in pred_lines])
        labels.append([{"line": line, "kept": True} for line in label_lines])
        label_files.append(label_file)

    if images:
        run_interface(images, predictions, labels, label_files, class_names)


if __name__ == "__main__":
    main()
