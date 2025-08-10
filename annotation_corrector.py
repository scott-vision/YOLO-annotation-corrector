import argparse
import glob
import os
import shutil
from typing import List, Tuple

import numpy as np
from PIL import Image
from ultralytics import YOLO

from preprocessing import preprocess
from qt_interface import run_interface


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


def format_predictions(boxes) -> List[Tuple[str, float]]:
    """Format model predictions as YOLO label strings with confidences."""
    lines: List[Tuple[str, float]] = []
    for b in boxes:
        cls = int(b.cls.item())
        xc, yc, w, h = b.xywhn.tolist()[0]
        conf = float(b.conf.item())
        lines.append((f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}", conf))
    return lines


def main():
    parser = argparse.ArgumentParser(description="YOLO Annotation Corrector")
    parser.add_argument("--images", required=True, help="Path to images directory")
    parser.add_argument("--labels", required=True, help="Path to original labels directory")
    parser.add_argument("--corrected", required=True, help="Directory to write corrected labels")
    parser.add_argument("--model", required=True, help="Path to YOLO model weights")
    args = parser.parse_args()

    os.makedirs(args.corrected, exist_ok=True)
    for src in glob.glob(os.path.join(args.labels, '*.txt')):
        dst = os.path.join(args.corrected, os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    model = YOLO(args.model)
    image_paths = sorted(glob.glob(os.path.join(args.images, '*')))

    for img_path in image_paths:
        image = Image.open(img_path).convert('RGB')
        processed = preprocess(image)
        results = model(np.array(processed))
        boxes = results[0].boxes

        pred_lines = format_predictions(boxes)
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(args.corrected, base + '.txt')
        label_lines = load_labels(label_file)

        if set(line for line, _ in pred_lines) == set(label_lines):
            continue

        run_interface(processed, pred_lines, label_lines, label_file)


if __name__ == "__main__":
    main()
