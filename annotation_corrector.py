import argparse
import glob
import os
import shutil
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from ultralytics import YOLO

from preprocessing import preprocess


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


def format_predictions(boxes) -> List[str]:
    """Format model predictions as YOLO label strings."""
    lines = []
    for b in boxes:
        cls = int(b.cls.item())
        xc, yc, w, h = b.xywhn.tolist()[0]
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def draw_boxes(ax, lines: List[str], img_w: int, img_h: int, color: str):
    """Draw YOLO-format boxes on a matplotlib axis."""
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = parts
        xc, yc, w, h = map(float, (xc, yc, w, h))
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        rect = patches.Rectangle((x1, y1), w * img_w, h * img_h,
                                 linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, cls, color=color)


def show_interface(image: Image.Image, pred_lines: List[str], label_lines: List[str], label_file: str):
    """Display image with predictions and labels and handle user input."""
    img_w, img_h = image.size
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    draw_boxes(ax, pred_lines, img_w, img_h, 'r')
    draw_boxes(ax, label_lines, img_w, img_h, 'g')
    plt.title("Red: predictions, Green: labels")
    plt.show(block=False)

    action = input("Accept predictions (a), Reject (r), Change (c)? ").strip().lower()
    plt.close(fig)

    if action == 'a':
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with open(label_file, 'w') as f:
            for line in pred_lines:
                f.write(line + '\n')
    elif action == 'c':
        new_lines = input("Enter new labels in YOLO format separated by semicolons: ").strip()
        lines = [l.strip() for l in new_lines.split(';') if l.strip()]
        os.makedirs(os.path.dirname(label_file), exist_ok=True)
        with open(label_file, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    # else: reject - keep existing labels


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

        if set(pred_lines) == set(label_lines):
            continue

        show_interface(processed, pred_lines, label_lines, label_file)


if __name__ == "__main__":
    main()
