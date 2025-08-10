import argparse
import glob
import os
import shutil
from typing import List

from PIL import Image

from inference import load_model, predict
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
        pred_lines = predict(model, processed)
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_file = os.path.join(args.corrected, base + '.txt')
        label_lines = load_labels(label_file)

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
