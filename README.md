# YOLO Annotation Corrector

Interactive tool for reviewing YOLO annotations. It runs an Ultralytics YOLO model on a set of images and compares the predictions with existing label files. Only images where the predictions and labels differ are presented to the user for review.

## Usage

```
python annotation_corrector.py --images path/to/images --labels path/to/labels --model path/to/weights.pt
```

Model predictions are drawn in **red** and existing labels in **green**. For each disagreement you can:

* **Accept** – overwrite labels with model predictions.
* **Reject** – keep existing labels.
* **Change** – enter new label lines in YOLO format.

Image preprocessing is defined in `preprocessing.py` and currently returns images unchanged.
