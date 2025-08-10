# YOLO Annotation Corrector

Interactive tool for reviewing YOLO annotations. It runs an Ultralytics YOLO model on a set of images and compares the predictions with existing label files. Only images where the predictions and labels differ are presented to the user for review.

The application now provides a PyQt6 interface. Model predictions (in red) can be toggled on and off, each showing its confidence and a check mark for accepting the prediction. Existing labels (in green) display a cross that can be clicked to remove them. A preview window shows the final label lines before saving.

## Usage

```
python annotation_corrector.py --images path/to/images --labels path/to/original_labels --corrected path/to/corrected_labels --model path/to/weights.pt
```

The script copies all label files from the `--labels` directory into the `--corrected` directory. Any accepted or edited labels are written to the corrected directory, leaving the originals untouched. Within the GUI you may:

* Toggle prediction boxes on or off.
* Click the **✓** on a prediction to include it.
* Click the **✗** on a ground-truth box to remove it.
* Use **Preview** to view the final labels before saving.

Image preprocessing is defined in `preprocessing.py` and currently returns images unchanged.
