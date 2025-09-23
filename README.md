# YOLO Annotation Corrector

Interactive tool for reviewing YOLO annotations. It runs an Ultralytics YOLO model on a set of images and compares the predictions with existing label files. Only images where the predictions and labels differ are presented to the user for review.

The application provides a single PyQt6 interface that lets you work through all images without reopening the window. Model predictions (in red) display a green "+" to add them and a red "-" to remove them once accepted. Existing labels (in green) show a red "-" for removal and switch to a green "+" to add them back. A preview window shows the final label lines before saving.

## Usage

```
python annotation_corrector.py --images path/to/images --labels path/to/original_labels --corrected path/to/corrected_labels --model path/to/weights.pt
```

If you prefer to avoid the command line, simply run the script without any arguments. A setup window will appear that lets you
select the required folders, YOLO model weights, and optional SAM/SAM2 weights interactively. The "Use cached predictions"
checkbox mirrors the `--predictions` flag described below.

Predictions are cached in a `predicted_labels` subdirectory inside the corrected labels folder. On subsequent runs you can skip
recomputing predictions by adding the `--predictions` flag (or enabling the checkbox in the setup window):

```
python annotation_corrector.py --images path/to/images --labels path/to/original_labels --corrected path/to/corrected_labels --model path/to/weights.pt --predictions --sam path/to/sam_weights.pt
```

The script copies all label files from the `--labels` directory into the `--corrected` directory. Any accepted or edited labels are written to the corrected directory, leaving the originals untouched. Within the GUI you may:

* Toggle prediction boxes on or off.
* Click the **+** on a prediction to include it, or **-** to remove it.
* Click the **-** on a ground-truth box to remove it, or **+** to add it back.
* Adjust image display using brightness and contrast sliders.
* Move between images with **Previous** and **Next**.
* Press **Save** to write labels for all images and **Exit** to close the tool.
* Use **Preview** to view the final labels for the current image.
* Click **SAM2 tighten** to refine every box in the current image using the loaded SAM/SAM2 model.
* Click **Single SAM2** to select a single tightening point; the next click inside the image runs SAM2 at that point and shrinks any containing boxes to the returned mask box.
* Use **Revert SAM2** to undo the most recent tightening adjustment for the current image.

Image preprocessing is defined in `preprocessing.py` and currently returns images unchanged.
