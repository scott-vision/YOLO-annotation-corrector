"""Configuration dialog for selecting application paths."""

from __future__ import annotations

import os
from typing import Dict, Optional

from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
)


class AnnotationSetupDialog(QDialog):
    """Dialog prompting the user to choose input and output locations."""

    def __init__(self, initial: Optional[Dict[str, str]] = None) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Annotation Corrector Setup")
        self.setModal(True)

        initial = initial or {}

        self.images_edit = QLineEdit(initial.get("images", ""))
        self.labels_edit = QLineEdit(initial.get("labels", ""))
        self.corrected_edit = QLineEdit(initial.get("corrected", ""))
        self.model_edit = QLineEdit(initial.get("model", ""))
        self.sam_edit = QLineEdit(initial.get("sam", ""))

        self.images_button = QPushButton("Browse…")
        self.images_button.clicked.connect(self._browse_images)

        self.labels_button = QPushButton("Browse…")
        self.labels_button.clicked.connect(self._browse_labels)

        self.corrected_button = QPushButton("Browse…")
        self.corrected_button.clicked.connect(self._browse_corrected)

        self.model_button = QPushButton("Browse…")
        self.model_button.clicked.connect(self._browse_model)

        self.sam_button = QPushButton("Browse…")
        self.sam_button.clicked.connect(self._browse_sam)

        self.predictions_checkbox = QCheckBox(
            "Use cached predictions if available"
        )
        self.predictions_checkbox.setChecked(bool(initial.get("predictions")))

        layout = QGridLayout(self)
        layout.addWidget(QLabel("Images folder"), 0, 0)
        layout.addWidget(self.images_edit, 0, 1)
        layout.addWidget(self.images_button, 0, 2)

        layout.addWidget(QLabel("Original labels"), 1, 0)
        layout.addWidget(self.labels_edit, 1, 1)
        layout.addWidget(self.labels_button, 1, 2)

        layout.addWidget(QLabel("Corrected labels"), 2, 0)
        layout.addWidget(self.corrected_edit, 2, 1)
        layout.addWidget(self.corrected_button, 2, 2)

        layout.addWidget(QLabel("Model weights"), 3, 0)
        layout.addWidget(self.model_edit, 3, 1)
        layout.addWidget(self.model_button, 3, 2)

        layout.addWidget(QLabel("SAM weights (optional)"), 4, 0)
        layout.addWidget(self.sam_edit, 4, 1)
        layout.addWidget(self.sam_button, 4, 2)

        layout.addWidget(self.predictions_checkbox, 5, 0, 1, 3)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel
            | QDialogButtonBox.StandardButton.Ok
        )
        self.button_box.accepted.connect(self._accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box, 6, 0, 1, 3)

        for edit in (
            self.images_edit,
            self.labels_edit,
            self.corrected_edit,
            self.model_edit,
        ):
            edit.textChanged.connect(self._update_accept_enabled)

        self._result: Optional[Dict[str, str]] = None
        self._update_accept_enabled()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def result(self) -> Optional[Dict[str, str]]:
        """Return the collected configuration values if the dialog was accepted."""

        return self._result

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------
    def _browse_images(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select images folder",
            self.images_edit.text() or os.getcwd(),
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.images_edit.setText(directory)

    def _browse_labels(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select original labels folder",
            self.labels_edit.text() or os.getcwd(),
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.labels_edit.setText(directory)

    def _browse_corrected(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select corrected labels folder",
            self.corrected_edit.text() or os.getcwd(),
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.corrected_edit.setText(directory)

    def _browse_model(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select YOLO model weights",
            self.model_edit.text() or os.getcwd(),
            "PyTorch models (*.pt *.pth);;All files (*)",
        )
        if file_path:
            self.model_edit.setText(file_path)

    def _browse_sam(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM weights",
            self.sam_edit.text() or os.getcwd(),
            "PyTorch models (*.pt *.pth);;All files (*)",
        )
        if file_path:
            self.sam_edit.setText(file_path)

    def _update_accept_enabled(self) -> None:
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is None:
            return
        ok_button.setEnabled(
            all(
                field.text().strip()
                for field in (
                    self.images_edit,
                    self.labels_edit,
                    self.corrected_edit,
                    self.model_edit,
                )
            )
        )

    # ------------------------------------------------------------------
    # Validation and acceptance
    # ------------------------------------------------------------------
    def _accept(self) -> None:
        images = self.images_edit.text().strip()
        labels = self.labels_edit.text().strip()
        corrected = self.corrected_edit.text().strip()
        model = self.model_edit.text().strip()
        sam = self.sam_edit.text().strip()

        if not os.path.isdir(images):
            self._show_error("Images folder does not exist.")
            return
        if not os.path.isdir(labels):
            self._show_error("Original labels folder does not exist.")
            return
        if corrected and not os.path.exists(corrected):
            create = QMessageBox.question(
                self,
                "Create folder?",
                (
                    "The corrected labels folder does not exist.\n"
                    "Create it before continuing?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if create == QMessageBox.StandardButton.Yes:
                try:
                    os.makedirs(corrected, exist_ok=True)
                except OSError as exc:  # pragma: no cover - unlikely path
                    self._show_error(f"Failed to create folder: {exc}")
                    return
            else:
                return
        if not os.path.isfile(model):
            self._show_error("Model weights file does not exist.")
            return
        if sam and not os.path.isfile(sam):
            self._show_error("SAM weights file does not exist.")
            return

        self._result = {
            "images": images,
            "labels": labels,
            "corrected": corrected,
            "model": model,
            "sam": sam,
            "predictions": self.predictions_checkbox.isChecked(),
        }
        self.accept()

    def _show_error(self, message: str) -> None:
        QMessageBox.warning(self, "Invalid configuration", message, QMessageBox.StandardButton.Ok)


def prompt_for_configuration(initial: Optional[Dict[str, str]] = None) -> Optional[Dict[str, str]]:
    """Display the setup dialog and return the chosen configuration."""

    dialog = AnnotationSetupDialog(initial)
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return dialog.result
    return None

