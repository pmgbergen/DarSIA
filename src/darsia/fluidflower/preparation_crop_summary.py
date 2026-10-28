"""Show the actual prepared crop without creating a second geometry setting."""

from math import isfinite

from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget


class PreparationCropSummary(QWidget):
    def __init__(self, choice, preview, parent=None):
        super().__init__(parent)
        self.choice = choice
        self._series = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        note = QLabel(
            "The approved reference quadrilateral defines the alignment and, unless "
            "a separate crop area is chosen under Rig Geometry, the crop. "
            "The preview shows the prepared crop on its physical image grid. "
            "Lighting and ColorChecker corrections use the complete photograph before this crop."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.review = QPushButton("Review crop reference points")
        self.review.setToolTip(
            "Open the existing reference-point editor. Changing the geometry requires "
            "Apply / Build again; this page does not change a completed job."
        )
        self.review.clicked.connect(self._review)
        layout.addWidget(self.review)
        choice.context_updated.connect(self._context_updated)
        preview.preview_loaded.connect(self._loaded)
        self._context_updated()

    def _context_updated(self):
        current = (str(self.choice.series_root or ""), self.choice.series_id)
        if current != self._series:
            self._series = current
            self.summary.setText(
                "Crop dimensions will appear with the baseline preview from the completed geometry Build."
            )
        self.review.setEnabled(bool(current[0] and current[1]))

    def _loaded(self, result):
        grid = result.get("grid", {})
        try:
            height, width = (float(value) * 1000 for value in grid["dimensions"])
            x0, y1 = (float(value) * 1000 for value in grid["origin"])
            rows, cols = grid["num_voxels"]
            if not all(isfinite(v) for v in (height, width, x0, y1)) or min(height, width, rows, cols) <= 0:
                raise ValueError("Invalid physical crop")
        except (KeyError, TypeError, ValueError):
            self.summary.setText("The baseline preview does not report a valid physical crop.")
            return
        self.summary.setText(
            f"{str(result.get('run', '')).upper()} prepared crop: {width:g} × {height:g} mm.\n"
            f"x: {x0:g}–{x0 + width:g} mm; y: {y1 - height:g}–{y1:g} mm.\n"
            f"Image grid: {int(cols)} × {int(rows)} pixels."
        )

    def _review(self):
        select = getattr(self.window(), "_select_row", None)
        if callable(select):
            select(("geometry_facies", "rig_frame"))

