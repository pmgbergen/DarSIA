"""Qt image-correction workflow page."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QPointF, QSize, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QImage, QImageReader, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.frame_quality import FrameQualityStore
from darsia.fluidflower.image_correction import (
    COLORCHECKER_ALGORITHMS,
    DARSIA_BASELINE_COLOR_ALGORITHM,
    IDENTITY_ALGORITHM,
    SWATCH_NAMES,
    UNIFORM24_ALGORITHM,
    CorrectionRevisionStore,
    calculate_color_correction_comparison_preview,
    calculate_identity_preview,
    darsia_baseline_correction_config,
    default_correction_config,
    discover_correction_frames,
    discover_project_correction_frames,
    identity_correction_config,
)
from darsia.fluidflower.image_formats import select_preferred_images
from darsia.fluidflower.pages.image_correction.frame_quality_panel import (
    FrameQualityPanel,
)
from darsia.fluidflower.pages.image_correction.selection_dialog import (
    ImageSelectionPlanDialog,
)
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.ui.common import (
    ModulePage,
    PathField,
    _number,
    deepcopy_settings,
)


class ZoomableImageView(QGraphicsView):
    zoom_requested = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.roi_item = QGraphicsPolygonItem()
        roi_pen = QPen(QColor("#ff3b30"))
        roi_pen.setWidthF(2.0)
        roi_pen.setCosmetic(True)
        self.roi_item.setPen(roi_pen)
        self.roi_item.setBrush(Qt.BrushStyle.NoBrush)
        self.roi_item.setZValue(10.0)
        self.roi_item.setVisible(False)
        self.scene().addItem(self.roi_item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#0b0d0f"))
        self.setMinimumSize(360, 260)
        self._has_image = False

    def set_array(self, image: np.ndarray) -> None:
        self.clear_roi()
        rgb = np.clip(np.asarray(image), 0.0, 1.0)
        rgb8 = np.ascontiguousarray(np.rint(rgb * 255.0).astype(np.uint8))
        height, width = rgb8.shape[:2]
        qimage = QImage(
            rgb8.data,
            width,
            height,
            int(rgb8.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(self.pixmap_item.boundingRect())
        self._has_image = True
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_image(self, image: QImage) -> None:
        """Display an already decoded image without a NumPy round trip."""

        self.clear_roi()
        self.pixmap_item.setPixmap(QPixmap.fromImage(image))
        self.scene().setSceneRect(self.pixmap_item.boundingRect())
        self._has_image = not image.isNull()
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear_roi(self) -> None:
        self.roi_item.setVisible(False)

    def set_roi(self, roi_rc: Any, *, source_shape: Any) -> None:
        """Outline a full-resolution row/column ROI on the displayed preview."""

        roi = np.asarray(roi_rc, dtype=np.float64)
        shape = tuple(int(value) for value in source_shape)
        pixmap = self.pixmap_item.pixmap()
        if roi.shape != (4, 2) or len(shape) < 2 or pixmap.isNull():
            self.clear_roi()
            return
        source_height, source_width = shape[:2]
        if source_height <= 0 or source_width <= 0:
            self.clear_roi()
            return
        scale_x = pixmap.width() / float(source_width)
        scale_y = pixmap.height() / float(source_height)
        polygon = QPolygonF(
            [QPointF(float(col) * scale_x, float(row) * scale_y) for row, col in roi]
        )
        self.roi_item.setPolygon(polygon)
        self.roi_item.setVisible(True)

    def wheelEvent(self, event: Any) -> None:
        if not self._has_image:
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.scale(factor, factor)
        self.zoom_requested.emit(factor)

    @Slot(float)
    def apply_linked_zoom(self, factor: float) -> None:
        if self._has_image:
            self.scale(float(factor), float(factor))


class SourceImagePreviewDialog(QDialog):
    """Read-only, bounded-resolution preview for one registered source image."""

    MAX_PREVIEW_SIZE = QSize(1600, 1000)

    def __init__(self, path: Path, *, role: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"{role} preview")
        self.resize(1100, 760)

        layout = QVBoxLayout(self)
        heading = QLabel(f"{role}: {path.name}")
        heading.setObjectName("sectionTitle")
        layout.addWidget(heading)
        location = QLabel(str(path))
        location.setObjectName("secondaryText")
        location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        location.setWordWrap(True)
        layout.addWidget(location)

        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        source_size = reader.size()
        if source_size.isValid() and (
            source_size.width() > self.MAX_PREVIEW_SIZE.width()
            or source_size.height() > self.MAX_PREVIEW_SIZE.height()
        ):
            reader.setScaledSize(
                source_size.scaled(
                    self.MAX_PREVIEW_SIZE,
                    Qt.AspectRatioMode.KeepAspectRatio,
                )
            )
        image = reader.read()
        if image.isNull():
            raise ValueError(reader.errorString() or "The image could not be decoded")

        view = ZoomableImageView()
        view.set_image(image)
        layout.addWidget(view, 1)
        dimensions = QLabel(
            f"Source: {source_size.width()} x {source_size.height()} px"
            if source_size.isValid()
            else f"Preview: {image.width()} x {image.height()} px"
        )
        dimensions.setObjectName("secondaryText")
        layout.addWidget(dimensions)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class CorrectionWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        algorithm: str,
        reference: Path | None,
        target: Path,
        corner: str,
        roi: Any,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.algorithm = algorithm
        self.reference = reference
        self.target = target
        self.corner = corner
        self.roi = roi
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            if self.reference is not None:
                variants = calculate_color_correction_comparison_preview(
                    self.reference,
                    self.target,
                    corner=self.corner,
                    roi_rc=self.roi,
                    settings=self.settings,
                )
                result = dict(variants[self.algorithm])
                result["comparison_variants"] = variants
            elif self.algorithm == IDENTITY_ALGORITHM:
                result = calculate_identity_preview(self.target)
                result["comparison_variants"] = {IDENTITY_ALGORITHM: result.copy()}
            else:
                raise ValueError("ColorChecker reference image is required")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(result)


class CorrectionInventoryWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        source_root: Path,
        *,
        recursive: bool,
        run_mapping: str,
        flat_run_id: str,
        project: FluidFlowerProject | None = None,
        use_project_mapping: bool = False,
    ) -> None:
        super().__init__()
        self.source_root = Path(source_root)
        self.recursive = recursive
        self.run_mapping = run_mapping
        self.flat_run_id = flat_run_id
        self.project = project
        self.use_project_mapping = use_project_mapping

    @Slot()
    def run(self) -> None:
        try:
            if self.use_project_mapping:
                if self.project is None:
                    raise RuntimeError("The registered project mapping is unavailable")
                frames = discover_project_correction_frames(self.project)
            else:
                frames = discover_correction_frames(
                    self.source_root,
                    recursive=self.recursive,
                    run_mapping=self.run_mapping,
                    flat_run_id=self.flat_run_id,
                )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.finished.emit(frames)


class ImageCorrectionPage(ModulePage):
    def __init__(self, project: FluidFlowerProject) -> None:
        super().__init__(project, "image_correction")
        self.store = CorrectionRevisionStore(project)
        self.preview_result: dict[str, Any] | None = None
        self._frame_option_cache: dict[str, list[dict[str, str]]] = {}
        self.batch_frames: list[dict[str, Any]] = []
        self._preview_roi_source = ""
        self._thread: QThread | None = None
        self._worker: CorrectionWorker | None = None
        self._inventory_thread: QThread | None = None
        self._inventory_worker: CorrectionInventoryWorker | None = None
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "Light and color correction makes the optical response comparable between "
            "registered frames before color paths, phase interpretation, and chemistry "
            "calibration are built. Compare the original and corrected image before saving "
            "a revision. Raw source, original DarSIA ColorCorrection, and improved "
            "Uniform24 can be compared pairwise on the same synchronized frame. The "
            "current ColorChecker methods estimate one spatially uniform "
            "transform per frame; it does not model a spatial lighting gradient. Saving a "
            "changed correction marks "
            "every dependent interpretation and result as outdated."
        )
        scope_group = QFrame()
        scope_group.setFrameShape(QFrame.Shape.StyledPanel)
        scope_layout = QGridLayout(scope_group)
        scope_title = QLabel("Series-wide correction profile")
        scope_title.setObjectName("sectionTitle")
        scope_layout.addWidget(scope_title, 0, 0, 1, 4)
        self.scope_explanation = QLabel(
            "Correction model, chart position, numerical settings, common reference "
            "appearance, and output format apply to every included experiment. A separate "
            "transform is still estimated for each frame. Select one representative experiment "
            "to supply the reference chart appearance and preview frames. This selection "
            "does not limit which experiments are corrected."
        )
        self.scope_explanation.setObjectName("secondaryText")
        self.scope_explanation.setWordWrap(True)
        scope_layout.addWidget(self.scope_explanation, 1, 0, 1, 4)
        scope_layout.addWidget(QLabel("Representative experiment"), 2, 0)
        self.representative_run = QComboBox()
        for row in project.manifest.get("runs") or []:
            if not isinstance(row, Mapping):
                continue
            run_id = str(row.get("run_id") or "").strip().casefold()
            if run_id:
                self.representative_run.addItem(
                    str(row.get("label") or run_id.upper()), run_id
                )
        self.representative_run.setToolTip(
            "Choose the experiment used to configure and visually inspect the shared "
            "correction procedure. All included experiments retain their registered "
            "image mappings."
        )
        scope_layout.addWidget(self.representative_run, 2, 1)
        scope_layout.addWidget(QLabel("Preview frame"), 2, 2)
        self.representative_frame = QComboBox()
        self.representative_frame.setToolTip(
            "Choose a registered frame from the representative experiment. The baseline "
            "supplies the reference ColorChecker appearance; this frame previews how the "
            "shared procedure behaves later in the experiment."
        )
        scope_layout.addWidget(self.representative_frame, 2, 3)
        self.representative_summary = QLabel()
        self.representative_summary.setObjectName("secondaryText")
        self.representative_summary.setWordWrap(True)
        scope_layout.addWidget(self.representative_summary, 3, 0, 1, 4)
        self.work_layout.addWidget(scope_group)
        image_plan_row = QHBoxLayout()
        self.image_selection_button = QPushButton("Plan images for review, calibration and production")
        self.image_selection_button.setToolTip(
            "Edit the series' versioned image-selection TOML and preview a native frame "
            "manifest without decoding images or starting a calculation."
        )
        self.image_selection_button.clicked.connect(self._open_image_selection_plan)
        image_plan_row.addWidget(self.image_selection_button)
        image_plan_row.addStretch(1)
        self.work_layout.addLayout(image_plan_row)
        settings_row = QGridLayout()
        self.reference_mode = QComboBox()
        self.reference_mode.addItem(
            "No radiometric correction (identity control)", IDENTITY_ALGORITHM
        )
        self.reference_mode.addItem(
            "Original DarSIA ColorCorrection (legacy)",
            DARSIA_BASELINE_COLOR_ALGORITHM,
        )
        self.reference_mode.addItem(
            "Improved Uniform24 correction (24 patches)", UNIFORM24_ALGORITHM
        )
        self.reference_mode.setToolTip(
            "Select how camera and lighting variation is corrected before optical phase "
            "interpretation. Additional validated correction models can be added here "
            "without changing source import."
        )
        self.correction_method_explanation = QLabel()
        self.correction_method_explanation.setObjectName("secondaryText")
        self.correction_method_explanation.setWordWrap(True)
        self.reference = PathField(
            directory=False,
            file_filter="Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All files (*)",
            placeholder="Frame defining the target ColorChecker appearance",
        )
        self.reference.setToolTip(
            "This is the automatically resolved or manually selected frame whose measured "
            "ColorChecker patches define the target appearance. Every inspected frame is "
            "corrected toward these patch colors."
        )
        self.target = PathField(
            directory=False,
            file_filter="Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All files (*)",
            placeholder="Frame to inspect",
        )
        self.manual_preview_sources = QCheckBox(
            "Override registered reference and preview files"
        )
        self.manual_preview_sources.setToolTip(
            "Normally leave this off. Enable it only to inspect files that are not the "
            "registered baseline or preview choices for the representative experiment."
        )
        settings_row.addWidget(QLabel("Correction model"), 0, 0)
        settings_row.addWidget(self.reference_mode, 0, 1, 1, 5)
        settings_row.addWidget(self.correction_method_explanation, 1, 1, 1, 5)
        reference_label = QLabel("Resolved color reference")
        reference_label.setToolTip(
            "Resolved means that DarSIA selected this file from the representative "
            "experiment's registered source mapping."
        )
        settings_row.addWidget(reference_label, 2, 0)
        settings_row.addWidget(self.reference, 2, 1, 1, 4)
        self.reference_source_preview = QPushButton("Preview")
        self.reference_source_preview.setMaximumWidth(88)
        self.reference_source_preview.setToolTip(
            "Open a quick read-only preview of the resolved ColorChecker reference image."
        )
        self.reference_source_preview.clicked.connect(
            lambda: self._show_source_preview(
                self.reference.text(), role="Color reference"
            )
        )
        settings_row.addWidget(self.reference_source_preview, 2, 5)
        target_label = QLabel("Resolved preview frame")
        target_label.setToolTip(
            "The one registered frame used to inspect the correction interactively. "
            "It does not limit or replace the frames processed during full execution."
        )
        settings_row.addWidget(target_label, 3, 0)
        settings_row.addWidget(self.target, 3, 1, 1, 4)
        self.target_source_preview = QPushButton("Preview")
        self.target_source_preview.setMaximumWidth(88)
        self.target_source_preview.setToolTip(
            "Open a quick read-only preview of the target frame selected for inspection."
        )
        self.target_source_preview.clicked.connect(
            lambda: self._show_source_preview(self.target.text(), role="Preview frame")
        )
        settings_row.addWidget(self.target_source_preview, 3, 5)
        settings_row.addWidget(self.manual_preview_sources, 4, 1, 1, 5)
        settings_row.addWidget(QLabel("Expected chart corner"), 5, 0)
        self.corner = QComboBox()
        for label, value in (
            ("Upper right", "upper_right"),
            ("Upper left", "upper_left"),
            ("Lower right", "lower_right"),
            ("Lower left", "lower_left"),
        ):
            self.corner.addItem(label, value)
        settings_row.addWidget(self.corner, 5, 1)
        self.neutral_tone = QCheckBox("Neutral tone")
        self.neutral_tone.setChecked(True)
        settings_row.addWidget(self.neutral_tone, 5, 2)
        self.robust_affine = QCheckBox("Robust 24-field affine")
        self.robust_affine.setChecked(True)
        settings_row.addWidget(self.robust_affine, 5, 3)
        self.preview_button = QPushButton("Calculate preview")
        self.preview_button.clicked.connect(self._start_preview)
        settings_row.addWidget(self.preview_button, 5, 4)
        self.save_revision = QPushButton("Save revision")
        self.save_revision.clicked.connect(self._save_revision)
        self.save_revision.setEnabled(False)
        settings_row.addWidget(self.save_revision, 5, 5)
        sequence_row = QHBoxLayout()
        self.previous_preview_frame = QPushButton("Previous frame")
        self.previous_preview_frame.setToolTip(
            "Move to the previous representative or stress-test frame and calculate it."
        )
        self.preview_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.preview_frame_slider.setMinimum(0)
        self.preview_frame_slider.setMaximum(0)
        self.preview_frame_slider.setToolTip(
            "Navigate the baseline, time-distributed frames, and detected lighting stress frames."
        )
        self.preview_frame_position = QLabel("0 / 0")
        self.next_preview_frame = QPushButton("Next frame")
        self.next_preview_frame.setToolTip(
            "Move to the next representative or stress-test frame and calculate it."
        )
        sequence_row.addWidget(self.previous_preview_frame)
        sequence_row.addWidget(self.preview_frame_slider, 1)
        sequence_row.addWidget(self.preview_frame_position)
        sequence_row.addWidget(self.next_preview_frame)
        settings_row.addLayout(sequence_row, 6, 1, 1, 5)
        self.work_layout.addLayout(settings_row)
        source_settings = dict(
            (project.manifest.get("tools") or {}).get("image_reference") or {}
        )
        source_location = str(
            source_settings.get("colorchecker_location") or "not registered"
        ).replace("_", " ")
        self.colorchecker_source_status = QLabel(
            f"Registered ColorChecker source: {source_location}. Source registration "
            "only records where the chart can be found; the correction model is selected here."
        )
        self.colorchecker_source_status.setWordWrap(True)
        self.colorchecker_source_status.setObjectName("secondaryText")
        self.work_layout.addWidget(self.colorchecker_source_status)
        reference_contract = dict(
            
                (project.manifest.get("chemistry") or {}).get(
                    "indicator_response_reference"
                )
                or {}
            
        )
        self.indicator_reference_status = QLabel(
            "Indicator references: "
            f"{int(reference_contract.get('entry_count') or 0)} image(s), "
            f"status {str(reference_contract.get('status') or 'not provided').replace('_', ' ')}. "
            "These references feed chemistry/phase calibration, not this light-correction transform."
        )
        self.indicator_reference_status.setWordWrap(True)
        self.indicator_reference_status.setObjectName("secondaryText")
        self.work_layout.addWidget(self.indicator_reference_status)
        batch_group = QFrame()
        batch_group.setFrameShape(QFrame.Shape.StyledPanel)
        batch_layout = QGridLayout(batch_group)
        batch_title = QLabel("Apply the profile to the registered series")
        batch_title.setObjectName("sectionTitle")
        batch_layout.addWidget(batch_title, 0, 0, 1, 6)
        self.batch_scope_explanation = QLabel(
            "DarSIA already knows the included experiments and their ordered CO2 image "
            "folders from Series setup. Build a content-addressed inventory when the "
            "profile is ready to execute; source files remain read-only."
        )
        self.batch_scope_explanation.setObjectName("secondaryText")
        self.batch_scope_explanation.setWordWrap(True)
        batch_layout.addWidget(self.batch_scope_explanation, 1, 0, 1, 6)
        self.batch_source = PathField(
            directory=True,
            placeholder="Folder containing one run or run subfolders",
        )
        self.batch_use_project_mapping = QCheckBox(
            "Use all included experiments from Series setup"
        )
        self.batch_use_project_mapping.setChecked(bool(project.manifest.get("runs")))
        self.batch_use_project_mapping.setToolTip(
            "Recommended for established projects. Only the CO2 image folders assigned "
            "to currently included experiments are inventoried; excluded sibling folders, "
            "overflow images, and superseded source revisions are not scanned."
        )
        self.batch_recursive = QCheckBox("Include subfolders")
        self.batch_recursive.setChecked(True)
        self.batch_run_mapping = QComboBox()
        self.batch_run_mapping.addItem("First subfolder is run ID", "parent_folder")
        self.batch_run_mapping.addItem("All images belong to one run", "flat")
        self.batch_flat_run = QLineEdit("run")
        self.batch_flat_run.setPlaceholderText("Run ID for a flat image folder")
        self.batch_encoding = QComboBox()
        self.batch_encoding.addItem("Lossless 16-bit PNG", "png16")
        self.batch_encoding.addItem("Float32 NumPy array", "npy-float32")
        self.batch_encoding.setToolTip(
            "Storage format used only when the full image-correction task is executed. "
            "Execution writes one corrected derivative for every inventoried frame; "
            "Calculate preview and Build series frame inventory do not. Source images "
            "are never modified."
        )
        self.batch_scan = QPushButton("Build series frame inventory")
        self.batch_scan.clicked.connect(self._start_inventory)
        self.batch_summary = QLabel("No immutable batch inventory has been created.")
        self.batch_summary.setObjectName("secondaryText")
        self.batch_summary.setWordWrap(True)
        batch_layout.addWidget(self.batch_use_project_mapping, 2, 0, 1, 3)
        self.batch_manual_label = QLabel("Manual image source override")
        batch_layout.addWidget(self.batch_manual_label, 3, 0)
        batch_layout.addWidget(self.batch_source, 3, 1, 1, 5)
        batch_layout.addWidget(self.batch_recursive, 4, 1)
        batch_layout.addWidget(self.batch_run_mapping, 4, 2)
        batch_layout.addWidget(self.batch_flat_run, 4, 3)
        corrected_output_label = QLabel("Corrected-frame storage")
        corrected_output_label.setToolTip(self.batch_encoding.toolTip())
        batch_layout.addWidget(corrected_output_label, 2, 3)
        batch_layout.addWidget(self.batch_encoding, 2, 4)
        batch_layout.addWidget(self.batch_scan, 2, 5)
        batch_layout.addWidget(self.batch_summary, 5, 0, 1, 6)
        self.work_layout.addWidget(batch_group)
        self.batch_source.changed.connect(self._invalidate_batch_inventory)
        self.batch_use_project_mapping.toggled.connect(
            self._batch_inventory_scope_changed
        )
        self.batch_recursive.toggled.connect(self._invalidate_batch_inventory)
        self.batch_run_mapping.currentIndexChanged.connect(
            self._invalidate_batch_inventory
        )
        self.batch_flat_run.textChanged.connect(self._invalidate_batch_inventory)
        self._batch_inventory_scope_changed()
        self.frame_quality = FrameQualityPanel(project)
        self.frame_quality.inspect_requested.connect(self._inspect_quality_frame)
        self.work_layout.addWidget(self.frame_quality)
        status_row = QHBoxLayout()
        self.correction_summary = QLabel("No preview calculated.")
        self.correction_summary.setObjectName("secondaryText")
        self.correction_summary.setWordWrap(True)
        status_row.addWidget(self.correction_summary, 1)
        self.left_preview_variant = QComboBox()
        self.right_preview_variant = QComboBox()
        for combo in (self.left_preview_variant, self.right_preview_variant):
            combo.addItem("Raw source", IDENTITY_ALGORITHM)
            combo.addItem(
                "Original DarSIA",
                DARSIA_BASELINE_COLOR_ALGORITHM,
            )
            combo.addItem("Improved Uniform24", UNIFORM24_ALGORITHM)
            combo.setToolTip(
                "Choose independently which version is shown on this side. Both views use "
                "the same target frame and synchronized zoom and pan."
            )
        self.right_preview_variant.setCurrentIndex(
            self.right_preview_variant.findData(UNIFORM24_ALGORITHM)
        )
        status_row.addWidget(QLabel("Left"))
        status_row.addWidget(self.left_preview_variant)
        status_row.addWidget(QLabel("Right"))
        status_row.addWidget(self.right_preview_variant)
        self.show_colorchecker_roi = QCheckBox("Show detected ColorChecker")
        self.show_colorchecker_roi.setChecked(True)
        self.show_colorchecker_roi.toggled.connect(self._update_checker_overlay)
        status_row.addWidget(self.show_colorchecker_roi)
        self.show_difference = QCheckBox("Amplify correction difference")
        self.show_difference.toggled.connect(self._update_right_image)
        status_row.addWidget(self.show_difference)
        self.swap_preview_sides = QPushButton("Swap image sides")
        self.swap_preview_sides.setToolTip(
            "Exchange the correction variants selected for the left and right previews."
        )
        self.swap_preview_sides.clicked.connect(self._swap_preview_sides)
        status_row.addWidget(self.swap_preview_sides)
        self.revisions = QComboBox()
        self.revisions.setMinimumWidth(260)
        status_row.addWidget(QLabel("Saved revision"))
        status_row.addWidget(self.revisions)
        self.activate_revision_button = QPushButton("Use selected revision")
        self.activate_revision_button.setToolTip(
            "Select an earlier immutable correction as the active analysis branch. "
            "Dependent color paths and analyses will require review or regeneration."
        )
        self.activate_revision_button.clicked.connect(self._activate_selected_revision)
        status_row.addWidget(self.activate_revision_button)
        self.work_layout.addLayout(status_row)
        self.colorchecker_detection_status = QLabel(
            "No ColorChecker detection has been calculated."
        )
        self.colorchecker_detection_status.setObjectName("secondaryText")
        self.colorchecker_detection_status.setWordWrap(True)
        self.work_layout.addWidget(self.colorchecker_detection_status)
        self.images_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.original_panel = QWidget()
        original_layout = QVBoxLayout(self.original_panel)
        original_layout.setContentsMargins(0, 0, 0, 0)
        self.original_title = QLabel("Raw source frame")
        self.original_title.setObjectName("sectionTitle")
        original_layout.addWidget(self.original_title)
        self.original_view = ZoomableImageView()
        original_layout.addWidget(self.original_view, 1)
        self.corrected_panel = QWidget()
        corrected_layout = QVBoxLayout(self.corrected_panel)
        corrected_layout.setContentsMargins(0, 0, 0, 0)
        self.corrected_title = QLabel("Uniform24 corrected frame")
        self.corrected_title.setObjectName("sectionTitle")
        corrected_layout.addWidget(self.corrected_title)
        self.corrected_view = ZoomableImageView()
        corrected_layout.addWidget(self.corrected_view, 1)
        self.images_splitter.addWidget(self.original_panel)
        self.images_splitter.addWidget(self.corrected_panel)
        self.images_splitter.setStretchFactor(0, 1)
        self.images_splitter.setStretchFactor(1, 1)
        self.original_view.zoom_requested.connect(
            self.corrected_view.apply_linked_zoom
        )
        self.corrected_view.zoom_requested.connect(
            self.original_view.apply_linked_zoom
        )
        self._link_scrollbars(self.original_view, self.corrected_view)
        self._link_scrollbars(self.corrected_view, self.original_view)
        self.work_layout.addWidget(self.images_splitter, 3)
        qc_row = QHBoxLayout()
        self.qc_table = QTableWidget(0, 3)
        self.qc_table.setHorizontalHeaderLabels(
            ["ColorChecker field", "Before residual", "After residual"]
        )
        self.qc_table.verticalHeader().setVisible(False)
        self.qc_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        qc_header = self.qc_table.horizontalHeader()
        qc_header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        qc_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        qc_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        qc_row.addWidget(self.qc_table, 1)
        revision_form = QWidget()
        revision_layout = QFormLayout(revision_form)
        self.reviewer = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.revision_reason = QLineEdit()
        self.revision_reason.setPlaceholderText("What changed or was verified?")
        self.pipeline_label = QLabel(
            "1. Drift, perspective, unfolding, and crop on the registered grid\n"
            "2. Selected light/color correction\n"
            "3. Geometry/facies, optical calibration, and phase interpretation"
        )
        self.pipeline_label.setObjectName("secondaryText")
        revision_layout.addRow("Reviewer", self.reviewer)
        revision_layout.addRow("Revision note", self.revision_reason)
        revision_layout.addRow("Required order", self.pipeline_label)
        qc_row.addWidget(revision_form)
        self.work_layout.addLayout(qc_row, 1)
        self.reference_mode.currentIndexChanged.connect(self._reference_mode_changed)
        self.left_preview_variant.currentIndexChanged.connect(
            self._update_comparison_images
        )
        self.right_preview_variant.currentIndexChanged.connect(
            self._update_comparison_images
        )
        self.previous_preview_frame.clicked.connect(
            lambda: self._step_preview_frame(-1)
        )
        self.next_preview_frame.clicked.connect(lambda: self._step_preview_frame(1))
        self.preview_frame_slider.valueChanged.connect(
            self._preview_frame_slider_changed
        )
        self.preview_frame_slider.sliderReleased.connect(self._start_preview)
        self._load_current()
        self.representative_run.currentIndexChanged.connect(
            self._representative_run_changed
        )
        self.representative_frame.currentIndexChanged.connect(
            self._representative_frame_changed
        )
        self.manual_preview_sources.toggled.connect(
            self._manual_preview_sources_changed
        )
        self.reference.changed.connect(self._update_source_preview_buttons)
        self.target.changed.connect(self._update_source_preview_buttons)
        self._update_source_preview_buttons()
        self.refresh()

    def _open_image_selection_plan(self) -> None:
        ImageSelectionPlanDialog(
            self.project.root,
            self,
            series_id=str(self.project.manifest.get("series_id") or ""),
        ).exec()

    def _update_source_preview_buttons(self, _value: Any = None) -> None:
        self.reference_source_preview.setEnabled(bool(self.reference.text()))
        self.target_source_preview.setEnabled(bool(self.target.text()))

    def _show_source_preview(self, path_text: str, *, role: str) -> None:
        path = Path(path_text).expanduser()
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Source image unavailable",
                f"The selected {role.casefold()} is currently unavailable:\n{path}",
            )
            return
        try:
            dialog = SourceImagePreviewDialog(path, role=role, parent=self)
        except Exception as exc:  # noqa: BLE001 - visible GUI boundary
            QMessageBox.critical(
                self,
                "Could not preview image",
                f"{type(exc).__name__}: {exc}",
            )
            return
        dialog.exec()

    def _swap_preview_sides(self) -> None:
        left = self.left_preview_variant.currentIndex()
        right = self.right_preview_variant.currentIndex()
        self.left_preview_variant.blockSignals(True)
        self.right_preview_variant.blockSignals(True)
        try:
            self.left_preview_variant.setCurrentIndex(right)
            self.right_preview_variant.setCurrentIndex(left)
        finally:
            self.left_preview_variant.blockSignals(False)
            self.right_preview_variant.blockSignals(False)
        self._update_comparison_images()

    @staticmethod
    def _sync_scrollbar(source: Any, destination: Any, value: int) -> None:
        source_span = source.maximum() - source.minimum()
        destination_span = destination.maximum() - destination.minimum()
        fraction = (
            (value - source.minimum()) / float(source_span) if source_span > 0 else 0.0
        )
        mapped = destination.minimum() + round(fraction * destination_span)
        destination.blockSignals(True)
        try:
            destination.setValue(mapped)
        finally:
            destination.blockSignals(False)

    def _link_scrollbars(
        self, source_view: ZoomableImageView, destination_view: ZoomableImageView
    ) -> None:
        source_view.horizontalScrollBar().valueChanged.connect(
            lambda value: self._sync_scrollbar(
                source_view.horizontalScrollBar(),
                destination_view.horizontalScrollBar(),
                value,
            )
        )
        source_view.verticalScrollBar().valueChanged.connect(
            lambda value: self._sync_scrollbar(
                source_view.verticalScrollBar(),
                destination_view.verticalScrollBar(),
                value,
            )
        )

    def _run_row(self, run_id: str) -> dict[str, Any] | None:
        selected = str(run_id).strip().casefold()
        return next(
            (
                dict(row)
                for row in self.project.manifest.get("runs") or []
                if isinstance(row, Mapping)
                and str(row.get("run_id") or "").strip().casefold() == selected
            ),
            None,
        )

    def _representative_frame_options(self, run_id: str) -> list[dict[str, str]]:
        cache_key = str(run_id).strip().casefold()
        if cache_key in self._frame_option_cache:
            return deepcopy(self._frame_option_cache[cache_key])
        row = self._run_row(run_id)
        mapping = dict((row or {}).get("source_mapping") or {})
        options: list[dict[str, str]] = []
        seen: set[str] = set()

        def add(label: str, role: str, value: Any) -> None:
            path = str(value or "").strip()
            key = path.casefold()
            if not path or key in seen:
                return
            seen.add(key)
            options.append({"label": label, "role": role, "path": path})

        add(
            "Baseline · before CO2 injection",
            "baseline",
            mapping.get("baseline_candidate") or mapping.get("first_image"),
        )
        sequences = [
            dict(sequence)
            for sequence in mapping.get("co2_sequences") or []
            if isinstance(sequence, Mapping)
        ]
        sequences.sort(key=lambda sequence: int(sequence.get("order") or 10**9))
        for sequence in sequences:
            order = int(sequence.get("order") or len(options) + 1)
            kind = str(sequence.get("kind") or "sequence").replace("_", " ")
            folder = Path(str(sequence.get("folder") or "")).expanduser()
            sequence_images: list[Path] = []
            if folder.is_dir():
                try:
                    sequence_images = sorted(
                        select_preferred_images(
                            (path for path in folder.iterdir() if path.is_file()),
                            self.project.image_file_policy(),
                        ),
                        key=lambda path: path.name.casefold(),
                    )
                except OSError:
                    sequence_images = []
            add(
                f"Sequence {order} ({kind}) · first frame",
                f"sequence_{order}_first",
                sequence.get("first_image"),
            )
            if cache_key == "ac52":
                stress = next(
                    (
                        image
                        for image in sequence_images
                        if image.name.casefold() == "dsc06103.jpg"
                    ),
                    None,
                )
                if stress is not None:
                    stress_index = sequence_images.index(stress)
                    for offset, suffix in ((-1, "previous"), (0, "known stress"), (1, "next")):
                        index = stress_index + offset
                        if 0 <= index < len(sequence_images):
                            image = sequence_images[index]
                            add(
                                f"AC52 9.42 h lighting test · {suffix} · {image.name}",
                                f"ac52_lighting_{suffix.replace(' ', '_')}",
                                image,
                            )
            if len(sequence_images) > 2:
                for sample_number, image_index in enumerate(
                    np.linspace(
                        0,
                        len(sequence_images) - 1,
                        min(9, len(sequence_images)),
                        dtype=int,
                    ),
                    start=1,
                ):
                    image = sequence_images[int(image_index)]
                    add(
                        f"Sequence {order} ({kind}) · sample {sample_number} of "
                        f"{min(9, len(sequence_images))} · {image.name}",
                        f"sequence_{order}_sample_{sample_number}",
                        image,
                    )
            add(
                f"Sequence {order} ({kind}) · last frame",
                f"sequence_{order}_last",
                sequence.get("last_image"),
            )
        try:
            quality_rows = [
                dict(item)
                for item in FrameQualityStore(self.project)
                .current_assessment()
                .get("frames", [])
                if str(item.get("run_id") or "").strip().casefold() == cache_key
                and str(item.get("source_path") or "")
            ]
        except (OSError, RuntimeError, ValueError):
            quality_rows = []
        quality_rows.sort(
            key=lambda item: max(
                [
                    float(value)
                    for value in dict(item.get("raw_metrics") or {}).values()
                    if isinstance(value, (int, float))
                ]
                or [0.0]
            ),
            reverse=True,
        )
        for rank, item in enumerate(quality_rows[:6], start=1):
            path = str(item.get("source_path") or "")
            add(
                f"Detected lighting stress {rank} · {Path(path).name}",
                f"quality_stress_{rank}",
                path,
            )
        add("Last registered CO2 frame", "last", mapping.get("last_image"))
        self._frame_option_cache[cache_key] = deepcopy(options)
        return options

    def _populate_representative_frames(
        self, *, preferred_path: str = ""
    ) -> None:
        run_id = str(self.representative_run.currentData() or "")
        options = self._representative_frame_options(run_id)
        self.representative_frame.blockSignals(True)
        try:
            self.representative_frame.clear()
            for option in options:
                self.representative_frame.addItem(option["label"], option)
            preferred = str(preferred_path).strip().casefold()
            selected_index = next(
                (
                    index
                    for index in range(self.representative_frame.count())
                    if str(
                        (self.representative_frame.itemData(index) or {}).get("path")
                        or ""
                    ).casefold()
                    == preferred
                ),
                -1,
            )
            if selected_index < 0 and self.representative_frame.count():
                selected_index = self.representative_frame.count() - 1
            self.representative_frame.setCurrentIndex(selected_index)
        finally:
            self.representative_frame.blockSignals(False)
        count = self.representative_frame.count()
        self.preview_frame_slider.blockSignals(True)
        try:
            self.preview_frame_slider.setMaximum(max(0, count - 1))
            self.preview_frame_slider.setValue(max(0, selected_index))
        finally:
            self.preview_frame_slider.blockSignals(False)
        self._update_preview_frame_position()
        self._apply_registered_preview_paths()

    def _apply_registered_preview_paths(self) -> None:
        if self.manual_preview_sources.isChecked():
            return
        run_id = str(self.representative_run.currentData() or "")
        options = self._representative_frame_options(run_id)
        baseline = next(
            (option["path"] for option in options if option["role"] == "baseline"),
            options[0]["path"] if options else "",
        )
        selected = self.representative_frame.currentData() or {}
        target = str(selected.get("path") or baseline)
        self.reference.setText(baseline)
        self.target.setText(target)
        run_label = run_id.upper() or "No experiment"
        if options:
            stress_count = sum(
                "stress" in str(option.get("role") or "")
                or "lighting" in str(option.get("role") or "")
                for option in options
            )
            self.representative_summary.setText(
                f"{run_label} supplies the reference and preview only. The saved profile "
                f"will be applied to all {len(self.project.manifest.get('runs') or [])} "
                f"included experiments using their registered image folders. {len(options)} "
                f"preview checkpoints are available, including {stress_count} lighting "
                "stress checkpoint(s)."
            )
        else:
            self.representative_summary.setText(
                f"{run_label} has no registered baseline or CO2 preview frames. Review "
                "Series setup or enable the manual file override."
            )

    def _representative_run_changed(self, _value: Any = None) -> None:
        self._populate_representative_frames()
        self.preview_result = None
        self.save_revision.setEnabled(False)

    def _representative_frame_changed(self, _value: Any = None) -> None:
        index = self.representative_frame.currentIndex()
        self.preview_frame_slider.blockSignals(True)
        try:
            self.preview_frame_slider.setValue(max(0, index))
        finally:
            self.preview_frame_slider.blockSignals(False)
        self._update_preview_frame_position()
        self._apply_registered_preview_paths()
        self.preview_result = None
        self.save_revision.setEnabled(False)

    def _update_preview_frame_position(self) -> None:
        count = self.representative_frame.count()
        index = self.representative_frame.currentIndex()
        self.preview_frame_position.setText(
            f"{index + 1} / {count}" if count and index >= 0 else "0 / 0"
        )
        self.previous_preview_frame.setEnabled(index > 0)
        self.next_preview_frame.setEnabled(0 <= index < count - 1)

    @Slot(int)
    def _preview_frame_slider_changed(self, value: int) -> None:
        if value != self.representative_frame.currentIndex():
            self.representative_frame.setCurrentIndex(value)

    def _step_preview_frame(self, offset: int) -> None:
        count = self.representative_frame.count()
        if not count:
            return
        index = min(max(0, self.representative_frame.currentIndex() + offset), count - 1)
        self.representative_frame.setCurrentIndex(index)
        self._start_preview()

    def _manual_preview_sources_changed(self, _value: Any = None) -> None:
        manual = self.manual_preview_sources.isChecked()
        has_registered_runs = self.representative_run.count() > 0
        colorchecker = str(self.reference_mode.currentData()) in COLORCHECKER_ALGORITHMS
        self.representative_run.setEnabled(has_registered_runs and not manual)
        self.representative_frame.setEnabled(has_registered_runs and not manual)
        self.reference.setEnabled(manual and colorchecker)
        self.target.setEnabled(manual)
        if not manual:
            self._apply_registered_preview_paths()

    def _load_current(self) -> None:
        current = self.store.current()
        saved_revisions = self.store.revisions()
        source_settings = dict(
            (self.project.manifest.get("tools") or {}).get("image_reference") or {}
        )
        configured_location = str(source_settings.get("colorchecker_location") or "")
        legacy_mode = str(source_settings.get("radiometric_mode") or "none")
        algorithm = (
            str(current.get("algorithm") or UNIFORM24_ALGORITHM)
            if saved_revisions
            else (
                UNIFORM24_ALGORITHM
                if configured_location
                in {"embedded_in_source_images", "dedicated_folder"}
                or legacy_mode == "colorchecker_uniform24"
                else IDENTITY_ALGORITHM
            )
        )
        mode_index = self.reference_mode.findData(algorithm)
        if mode_index >= 0:
            self.reference_mode.setCurrentIndex(mode_index)
        reference = str(current.get("reference_image") or "")
        target = str(current.get("target_image") or "")
        representative_run_id = str(current.get("representative_run_id") or "")
        representative_index = self.representative_run.findData(representative_run_id)
        if representative_index >= 0:
            self.representative_run.setCurrentIndex(representative_index)
        self._populate_representative_frames(preferred_path=target)
        options = self._representative_frame_options(
            str(self.representative_run.currentData() or "")
        )
        known_paths = {option["path"].casefold() for option in options}
        baseline = next(
            (option["path"] for option in options if option["role"] == "baseline"),
            options[0]["path"] if options else "",
        )
        manual_sources = not options or bool(
            (target and target.casefold() not in known_paths)
            or (reference and reference.casefold() != baseline.casefold())
        )
        self.manual_preview_sources.setChecked(manual_sources)
        if reference:
            self.reference.setText(reference)
        if target:
            self.target.setText(target)
        corner_index = self.corner.findData(current.get("colorchecker_corner"))
        if corner_index >= 0:
            self.corner.setCurrentIndex(corner_index)
        correction_settings = dict(current.get("settings") or {})
        self.neutral_tone.setChecked(
            bool(correction_settings.get("neutral_tone", True))
        )
        self.robust_affine.setChecked(
            bool(correction_settings.get("robust_affine", True))
        )
        batch = dict(current.get("batch") or {})
        if batch.get("source_root"):
            self.batch_source.setText(str(batch["source_root"]))
        elif not self.batch_source.text():
            experiment_source = next(
                (
                    str(row.get("path") or "")
                    for row in self.project.source_manifest().get("sources") or []
                    if str(row.get("kind") or "") == "experiment_images"
                ),
                "",
            )
            if experiment_source:
                self.batch_source.setText(experiment_source)
        self.batch_recursive.setChecked(bool(batch.get("recursive", True)))
        use_project_mapping = str(batch.get("inventory_scope") or "") == (
            "registered_run_mappings"
        )
        if "inventory_scope" not in batch:
            use_project_mapping = bool(self.project.manifest.get("runs"))
        self.batch_use_project_mapping.setChecked(use_project_mapping)
        mapping_index = self.batch_run_mapping.findData(
            str(batch.get("run_mapping") or "parent_folder")
        )
        if mapping_index >= 0:
            self.batch_run_mapping.setCurrentIndex(mapping_index)
        self.batch_flat_run.setText(str(batch.get("flat_run_id") or "run"))
        encoding_index = self.batch_encoding.findData(
            str(batch.get("output_encoding") or "png16")
        )
        if encoding_index >= 0:
            self.batch_encoding.setCurrentIndex(encoding_index)
        self.batch_frames = [
            dict(row) for row in batch.get("frames") or [] if isinstance(row, Mapping)
        ]
        if self.batch_frames:
            run_count = len({str(row.get("run_id") or "") for row in self.batch_frames})
            self.batch_summary.setText(
                f"Saved inventory: {len(self.batch_frames)} frames across {run_count} run(s)."
            )
        self._batch_inventory_scope_changed(invalidate=False)
        self._reference_mode_changed()
        self._manual_preview_sources_changed()

    def _reference_mode_changed(self, _value: Any = None) -> None:
        algorithm = str(self.reference_mode.currentData())
        colorchecker = algorithm in COLORCHECKER_ALGORITHMS
        right_index = self.right_preview_variant.findData(algorithm)
        if right_index >= 0:
            self.right_preview_variant.setCurrentIndex(right_index)
        if algorithm == UNIFORM24_ALGORITHM:
            explanation = (
                "Uses all 24 chart patches to estimate one global color transform for each "
                "registered frame. This corrects frame-to-frame camera and lighting color "
                "drift, but does not compensate for a left-to-right spatial light gradient."
            )
        elif algorithm == DARSIA_BASELINE_COLOR_ALGORITHM:
            explanation = (
                "Reproduces DarSIA's original baseline-relative ColorCorrection using "
                "adaptive white and affine color balancing. Keep it as a legacy comparator; "
                "selecting it creates a distinct, fully traced analysis branch."
            )
        else:
            explanation = (
                "Applies no radiometric transform. Use this as an explicit control when no "
                "validated color reference is available, and review frame stability before "
                "building color paths."
            )
        self.correction_method_explanation.setText(explanation)
        self.corner.setEnabled(colorchecker)
        self.neutral_tone.setEnabled(colorchecker)
        self.robust_affine.setEnabled(colorchecker)
        self.qc_table.setVisible(colorchecker)
        self.show_difference.setEnabled(colorchecker)
        self.show_colorchecker_roi.setEnabled(colorchecker)
        if not colorchecker:
            self.show_difference.setChecked(False)
            self.show_colorchecker_roi.setChecked(False)
            self.original_view.clear_roi()
            self.corrected_view.clear_roi()
            self.colorchecker_detection_status.setText(
                "ColorChecker detection is not used by the identity control."
            )
        else:
            self.show_colorchecker_roi.setChecked(True)
            self.colorchecker_detection_status.setText(
                "Calculate a preview to verify that the detected ColorChecker region is correct."
            )
        self.pipeline_label.setText(
            "1. Drift, perspective, unfolding, and crop\n"
            + (
                "2. Selected ColorChecker correction\n"
                if colorchecker
                else "2. No radiometric correction (identity control)\n"
            )
            + "3. Geometry/facies, optical calibration, and phase interpretation"
        )
        self.preview_result = None
        self._preview_roi_source = ""
        self.save_revision.setEnabled(False)
        self.correction_summary.setText(
            "Choose a target frame to verify the explicit no-target policy."
            if not colorchecker
            else "Choose a ColorChecker reference and target frame."
        )
        self._manual_preview_sources_changed()

    def refresh(self) -> None:
        self._refresh_gate()
        current_id = ""
        current = self.store.current()
        if isinstance(current, dict):
            current_id = str(current.get("revision_id") or "")
        self.revisions.clear()
        rows = list(reversed(self.store.revisions()))
        if not rows:
            self.revisions.addItem("No saved revisions", "")
        for revision in rows:
            algorithm_label = self._variant_title(str(revision.get("algorithm") or ""))
            label = (
                f"{algorithm_label} · {str(revision.get('recorded_utc') or '')[:19]} · "
                f"{str(revision.get('revision_id') or '')[:8]}"
            )
            self.revisions.addItem(label, str(revision.get("revision_id") or ""))
            if str(revision.get("revision_id") or "") == current_id:
                self.revisions.setCurrentIndex(self.revisions.count() - 1)
        self.activate_revision_button.setEnabled(bool(rows))
        if hasattr(self, "frame_quality"):
            self.frame_quality.refresh()

    @Slot(str)
    def _inspect_quality_frame(self, source_path: str) -> None:
        path = Path(source_path)
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Source image unavailable",
                f"The selected source image is currently unavailable:\n{path}",
            )
            return
        self.target.setText(str(path))
        self._start_preview()

    def _activate_selected_revision(self) -> None:
        revision_id = str(self.revisions.currentData() or "")
        if not revision_id:
            return
        try:
            revision = self.store.activate_revision(
                revision_id,
                actor=self.reviewer.text().strip(),
                reason=(
                    "An existing image-correction revision was selected as the active "
                    "analysis branch."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not select correction revision", str(exc))
            return
        self._load_current()
        self.refresh()
        QMessageBox.information(
            self,
            "Correction branch selected",
            f"Revision {str(revision['revision_id'])[:8]} is active. Dependent color "
            "paths and analyses must match this revision before approval.",
        )

    def _start_inventory(self) -> None:
        if self._inventory_thread is not None:
            return
        use_project_mapping = self.batch_use_project_mapping.isChecked()
        source = Path(self.batch_source.text()).expanduser()
        if not use_project_mapping and not source.is_dir():
            QMessageBox.warning(
                self,
                "Select a batch image folder",
                "Choose an existing folder before building the immutable inventory.",
            )
            return
        self.batch_scan.setEnabled(False)
        self.batch_summary.setText(
            "Discovering images and calculating source hashes. This happens once per saved plan..."
        )
        thread = QThread(self)
        worker = CorrectionInventoryWorker(
            source.resolve(),
            recursive=self.batch_recursive.isChecked(),
            run_mapping=str(self.batch_run_mapping.currentData()),
            flat_run_id=self.batch_flat_run.text().strip() or "run",
            project=self.project,
            use_project_mapping=use_project_mapping,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._inventory_ready)
        worker.failed.connect(self._inventory_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._inventory_thread_finished)
        self._inventory_thread = thread
        self._inventory_worker = worker
        thread.start()

    def _invalidate_batch_inventory(self, _value: Any = None) -> None:
        if not self.batch_frames:
            return
        self.batch_frames = []
        self.batch_summary.setText(
            "Batch settings changed. Rebuild the immutable frame inventory before saving."
        )

    def _batch_inventory_scope_changed(
        self,
        _value: Any = None,
        *,
        invalidate: bool = True,
    ) -> None:
        registered = self.batch_use_project_mapping.isChecked()
        self.batch_manual_label.setVisible(not registered)
        self.batch_source.setVisible(not registered)
        self.batch_recursive.setVisible(not registered)
        self.batch_run_mapping.setVisible(not registered)
        self.batch_flat_run.setVisible(not registered)
        if invalidate:
            self._invalidate_batch_inventory()

    def _inventory_thread_finished(self) -> None:
        if self._inventory_thread is not None:
            # After wait() the worker is freed here, in the GUI thread (Q185).
            self._inventory_thread.wait()
            self._inventory_thread.deleteLater()
        self._inventory_worker = None
        self._inventory_thread = None
        self.batch_scan.setEnabled(True)

    @Slot(object)
    def _inventory_ready(self, raw_frames: object) -> None:
        self.batch_frames = [
            dict(row) for row in raw_frames if isinstance(row, Mapping)
        ]
        run_count = len({str(row.get("run_id") or "") for row in self.batch_frames})
        self.batch_summary.setText(
            f"Immutable inventory ready: {len(self.batch_frames)} frames across "
            f"{run_count} run(s). Save the correction revision to bind it to execution."
        )

    @Slot(str)
    def _inventory_failed(self, message: str) -> None:
        self.batch_frames = []
        self.batch_summary.setText("Batch inventory failed; no frame list was changed.")
        QMessageBox.critical(self, "Could not inventory correction frames", message)

    def _settings(self) -> dict[str, Any]:
        settings = deepcopy_settings(default_correction_config()["settings"])
        settings["neutral_tone"] = self.neutral_tone.isChecked()
        settings["robust_affine"] = self.robust_affine.isChecked()
        return settings

    def _start_preview(self) -> None:
        algorithm = str(self.reference_mode.currentData())
        reference = (
            Path(self.reference.text()).expanduser()
            if self.reference.text().strip()
            else None
        )
        target = Path(self.target.text()).expanduser()
        if not target.is_file() or (
            algorithm in COLORCHECKER_ALGORITHMS
            and (reference is None or not reference.is_file())
        ):
            QMessageBox.warning(
                self,
                "Select required images",
                (
                    "Choose an existing ColorChecker reference and target frame."
                    if algorithm in COLORCHECKER_ALGORITHMS
                    else "Choose an existing target frame."
                ),
            )
            return
        if self._thread is not None and self._thread.isRunning():
            return
        self.preview_button.setEnabled(False)
        self.save_revision.setEnabled(False)
        self.correction_summary.setText(
            "Calculating raw, original DarSIA, and improved Uniform24 views..."
            if algorithm in COLORCHECKER_ALGORITHMS
            else "Loading the native frame without an external color transform..."
        )
        current = self.store.current()
        roi = current.get("colorchecker_roi_rc") if isinstance(current, dict) else None
        if reference is None or str(current.get("reference_image") or "") != str(
            reference.resolve()
        ):
            roi = None
        self._preview_roi_source = "saved" if roi is not None else "detected"
        self._thread = QThread(self)
        self._worker = CorrectionWorker(
            algorithm,
            reference.resolve() if reference is not None else None,
            target.resolve(),
            str(self.corner.currentData()),
            roi,
            self._settings(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._preview_ready)
        self._worker.failed.connect(self._preview_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot(object)
    def _preview_ready(self, result: object) -> None:
        self.preview_result = dict(result)
        self._set_available_comparison_variants()
        self._update_comparison_images()
        diagnostics = dict(self.preview_result["diagnostics"])
        if diagnostics.get("algorithm") == IDENTITY_ALGORITHM:
            self.correction_summary.setText(
                "NO EXTERNAL COLOR TARGET · image values are unchanged by this stage. "
                "Registration and spatial-illumination QC remain required separately."
            )
            self.qc_table.setRowCount(0)
            self.colorchecker_detection_status.setText(
                "ColorChecker detection is not used by the identity control."
            )
        else:
            before = dict(diagnostics["before"])
            after = dict(diagnostics["after"])
            flag = "REVIEW REQUIRED" if diagnostics.get("flagged") else "QC PASS"
            reasons = ", ".join(diagnostics.get("flag_reasons") or [])
            self.correction_summary.setText(
                f"{flag} · mean residual {_number(before['mean'])} → {_number(after['mean'])} · "
                f"maximum {_number(before['max'])} → {_number(after['max'])}"
                + (f" · {reasons}" if reasons else "")
            )
            before_fields = list(before["swatch_residuals"])
            after_fields = list(after["swatch_residuals"])
            self.qc_table.setRowCount(len(SWATCH_NAMES))
            for row, name in enumerate(SWATCH_NAMES):
                self.qc_table.setItem(row, 0, QTableWidgetItem(name))
                self.qc_table.setItem(
                    row, 1, QTableWidgetItem(f"{before_fields[row]:.5f}")
                )
                after_item = QTableWidgetItem(f"{after_fields[row]:.5f}")
                if after_fields[row] > float(
                    self._settings()["max_swatch_warning_threshold"]
                ):
                    after_item.setForeground(QColor("#e27979"))
                self.qc_table.setItem(row, 2, after_item)
            roi = np.asarray(self.preview_result.get("roi_rc"), dtype=np.float64)
            if roi.shape == (4, 2):
                row_min, col_min = np.floor(np.min(roi, axis=0)).astype(int)
                row_max, col_max = np.ceil(np.max(roi, axis=0)).astype(int)
                action = (
                    "Reused the saved ColorChecker region"
                    if self._preview_roi_source == "saved"
                    else "Detected the ColorChecker in the reference frame"
                )
                self.colorchecker_detection_status.setText(
                    f"{action}: rows {row_min}-{row_max}, columns {col_min}-{col_max}. "
                    "The same registered region is outlined on both target previews; "
                    "verify that it encloses the physical 24-patch chart."
                )
            else:
                self.colorchecker_detection_status.setText(
                    "No valid ColorChecker region was returned. Review is required."
                )
        self._update_checker_overlay()
        self.preview_button.setEnabled(True)
        self.save_revision.setEnabled(True)
        self._worker = None
        self._thread = None

    @Slot(str)
    def _preview_failed(self, message: str) -> None:
        self.preview_result = None
        self.preview_button.setEnabled(True)
        self.save_revision.setEnabled(False)
        self.correction_summary.setText(
            "Preview failed. No correction revision was changed."
        )
        self.original_view.clear_roi()
        self.corrected_view.clear_roi()
        self.colorchecker_detection_status.setText(
            "ColorChecker detection failed; no region was accepted."
        )
        self._worker = None
        self._thread = None
        QMessageBox.critical(self, "Image correction failed", message)

    @staticmethod
    def _variant_title(algorithm: str) -> str:
        return {
            IDENTITY_ALGORITHM: "Raw source frame",
            DARSIA_BASELINE_COLOR_ALGORITHM: "Original DarSIA correction",
            UNIFORM24_ALGORITHM: "Improved Uniform24 correction",
        }.get(str(algorithm), "Correction preview")

    def _set_available_comparison_variants(self) -> None:
        variants = dict((self.preview_result or {}).get("comparison_variants") or {})
        for combo in (self.left_preview_variant, self.right_preview_variant):
            for index in range(combo.count()):
                model = combo.model().item(index)
                if model is not None:
                    model.setEnabled(str(combo.itemData(index)) in variants)
        selected = str(self.reference_mode.currentData())
        if selected in variants:
            index = self.right_preview_variant.findData(selected)
            if index >= 0:
                self.right_preview_variant.setCurrentIndex(index)

    def _update_right_image(self) -> None:
        self._update_comparison_images()

    def _update_comparison_images(self, _value: Any = None) -> None:
        if not self.preview_result:
            return
        variants = dict(self.preview_result.get("comparison_variants") or {})
        left_algorithm = str(self.left_preview_variant.currentData())
        right_algorithm = str(self.right_preview_variant.currentData())
        left_result = dict(variants.get(left_algorithm) or self.preview_result)
        right_result = dict(variants.get(right_algorithm) or self.preview_result)
        left = np.asarray(left_result["corrected"])
        right = np.asarray(right_result["corrected"])
        self.original_title.setText(self._variant_title(left_algorithm))
        self.original_view.set_array(left)
        if self.show_difference.isChecked():
            self.corrected_title.setText("Amplified left/right difference (4×)")
            self.corrected_view.set_array(
                np.clip(np.abs(right - left) * 4.0, 0.0, 1.0)
            )
        else:
            self.corrected_title.setText(self._variant_title(right_algorithm))
            self.corrected_view.set_array(right)
        self._update_checker_overlay()

    def _update_checker_overlay(self, _value: Any = None) -> None:
        if (
            not self.preview_result
            or not self.show_colorchecker_roi.isChecked()
            or str(self.reference_mode.currentData()) not in COLORCHECKER_ALGORITHMS
        ):
            self.original_view.clear_roi()
            self.corrected_view.clear_roi()
            return
        diagnostics = dict(self.preview_result.get("diagnostics") or {})
        source_shape = diagnostics.get("source_shape") or []
        roi = self.preview_result.get("roi_rc")
        self.original_view.set_roi(roi, source_shape=source_shape)
        self.corrected_view.set_roi(roi, source_shape=source_shape)

    def _save_revision(self) -> None:
        if not self.preview_result:
            return
        algorithm = str(self.reference_mode.currentData())
        config = (
            identity_correction_config()
            if algorithm == IDENTITY_ALGORITHM
            else (
                darsia_baseline_correction_config()
                if algorithm == DARSIA_BASELINE_COLOR_ALGORITHM
                else default_correction_config()
            )
        )
        saved_settings = deepcopy(dict(config.get("settings") or {}))
        saved_settings.update(self._settings())
        config.update(
            {
                "application_scope": "series",
                "representative_run_id": str(
                    self.representative_run.currentData() or ""
                ),
                "representative_preview_role": str(
                    "manual_override"
                    if self.manual_preview_sources.isChecked()
                    else (self.representative_frame.currentData() or {}).get("role")
                    or "registered_preview"
                ),
                "reference_image": (
                    str(Path(self.reference.text()).resolve())
                    if algorithm in COLORCHECKER_ALGORITHMS
                    else ""
                ),
                "target_image": str(Path(self.target.text()).resolve()),
                "colorchecker_corner": str(self.corner.currentData()),
                "colorchecker_roi_rc": self.preview_result["roi_rc"],
                "settings": saved_settings,
                "batch": {
                    "inventory_scope": (
                        "registered_run_mappings"
                        if self.batch_use_project_mapping.isChecked()
                        else "manual_source_root"
                    ),
                    "source_root": (
                        str(Path(self.batch_source.text()).expanduser().resolve())
                        if self.batch_source.text()
                        else ""
                    ),
                    "recursive": self.batch_recursive.isChecked(),
                    "run_mapping": str(self.batch_run_mapping.currentData()),
                    "flat_run_id": self.batch_flat_run.text().strip() or "run",
                    "output_encoding": str(self.batch_encoding.currentData()),
                    "frames": deepcopy(self.batch_frames),
                },
            }
        )
        try:
            revision = self.store.save_revision(
                config,
                actor=self.reviewer.text().strip(),
                reason=self.revision_reason.text().strip(),
                diagnostics=self.preview_result["diagnostics"],
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save correction revision", str(exc))
            return
        self.revision_reason.clear()
        self.refresh()
        QMessageBox.information(
            self,
            "Correction revision saved",
            f"Revision {str(revision['revision_id'])[:8]} was stored locally with full provenance.",
        )
