"""Qt geometry-and-facies workflow page."""

from __future__ import annotations

import json
import math
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PySide6.QtCore import QObject, QPointF, QRectF, QSignalBlocker, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QImage,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QAbstractButton,
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.facies_learning import (
    FaciesAnnotationStore,
)
from darsia.fluidflower.facies_examples import (
    FACIES_MATERIAL_ROLES,
    FaciesExampleStore,
    create_facies_example_prediction,
    default_facies_example_session,
    facies_config_from_example_prediction,
    facies_example_quality,
    load_facies_example_image,
    normalize_facies_example_session,
)
from darsia.fluidflower.facies_model import (
    FaciesModelConfig,
    FaciesModelStore,
    train_facies_baseline,
)
from darsia.fluidflower.geometry_facies import (
    ANALYSIS_ROLES,
    GEOMETRY_REVISION_FORMAT,
    INJECTION_POINT_CONFLICT_TOLERANCE_M,
    RUN_GEOMETRY_ACQUISITION_STATES,
    RUN_GEOMETRY_CHOICES,
    GeometryLabelEditSession,
    GeometryRevisionStore,
    create_optical_facies_draft,
    default_run_geometry_config,
    geometry_config_digest,
    import_darsia_rig,
    load_geometry_preview,
    load_protocol_injection_points,
    normalize_run_geometry_config,
    validate_geometry_config,
)
from darsia.fluidflower.modules import dependencies_satisfied
from darsia.fluidflower.page_registry import (
    FluidFlowerSubpageRegistry,
    default_subpage_registry,
)
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.ui.common import (
    DIRECTORY_DIALOG_OPTIONS,
    FILE_DIALOG_OPTIONS,
    INJECTOR_RECORD_ROLE,
    ModulePage,
    _gui_float,
    _gui_int,
    _millimetres,
    _percent,
    _table_number,
    deepcopy_settings,
)


class GeometryImageView(QGraphicsView):
    physical_point_selected = Signal(float, float)
    # Clicked position as fractions (0..1) of the image width and height.
    image_point_selected = Signal(float, float)
    label_selected = Signal(int)
    label_edit_drawn = Signal(str, object)
    facies_example_box_drawn = Signal(object)

    _PALETTE = np.asarray(
        [
            (36, 41, 46),
            (77, 184, 166),
            (231, 178, 68),
            (89, 148, 205),
            (220, 105, 112),
            (151, 122, 201),
            (103, 178, 92),
            (225, 137, 67),
            (74, 194, 207),
            (205, 103, 166),
            (168, 174, 83),
            (113, 135, 177),
            (235, 199, 117),
            (132, 194, 168),
            (195, 125, 95),
            (143, 157, 168),
        ],
        dtype=np.uint8,
    )

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#090b0d"))
        self.setMinimumSize(520, 380)
        self.preview: dict[str, Any] | None = None
        self.config: dict[str, Any] = {}
        self.background = "baseline"
        self.overlay = "facies"
        self.opacity = 0.45
        self.boundaries_only = False
        self.show_injectors = True
        self.place_mode = False
        self.select_mode = False
        self.edit_mode = "pan"
        self._edit_points: list[QPointF] = []
        self._edit_item: QGraphicsPathItem | None = None
        self._brush_active = False
        self._example_draft_item: QGraphicsRectItem | None = None
        self._example_items: list[Any] = []
        self._example_boxes: list[dict[str, Any]] = []
        self._example_materials: list[dict[str, Any]] = []
        self._show_example_boxes = True

    def set_model(self, preview: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.cancel_edit()
        self.preview = dict(preview)
        self.config = deepcopy_settings(config)
        self._render(reset_view=True)

    def update_preview(
        self, preview: Mapping[str, Any], config: Mapping[str, Any]
    ) -> None:
        """Refresh raster fields while preserving the current pan and zoom."""

        self.preview = dict(preview)
        self.config = deepcopy_settings(config)
        self._render(reset_view=False)

    def set_config(self, config: Mapping[str, Any]) -> None:
        self.config = deepcopy_settings(config)
        self._render(reset_view=False)

    def set_options(
        self,
        *,
        background: str | None = None,
        overlay: str | None = None,
        opacity: float | None = None,
        boundaries_only: bool | None = None,
        show_injectors: bool | None = None,
    ) -> None:
        if background is not None:
            self.background = background
        if overlay is not None:
            self.overlay = overlay
        if opacity is not None:
            self.opacity = float(np.clip(opacity, 0.0, 1.0))
        if boundaries_only is not None:
            self.boundaries_only = boundaries_only
        if show_injectors is not None:
            self.show_injectors = show_injectors
        self._render(reset_view=False)

    @staticmethod
    def _boundaries(field: np.ndarray) -> np.ndarray:
        values = np.asarray(field)
        boundary = np.zeros(values.shape, dtype=bool)
        boundary[:, 1:] |= values[:, 1:] != values[:, :-1]
        boundary[:, :-1] |= values[:, 1:] != values[:, :-1]
        boundary[1:, :] |= values[1:, :] != values[:-1, :]
        boundary[:-1, :] |= values[1:, :] != values[:-1, :]
        return boundary

    def _region_scalar(self, name: str) -> np.ndarray:
        assert self.preview is not None
        labels = np.asarray(self.preview["labels"])
        output = np.full(labels.shape, np.nan, dtype=np.float32)
        for row in self.config.get("regions") or []:
            try:
                value = float(row.get(name))
                label_id = int(row.get("label_id"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                output[labels == label_id] = value
        if name == "permeability":
            positive = output > 0.0
            output[positive] = np.log10(output[positive])
        return output

    @staticmethod
    def _scalar_colors(field: np.ndarray) -> np.ndarray:
        import cv2

        values = np.asarray(field, dtype=np.float32)
        finite = values[np.isfinite(values)]
        normalized = np.zeros(values.shape, dtype=np.uint8)
        if finite.size:
            low, high = np.percentile(finite, [2.0, 98.0])
            if high <= low:
                high = low + 1.0
            scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
            normalized[np.isfinite(values)] = np.rint(
                255.0 * scaled[np.isfinite(values)]
            ).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
        return np.asarray(colored[..., ::-1], dtype=np.uint8)

    def _render(self, *, reset_view: bool) -> None:
        if self.preview is None:
            return
        import cv2

        background_key = (
            "review_frame"
            if self.background == "review_frame" and "review_frame" in self.preview
            else "baseline"
        )
        # The 8-bit photograph is kept while the same photograph array is shown,
        # so a redraw for a stroke or an overlay does not convert it again.
        source = self.preview[background_key]
        kept = self.__dict__.get("_base8_kept")
        if kept is not None and kept[0] is source:
            base8 = kept[1]
        else:
            base = np.clip(np.asarray(source, dtype=np.float32), 0.0, 1.0)
            base8 = np.rint(255.0 * base).astype(np.uint8)
            self._base8_kept = (source, base8)
        field: np.ndarray | None = None
        colors: np.ndarray | None = None
        if self.overlay == "labels":
            field = np.asarray(self.preview["labels"], dtype=np.int32)
            colors = self._PALETTE[np.mod(field, len(self._PALETTE))]
        elif self.overlay == "facies":
            field = np.asarray(self.preview["facies"], dtype=np.int32)
            colors = self._PALETTE[np.mod(field, len(self._PALETTE))]
        elif self.overlay == "facies_proposal" and "facies_proposal" in self.preview:
            field = np.asarray(self.preview["facies_proposal"], dtype=np.int32)
            colors = self._PALETTE[np.mod(field, len(self._PALETTE))]
        elif (
            self.overlay == "facies_uncertainty"
            and "facies_uncertainty" in self.preview
        ):
            field = np.asarray(self.preview["facies_uncertainty"], dtype=np.float32)
            colors = self._scalar_colors(field)
        elif self.overlay == "depth":
            field = np.asarray(self.preview["depth"], dtype=np.float32)
            colors = self._scalar_colors(field)
        elif self.overlay in {"porosity", "permeability"}:
            field = self._region_scalar(self.overlay)
            colors = self._scalar_colors(field)

        display = base8.copy()
        if field is not None and colors is not None:
            if self.boundaries_only and self.overlay in {
                "labels",
                "facies",
                "facies_proposal",
            }:
                boundary = self._boundaries(field)
                display[boundary] = colors[boundary]
            else:
                valid = np.ones(field.shape, dtype=bool)
                if np.issubdtype(field.dtype, np.floating):
                    valid = np.isfinite(field)
                blended = np.rint(
                    (1.0 - self.opacity) * display.astype(np.float32)
                    + self.opacity * colors.astype(np.float32)
                ).astype(np.uint8)
                display[valid] = blended[valid]

        if self.show_injectors:
            height, width = display.shape[:2]
            width_m = _gui_float(self.config.get("width_m"))
            height_m = _gui_float(self.config.get("height_m"))
            if width_m and height_m:
                for index, injector in enumerate(
                    self.config.get("injectors") or [], start=1
                ):
                    x_m = _gui_float(injector.get("x_m"))
                    y_m = _gui_float(injector.get("y_m"))
                    if x_m is None or y_m is None:
                        continue
                    column = int(round(np.clip(x_m / width_m, 0.0, 1.0) * (width - 1)))
                    row = int(
                        round((1.0 - np.clip(y_m / height_m, 0.0, 1.0)) * (height - 1))
                    )
                    radius = max(5, int(round(min(height, width) * 0.009)))
                    cv2.circle(
                        display, (column, row), radius, (238, 74, 83), 2, cv2.LINE_AA
                    )
                    cv2.circle(
                        display,
                        (column, row),
                        max(2, radius // 3),
                        (238, 74, 83),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        display,
                        str(index),
                        (column + radius + 3, row + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        max(0.4, min(height, width) / 1600.0),
                        (248, 92, 99),
                        1,
                        cv2.LINE_AA,
                    )

        image = np.ascontiguousarray(display)
        height, width = image.shape[:2]
        qimage = QImage(
            image.data,
            width,
            height,
            int(image.strides[0]),
            QImage.Format.Format_RGB888,
        ).copy()
        self.pixmap_item.setPixmap(QPixmap.fromImage(qimage))
        self.scene().setSceneRect(self.pixmap_item.boundingRect())
        self._render_example_boxes()
        if reset_view:
            self.fit_image()

    def set_example_boxes(
        self,
        examples: list[dict[str, Any]],
        materials: list[dict[str, Any]],
        *,
        visible: bool = True,
    ) -> None:
        self._example_boxes = deepcopy(examples)
        self._example_materials = deepcopy(materials)
        self._show_example_boxes = bool(visible)
        self._render_example_boxes()

    def _clear_example_items(self) -> None:
        for item in self._example_items:
            if item.scene() is self.scene():
                self.scene().removeItem(item)
        self._example_items.clear()

    def _render_example_boxes(self) -> None:
        self._clear_example_items()
        if not self._show_example_boxes or self.pixmap_item.pixmap().isNull():
            return
        bounds = self.pixmap_item.boundingRect()
        materials = {
            str(row.get("material_id") or ""): (index, str(row.get("name") or ""))
            for index, row in enumerate(self._example_materials, start=1)
        }
        for row in self._example_boxes:
            material_id = str(row.get("material_id") or "")
            material_index, name = materials.get(material_id, (0, "Unknown"))
            color_values = self._PALETTE[material_index % len(self._PALETTE)]
            color = QColor(*(int(value) for value in color_values))
            rectangle = QRectF(
                float(row.get("x") or 0.0) * bounds.width(),
                float(row.get("y") or 0.0) * bounds.height(),
                float(row.get("width") or 0.0) * bounds.width(),
                float(row.get("height") or 0.0) * bounds.height(),
            ).normalized()
            item = QGraphicsRectItem(rectangle)
            pen = QPen(color, 2.0)
            pen.setCosmetic(True)
            item.setPen(pen)
            item.setZValue(14.0)
            item.setToolTip(name)
            self.scene().addItem(item)
            self._example_items.append(item)
            label = QGraphicsSimpleTextItem(name)
            label.setBrush(color)
            label.setPos(rectangle.left() + 2.0, rectangle.top() + 2.0)
            label.setFlag(
                QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations, True
            )
            label.setZValue(15.0)
            self.scene().addItem(label)
            self._example_items.append(label)

    def fit_image(self) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        if self.pixmap_item.pixmap().isNull():
            return
        factor = 1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2
        self.scale(factor, factor)

    def set_edit_mode(self, mode: str) -> None:
        if mode not in {"pan", "polygon", "brush", "example_box"}:
            raise ValueError(f"Unsupported geometry edit mode: {mode}")
        self.cancel_edit()
        self.edit_mode = mode
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if mode == "pan"
            else QGraphicsView.DragMode.NoDrag
        )

    def cancel_edit(self) -> None:
        self._edit_points.clear()
        self._brush_active = False
        if self._edit_item is not None:
            self.scene().removeItem(self._edit_item)
            self._edit_item = None
        if self._example_draft_item is not None:
            self.scene().removeItem(self._example_draft_item)
            self._example_draft_item = None

    def finish_edit(self) -> None:
        self._finish_edit()

    def _scene_point(self, event: Any) -> QPointF | None:
        point = self.mapToScene(event.position().toPoint())
        return point if self.pixmap_item.boundingRect().contains(point) else None

    def _normalized_points(self) -> list[list[float]]:
        bounds = self.pixmap_item.boundingRect()
        width = max(1.0, bounds.width() - 1.0)
        height = max(1.0, bounds.height() - 1.0)
        return [
            [
                float(np.clip(point.x() / width, 0.0, 1.0)),
                float(np.clip(point.y() / height, 0.0, 1.0)),
            ]
            for point in self._edit_points
        ]

    def _update_edit_path(self) -> None:
        if not self._edit_points:
            return
        path = QPainterPath(self._edit_points[0])
        for point in self._edit_points[1:]:
            path.lineTo(point)
        if self.edit_mode == "polygon" and len(self._edit_points) >= 3:
            path.lineTo(self._edit_points[0])
        if self._edit_item is None:
            self._edit_item = QGraphicsPathItem()
            pen = QPen(QColor("#ff5161"), 2.0)
            pen.setCosmetic(True)
            self._edit_item.setPen(pen)
            self._edit_item.setZValue(20.0)
            self.scene().addItem(self._edit_item)
        self._edit_item.setPath(path)

    def _update_example_rectangle(self) -> None:
        if len(self._edit_points) < 2:
            return
        rectangle = QRectF(self._edit_points[0], self._edit_points[-1]).normalized()
        if self._example_draft_item is None:
            self._example_draft_item = QGraphicsRectItem()
            pen = QPen(QColor("#ff5161"), 2.0)
            pen.setCosmetic(True)
            self._example_draft_item.setPen(pen)
            self._example_draft_item.setZValue(20.0)
            self.scene().addItem(self._example_draft_item)
        self._example_draft_item.setRect(rectangle)

    def _finish_example_box(self) -> None:
        if len(self._edit_points) < 2:
            self.cancel_edit()
            return
        rectangle = QRectF(self._edit_points[0], self._edit_points[-1]).normalized()
        bounds = self.pixmap_item.boundingRect()
        if rectangle.width() < 2.0 or rectangle.height() < 2.0:
            self.cancel_edit()
            return
        payload = {
            "x": float(np.clip(rectangle.left() / bounds.width(), 0.0, 1.0)),
            "y": float(np.clip(rectangle.top() / bounds.height(), 0.0, 1.0)),
            "width": float(np.clip(rectangle.width() / bounds.width(), 0.0, 1.0)),
            "height": float(np.clip(rectangle.height() / bounds.height(), 0.0, 1.0)),
        }
        self.cancel_edit()
        self.facies_example_box_drawn.emit(payload)

    def _finish_edit(self) -> None:
        minimum = 3 if self.edit_mode == "polygon" else 1
        if len(self._edit_points) >= minimum:
            kind = self.edit_mode
            points = self._normalized_points()
            self.cancel_edit()
            self.label_edit_drawn.emit(kind, points)
        else:
            self.cancel_edit()

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if (
            self.edit_mode == "example_box"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            point = self._scene_point(event)
            if point is not None:
                self.cancel_edit()
                self._edit_points = [point, point]
                self._update_example_rectangle()
                event.accept()
                return
        if (
            self.edit_mode in {"polygon", "brush"}
            and event.button() == Qt.MouseButton.LeftButton
        ):
            point = self._scene_point(event)
            if point is not None:
                if self.edit_mode == "brush":
                    self.cancel_edit()
                    self._brush_active = True
                self._edit_points.append(point)
                self._update_edit_path()
                event.accept()
                return
        if (
            self.place_mode or self.select_mode
        ) and event.button() == Qt.MouseButton.LeftButton:
            point = self.mapToScene(event.position().toPoint())
            bounds = self.pixmap_item.boundingRect()
            if bounds.contains(point):
                if self.select_mode and self.preview is not None:
                    labels = np.asarray(self.preview["labels"])
                    column = int(
                        np.clip(
                            round(
                                point.x()
                                / max(1.0, bounds.width() - 1.0)
                                * (labels.shape[1] - 1)
                            ),
                            0,
                            labels.shape[1] - 1,
                        )
                    )
                    row = int(
                        np.clip(
                            round(
                                point.y()
                                / max(1.0, bounds.height() - 1.0)
                                * (labels.shape[0] - 1)
                            ),
                            0,
                            labels.shape[0] - 1,
                        )
                    )
                    self.label_selected.emit(int(labels[row, column]))
                    return
                self.image_point_selected.emit(
                    float(point.x() / max(1.0, bounds.width() - 1.0)),
                    float(point.y() / max(1.0, bounds.height() - 1.0)),
                )
                width_m = _gui_float(self.config.get("width_m"))
                height_m = _gui_float(self.config.get("height_m"))
                if width_m and height_m:
                    x_m = float(point.x() / max(1.0, bounds.width() - 1.0) * width_m)
                    y_m = float(
                        (1.0 - point.y() / max(1.0, bounds.height() - 1.0)) * height_m
                    )
                    self.physical_point_selected.emit(x_m, y_m)
                    return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self.edit_mode == "example_box" and len(self._edit_points) == 2:
            point = self._scene_point(event)
            if point is not None:
                self._edit_points[-1] = point
                self._update_example_rectangle()
            event.accept()
            return
        if self.edit_mode == "brush" and self._brush_active:
            point = self._scene_point(event)
            if point is not None and (
                not self._edit_points
                or (point - self._edit_points[-1]).manhattanLength() >= 1.0
            ):
                self._edit_points.append(point)
                self._update_edit_path()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if (
            self.edit_mode == "example_box"
            and len(self._edit_points) == 2
            and event.button() == Qt.MouseButton.LeftButton
        ):
            point = self._scene_point(event)
            if point is not None:
                self._edit_points[-1] = point
            self._finish_example_box()
            event.accept()
            return
        if (
            self.edit_mode == "brush"
            and self._brush_active
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._finish_edit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: Any) -> None:  # noqa: N802
        if self.edit_mode == "polygon" and event.button() == Qt.MouseButton.LeftButton:
            self._finish_edit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event: Any) -> None:  # noqa: N802
        if self.edit_mode == "polygon" and event.key() in {
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        }:
            self._finish_edit()
            event.accept()
            return
        if self.edit_mode in {"polygon", "brush", "example_box"} and event.key() == Qt.Key.Key_Escape:
            self.cancel_edit()
            event.accept()
            return
        super().keyPressEvent(event)


class GeometryWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        mode: str,
        config: Mapping[str, Any] | None = None,
        rig_folder: Path | None = None,
        facies_properties_path: Path | None = None,
        protocol_path: Path | None = None,
        rig_type: str = "custom",
        draft_source_path: Path | None = None,
        draft_destination: Path | None = None,
        draft_width_m: float | None = None,
        draft_height_m: float | None = None,
        draft_depth_m: float | None = None,
        draft_classes: int = 6,
        preview_run_id: str | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.config = deepcopy_settings(config or {})
        self.rig_folder = rig_folder
        self.facies_properties_path = facies_properties_path
        self.protocol_path = protocol_path
        self.rig_type = rig_type
        self.draft_source_path = draft_source_path
        self.draft_destination = draft_destination
        self.draft_width_m = draft_width_m
        self.draft_height_m = draft_height_m
        self.draft_depth_m = draft_depth_m
        self.draft_classes = draft_classes
        self.preview_run_id = str(preview_run_id or "")

    @Slot()
    def run(self) -> None:
        try:
            if self.mode == "import":
                if self.rig_folder is None:
                    raise ValueError("No DarSIA rig folder was selected")
                config = import_darsia_rig(
                    self.rig_folder,
                    facies_properties_path=self.facies_properties_path,
                    protocol_path=self.protocol_path,
                    rig_type=self.rig_type,
                )
            elif self.mode == "draft":
                if self.draft_source_path is None or self.draft_destination is None:
                    raise ValueError(
                        "No source image or draft destination was selected"
                    )
                config = create_optical_facies_draft(
                    self.draft_source_path,
                    self.draft_destination,
                    width_m=float(self.draft_width_m or 0.0),
                    height_m=float(self.draft_height_m or 0.0),
                    depth_m=float(self.draft_depth_m or 0.0),
                    optical_classes=int(self.draft_classes),
                    rig_type=self.rig_type,
                )
            else:
                config = self.config
            validation = validate_geometry_config(config)
            preview = None
            preview_error = ""
            try:
                preview = load_geometry_preview(
                    config,
                    run_id=(
                        self.preview_run_id if self.mode == "preview_run" else None
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                preview_error = f"{type(exc).__name__}: {exc}"
            self.finished.emit(
                {
                    "mode": self.mode,
                    "config": config,
                    "validation": validation,
                    "preview": preview,
                    "preview_error": preview_error,
                    "preview_only": self.mode in {"preview_run", "preview_injector"},
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class FaciesAnnotationExportWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        project: FluidFlowerProject,
        dataset_root: Path,
        geometry_revision: Mapping[str, Any],
        split: str,
        group_id: str,
        actor: str,
        include_registered_frame: bool,
    ) -> None:
        super().__init__()
        self.project = project
        self.dataset_root = Path(dataset_root)
        self.geometry_revision = deepcopy_settings(geometry_revision)
        self.split = split
        self.group_id = group_id
        self.actor = actor
        self.include_registered_frame = include_registered_frame

    @Slot()
    def run(self) -> None:
        try:
            store = FaciesAnnotationStore(self.dataset_root)
            record = store.export(
                self.project,
                self.geometry_revision,
                split=self.split,
                group_id=self.group_id,
                actor=self.actor,
                include_registered_frame=self.include_registered_frame,
            )
            self.finished.emit(
                {
                    "record": record,
                    "summary": store.summary(),
                    "dataset_root": str(store.root),
                }
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class FaciesModelTrainingWorker(QObject):
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        dataset_roots: list[Path],
        output_root: Path,
        config: FaciesModelConfig,
        split_plan: dict[str, Any] | None,
    ) -> None:
        super().__init__()
        self.dataset_roots = list(dataset_roots)
        self.output_root = Path(output_root)
        self.config = config
        self.split_plan = deepcopy_settings(split_plan)

    @Slot()
    def run(self) -> None:
        try:
            result = train_facies_baseline(
                self.dataset_roots,
                self.output_root,
                config=self.config,
                split_plan=self.split_plan,
                progress=self.progress.emit,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class FaciesExamplePredictionWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        *,
        session: Mapping[str, Any],
        destination_root: Path,
        mode: str,
        cpu_workers: int = 0,
    ) -> None:
        super().__init__()
        self.session = deepcopy_settings(session)
        self.destination_root = Path(destination_root)
        self.mode = str(mode)
        self.cpu_workers = int(cpu_workers)

    @Slot()
    def run(self) -> None:
        try:
            result = create_facies_example_prediction(
                self.session,
                self.destination_root,
                mode=self.mode,
                cpu_workers=self.cpu_workers,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")


class GeometryFaciesPage(ModulePage):
    def __init__(
        self,
        project: FluidFlowerProject,
        *,
        subpage_registry: FluidFlowerSubpageRegistry | None = None,
    ) -> None:
        super().__init__(project, "geometry_facies")
        self.subpage_registry = subpage_registry or default_subpage_registry()
        self.store = GeometryRevisionStore(project)
        self.config = self.store.current_config()
        self.facies_example_store = FaciesExampleStore(project)
        try:
            self.facies_example_session = self.facies_example_store.current()
            self._facies_example_load_error = ""
        except Exception as exc:  # noqa: BLE001
            self.facies_example_session = default_facies_example_session(project)
            self._facies_example_load_error = f"{type(exc).__name__}: {exc}"
        self.validation: dict[str, Any] | None = None
        current = self.store.current()
        if current.get("format") == GEOMETRY_REVISION_FORMAT:
            self.validation = deepcopy_settings(current.get("validation") or {})
        self.preview_data: dict[str, Any] | None = None
        self._thread: QThread | None = None
        self._worker: GeometryWorker | None = None
        self._facies_export_thread: QThread | None = None
        self._facies_export_worker: FaciesAnnotationExportWorker | None = None
        self._facies_model_thread: QThread | None = None
        self._facies_model_worker: FaciesModelTrainingWorker | None = None
        self._facies_example_thread: QThread | None = None
        self._facies_example_worker: FaciesExamplePredictionWorker | None = None
        self._facies_example_result: dict[str, Any] | None = None
        self._facies_example_applying = False
        self._applying = False
        self.label_session: GeometryLabelEditSession | None = None
        self._label_edits_need_materialization = False
        self.work_area.setVisible(True)
        self.layout().takeAt(self.layout().count() - 1)
        self.integration.setText(
            "Detailed labels preserve physical layers and curvature; facies may share material "
            "properties or remain layer-specific. Saved revisions materialize a canonical DarSIA "
            "facies map while all imported rig files remain unchanged."
        )

        action_row = QHBoxLayout()
        self.import_rig = QPushButton("Import DarSIA rig")
        self.import_rig.clicked.connect(self._import_rig)
        action_row.addWidget(self.import_rig)
        self.validate_button = QPushButton("Validate and preview")
        self.validate_button.clicked.connect(self._start_validation)
        action_row.addWidget(self.validate_button)
        self.save_button = QPushButton("Save revision")
        self.save_button.clicked.connect(self._save_revision)
        self.save_button.setEnabled(False)
        action_row.addWidget(self.save_button)
        self.canonical_edit_button = QPushButton("Create canonical edit draft")
        self.canonical_edit_button.setToolTip(
            "Explicitly derive edited facies from labels and the full original CSV; "
            "the imported parent and boundary differences remain recorded. This does not register geometry.")
        self.canonical_edit_button.clicked.connect(self._begin_canonical_edit)
        self.canonical_edit_button.setVisible(False)
        action_row.addWidget(self.canonical_edit_button)
        self.busy = QProgressBar()
        self.busy.setRange(0, 1)
        self.busy.setValue(0)
        self.busy.setMaximumWidth(170)
        action_row.addWidget(self.busy)
        self.geometry_summary = QLabel("No validated geometry revision.")
        self.geometry_summary.setObjectName("secondaryText")
        action_row.addWidget(self.geometry_summary, 1)
        self.actor = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.actor.setPlaceholderText("Reviewer")
        self.actor.setMaximumWidth(220)
        action_row.addWidget(self.actor)
        self.work_layout.addLayout(action_row)

        self.geometry_quality_basis = QLabel(
            "Working basis: native registered image grid | categorical layer and "
            "facies labels | numerical depth, porosity, and permeability retain "
            "their recorded artifact precision."
        )
        self.geometry_quality_basis.setObjectName("integrationNotice")
        self.geometry_quality_basis.setWordWrap(True)
        self.work_layout.addWidget(self.geometry_quality_basis)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.canvas = GeometryImageView()
        self.canvas.physical_point_selected.connect(self._injector_placed)
        self.canvas.label_selected.connect(self._layer_selected)
        self.canvas.label_edit_drawn.connect(self._label_edit_drawn)
        self.canvas.facies_example_box_drawn.connect(
            self._facies_example_box_drawn
        )
        left_layout.addWidget(self.canvas, 1)
        display_row = QHBoxLayout()
        display_row.addWidget(QLabel("Background"))
        self.background = QComboBox()
        self.background.addItem("Baseline", "baseline")
        self.background.addItem("Registered / early CO2 frame", "review_frame")
        self.background.currentIndexChanged.connect(self._display_changed)
        display_row.addWidget(self.background)
        display_row.addWidget(QLabel("Overlay"))
        self.overlay = QComboBox()
        for label, value in (
            ("Baseline", "baseline"),
            ("Physical layers", "labels"),
            ("Facies", "facies"),
            ("Material proposal", "facies_proposal"),
            ("Proposal uncertainty", "facies_uncertainty"),
            ("Depth", "depth"),
            ("Porosity", "porosity"),
            ("Permeability", "permeability"),
        ):
            self.overlay.addItem(label, value)
        self.overlay.setCurrentIndex(self.overlay.findData("facies"))
        self.overlay.currentIndexChanged.connect(self._display_changed)
        display_row.addWidget(self.overlay)
        display_row.addWidget(QLabel("Opacity"))
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        self.opacity.setFixedWidth(130)
        self.opacity.valueChanged.connect(self._display_changed)
        display_row.addWidget(self.opacity)
        self.boundaries = QCheckBox("Boundaries only")
        self.boundaries.stateChanged.connect(self._display_changed)
        display_row.addWidget(self.boundaries)
        self.show_injectors = QCheckBox("Injection points")
        self.show_injectors.setChecked(True)
        self.show_injectors.stateChanged.connect(self._display_changed)
        display_row.addWidget(self.show_injectors)
        self.place_injector = QPushButton("Place selected injection point")
        self.place_injector.setCheckable(True)
        self.place_injector.setToolTip(
            "Select an injection point in the table, activate this button, then click its "
            "position in the image."
        )
        self.place_injector.toggled.connect(self._placement_toggled)
        display_row.addWidget(self.place_injector)
        self.select_layer = QPushButton("Select layer")
        self.select_layer.setCheckable(True)
        self.select_layer.toggled.connect(self._selection_toggled)
        display_row.addWidget(self.select_layer)
        fit = QPushButton("Fit")
        fit.clicked.connect(self.canvas.fit_image)
        display_row.addWidget(fit)
        display_row.addStretch(1)
        left_layout.addLayout(display_row)
        splitter.addWidget(left)

        self.tabs = QTabWidget()
        self.tabs.setMinimumWidth(470)
        self.subpage_registry.populate_tabs("geometry_facies", self, self.tabs)
        self._initialize_facies_example_ui()
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        self.work_layout.addWidget(splitter, 1)

        self._connect_dirty_signals()
        self._populate_config(self.config)
        self._populate_history()
        self._show_validation(self.validation)
        self.refresh()







    def _append_run_geometry_row(self, row: Mapping[str, Any]) -> None:
        index = self.run_geometry_table.rowCount()
        self.run_geometry_table.insertRow(index)
        enabled = QTableWidgetItem("")
        enabled.setFlags(enabled.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        enabled.setCheckState(
            Qt.CheckState.Checked
            if bool(row.get("enabled", True))
            else Qt.CheckState.Unchecked
        )
        self.run_geometry_table.setItem(index, 0, enabled)
        run_item = QTableWidgetItem(str(row.get("run_id") or ""))
        # Keep the original immutable registration/provenance with this row.
        # Reading visible controls must not silently discard its source proof.
        run_item.setData(Qt.ItemDataRole.UserRole, deepcopy_settings(dict(row)))
        self.run_geometry_table.setItem(index, 1, run_item)
        acquisition = QComboBox()
        for value in RUN_GEOMETRY_ACQUISITION_STATES:
            acquisition.addItem(value.title(), value)
        selected = acquisition.findData(str(row.get("acquisition_state") or "unknown"))
        acquisition.setCurrentIndex(max(0, selected))
        acquisition.currentIndexChanged.connect(self._mark_dirty)
        self.run_geometry_table.setCellWidget(index, 2, acquisition)
        geometry_choice = QComboBox()
        for label, value in (
            ("Series reference", "series_reference"),
            ("Reviewed run map", "run_override"),
        ):
            if value in RUN_GEOMETRY_CHOICES:
                geometry_choice.addItem(label, value)
        inferred_choice = str(row.get("geometry_choice") or "")
        if inferred_choice not in RUN_GEOMETRY_CHOICES:
            inferred_choice = (
                "run_override"
                if str(row.get("labels_path") or "").strip()
                or str(row.get("facies_path") or "").strip()
                else "series_reference"
            )
        selected = geometry_choice.findData(inferred_choice)
        geometry_choice.setCurrentIndex(max(0, selected))
        geometry_choice.currentIndexChanged.connect(self._mark_dirty)
        self.run_geometry_table.setCellWidget(index, 3, geometry_choice)
        for column, key in (
            (4, "source_image_path"),
            (5, "labels_path"),
            (6, "facies_path"),
        ):
            self.run_geometry_table.setItem(
                index, column, QTableWidgetItem(str(row.get(key) or ""))
            )
        screen = QTableWidgetItem("Not checked")
        screen.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        screen.setToolTip(
            "Validation compares this run's layer labels with the approved series map."
        )
        self.run_geometry_table.setItem(index, 7, screen)
        properties = QTableWidgetItem("Inherited (locked)")
        properties.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
        properties.setToolTip(
            "Layer packing may move boundaries, but it does not silently change "
            "porosity or permeability."
        )
        self.run_geometry_table.setItem(index, 8, properties)

    def _populate_run_geometry(self, raw_contract: Mapping[str, Any] | None) -> None:
        contract = normalize_run_geometry_config(raw_contract)
        self.run_geometry_table.blockSignals(True)
        try:
            policy_index = self.run_geometry_policy.findData(
                str(contract.get("geometry_policy") or "allow_reviewed_run_overrides")
            )
            self.run_geometry_policy.setCurrentIndex(max(0, policy_index))
            detection = dict(contract.get("change_detection") or {})
            self.run_geometry_change_detection.setChecked(
                bool(detection.get("enabled", True))
            )
            self.run_geometry_displacement_threshold.setValue(
                100.0
                * float(detection.get("warning_displacement_fraction_of_height", 0.005))
            )
            self.run_geometry_affected_threshold.setValue(
                100.0 * float(detection.get("minimum_affected_boundary_fraction", 0.10))
            )
            self.run_geometry_table.setRowCount(0)
            for row in contract.get("runs") or []:
                self._append_run_geometry_row(row)
        finally:
            self.run_geometry_table.blockSignals(False)

    def _collect_run_geometry(self) -> dict[str, Any]:
        contract = normalize_run_geometry_config(
            getattr(self, "config", {}).get("run_geometry")
        )
        contract["geometry_policy"] = str(
            self.run_geometry_policy.currentData() or "allow_reviewed_run_overrides"
        )
        contract["change_detection"].update({
            "enabled": self.run_geometry_change_detection.isChecked(),
            "warning_displacement_fraction_of_height": (
                self.run_geometry_displacement_threshold.value() / 100.0
            ),
            "minimum_affected_boundary_fraction": (
                self.run_geometry_affected_threshold.value() / 100.0
            ),
        })
        rows: list[dict[str, Any]] = []
        for index in range(self.run_geometry_table.rowCount()):
            acquisition = self.run_geometry_table.cellWidget(index, 2)
            geometry_choice = self.run_geometry_table.cellWidget(index, 3)
            run_item = self.run_geometry_table.item(index, 1)
            run_id = run_item.text().strip().casefold()
            original = run_item.data(Qt.ItemDataRole.UserRole)
            # Renaming a row cannot transfer another run's registration. The
            # unchanged strict source validator then requires genuine new proof.
            row = (deepcopy_settings(original) if isinstance(original, Mapping)
                   and str(original.get("run_id") or "").strip().casefold() == run_id else {})
            values = {
                    "enabled": self.run_geometry_table.item(index, 0).checkState()
                    == Qt.CheckState.Checked,
                    "run_id": run_id,
                    "acquisition_state": str(
                        acquisition.currentData()
                        if isinstance(acquisition, QComboBox)
                        else "unknown"
                    ),
                    "geometry_choice": str(
                        geometry_choice.currentData()
                        if isinstance(geometry_choice, QComboBox)
                        else "series_reference"
                    ),
                    "source_image_path": self.run_geometry_table.item(index, 4)
                    .text()
                    .strip(),
                    "labels_path": self.run_geometry_table.item(index, 5)
                    .text()
                    .strip(),
                    "facies_path": self.run_geometry_table.item(index, 6)
                    .text()
                    .strip(),
                }
            if row:
                # Empty optional fields were absent in older saved revisions.
                # Opening the editor must not add them or erase opaque proposals.
                defaults = {"enabled": True, "acquisition_state": "unknown",
                    "geometry_choice": "series_reference", "source_image_path": "",
                    "labels_path": "", "facies_path": ""}
                for key, value in values.items():
                    if key in row or value != defaults.get(key):
                        row[key] = value
            else:
                row.update(values, property_change_proposals=[])
            rows.append(row)
        contract["runs"] = rows
        return contract

    def _add_project_run_geometry_rows(self) -> None:
        existing = {
            self.run_geometry_table.item(row, 1).text().strip().casefold()
            for row in range(self.run_geometry_table.rowCount())
            if self.run_geometry_table.item(row, 1) is not None
        }
        self._applying = True
        try:
            for row in self.project.manifest.get("runs") or []:
                run_id = str(row.get("run_id") or "").strip().casefold()
                if run_id and run_id not in existing:
                    self._append_run_geometry_row(
                        {
                            "run_id": run_id,
                            "enabled": True,
                            "acquisition_state": str(
                                row.get("indicator_state") or "unknown"
                            )
                            if str(row.get("indicator_state") or "unknown")
                            in RUN_GEOMETRY_ACQUISITION_STATES
                            else "unknown",
                            "geometry_choice": "series_reference",
                        }
                    )
                    existing.add(run_id)
        finally:
            self._applying = False
        self._mark_dirty()

    def _add_custom_run_geometry_row(self) -> None:
        self._append_run_geometry_row(
            {
                "run_id": "run",
                "enabled": True,
                "acquisition_state": "unknown",
                "geometry_choice": "series_reference",
            }
        )
        self._mark_dirty()

    def _remove_run_geometry_rows(self) -> None:
        rows = sorted(
            {index.row() for index in self.run_geometry_table.selectedIndexes()},
            reverse=True,
        )
        for row in rows:
            self.run_geometry_table.removeRow(row)
        if rows:
            self._mark_dirty()

    def _clear_selected_run_geometry_maps(self) -> None:
        rows = {index.row() for index in self.run_geometry_table.selectedIndexes()}
        for row in rows:
            for column in (5, 6):
                self.run_geometry_table.item(row, column).setText("")
            choice = self.run_geometry_table.cellWidget(row, 3)
            if isinstance(choice, QComboBox):
                choice.setCurrentIndex(max(0, choice.findData("series_reference")))
        if rows:
            self._mark_dirty()

    def _browse_run_geometry_cell(self, row: int, column: int) -> None:
        if column not in {4, 5, 6}:
            return
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Select registered run geometry input",
            self.run_geometry_table.item(row, column).text(),
            "Images and arrays (*.npz *.npy *.jpg *.jpeg *.png *.tif *.tiff);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if selected:
            self.run_geometry_table.item(row, column).setText(selected)

    def _preview_selected_run_geometry(self) -> None:
        rows = sorted(
            {index.row() for index in self.run_geometry_table.selectedIndexes()}
        )
        if len(rows) != 1:
            QMessageBox.information(
                self,
                "Select one run",
                "Select exactly one per-run geometry row to preview.",
            )
            return
        run_id = self.run_geometry_table.item(rows[0], 1).text().strip().casefold()
        self._start_worker(
            GeometryWorker(
                mode="preview_run",
                config=self._collect_config(),
                preview_run_id=run_id,
            )
        )


    def _connect_dirty_signals(self) -> None:
        for field in (
            self.rig_folder,
            self.baseline_path,
            self.depth_path,
            self.labels_path,
            self.facies_path,
            self.properties_path,
            self.protocol_path,
            self.review_frame_path,
        ):
            field.changed.connect(self._mark_dirty)
        self.width_m.valueChanged.connect(self._mark_dirty)
        self.height_m.valueChanged.connect(self._mark_dirty)
        self.grouping_mode.currentIndexChanged.connect(self._mark_dirty)
        self.permeability_unit.currentTextChanged.connect(self._mark_dirty)
        self.layers.itemChanged.connect(self._mark_dirty)
        self.injector_table.itemChanged.connect(self._injector_item_changed)
        self.run_geometry_table.itemChanged.connect(self._mark_dirty)

    def _populate_config(self, config: Mapping[str, Any]) -> None:
        incoming_reference = dict(config.get("label_edit_session") or {})
        incoming_session_id = str(incoming_reference.get("session_id") or "")
        if self.label_session is not None and (
            not incoming_session_id
            or incoming_session_id != self.label_session.session_id
        ):
            self.label_session = None
        self._applying = True
        try:
            self.config = deepcopy_settings(config)
            self.rig_folder.setText(str(config.get("source_rig_path") or ""))
            self.baseline_path.setText(str(config.get("baseline_path") or ""))
            self.depth_path.setText(str(config.get("depth_path") or ""))
            self.labels_path.setText(str(config.get("labels_path") or ""))
            self.facies_path.setText(str(config.get("facies_source_path") or ""))
            self.properties_path.setText(
                str(config.get("facies_properties_source_path") or "")
            )
            self.protocol_path.setText(
                str(
                    config.get("protocol_source_path")
                    or self._registered_protocol_path()
                    or ""
                )
            )
            self.review_frame_path.setText(str(config.get("display_image_path") or ""))
            self.draft_source_path.setText(
                str(config.get("draft_source_image_path") or "")
            )
            self.width_m.setValue(_gui_float(config.get("width_m")) or 0.0)
            self.height_m.setValue(_gui_float(config.get("height_m")) or 0.0)
            grouping_index = self.grouping_mode.findData(
                str(config.get("grouping_mode") or "shared_facies")
            )
            self.grouping_mode.setCurrentIndex(max(0, grouping_index))
            unit = str(config.get("permeability_unit") or "source_unit")
            unit_index = self.permeability_unit.findText(unit)
            if unit_index < 0:
                self.permeability_unit.addItem(unit)
                unit_index = self.permeability_unit.findText(unit)
            self.permeability_unit.setCurrentIndex(unit_index)
            self._populate_layers(list(config.get("regions") or []))
            self._populate_injectors(list(config.get("injectors") or []))
            self._populate_run_geometry(config.get("run_geometry"))
        finally:
            self._applying = False
        materialized_count = int(
            incoming_reference.get("materialized_active_operation_count") or 0
        )
        active_count = int(incoming_reference.get("active_operation_count") or 0)
        self._label_edits_need_materialization = active_count != materialized_count
        if incoming_session_id and self.label_session is None:
            self.label_edit_status.setText(
                f"Saved edit session {incoming_session_id[:8]} is available."
            )
        self._refresh_label_edit_controls()
        self._sync_imported_source_mode()

    def _sync_imported_source_mode(self) -> None:
        imported = self.config.get("source_mode") == "imported-rasters"
        self.canonical_edit_button.setVisible(imported)
        previous = getattr(self, "_imported_editor_enabled_states", [])
        if not imported:
            for widget, enabled in previous:
                widget.setEnabled(enabled)
            self._imported_editor_enabled_states = []
            if hasattr(self, "_imported_quality_basis_before"):
                self.geometry_quality_basis.setText(self._imported_quality_basis_before)
            return
        if not previous:
            editor_types = (QAbstractButton, QAbstractSpinBox, QComboBox, QLineEdit, QTableWidget)
            view_controls = {getattr(self, name, None) for name in ("background", "overlay", "revisions")}
            previous = []
            for widget in self.tabs.findChildren(QWidget):
                if not isinstance(widget, editor_types) or widget in view_controls:
                    continue
                parent = widget.parentWidget()
                while parent is not None and parent is not self.tabs:
                    if isinstance(parent, editor_types):
                        break
                    parent = parent.parentWidget()
                else:
                    parent = None
                # A table owns transient cell editors. Lock its persistent table,
                # not wrappers which a later _populate_layers call replaces.
                if parent is None or parent is self.tabs:
                    previous.append((widget, widget.isEnabled()))
            for name in ("import_rig", "place_injector"):
                button = getattr(self, name, None)
                if button is not None and all(button is not widget for widget, _enabled in previous):
                    previous.append((button, button.isEnabled()))
            self._imported_editor_enabled_states = previous
            self._imported_quality_basis_before = self.geometry_quality_basis.text()
        for widget, _enabled in previous:
            widget.setEnabled(False)
        self.save_button.setEnabled(False)
        self.geometry_quality_basis.setText(
            "Imported rasters: labels and facies are independent authoritative maps; "
            "the FULL original properties CSV is retained. Fields are read-only. "
            "Create a canonical edit draft to edit; registered geometry still requires the project owner's decision.")

    def _begin_canonical_edit(self) -> None:
        from darsia.fluidflower.geometry_raster_sources import prepare_canonical_edit
        try:
            draft = prepare_canonical_edit(
                self.project, actor=self.actor.text().strip(),
                reason=self.reason.text().strip() or "User requested an explicit canonical edit draft")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not create canonical edit draft", str(exc))
            return
        self._populate_config(draft)
        self.validation = None
        self.save_button.setEnabled(False)
        self.geometry_quality_basis.setText(
            "Explicit canonical edit draft. Future facies are derived from edited labels and the "
            "FULL original CSV; Save records the difference to the imported parent. "
            "This is not registered-v2 and does not authorize a correction.")
        self.geometry_summary.setText("Canonical edit draft; validate before saving a new revision.")

    def _populate_layers(self, rows: list[dict[str, Any]]) -> None:
        self.layers.setRowCount(len(rows))
        for index, row in enumerate(rows):
            enabled = QTableWidgetItem("")
            enabled.setFlags(enabled.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            enabled.setCheckState(
                Qt.CheckState.Checked
                if row.get("enabled", True)
                else Qt.CheckState.Unchecked
            )
            self.layers.setItem(index, 0, enabled)
            values = (
                str(row.get("label_id") if row.get("label_id") is not None else ""),
                str(row.get("facies_id") if row.get("facies_id") is not None else ""),
                str(row.get("name") or ""),
            )
            for column, value in enumerate(values, start=1):
                self.layers.setItem(index, column, QTableWidgetItem(value))
            role = QComboBox()
            for value in sorted(ANALYSIS_ROLES):
                role.addItem(value.replace("_", " ").title(), value)
            role_index = role.findData(str(row.get("analysis_role") or "excluded"))
            role.setCurrentIndex(max(0, role_index))
            role.currentIndexChanged.connect(self._mark_dirty)
            self.layers.setCellWidget(index, 4, role)
            for column, key in ((5, "porosity"), (6, "permeability")):
                value = row.get(key)
                self.layers.setItem(
                    index,
                    column,
                    QTableWidgetItem("" if value is None else f"{float(value):.17g}"),
                )
        if hasattr(self, "edit_target"):
            self._refresh_label_targets()

    def _populate_injectors(self, rows: list[dict[str, Any]]) -> None:
        self.injector_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            values = (
                str(
                    row.get("injector_id")
                    or row.get("protocol_location_id")
                    or f"ip{index + 1}"
                ),
                str(row.get("name") or f"Injection Point {index + 1}"),
                _table_number(row.get("x_m")),
                _table_number(row.get("y_m")),
                _table_number(row.get("uncertainty_m")),
                str(row.get("coordinate_source") or "manual"),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        str(
                            row.get("injector_id")
                            or row.get("protocol_location_id")
                            or f"ip{index + 1}"
                        ),
                    )
                    item.setData(INJECTOR_RECORD_ROLE, deepcopy_settings(row))
                self.injector_table.setItem(index, column, item)
        self._update_injector_position_status()

    def _registered_protocol_path(self) -> str:
        sources = list(self.project.source_manifest().get("sources") or [])
        source_id = str(
            (self.project.manifest.get("protocol") or {}).get("source_id") or ""
        )
        selected = next(
            (
                row
                for row in sources
                if source_id and str(row.get("source_id") or "") == source_id
            ),
            None,
        )
        if selected is None:
            selected = next(
                (row for row in sources if str(row.get("kind") or "") == "protocol"),
                None,
            )
        return str((selected or {}).get("path") or "")

    def _load_injection_points_from_protocol(self) -> None:
        source_text = (
            self.protocol_path.text().strip() or self._registered_protocol_path()
        )
        if not source_text:
            QMessageBox.warning(
                self,
                "Excel protocol required",
                "Select an Excel protocol in Inputs before loading injection points.",
            )
            return
        source = Path(source_text).expanduser()
        try:
            incoming = load_protocol_injection_points(source)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Could not read injection points",
                f"The protocol could not be read:\n{type(exc).__name__}: {exc}",
            )
            return
        if not incoming:
            QMessageBox.warning(
                self,
                "No injection points found",
                "No ip1/ip2-style locations or usable legacy injection-point rows "
                "were found in the selected protocol.",
            )
            return

        current = list(self._collect_config().get("injectors") or [])
        current_by_id = {
            str(row.get("injector_id") or row.get("protocol_location_id") or "")
            .strip()
            .casefold(): row
            for row in current
            if str(
                row.get("injector_id") or row.get("protocol_location_id") or ""
            ).strip()
        }
        merged_rows: list[dict[str, Any]] = []
        consumed: set[str] = set()
        for protocol_row in incoming:
            location_id = str(
                protocol_row.get("injector_id")
                or protocol_row.get("protocol_location_id")
                or ""
            ).strip()
            key = location_id.casefold()
            existing = deepcopy_settings(current_by_id.get(key) or {})
            merged = existing
            merged["injector_id"] = location_id
            merged["protocol_location_id"] = str(
                protocol_row.get("protocol_location_id") or location_id
            )
            merged["name"] = str(protocol_row.get("name") or existing.get("name") or "")
            for field in (
                "protocol_x_m",
                "protocol_y_m",
                "protocol_coordinate_candidates",
            ):
                merged[field] = deepcopy(protocol_row.get(field))

            current_x = _gui_float(existing.get("x_m"))
            current_y = _gui_float(existing.get("y_m"))
            current_source = str(existing.get("coordinate_source") or "")
            preserve_manual = (
                current_x is not None
                and current_y is not None
                and not current_source.casefold().startswith("protocol")
            )
            if preserve_manual:
                merged["x_m"] = current_x
                merged["y_m"] = current_y
                merged["coordinate_source"] = current_source
            else:
                merged["x_m"] = _gui_float(protocol_row.get("x_m"))
                merged["y_m"] = _gui_float(protocol_row.get("y_m"))
                merged["coordinate_source"] = str(
                    protocol_row.get("coordinate_source") or "protocol"
                )
            merged_rows.append(merged)
            consumed.add(key)

        merged_rows.extend(
            deepcopy_settings(row)
            for row in current
            if str(row.get("injector_id") or row.get("protocol_location_id") or "")
            .strip()
            .casefold()
            not in consumed
        )
        self._applying = True
        try:
            self.protocol_path.setText(str(source))
            self._populate_injectors(merged_rows)
        finally:
            self._applying = False
        self.config = self._collect_config()
        self._mark_dirty()
        if self.injector_table.rowCount():
            self.injector_table.selectRow(0)
        self._update_injector_position_status()

    def _collect_config(self) -> dict[str, Any]:
        if self.config.get("source_mode") == "imported-rasters":
            # Visible read-only editor until the explicit canonical draft action.
            return deepcopy_settings(self.config)
        regions: list[dict[str, Any]] = []
        for row in range(self.layers.rowCount()):
            role = self.layers.cellWidget(row, 4)
            regions.append(
                {
                    "label_id": _gui_int(self.layers.item(row, 1).text()),
                    "facies_id": _gui_int(self.layers.item(row, 2).text()),
                    "name": self.layers.item(row, 3).text().strip(),
                    "analysis_role": str(
                        role.currentData()
                        if isinstance(role, QComboBox)
                        else "excluded"
                    ),
                    "porosity": _gui_float(self.layers.item(row, 5).text()),
                    "permeability": _gui_float(self.layers.item(row, 6).text()),
                    "enabled": self.layers.item(row, 0).checkState()
                    == Qt.CheckState.Checked,
                }
            )
        injectors: list[dict[str, Any]] = []
        for row in range(self.injector_table.rowCount()):
            identity_item = self.injector_table.item(row, 0)
            name_item = self.injector_table.item(row, 1)
            original = identity_item.data(INJECTOR_RECORD_ROLE)
            injector = (
                deepcopy_settings(original) if isinstance(original, Mapping) else {}
            )
            location_id = identity_item.text().strip()
            injector.update(
                {
                    "injector_id": location_id
                    or str(
                        identity_item.data(Qt.ItemDataRole.UserRole) or uuid.uuid4().hex
                    ),
                    "name": name_item.text().strip(),
                    "x_m": _gui_float(self.injector_table.item(row, 2).text()),
                    "y_m": _gui_float(self.injector_table.item(row, 3).text()),
                    "uncertainty_m": _gui_float(
                        self.injector_table.item(row, 4).text()
                    ),
                    "coordinate_source": self.injector_table.item(row, 5)
                    .text()
                    .strip(),
                }
            )
            injectors.append(injector)
        candidate = deepcopy_settings(self.config)
        candidate.update(
            {
                "schema": 1,
                "format": "darsia-fluidflower-geometry-config-v1",
                "rig_type": str(
                    (self.project.manifest.get("rig") or {}).get("type") or "custom"
                ),
                "source_rig_path": self.rig_folder.text(),
                "baseline_path": self.baseline_path.text(),
                "depth_path": self.depth_path.text(),
                "labels_path": self.labels_path.text(),
                "facies_source_path": self.facies_path.text(),
                "facies_properties_source_path": self.properties_path.text(),
                "protocol_source_path": self.protocol_path.text(),
                "display_image_path": self.review_frame_path.text(),
                "width_m": self.width_m.value() or None,
                "height_m": self.height_m.value() or None,
                "coordinate_convention": "origin_lower_left_x_right_y_up",
                "grouping_mode": str(self.grouping_mode.currentData()),
                "permeability_unit": self.permeability_unit.currentText().strip()
                or "source_unit",
                "regions": regions,
                "injectors": injectors,
                "run_geometry": self._collect_run_geometry(),
            }
        )
        generation = dict(candidate.get("facies_generation") or {})
        if generation and any(
            bool(row.get("enabled"))
            and str(row.get("analysis_role") or "") in {"porous_medium", "seal"}
            for row in regions
        ):
            generation["status"] = "expert_edited_draft"
            candidate["facies_generation"] = generation
        return candidate

    def _injector_item_changed(self, item: QTableWidgetItem) -> None:
        if self._applying:
            return
        if item.column() in {2, 3}:
            identity_item = self.injector_table.item(item.row(), 0)
            original = (
                identity_item.data(INJECTOR_RECORD_ROLE)
                if identity_item is not None
                else None
            )
            metadata = dict(original) if isinstance(original, Mapping) else {}
            protocol_x = _gui_float(metadata.get("protocol_x_m"))
            protocol_y = _gui_float(metadata.get("protocol_y_m"))
            current_x = _gui_float(self.injector_table.item(item.row(), 2).text())
            current_y = _gui_float(self.injector_table.item(item.row(), 3).text())
            tolerance_m = max(
                INJECTION_POINT_CONFLICT_TOLERANCE_M,
                _gui_float(self.injector_table.item(item.row(), 4).text()) or 0.0,
            )
            source_item = self.injector_table.item(item.row(), 5)
            if (
                source_item is not None
                and current_x is not None
                and current_y is not None
            ):
                matches_protocol = (
                    protocol_x is not None
                    and protocol_y is not None
                    and math.hypot(current_x - protocol_x, current_y - protocol_y)
                    <= tolerance_m
                )
                source = (
                    str(metadata.get("coordinate_source") or "protocol")
                    if matches_protocol
                    else "manual_coordinate_entry"
                )
                self._applying = True
                try:
                    source_item.setText(source)
                finally:
                    self._applying = False
        self._update_injector_position_status()
        self._mark_dirty()

    def _update_injector_position_status(self) -> None:
        if not hasattr(self, "injector_position_status"):
            return
        pending: list[str] = []
        conflicts: list[str] = []
        for row in range(self.injector_table.rowCount()):
            identity_item = self.injector_table.item(row, 0)
            if identity_item is None:
                continue
            location_id = identity_item.text().strip() or f"row {row + 1}"
            current_x = _gui_float(self.injector_table.item(row, 2).text())
            current_y = _gui_float(self.injector_table.item(row, 3).text())
            if current_x is None or current_y is None:
                pending.append(location_id)
                continue
            original = identity_item.data(INJECTOR_RECORD_ROLE)
            metadata = dict(original) if isinstance(original, Mapping) else {}
            protocol_x = _gui_float(metadata.get("protocol_x_m"))
            protocol_y = _gui_float(metadata.get("protocol_y_m"))
            tolerance_m = max(
                INJECTION_POINT_CONFLICT_TOLERANCE_M,
                _gui_float(self.injector_table.item(row, 4).text()) or 0.0,
            )
            if (
                protocol_x is not None
                and protocol_y is not None
                and math.hypot(current_x - protocol_x, current_y - protocol_y)
                > tolerance_m
            ):
                conflicts.append(location_id)
        if conflicts:
            self.injector_position_status.setText(
                "Warning: GUI and Excel positions differ for "
                f"{', '.join(conflicts)}. Both positions are retained in provenance; "
                "Geometry validation will report the difference."
            )
            self.injector_position_status.setStyleSheet("color: #e3b65c;")
        elif pending:
            self.injector_position_status.setText(
                f"Position still required for {', '.join(pending)}. Enter x/y or "
                "select a row and place it on the image."
            )
            self.injector_position_status.setStyleSheet("color: #e3b65c;")
        else:
            self.injector_position_status.setText(
                "Injection-point positions are defined. Red markers update when x/y changes."
            )
            self.injector_position_status.setStyleSheet("")

    def _mark_dirty(self, *_args: Any) -> None:
        if self._applying:
            return
        self.validation = None
        self.save_button.setEnabled(False)
        self.geometry_summary.setText("Local changes need validation.")
        if self.preview_data is not None:
            self.config = self._collect_config()
            self.canvas.set_config(self.config)

    def _preview_injector_reference(self) -> None:
        config = self._collect_config()
        required = {
            "baseline": str(config.get("baseline_path") or ""),
            "layer labels": str(config.get("labels_path") or ""),
            "depth map": str(config.get("depth_path") or ""),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            QMessageBox.warning(
                self,
                "Geometry inputs required",
                "Add the "
                + ", ".join(missing)
                + " before previewing injection-point placement.",
            )
            return
        placement_path = str(config.get("display_image_path") or "")
        if placement_path and not Path(placement_path).expanduser().is_file():
            QMessageBox.warning(
                self,
                "Placement image unavailable",
                f"The selected placement image could not be found:\n{placement_path}",
            )
            return
        self.tabs.setCurrentIndex(2)
        self._start_worker(GeometryWorker(mode="preview_injector", config=config))

    def _show_injector_baseline(self) -> None:
        if self.preview_data is None:
            self._preview_injector_reference()
            return
        index = self.background.findData("baseline")
        self.background.setCurrentIndex(max(0, index))
        self._display_changed()

    def _refresh_label_targets(self) -> None:
        selected = self.edit_target.currentData()
        self.edit_target.blockSignals(True)
        self.edit_target.clear()
        for row in range(self.layers.rowCount()):
            label_item = self.layers.item(row, 1)
            name_item = self.layers.item(row, 3)
            label_id = _gui_int(label_item.text()) if label_item is not None else None
            if label_id is None:
                continue
            name = name_item.text().strip() if name_item is not None else ""
            self.edit_target.addItem(
                f"{label_id} · {name or f'Layer {label_id}'}", label_id
            )
        restored = self.edit_target.findData(selected)
        self.edit_target.setCurrentIndex(restored if restored >= 0 else 0)
        self.edit_target.blockSignals(False)

    def _refresh_label_edit_controls(self) -> None:
        session = self.label_session
        self.undo_label_edit.setEnabled(bool(session and session.can_undo()))
        self.redo_label_edit.setEnabled(bool(session and session.can_redo()))
        self.materialize_label_edits.setEnabled(
            bool(session and self._label_edits_need_materialization)
        )
        self.label_edit_history.setRowCount(0)
        if session is None:
            return
        operations = {
            str(event.get("event_id") or ""): event
            for event in session.events
            if str(event.get("kind") or "") in {"polygon", "brush"}
        }
        rows = list(reversed(session.events))
        self.label_edit_history.setRowCount(len(rows))
        names = {
            _gui_int(self.layers.item(row, 1).text()): self.layers.item(row, 3)
            .text()
            .strip()
            for row in range(self.layers.rowCount())
            if self.layers.item(row, 1) is not None
            and self.layers.item(row, 3) is not None
        }
        for row_index, event in enumerate(rows):
            kind = str(event.get("kind") or "")
            operation = (
                operations.get(str(event.get("target_event_id") or ""), {})
                if kind in {"undo", "redo"}
                else event
            )
            label_id = _gui_int(operation.get("target_label"))
            values = (
                str(event.get("recorded_utc") or "")[11:19],
                kind.title(),
                (
                    f"{label_id} · {names.get(label_id, f'Layer {label_id}')}"
                    if label_id is not None
                    else ""
                ),
                str(operation.get("affected_pixel_count") or ""),
            )
            for column, value in enumerate(values):
                self.label_edit_history.setItem(
                    row_index, column, QTableWidgetItem(value)
                )

    def _ensure_label_session(self) -> GeometryLabelEditSession | None:
        if self.label_session is not None:
            return self.label_session
        config = self._collect_config()
        labels_path = Path(str(config.get("labels_path") or "")).expanduser()
        if (
            not labels_path.is_file()
            or not config.get("width_m")
            or not config.get("height_m")
        ):
            QMessageBox.warning(
                self,
                "Layer map required",
                "Load a layer-label map and enter the physical rig dimensions first.",
            )
            return None
        try:
            self.label_session = GeometryLabelEditSession.create(
                config,
                self.project.root / "overrides" / "geometry_facies" / "edit_sessions",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not start boundary editing", str(exc))
            return None
        reference = dict(config.get("label_edit_session") or {})
        materialized_count = int(
            reference.get("materialized_active_operation_count") or 0
        )
        self._label_edits_need_materialization = (
            self.label_session.active_operation_count() != materialized_count
        )
        self.config["label_edit_session"] = {
            **reference,
            **self.label_session.config_reference(),
        }
        self._refresh_label_edit_controls()
        return self.label_session

    def _boundary_mode_changed(self, mode: str) -> None:
        if mode != "pan":
            if hasattr(self, "draw_facies_example"):
                self.draw_facies_example.setChecked(False)
            if self.place_injector.isChecked():
                self.place_injector.setChecked(False)
            if self.select_layer.isChecked():
                self.select_layer.setChecked(False)
        self.canvas.set_edit_mode(mode)
        if hasattr(self, "finish_polygon"):
            self.finish_polygon.setEnabled(mode == "polygon")
        if hasattr(self, "brush_radius_mm"):
            self.brush_radius_mm.setEnabled(mode == "brush")
        if hasattr(self, "boundary_tab"):
            self.tabs.setCurrentWidget(self.boundary_tab)

    @Slot(str, object)
    def _label_edit_drawn(self, kind: str, raw_points: object) -> None:
        session = self._ensure_label_session()
        target_label = _gui_int(self.edit_target.currentData())
        if session is None or target_label is None:
            return
        points = list(raw_points)
        points_m = [
            [float(x) * session.width_m, (1.0 - float(y)) * session.height_m]
            for x, y in points
        ]
        try:
            event = session.apply(
                kind=kind,
                points_m=points_m,
                target_label=target_label,
                actor=self.actor.text().strip() or "Unspecified reviewer",
                brush_radius_m=self.brush_radius_mm.value() / 1000.0,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Boundary edit failed", str(exc))
            return
        self.config["label_edit_session"] = session.config_reference()
        self._label_edits_need_materialization = True
        self.validation = None
        self.validate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.geometry_summary.setText(
            "Boundary edits need a revised layer-map materialization."
        )
        self.label_edit_status.setText(
            f"Autosaved {kind} edit · {event['affected_pixel_count']:,} changed pixels · "
            f"{session.active_operation_count()} active operations."
        )
        self._update_label_edit_preview()
        self._refresh_label_edit_controls()

    def _update_label_edit_preview(self) -> None:
        if self.label_session is None or self.preview_data is None:
            return
        preview = dict(self.preview_data)
        shape = tuple(np.asarray(preview["labels"]).shape)
        labels = self.label_session.preview((int(shape[0]), int(shape[1])))
        facies = np.zeros(labels.shape, dtype=np.int32)
        for region in self._collect_config().get("regions") or []:
            label_id = _gui_int(region.get("label_id"))
            facies_id = _gui_int(region.get("facies_id"))
            if label_id is not None and facies_id is not None:
                facies[labels == label_id] = facies_id
        preview["labels"] = labels
        preview["facies"] = facies
        self.preview_data = preview
        self.canvas.update_preview(preview, self._collect_config())

    def _undo_label_edit(self) -> None:
        session = self._ensure_label_session()
        if session is None:
            return
        event = session.undo(actor=self.actor.text().strip() or "Unspecified reviewer")
        if event is None:
            return
        self._after_label_history_change("Undo autosaved.")

    def _redo_label_edit(self) -> None:
        session = self._ensure_label_session()
        if session is None:
            return
        event = session.redo(actor=self.actor.text().strip() or "Unspecified reviewer")
        if event is None:
            return
        self._after_label_history_change("Redo autosaved.")

    def _after_label_history_change(self, message: str) -> None:
        assert self.label_session is not None
        self.config["label_edit_session"] = self.label_session.config_reference()
        self._label_edits_need_materialization = True
        self.validation = None
        self.validate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.geometry_summary.setText(
            "Boundary edits need a revised layer-map materialization."
        )
        self.label_edit_status.setText(
            f"{message} {self.label_session.active_operation_count()} active operations."
        )
        self._update_label_edit_preview()
        self._refresh_label_edit_controls()

    def _add_edit_target(self) -> None:
        config = self._collect_config()
        current_labels = [
            _gui_int(region.get("label_id")) for region in config.get("regions") or []
        ]
        current_facies = [
            _gui_int(region.get("facies_id")) for region in config.get("regions") or []
        ]
        label_id = (
            max((value for value in current_labels if value is not None), default=0) + 1
        )
        facies_id = (
            max((value for value in current_facies if value is not None), default=0) + 1
        )
        config["regions"].append(
            {
                "label_id": label_id,
                "facies_id": facies_id,
                "name": f"Layer {label_id}",
                "analysis_role": "excluded",
                "porosity": None,
                "permeability": None,
                "enabled": False,
            }
        )
        self._applying = True
        try:
            self._populate_layers(config["regions"])
        finally:
            self._applying = False
        self._refresh_label_targets()
        target_index = self.edit_target.findData(label_id)
        self.edit_target.setCurrentIndex(max(0, target_index))
        self._mark_dirty()

    def _materialize_label_edits(self) -> None:
        session = self._ensure_label_session()
        if session is None or not self._label_edits_need_materialization:
            return
        self.materialize_label_edits.setEnabled(False)
        self.busy.setRange(0, 0)
        self.label_edit_status.setText("Writing a new full-resolution layer map...")
        QApplication.processEvents()
        try:
            materialization = session.materialize()
            config = self._collect_config()
            reference = session.config_reference()
            reference.update(
                {
                    "materialization_id": materialization["materialization_id"],
                    "materialization_manifest_path": materialization["manifest_path"],
                    "materialized_active_operation_count": session.active_operation_count(),
                }
            )
            config["labels_path"] = materialization["labels_path"]
            config["facies_source_path"] = ""
            config["label_edit_session"] = reference
            from darsia.fluidflower.geometry_raster_sources import bind_private_manual_label_edit
            config = bind_private_manual_label_edit(self.project, config, session, materialization)
            preview = load_geometry_preview(config)
        except Exception as exc:  # noqa: BLE001
            self.busy.setRange(0, 1)
            self.busy.setValue(0)
            self.materialize_label_edits.setEnabled(True)
            self.label_edit_status.setText("Layer-map materialization failed.")
            QMessageBox.critical(self, "Could not materialize layer map", str(exc))
            return
        self._applying = True
        try:
            self.config = config
            self.labels_path.setText(materialization["labels_path"])
            self.facies_path.setText("")
        finally:
            self._applying = False
        self.preview_data = preview
        self.canvas.update_preview(preview, config)
        self._label_edits_need_materialization = False
        self.validation = None
        self.validate_button.setEnabled(True)
        self.save_button.setEnabled(False)
        self.busy.setRange(0, 1)
        self.busy.setValue(1)
        self.geometry_summary.setText("Revised layer map needs validation.")
        self.label_edit_status.setText(
            f"Materialized {session.active_operation_count()} active operations without "
            "modifying the imported label map."
        )
        self._refresh_label_edit_controls()

    def _initialize_facies_example_ui(self) -> None:
        if not hasattr(self, "facies_materials"):
            return
        self._facies_example_applying = True
        try:
            session = normalize_facies_example_session(
                self.facies_example_session, project=self.project
            )
            if not session["source_image_path"]:
                session["source_image_path"] = str(
                    self.config.get("baseline_path") or ""
                )
            self.facies_example_session = session
            self.facies_example_strategy.setCurrentIndex(
                max(
                    0,
                    self.facies_example_strategy.findData(
                        session["start_strategy"]
                    ),
                )
            )
            self.facies_example_source.setText(session["source_image_path"])
            self.facies_example_model_manifest.setText(
                session["pretrained_model_manifest"]
            )
            self.facies_example_minimum.setValue(
                int(session["minimum_examples_per_material"])
            )
        finally:
            self._facies_example_applying = False
        self._refresh_facies_example_ui()
        if self._facies_example_load_error:
            self.facies_example_status.setText(
                "Saved material examples could not be loaded: "
                + self._facies_example_load_error
            )

    def _collect_facies_example_session(self) -> dict[str, Any]:
        session = deepcopy_settings(self.facies_example_session)
        session.update(
            {
                "start_strategy": str(
                    self.facies_example_strategy.currentData()
                    or "guided_examples"
                ),
                "source_image_path": self.facies_example_source.text(),
                "pretrained_model_manifest": (
                    self.facies_example_model_manifest.text()
                ),
                "minimum_examples_per_material": (
                    self.facies_example_minimum.value()
                ),
            }
        )
        previous = {
            str(row.get("material_id") or ""): dict(row)
            for row in session.get("materials") or []
        }
        materials: list[dict[str, Any]] = []
        for row in range(self.facies_materials.rowCount()):
            name_item = self.facies_materials.item(row, 0)
            role_widget = self.facies_materials.cellWidget(row, 1)
            material_id = str(
                name_item.data(Qt.ItemDataRole.UserRole) if name_item else ""
            ) or uuid.uuid4().hex
            original = previous.get(material_id, {})
            name = name_item.text().strip() if name_item is not None else ""
            materials.append(
                {
                    **original,
                    "material_id": material_id,
                    "name": name,
                    "canonical_key": str(
                        original.get("canonical_key") or name
                    ).strip().casefold(),
                    "analysis_role": str(
                        role_widget.currentData()
                        if isinstance(role_widget, QComboBox)
                        else "porous_medium"
                    ),
                }
            )
        session["materials"] = materials
        return session

    def _save_facies_example_session(
        self,
        reason: str,
        *,
        show_error: bool = False,
        session_override: Mapping[str, Any] | None = None,
    ) -> bool:
        if self._facies_example_applying:
            return False
        actor = self.actor.text().strip()
        if not actor:
            self.facies_example_status.setText(
                "Enter a reviewer name at the top of the page before editing examples."
            )
            return False
        try:
            collected = (
                deepcopy_settings(session_override)
                if session_override is not None
                else self._collect_facies_example_session()
            )
            saved = self.facies_example_store.save(
                collected,
                actor=actor,
                reason=reason,
            )
            self.facies_example_session = normalize_facies_example_session(
                saved, project=self.project
            )
        except Exception as exc:  # noqa: BLE001
            self.facies_example_status.setText(
                f"Material examples were not saved: {type(exc).__name__}: {exc}"
            )
            if show_error:
                QMessageBox.warning(
                    self,
                    "Could not save material examples",
                    f"{type(exc).__name__}: {exc}",
                )
            return False
        return True

    def _facies_example_session_changed(self, *_args: Any) -> None:
        if self._facies_example_applying:
            return
        if self._save_facies_example_session("Updated material-example settings"):
            self._refresh_facies_example_ui()

    def _facies_material_item_changed(self, _item: QTableWidgetItem) -> None:
        if self._facies_example_applying:
            return
        if self._save_facies_example_session("Updated a named sand type"):
            self._refresh_facies_example_ui()

    def _facies_material_role_changed(self, *_args: Any) -> None:
        if self._facies_example_applying:
            return
        if self._save_facies_example_session("Updated a sand-type role"):
            self._refresh_facies_example_ui()

    def _refresh_facies_example_ui(self) -> None:
        if not hasattr(self, "facies_materials"):
            return
        try:
            session = normalize_facies_example_session(
                self.facies_example_session, project=self.project
            )
            quality = facies_example_quality(session)
        except Exception as exc:  # noqa: BLE001
            self.facies_example_status.setText(
                f"Material examples are incomplete: {type(exc).__name__}: {exc}"
            )
            return
        quality_by_id = {
            str(row["material_id"]): row for row in quality["materials"]
        }
        self._facies_example_applying = True
        try:
            self.facies_material_count.setValue(len(session["materials"]))
            self.facies_materials.setRowCount(len(session["materials"]))
            selected_material_id = str(
                self.facies_example_material.currentData() or ""
            )
            self.facies_example_material.clear()
            for row_index, material in enumerate(session["materials"]):
                material_id = str(material["material_id"])
                state = quality_by_id.get(material_id, {})
                name = QTableWidgetItem(str(material["name"]))
                name.setData(Qt.ItemDataRole.UserRole, material_id)
                self.facies_materials.setItem(row_index, 0, name)
                role = QComboBox()
                for role_value in FACIES_MATERIAL_ROLES:
                    role.addItem(role_value.replace("_", " ").title(), role_value)
                role.setCurrentIndex(
                    max(0, role.findData(str(material["analysis_role"])))
                )
                role.currentIndexChanged.connect(
                    self._facies_material_role_changed
                )
                self.facies_materials.setCellWidget(row_index, 1, role)
                values = (
                    str(state.get("example_count") or 0),
                    str(state.get("spatial_bin_count") or 0),
                    "Yes" if state.get("ready") else "Needs examples",
                )
                for column, value in enumerate(values, start=2):
                    item = QTableWidgetItem(value)
                    item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    self.facies_materials.setItem(row_index, column, item)
                self.facies_example_material.addItem(
                    str(material["name"]), material_id
                )
            restored = self.facies_example_material.findData(selected_material_id)
            self.facies_example_material.setCurrentIndex(
                restored if restored >= 0 else 0
            )

            names = {
                str(row["material_id"]): str(row["name"])
                for row in session["materials"]
            }
            self.facies_examples.setRowCount(len(session["examples"]))
            for row_index, example in enumerate(session["examples"]):
                values = (
                    names.get(str(example["material_id"]), "Unknown"),
                    Path(str(example["source_image_path"])).name,
                    f"{float(example['x']):.3f}",
                    f"{float(example['y']):.3f}",
                    f"{float(example['width']):.3f} x {float(example['height']):.3f}",
                    str(example.get("created_by") or ""),
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setToolTip(
                        str(example["source_image_path"])
                        if column == 1
                        else value
                    )
                    item.setData(
                        Qt.ItemDataRole.UserRole, str(example["example_id"])
                    )
                    self.facies_examples.setItem(row_index, column, item)
        finally:
            self._facies_example_applying = False
        ready = bool(quality["ready_for_guided_prediction"])
        material_summary = ", ".join(
            f"{row['name']}: {row['example_count']}/{row['minimum_count']} boxes, "
            f"{row['spatial_bin_count']}/{row['minimum_spatial_bins']} areas"
            for row in quality["materials"]
        )
        self.facies_example_quality_status.setText(
            ("Ready for guided proposal. " if ready else "More examples required. ")
            + (material_summary or "Define at least two named sand types.")
        )
        self.generate_guided_facies.setEnabled(
            ready and self._facies_example_thread is None
        )
        self.generate_blind_facies.setEnabled(
            bool(session["pretrained_model_manifest"])
            and self._facies_example_thread is None
        )
        latest = str(session.get("latest_prediction_manifest") or "")
        self.apply_facies_example_prediction.setEnabled(Path(latest).is_file())
        self._refresh_facies_example_overlay()

    def _set_facies_material_count(self) -> None:
        session = self._collect_facies_example_session()
        materials = list(session.get("materials") or [])
        target = self.facies_material_count.value()
        if target < len(materials):
            removed_ids = {
                str(row["material_id"]) for row in materials[target:]
            }
            removed_examples = sum(
                str(row.get("material_id") or "") in removed_ids
                for row in session.get("examples") or []
            )
            if removed_examples and QMessageBox.question(
                self,
                "Remove sand types and examples?",
                f"This removes {removed_examples} saved example box(es). Continue?",
            ) != QMessageBox.StandardButton.Yes:
                self.facies_material_count.setValue(len(materials))
                return
            materials = materials[:target]
            session["examples"] = [
                row
                for row in session.get("examples") or []
                if str(row.get("material_id") or "") not in removed_ids
            ]
        while len(materials) < target:
            number = len(materials) + 1
            materials.append(
                {
                    "material_id": uuid.uuid4().hex,
                    "name": f"Sand type {number}",
                    "canonical_key": f"sand type {number}",
                    "analysis_role": "porous_medium",
                    "porosity": None,
                    "permeability": None,
                }
            )
        session["materials"] = materials
        self.facies_example_session = session
        if self._save_facies_example_session(
            "Changed the number of sand types", session_override=session
        ):
            self._refresh_facies_example_ui()

    def _add_facies_material(self) -> None:
        self.facies_material_count.setValue(self.facies_materials.rowCount() + 1)
        self._set_facies_material_count()

    def _remove_facies_material(self) -> None:
        rows = sorted(
            {index.row() for index in self.facies_materials.selectedIndexes()}
        )
        if not rows:
            return
        session = self._collect_facies_example_session()
        remove_ids = {
            str(session["materials"][row]["material_id"])
            for row in rows
            if row < len(session["materials"])
        }
        count = sum(
            str(row.get("material_id") or "") in remove_ids
            for row in session.get("examples") or []
        )
        if count and QMessageBox.question(
            self,
            "Remove sand types and examples?",
            f"This removes {count} saved example box(es). Continue?",
        ) != QMessageBox.StandardButton.Yes:
            return
        session["materials"] = [
            row
            for row in session["materials"]
            if str(row["material_id"]) not in remove_ids
        ]
        session["examples"] = [
            row
            for row in session.get("examples") or []
            if str(row.get("material_id") or "") not in remove_ids
        ]
        self.facies_example_session = session
        if self._save_facies_example_session(
            "Removed named sand types", session_override=session
        ):
            self._refresh_facies_example_ui()

    def _facies_example_use_current_baseline(self) -> None:
        path = self.baseline_path.text() or str(
            self.config.get("baseline_path") or ""
        )
        if not path:
            QMessageBox.information(
                self,
                "No baseline selected",
                "Select a baseline image in Inputs first.",
            )
            return
        self.facies_example_source.setText(path)
        self._facies_example_session_changed()
        self._load_facies_example_source()

    def _latest_facies_model_manifest(self) -> Path | None:
        if not hasattr(self, "facies_model_root"):
            return None
        store = self._facies_model_store()
        if store is None:
            return None
        records = store.records()
        if not records:
            return None
        path = store.root / str(records[-1].get("manifest_relpath") or "")
        return path if path.is_file() else None

    def _facies_example_select_latest_model(self) -> None:
        try:
            path = self._latest_facies_model_manifest()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Model index unavailable", str(exc))
            return
        if path is None:
            QMessageBox.information(
                self,
                "No compatible project model",
                "No supervised facies model is registered in this project's model folder.",
            )
            return
        self.facies_example_model_manifest.setText(str(path))
        self.facies_example_strategy.setCurrentIndex(
            max(0, self.facies_example_strategy.findData("blind_model_first"))
        )
        self._facies_example_session_changed()

    def _load_facies_example_source(self) -> None:
        path_text = self.facies_example_source.text()
        if not path_text:
            QMessageBox.warning(
                self, "Example image required", "Choose a representative image first."
            )
            return
        try:
            image = load_facies_example_image(Path(path_text))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Could not load example image",
                f"{type(exc).__name__}: {exc}",
            )
            return
        shape = image.shape[:2]
        preview = {
            "baseline": image,
            "labels": np.zeros(shape, dtype=np.int32),
            "facies": np.zeros(shape, dtype=np.int32),
            "depth": np.full(shape, np.nan, dtype=np.float32),
        }
        self._facies_example_preview = preview
        self.canvas.set_model(preview, self._collect_config())
        self.background.setCurrentIndex(
            max(0, self.background.findData("baseline"))
        )
        self.overlay.setCurrentIndex(max(0, self.overlay.findData("baseline")))
        self._refresh_facies_example_overlay()
        self.facies_example_status.setText(
            f"Full-resolution example image loaded: {image.shape[1]} x {image.shape[0]} px."
        )

    def _facies_example_drawing_toggled(self, active: bool) -> None:
        if active:
            if self.facies_example_material.count() == 0:
                QMessageBox.information(
                    self,
                    "Define sand types first",
                    "Create and name at least one sand type before drawing examples.",
                )
                self.draw_facies_example.setChecked(False)
                return
            if self.canvas.pixmap_item.pixmap().isNull():
                self._load_facies_example_source()
            if self.canvas.pixmap_item.pixmap().isNull():
                self.draw_facies_example.setChecked(False)
                return
            if self.place_injector.isChecked():
                self.place_injector.setChecked(False)
            if self.select_layer.isChecked():
                self.select_layer.setChecked(False)
            self.canvas.set_edit_mode("example_box")
        elif self.canvas.edit_mode == "example_box":
            self.canvas.set_edit_mode("pan")

    @Slot(object)
    def _facies_example_box_drawn(self, raw_box: object) -> None:
        if not isinstance(raw_box, Mapping):
            return
        material_id = str(self.facies_example_material.currentData() or "")
        source = self.facies_example_source.text()
        if not material_id or not source:
            return
        box = {key: float(raw_box.get(key) or 0.0) for key in ("x", "y", "width", "height")}
        box["width"] = min(box["width"], 1.0 - box["x"])
        box["height"] = min(box["height"], 1.0 - box["y"])
        session = self._collect_facies_example_session()
        session.setdefault("examples", []).append(
            {
                "example_id": uuid.uuid4().hex,
                "material_id": material_id,
                "source_image_path": source,
                **box,
                "created_by": self.actor.text().strip(),
                "note": "Representative homogeneous material interior",
            }
        )
        self.facies_example_session = session
        if self._save_facies_example_session(
            "Added representative material box", session_override=session
        ):
            self._refresh_facies_example_ui()
            self.facies_example_status.setText(
                "Example box autosaved. Continue drawing or choose another sand type."
            )
        if self.draw_facies_example.isChecked():
            self.canvas.set_edit_mode("example_box")

    def _delete_facies_examples(self) -> None:
        identifiers = {
            str(self.facies_examples.item(index.row(), 0).data(Qt.ItemDataRole.UserRole))
            for index in self.facies_examples.selectedIndexes()
            if self.facies_examples.item(index.row(), 0) is not None
        }
        if not identifiers:
            return
        session = self._collect_facies_example_session()
        session["examples"] = [
            row
            for row in session.get("examples") or []
            if str(row.get("example_id") or "") not in identifiers
        ]
        self.facies_example_session = session
        if self._save_facies_example_session(
            "Deleted representative material boxes", session_override=session
        ):
            self._refresh_facies_example_ui()

    def _refresh_facies_example_overlay(self, *_args: Any) -> None:
        if not hasattr(self, "show_facies_examples"):
            return
        source_text = self.facies_example_source.text()
        try:
            selected = str(Path(source_text).expanduser().resolve())
        except Exception:  # noqa: BLE001
            selected = source_text
        examples = [
            row
            for row in self.facies_example_session.get("examples") or []
            if str(row.get("source_image_path") or "") == selected
        ]
        self.canvas.set_example_boxes(
            examples,
            list(self.facies_example_session.get("materials") or []),
            visible=self.show_facies_examples.isChecked(),
        )

    def _run_facies_example_prediction(self, mode: str) -> None:
        if self._facies_example_thread is not None:
            return
        strategy_index = self.facies_example_strategy.findData(mode)
        self._facies_example_applying = True
        try:
            self.facies_example_strategy.setCurrentIndex(max(0, strategy_index))
        finally:
            self._facies_example_applying = False
        if not self._save_facies_example_session(
            f"Prepared {mode} material proposal", show_error=True
        ):
            return
        session = self._collect_facies_example_session()
        if not Path(str(session.get("source_image_path") or "")).expanduser().is_file():
            QMessageBox.warning(
                self,
                "Example image unavailable",
                "Choose an available source image before generating a proposal.",
            )
            return
        destination = self.project.root / "results" / "facies_example_predictions"
        workers = (
            self.facies_model_cpu_workers.value()
            if hasattr(self, "facies_model_cpu_workers")
            else 0
        )
        thread = QThread(self)
        worker = FaciesExamplePredictionWorker(
            session=session,
            destination_root=destination,
            mode=mode,
            cpu_workers=workers,
        )
        self._facies_example_thread = thread
        self._facies_example_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._facies_example_prediction_finished)
        worker.failed.connect(self._facies_example_prediction_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._facies_example_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.generate_guided_facies.setEnabled(False)
        self.generate_blind_facies.setEnabled(False)
        self.facies_example_busy.setRange(0, 0)
        self.facies_example_status.setText(
            "Building a full-resolution unreviewed material and layer proposal..."
        )
        thread.start()

    @Slot(object)
    def _facies_example_prediction_finished(self, raw_result: object) -> None:
        result = dict(raw_result)
        self._facies_example_result = result
        session = self._collect_facies_example_session()
        session["latest_prediction_manifest"] = str(result["manifest_path"])
        self.facies_example_session = session
        self._save_facies_example_session("Recorded a new facies material proposal")
        try:
            source = load_facies_example_image(
                Path(str((result.get("source_image") or {}).get("path") or ""))
            )
            root = Path(str(result["prediction_path"]))
            with np.load(root / "facies.npz", allow_pickle=False) as archive:
                facies = np.asarray(archive["array"])
            with np.load(root / "labels.npz", allow_pickle=False) as archive:
                labels = np.asarray(archive["array"])
            with np.load(root / "uncertainty.npz", allow_pickle=False) as archive:
                uncertainty = np.asarray(archive["array"])
            preview = {
                "baseline": source,
                "labels": labels,
                "facies": facies,
                "facies_proposal": facies,
                "facies_uncertainty": uncertainty,
                "depth": np.full(facies.shape, np.nan, dtype=np.float32),
            }
            self._facies_example_preview = preview
            self.canvas.set_model(preview, self._collect_config())
            self.overlay.setCurrentIndex(
                max(0, self.overlay.findData("facies_proposal"))
            )
            self._refresh_facies_example_overlay()
        except Exception as exc:  # noqa: BLE001
            self.facies_example_status.setText(
                "Proposal was saved, but its preview could not be loaded: "
                f"{type(exc).__name__}: {exc}"
            )
        else:
            mode = str(result.get("mode") or "").replace("_", " ")
            self.facies_example_status.setText(
                f"{mode.title()} proposal saved. Inspect materials and uncertainty; "
                "it is not approved for analysis."
            )
        self._finish_facies_example_prediction()
        self._refresh_facies_example_ui()

    @Slot(str)
    def _facies_example_prediction_failed(self, message: str) -> None:
        self.facies_example_status.setText("Facies material proposal failed.")
        self._finish_facies_example_prediction()
        QMessageBox.critical(self, "Could not generate facies proposal", message)

    def _finish_facies_example_prediction(self) -> None:
        self.facies_example_busy.setRange(0, 1)
        self.facies_example_busy.setValue(1)

    @Slot()
    def _facies_example_thread_finished(self) -> None:
        self._facies_example_worker = None
        self._facies_example_thread = None
        self._refresh_facies_example_ui()

    def _apply_facies_example_prediction(self) -> None:
        manifest = str(
            (self._facies_example_result or {}).get("manifest_path")
            or self.facies_example_session.get("latest_prediction_manifest")
            or ""
        )
        if not manifest:
            return
        try:
            updated = facies_config_from_example_prediction(
                self._collect_config(), Path(manifest)
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not use facies proposal", str(exc))
            return
        self.validation = None
        self._populate_config(updated)
        preview = getattr(self, "_facies_example_preview", None)
        if preview is not None:
            self.preview_data = dict(preview)
            self.canvas.set_model(self.preview_data, updated)
        self.save_button.setEnabled(False)
        self.geometry_summary.setText(
            "Unreviewed material proposal applied locally. Assign properties, review "
            "every physical layer, then validate before saving a geometry revision."
        )
        self.tabs.setCurrentIndex(1)

    def _facies_annotation_store(self) -> FaciesAnnotationStore | None:
        root = self.facies_dataset_root.text()
        if not root:
            return None
        return FaciesAnnotationStore(Path(root))

    def _refresh_facies_exports(self) -> None:
        store = self._facies_annotation_store()
        self.facies_exports.setRowCount(0)
        if store is None:
            self.facies_dataset_summary.setText("Choose a dataset folder.")
            return
        try:
            records = list(reversed(store.records()))
            summary = store.summary()
        except Exception as exc:  # noqa: BLE001
            self.facies_dataset_summary.setText(f"Dataset index unavailable: {exc}")
            return
        self.facies_exports.setRowCount(len(records))
        for row_index, record in enumerate(records):
            values = (
                str(record.get("created_utc") or "")[:19],
                str(record.get("series_id") or ""),
                str(record.get("group_id") or ""),
                str(record.get("split") or "").title(),
                str(record.get("review_status") or "").title(),
                str(record.get("bundle_id") or "")[:8],
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if column == 5:
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        str(record.get("bundle_relpath") or ""),
                    )
                self.facies_exports.setItem(row_index, column, item)
        splits = dict(summary.get("split_counts") or {})
        conflicts = dict(summary.get("split_conflicts") or {})
        self.facies_dataset_summary.setText(
            f"{summary.get('bundle_count', 0)} bundles · "
            f"{splits.get('train', 0)} train · "
            f"{splits.get('validation', 0)} validation · "
            f"{splits.get('holdout', 0)} holdout · "
            f"{splits.get('pool', 0)} pool · "
            f"{len(conflicts)} leakage conflicts"
        )
        if not records:
            self.facies_export_status.setText("No facies annotations exported.")
        elif self.facies_export_status.text() == "No facies annotations exported.":
            latest = records[0]
            self.facies_export_status.setText(
                f"Latest indexed bundle · "
                f"{str(latest.get('review_status') or '').title()} · "
                f"{str(latest.get('bundle_id') or '')[:8]}"
            )

    def _export_facies_annotations(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        if (
            self._facies_export_thread is not None
            and self._facies_export_thread.isRunning()
        ):
            return
        store = self._facies_annotation_store()
        if store is None:
            QMessageBox.warning(
                self,
                "Dataset folder required",
                "Choose a local or shared dataset folder first.",
            )
            return
        revision = self.store.current()
        if revision.get("format") != GEOMETRY_REVISION_FORMAT:
            QMessageBox.warning(
                self,
                "Saved geometry required",
                "Validate and save the current geometry/facies interpretation first.",
            )
            return
        saved_config = dict(revision.get("configuration") or {})
        if geometry_config_digest(saved_config) != geometry_config_digest(
            self._collect_config()
        ):
            QMessageBox.warning(
                self,
                "Unsaved geometry changes",
                "Validate and save the current geometry changes before exporting annotations.",
            )
            return
        group_id = self.facies_group_id.text().strip()
        if not group_id:
            QMessageBox.warning(
                self,
                "Leakage group required",
                "Assign every related set of rig images to one leakage group.",
            )
            return
        self.export_facies_bundle.setEnabled(False)
        self.import_rig.setEnabled(False)
        self.generate_draft.setEnabled(False)
        self.validate_button.setEnabled(False)
        self.busy.setRange(0, 0)
        self.facies_export_status.setText(
            "Building and verifying an immutable annotation bundle..."
        )
        thread = QThread(self)
        worker = FaciesAnnotationExportWorker(
            project=self.project,
            dataset_root=store.root,
            geometry_revision=revision,
            split=str(self.facies_split.currentData()),
            group_id=group_id,
            actor=self.actor.text().strip() or "Unspecified reviewer",
            include_registered_frame=self.include_facies_review_frame.isChecked(),
        )
        self._facies_export_thread = thread
        self._facies_export_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._facies_export_finished)
        worker.failed.connect(self._facies_export_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._facies_export_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    @Slot(object)
    def _facies_export_finished(self, raw_result: object) -> None:
        result = dict(raw_result)
        record = dict(result.get("record") or {})
        self.facies_export_status.setText(
            (
                "Existing identical bundle reused"
                if record.get("reused")
                else "Annotation bundle exported and verified"
            )
            + f" · {str(record.get('review_status') or '').title()} · "
            + f"{str(record.get('bundle_id') or '')[:8]}"
        )
        self._finish_facies_export()
        self._refresh_facies_exports()

    @Slot(str)
    def _facies_export_failed(self, message: str) -> None:
        self.facies_export_status.setText("Annotation export failed.")
        self._finish_facies_export()
        QMessageBox.critical(self, "Could not export facies annotations", message)

    def _finish_facies_export(self) -> None:
        self.export_facies_bundle.setEnabled(True)
        self.import_rig.setEnabled(True)
        self.generate_draft.setEnabled(True)
        self.validate_button.setEnabled(not self._label_edits_need_materialization)
        self.busy.setRange(0, 1)
        self.busy.setValue(1)

    @Slot()
    def _facies_export_thread_finished(self) -> None:
        self._facies_export_worker = None
        self._facies_export_thread = None

    def _open_facies_dataset(self) -> None:
        store = self._facies_annotation_store()
        if store is None:
            return
        store.root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(store.root)))

    @Slot(int, int)
    def _open_selected_facies_bundle(self, row: int, _column: int) -> None:
        store = self._facies_annotation_store()
        item = self.facies_exports.item(row, 5)
        if store is None or item is None:
            return
        relative = str(item.data(Qt.ItemDataRole.UserRole) or "")
        bundle = store.root / relative
        if bundle.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(bundle)))

    def _facies_model_store(self) -> FaciesModelStore | None:
        root = self.facies_model_root.text()
        if not root:
            return None
        return FaciesModelStore(Path(root))

    def _facies_training_roots(self) -> list[Path]:
        raw = self.facies_model_dataset_roots.text().replace("\n", ";")
        roots = [Path(value.strip()) for value in raw.split(";") if value.strip()]
        if not roots:
            store = self._facies_annotation_store()
            if store is not None:
                roots = [store.root]
        return roots

    def _refresh_facies_models(self) -> None:
        self.facies_models.setRowCount(0)
        store = self._facies_model_store()
        if store is None:
            self.facies_model_status.setText("Choose a model folder.")
            return
        try:
            records = list(reversed(store.records()))
        except Exception as exc:  # noqa: BLE001
            self.facies_model_status.setText(f"Model index unavailable: {exc}")
            return
        self.facies_models.setRowCount(len(records))
        for row_index, record in enumerate(records):
            validation_metric = record.get("validation_hydraulic_mean_iou")
            holdout_metric = record.get("holdout_hydraulic_mean_iou")
            values = (
                str(record.get("created_utc") or "")[:19],
                str(record.get("model_id") or "")[:8],
                (
                    "Not evaluated"
                    if validation_metric is None
                    else f"{float(validation_metric):.3f}"
                ),
                (
                    "Not evaluated"
                    if holdout_metric is None
                    else f"{float(holdout_metric):.3f}"
                ),
                str(record.get("status") or "").replace("_", " ").title(),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(value)
                if column == 1:
                    item.setData(
                        Qt.ItemDataRole.UserRole,
                        str(record.get("model_relpath") or ""),
                    )
                self.facies_models.setItem(row_index, column, item)
        if (
            records
            and self.facies_model_status.text() == "No supervised model trained."
        ):
            latest = records[0]
            self.facies_model_status.setText(
                "Latest research baseline · "
                f"{str(latest.get('model_id') or '')[:8]} · double-click for model card"
            )

    def _train_facies_baseline(self) -> None:
        if (
            self._facies_model_thread is not None
            and self._facies_model_thread.isRunning()
        ):
            return
        roots = self._facies_training_roots()
        if not roots:
            QMessageBox.warning(
                self,
                "Annotation folders required",
                "Choose at least one folder containing approved annotation bundles.",
            )
            return
        missing = [str(root) for root in roots if not root.expanduser().is_dir()]
        if missing:
            QMessageBox.warning(
                self,
                "Annotation folder unavailable",
                "These annotation folders are unavailable:\n" + "\n".join(missing),
            )
            return
        store = self._facies_model_store()
        if store is None:
            QMessageBox.warning(
                self,
                "Model folder required",
                "Choose a local or shared model folder first.",
            )
            return
        config = FaciesModelConfig(
            n_estimators=self.facies_model_estimators.value(),
            max_pixels_per_bundle=self.facies_model_pixels.value(),
            cpu_workers=self.facies_model_cpu_workers.value(),
            calibrate_probabilities=self.facies_model_calibrate.isChecked(),
            include_normalized_position=self.facies_model_position.isChecked(),
        )
        split_plan: dict[str, Any] | None = None
        split_plan_path = self.facies_model_split_plan.text()
        if split_plan_path:
            try:
                split_plan = json.loads(
                    Path(split_plan_path).expanduser().read_text(encoding="utf-8-sig")
                )
                if not isinstance(split_plan, dict):
                    raise TypeError("Split plan must contain one JSON object")
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Split plan unavailable",
                    f"The split plan could not be loaded:\n{type(exc).__name__}: {exc}",
                )
                return
        self.train_facies_model.setEnabled(False)
        self.export_facies_bundle.setEnabled(False)
        self.facies_model_busy.setRange(0, 0)
        self.facies_model_status.setText(
            "Collecting approved bundles and checking leakage groups..."
        )
        thread = QThread(self)
        worker = FaciesModelTrainingWorker(
            dataset_roots=roots,
            output_root=store.root,
            config=config,
            split_plan=split_plan,
        )
        self._facies_model_thread = thread
        self._facies_model_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._facies_model_progress)
        worker.finished.connect(self._facies_model_finished)
        worker.failed.connect(self._facies_model_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._facies_model_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    @Slot(str)
    def _facies_model_progress(self, message: str) -> None:
        self.facies_model_status.setText(message)

    @Slot(object)
    def _facies_model_finished(self, raw_result: object) -> None:
        result = dict(raw_result)
        self.facies_model_status.setText(
            "Supervised research baseline trained and independently evaluated · "
            f"{str(result.get('model_id') or '')[:8]}"
        )
        self._finish_facies_model_training()
        self._refresh_facies_models()

    @Slot(str)
    def _facies_model_failed(self, message: str) -> None:
        self.facies_model_status.setText("Supervised baseline training failed.")
        self._finish_facies_model_training()
        QMessageBox.critical(self, "Could not train facies baseline", message)

    def _finish_facies_model_training(self) -> None:
        self.train_facies_model.setEnabled(True)
        self.export_facies_bundle.setEnabled(True)
        self.facies_model_busy.setRange(0, 1)
        self.facies_model_busy.setValue(1)

    @Slot()
    def _facies_model_thread_finished(self) -> None:
        self._facies_model_worker = None
        self._facies_model_thread = None

    def _open_facies_models(self) -> None:
        store = self._facies_model_store()
        if store is None:
            return
        store.root.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(store.root)))

    @Slot(int, int)
    def _open_selected_facies_model(self, row: int, _column: int) -> None:
        store = self._facies_model_store()
        item = self.facies_models.item(row, 1)
        if store is None or item is None:
            return
        relative = str(item.data(Qt.ItemDataRole.UserRole) or "")
        model_root = store.root / relative
        model_card = model_root / "model_card.md"
        selected = model_card if model_card.is_file() else model_root
        if selected.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(selected)))

    def _import_rig(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Import saved DarSIA rig",
            self.rig_folder.text() or str(self.project.root),
            options=DIRECTORY_DIALOG_OPTIONS,
        )
        if not selected:
            return
        properties = (
            Path(self.properties_path.text()) if self.properties_path.text() else None
        )
        protocol = (
            Path(self.protocol_path.text()) if self.protocol_path.text() else None
        )
        self._start_worker(
            GeometryWorker(
                mode="import",
                rig_folder=Path(selected),
                facies_properties_path=properties,
                protocol_path=protocol,
                rig_type=str(
                    (self.project.manifest.get("rig") or {}).get("type") or "custom"
                ),
            )
        )

    def _generate_optical_draft(self) -> None:
        source_text = self.draft_source_path.text() or self.baseline_path.text()
        if not source_text:
            QMessageBox.warning(
                self,
                "Baseline image required",
                "Choose a baseline image before generating an optical facies draft.",
            )
            return
        width_m = self.width_m.value()
        height_m = self.height_m.value()
        if width_m <= 0.0 or height_m <= 0.0:
            QMessageBox.warning(
                self,
                "Physical dimensions required",
                "Enter the physical width and height of the rig first.",
            )
            return
        destination = (
            self.project.root
            / "overrides"
            / "geometry_facies"
            / "drafts"
            / uuid.uuid4().hex
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        self._start_worker(
            GeometryWorker(
                mode="draft",
                rig_type=str(
                    (self.project.manifest.get("rig") or {}).get("type") or "custom"
                ),
                draft_source_path=Path(source_text),
                draft_destination=destination,
                draft_width_m=width_m,
                draft_height_m=height_m,
                draft_depth_m=self.draft_depth_mm.value() / 1000.0,
                draft_classes=self.draft_classes.value(),
            )
        )

    def _start_validation(self) -> None:
        self._start_worker(
            GeometryWorker(mode="validate", config=self._collect_config())
        )

    def _start_worker(self, worker: GeometryWorker) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        if (
            self._facies_export_thread is not None
            and self._facies_export_thread.isRunning()
        ):
            return
        self.import_rig.setEnabled(False)
        self.generate_draft.setEnabled(False)
        self.export_facies_bundle.setEnabled(False)
        self.validate_button.setEnabled(False)
        self.save_button.setEnabled(False)
        self.busy.setRange(0, 0)
        self.geometry_summary.setText("Reading and validating geometry...")
        self.last_geometry_preview_error = None
        self.last_geometry_worker_error = None
        self._thread = QThread(self)
        self._worker = worker
        worker.moveToThread(self._thread)
        self._thread.started.connect(worker.run)
        worker.finished.connect(self._worker_finished)
        worker.failed.connect(self._worker_failed)
        worker.finished.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.start()

    @Slot(object)
    def _worker_finished(self, raw_result: object) -> None:
        result = dict(raw_result)
        self.last_geometry_preview_error = str(result.get("preview_error") or "")
        self.last_geometry_worker_error = ""
        if bool(result.get("preview_only")):
            self.preview_data = result.get("preview")
            if self.preview_data is not None:
                self.canvas.set_model(self.preview_data, self._collect_config())
                mode = str(result.get("mode") or "")
                background = (
                    "review_frame"
                    if mode == "preview_injector"
                    and "review_frame" in self.preview_data
                    else "baseline"
                )
                self.background.setCurrentIndex(
                    max(0, self.background.findData(background))
                )
                if mode == "preview_injector":
                    self.tabs.setCurrentIndex(2)
                    self.geometry_summary.setText(
                        "Injection-point placement preview ready. Select a point and "
                        "click Place selected injection point, or edit x/y in the table."
                    )
                else:
                    self.geometry_summary.setText(
                        f"Previewing per-run geometry for "
                        f"{str(self.preview_data.get('run_id') or '').upper()}."
                    )
            preview_error = str(result.get("preview_error") or "")
            self._finish_worker()
            if preview_error:
                QMessageBox.warning(self, "Preview unavailable", preview_error)
            return
        self.config = deepcopy_settings(result["config"])
        self.validation = deepcopy_settings(result["validation"])
        self.preview_data = result.get("preview")
        self._populate_config(self.config)
        if self.preview_data is not None:
            self.canvas.set_model(self.preview_data, self.config)
            if "review_frame" in self.preview_data:
                index = self.background.findData("review_frame")
                self.background.setCurrentIndex(max(0, index))
        preview_error = str(result.get("preview_error") or "")
        if preview_error:
            QMessageBox.warning(self, "Preview unavailable", preview_error)
        self._finish_worker()
        self._show_validation(self.validation)

    @Slot(str)
    def _worker_failed(self, message: str) -> None:
        self.last_geometry_worker_error = str(message)
        self._finish_worker()
        self.geometry_summary.setText("Geometry operation failed.")
        QMessageBox.critical(self, "Geometry operation failed", message)

    def _finish_worker(self) -> None:
        self.import_rig.setEnabled(True)
        self.generate_draft.setEnabled(True)
        self.export_facies_bundle.setEnabled(True)
        self.validate_button.setEnabled(not self._label_edits_need_materialization)
        self.busy.setRange(0, 1)
        self.busy.setValue(1)
        self._worker = None
        self._thread = None

    def _show_run_geometry_diagnostics(self, summary: Mapping[str, Any]) -> None:
        diagnostics = {
            str(row.get("run_id") or "").strip().casefold(): dict(row)
            for row in summary.get("run_geometry_change_diagnostics") or []
        }
        # Derived status is not a user edit to the selected geometry.
        with QSignalBlocker(self.run_geometry_table):
            for row_index in range(self.run_geometry_table.rowCount()):
                run_item = self.run_geometry_table.item(row_index, 1)
                status_item = self.run_geometry_table.item(row_index, 7)
                if run_item is None or status_item is None:
                    continue
                run_id = run_item.text().strip().casefold()
                diagnostic = diagnostics.get(run_id)
                if diagnostic is None:
                    labels_item = self.run_geometry_table.item(row_index, 5)
                    status_item.setText(
                        "No run labels"
                        if labels_item is None or not labels_item.text().strip()
                        else "Not checked"
                    )
                    status_item.setToolTip(
                        "A comparable per-run layer-label map is required for screening."
                    )
                    status_item.setForeground(QColor("#9aa4ad"))
                    continue
                status = str(diagnostic.get("status") or "not_comparable")
                if status == "not_comparable":
                    status_item.setText("Not comparable")
                    status_item.setToolTip(str(diagnostic.get("reason") or ""))
                    status_item.setForeground(QColor("#e3b65c"))
                    continue
                p95_mm = 1000.0 * float(diagnostic.get("p95_displacement_m") or 0.0)
                affected = 100.0 * float(
                    diagnostic.get("affected_boundary_fraction") or 0.0
                )
                if bool(diagnostic.get("substantial_change")):
                    status_item.setText(f"Review · {p95_mm:.2f} mm")
                    status_item.setForeground(QColor("#e3b65c"))
                else:
                    status_item.setText(f"Within threshold · {p95_mm:.2f} mm")
                    status_item.setForeground(QColor("#76c893"))
                status_item.setToolTip(
                    f"95th-percentile boundary displacement: {p95_mm:.3f} mm\n"
                    f"Boundary samples beyond threshold: {affected:.1f}%\n"
                    "This screen is advisory and does not change the selected geometry."
                )

    def _show_validation(self, report: Mapping[str, Any] | None) -> None:
        self.issues.setRowCount(0)
        if not report:
            self.geometry_summary.setText("No validated geometry revision.")
            self.validation_details.setText("No validation report.")
            self.save_button.setEnabled(False)
            return
        summary = dict(report.get("summary") or {})
        source_notice = str(summary.get("source_mode_notice") or "")
        if source_notice:
            diagnosis = (self.config.get("imported_rasters") or {}).get("diagnosis") or {}
            self.geometry_quality_basis.setText(
                source_notice + ("; diagnosis SHA-256 " + diagnosis["sha256"] if diagnosis.get("sha256") else ""))
        self._show_run_geometry_diagnostics(summary)
        errors = int(summary.get("error_count") or 0)
        warnings = int(summary.get("warning_count") or 0)
        pore_volume = summary.get("estimated_pore_volume_m3")
        volume_text = (
            f" · {1000.0 * float(pore_volume):.3f} L pore volume"
            if pore_volume is not None
            else ""
        )
        self.geometry_summary.setText(
            (
                "Unreviewed optical draft · "
                if str((self.config.get("facies_generation") or {}).get("status") or "")
                == "unreviewed_optical_draft"
                else ""
            )
            + f"{summary.get('label_count', 0)} layers · {summary.get('facies_count', 0)} facies · "
            f"{summary.get('selected_run_override_count', 0)} selected run override(s) · "
            f"{summary.get('substantial_layer_change_count', 0)} layer-change warning(s) · "
            f"{summary.get('injector_count', 0)} injection points · {errors} errors · {warnings} warnings"
            f"{volume_text}"
        )
        depth = dict(summary.get("depth") or {})
        self.validation_details.setText(
            f"Shape: {summary.get('spatial_shape') or 'unknown'}\n"
            f"Active porous area: {_percent(summary.get('active_area_fraction'))}\n"
            f"Run layer maps screened: {summary.get('screened_run_geometry_count', 0)}\n"
            f"Depth min / median / max: {_millimetres(depth.get('minimum_m'))} / "
            f"{_millimetres(depth.get('median_m'))} / {_millimetres(depth.get('maximum_m'))}"
        )
        rows = list(report.get("issues") or [])
        self.issues.setRowCount(len(rows))
        colors = {"error": "#e27979", "warning": "#e3b65c", "info": "#73b9e6"}
        for row_index, issue in enumerate(rows):
            values = (
                str(issue.get("severity") or "").upper(),
                str(issue.get("code") or "").replace("_", " "),
                str(issue.get("message") or ""),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setForeground(
                        QColor(colors.get(str(issue.get("severity")), "#d7dce0"))
                    )
                self.issues.setItem(row_index, column, item)
        current_digest = geometry_config_digest(self._collect_config())
        self.save_button.setEnabled(
            errors == 0 and str(report.get("config_sha256") or "") == current_digest
            and self.config.get("source_mode") != "imported-rasters"
        )
        if errors:
            self.tabs.setCurrentWidget(self.qc_tab)

    def _save_revision(self) -> None:
        config = self._collect_config()
        actor = self.actor.text().strip()
        reason = self.reason.text().strip() or "Validated geometry and facies revision"
        try:
            if config.get("source_mode") == "imported-rasters":
                from darsia.fluidflower.geometry_raster_sources import SILENT_EDIT
                raise ValueError(SILENT_EDIT)
            revision = self.store.save_revision(
                config,
                actor=actor,
                reason=reason,
                validation=self.validation,
                **({"preserve_failed_artifacts": True}
                   if config.get("frame", "raw") == "rig-corrected" else {}),
            )
            self.project.set_module_status(
                "geometry_facies",
                "review_required",
                updated_by=actor,
                reason=f"Geometry revision {revision['revision_id']} saved",
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save geometry revision", str(exc))
            return
        self.config = deepcopy_settings(revision["configuration"])
        self.validation = deepcopy_settings(revision["validation"])
        self.reason.clear()
        self._populate_history()
        self._show_validation(self.validation)
        self.refresh()
        self.status_changed.emit()

    def _populate_history(self) -> None:
        revisions = list(reversed(self.store.revisions()))
        self.revisions.clear()
        if not revisions:
            self.revisions.addItem("No saved revisions", "")
            return
        for revision in revisions:
            summary = dict((revision.get("validation") or {}).get("summary") or {})
            self.revisions.addItem(
                f"{str(revision.get('recorded_utc') or '')[:19]} · "
                f"{summary.get('label_count', 0)} layers · {summary.get('error_count', 0)} errors",
                str(revision.get("revision_id") or ""),
            )

    def _load_selected_revision(self) -> None:
        revision_id = str(self.revisions.currentData() or "")
        revision = next(
            (
                row
                for row in self.store.revisions()
                if str(row.get("revision_id") or "") == revision_id
            ),
            None,
        )
        if revision is None:
            return
        self.label_session = None
        self.preview_data = None
        self.validation = deepcopy_settings(revision.get("validation") or {})
        self._populate_config(revision.get("configuration") or {})
        self._show_validation(self.validation)

    def _display_changed(self, *_args: Any) -> None:
        self.canvas.set_options(
            background=str(self.background.currentData()),
            overlay=str(self.overlay.currentData()),
            opacity=self.opacity.value() / 100.0,
            boundaries_only=self.boundaries.isChecked(),
            show_injectors=self.show_injectors.isChecked(),
        )

    def _placement_toggled(self, active: bool) -> None:
        if active and hasattr(self, "draw_facies_example"):
            self.draw_facies_example.setChecked(False)
        if active and hasattr(self, "pan_boundaries"):
            self.pan_boundaries.setChecked(True)
        if active and self.select_layer.isChecked():
            self.select_layer.setChecked(False)
        self.canvas.place_mode = active
        self.canvas.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if active
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        if active:
            self.tabs.setCurrentIndex(2)

    def _selection_toggled(self, active: bool) -> None:
        if active and hasattr(self, "draw_facies_example"):
            self.draw_facies_example.setChecked(False)
        if active and hasattr(self, "pan_boundaries"):
            self.pan_boundaries.setChecked(True)
        if active and self.place_injector.isChecked():
            self.place_injector.setChecked(False)
        self.canvas.select_mode = active
        self.canvas.setDragMode(
            QGraphicsView.DragMode.NoDrag
            if active
            else QGraphicsView.DragMode.ScrollHandDrag
        )
        if active:
            self.tabs.setCurrentIndex(1)

    def _layer_selected(self, label_id: int) -> None:
        for row in range(self.layers.rowCount()):
            item = self.layers.item(row, 1)
            if item is not None and _gui_int(item.text()) == label_id:
                self.layers.selectRow(row)
                self.layers.scrollToItem(item)
                self.tabs.setCurrentIndex(1)
                return

    def _apply_properties_to_facies(self) -> None:
        selected = self.layers.currentRow()
        if selected < 0:
            return
        facies_item = self.layers.item(selected, 2)
        facies_id = _gui_int(facies_item.text()) if facies_item is not None else None
        if facies_id is None:
            return
        source_role = self.layers.cellWidget(selected, 4)
        source_role_value = (
            str(source_role.currentData())
            if isinstance(source_role, QComboBox)
            else "excluded"
        )
        source_enabled = self.layers.item(selected, 0).checkState()
        source_porosity = self.layers.item(selected, 5).text()
        source_permeability = self.layers.item(selected, 6).text()
        self._applying = True
        try:
            for row in range(self.layers.rowCount()):
                item = self.layers.item(row, 2)
                if item is None or _gui_int(item.text()) != facies_id:
                    continue
                self.layers.item(row, 0).setCheckState(source_enabled)
                role = self.layers.cellWidget(row, 4)
                if isinstance(role, QComboBox):
                    role.setCurrentIndex(max(0, role.findData(source_role_value)))
                self.layers.item(row, 5).setText(source_porosity)
                self.layers.item(row, 6).setText(source_permeability)
        finally:
            self._applying = False
        self._mark_dirty()

    def _add_injector(self) -> None:
        config = self._collect_config()
        width_m = _gui_float(config.get("width_m")) or 0.0
        height_m = _gui_float(config.get("height_m")) or 0.0
        existing_ids = {
            str(row.get("injector_id") or "").strip().casefold()
            for row in config.get("injectors") or []
        }
        number = len(config["injectors"]) + 1
        location_id = f"ip{number}"
        while location_id.casefold() in existing_ids:
            number += 1
            location_id = f"ip{number}"
        config["injectors"].append(
            {
                "injector_id": location_id,
                "name": f"Injection Point {number}",
                "x_m": width_m / 2.0,
                "y_m": height_m / 2.0,
                "uncertainty_m": None,
                "coordinate_source": "manual",
            }
        )
        self._applying = True
        self._populate_injectors(config["injectors"])
        self._applying = False
        self._mark_dirty()
        self.injector_table.selectRow(self.injector_table.rowCount() - 1)

    def _remove_injector(self) -> None:
        row = self.injector_table.currentRow()
        if row >= 0:
            self.injector_table.removeRow(row)
            self._mark_dirty()

    def _injector_placed(self, x_m: float, y_m: float) -> None:
        row = self.injector_table.currentRow()
        if row < 0:
            self._add_injector()
            row = self.injector_table.currentRow()
        self.injector_table.item(row, 2).setText(f"{x_m:.6f}")
        self.injector_table.item(row, 3).setText(f"{y_m:.6f}")
        self.injector_table.item(row, 5).setText("manual_image_pick")
        self._mark_dirty()

    def refresh(self) -> None:
        self._refresh_gate()
        current = self.store.current()
        latest_validation = (
            dict(current.get("validation") or {})
            if current.get("format") == GEOMETRY_REVISION_FORMAT
            else {}
        )
        errors = int((latest_validation.get("summary") or {}).get("error_count") or 0)
        issue_codes = {
            str(issue.get("code") or "")
            for issue in latest_validation.get("issues") or []
        }
        approval_blocked = "no_active_porous_regions" in issue_codes
        has_revision = current.get("format") == GEOMETRY_REVISION_FORMAT
        state = str(self.project.module_states[self.module_key].get("status") or "")
        available = dependencies_satisfied(self.module_key, self.project.module_states)
        self.approve.setEnabled(
            has_revision
            and errors == 0
            and not approval_blocked
            and available
            and state in {"review_required", "in_progress", "ready", "stale"}
        )
        self.complete.setEnabled(
            has_revision
            and errors == 0
            and not approval_blocked
            and available
            and state in {"approved", "review_required", "in_progress"}
        )

    def _set_status(self, status: str) -> None:
        if status in {"approved", "complete"}:
            current = self.store.current()
            validation = dict(current.get("validation") or {})
            if current.get("format") != GEOMETRY_REVISION_FORMAT:
                QMessageBox.warning(
                    self,
                    "Geometry revision required",
                    "Save and review a validated geometry revision before approval.",
                )
                return
            if int((validation.get("summary") or {}).get("error_count") or 0) > 0:
                QMessageBox.warning(
                    self,
                    "Blocking geometry findings",
                    "Resolve every validation error and save a new revision before approval.",
                )
                return
            if "no_active_porous_regions" in {
                str(issue.get("code") or "") for issue in validation.get("issues") or []
            }:
                QMessageBox.warning(
                    self,
                    "Facies review required",
                    "Assign and enable at least one porous-medium or seal region "
                    "before approving the geometry.",
                )
                return
        super()._set_status(status)
