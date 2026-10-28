"""Reusable Qt editor for spatial analysis domains."""

from __future__ import annotations

import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.analysis_domains import (
    ANALYSIS_DOMAIN_REVISION_FORMAT,
    AnalysisDomainStore,
    canonicalize_analysis_domain,
    default_analysis_domain_config,
    domain_from_normalized_rectangle,
    import_analysis_domains,
    load_analysis_domain_reference,
    validate_analysis_domain_config,
)
from darsia.fluidflower.geometry_facies import GeometryRevisionStore
from darsia.fluidflower.image_correction import CorrectionRevisionStore
from darsia.fluidflower.project import FluidFlowerProject
from darsia.fluidflower.ui.common import (
    FILE_DIALOG_OPTIONS,
    PathField,
    deepcopy_settings,
)


class AnalysisDomainImageView(QGraphicsView):
    """Pan/zoom image view with physical rectangle drawing."""

    rectangle_drawn = Signal(object)
    domain_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene().addItem(self.pixmap_item)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(QColor("#090b0d"))
        self.setMinimumSize(720, 460)
        self.mode = "pan"
        self._start: QPointF | None = None
        self._draft: QGraphicsRectItem | None = None
        self._domain_items: dict[str, QGraphicsRectItem] = {}

    def set_image(self, image: np.ndarray) -> None:
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
        self.fit_image()

    def set_domains(self, domains: list[dict[str, Any]]) -> None:
        for item in self._domain_items.values():
            self.scene().removeItem(item)
        self._domain_items.clear()
        bounds = self.pixmap_item.boundingRect()
        for index, domain in enumerate(domains):
            values = domain.get("normalized_xyxy_top_left") or []
            if len(values) != 4:
                continue
            x0, y0, x1, y1 = [float(value) for value in values]
            item = QGraphicsRectItem(
                x0 * bounds.width(),
                y0 * bounds.height(),
                (x1 - x0) * bounds.width(),
                (y1 - y0) * bounds.height(),
            )
            color = QColor.fromHsv((index * 67 + 15) % 360, 190, 245)
            pen = QPen(color, 2.0)
            pen.setCosmetic(True)
            item.setPen(pen)
            fill = QColor(color)
            fill.setAlpha(34 if bool(domain.get("enabled", True)) else 10)
            item.setBrush(QBrush(fill))
            item.setZValue(10.0)
            item.setData(0, str(domain.get("domain_id") or ""))
            self.scene().addItem(item)
            self._domain_items[str(domain.get("domain_id") or "")] = item

    def select_domain(self, domain_id: str) -> None:
        for key, item in self._domain_items.items():
            pen = item.pen()
            pen.setWidthF(4.0 if key == domain_id else 2.0)
            item.setPen(pen)

    def set_mode(self, mode: str) -> None:
        if mode not in {"pan", "draw"}:
            raise ValueError(f"Unsupported analysis-domain mode: {mode}")
        self.mode = mode
        self.setDragMode(
            QGraphicsView.DragMode.ScrollHandDrag
            if mode == "pan"
            else QGraphicsView.DragMode.NoDrag
        )

    def fit_image(self) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event: Any) -> None:  # noqa: N802
        if self.pixmap_item.pixmap().isNull():
            return
        self.scale(
            1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2,
            1.2 if event.angleDelta().y() > 0 else 1.0 / 1.2,
        )

    def _image_point(self, event: Any) -> QPointF | None:
        point = self.mapToScene(event.position().toPoint())
        return point if self.pixmap_item.boundingRect().contains(point) else None

    def mousePressEvent(self, event: Any) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.mode == "draw":
            point = self._image_point(event)
            if point is not None:
                self._start = point
                self._draft = QGraphicsRectItem(point.x(), point.y(), 0.0, 0.0)
                pen = QPen(QColor("#ff5161"), 2.0)
                pen.setCosmetic(True)
                self._draft.setPen(pen)
                self._draft.setBrush(QBrush(QColor(255, 81, 97, 35)))
                self._draft.setZValue(30.0)
                self.scene().addItem(self._draft)
                event.accept()
                return
        if event.button() == Qt.MouseButton.LeftButton and self.mode == "pan":
            item = self.itemAt(event.position().toPoint())
            if isinstance(item, QGraphicsRectItem) and item.data(0):
                self.domain_selected.emit(str(item.data(0)))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:  # noqa: N802
        if self.mode == "draw" and self._start is not None and self._draft is not None:
            point = self._image_point(event)
            if point is not None:
                x0, x1 = sorted((self._start.x(), point.x()))
                y0, y1 = sorted((self._start.y(), point.y()))
                self._draft.setRect(x0, y0, x1 - x0, y1 - y0)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:  # noqa: N802
        if (
            self.mode == "draw"
            and event.button() == Qt.MouseButton.LeftButton
            and self._start is not None
            and self._draft is not None
        ):
            rectangle = self._draft.rect()
            bounds = self.pixmap_item.boundingRect()
            self.scene().removeItem(self._draft)
            self._draft = None
            self._start = None
            if rectangle.width() >= 3.0 and rectangle.height() >= 3.0:
                self.rectangle_drawn.emit(
                    [
                        rectangle.left() / bounds.width(),
                        rectangle.top() / bounds.height(),
                        rectangle.right() / bounds.width(),
                        rectangle.bottom() / bounds.height(),
                    ]
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AnalysisDomainDialog(QDialog):
    """Edit named rectangles on the corrected, registered rig image."""

    domains_saved = Signal(object)

    def __init__(
        self, project: FluidFlowerProject, parent: QWidget | None = None,
        *, store: AnalysisDomainStore | None = None,
        configuration: dict[str, Any] | None = None,
        load_reference: bool = True,
        canvas: AnalysisDomainImageView | None = None,
    ) -> None:
        super().__init__(parent)
        self.project = project
        self.store = store if store is not None else AnalysisDomainStore(project)
        current = self.store.current()
        self.config = deepcopy_settings(
            configuration if configuration is not None else current.get("configuration")
            if configuration is not None or current.get("format") == ANALYSIS_DOMAIN_REVISION_FORMAT
            else default_analysis_domain_config(project)
        )
        self.reference: dict[str, Any] | None = None
        self.domains = [
            deepcopy_settings(row) for row in self.config.get("domains") or []
        ]
        self._updating_table = False
        self.setWindowTitle("Analysis domains · corrected rig grid")
        self.resize(1380, 900)
        outer = QVBoxLayout(self)

        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Reference frame"))
        self.domain_source = PathField(
            directory=False,
            file_filter="Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.npz);;All files (*)",
            placeholder="Current correction target or corrected rig baseline",
        )
        self.domain_source.setText(str(self.config.get("reference_image_path") or ""))
        source_row.addWidget(self.domain_source, 1)
        load_button = QPushButton("Load corrected image")
        load_button.clicked.connect(self._load_reference)
        source_row.addWidget(load_button)
        outer.addLayout(source_row)
        self.domain_reference_status = QLabel(
            "The displayed image must use the same corrected, cropped, and unwrapped grid as the mass fields."
        )
        self.domain_reference_status.setObjectName("secondaryText")
        self.domain_reference_status.setWordWrap(True)
        outer.addWidget(self.domain_reference_status)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        tools = QHBoxLayout()
        self.domain_pan = QPushButton("Pan")
        self.domain_pan.setCheckable(True)
        self.domain_pan.setChecked(True)
        self.domain_pan.clicked.connect(lambda: self._set_mode("pan"))
        tools.addWidget(self.domain_pan)
        self.domain_draw = QPushButton("Draw rectangle")
        self.domain_draw.setCheckable(True)
        self.domain_draw.clicked.connect(lambda: self._set_mode("draw"))
        tools.addWidget(self.domain_draw)
        fit_button = QPushButton("Fit image")
        fit_button.clicked.connect(lambda: self.domain_canvas.fit_image())
        tools.addWidget(fit_button)
        tools.addStretch(1)
        image_layout.addLayout(tools)
        self.domain_canvas = canvas if canvas is not None else AnalysisDomainImageView()
        self.domain_canvas.rectangle_drawn.connect(self._rectangle_drawn)
        self.domain_canvas.domain_selected.connect(self._canvas_domain_selected)
        image_layout.addWidget(self.domain_canvas, 1)
        splitter.addWidget(image_panel)

        table_panel = QWidget()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        self.domain_table = QTableWidget(0, 6)
        self.domain_table.setHorizontalHeaderLabels(
            ["Use", "Name", "x min [m]", "x max [m]", "y min [m]", "y max [m]"]
        )
        self.domain_table.verticalHeader().setVisible(False)
        self.domain_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.domain_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        for column in range(2, 6):
            self.domain_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.domain_table.itemChanged.connect(self._domain_table_changed)
        self.domain_table.itemSelectionChanged.connect(self._table_selection_changed)
        table_layout.addWidget(self.domain_table, 1)
        table_actions = QHBoxLayout()
        import_button = QPushButton("Import coordinates")
        import_button.clicked.connect(self._import_domains)
        table_actions.addWidget(import_button)
        delete_button = QPushButton("Delete selected")
        delete_button.clicked.connect(self._delete_selected)
        table_actions.addWidget(delete_button)
        table_actions.addStretch(1)
        table_layout.addLayout(table_actions)
        splitter.addWidget(table_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, 1)

        review = QFormLayout()
        self.domain_reviewer = QLineEdit(str(project.manifest.get("created_by") or ""))
        self.domain_reason = QLineEdit()
        self.domain_reason.setPlaceholderText(
            "Why were these physical regions selected?"
        )
        review.addRow("Reviewer", self.domain_reviewer)
        review.addRow("Revision note", self.domain_reason)
        outer.addLayout(review)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._save_domains)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self._refresh_domain_table()
        if load_reference:
            self._load_reference(show_errors=False)

    def _set_mode(self, mode: str) -> None:
        self.domain_pan.setChecked(mode == "pan")
        self.domain_draw.setChecked(mode == "draw")
        self.domain_canvas.set_mode(mode)

    def _load_reference(self, *, show_errors: bool = True) -> None:
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            reference = load_analysis_domain_reference(
                self.project,
                source_path=self.domain_source.text(),
            )
        except Exception as exc:  # noqa: BLE001
            self.domain_reference_status.setText(
                f"Corrected reference could not be loaded: {type(exc).__name__}: {exc}"
            )
            if show_errors:
                QMessageBox.critical(self, "Could not load corrected image", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        self.reference = reference
        provenance = dict(reference["provenance"])
        self.domain_source.setText(str(provenance.get("source_path") or ""))
        self.domain_canvas.set_image(reference["image"])
        self.domain_canvas.set_domains(self.domains)
        registration = dict(provenance.get("registration") or {})
        correction = dict(provenance.get("uniform24") or {})
        corrections = (
            ", ".join(registration.get("corrections") or []) or "registered artifact"
        )
        self.domain_reference_status.setText(
            f"Grid {provenance.get('source_shape_rc')} · preview {provenance.get('preview_shape_rc')} · "
            f"shape/crop/unwrap: {corrections} · ColorChecker: {correction.get('status')}"
        )

    def _rectangle_drawn(self, normalized: object) -> None:
        if self.reference is None:
            return
        provenance = dict(self.reference["provenance"])
        domain = domain_from_normalized_rectangle(
            name=f"Analysis box {len(self.domains) + 1}",
            normalized_xyxy_top_left=list(normalized),
            source_shape_rc=list(provenance["source_shape_rc"]),
            width_m=float(provenance["width_m"]),
            height_m=float(provenance["height_m"]),
        )
        self.domains.append(domain)
        self._refresh_domain_table(select_id=domain["domain_id"])
        self._set_mode("pan")

    def _refresh_domain_table(self, *, select_id: str = "") -> None:
        self._updating_table = True
        self.domain_table.setRowCount(len(self.domains))
        selected_row = -1
        for row, domain in enumerate(self.domains):
            enabled = QTableWidgetItem()
            enabled.setCheckState(
                Qt.CheckState.Checked
                if bool(domain.get("enabled", True))
                else Qt.CheckState.Unchecked
            )
            enabled.setData(
                Qt.ItemDataRole.UserRole, str(domain.get("domain_id") or "")
            )
            enabled.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsUserCheckable
            )
            self.domain_table.setItem(row, 0, enabled)
            self.domain_table.setItem(
                row, 1, QTableWidgetItem(str(domain.get("name") or ""))
            )
            bounds = dict(domain.get("physical_bounds_m") or {})
            for column, key in enumerate(("x_min", "x_max", "y_min", "y_max"), start=2):
                item = QTableWidgetItem(f"{float(bounds.get(key) or 0.0):.6g}")
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsEditable
                )
                self.domain_table.setItem(row, column, item)
            if str(domain.get("domain_id") or "") == select_id:
                selected_row = row
        self._updating_table = False
        self.domain_canvas.set_domains(self.domains)
        if selected_row >= 0:
            self.domain_table.selectRow(selected_row)

    def _domain_table_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table or item.row() >= len(self.domains):
            return
        domain = self.domains[item.row()]
        if item.column() == 0:
            domain["enabled"] = item.checkState() == Qt.CheckState.Checked
        elif item.column() == 1:
            domain["name"] = item.text().strip()
        elif 2 <= item.column() <= 5 and self.reference is not None:
            values: list[float] = []
            for column in range(2, 6):
                cell = self.domain_table.item(item.row(), column)
                try:
                    values.append(
                        float((cell.text() if cell else "").replace(",", "."))
                    )
                except ValueError:
                    self._refresh_domain_table(
                        select_id=str(domain.get("domain_id") or "")
                    )
                    return
            provenance = dict(self.reference["provenance"])
            try:
                updated = canonicalize_analysis_domain(
                    {
                        "domain_id": domain.get("domain_id"),
                        "name": domain.get("name"),
                        "enabled": domain.get("enabled", True),
                        "x_min_m": values[0],
                        "x_max_m": values[1],
                        "y_min_m": values[2],
                        "y_max_m": values[3],
                    },
                    source_shape_rc=list(provenance["source_shape_rc"]),
                    width_m=float(provenance["width_m"]),
                    height_m=float(provenance["height_m"]),
                )
            except (TypeError, ValueError):
                self._refresh_domain_table(select_id=str(domain.get("domain_id") or ""))
                return
            self.domains[item.row()] = updated
        self.domain_canvas.set_domains(self.domains)

    def _table_selection_changed(self) -> None:
        rows = self.domain_table.selectionModel().selectedRows()
        if not rows:
            return
        domain_id = str(self.domains[rows[0].row()].get("domain_id") or "")
        self.domain_canvas.select_domain(domain_id)

    def _canvas_domain_selected(self, domain_id: str) -> None:
        for row, domain in enumerate(self.domains):
            if str(domain.get("domain_id") or "") == domain_id:
                self.domain_table.selectRow(row)
                break

    def _delete_selected(self) -> None:
        rows = sorted(
            {
                index.row()
                for index in self.domain_table.selectionModel().selectedRows()
            },
            reverse=True,
        )
        for row in rows:
            if 0 <= row < len(self.domains):
                self.domains.pop(row)
        self._refresh_domain_table()

    def _import_domains(self) -> None:
        if self.reference is None:
            QMessageBox.warning(
                self, "Load image first", "Load the corrected grid before import."
            )
            return
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Import analysis-domain coordinates",
            "",
            "Domain files (*.json *.csv *.tsv);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if not selected:
            return
        provenance = dict(self.reference["provenance"])
        try:
            imported = import_analysis_domains(
                Path(selected),
                source_shape_rc=list(provenance["source_shape_rc"]),
                width_m=float(provenance["width_m"]),
                height_m=float(provenance["height_m"]),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not import domains", str(exc))
            return
        existing_ids = {str(row.get("domain_id") or "") for row in self.domains}
        existing_names = {str(row.get("name") or "").casefold() for row in self.domains}
        for domain in imported:
            if str(domain["domain_id"]) in existing_ids:
                domain["domain_id"] = uuid.uuid4().hex
            base_name = str(domain["name"])
            suffix = 2
            while str(domain["name"]).casefold() in existing_names:
                domain["name"] = f"{base_name} {suffix}"
                suffix += 1
            existing_ids.add(str(domain["domain_id"]))
            existing_names.add(str(domain["name"]).casefold())
            self.domains.append(domain)
        self._refresh_domain_table()

    def _save_domains(self) -> None:
        if self.reference is None:
            QMessageBox.warning(
                self, "Corrected image required", "Load the corrected image first."
            )
            return
        provenance = deepcopy_settings(self.reference["provenance"])
        geometry = GeometryRevisionStore(self.project).current()
        correction = CorrectionRevisionStore(self.project).current()
        config = default_analysis_domain_config(self.project)
        config.update(
            {
                "geometry_revision_id": str(geometry.get("revision_id") or ""),
                "geometry_revision_sha256": str(geometry.get("content_sha256") or ""),
                "image_correction_revision_id": str(
                    correction.get("revision_id") or ""
                ),
                "reference_image_path": str(provenance.get("source_path") or ""),
                "reference_provenance": provenance,
                "width_m": float(provenance.get("width_m") or 0.0),
                "height_m": float(provenance.get("height_m") or 0.0),
                "source_shape_rc": list(provenance.get("source_shape_rc") or []),
                "domains": deepcopy(self.domains),
            }
        )
        report = validate_analysis_domain_config(self.project, config)
        if not report["valid"]:
            QMessageBox.critical(
                self,
                "Analysis domains are invalid",
                "\n".join(str(issue["message"]) for issue in report["issues"]),
            )
            return
        try:
            revision = self.store.save_revision(
                config,
                actor=self.domain_reviewer.text().strip(),
                reason=self.domain_reason.text().strip(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Could not save analysis domains", str(exc))
            return
        self.domains_saved.emit(revision)
        self.accept()
