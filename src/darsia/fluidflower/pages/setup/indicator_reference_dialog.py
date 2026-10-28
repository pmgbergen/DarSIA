"""Editor for pH-buffer and alternate-state indicator reference images."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from darsia.fluidflower.indicator_references import (
    REFERENCE_KIND_ALTERNATE_STATE,
    REFERENCE_KIND_PH_BUFFER,
    discover_indicator_reference_images,
    infer_ph,
    merge_indicator_reference_entries,
    normalize_indicator_reference_entry,
)
from darsia.fluidflower.ui.common import (
    DIRECTORY_DIALOG_OPTIONS,
    FILE_DIALOG_OPTIONS,
)


IMAGE_FILTER = "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp);;All files (*)"


class IndicatorReferenceDialog(QDialog):
    """Edit exact image-to-pH/state metadata without changing source files."""

    def __init__(
        self,
        parent: Any = None,
        *,
        entries: Sequence[Mapping[str, Any]] = (),
        source_folders: Sequence[tuple[Path, str]] = (),
        ph_buffer_folders: Sequence[tuple[Path, str]] = (),
        alternate_state_folders: Sequence[tuple[Path, str]] = (),
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Indicator response references")
        self.resize(1180, 680)
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "Register either (1) the same indicator mixture used in the experiment at "
            "a measured pH, or (2) a different, explicitly named liquid/color state in "
            "the rig. Only the first category describes the indicator's chemical "
            "response curve. Neither category is used as an illumination or "
            "ColorChecker target."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("secondaryText")
        layout.addWidget(explanation)

        discovered = discover_indicator_reference_images(source_folders)
        discovered = merge_indicator_reference_entries(
            discovered,
            discover_indicator_reference_images(
                ph_buffer_folders,
                default_kind=REFERENCE_KIND_PH_BUFFER,
            ),
        )
        discovered = merge_indicator_reference_entries(
            discovered,
            discover_indicator_reference_images(
                alternate_state_folders,
                default_kind=REFERENCE_KIND_ALTERNATE_STATE,
            ),
        )
        merged = merge_indicator_reference_entries(entries, discovered)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Reference type", "Measured pH", "State / note", "Scope", "Image"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        header = self.table.horizontalHeader()
        for column in (0, 1, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table, 1)
        for row in merged:
            self._append_entry(row)

        actions = QHBoxLayout()
        add_images = QPushButton("Add image files...")
        add_images.clicked.connect(self._add_images)
        actions.addWidget(add_images)
        add_folder = QPushButton("Add a folder...")
        add_folder.clicked.connect(self._add_folder)
        actions.addWidget(add_folder)
        remove = QPushButton("Remove selected")
        remove.clicked.connect(self._remove_selected)
        actions.addWidget(remove)
        actions.addStretch(1)
        layout.addLayout(actions)

        note = QLabel(
            "Use scope 'series' for references shared by all experiments, or a run ID "
            "such as ac52 when a reference only applies to one experiment. A usable pH "
            "curve needs at least two distinct measured pH values."
        )
        note.setWordWrap(True)
        note.setObjectName("secondaryText")
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _kind_combo(value: str) -> QComboBox:
        combo = QComboBox()
        combo.addItem(
            "Same indicator at measured pH", REFERENCE_KIND_PH_BUFFER
        )
        combo.addItem(
            "Different liquid / known rig-color state",
            REFERENCE_KIND_ALTERNATE_STATE,
        )
        index = combo.findData(str(value or REFERENCE_KIND_PH_BUFFER))
        combo.setCurrentIndex(max(0, index))
        combo.setToolTip(
            "Known-pH buffers constrain indicator chemistry. Other states are labelled "
            "optical references or controls without an assumed pH."
        )
        return combo

    def _append_entry(self, raw: Mapping[str, Any]) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setCellWidget(
            row, 0, self._kind_combo(str(raw.get("kind") or ""))
        )
        ph = raw.get("ph")
        ph_item = QTableWidgetItem(
            "" if ph is None or not str(ph).strip() else str(ph)
        )
        ph_item.setToolTip("Measured pH, from 0 to 14. Comma or point is accepted.")
        self.table.setItem(row, 1, ph_item)
        self.table.setItem(row, 2, QTableWidgetItem(str(raw.get("label") or "")))
        self.table.setItem(
            row, 3, QTableWidgetItem(str(raw.get("scope") or "series"))
        )
        path = str(raw.get("image_path") or "")
        path_item = QTableWidgetItem(path)
        path_item.setToolTip(path)
        path_item.setFlags(path_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self.table.setItem(row, 4, path_item)

    def _existing_paths(self) -> set[str]:
        return {
            str(Path(self.table.item(row, 4).text()).expanduser().resolve()).casefold()
            for row in range(self.table.rowCount())
            if self.table.item(row, 4) is not None
            and self.table.item(row, 4).text().strip()
        }

    def _append_paths(self, paths: Sequence[Path], *, scope: str = "series") -> None:
        known = self._existing_paths()
        for path in paths:
            resolved = Path(path).expanduser().resolve()
            identity = str(resolved).casefold()
            if identity in known:
                continue
            ph = infer_ph(resolved)
            self._append_entry(
                {
                    "kind": (
                        REFERENCE_KIND_PH_BUFFER
                        if ph is not None
                        else REFERENCE_KIND_ALTERNATE_STATE
                    ),
                    "ph": ph,
                    "label": "" if ph is not None else resolved.parent.name,
                    "scope": scope,
                    "image_path": str(resolved),
                }
            )
            known.add(identity)

    def _add_images(self) -> None:
        selected, _ = QFileDialog.getOpenFileNames(
            self,
            "Choose indicator reference images",
            str(Path.home()),
            IMAGE_FILTER,
            options=FILE_DIALOG_OPTIONS,
        )
        self._append_paths([Path(value) for value in selected])

    def _add_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Choose an indicator reference folder",
            str(Path.home()),
            options=DIRECTORY_DIALOG_OPTIONS,
        )
        if not selected:
            return
        discovered = discover_indicator_reference_images([(Path(selected), "series")])
        self._append_paths(
            [Path(str(row["image_path"])) for row in discovered], scope="series"
        )

    def _remove_selected(self) -> None:
        for row in sorted({index.row() for index in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(row)

    def entries(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self.table.rowCount()):
            kind_widget = self.table.cellWidget(row, 0)
            kind = (
                str(kind_widget.currentData())
                if isinstance(kind_widget, QComboBox)
                else REFERENCE_KIND_PH_BUFFER
            )
            rows.append(
                {
                    "kind": kind,
                    "ph": self.table.item(row, 1).text().strip().replace(",", "."),
                    "label": self.table.item(row, 2).text().strip(),
                    "scope": self.table.item(row, 3).text().strip() or "series",
                    "image_path": self.table.item(row, 4).text().strip(),
                }
            )
        return rows

    def _validate_and_accept(self) -> None:
        try:
            normalized = [
                normalize_indicator_reference_entry(row) for row in self.entries()
            ]
        except (TypeError, ValueError) as exc:
            QMessageBox.warning(self, "Reference metadata is incomplete", str(exc))
            return
        self._normalized_entries = normalized
        self.accept()

    def normalized_entries(self) -> list[dict[str, Any]]:
        return list(getattr(self, "_normalized_entries", self.entries()))


__all__ = ["IndicatorReferenceDialog"]
