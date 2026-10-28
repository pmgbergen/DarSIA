"""Small Qt editor for the CLI image-selection plan.

This is deliberately a view over ``image_selection_plan``. It does not inspect
image pixels, prepare phase maps, or launch a calibration job.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tomllib
from typing import Any, Mapping
from uuid import uuid4

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.image_selection_plan import (
    PURPOSES,
    resolve_selection,
    resolve_selection_files,
)
from darsia.fluidflower.series_paths import darsia_source_checkout


_SECTION_RE = re.compile(r"(?m)^\s*\[([^]\r\n]+)\]\s*(?:#.*)?$")
_SELECTED_PAGE_SIZE = 250


def _toml_number(value: float) -> str:
    return repr(float(value))


def render_purpose_section(purpose: str, settings: Mapping[str, Any]) -> str:
    """Render only the selected purpose; other TOML sections remain untouched."""

    if purpose not in PURPOSES:
        raise ValueError(f"Unknown purpose: {purpose}")
    mode = settings["mode"]
    lines = [f"[purposes.{purpose}]", f'mode = "{mode}"']
    if mode == "target_hours":
        targets = ", ".join(_toml_number(value) for value in settings["target_hours"])
        lines.append(f"target_hours = [{targets}]")
        lines.append(
            f'on_unmatched_target = "{settings.get("on_unmatched_target", "error")}"'
        )
    elif mode == "all":
        for key in ("start_hour", "end_hour"):
            if key in settings:
                lines.append(f"{key} = {_toml_number(settings[key])}")
    else:
        raise ValueError("Mode must be all or target_hours")
    lines.append(f'tolerance_seconds = {_toml_number(settings["tolerance_seconds"])}')
    if "required_end_hour" in settings:
        lines.append(f'required_end_hour = {_toml_number(settings["required_end_hour"])}')
    return "\n".join(lines) + "\n"


def update_plan_text(existing: str, purpose: str, settings: Mapping[str, Any]) -> str:
    """Replace one purpose in a versioned plan, preserving unrelated sections."""

    if existing.strip():
        document = tomllib.loads(existing)
        if document.get("schema_version") != 1:
            raise ValueError("Existing plan must have schema_version = 1")
    else:
        existing = "schema_version = 1\n"
    section = render_purpose_section(purpose, settings)
    matches = list(_SECTION_RE.finditer(existing))
    selected = next(
        (match for match in matches if match.group(1).strip() == f"purposes.{purpose}"),
        None,
    )
    if selected is None:
        updated = existing.rstrip() + "\n\n" + section
    else:
        following = next(
            (match.start() for match in matches if match.start() > selected.start()),
            len(existing),
        )
        old_lines = existing[selected.start() : following].splitlines(keepends=True)
        footer_start = len(old_lines)
        while footer_start and (
            not old_lines[footer_start - 1].strip()
            or old_lines[footer_start - 1].lstrip().startswith("#")
        ):
            footer_start -= 1
        # Keep comments adjacent to the next section, such as a reason for a
        # separately approved calibration branch.
        footer = "".join(old_lines[footer_start:])
        updated = existing[: selected.start()].rstrip() + "\n\n" + section
        updated += footer or ("\n" if following < len(existing) else "")
        updated += existing[following:]
    tomllib.loads(updated)
    return updated


def _format_hour(value: Any) -> str:
    return "—" if value is None else f"{float(value):g} h"


class ImageSelectionPlanDialog(QDialog):
    """Edit and preview review/calibration/production frame selection."""

    def __init__(
        self,
        project_root: Path,
        parent: QWidget | None = None,
        *,
        series_id: str = "",
    ) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).expanduser().resolve()
        self.series_id = str(series_id).strip().casefold().replace("-", "_")
        self._active_purpose = "review"
        self._drafts: dict[str, dict[str, Any]] = {}
        self._last_preview: dict[str, Any] | None = None
        self._selected_rows: list[tuple[str, int, str, str, str, str, str]] = []
        self.setWindowTitle("Image selection plan")
        self.resize(1040, 800)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Choose images separately for phase review, mass calibration and full "
            "production. A sparse native review manifest can select only images "
            "already prepared in that manifest; full production needs a complete "
            "acquisition inventory. Preview reads metadata only. A receipt records "
            "the selection and does not start phase or mass calculation."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        storage = QLabel(
            "Save the plan and receipts in the selected project folder. "
            "The project may be stored locally or on a network drive."
        )
        storage.setWordWrap(True)
        layout.addWidget(storage)

        form = QFormLayout()
        self.manifest_path = QLineEdit()
        self.manifest_path.setPlaceholderText("Existing JSON frame manifest")
        form.addRow(
            "Frame manifest",
            self._file_row(self.manifest_path, save=False, filter_text="JSON (*.json)"),
        )
        self.plan_path = QLineEdit()
        self.plan_path.setPlaceholderText("TOML path in the project's config folder")
        form.addRow(
            "Selection plan",
            self._file_row(self.plan_path, save=True, filter_text="TOML (*.toml)"),
        )
        self.plan_path.setText(str(self.project_root / "config" / "image_selection_plan.toml"))

        self.purpose = QComboBox()
        for value, label in (
            ("review", "Phase review"),
            ("calibration", "Concentration / mass calibration"),
            ("production", "Full production"),
        ):
            self.purpose.addItem(label, value)
        form.addRow("Purpose", self.purpose)
        self.mode = QComboBox()
        self.mode.addItem("Nearest images to target hours", "target_hours")
        self.mode.addItem("Every image in time window", "all")
        form.addRow("Selection", self.mode)
        self.targets = QLineEdit()
        self.targets.setPlaceholderText("Comma-separated hours, e.g. -0.05, 0.25, 1, 24, 96")
        form.addRow("Target hours", self.targets)
        self.tolerance = QLineEdit("300")
        form.addRow("Maximum time difference (s)", self.tolerance)
        self.required_end = QLineEdit()
        self.required_end.setPlaceholderText("Required last hour; explicit for full production")
        form.addRow("Required last hour", self.required_end)
        self.start_hour = QLineEdit()
        self.start_hour.setPlaceholderText("Optional")
        form.addRow("Start hour (all)", self.start_hour)
        self.end_hour = QLineEdit()
        self.end_hour.setPlaceholderText("Optional")
        form.addRow("End hour (all)", self.end_hour)
        self.unmatched = QComboBox()
        self.unmatched.addItem("Fail if a target is absent", "error")
        self.unmatched.addItem("Skip early missing targets; still fail on late coverage", "skip")
        form.addRow("Unmatched targets", self.unmatched)
        self.check_files = QCheckBox(
            "Check source file existence (metadata only; may be slow on network storage)"
        )
        form.addRow("Source check", self.check_files)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.load_button = QPushButton("Load saved plan")
        self.preview_button = QPushButton("Preview image selection")
        self.save_button = QPushButton("Save this purpose to plan")
        for button in (self.load_button, self.preview_button, self.save_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.summary = QLabel("Choose a manifest and a purpose, then preview the selection.")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(
            [
                "Run", "Status", "Input", "Eligible", "Selected", "Missing",
                "Last available", "Last selected",
            ]
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.table, 1)
        selected_heading = QHBoxLayout()
        selected_heading.addWidget(QLabel("Selected source images (original indices shown)"))
        selected_heading.addStretch(1)
        self.selected_count = QLabel("0 selected")
        selected_heading.addWidget(self.selected_count)
        selected_heading.addWidget(QLabel("Page"))
        self.selected_page = QSpinBox()
        self.selected_page.setMinimum(1)
        self.selected_page.setMaximum(1)
        selected_heading.addWidget(self.selected_page)
        layout.addLayout(selected_heading)
        self.selected_table = QTableWidget(0, 7)
        self.selected_table.setHorizontalHeaderLabels(
            [
                "Run", "Original index", "File", "Actual time", "Target hour(s)",
                "Deviation(s)", "Source path",
            ]
        )
        self.selected_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.selected_table.setMinimumHeight(150)
        layout.addWidget(self.selected_table, 2)
        self.audit = QPlainTextEdit()
        self.audit.setReadOnly(True)
        self.audit.setPlaceholderText(
            "Unmatched targets and omitted/missing-image reasons appear here."
        )
        self.audit.setMaximumHeight(130)
        layout.addWidget(self.audit)

        receipt_row = QHBoxLayout()
        self.receipt_path = QLineEdit()
        self.receipt_path.setPlaceholderText(
            "New JSON receipt path in the project's results folder"
        )
        receipt_row.addWidget(self.receipt_path, 1)
        choose_receipt = QPushButton("Browse...")
        choose_receipt.clicked.connect(
            lambda: self._browse(
                self.receipt_path, save=True, filter_text="JSON (*.json)"
            )
        )
        receipt_row.addWidget(choose_receipt)
        self.receipt_button = QPushButton("Create selection receipt")
        receipt_row.addWidget(self.receipt_button)
        layout.addLayout(receipt_row)

        self.mode.currentIndexChanged.connect(self._mode_changed)
        self.purpose.currentIndexChanged.connect(self._purpose_changed)
        self.plan_path.textChanged.connect(self._plan_path_changed)
        self.load_button.clicked.connect(self._load_clicked)
        self.preview_button.clicked.connect(self._preview_clicked)
        self.save_button.clicked.connect(self._save_clicked)
        self.receipt_button.clicked.connect(self._receipt_clicked)
        self.selected_page.valueChanged.connect(self._display_selected_page)
        self._mode_changed()
        if self.plan_path.text():
            self.load_selected_purpose()

    def _file_row(self, editor: QLineEdit, *, save: bool, filter_text: str) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(editor, 1)
        browse = QPushButton("Browse...")
        browse.clicked.connect(lambda: self._browse(editor, save=save, filter_text=filter_text))
        layout.addWidget(browse)
        return row

    def _browse(self, editor: QLineEdit, *, save: bool, filter_text: str) -> None:
        initial = editor.text().strip() or str(self.project_root)
        if save:
            selected, _ = QFileDialog.getSaveFileName(
                self, "Choose output path", initial, filter_text
            )
        else:
            selected, _ = QFileDialog.getOpenFileName(
                self, "Choose frame manifest", initial, filter_text
            )
        if selected:
            editor.setText(selected)

    def _form_snapshot(self) -> dict[str, Any]:
        return {
            "mode": self.mode.currentData(),
            "targets": self.targets.text(),
            "tolerance": self.tolerance.text(),
            "required_end": self.required_end.text(),
            "start_hour": self.start_hour.text(),
            "end_hour": self.end_hour.text(),
            "unmatched": self.unmatched.currentData(),
        }

    def _set_form(self, values: Mapping[str, Any]) -> None:
        self.mode.setCurrentIndex(max(0, self.mode.findData(values.get("mode", "target_hours"))))
        targets = values.get("target_hours", values.get("targets", ""))
        self.targets.setText(
            ", ".join(f"{value:g}" for value in targets)
            if isinstance(targets, list)
            else str(targets)
        )
        self.tolerance.setText(
            str(values.get("tolerance_seconds", values.get("tolerance", "300")))
        )
        self.required_end.setText(
            str(values.get("required_end_hour", values.get("required_end", "")))
        )
        self.start_hour.setText(str(values.get("start_hour", "")))
        self.end_hour.setText(str(values.get("end_hour", "")))
        choice = values.get("on_unmatched_target", values.get("unmatched", "error"))
        self.unmatched.setCurrentIndex(max(0, self.unmatched.findData(choice)))
        self._mode_changed()

    def _purpose_changed(self) -> None:
        self._drafts[self._active_purpose] = self._form_snapshot()
        self._active_purpose = str(self.purpose.currentData())
        if self._active_purpose in self._drafts:
            self._set_form(self._drafts[self._active_purpose])
        else:
            self.load_selected_purpose()
        self._last_preview = None

    def _mode_changed(self) -> None:
        is_target = self.mode.currentData() == "target_hours"
        self.targets.setEnabled(is_target)
        self.unmatched.setEnabled(is_target)
        self.start_hour.setEnabled(not is_target)
        self.end_hour.setEnabled(not is_target)

    def _plan_path_changed(self) -> None:
        self._drafts.clear()
        self._last_preview = None
        self.summary.setText(
            "Plan path changed. Load the saved plan to inspect it, or edit and save "
            "one purpose to this explicit destination."
        )

    def _plan_document(self) -> dict[str, Any]:
        path_text = self.plan_path.text().strip()
        if path_text and Path(path_text).is_file():
            with Path(path_text).open("rb") as stream:
                data = tomllib.load(stream)
            if data.get("schema_version") != 1:
                raise ValueError("Existing plan must have schema_version = 1")
            return data
        return {"schema_version": 1, "purposes": {}}

    def load_selected_purpose(self) -> None:
        document = self._plan_document()
        section = (document.get("purposes") or {}).get(self._active_purpose) or {}
        self._set_form(section)
        self._drafts.pop(self._active_purpose, None)
        self._last_preview = None

    def _settings(self) -> dict[str, Any]:
        settings: dict[str, Any] = {
            "mode": str(self.mode.currentData()),
            "tolerance_seconds": float(self.tolerance.text().strip()),
        }
        if self.required_end.text().strip():
            settings["required_end_hour"] = float(self.required_end.text().strip())
        if settings["mode"] == "target_hours":
            settings["target_hours"] = [
                float(piece.strip())
                for piece in self.targets.text().split(",")
                if piece.strip()
            ]
            settings["on_unmatched_target"] = str(self.unmatched.currentData())
        else:
            for key, widget in (("start_hour", self.start_hour), ("end_hour", self.end_hour)):
                if widget.text().strip():
                    settings[key] = float(widget.text().strip())
        return settings

    def _edited_document(self) -> dict[str, Any]:
        document = self._plan_document()
        document.setdefault("purposes", {})[self._active_purpose] = self._settings()
        # The resolver has the plan-schema validation. This synthetic metadata
        # input exercises only that validation; its late-coverage status is not
        # a prediction about the user's actual source manifest.
        resolve_selection(
            {"run": "settings-check", "frames": [{"source_path": "check", "elapsed_h": 0.0}]},
            document,
            self._active_purpose,
        )
        return document

    def preview_selection(self) -> dict[str, Any]:
        manifest_path = Path(self.manifest_path.text().strip())
        if not self.manifest_path.text().strip() or not manifest_path.is_file():
            raise ValueError("Choose an existing JSON frame manifest")
        with manifest_path.open("r", encoding="utf-8-sig") as stream:
            manifest = json.load(stream)
        document = self._edited_document()
        receipt = resolve_selection(
            manifest,
            document,
            self._active_purpose,
            check_files=self.check_files.isChecked(),
            source_root=manifest_path.parent,
            manifest_path=manifest_path,
            plan_path=self.plan_path.text().strip() or None,
        )
        self._last_preview = receipt
        self._display_receipt(receipt)
        return receipt

    def _display_receipt(self, receipt: Mapping[str, Any]) -> None:
        runs = receipt["runs"]
        self.table.setRowCount(len(runs))
        details: list[str] = []
        selected_rows: list[tuple[str, int, str, str, str, str, str]] = []
        for row_index, run in enumerate(runs):
            missing = sum("missing" in str(row["reason"]) for row in run["omitted"])
            values = (
                run["run_id"], run["status"], run["input_frame_count"],
                run["eligible_frame_count"], run["selected_frame_count"], missing,
                _format_hour(run["last_available_hour"]), _format_hour(run["last_selected_hour"]),
            )
            for col, value in enumerate(values):
                self.table.setItem(row_index, col, QTableWidgetItem(str(value)))
            details.append(
                f'{run["run_id"]}: required last {_format_hour(run["required_end_hour"])}; '
                f'{len(run["unmatched_targets"])} unmatched target(s)'
            )
            details.extend(f'  {item["code"]}: {item["message"]}' for item in run["issues"])
            reasons: dict[str, int] = {}
            for item in run["omitted"]:
                reason = str(item["reason"])
                reasons[reason] = reasons.get(reason, 0) + 1
            if reasons:
                details.append(
                    "  Omitted: "
                    + ", ".join(
                        f"{key}={value}" for key, value in sorted(reasons.items())
                    )
                )
            matches_by_index: dict[int, list[Mapping[str, Any]]] = {}
            for match in run["target_matches"]:
                matches_by_index.setdefault(int(match["input_index"]), []).append(match)
            for frame in run["selected"]:
                matches = matches_by_index.get(int(frame["input_index"]), [])
                source = str(frame["source_path"])
                selected_rows.append(
                    (
                        str(run["run_id"]),
                        int(frame["input_index"]),
                        Path(source).name,
                        _format_hour(frame["elapsed_h"]),
                        ", ".join(f'{float(item["target_hour"]):g}' for item in matches),
                        ", ".join(
                            f'{float(item["distance_seconds"]):g} s' for item in matches
                        ),
                        source,
                    )
                )
        self.audit.setPlainText("\n".join(details))
        selected = sum(int(run["selected_frame_count"]) for run in runs)
        self._selected_rows = selected_rows
        pages = max(1, (len(selected_rows) + _SELECTED_PAGE_SIZE - 1) // _SELECTED_PAGE_SIZE)
        self.selected_page.blockSignals(True)
        self.selected_page.setMaximum(pages)
        self.selected_page.setValue(1)
        self.selected_page.blockSignals(False)
        self._display_selected_page()
        self.summary.setText(
            f'{receipt["status"].upper()}: {selected} image(s) selected across '
            f'{len(runs)} run(s). Preview is metadata only; save the plan before '
            "creating an immutable receipt."
        )
        self.table.resizeColumnsToContents()

    def _display_selected_page(self) -> None:
        start = (self.selected_page.value() - 1) * _SELECTED_PAGE_SIZE
        rows = self._selected_rows[start : start + _SELECTED_PAGE_SIZE]
        self.selected_table.setRowCount(len(rows))
        for row_index, values in enumerate(rows):
            for col, value in enumerate(values):
                cell = QTableWidgetItem(str(value))
                if col == 2:
                    cell.setToolTip(values[-1])
                self.selected_table.setItem(row_index, col, cell)
        self.selected_count.setText(
            f"{len(self._selected_rows)} selected; showing {start + 1 if rows else 0}–"
            f"{start + len(rows)}"
        )
        self.selected_table.resizeColumnsToContents()

    def _output_path(self, text: str, *, label: str) -> Path:
        if not text.strip():
            raise ValueError(f"Choose a {label} path in the selected project")
        path = Path(text.strip()).expanduser().resolve()
        if not path.is_relative_to(self.project_root):
            raise ValueError("Save in the selected project")
        # The project folder may be its own Git repository; only the DarSIA
        # source checkout itself never receives generated plans or receipts.
        checkout = darsia_source_checkout()
        if checkout is not None and path.is_relative_to(checkout):
            raise ValueError(
                "Generated plans and receipts must be outside the DarSIA source checkout"
            )
        expected_folder = "results" if label.startswith("new JSON") else "config"
        relative = path.relative_to(self.project_root)
        if not relative.parts or relative.parts[0].casefold() != expected_folder:
            raise ValueError(f"Save the {label} under {self.project_root / expected_folder}")
        return path

    def save_plan(self) -> Path:
        path = self._output_path(self.plan_path.text(), label="TOML plan")
        settings = self._settings()
        self._edited_document()
        previous_bytes = path.read_bytes() if path.is_file() else None
        existing = (
            previous_bytes.decode("utf-8-sig") if previous_bytes is not None else ""
        )
        updated = update_plan_text(existing, self._active_purpose, settings)
        # Confirm the persisted representation resolves exactly as the UI form.
        parsed = tomllib.loads(updated)
        resolve_selection(
            {"run": "settings-check", "frames": [{"source_path": "check", "elapsed_h": 0.0}]},
            parsed,
            self._active_purpose,
        )
        if parsed["purposes"][self._active_purpose] != settings:
            raise ValueError("The saved TOML section does not match the edited values")
        updated_bytes = updated.encode("utf-8")
        if previous_bytes == updated_bytes:
            self.summary.setText(f"Plan is already current: {path}")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        backup: Path | None = None
        if previous_bytes is not None:
            digest = hashlib.sha256(previous_bytes).hexdigest()
            backup = path.with_name(f"{path.stem}.{digest}.backup{path.suffix}")
            try:
                with backup.open("xb") as stream:
                    stream.write(previous_bytes)
            except FileExistsError:
                if backup.read_bytes() != previous_bytes:
                    raise ValueError("Existing plan backup does not match its SHA-256 name")
            if path.read_bytes() != previous_bytes:
                raise ValueError("Plan changed during editing; reload before saving")
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(updated_bytes)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        self._drafts.pop(self._active_purpose, None)
        self._last_preview = None
        self.summary.setText(
            f"Saved {self._active_purpose} selection plan: {path}. "
            + (f"Previous bytes preserved at {backup}. " if backup else "")
            + "Create a new receipt before using the changed plan."
        )
        return path

    def create_receipt(self) -> Path:
        plan_path = self._output_path(self.plan_path.text(), label="TOML plan")
        manifest_path = Path(self.manifest_path.text().strip())
        if not plan_path.is_file() or not manifest_path.is_file():
            raise ValueError("Save the plan and choose an existing frame manifest first")
        document = self._plan_document()
        saved = (document.get("purposes") or {}).get(self._active_purpose)
        if saved != self._settings():
            raise ValueError("The form differs from the saved plan; save this purpose first")
        output = self._output_path(self.receipt_path.text(), label="new JSON receipt")
        if output in {plan_path.resolve(), manifest_path.resolve()}:
            raise ValueError("Receipt path must differ from the plan and manifest")
        receipt = resolve_selection_files(
            manifest_path, plan_path, self._active_purpose,
            check_files=self.check_files.isChecked(),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        self._display_receipt(receipt)
        self.summary.setText(
            f'Selection receipt saved ({receipt["status"]}): {output}. '
            "No phase or mass computation was started."
        )
        return output

    def _show_error(self, operation: str, error: Exception) -> None:
        QMessageBox.warning(self, operation, f"{type(error).__name__}: {error}")

    def _load_clicked(self) -> None:
        try:
            self.load_selected_purpose()
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            self._show_error("Load image selection plan", error)

    def _preview_clicked(self) -> None:
        try:
            self.preview_selection()
        except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
            self._show_error("Preview image selection", error)

    def _save_clicked(self) -> None:
        try:
            self.save_plan()
        except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
            self._show_error("Save image selection plan", error)

    def _receipt_clicked(self) -> None:
        try:
            self.create_receipt()
        except (OSError, ValueError, json.JSONDecodeError, tomllib.TOMLDecodeError) as error:
            self._show_error("Create image selection receipt", error)
