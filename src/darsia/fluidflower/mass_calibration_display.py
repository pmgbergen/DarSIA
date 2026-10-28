"""Read-only saved mass-calibration metadata for the native Calibration page.

Only explicit paths and an optional, already bound current result are read.
Nothing here fits a calibration, calculates a metric, changes parameters, starts
jobs, discovers history, opens images/NPZ, or writes any file. The Phase-map
method's page also shows the series' refit job (``refit_job_panel``), which is
the one place on this page that creates and starts a job, and only on request.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

NOT_REPORTED = "Not reported"
NOT_IN_CALIBRATION = "Not in this calibration document"
MAX_JSON_BYTES = 1024 * 1024
MAX_CSV_BYTES = 256 * 1024
MAX_SIGNAL_FILES = 64
MAX_DIRECTORY_ENTRIES = 256
MAX_SIGNAL_ROWS = 128
PRESSURE_COLUMNS = ("Run", "Source", "Pressure · temperature", "Time span", "Status", "In the fit")
MODE_NOTICE = (
    "Manual and automatic calibration runs will be available on the Jobs page. "
    "This page shows saved values and reported errors."
)
REFIT_MODE_NOTICE = (
    "Automatic mass calibration fits the gain and beta to the injected mass of the fit runs. "
    "It is a job like any other (it also shows on the Jobs page); this page creates it, follows "
    "it and shows its frozen result. Manual parameters are not available yet."
)


@dataclass(frozen=True)
class SavedMassCalibrationData:
    # (parameter or metric name, literal saved value, scope/provenance)
    rows: tuple[tuple[str, str, str], ...]
    source_label: str
    notes: tuple[str, ...] = ()


def _text(value) -> str:
    """Format a saved scalar only; no derived quantity or guessed default."""
    if value is None or isinstance(value, (bool, Mapping, list, tuple)):
        return NOT_REPORTED
    if isinstance(value, float) and not math.isfinite(value):
        return NOT_REPORTED
    return str(value)


def _path(reference, series_root) -> Path | None:
    value = reference.get("path") if isinstance(reference, Mapping) else reference
    if value is None or not str(value).strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        if series_root is None:
            raise ValueError("A relative reference needs the selected series folder.")
        path = Path(series_root) / path
    from .storage_paths import resolve_storage_path
    return resolve_storage_path(path, series_roots=(() if series_root is None else (Path(series_root),)))


def _small_bytes(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"Metadata exceeds the display limit ({limit} bytes): {path}")
    return raw


def _json(path: Path) -> dict:
    value = json.loads(_small_bytes(path, MAX_JSON_BYTES))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a saved JSON object: {path}")
    return value


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _error_row(values, source: str) -> tuple[str, str, str] | None:
    """Use known saved error fields verbatim; never derive an error from mass."""
    metrics = _mapping(values)
    for key in ("mean_run_nmae", "nmae", "nrmse", "rmse_g"):
        value = _text(metrics.get(key))
        if value != NOT_REPORTED:
            units = "g" if key == "rmse_g" else "saved fraction"
            return ("Saved mass-balance error", f"{key} = {value} ({units})", source)
    return None


def _bound_matching_result(reference, matching_result, notes) -> Mapping:
    """Caller supplies an already matched result, never an arbitrary selection.

    When the explicit reference also has a SHA and the result declares its SHA,
    reject disagreement. This does not claim to verify a file's content digest.
    """
    result = _mapping(matching_result)
    expected = _mapping(reference).get("sha256")
    found = (
        _mapping(result.get("bindings")).get("calibration_sha256")
        or _mapping(result.get("scalar_result")).get("calibration_sha256")
        or _mapping(result.get("calibration")).get("sha256")
        or result.get("calibration_sha256")
    )
    if expected and found and expected != found:
        notes.append(
            "The supplied result declares another calibration; its frame values are not shown."
        )
        return {}
    return result


def _phase_display(reference, series_root, matching_result) -> SavedMassCalibrationData:
    rows, notes, saved = [], [], {}
    try:
        path = _path(reference, series_root)
        if path is not None:
            saved = _json(path)
        else:
            notes.append("No phase-model mass calibration is referenced.")
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        path = None
        notes.append(str(error))
    source = (
        str(path)
        if path is not None
        else "No readable explicit phase calibration reference"
    )
    frozen_round4 = saved.get("contract") == "ffac-common-mass-round3-frozen-v1" and saved.get("frozen") is True
    absent = NOT_IN_CALIBRATION if frozen_round4 else NOT_REPORTED
    parameters = _mapping(saved.get("parameters"))
    if not parameters:
        parameters = _mapping(_mapping(saved.get("calibration")).get("parameters"))
    if not parameters:
        parameters = _mapping(_mapping(saved.get("scalar_result")).get("parameters"))
    for title, key in (("Gain", "gain"), ("Beta", "beta")):
        rows.append(
            (title, _text(parameters.get(key)) if key in parameters else absent, "Shared saved calibration; " + source)
        )
    chemistry = saved.get("chemistry_model")
    rows.append(("Chemistry model", "Titration" if chemistry == "titration" else (
        str(chemistry) if chemistry is not None else absent), "Named saved calibration; " + source))

    result = _bound_matching_result(reference, matching_result, notes)
    # A full result may have many frames. Never silently choose its first frame.
    frame = _mapping(result.get("frame"))
    if not frame and "frames" not in result:
        frame = result
    scalar = _mapping(frame.get("scalar_result"))
    factor = frame.get("dilution_factor")
    if factor is None:
        factor = scalar.get("dilution_factor")
    if factor is None:
        factor = _mapping(frame.get("prediction")).get("dilution_factor")
    anchor = _mapping(frame.get("anchor"))
    if not anchor:
        anchor = _mapping(result.get("anchor"))
    anchor_fields = {
        key: anchor[key]
        for key in ("elapsed_h", "affected_pore_volume_m3")
        if _text(anchor.get(key)) != NOT_REPORTED
    }
    anchor_text = (
        json.dumps(anchor_fields, ensure_ascii=False) if anchor_fields else absent
    )
    frame_source = "Selected saved image"
    rows.extend(
        (
            ("Dilution factor", _text(factor) if factor is not None else absent, "Per image; " + frame_source),
            (
                "Anchor",
                anchor_text,
                "Per run: time [h], affected pore volume [m3]; " + frame_source,
            ),
        )
    )
    error = None
    for metrics, where in (
        (saved.get("metrics"), source),
        (_mapping(saved.get("calibration")).get("metrics"), source),
        (
            _mapping(result.get("calibration")).get("metrics"),
            "Saved result calibration",
        ),
        (result.get("metrics"), "Saved result"),
    ):
        error = _error_row(metrics, where)
        if error is not None:
            break
    rows.append(
        error
        or (
            "Saved mass-balance error",
            absent,
            "No error is reported with this saved calibration.",
        )
    )
    notes.append(
        "Gain and beta are shared by the calibration. View a saved result under Analysis for its per-image dilution factor and per-run anchor."
    )
    notes.append(
        "The fit error describes agreement with the calibration observations. A frozen calibration may have no error metric reported."
    )
    if saved.get("overflow_phase_split_policy"):
        counts = {}
        for item in saved.get("overflow_phase_split_status", []):
            if not isinstance(item, dict):
                continue
            target = _mapping(item.get("target"))
            status = {"own_measurement": "Measured", "pressure_matched_estimate": "ESTIMATED"}.get(
                item.get("kind") or target.get("kind"), "Phase model — no overflow reference")
            if item.get("donor_runs"):
                status += " · donors " + ", ".join(map(str, item["donor_runs"]))
            if item.get("pressure_difference_bar_range"):
                status += " · ΔP [bar] " + str(item["pressure_difference_bar_range"])
            key = (str(item.get("run") or "Unknown run"), status)
            counts[key] = counts.get(key, 0) + int(item.get("image_count", 1))
        for (run, status), count in sorted(counts.items()):
            rows.append(("Overflow phase split · " + run, status + f" · {count} image(s)", source))
        notes.append("Overflow constrains the phase split. ESTIMATED values record donor runs and pressure differences in the saved preparation; total mass is fitted to the injection protocol.")
    return SavedMassCalibrationData(tuple(rows), source, tuple(notes))


def _signal_rows(folder: Path, notes) -> list[tuple[str, str, str]]:
    signal_folder = folder / "signal_model"
    selected = []
    try:
        for count, path in enumerate(signal_folder.iterdir(), 1):
            if count > MAX_DIRECTORY_ENTRIES:
                raise ValueError(
                    "Signal-model folder exceeds the bounded display listing."
                )
            if (
                path.name.startswith("signal_model_")
                and path.suffix.lower() == ".csv"
                and path.is_file()
            ):
                selected.append(path)
                if len(selected) > MAX_SIGNAL_FILES:
                    raise ValueError("Too many signal CSV files for this display.")
    except (OSError, ValueError) as error:
        notes.append(str(error))
        return []
    rows = []
    for path in sorted(selected, key=lambda item: item.name):
        try:
            text = _small_bytes(path, MAX_CSV_BYTES).decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            if not {"supports", "values"}.issubset(reader.fieldnames or []):
                raise ValueError(f"Signal CSV lacks supports/values: {path}")
            supports, values = [], []
            for index, row in enumerate(reader, 1):
                if index > MAX_SIGNAL_ROWS:
                    raise ValueError(
                        f"Signal CSV exceeds the display row limit: {path}"
                    )
                supports.append(str(row.get("supports") or ""))
                values.append(str(row.get("values") or ""))
            label = path.stem.removeprefix("signal_model_")
            rows.append(
                (f"Signal supports (facies {label})", json.dumps(supports), str(path))
            )
            rows.append(
                (f"Signal values (facies {label})", json.dumps(values), str(path))
            )
        except (OSError, ValueError, TypeError, UnicodeError) as error:
            notes.append(str(error))
    return rows


def _same_path(first: Path, second: Path) -> bool:
    # Absolute explicit paths only. No is_relative_to or filesystem traversal.
    return os.path.abspath(first).casefold() == os.path.abspath(second).casefold()


def _colour_display(
    reference, series_root, optuna_summary_reference
) -> SavedMassCalibrationData:
    rows, notes, flash = [], [], {}
    try:
        folder = _path(reference, series_root)
    except (ValueError, TypeError) as error:
        folder = None
        notes.append(str(error))
    source = (
        str(folder)
        if folder is not None
        else "No explicit colour-path calibration folder"
    )
    if folder is not None:
        rows.extend(_signal_rows(folder, notes))
        try:
            flash = _json(folder / "flash" / "flash.json")
        except (OSError, ValueError, TypeError, UnicodeError) as error:
            notes.append(str(error))
    if not rows:
        rows.append(("Signal values by facies", NOT_REPORTED, source))
    for key in ("min_value_aq", "max_value_aq", "min_value_g", "max_value_g"):
        rows.append(
            ("Flash " + key, _text(flash.get(key)), source + "/flash/flash.json")
        )
    report_error = None
    if optuna_summary_reference:
        try:
            report_path = _path(optuna_summary_reference, series_root)
            report = _json(report_path)
            bound_folder = _path(report.get("calibration_folder"), series_root)
            if (
                folder is None
                or bound_folder is None
                or not _same_path(folder, bound_folder)
            ):
                raise ValueError(
                    "The explicit Optuna report is not bound to the selected colour calibration folder."
                )
            date = _text(report.get("finished"))
            saved = report.get("saved")
            notes.append(
                f"Explicit Optuna report: {report_path}; finished: {date}; saved: {saved!r}."
            )
            notes.append(
                "Its error and best values describe that report, not a new measurement of current manual slider values."
            )
            notes.append(
                "Report configuration: "
                + json.dumps(report.get("configuration"), ensure_ascii=False)
            )
            if saved is not True:
                notes.append(
                    "The report does not say its best values were adopted into the calibration."
                )
            percentage = _text(report.get("mean_error_percent_best"))
            if percentage != NOT_REPORTED:
                report_error = (
                    "Saved mass-balance error (Optuna report)",
                    percentage + " %",
                    f"Historical report at {date}; {report_path}; saved={saved!r}",
                )
            elif _text(report.get("objective_best")) != NOT_REPORTED:
                report_error = (
                    "Saved objective_best (Optuna report)",
                    _text(report["objective_best"]),
                    "Report objective: "
                    + str(
                        _mapping(report.get("settings")).get("objective", NOT_REPORTED)
                    ),
                )
            for label, values in _mapping(report.get("values_best")).items():
                rows.append(
                    (
                        f"Report best signal values (facies {label})",
                        json.dumps(values, ensure_ascii=False),
                        f"Optuna report only; {report_path}",
                    )
                )
        except (OSError, ValueError, TypeError, UnicodeError) as error:
            notes.append(str(error))
    else:
        notes.append(
            "No Optuna report is explicitly referenced; history folders are not searched."
        )
    rows.append(
        report_error
        or (
            "Saved mass-balance error",
            NOT_REPORTED,
            "No separate mass-balance error is reported with this calibration.",
        )
    )
    return SavedMassCalibrationData(tuple(rows), source, tuple(notes))


def read_mass_calibration_display(
    method: str,
    reference,
    *,
    series_root: str | Path | None = None,
    matching_result: Mapping | None = None,
    optuna_summary_reference=None,
) -> SavedMassCalibrationData:
    """Read bounded, explicitly referenced metadata only.

    ``reference`` is a calibration JSON path for phase_model, a calibration folder
    for colour_path, or an existing {path, sha256, ...} descriptor. Relative paths
    require series_root. The caller may supply a selected frame/full result only
    after matching it to this exact calibration. To select a frame from a full
    result, pass it explicitly as matching_result['frame']; no first-frame default
    is made. Optuna summaries must be explicitly supplied and bind the same folder.
    No library target, old global fit, latest report or result path is followed.
    """
    if method == "phase_model":
        return _phase_display(reference, series_root, matching_result)
    if method == "colour_path":
        return _colour_display(reference, series_root, optuna_summary_reference)
    return SavedMassCalibrationData(
        (("Saved calibration", NOT_REPORTED, "No method is chosen for this series."),),
        "Method: not chosen",
        (
            "Choose the method explicitly under Calibration. This display does not choose one.",
        ),
    )


class MassCalibrationDisplay(QWidget):
    """Reusable read-only table; both calculation modes stay disabled by default.

    A page may also show each run's pressure and temperature source below it,
    from rows that ``pressure_source`` reads; the widget itself reads nothing.
    The Phase-map method's page offers the series' refit job beside the saved
    values (:meth:`set_refit_series`); Automatic is then the chosen mode.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Mass calibration — saved values and saved error"))
        self.source = QLabel()
        self.source.setTextFormat(Qt.TextFormat.PlainText)
        self.source.setWordWrap(True)
        self.source.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.source)
        modes = QHBoxLayout()
        modes.addWidget(QLabel("Parameters:"))
        self.manual_button = QRadioButton("Manual")
        self.automatic_button = QRadioButton("Automatic (fitted to injected mass)")
        self.mode_buttons = QButtonGroup(self)
        for button in (self.manual_button, self.automatic_button):
            self.mode_buttons.addButton(button)
            button.setEnabled(False)
            modes.addWidget(button)
        layout.addLayout(modes)
        self.mode_notice = QLabel(MODE_NOTICE)
        self.mode_notice.setWordWrap(True)
        layout.addWidget(self.mode_notice)
        # The saved values beside the refit job (N141 spec 2.4), in an adjustable
        # splitter: the page grows in width, not in height.
        self.columns = QSplitter(Qt.Orientation.Horizontal)
        self.columns.setChildrenCollapsible(False)
        saved = QWidget()
        column = QVBoxLayout(saved)
        column.setContentsMargins(0, 0, 0, 0)
        self.parameters = QTableWidget(0, 3)
        self.parameters.setHorizontalHeaderLabels(
            ("Parameter / metric", "Saved value", "Scope / source")
        )
        self.parameters.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.parameters.horizontalHeader().setStretchLastSection(True)
        self.parameters.verticalHeader().hide()
        column.addWidget(self.parameters, 1)
        self.notes = QLabel()
        self.notes.setTextFormat(Qt.TextFormat.PlainText)
        self.notes.setWordWrap(True)
        self.notes.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        column.addWidget(self.notes)
        # Pressure and temperature per run (whole series). Hidden until a page
        # hands it rows; the read-only calibration dialog never does.
        self.pressure_box = QWidget()
        pressure = QVBoxLayout(self.pressure_box)
        pressure.setContentsMargins(0, 0, 0, 0)
        self.pressure_title = QLabel()
        self.pressure_title.setWordWrap(True)
        pressure.addWidget(self.pressure_title)
        self.pressure_table = QTableWidget(0, len(PRESSURE_COLUMNS))
        self.pressure_table.setHorizontalHeaderLabels(PRESSURE_COLUMNS)
        self.pressure_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pressure_table.horizontalHeader().setStretchLastSection(True)
        self.pressure_table.verticalHeader().hide()
        pressure.addWidget(self.pressure_table, 1)
        self.pressure_notes = QLabel()
        self.pressure_notes.setTextFormat(Qt.TextFormat.PlainText)
        self.pressure_notes.setWordWrap(True)
        self.pressure_notes.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        pressure.addWidget(self.pressure_notes)
        column.addWidget(self.pressure_box, 1)
        self.pressure_box.hide()
        self.columns.addWidget(saved)
        from darsia.fluidflower.refit_job_panel import RefitJobPanel

        self.refit_panel = RefitJobPanel()
        # The refit column scrolls on its own when the window is short, so the
        # page's minimum size, and the window's, stays as it was. It is the
        # page's only scroll area.
        self.refit_area = QScrollArea()
        self.refit_area.setWidgetResizable(True)
        self.refit_area.setFrameShape(QFrame.Shape.NoFrame)
        self.refit_area.setWidget(self.refit_panel)
        self.refit_area.hide()
        self.columns.addWidget(self.refit_area)
        self.columns.setStretchFactor(0, 1)
        self.columns.setStretchFactor(1, 1)
        layout.addWidget(self.columns, 1)
        self.overflow_choice = OverflowChoicePage(diagnostic=True)
        self.overflow_choice.steer.setEnabled(True)
        self.overflow_choice.steer.clicked.connect(
            lambda: self.refit_panel.set_overflow_enabled(True))
        self.overflow_choice.hide()
        layout.addWidget(self.overflow_choice)
        self._refit_series = None
        self.pressure_complete = True
        self._pressure_steps = None
        self._pressure_timer = QTimer(self)
        self._pressure_timer.setInterval(0)
        self._pressure_timer.timeout.connect(self._next_pressure_step)
        self.set_display(read_mass_calibration_display("", None))

    def set_refit_series(self, series_root) -> None:
        """Offer this series' refit job beside the saved values; None hides it.

        The refit job is the automatic mode, so Automatic is shown chosen while
        it is offered. Both buttons stay disabled: Manual is not available, so
        there is nothing to switch to. The read-only calibration dialog never
        calls this, so neither mode is chosen there.
        """
        offered = series_root is not None
        if series_root != self._refit_series:
            self.overflow_choice.reset_diagnostic()
            self._refit_series = series_root
        self.refit_panel.set_series(series_root)
        self.refit_area.setVisible(offered)
        self.overflow_choice.setVisible(offered)
        self.mode_buttons.setExclusive(False)
        self.manual_button.setChecked(False)
        self.automatic_button.setChecked(offered)
        self.mode_buttons.setExclusive(True)
        self.mode_notice.setText(REFIT_MODE_NOTICE if offered else MODE_NOTICE)

    def set_pressure_sources(self, sources) -> None:
        """Render read pressure rows (pressure_source.PressureSources); no I/O.

        None hides the table.
        """
        if sources is None:
            self.pressure_table.setRowCount(0)
            self.pressure_box.hide()
            return
        rows = tuple(sources.rows)
        self.pressure_complete = bool(sources.complete)
        self.pressure_title.setText(
            f"Pressure and temperature per run — whole series ({len(rows)} runs)"
        )
        self.pressure_table.setRowCount(len(rows))
        for index, row in enumerate(rows):
            cells = (str(row.run).upper(), row.source, row.values, row.span, row.status, row.fit)
            for column, value in enumerate(cells):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(row.detail or str(value))
                self.pressure_table.setItem(index, column, item)
        self.pressure_table.resizeColumnsToContents()
        self.pressure_notes.setText("\n".join(sources.notes))
        self.pressure_box.show()

    def follow_pressure_sources(self, snapshots) -> None:
        """Show each snapshot of an iterator of PressureSources as it is read.

        One snapshot is taken per Qt timer tick in this thread, so the window
        stays responsive while network files are read; a new call replaces the
        previous iterator. None hides the table.
        """
        self._pressure_timer.stop()
        self._pressure_steps = None
        if snapshots is None:
            self.pressure_complete = True
            self.set_pressure_sources(None)
            return
        self.pressure_complete = False
        self._pressure_steps = iter(snapshots)
        self._next_pressure_step()
        if self._pressure_steps is not None:
            self._pressure_timer.start()

    def _next_pressure_step(self) -> None:
        steps = self._pressure_steps
        if steps is None:
            self._pressure_timer.stop()
            return
        try:
            snapshot = next(steps)
        except StopIteration:
            snapshot = None
        except (OSError, ValueError, TypeError, KeyError) as error:
            snapshot = None
            self.pressure_notes.setText(f"Pressure sources could not be read: {error}")
            self.pressure_box.show()
        if snapshot is None:
            self._pressure_timer.stop()
            self._pressure_steps = None
            self.pressure_complete = True
            return
        self.set_pressure_sources(snapshot)
        if snapshot.complete:
            self._pressure_timer.stop()
            self._pressure_steps = None

    def set_display(self, display: SavedMassCalibrationData) -> None:
        """Render already loaded metadata; this method performs no I/O."""
        self.source.setText(display.source_label)
        self.notes.setText("\n".join(display.notes))
        self.parameters.setRowCount(len(display.rows))
        for index, values in enumerate(display.rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                item.setToolTip(str(value))
                self.parameters.setItem(index, column, item)
        self.parameters.resizeColumnsToContents()


OVERFLOW_MODES = ("off", "compare", "steer", "constrain", "hard")


def read_overflow_use(reference, series_root=None):
    """How the referenced phase-model calibration used overflow: (mode, explanation).

    mode is one of OVERFLOW_MODES as the calibration document records it, or
    None when nothing is referenced or recorded. A frozen calibration with
    overflow_in_objective = false is compare-only by its contract. Only the
    explicit reference is read.
    """
    try:
        path = _path(reference, series_root)
        if path is None:
            return None, "No phase-model mass calibration is referenced in series.toml."
        saved = _json(path)
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return None, f"The referenced calibration could not be read: {error}"
    if (saved.get("format") == "darsia-local-overflow-frozen-calibration-v1"
            and saved.get("variant_f") is True):
        return "steer", "The referenced calibration uses reviewed overflow as its free-gas target."
    if saved.get("overflow_in_objective") is False:
        return "compare", (
            "The referenced calibration keeps overflow out of the fit "
            "(overflow_in_objective = false): overflow is compared, not fitted."
        )
    mode = saved.get("overflow_mode") or _mapping(saved.get("calibration")).get("overflow_mode")
    if mode in OVERFLOW_MODES:
        return mode, f"The referenced calibration records overflow_mode = {mode}."
    return None, "The referenced calibration does not record how it used overflow."


class OverflowChoicePage(QWidget):
    """Free gas vs overflow (N141 spec 2.3): the series calibration's overflow use.

    Off and Compare only show what the referenced calibration does; a different
    choice needs a new calibration. Steer per run selects the new refit job.
    Constrain stays with the legacy comparison method, unchanged.
    """

    LABELS = {
        "off": "Off",
        "compare": "Compare only — shown in figures and tables, never fitted",
    }
    mode_changed = Signal(str)
    comparison_requested = Signal()

    def __init__(self, parent=None, *, diagnostic=False):
        super().__init__(parent)
        self.diagnostic = bool(diagnostic)
        self._user_mode = None
        self._recorded_mode = None
        self._explanation = ""
        self._comparison_available = False
        layout = QVBoxLayout(self)
        title = QLabel("Free gas vs overflow")
        font = title.font()
        font.setBold(True)
        title.setFont(font)
        layout.addWidget(title)
        self.buttons = QButtonGroup(self)
        self.buttons.setExclusive(True)
        self.choice = {}
        for mode in ("off", "compare"):
            button = QRadioButton(self.LABELS[mode])
            button.setEnabled(self.diagnostic)
            button.setToolTip(
                "Changes only the saved overflow comparison view; it does not change or fit a calibration."
                if self.diagnostic else
                "Set by the referenced mass calibration; a different choice needs a new calibration.")
            button.clicked.connect(lambda _checked=False, selected=mode: self._choose_diagnostic(selected))
            self.buttons.addButton(button)
            self.choice[mode] = button
            layout.addWidget(button)
        self.steer = QRadioButton("Steer per run — use reviewed overflow in a new calibration")
        self.steer.setEnabled(False)
        self.steer.setToolTip(
            "Select the mass calibration job's reviewed overflow target; its sources are required before starting."
        )
        self.buttons.addButton(self.steer)
        layout.addWidget(self.steer)
        self.detail = QLabel()
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        self.comparison_button = QPushButton("Open saved overflow comparison…")
        self.comparison_button.clicked.connect(lambda _checked=False: self.comparison_requested.emit())
        self.comparison_button.setVisible(self.diagnostic)
        layout.addWidget(self.comparison_button)
        note = QLabel(
            "Steer per run uses reviewed overflow as the free-gas target in a new mass calibration. "
            "Off and Compare only control the saved comparison view."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        self.mode = None
        self._update_diagnostic()

    def reset_diagnostic(self) -> None:
        """A different series starts without a local display override."""
        self._user_mode = None
        self._comparison_available = False

    def set_comparison_available(self, available: bool) -> None:
        """Describe only the already selected saved result, without searching or calculating."""
        self._comparison_available = bool(available)
        self._update_diagnostic()

    def _choose_diagnostic(self, mode) -> None:
        if not self.diagnostic or mode not in self.choice:
            return
        self._user_mode = mode
        self.set_use(self._recorded_mode, self._explanation)
        self.mode_changed.emit(mode)

    def _update_diagnostic(self) -> None:
        if not self.diagnostic:
            return
        if self.mode == "off":
            state = "The saved overflow comparison is hidden."
        elif self._comparison_available:
            state = "Compare only shows the existing saved model free-gas versus measured-overflow figure; it fits nothing."
        else:
            state = (
                "No saved overflow comparison is selected. Open the saved-results browser and choose a result "
                "with an overflow comparison. A new comparison needs measured overflow targets and calculated mass results.")
        self.detail.setText(self._explanation + "\n" + state)
        self.comparison_button.setEnabled(self.mode == "compare")

    def set_use(self, mode, explanation) -> None:
        """Show the calibration's overflow use; nothing here writes a setting."""
        self._recorded_mode, self._explanation = mode, explanation
        self.mode = (self._user_mode or (mode if mode in self.choice else "off")) if self.diagnostic else (
            mode if mode in OVERFLOW_MODES else None)
        self.buttons.setExclusive(False)
        for button in (*self.choice.values(), self.steer):
            button.setChecked(False)
        if self.mode in self.choice:
            self.choice[self.mode].setChecked(True)
        elif self.mode == "steer":
            self.steer.setChecked(True)
        self.buttons.setExclusive(True)
        extra = ""
        if self.mode in ("constrain", "hard"):
            extra = " That is the legacy comparison method's constraint, not one of these choices."
        self.detail.setText(explanation + extra)
        self._update_diagnostic()
