"""Pressure and temperature source of each run of a series, read only.

Two routes bring pressure and temperature into a mass computation:

- The Direct colour method reads the run configuration's ``[protocols]``
  ``pressure_temperature`` file with DarSIA's ``PressureTemperatureProtocol``.
  With ``pressure_temperature_mode = "constant"`` DarSIA's protocol setup
  writes that file from ``pressure_bar`` and ``temperature_celsius``; with
  ``"Load from CSV"`` the file is written by hand. The method reads the file
  in both modes.
- Phase-map method jobs read no run configuration for this. Their mass uses
  the pressure and temperature bound in the adopted execution plan's run data:
  an ``atmosphere_series`` table interpolated in time, or one value per image.

The functions here read a run configuration and its protocol file with
DarSIA's own readers and return plain text rows. They never write, never run
protocol setup and never open an execution plan. When a value cannot be read,
the row says so instead of assuming a default.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

NOT_RECORDED = "Not recorded"
READING = "Reading…"
#: A protocol file larger than this is not summarised on a page.
MAX_PROTOCOL_BYTES = 32 * 1024 * 1024
#: Calibration documents are read only up to this size, as on the saved-values page.
MAX_CALIBRATION_BYTES = 1024 * 1024
#: Key words that mark a pressure or temperature field in a calibration document.
PRESSURE_WORDS = ("pressure", "atmosphere", "temperature")

DIRECT_COLOUR_NOTE = (
    "Direct colour method: reads each run's [protocols] pressure_temperature file, "
    "as listed here."
)
PHASE_MAP_NOTE = (
    "Phase-map method: jobs do not read these files. Their mass uses the pressure and "
    "temperature bound in the adopted execution plan's run data (an atmosphere table "
    "over time, or one value per image); this page does not open the plan."
)


class PressureSourceRow(NamedTuple):
    """One run's pressure and temperature source, as display text.

    ``setup`` is the configured setup constant when protocol setup is in
    "constant" mode and the file holds other values, else empty.
    """

    run: str
    source: str
    values: str
    span: str
    status: str
    fit: str
    detail: str
    setup: str = ""


class PressureSources(NamedTuple):
    """The rows of every run of a series and the notes shown with them.

    ``complete`` is False while later rows are still being read.
    """

    rows: tuple[PressureSourceRow, ...]
    notes: tuple[str, ...]
    complete: bool = True


class ProtocolSummary(NamedTuple):
    """What DarSIA's pressure-temperature protocol reader found in one file."""

    rows: int
    first: str
    last: str
    ordered: bool
    pressure: tuple[float, float]
    temperature: tuple[float, float]
    pressure_gradient: tuple[float, float]
    temperature_gradient: tuple[float, float]


class FitPressure(NamedTuple):
    """The pressure a referenced calibration records for its fit, as text."""

    text: str
    detail: str


def _number(value: float) -> str:
    return f"{value:.6g}"


def _range(values: tuple[float, float], unit: str = "") -> str:
    low, high = values
    text = _number(low) if low == high else f"{_number(low)}–{_number(high)}"
    return f"{text} {unit}".rstrip()


def _bounds(series) -> tuple[float, float]:
    return float(series.min()), float(series.max())


def _all_equal(values: tuple[float, float], constant: float) -> bool:
    # The file is parsed by pandas, the constant by Python; allow parser ulps only.
    return all(math.isclose(value, constant, rel_tol=1e-12, abs_tol=0.0) for value in values)


def summarise_protocol(protocol: Path | tuple[Path, str]) -> ProtocolSummary:
    """Summarise one pressure-temperature protocol as DarSIA reads it.

    ``protocol`` is a file, or a ``(workbook, sheet)`` pair, exactly as the
    run configuration gives it. The file is parsed by DarSIA's own
    ``PressureTemperatureProtocol``, so the required columns, number parsing
    and errors are the method's own. Raises OSError, ValueError or
    AssertionError when the method could not read the file.
    """
    from darsia.experiment.protocols import PressureTemperatureProtocol

    path = Path(protocol[0] if isinstance(protocol, (list, tuple)) else protocol)
    size = path.stat().st_size
    if size > MAX_PROTOCOL_BYTES:
        raise ValueError(
            f"The file has {size} bytes, more than this page summarises "
            f"({MAX_PROTOCOL_BYTES} bytes)."
        )
    frame = PressureTemperatureProtocol(protocol).df
    if frame.empty:
        empty = (math.nan, math.nan)
        return ProtocolSummary(0, "", "", True, empty, empty, empty, empty)
    times = frame["datetime"]
    return ProtocolSummary(
        rows=len(frame),
        first=str(times.iloc[0]),
        last=str(times.iloc[-1]),
        ordered=bool(times.is_monotonic_increasing),
        pressure=_bounds(frame["pressure_bar"]),
        temperature=_bounds(frame["temperature_celsius"]),
        pressure_gradient=_bounds(frame["pressure_gradient_bar"]),
        temperature_gradient=_bounds(frame["temperature_gradient_celsius"]),
    )


def read_pressure_settings(config: str | Path):
    """The ``[protocols]`` pressure and temperature settings of one run file.

    Uses DarSIA's own section reader (with this machine's storage path
    aliases) and ``PressureTemperatureProtocolConfig``, so the protocol path,
    the mode and the constants mean what they mean to the method. A file
    without a ``[protocols]`` table gives the method's defaults and no file.
    """
    from darsia.presets.workflows.config.protocols import (
        PressureTemperatureProtocolConfig,
    )
    from darsia.presets.workflows.config.utils import _get_section_from_toml

    path = Path(config)
    try:
        section = _get_section_from_toml(path, "protocols")
    except KeyError:
        section = {}
    if not isinstance(section, Mapping):
        raise ValueError(f"[protocols] in {path} is not a table.")
    settings = PressureTemperatureProtocolConfig()
    settings.load_pressure_temperature(dict(section), path)
    return settings


def _file_label(protocol) -> str:
    path = Path(protocol[0] if isinstance(protocol, (list, tuple)) else protocol)
    label = f"{path.parent.name}/{path.name}" if path.parent.name else path.name
    if isinstance(protocol, (list, tuple)):
        label += f" (sheet {protocol[1]})"
    return label


def _values_text(summary: ProtocolSummary) -> str:
    gradients = summary.pressure_gradient + summary.temperature_gradient
    gradient = (
        "gradients 0" if all(value == 0 for value in gradients) else "gradients not 0"
    )
    return (
        f"{_range(summary.pressure, 'bar')} · {_range(summary.temperature, '°C')} · "
        f"{gradient}"
    )


def run_pressure_source(
    run: str,
    config: str | Path | None,
    *,
    note: str = "",
    fit: FitPressure = FitPressure(NOT_RECORDED, ""),
) -> PressureSourceRow:
    """The pressure and temperature source of one run, as one display row.

    ``config`` is the file DarSIA's run selector opens for the run, or None
    with ``note`` saying why there is none. The row names the protocol file
    with its row count, time span and value range, the constant that protocol
    setup would write, or that the source is missing. ``fit`` is the series
    calibration's own record, the same for every run.
    """
    run = str(run)

    def row(source, values, span, status, *lines, setup=""):
        detail = "\n".join(line for line in (*lines, fit.detail) if line)
        return PressureSourceRow(run, source, values, span, status, fit.text, detail, setup)

    if config is None:
        return row("No run configuration", "", "", NOT_RECORDED,
                   note or "No single run configuration file is registered.")
    where = f"Run configuration: {config}"
    try:
        settings = read_pressure_settings(config)
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return row(NOT_RECORDED, "", "", "Not readable", where, str(error))
    mode = settings.pressure_temperature_mode
    constant = (float(settings.pressure_bar), float(settings.temperature_celsius))
    constant_text = f"{_number(constant[0])} bar · {_number(constant[1])} °C"
    mode_line = (
        f"pressure_temperature_mode = constant ({constant_text}): DarSIA's protocol "
        "setup writes the file from these values."
        if mode == "constant"
        else "pressure_temperature_mode = Load from CSV: the file is written by hand."
    )
    protocol = settings.pressure_temperature
    if protocol is None:
        return row("Missing", "", "", "No file configured", where,
                   "[protocols] pressure_temperature is not set; the Direct colour "
                   "method needs this file.", mode_line)
    path = Path(protocol[0] if isinstance(protocol, (list, tuple)) else protocol)
    file_line = f"Protocol file: {path}"
    try:
        present = path.is_file()
    except OSError as error:
        return row(_file_label(protocol), "", "", "Not readable", where, file_line,
                   str(error))
    if not present:
        if mode == "constant":
            return row("Constant (file not written)", constant_text, "",
                       "Not written yet", where, file_line, mode_line,
                       "Until protocol setup writes it, the method has no file to read.")
        return row(_file_label(protocol), "", "", "File missing", where, file_line,
                   mode_line)
    try:
        summary = summarise_protocol(protocol)
    except (OSError, ValueError, TypeError, AssertionError, UnicodeError) as error:
        return row(_file_label(protocol), "", "", "Not readable", where, file_line,
                   mode_line, str(error) or type(error).__name__)
    if summary.rows == 0:
        return row(_file_label(protocol), "", "0 rows", "No rows", where, file_line,
                   mode_line)
    span = f"{summary.rows} rows · {summary.first} to {summary.last}"
    lines = [
        where,
        file_line,
        mode_line,
        f"pressure_gradient_bar {_range(summary.pressure_gradient)} · "
        f"temperature_gradient_celsius {_range(summary.temperature_gradient)}",
    ]
    status, setup = "Ready", ""
    if not summary.ordered:
        status = "Rows not in time order"
        lines.append("DarSIA interpolates between neighbouring rows in file order.")
    if mode == "constant" and not (
        _all_equal(summary.pressure, constant[0])
        and _all_equal(summary.temperature, constant[1])
    ):
        setup = constant_text
        lines.append(
            "The file holds other values than the setup constant. The method reads "
            "the file; protocol setup replaces it only when told to overwrite."
        )
    return row(_file_label(protocol), _values_text(summary), span, status, *lines,
               setup=setup)


def _reference_path(reference, series_root) -> Path | None:
    value = reference.get("path") if isinstance(reference, Mapping) else reference
    if value is None or not str(value).strip():
        return None
    path = Path(str(value))
    if not path.is_absolute():
        if series_root is None:
            raise ValueError("A relative calibration reference needs the series folder.")
        path = Path(series_root) / path
    return path


def _bounded_json(path: Path) -> dict:
    with path.open("rb") as stream:
        raw = stream.read(MAX_CALIBRATION_BYTES + 1)
    if len(raw) > MAX_CALIBRATION_BYTES:
        raise ValueError(f"The calibration exceeds {MAX_CALIBRATION_BYTES} bytes: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a saved JSON object: {path}")
    return value


def recorded_pressure_field(document, prefix: str = "", depth: int = 0):
    """The first ``(key path, value)`` whose key names pressure, atmosphere or temperature.

    Only keys are matched, never text values (a path may contain those words).
    Returns None when the document has no such key. The search is bounded in
    depth and list length.
    """
    if depth > 12:
        return None
    if isinstance(document, Mapping):
        for key, value in document.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(word in str(key).casefold() for word in PRESSURE_WORDS):
                return path, value
            found = recorded_pressure_field(value, path, depth + 1)
            if found is not None:
                return found
    elif isinstance(document, list):
        for index, value in enumerate(document[:1024]):
            found = recorded_pressure_field(value, f"{prefix}[{index}]", depth + 1)
            if found is not None:
                return found
    return None


def calibration_pressure(method: str, reference, series_root=None) -> FitPressure:
    """What the series' referenced calibration records about its fit's pressure.

    For the Phase-map method ``reference`` is the calibration JSON, for the
    Direct colour method the calibration folder (its ``flash/flash.json`` is
    read). A field whose key names pressure, atmosphere or temperature is
    reported with its key and, for a single value, the value itself; without
    one the answer is "Not recorded". No other file is followed.
    """
    if method not in {"phase_model", "colour_path"}:
        return FitPressure("No method chosen", "The series has no chosen method.")
    try:
        path = _reference_path(reference, series_root)
        if path is None:
            return FitPressure("No calibration referenced",
                               "series.toml references no calibration for this method.")
        document_path = path if method == "phase_model" else path / "flash" / "flash.json"
        if method == "colour_path" and path.is_dir() and not document_path.exists():
            return FitPressure(NOT_RECORDED,
                               f"In the fit: {path.name} records no pressure or temperature.")
        document = _bounded_json(document_path)
    except (OSError, ValueError, TypeError, UnicodeError) as error:
        return FitPressure("Calibration not readable", f"In the fit: {error}")
    found = recorded_pressure_field(document)
    if found is None:
        return FitPressure(
            NOT_RECORDED,
            f"In the fit: {document_path.name} records no pressure or temperature "
            "used by its fit.",
        )
    key, value = found
    if isinstance(value, (int, float, str)) and not isinstance(value, bool):
        text = f"{key} = {value}"
    else:
        text = f"Recorded under {key}"
    return FitPressure(text, f"In the fit: {document_path.name} records {key}.")


def run_configuration_choices(series_root) -> list[tuple[str, Path | None, str]]:
    """``(run, file, note)`` for every run: the file DarSIA's run selector opens.

    This is the native window's own ``run_configurations``, so these pages
    read exactly the file that the run selector and the Direct colour method
    use.
    """
    from darsia.fluidflower.native_darsia_app import run_configurations

    return run_configurations(series_root)


def _setup_note(rows: Sequence[PressureSourceRow]) -> str:
    count = sum(bool(row.setup) for row in rows)
    if not count:
        return ""
    return (
        f"{count} of {len(rows)} runs: pressure_temperature_mode is 'constant' and the file "
        "holds other values than that constant. The method reads the file; DarSIA's "
        "protocol setup replaces the file with the constant only when told to overwrite."
    )


def iter_series_pressure_sources(
    series_root,
    *,
    method: str = "",
    calibration=None,
    choices: Sequence[tuple[str, Path | None, str]] | None = None,
) -> Iterator[PressureSources]:
    """Read the series run by run, yielding the table after each step.

    The first snapshot lists every run as "Reading…"; each later one adds one
    read run, and the last one is complete. A page can show each snapshot as
    it comes, so one slow network file does not hold the window. ``choices``
    are the run selector's ``(run, file, note)`` rows; without them they are
    read for ``series_root``. ``method`` and ``calibration`` name the series'
    referenced calibration, whose own pressure record fills "In the fit".
    """
    root = Path(series_root)
    choices = list(run_configuration_choices(root) if choices is None else choices)
    fit = calibration_pressure(method, calibration, root)
    notes = [DIRECT_COLOUR_NOTE, PHASE_MAP_NOTE, fit.detail]
    if not choices:
        notes.insert(0, "No runs are registered in series.toml.")
    rows = [
        PressureSourceRow(str(run), "", "", "", READING, fit.text, "")
        for run, _path, _note in choices
    ]
    yield PressureSources(tuple(rows), tuple(notes), complete=not choices)
    for index, (run, path, note) in enumerate(choices):
        rows[index] = run_pressure_source(run, path, note=note, fit=fit)
        done = index == len(choices) - 1
        final = [*notes, _setup_note(rows)] if done else notes
        yield PressureSources(tuple(rows), tuple(line for line in final if line), done)


def series_pressure_sources(
    series_root,
    *,
    method: str = "",
    calibration=None,
    choices: Sequence[tuple[str, Path | None, str]] | None = None,
) -> PressureSources:
    """One row per run of the series, in ``series.toml`` order, and the notes.

    The complete last snapshot of :func:`iter_series_pressure_sources`.
    """
    last = None
    for last in iter_series_pressure_sources(
        series_root, method=method, calibration=calibration, choices=choices
    ):
        pass
    return last


def pressure_line(row: PressureSourceRow | None, run=None) -> str:
    """One line for a run page: the run's configured source and its status."""
    if row is None and not run:
        return "Pressure and temperature: choose a run."
    label = str(row.run if row is not None else run).upper()
    if row is None:
        return f"Pressure and temperature for {label}: {NOT_RECORDED} (not a run of this series)."
    parts = " · ".join(part for part in (row.source, row.values, row.span) if part)
    return (
        f"Pressure and temperature for {label}: {parts} — {row.status}. "
        f"In the fit: {row.fit}. Source: the run configuration, read by the Direct "
        "colour method; Phase-map jobs use the execution plan's run data."
    )
