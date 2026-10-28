"""Setup > Protocols > Pressure and temperature > Pressure datasets: every run's pressure, read only.

A pressure dataset index (format ``darsia-pressure-dataset-index-v1``)
lists, per series and run, whether the run's pressure data is
complete, its row count, its recorded gaps, whether a gap is filled
provisionally by default and whether it is adopted, and binds the run's file
(format ``darsia-pressure-dataset-v1``) by path, SHA-256 and size. This
window lists every run of an index and plots the selected run's pressure
against UTC time, with its imaging window, its recorded gaps and the rows that
are not plain observations.

The historical ``darsia-running-gfi-pressure-index-v1`` and
``darsia-running-gfi-pressure-v1`` formats use the same schema and checks.
Their original identifiers remain valid; the supplier never selects a separate
reader, default dataset, unit conversion or clock correction.

A run's file is checked against its size and SHA-256 before it is parsed; a
file that does not match shows as unreadable, and the other runs stay
available. Nothing is fitted, adopted or started, no pressure is overridden
and no file is written. The one stored value is the index chosen last, in the
application's settings.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
from PySide6.QtCore import QSettings, QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QCursor, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

INDEX_FORMAT = "darsia-running-gfi-pressure-index-v1"
SERIES_FORMAT = "darsia-running-gfi-pressure-v1"
DATASET_INDEX_FORMAT = "darsia-pressure-dataset-index-v1"
DATASET_SERIES_FORMAT = "darsia-pressure-dataset-v1"
_INDEX_FORMATS = (DATASET_INDEX_FORMAT, INDEX_FORMAT)
_SERIES_FORMATS = (DATASET_SERIES_FORMAT, SERIES_FORMAT)
#: The quality of a row that is a plain station observation.
OBSERVED = "observed"
#: Organisation and application of the settings that remember the chosen index.
SETTINGS = ("DarSIA", "DarSIA GUI")
INDEX_KEY = "running_pressure/index"
#: The smallest drawn plot, in pixels.
MIN_PLOT_PX = (480, 260)
#: The largest first size of the window, in pixels.
WINDOW_PX = (1280, 860)
LINE_COLOUR = "#1f5f9f"
IMAGING_COLOUR = (0.18, 0.49, 0.2)
GAP_COLOUR = (0.75, 0.22, 0.17)
FLAG_COLOUR = "#e67e22"
FLAG_MARKERS = ("o", "s", "^", "D", "v")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COLUMNS = (
    "Series",
    "Run",
    "Complete",
    "Rows",
    "Gaps",
    "Provisional by default",
    "Adopted",
    "File",
)
_FILE_COLUMN = len(_COLUMNS) - 1


@dataclass(frozen=True)
class PressureGap:
    """One recorded gap: the missing stamps from the first to the last."""

    first_utc: datetime
    last_utc: datetime
    missing_stamp_count: int
    filled_count: int
    resolved_provisionally: bool


@dataclass(frozen=True)
class PressureIndexRow:
    """One run of a running pressure index, as the table shows it."""

    series: str
    run: str
    complete: bool
    row_count: int
    gaps: tuple[PressureGap, ...]
    provisional_gap_default: bool
    adopted: bool
    #: The run's file: ``path``, ``sha256`` and ``size_bytes``.
    atmosphere_series: Mapping[str, Any]

    @property
    def gap_count(self) -> int:
        """The number of recorded gaps."""

        return len(self.gaps)


@dataclass(frozen=True)
class PressureRow:
    """One row of a run's running pressure."""

    utc: datetime
    time_hours: float | None
    pressure_bar: float
    temperature_c: float | None
    station_pressure_hpa: float | None
    quality: str


@dataclass(frozen=True)
class RunningPressureSeries:
    """A run's running pressure file, read after its size and SHA-256 matched."""

    path: Path
    sha256: str
    size_bytes: int
    series: str
    run: str
    complete: bool
    rows: tuple[PressureRow, ...]
    gaps: tuple[PressureGap, ...]
    imaging_first_utc: datetime | None
    imaging_last_utc: datetime | None
    #: Why the file holds no rows, when it says so.
    reason: str


@dataclass(frozen=True)
class PlotElements:
    """What the plot of a run draws."""

    times: tuple[datetime, ...]
    pressures: tuple[float, ...]
    #: Index ranges (start, stop) of the rows the line joins; it stops at a
    #: recorded gap that no row fills.
    segments: tuple[tuple[int, int], ...]
    imaging: tuple[datetime, datetime] | None
    #: Each recorded gap, from its first to its last missing stamp.
    gaps: tuple[tuple[datetime, datetime], ...]
    #: The rows whose quality is not "observed", by quality.
    not_observed: Mapping[str, tuple[tuple[datetime, float], ...]]


def _flag(entry: Mapping[str, Any], key: str, where: str) -> bool:
    """A true/false field of ``entry``; anything else is refused."""

    value = entry.get(key)
    if not isinstance(value, bool):
        raise ValueError(f"{where}: '{key}' must be true or false")
    return value


def _count(entry: Mapping[str, Any], key: str, where: str) -> int:
    """A count field of ``entry``: a whole number, zero or more."""

    value = entry.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: '{key}' must be a whole number, zero or more")
    return value


def _utc(value: Any, where: str) -> datetime:
    """An ISO 8601 time with an explicit offset, in UTC."""

    from darsia.fluidflower.round5_pressure import utc

    if not isinstance(value, str):
        raise ValueError(f"{where}: the time must be an ISO 8601 text with an offset")
    try:
        return utc(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{where}: {value!r} is not a time with an offset ({error})") from None


def _number(row: Mapping[str, Any], key: str, where: str, *, required: bool) -> float | None:
    """A finite number field of ``row``; None when it is absent and not required."""

    value = row.get(key)
    if value is None and not required:
        return None
    number = None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
        except OverflowError:  # A whole number beyond any float.
            number = None
    if number is None or not math.isfinite(number):
        raise ValueError(f"{where}: '{key}' must be a finite number")
    return number


def _gaps(value: Any, where: str) -> tuple[PressureGap, ...]:
    """The recorded gaps of a run, each from its first to its last missing stamp."""

    if not isinstance(value, list):
        raise ValueError(f"{where}: 'gaps' must be a list")
    gaps = []
    for number, gap in enumerate(value, 1):
        at = f"{where}, gap {number}"
        if not isinstance(gap, Mapping):
            raise ValueError(f"{at}: the gap must be an object")
        first, last = _utc(gap.get("first_utc"), at), _utc(gap.get("last_utc"), at)
        if last < first:
            raise ValueError(f"{at}: the last missing stamp is before the first")
        gaps.append(
            PressureGap(
                first,
                last,
                _count(gap, "missing_stamp_count", at),
                _count(gap, "filled_count", at),
                _flag(gap, "resolved_provisionally", at),
            )
        )
    return tuple(gaps)


def read_running_pressure_index(path: Path | str) -> list[PressureIndexRow]:
    """The table rows of a running pressure index: one per run, in index order.

    The file must have the generic pressure dataset index format or its
    historical running-pressure format. Both use the same reader and checks.
    Each run records whether it is complete, its row count, its recorded gaps,
    whether a gap is filled provisionally by default, whether it is adopted,
    and its file (``atmosphere_series``). That file is not read here:
    :func:`read_running_pressure_series` checks it. Raises ValueError when
    the index is not such a file, OSError when it cannot be read.
    """

    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{path} is not a JSON file: {error}") from None
    found = document.get("format") if isinstance(document, Mapping) else None
    if found not in _INDEX_FORMATS:
        raise ValueError(
            f"{path} is not a running pressure index: its format is {found!r}, "
            f"expected one of {_INDEX_FORMATS!r}"
        )
    runs = document.get("runs")
    if not isinstance(runs, Mapping):
        raise ValueError(f"{path}: 'runs' must map each series to its runs")
    rows = []
    for series, entries in runs.items():
        if not isinstance(entries, Mapping):
            raise ValueError(f"{path}: series {series!r} must map each run to its entry")
        for run, entry in entries.items():
            where = f"{path}: {series} {run}"
            if not isinstance(entry, Mapping):
                raise ValueError(f"{where}: the entry must be an object")
            if "run" in entry and entry["run"] != run:
                raise ValueError(f"{where}: the entry names another run, {entry['run']!r}")
            descriptor = entry.get("atmosphere_series")
            if not isinstance(descriptor, Mapping):
                raise ValueError(f"{where}: 'atmosphere_series' must bind the run's file")
            rows.append(
                PressureIndexRow(
                    series=str(series),
                    run=str(run),
                    complete=_flag(entry, "complete", where),
                    row_count=_count(entry, "row_count", where),
                    gaps=_gaps(entry.get("gaps"), where),
                    provisional_gap_default=_flag(entry, "provisional_gap_default", where),
                    adopted=_flag(entry, "adopted", where),
                    atmosphere_series=dict(descriptor),
                )
            )
    return rows


def _verified_bytes(descriptor: Any) -> tuple[Path, str, bytes]:
    """The bytes of the file ``descriptor`` binds, after its size and SHA-256 matched."""

    if not isinstance(descriptor, Mapping):
        raise ValueError("The run's file is not bound by path, SHA-256 and size")
    path_text = descriptor.get("path")
    expected = descriptor.get("sha256")
    size = descriptor.get("size_bytes")
    expected = expected.lower() if isinstance(expected, str) else expected
    if (
        not isinstance(path_text, str)
        or not path_text
        or not isinstance(expected, str)
        or not _SHA256.fullmatch(expected)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
    ):
        raise ValueError("The run's file is not bound by path, SHA-256 and size")
    path = Path(path_text)
    found = path.stat().st_size
    if found != size:
        raise ValueError(f"{path}: the file has {found} bytes, the index records {size}")
    raw = path.read_bytes()
    if len(raw) != size:
        raise ValueError(f"{path}: the file has {len(raw)} bytes, the index records {size}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected:
        raise ValueError(
            f"{path}: the file's SHA-256 {digest} differs from the index's {expected}"
        )
    return path, digest, raw


def _rows(value: Any, where: str) -> tuple[PressureRow, ...]:
    """The rows of a run's file, in strictly increasing UTC time."""

    if not isinstance(value, list):
        raise ValueError(f"{where}: 'rows' must be a list")
    rows = []
    for number, row in enumerate(value, 1):
        at = f"{where}, row {number}"
        if not isinstance(row, Mapping):
            raise ValueError(f"{at}: the row must be an object")
        quality = row.get("quality")
        if not isinstance(quality, str) or not quality:
            raise ValueError(f"{at}: 'quality' must be a text")
        pressure = _number(row, "pressure_bar", at, required=True)
        if pressure <= 0:
            raise ValueError(f"{at}: 'pressure_bar' must be positive")
        stamp = _utc(row.get("utc"), at)
        if rows and stamp <= rows[-1].utc:
            raise ValueError(f"{at}: the rows are not in increasing UTC time")
        rows.append(
            PressureRow(
                utc=stamp,
                time_hours=_number(row, "time_hours", at, required=False),
                pressure_bar=pressure,
                temperature_c=_number(row, "temperature_c", at, required=False),
                station_pressure_hpa=_number(row, "station_pressure_hpa", at, required=False),
                quality=quality,
            )
        )
    return tuple(rows)


def read_running_pressure_series(
    descriptor: Mapping[str, Any],
    *,
    series: str | None = None,
    run: str | None = None,
) -> RunningPressureSeries:
    """A run's running pressure file, checked against ``descriptor`` before it is parsed.

    ``descriptor`` binds the file by ``path``, ``sha256`` and ``size_bytes``
    (an index entry's ``atmosphere_series``). The file's size and SHA-256 are
    compared with it first; a difference raises a ValueError that names both
    values, and the file is not parsed. The file must then have the generic
    pressure dataset format or its historical running-pressure format and,
    when ``series`` or ``run`` is
    given, name that series and run. Its rows must be in strictly increasing
    UTC time, each with a ``quality`` and a finite positive ``pressure_bar``.
    Raises OSError when the file cannot be read.
    """

    path, digest, raw = _verified_bytes(descriptor)
    # Parsed only now that its bytes are the ones the index binds.
    try:
        document = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{path} is not a JSON file: {error}") from None
    found = document.get("format") if isinstance(document, Mapping) else None
    if found not in _SERIES_FORMATS:
        raise ValueError(
            f"{path} is not a running pressure file: its format is {found!r}, "
            f"expected one of {_SERIES_FORMATS!r}"
        )
    named = (str(document.get("series")), str(document.get("run")))
    if (series is not None and named[0] != series) or (run is not None and named[1] != run):
        raise ValueError(
            f"{path} belongs to {named[0]} {named[1]}, not {series or named[0]} {run or named[1]}"
        )
    where = str(path)
    first = document.get("imaging_first_utc")
    last = document.get("imaging_last_utc")
    first = None if first is None else _utc(first, f"{where}: imaging_first_utc")
    last = None if last is None else _utc(last, f"{where}: imaging_last_utc")
    if first is not None and last is not None and last < first:
        raise ValueError(f"{where}: the last imaging time is before the first")
    reason = document.get("reason")
    return RunningPressureSeries(
        path=path,
        sha256=digest,
        size_bytes=len(raw),
        series=named[0],
        run=named[1],
        complete=_flag(document, "complete", where),
        rows=_rows(document.get("rows"), where),
        gaps=_gaps(document.get("gaps", []), where),
        imaging_first_utc=first,
        imaging_last_utc=last,
        reason=reason if isinstance(reason, str) else "",
    )


def plot_elements(series: RunningPressureSeries) -> PlotElements:
    """The drawn parts of a run's plot.

    The pressure against UTC time; the line stops at a recorded gap that lies
    between two rows with no row inside it, so a missing period is never
    drawn as data. The imaging window from the first to the last imaging
    time; each recorded gap from its first to its last missing stamp; and
    the rows whose quality is not "observed", grouped by quality.
    """

    times = tuple(row.utc for row in series.rows)
    breaks = [
        number
        for number in range(1, len(times))
        if any(times[number - 1] < gap.first_utc and gap.last_utc < times[number] for gap in series.gaps)
    ]
    edges = [0, *breaks, len(times)]
    segments = tuple((start, stop) for start, stop in zip(edges, edges[1:]) if stop > start)
    flagged: dict[str, list[tuple[datetime, float]]] = {}
    for row in series.rows:
        if row.quality != OBSERVED:
            flagged.setdefault(row.quality, []).append((row.utc, row.pressure_bar))
    imaging = None
    if series.imaging_first_utc is not None and series.imaging_last_utc is not None:
        imaging = (series.imaging_first_utc, series.imaging_last_utc)
    return PlotElements(
        times=times,
        pressures=tuple(row.pressure_bar for row in series.rows),
        segments=segments,
        imaging=imaging,
        gaps=tuple((gap.first_utc, gap.last_utc) for gap in series.gaps),
        not_observed={quality: tuple(points) for quality, points in flagged.items()},
    )


def imaging_window_pressure(series: RunningPressureSeries) -> tuple[float, float, int] | None:
    """The lowest and highest pressure (bar) of the rows inside the imaging
    window, both ends included, and their number; None without an imaging
    window or without a row inside it."""

    if series.imaging_first_utc is None or series.imaging_last_utc is None:
        return None
    inside = [
        row.pressure_bar
        for row in series.rows
        if series.imaging_first_utc <= row.utc <= series.imaging_last_utc
    ]
    if not inside:
        return None
    return min(inside), max(inside), len(inside)


def _shown_time(value: datetime) -> str:
    """A UTC time to the second, as the window shows it."""

    return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def series_details(series: RunningPressureSeries, index_rows: int | None = None) -> str:
    """The details of a run: its file, SHA-256 and row count, and the lowest
    and highest pressure in its imaging window. ``index_rows`` is the row
    count the index records; a different count is said."""

    flagged = sum(row.quality != OBSERVED for row in series.rows)
    count = f"{len(series.rows)} rows" + (f", {flagged} not observed" if flagged else "")
    if index_rows is not None and index_rows != len(series.rows):
        count += f" (the index records {index_rows})"
    if series.imaging_first_utc is None or series.imaging_last_utc is None:
        window = "No imaging window is recorded."
    else:
        span = (
            f"Imaging window {_shown_time(series.imaging_first_utc)} to "
            f"{_shown_time(series.imaging_last_utc)} UTC"
        )
        values = imaging_window_pressure(series)
        if values is None:
            window = f"{span}: no row inside it."
        else:
            low, high, inside = values
            window = f"{span}: pressure min {low:.5f} bar, max {high:.5f} bar ({inside} rows)."
    if not series.rows and series.reason:
        window += f" No rows: {series.reason}"
    return f"File: {series.path}\nSHA-256: {series.sha256} · {count}\n{window}"


def render_pressure_plot(
    series: RunningPressureSeries,
    *,
    size_px: tuple[int, int] = (1200, 480),
    scale: float = 1.0,
) -> np.ndarray:
    """The plot of a run's pressure against UTC time as RGBA pixels.

    It shows the pressure line, the imaging window, the recorded gaps and
    the rows that are not observed (see :func:`plot_elements`); a run
    without rows shows why instead. ``size_px`` is the size in screen pixels
    (at least MIN_PLOT_PX) and ``scale`` the screen's pixel ratio, so the
    result has ``scale`` times as many pixels each way.
    """

    from matplotlib import dates
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from darsia.fluidflower.storage_saved_rendering import release_stored_figure

    width = max(int(size_px[0]), MIN_PLOT_PX[0])
    height = max(int(size_px[1]), MIN_PLOT_PX[1])
    scale = scale if math.isfinite(scale) and scale > 0 else 1.0
    figure = Figure(figsize=(width / 100, height / 100), dpi=100 * scale, layout="constrained")
    FigureCanvasAgg(figure)
    try:
        axis = figure.subplots()
        title = f"{series.series} {series.run}: pressure"
        elements = plot_elements(series)
        if not elements.times:
            text = "This run's file holds no pressure rows."
            if series.reason:
                text += f"\n{series.reason}"
            axis.text(0.5, 0.5, text, ha="center", va="center", transform=axis.transAxes, wrap=True)
            axis.set_title(title)
            axis.set_axis_off()
        else:

            def naive(value: datetime) -> datetime:
                """The UTC time without its offset, which Matplotlib reads as UTC."""

                return value.astimezone(timezone.utc).replace(tzinfo=None)

            times = [naive(value) for value in elements.times]
            for number, (start, stop) in enumerate(elements.segments):
                axis.plot(
                    times[start:stop],
                    elements.pressures[start:stop],
                    color=LINE_COLOUR,
                    linewidth=1.2,
                    zorder=2,
                    label="Pressure" if number == 0 else None,
                )
            if elements.imaging is not None:
                axis.axvspan(
                    naive(elements.imaging[0]),
                    naive(elements.imaging[1]),
                    facecolor=(*IMAGING_COLOUR, 0.08),
                    edgecolor=(*IMAGING_COLOUR, 0.8),
                    linestyle="--",
                    linewidth=1.0,
                    zorder=0,
                    label="Imaging window",
                )
            for number, (first, last) in enumerate(elements.gaps):
                # The edge keeps a gap of one missing stamp visible.
                axis.axvspan(
                    naive(first),
                    naive(last),
                    facecolor=(*GAP_COLOUR, 0.25),
                    edgecolor=(*GAP_COLOUR, 0.9),
                    linewidth=1.5,
                    zorder=1,
                    label="Recorded gap" if number == 0 else None,
                )
            for number, (quality, points) in enumerate(elements.not_observed.items()):
                axis.scatter(
                    [naive(stamp) for stamp, _value in points],
                    [value for _stamp, value in points],
                    marker=FLAG_MARKERS[number % len(FLAG_MARKERS)],
                    s=36,
                    facecolors="none",
                    edgecolors=FLAG_COLOUR,
                    linewidths=1.4,
                    zorder=3,
                    label=f"Not observed: {quality} ({len(points)})",
                )
            locator = dates.AutoDateLocator()
            axis.xaxis.set_major_locator(locator)
            axis.xaxis.set_major_formatter(dates.ConciseDateFormatter(locator))
            axis.set(xlabel="UTC", ylabel="Pressure (bar)", title=title)
            axis.grid(alpha=0.25)
            axis.legend(loc="best", fontsize=9, framealpha=0.85)
        figure.canvas.draw()
        return np.asarray(figure.canvas.buffer_rgba()).copy()
    finally:
        release_stored_figure(figure)


class PressurePlot(QLabel):
    """The selected run's pressure plot, drawn again at the label's size."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """An empty plot that shows a short message until a run is set."""

        super().__init__(parent)
        self.series: RunningPressureSeries | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # An explicit minimum, so that a drawn plot never sets the window's minimum.
        self.setMinimumSize(*MIN_PLOT_PX)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._redraw = QTimer(self)
        self._redraw.setSingleShot(True)
        self._redraw.setInterval(150)
        self._redraw.timeout.connect(self.redraw)
        self.setText("Choose a run.")

    def show_series(self, series: RunningPressureSeries) -> bool:
        """Plot ``series`` at the label's size."""

        self.series = series
        return self.redraw()

    def show_message(self, text: str) -> None:
        """Show ``text`` instead of a plot."""

        self.series = None
        self._redraw.stop()
        self.clear()
        self.setText(text)

    def redraw(self) -> bool:
        """Draw the current run at the label's size; False without a run."""

        if self.series is None:
            return False
        scale = float(self.devicePixelRatioF())
        pixels = np.ascontiguousarray(
            render_pressure_plot(self.series, size_px=(self.width(), self.height()), scale=scale)
        )
        height, width = pixels.shape[:2]
        image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGBA8888)
        # The QImage only borrows the array: the pixmap gets its own copy.
        pixmap = QPixmap.fromImage(image.copy())
        pixmap.setDevicePixelRatio(scale)
        self.setPixmap(pixmap)
        return True

    def resizeEvent(self, event: Any) -> None:
        """Draw the plot again once the new size has settled."""

        super().resizeEvent(event)
        if self.series is not None:
            self._redraw.start()


def _cells(row: PressureIndexRow) -> tuple[str, ...]:
    """The table cells of one run; the File cell is filled once the run is read."""

    def yes(flag: bool) -> str:
        """'yes' or 'no'."""

        return "yes" if flag else "no"

    return (
        row.series,
        row.run,
        yes(row.complete),
        str(row.row_count),
        str(row.gap_count),
        yes(row.provisional_gap_default),
        yes(row.adopted),
        "",
    )


class RunningPressureDialog(QDialog):
    """Every run's pressure from the selected dataset index, read only."""

    def __init__(self, index_path: Path | str, *, parent: QWidget | None = None) -> None:
        """List the runs of the index at ``index_path`` and plot the first.

        Raises OSError or ValueError when the index cannot be read.
        """

        super().__init__(parent)
        self.index_path: Path | None = None
        self.rows: list[PressureIndexRow] = []
        self.current: int | None = None
        self.series: RunningPressureSeries | None = None
        self.setWindowTitle("Pressure datasets")
        layout = QVBoxLayout(self)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.note.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # A wrapped label is as wide as its longest word, here a file path: an
        # explicit minimum keeps a long path from setting the window's minimum.
        self.note.setMinimumWidth(1)
        layout.addWidget(self.note)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(list(_COLUMNS))
        self.table.horizontalHeaderItem(_FILE_COLUMN).setToolTip(
            "Checked when the run is selected: its file's size and SHA-256 against the index"
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.setMinimumHeight(120)
        self.table.itemSelectionChanged.connect(self._selected)
        splitter.addWidget(self.table)
        self.plot = PressurePlot()
        splitter.addWidget(self.plot)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 520])
        layout.addWidget(splitter, 1)
        self.details = QLabel("Choose a run.")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.details.setMinimumWidth(1)
        layout.addWidget(self.details)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.choose_button = buttons.addButton(
            "Choose dataset…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.choose_button.setToolTip("Show another pressure dataset index (*.json)")
        self.choose_button.clicked.connect(lambda: self.choose_index())
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._fit_screen()
        self.load_index(index_path)

    def _fit_screen(self) -> None:
        """Open at WINDOW_PX or nine tenths of the screen, whichever is smaller."""

        screen = self.screen()
        if screen is None:
            self.resize(*WINDOW_PX)
            return
        area = screen.availableGeometry()
        self.resize(
            min(WINDOW_PX[0], int(area.width() * 0.9)),
            min(WINDOW_PX[1], int(area.height() * 0.9)),
        )

    def load_index(self, index_path: Path | str) -> None:
        """List the runs of the index at ``index_path`` and plot the first.

        Raises OSError or ValueError, before anything changes, when the index
        cannot be read; the shown index then stays.
        """

        rows = read_running_pressure_index(index_path)
        self.index_path, self.rows = Path(index_path), rows
        self.current, self.series = None, None
        with QSignalBlocker(self.table):
            self.table.clearSelection()
            self.table.setRowCount(len(rows))
            for number, row in enumerate(rows):
                for column, text in enumerate(_cells(row)):
                    self.table.setItem(number, column, QTableWidgetItem(text))
        series = list(dict.fromkeys(row.series for row in rows))
        complete = sum(row.complete for row in rows)
        self.note.setText(
            "Read only: the pressure data of every run in the selected dataset, with "
            "whether it is complete, its rows and recorded gaps, whether a gap is filled "
            "provisionally by default and whether it is adopted. A run's file is checked "
            "against its size and SHA-256 in the index before it is read. Nothing is "
            "fitted, adopted or started, and no file is written. "
            f"{len(rows)} runs in {len(series)} series, {complete} complete. "
            f"Index: {self.index_path}"
        )
        self.plot.show_message("Choose a run." if rows else "The index lists no run.")
        self.details.setText("Choose a run." if rows else "")
        if rows:
            self.table.selectRow(0)

    def choose_index(self) -> bool:
        """Ask for another index with a file dialog and list its runs; False
        when none is chosen or it cannot be read (the details line says why)."""

        path = choose_running_pressure_index(self)
        if not path:
            return False
        try:
            self.load_index(path)
        except (OSError, ValueError) as error:
            self.details.setText(f"The index could not be read: {error}")
            return False
        return True

    def file_state(self, number: int) -> str:
        """The File cell of the run in row ``number``: '', 'verified' or 'unreadable'."""

        item = self.table.item(number, _FILE_COLUMN)
        return item.text() if item is not None else ""

    def _set_file_state(self, number: int, text: str, tooltip: str) -> None:
        """Say in the table whether the run's file was read."""

        item = self.table.item(number, _FILE_COLUMN)
        if item is not None:
            item.setText(text)
            item.setToolTip(tooltip)

    def show_run(self, number: int) -> bool:
        """Read the run in row ``number`` and plot it.

        The file is checked against its size and SHA-256 in the index before
        it is parsed. False when it cannot be read: the table, the plot and
        the details line then say that the run is unreadable and why, and
        every other run stays available.
        """

        row = self.rows[number]
        self.current = number
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.WaitCursor))
        try:
            try:
                series = read_running_pressure_series(
                    row.atmosphere_series, series=row.series, run=row.run
                )
            except (OSError, ValueError) as error:
                self.series = None
                self._set_file_state(number, "unreadable", str(error))
                self.plot.show_message(f"{row.series} {row.run}: the file is unreadable.\n{error}")
                self.details.setText(
                    f"File: {row.atmosphere_series.get('path') or '(none)'}\nUnreadable: {error}"
                )
                return False
            self.series = series
            self._set_file_state(number, "verified", "Its size and SHA-256 match the index")
            self.plot.show_series(series)
            self.details.setText(series_details(series, row.row_count))
            return True
        finally:
            QApplication.restoreOverrideCursor()

    def _selected(self) -> None:
        """Show the selected run."""

        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.show_run(rows[0].row())


def choose_running_pressure_index(parent: QWidget | None = None) -> str:
    """Ask for a running pressure index with a file dialog that starts at the
    index chosen last; remember the choice in the settings and return it, or
    '' when the dialog is cancelled."""

    settings = QSettings(*SETTINGS)
    start = str(settings.value(INDEX_KEY, "") or "")
    path, _ = QFileDialog.getOpenFileName(
        parent,
        "Pressure dataset index",
        start,
        "Pressure dataset index (*.json)",
    )
    if not path:
        return ""
    settings.setValue(INDEX_KEY, path)
    settings.sync()
    return path


def open_running_pressure(parent: QWidget | None = None) -> RunningPressureDialog | None:
    """Ask for a running pressure index and open its read-only window; None
    when no index is chosen or it cannot be read (a message says why)."""

    path = choose_running_pressure_index(parent)
    if not path:
        return None
    try:
        dialog = RunningPressureDialog(path, parent=parent)
    except (OSError, ValueError) as error:
        QMessageBox.warning(parent, "Could not read the pressure dataset index", str(error))
        return None
    dialog.show()
    return dialog


class ProtocolsRunningPressure:
    """The window's read-only pressure dataset view.

    It opens from the "Show pressure datasets…" button on Setup > Protocols >
    Pressure and temperature (native_protocol_sections).
    """

    def __init__(self, window: Any) -> None:
        self.window = window
        self.dialog: RunningPressureDialog | None = None

    def open(self) -> RunningPressureDialog | None:
        """Ask for the index and open the read-only window; None when none opens."""

        dialog = open_running_pressure(self.window)
        if dialog is not None:
            self.dialog = dialog
        return dialog


__all__ = [
    "DATASET_INDEX_FORMAT",
    "DATASET_SERIES_FORMAT",
    "INDEX_FORMAT",
    "SERIES_FORMAT",
    "PlotElements",
    "PressureGap",
    "PressureIndexRow",
    "PressurePlot",
    "PressureRow",
    "ProtocolsRunningPressure",
    "RunningPressureDialog",
    "RunningPressureSeries",
    "choose_running_pressure_index",
    "imaging_window_pressure",
    "open_running_pressure",
    "plot_elements",
    "read_running_pressure_index",
    "read_running_pressure_series",
    "render_pressure_plot",
    "series_details",
]
