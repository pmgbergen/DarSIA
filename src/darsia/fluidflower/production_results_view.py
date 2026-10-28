"""Analysis > Production results: a finished production job, read-only, and its films.

The page opens one production job folder written by the job layer
(darsia-alias-job-v1). It lists every run of the job with its frame count and
state. For a finished run it shows each frame's stored field on a time slider
(phase labels, concentration, gas saturation, aqueous mass or free-gas mass;
only the fields the run stored, from its frame products.npz or its stored
coarse layer) and the run's mass curves from its mass table.

Everything is read through the existing readers:

* each finished run through :class:`FinishedJob` (published_dataset), which
  checks the run's records and small products against their SHA-256;
* a frame's ``products.npz`` through :func:`storage_profile.read_products`
  with the analysis data reader, which verifies the archive's SHA-256 and
  rebuilds a lean frame's omitted total mass;
* a stored coarse layer as the verified bytes FinishedJob returns.

Opening a job starts no calculation and writes nothing. "Export film…" renders
the selected run and field into a new folder inside a folder the user chooses,
never inside the job folder or the run's product folders, and encodes an MP4
with the existing movie writer at the chosen frames per second. Without an
MP4 encoder it leaves the numbered PNG frames and says so.
"""

from __future__ import annotations

from collections import OrderedDict
import contextlib
import csv
from dataclasses import dataclass
import functools
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import threading

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QDoubleValidator, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressDialog,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from darsia.fluidflower.phase_rendering import AQUEOUS_RGBA, GAS_RGBA
from darsia.fluidflower.published_dataset.job_reader import LAYER_RECEIPT, FinishedJob, JobError
from darsia.fluidflower.storage_resolution import LABEL_LEGEND

FILM_FORMAT = "darsia-production-film-v1"
#: The longest side of a displayed map; larger stored grids show every n-th cell.
MAX_DISPLAY_PIXELS = 1280
#: The size of a rendered map: the view and every film frame.
MAP_SIZE_PX = (1280, 720)
CURVE_SIZE_PX = (1280, 400)
DEFAULT_FPS = 3.0
#: Frame result records kept in memory (a production record is about half a megabyte).
RESULT_CACHE_SIZE = 4
COMPLETE = "Complete"
#: Background of cells without a value (outside the domain or excluded).
BACKGROUND = "#e6edf2"


@dataclass(frozen=True)
class FieldSpec:
    """One field the page can show: its label and its array in each product."""

    label: str
    products_array: str
    layer_array: str | None


FIELDS = {
    "phase_labels": FieldSpec("Phase labels", "phase_labels", None),
    "concentration": FieldSpec("Concentration", "concentration_aqueous", "concentration_aqueous"),
    "saturation": FieldSpec("Gas saturation", "saturation_free_gas", "saturation_free_gas"),
    "aqueous_mass": FieldSpec("Aqueous mass", "mass_aqueous", "cell_mass_aqueous_g"),
    "gas_mass": FieldSpec("Free-gas mass", "mass_free_gas", "cell_mass_gas_g"),
}
SOURCES = {"products": "Frame products", "layer": "Stored coarse layer"}
_DEFAULT_UNITS = {"concentration": "g/L", "saturation": "fraction",
                  "aqueous_mass": "g/L", "gas_mass": "g/L"}
_LAYER_UNITS = {"concentration": "g/L", "saturation": "fraction",
                "aqueous_mass": "g per cell", "gas_mass": "g per cell"}
#: Fields with a physical upper bound start at that bound; others fit the frame.
_PHYSICAL_LIMITS = {"saturation": 1.0}
_OTHER_LABEL = ("Other label", "#555b61")
#: Display names and colours of the stored phase labels; dissolved and gas keep
#: the established two-phase display colours.
_PHASE_DISPLAY = {"water": ("Water", "#a9cfee"),
                  "dissolved": ("Dissolved CO2", "#%02x%02x%02x" % tuple(AQUEOUS_RGBA[:3])),
                  "gas": ("Free gas", "#%02x%02x%02x" % tuple(GAS_RGBA[:3])),
                  "excluded": ("Excluded", BACKGROUND)}
#: (code, legend key, display name, colour) of each phase label code in LABEL_LEGEND.
_PHASES = tuple((int(code), key, *_PHASE_DISPLAY.get(key, (key.capitalize(), _OTHER_LABEL[1])))
                for code, key in LABEL_LEGEND.items())
_CURVES = (("total_mass_g", "Total", "#175aad", "-"), ("aqueous_mass_g", "Dissolved", "#169879", "-"),
           ("gas_mass_g", "Free gas", "#dc398b", "-"), ("injected_mass_g", "Injected", "black", "--"))


@dataclass(frozen=True)
class RunState:
    """One run of a job: its state, its frame count when known, and a note."""

    run: str
    state: str
    frames: int | None
    note: str = ""


@dataclass
class FieldImage:
    """A stored field prepared for display: every ``stride``-th cell of its grid."""

    values: np.ndarray
    mask: np.ndarray | None
    categorical: bool
    unit: str
    origin: str
    extent: tuple | None
    stride: int
    shape: tuple


@dataclass(frozen=True)
class FilmResult:
    """What a film export wrote: the folder, the MP4 or PNG frames, and a message."""

    folder: Path
    movie: Path | None
    frames_folder: Path | None
    frame_count: int
    encoder: str | None
    message: str


class _FinishedRun(FinishedJob):
    """FinishedJob over one run of a job whose other runs may not be finished.

    FinishedJob refuses a job with any unfinished run. Its own reading and
    checks apply here unchanged, to one run: the job record it reads names only
    that run, so one unreadable or unfinished run never hides the others.
    """

    def __init__(self, root, run: str):
        """Read and check one finished run of the job in ``root``."""
        self._only_run = str(run)
        super().__init__(root)

    @property
    def job(self) -> dict:
        """The job record, naming only this reader's run as expected."""
        return self._job_record

    @job.setter
    def job(self, value) -> None:
        """Keep the job record with its expected runs narrowed to this reader's run."""
        if isinstance(value, dict):
            runs = [item for item in value.get("expected_runs") or () if str(item) == self._only_run]
            value = dict(value, expected_runs=runs)
        self._job_record = value


def _marker_state(root: Path, key: str) -> str:
    """The job-layer state of a run without a done record, from its state files."""
    if (root / "failed" / f"{key}.json").is_file():
        return "Failed"
    if any((root / "claimed").glob(f"*/{key}.json")):
        return "Claimed"
    if (root / "publishing" / key).exists():
        return "Publishing"
    if (root / "queue" / f"{key}.json").is_file():
        return "Queued"
    return "Not started"


def _task_frames(root: Path, key: str) -> int | None:
    """The number of frames a run's task selects, or None when it cannot be read."""
    from darsia.fluidflower.pipeline.job_queue import read_json

    try:
        cases = read_json(root / "results" / key / "task.json").get("ordered_case_ids")
    except (OSError, ValueError):
        return None
    return len(cases) if isinstance(cases, list) else None


def display_stride(shape, limit: int | None = None) -> int:
    """The step that keeps a grid's longest displayed side at or below ``limit`` cells.

    The default limit is :data:`MAX_DISPLAY_PIXELS`.
    """
    limit = MAX_DISPLAY_PIXELS if limit is None else int(limit)
    return max(1, math.ceil(max(int(size) for size in shape) / limit))


def nice_limit(value) -> float:
    """A colour-scale maximum: ``value`` rounded up to two significant digits (1.0 if none)."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(value) or value <= 0:
        return 1.0
    step = 10.0 ** (math.floor(math.log10(value)) - 1)
    # The tolerance keeps a float32 value such as 1.2000000477 at 1.2.
    return float(f"{math.ceil(value / step * (1 - 1e-6)) * step:.3g}")


def fit_limit(image: FieldImage) -> float:
    """A colour-scale maximum for the shown values of a continuous field image."""
    values = np.asarray(image.values, dtype=np.float64)
    shown = np.isfinite(values)
    if image.mask is not None:
        shown &= ~image.mask
    return nice_limit(values[shown].max()) if shown.any() else 1.0


def inside_folder(path, root) -> bool:
    """True when ``path`` is ``root`` or lies inside it (links resolved)."""
    try:
        return Path(path).resolve().is_relative_to(Path(root).resolve())
    except (OSError, ValueError):
        return False


def fresh_folder(parent, name: str) -> Path:
    """The first of ``name``, ``name-2``, ``name-3``… that does not exist in ``parent``."""
    parent = Path(parent)
    candidate, number = parent / name, 1
    while candidate.exists():
        number += 1
        candidate = parent / f"{name}-{number}"
    return candidate


def format_hours(elapsed_h) -> str:
    """An elapsed time as text in hours, or 'time not recorded'."""
    try:
        value = float(elapsed_h)
    except (TypeError, ValueError):
        return "time not recorded"
    return f"{value:.2f} h" if math.isfinite(value) else "time not recorded"


def frame_title(run: str, field: str, elapsed_h) -> str:
    """The English title of a map: run, field and elapsed time."""
    return f"{run} · {FIELDS[field].label} · {format_hours(elapsed_h)}"


def frame_footer(number: int, count: int, case: str, image: FieldImage | None) -> str:
    """The second title line of a map: frame number, case id and the display step."""
    text = f"Frame {number} of {count} ({case})"
    if image is None or image.stride == 1:
        return text
    stride = image.stride
    suffix = "th" if 10 <= stride % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(stride % 10, "th")
    return f"{text} · every {stride}{suffix} cell shown"


def label_colours(codes: np.ndarray) -> tuple[np.ndarray, list[tuple[str, str]]]:
    """An RGB image of phase label codes and the (name, colour) legend of the codes present."""
    from matplotlib.colors import to_rgb

    codes = np.asarray(codes)
    rgb = np.empty(codes.shape + (3,), dtype=np.uint8)
    rgb[...] = np.round(np.asarray(to_rgb(_OTHER_LABEL[1])) * 255)
    legend, known = [], np.zeros(codes.shape, dtype=bool)
    for code, _key, name, colour in _PHASES:
        where = codes == code
        known |= where
        rgb[where] = np.round(np.asarray(to_rgb(colour)) * 255)
        if where.any():
            legend.append((name, colour))
    if not known.all():
        legend.append(_OTHER_LABEL)
    return rgb, legend


def _phase_codes_from_fractions(arrays: dict) -> np.ndarray | None:
    """Each layer cell's phase code with the largest area share; no area is 'excluded'."""
    from darsia.fluidflower.published_dataset.comparison import label_fractions

    fractions = label_fractions(arrays)
    names = [(code, key) for code, key, _name, _colour in _PHASES if key in fractions]
    if not names:
        return None
    stack = np.stack([np.nan_to_num(fractions[key], nan=-1.0) for _code, key in names])
    codes = np.asarray([code for code, _key in names], dtype=np.uint8)[np.argmax(stack, axis=0)]
    codes[np.all(stack < 0, axis=0)] = 255
    return codes


def _figure(size_px):
    """A constrained Agg figure of ``size_px`` pixels at 100 dpi."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(size_px[0] / 100, size_px[1] / 100), dpi=100, layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def _pixels(figure) -> np.ndarray:
    """Draw a figure and return its RGBA pixels; its buffers are released."""
    from darsia.fluidflower.storage_saved_rendering import release_stored_figure

    try:
        figure.canvas.draw()
        return np.asarray(figure.canvas.buffer_rgba()).copy()
    finally:
        release_stored_figure(figure)


# Matplotlib's text and font caches are shared by every figure, and the page
# draws frames off the GUI thread: one figure is drawn at a time.
_RENDER_LOCK = threading.RLock()


def _one_render_at_a_time(render):
    """Draw ``render``'s figure while holding the module's render lock."""

    @functools.wraps(render)
    def locked(*args, **kwargs):
        with _RENDER_LOCK:
            return render(*args, **kwargs)

    return locked


@_one_render_at_a_time
def render_map(image: FieldImage | None, *, title: str, footer: str = "", scale_max: float | None = None,
               label: str = "", message: str = "", size_px=MAP_SIZE_PX) -> np.ndarray:
    """Render one field image as RGBA pixels: the page's view and a film frame alike.

    Continuous fields use a fixed linear scale from 0 to ``scale_max``; phase
    labels use fixed colours with a legend. Cells without a value show the
    background colour. ``message`` replaces the map when there is no image.
    """
    from matplotlib.patches import Patch

    figure = _figure(size_px)
    axis = figure.subplots()
    axis.set_title(title + (f"\n{footer}" if footer else ""), fontsize=13)
    if image is None:
        axis.text(0.5, 0.5, message or "No stored field for this frame.", ha="center", va="center",
                  transform=axis.transAxes, wrap=True, fontsize=13)
        axis.set_axis_off()
        return _pixels(figure)
    axis.set_facecolor(BACKGROUND)
    options = dict(origin=image.origin, extent=image.extent, interpolation="nearest")
    if image.categorical:
        rgb, legend = label_colours(image.values)
        axis.imshow(rgb, **options)
        handles = [Patch(facecolor=colour, edgecolor="#667085", label=name) for name, colour in legend]
        figure.legend(handles=handles, loc="outside right upper", frameon=False, title=label or None)
    else:
        from matplotlib import colormaps

        limit = float(scale_max) if scale_max and math.isfinite(scale_max) and scale_max > 0 else 1.0
        values = np.ma.masked_invalid(np.asarray(image.values, dtype=np.float64))
        if image.mask is not None:
            values = np.ma.masked_where(image.mask, values)
        # Cells without a value show the background, as excluded phase labels do.
        colours = colormaps["viridis"].with_extremes(bad=BACKGROUND)
        mapped = axis.imshow(values, cmap=colours, vmin=0.0, vmax=limit, **options)
        unit = f" ({image.unit})" if image.unit else ""
        figure.colorbar(mapped, ax=axis, fraction=0.035, label=f"{label}{unit} · fixed scale 0 to {limit:g}")
    if image.extent is None:
        axis.set_axis_off()
    else:
        axis.set(xlabel="x (m)", ylabel="y (m)", aspect="equal")
    return _pixels(figure)


def curve_series(rows: list[dict]) -> tuple[str, list, dict]:
    """The x axis label, x values and the mass series of a run's mass table rows."""
    times = [row.get("elapsed_h") for row in rows]
    if rows and all(isinstance(value, float) and math.isfinite(value) for value in times):
        axis, xs = "Elapsed time (h)", times
    else:
        axis, xs = "Frame", list(range(1, len(rows) + 1))
    series = {}
    for key, *_style in _CURVES:
        values = [row.get(key) for row in rows]
        if any(isinstance(value, float) and math.isfinite(value) for value in values):
            series[key] = [value if isinstance(value, float) else math.nan for value in values]
    return axis, xs, series


def curve_marker(rows: list[dict], case: str):
    """Where frame ``case`` lies on the mass curves' x axis, or None."""
    _axis, xs, _series = curve_series(rows)
    marker = None
    for row, x in zip(rows, xs):
        if row.get("case_id") == case:
            marker = float(x)
    return marker


def mass_table_hours(rows: list[dict]) -> dict:
    """Each frame's elapsed hours in a run's mass table rows: the first finite value per frame."""
    hours = {}
    for row in rows:
        value = row.get("elapsed_h")
        if isinstance(value, float) and math.isfinite(value):
            hours.setdefault(row.get("case_id"), value)
    return hours


@_one_render_at_a_time
def render_mass_curves(rows: list[dict], *, run: str, marker=None, size_px=CURVE_SIZE_PX) -> np.ndarray:
    """Render a run's total, dissolved and free-gas CO2 mass from its mass table as RGBA pixels."""
    figure = _figure(size_px)
    axis = figure.subplots()
    label, xs, series = curve_series(rows)
    if not series:
        axis.text(0.5, 0.5, "This run has no mass table with CO2 masses.", ha="center", va="center",
                  transform=axis.transAxes)
        axis.set_axis_off()
        return _pixels(figure)
    for key, name, colour, style in _CURVES:
        if key in series:
            axis.plot(xs, series[key], style, color=colour, label=name, linewidth=1.6)
    if marker is not None and math.isfinite(marker):
        axis.axvline(marker, color="#667085", linewidth=1.0, linestyle=":", label="Shown frame")
    axis.set(xlabel=label, ylabel="CO2 mass (g)", title=f"{run} · CO2 mass from the run's mass table")
    axis.grid(alpha=0.25)
    figure.legend(loc="outside right upper", fontsize=9, frameon=False)
    return _pixels(figure)


def _number(text):
    """A finite float from a mass-table cell, 'True'/'False' as text, or the text itself."""
    try:
        value = float(text)
    except (TypeError, ValueError):
        return text
    return value if math.isfinite(value) else math.nan


def read_job_record(root) -> dict:
    """The record (job.json) of a production job written by the job layer.

    Raises JobError for a folder without one, another kind of job, or a job
    that names no runs.
    """
    from darsia.fluidflower.pipeline.job_queue import read_json

    path = Path(root) / "job.json"
    if not path.is_file():
        raise JobError(f"No job.json in {Path(root)}")
    record = read_json(path)
    if record.get("format") != "darsia-alias-job-v1":
        raise JobError("A production job written by the job layer (darsia-alias-job-v1) is required")
    if not record.get("expected_runs"):
        raise JobError("The job names no runs")
    return record


class ProductionJob:
    """A production job's runs and, for its finished runs, their stored frames and masses.

    Opening reads only the job's records. Fields are read when they are asked
    for, through the existing readers, and the last few are kept in memory.
    A reader on another thread holds ``lock`` while it uses the job.
    """

    def __init__(self, root, *, cache_size: int = 12):
        """Read the job in ``root``: every run's state, and a checked reader per finished run."""
        from darsia.fluidflower.pipeline.job_queue import name

        self.root = Path(root)
        self.record = read_job_record(self.root)
        runs = [str(run) for run in self.record.get("expected_runs") or ()]
        self.lock = threading.RLock()
        self.readers, self.runs = {}, []
        for run in runs:
            key = name(run)
            if not (self.root / "done" / f"{key}.json").is_file():
                self.runs.append(RunState(run, _marker_state(self.root, key), _task_frames(self.root, key)))
                continue
            try:
                reader = _FinishedRun(self.root, run)
            except (JobError, OSError, ValueError, KeyError, TypeError) as error:
                self.runs.append(RunState(run, "Unreadable", _task_frames(self.root, key), str(error)))
                continue
            self.readers[run] = reader
            self.runs.append(RunState(run, COMPLETE, len(reader.frames(run))))
        self._data_reader = None
        self._cache, self._results = OrderedDict(), OrderedDict()
        self._cache_size = int(cache_size)
        self._masses, self._fields, self._layer_grids = {}, {}, {}

    def facts(self) -> str:
        """One line about the job: series, code commit, precision and storage profile."""
        parts = [f"series {self.record.get('series_id') or 'not recorded'}"]
        if self.record.get("commit"):
            parts.append(f"code {str(self.record['commit'])[:8]}")
        for key in ("precision", "storage_profile"):
            if self.record.get(key):
                parts.append(f"{key.replace('_', ' ')} {self.record[key]}")
        complete = sum(state.state == COMPLETE for state in self.runs)
        parts.append(f"{len(self.runs)} runs, {complete} complete")
        return " · ".join(parts)

    def state(self, run: str) -> RunState:
        """The state of one run of the job."""
        return next(state for state in self.runs if state.run == run)

    def frames(self, run: str) -> list[str]:
        """A finished run's frames (case ids) in time order; none for other runs."""
        reader = self.readers.get(run)
        return reader.frames(run) if reader is not None else []

    def _reader(self, run: str) -> FinishedJob:
        """The checked reader of a finished run."""
        try:
            return self.readers[run]
        except KeyError:
            raise JobError(f"Run {run} is not finished") from None

    def _data(self):
        """The analysis data reader, which verifies each archive once by SHA-256."""
        if self._data_reader is None:
            from darsia.fluidflower.mass.n1_pipeline import DataReader

            self._data_reader = DataReader()
        return self._data_reader

    def result(self, run: str, case: str) -> dict:
        """A frame's checked result record (the last few read stay in memory)."""
        key = (run, case)
        if key in self._results:
            self._results.move_to_end(key)
        else:
            self._results[key] = self._reader(run).frame_result(run, case)
            while len(self._results) > RESULT_CACHE_SIZE:
                self._results.popitem(last=False)
        return self._results[key]

    def _has_products(self, run: str, case: str) -> bool:
        """True when the frame's products index lists its products.npz."""
        return f"frames/{case}/products.npz" in self._reader(run).products[run]

    def _source(self, run: str, product) -> Path:
        """A readable file with a product's bytes, or JobError naming where the product is."""
        try:
            return self._reader(run).product_source(product)
        except JobError:
            raise JobError(f"{product.product_id} is only on the machine that produced it; "
                           "open the job there to see it") from None

    def _products_folder(self, run: str, case: str) -> Path:
        """The readable folder that holds a frame's products.npz."""
        return self._source(run, self._reader(run).product(run, f"frames/{case}/products.npz")).parent

    def sources(self, run: str) -> list[str]:
        """The stored products of a finished run's frames: 'products', 'layer' or both."""
        frames = self.frames(run)
        if not frames:
            return []
        found = []
        if self._has_products(run, frames[0]):
            found.append("products")
        if self._reader(run).layer(run, frames[0]) is not None:
            found.append("layer")
        return found

    def fields(self, run: str, source: str) -> list[str]:
        """The fields a run's first frame stores in one source, in page order."""
        key = (run, source)
        if key not in self._fields:
            case = self.frames(run)[0]
            if source == "products":
                names = self._product_names(run, case)
                found = [field for field, spec in FIELDS.items() if spec.products_array in names]
            else:
                arrays = self._layer_arrays(run, case)
                found = [field for field, spec in FIELDS.items()
                         if (spec.layer_array in arrays if spec.layer_array
                             else _phase_codes_from_fractions(arrays) is not None)]
            self._fields[key] = found
        return list(self._fields[key])

    def _product_names(self, run: str, case: str) -> set[str]:
        """The arrays a frame's products.npz stores or can rebuild (a lean frame's total mass)."""
        from darsia.fluidflower.storage_profile import frame_storage

        result = self.result(run, case)
        folder = self._products_folder(run, case)
        bound = dict(result["products"], path=str(folder / result["products"]["path"]))
        with np.load(self._data().path(bound), allow_pickle=False) as archive:
            names = set(archive.files)
        record = frame_storage(result)
        return names | (set(record["products"]["omitted"]) if record is not None else set())

    def _layer_arrays(self, run: str, case: str) -> dict:
        """The arrays of a frame's stored coarse layer, from its checked bytes."""
        reader = self._reader(run)
        layer = reader.layer(run, case)
        if layer is None:
            raise JobError(f"Frame {case} has no stored coarse layer")
        self._source(run, layer)
        with np.load(io.BytesIO(reader.product_bytes(layer)), allow_pickle=False) as archive:
            return {item: archive[item] for item in archive.files}

    def _layer_grid(self, run: str, case: str) -> dict:
        """The stored layer's grid (extent, resolution, axes) from its receipt, if recorded."""
        key = (run, case)
        if key not in self._layer_grids:
            reader, grid = self._reader(run), {}
            receipt = reader.layer(run, case, LAYER_RECEIPT)
            if receipt is not None:
                try:
                    grid = (json.loads(reader.product_bytes(receipt)).get("coarse") or {}).get("grid") or {}
                except (JobError, OSError, ValueError):
                    grid = {}
            self._layer_grids[key] = grid
        return self._layer_grids[key]

    def mass_rows(self, run: str) -> list[dict]:
        """A finished run's mass table rows with numbers as floats (empty without a table)."""
        if run not in self._masses:
            reader = self._reader(run)
            try:
                raw = reader.product_bytes(reader.mass_table(run))
            except JobError:
                rows = []
            else:
                rows = [{key: (value if key == "case_id" else _number(value)) for key, value in row.items()}
                        for row in csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))]
            self._masses[run] = rows
        return self._masses[run]

    def elapsed_h(self, run: str, case: str):
        """A frame's elapsed time in hours: from the mass table, else from its result record."""
        hours = mass_table_hours(self.mass_rows(run))
        if case in hours:
            return hours[case]
        try:
            value = self.result(run, case).get("elapsed_h")
        except (JobError, OSError, ValueError, KeyError):
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            return None
        return float(value)

    def product_folders(self, run: str) -> set[Path]:
        """The folders where a finished run's products lie: the job folder and the producer's."""
        folders = {self.root}
        reader = self.readers.get(run)
        if reader is None:
            return folders
        for product in reader.products[run].values():
            parts = PurePosixPath(product.product_id).parts
            if parts[:1] == ("frames",) and len(parts) >= 3:
                # The run's output folder: <output>/frames/<case>/<file>.
                folders.add(Path(product.local_path).parent.parent.parent)
            elif parts[:1] == ("attempts",) and len(parts) >= 3:
                folders.add(Path(product.local_path).parents[len(parts) - 3])
        return folders

    def field_image(self, run: str, case: str, field: str, source: str, *, cache: bool = True) -> FieldImage:
        """One frame's field prepared for display, read through the existing readers."""
        key = (run, case, field, source)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        image = (self._products_image(run, case, field) if source == "products"
                 else self._layer_image(run, case, field))
        if cache:
            self._cache[key] = image
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return image

    def _products_image(self, run: str, case: str, field: str) -> FieldImage:
        """A field of a frame's products.npz, with excluded pixels masked by its phase labels."""
        from darsia.fluidflower.storage_profile import read_products

        spec, result = FIELDS[field], self.result(run, case)
        names = [spec.products_array]
        if field != "phase_labels" and "phase_labels" in self.fields(run, "products"):
            names.append("phase_labels")
        arrays = read_products(result, self._products_folder(run, case), names, reader=self._data())
        values = np.asarray(arrays[spec.products_array])
        step = display_stride(values.shape)
        shown = np.ascontiguousarray(values[::step, ::step])
        if field == "phase_labels":
            return FieldImage(shown, None, True, "", "upper", None, step, values.shape)
        mask = ~np.isfinite(shown.astype(np.float64))
        if "phase_labels" in arrays:
            mask |= np.asarray(arrays["phase_labels"])[::step, ::step] == 255
        unit = str((result.get("units") or {}).get(spec.products_array) or _DEFAULT_UNITS[field])
        return FieldImage(shown, mask, False, "fraction" if unit == "1" else unit, "upper", None, step,
                          values.shape)

    def _layer_image(self, run: str, case: str, field: str) -> FieldImage:
        """A field of a frame's stored coarse layer; cells without area are masked."""
        arrays, grid = self._layer_arrays(run, case), self._layer_grid(run, case)
        spec = FIELDS[field]
        if spec.layer_array is None:
            values = _phase_codes_from_fractions(arrays)
        else:
            values = arrays[spec.layer_array]
        if values is None:
            raise JobError(f"Frame {case} stores no {spec.label.lower()}")
        values = np.asarray(values)
        step = display_stride(values.shape)
        shown = np.ascontiguousarray(values[::step, ::step])
        extent = grid.get("extent_m")
        if isinstance(extent, list) and len(extent) == 4:
            extent = tuple(float(item) for item in extent)
        else:
            extent = None
        if spec.layer_array is None:
            return FieldImage(shown, None, True, "", "lower", extent, step, values.shape)
        mask = ~np.isfinite(shown.astype(np.float64))
        area = arrays.get("static_pixel_area_m2_sum")
        if area is not None and np.shape(area) == values.shape:
            mask |= ~(np.asarray(area)[::step, ::step] > 0)
        return FieldImage(shown, mask, False, _LAYER_UNITS[field], "lower", extent, step, values.shape)


def read_run_contents(job: ProductionJob, run: str) -> dict:
    """A finished run's frames, sources with their names, fields per source and mass rows.

    Read through the job's checked readers while holding the job's lock; no
    Qt is touched, so the page reads it off the GUI thread. A source or field
    list that cannot be read is empty, with the reason in ``problem`` or
    ``problems``.
    """
    with job.lock:
        frames = job.frames(run)
        contents = dict(frames=frames, sources=[], fields={}, problems={}, problem="")
        try:
            sources = job.sources(run)
        except (JobError, OSError, ValueError, KeyError) as error:
            sources = []
            contents["problem"] = f"Run {run}: {error}"
        for source in sources:
            text = SOURCES[source]
            if source == "layer":
                resolution = job._layer_grid(run, frames[0]).get("resolution_mm")
                if isinstance(resolution, (int, float)) and not isinstance(resolution, bool):
                    text += f" ({resolution:g} mm)"
            contents["sources"].append((source, text))
            try:
                contents["fields"][source] = job.fields(run, source)
            except (JobError, OSError, ValueError, KeyError) as error:
                contents["fields"][source] = []
                contents["problems"][source] = f"Run {run}: {error}"
        contents["mass_rows"] = job.mass_rows(run)
    return contents


def read_frame_view(job: ProductionJob, run: str, case: str, field: str, source: str, *, number: int,
                    count: int, scale_max, mass_rows: list[dict], cancelled=lambda: False) -> dict | None:
    """One frame's field and elapsed time, drawn as the map and the run's mass curves.

    The field is read while holding the job's lock and drawn after it; no Qt
    is touched, so the page reads frames off the GUI thread. A continuous
    field without ``scale_max`` gets the field's physical limit, else a limit
    fitted to the frame. None when ``cancelled()`` is true before reading.
    """
    with job.lock:
        if cancelled():
            return None
        try:
            image, message = job.field_image(run, case, field, source), ""
        except (JobError, OSError, ValueError, KeyError) as error:
            image, message = None, f"{case} is not readable here: {error}"
        elapsed = job.elapsed_h(run, case)
    if scale_max is None and image is not None and not image.categorical:
        scale_max = _PHYSICAL_LIMITS.get(field) or fit_limit(image)
    pixels = render_map(image, title=frame_title(run, field, elapsed),
                        footer=frame_footer(number, count, case, image), scale_max=scale_max,
                        label=FIELDS[field].label, message=message)
    curves = render_mass_curves(mass_rows, run=run, marker=curve_marker(mass_rows, case))
    return dict(image=image, message=message, elapsed=elapsed, scale_max=scale_max, map=pixels,
                curves=curves)


def read_export_problems(job, runs, selected, field, source, cancelled=lambda: False):
    """The export resolutions unavailable for ``runs`` and for the ``selected`` run, with reasons.

    Reads every frame's grid receipt while holding the job's lock; no Qt is
    touched. None when ``cancelled()`` is true before reading.
    """
    with job.lock if job is not None else contextlib.nullcontext():
        if cancelled():
            return None
        return (export_resolution_problems(job, runs, field, source),
                export_resolution_problems(job, selected, field, source))


def export_resolution_problems(job, runs, field, source):
    """Explain unavailable choices using the finished runs' bound grid receipts."""
    problems = dict.fromkeys((None, 1, 2, 5, 10), "")
    if not runs or field not in FIELDS or source not in SOURCES:
        return dict.fromkeys(problems, "Choose a finished run, source and field first.")
    for run in runs:
        current_checked = False
        try:
            if job.state(run).state != COMPLETE:
                raise JobError(f"Run {run} is not finished.")
            sources = job.sources(run)
            if source not in sources or field not in job.fields(run, source):
                problems[None] = problems[None] or f"Run {run} has no {FIELDS[field].label} in {SOURCES[source]}."
            current_checked = True
            if "layer" not in sources or field not in job.fields(run, "layer"):
                raise JobError(f"Run {run} has no saved metric layer for {FIELDS[field].label}.")
            resolutions = []
            for case in job.frames(run):
                grid = job._layer_grid(run, case)
                value = grid.get("resolution_mm")
                if (not isinstance(value, (int, float)) or isinstance(value, bool)
                        or not math.isfinite(value) or value <= 0):
                    raise JobError(f"Run {run}, {case} has no documented metric resolution.")
                resolutions.append(value)
            if not resolutions:
                raise JobError(f"Run {run} has no saved metric frames.")
            coarsest = max(resolutions)
            for target in (1, 2, 5, 10):
                if target < coarsest:
                    problems[target] = problems[target] or f"Run {run} includes a {coarsest:g} mm layer; finer exports are unavailable."
                elif field == "phase_labels" and any(value != target for value in resolutions):
                    problems[target] = problems[target] or "Phase labels require the exact stored resolution; no coarser reduction is defined."
        except (JobError, OSError, ValueError, KeyError, TypeError) as error:
            if not current_checked:
                problems[None] = problems[None] or str(error)
            for target in (1, 2, 5, 10):
                problems[target] = problems[target] or str(error)
    return problems


def export_field_images(job, run, field, source, resolution_mm=None):
    """Read exact saved fields or reduce extensive cells using the existing rule."""
    frames = job.frames(run)
    if resolution_mm is None:
        for case in frames:
            yield case, job.field_image(run, case, field, source, cache=False)
        return
    from .storage_coarsening import _target
    from .published_dataset.export_mass_maps import iter_export_products

    target = _target(resolution_mm)
    reader = job._reader(run)
    # An explicit millimetre choice always refers to saved metric maps.
    resolutions = [job._layer_grid(run, case).get("resolution_mm") for case in frames]
    if not resolutions or any(value is None or target < value for value in resolutions):
        raise ValueError("Export maps need a saved metric layer at this resolution or finer")
    if all(value == target for value in resolutions):
        for case in frames:
            yield case, job._layer_image(run, case, field)
        return
    if field == "phase_labels":
        raise ValueError("Coarser phase-label figures have no prescribed reduction; choose a mass or concentration field")
    for case, product in iter_export_products(reader, run, target):
        arrays, grid = product["arrays"], product["metadata"]["grid"]
        values = arrays[FIELDS[field].layer_array]
        step = display_stride(values.shape)
        shown = np.ascontiguousarray(values[::step, ::step])
        mask = ~np.isfinite(shown) | ~arrays["cell_has_pore_volume"][::step, ::step]
        yield case, FieldImage(shown, mask, False, _LAYER_UNITS[field], "lower",
                               tuple(grid["extent_m"]), step, values.shape)
        del product, arrays, values, shown


def write_figures(job, run, field, source, folder, *, resolution_mm=None, scale_max=None):
    """Write every saved frame and the run's actual mass curves into a fresh folder."""
    from PIL import Image

    folder = Path(folder)
    if any(inside_folder(folder, protected) for protected in job.product_folders(run)):
        raise ValueError("Choose a folder outside the job folder and its product folders")
    target = fresh_folder(folder, f"{run}_{field}_figures")
    rows = []
    # Preflight and render one frame at a time. A failure preserves any written
    # figures and never records the package as complete.
    for number, (case, image) in enumerate(export_field_images(job, run, field, source, resolution_mm), 1):
        pixels = render_map(image, title=frame_title(run, field, job.elapsed_h(run, case)),
            footer=frame_footer(number, len(job.frames(run)), case, image),
            scale_max=scale_max, label=FIELDS[field].label)
        if not rows:
            target.mkdir(parents=True, exist_ok=False)
        name = f"frame_{number:06d}.png"
        with (target / name).open("xb") as stream:
            Image.fromarray(pixels).save(stream, format="PNG")
        rows.append(dict(case_id=case, path=name, elapsed_h=job.elapsed_h(run, case)))
    if not rows:
        raise ValueError("The run has no finished frames")
    masses = job.mass_rows(run)
    curves = None
    if curve_series(masses)[2]:
        curves = "mass_curves.png"
        with (target / curves).open("xb") as stream:
            Image.fromarray(render_mass_curves(masses, run=run)).save(stream, format="PNG")
    document = dict(format="darsia-production-figures-v1", complete=True, run=run,
        field=field, source="layer" if resolution_mm is not None else source, view_source=source,
        export_resolution_mm=resolution_mm, frames=rows,
        mass_curves=curves, mass_curve_state="saved table" if curves else "not available",
        source_products_index=job._reader(run).done[run]["products_index"],
        source_completion=job._reader(run).done[run]["completed"],
        job=dict(series_id=job.record.get("series_id"), commit=job.record.get("commit")))
    with (target / "figures.json").open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2, allow_nan=False)
    return target


def write_film(job: ProductionJob, run: str, field: str, source: str, folder, *, fps: float,
               scale_max: float | None = None, progress=None, resolution_mm=None) -> FilmResult | None:
    """Render a finished run's frames of one field and encode them as an MP4.

    The film goes to a new folder inside ``folder``, which must lie outside the
    job folder and the run's product folders. ``progress(done, total)`` may
    return False to cancel; a cancelled export writes nothing and returns None.
    Without an MP4 encoder (or when the MP4 does not decode as written) the
    numbered PNG frames are written instead, or as well, and the message says so.
    """
    from darsia.fluidflower.post_materialization import _write_movie
    from darsia.fluidflower.storage_saved_movies import _decoded_movie
    import cv2

    folder = Path(folder)
    if any(inside_folder(folder, protected) for protected in job.product_folders(run)):
        raise ValueError("Choose a folder outside the job folder and its product folders")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("A film needs a positive number of frames per second")
    frames = job.frames(run)
    if not frames:
        raise ValueError(f"Run {run} has no finished frames")
    label = FIELDS[field].label
    rows = []
    with tempfile.TemporaryDirectory(prefix="darsia-film-") as temporary:
        temporary = Path(temporary)
        paths = []
        for number, (case, image) in enumerate(export_field_images(job, run, field, source, resolution_mm), 1):
            elapsed = job.elapsed_h(run, case)
            pixels = render_map(image, title=frame_title(run, field, elapsed),
                                footer=frame_footer(number, len(frames), case, image),
                                scale_max=scale_max, label=label)
            path = temporary / f"frame_{number:06d}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)):
                raise OSError(f"Could not write a film frame: {path}")
            paths.append(path)
            rows.append({"case_id": case, "elapsed_h": elapsed, "shown": image is not None})
            if progress is not None and progress(number, len(frames)) is False:
                return None
        target = fresh_folder(folder, f"{run}_{field}_film")
        stem = f"{run}_{field}"
        movie, frames_folder, encoder, problem = None, None, None, ""
        try:
            encoded = temporary / f"{stem}.mp4"
            encoder = _write_movie(paths, encoded, float(fps))
            _decoded_movie(encoded, expected_count=len(paths), shape=(MAP_SIZE_PX[1], MAP_SIZE_PX[0]),
                           fps=float(fps))
        except (RuntimeError, ValueError, OSError, cv2.error) as error:
            problem = str(error)
            if encoder is None or not encoded.is_file():
                encoder = None
        target.mkdir(parents=True)
        if encoder is not None:
            movie = target / f"{stem}.mp4"
            shutil.copyfile(encoded, movie)
        if problem:
            frames_folder = target / "frames"
            frames_folder.mkdir()
            for path in paths:
                shutil.copyfile(path, frames_folder / path.name)
    if movie is not None and not problem:
        message = f"Film written: {movie} ({len(paths)} frames at {fps:g} frames per second)."
    elif movie is not None:
        message = (f"The MP4 could not be checked ({problem}); it is in {movie}, and the "
                   f"{len(paths)} numbered PNG frames are in {frames_folder}.")
    else:
        message = (f"No MP4 encoder is available here ({problem}); the {len(paths)} numbered PNG frames "
                   f"are in {frames_folder}.")
    document = dict(format=FILM_FORMAT, run=run, field=field, field_label=label,
                    source="layer" if resolution_mm is not None else source, view_source=source,
                    fps=float(fps), colour_scale=None if field == "phase_labels" else [0.0, scale_max],
                    movie=None if movie is None else movie.name,
                    png_frames=None if frames_folder is None else frames_folder.name, encoder=encoder,
                    frames=rows, export_resolution_mm=resolution_mm,
                    source_products_index=job._reader(run).done[run]["products_index"],
                    source_completion=job._reader(run).done[run]["completed"],
                    job=dict(series_id=job.record.get("series_id"), commit=job.record.get("commit")),
                    written_by="Results and export; the job folder was only read")
    text = json.dumps(document, indent=1, allow_nan=False) + "\n"
    (target / "film.json").write_text(text, encoding="utf-8")
    return FilmResult(target, movie, frames_folder, len(paths), encoder, message)


def _pixmap(pixels: np.ndarray) -> QPixmap:
    """A QPixmap holding its own copy of RGBA pixels."""
    pixels = np.ascontiguousarray(pixels)
    height, width = pixels.shape[:2]
    image = QImage(pixels.data, width, height, pixels.strides[0], QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())


class _ScaledImage(QLabel):
    """An image that fits its label, keeping its aspect ratio, or a short message."""

    def __init__(self, text: str = "", minimum_height: int = 200):
        """An empty view showing ``text`` until an image is set."""
        super().__init__(text)
        self._source = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setMinimumSize(1, minimum_height)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_pixels(self, pixels: np.ndarray) -> None:
        """Show RGBA pixels, scaled to fit."""
        self._source = _pixmap(pixels)
        self.setText("")
        self._rescale()

    def set_message(self, text: str) -> None:
        """Show a message instead of an image."""
        self._source = QPixmap()
        self.clear()
        self.setText(text)

    def source_pixmap(self) -> QPixmap:
        """The full-size image shown (null when a message is shown)."""
        return self._source

    def resizeEvent(self, event) -> None:
        """Fit the image to the new size."""
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        """Scale the image to the label, keeping its aspect ratio."""
        if not self._source.isNull():
            self.setPixmap(self._source.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))


class ProductionResultsPage(QWidget):
    """A finished production job, read-only: every run, its frames, maps and mass curves, and films."""

    def __init__(self, parent=None):
        """An empty page; "Open finished job…" or :meth:`open_job` fills it."""
        super().__init__(parent)
        self.setObjectName("productionResultsPage")
        self.job: ProductionJob | None = None
        self.run: str | None = None
        self.image: FieldImage | None = None
        self.shown: tuple | None = None
        self.last_film: FilmResult | None = None
        self._scales = {}
        self._folder = ""
        # Read in the background: the run's contents, each frame's elapsed
        # hours, and the last export check (its scope and problems).
        self._readers, self._contents, self._elapsed, self._export_check = {}, None, {}, None
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(self.scroll)
        content = QWidget()
        self.scroll.setWidget(content)
        layout = QVBoxLayout(content)
        self.title = QLabel("Results")
        font = self.title.font()
        font.setBold(True)
        self.title.setFont(font)
        layout.addWidget(self.title)
        intro = QLabel("Open a finished production job to see every run, its frames, maps and mass curves. "
                       "Reading starts no calculation and writes nothing; a film goes to a new folder you "
                       "choose, outside the job.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        opening = QHBoxLayout()
        self.open_button = QPushButton("Open finished job…")
        self.open_button.clicked.connect(self.choose_job)
        opening.addWidget(self.open_button)
        self.job_label = QLabel("No job is open.")
        self.job_label.setWordWrap(True)
        self.job_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        opening.addWidget(self.job_label, 1)
        layout.addLayout(opening)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)

        body = QHBoxLayout()
        layout.addLayout(body, 1)
        self.runs_table = QTableWidget(0, 3)
        self.runs_table.setHorizontalHeaderLabels(["Run", "State", "Frames"])
        self.runs_table.verticalHeader().hide()
        self.runs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.runs_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.runs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.runs_table.horizontalHeader().setStretchLastSection(True)
        self.runs_table.setMinimumWidth(240)
        self.runs_table.setMaximumWidth(360)
        self.runs_table.itemSelectionChanged.connect(self._run_selected)
        body.addWidget(self.runs_table)

        right = QVBoxLayout()
        body.addLayout(right, 1)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("Field"))
        self.field_combo = QComboBox()
        self.field_combo.currentIndexChanged.connect(self._field_changed)
        controls.addWidget(self.field_combo)
        self.source_label = QLabel("Stored as")
        controls.addWidget(self.source_label)
        self.source_combo = QComboBox()
        self.source_combo.currentIndexChanged.connect(self._source_changed)
        controls.addWidget(self.source_combo)
        self.scale_label = QLabel("Colour scale 0 to")
        controls.addWidget(self.scale_label)
        self.scale_edit = QLineEdit()
        self.scale_edit.setValidator(QDoubleValidator(0.0, 1e12, 12, self.scale_edit))
        self.scale_edit.setMaximumWidth(110)
        self.scale_edit.setToolTip("Fixed for every frame of the run and in its film.")
        self.scale_edit.editingFinished.connect(self._scale_edited)
        controls.addWidget(self.scale_edit)
        self.fit_button = QPushButton("Fit to this frame")
        self.fit_button.setToolTip("Set the colour scale to the largest value of the shown frame.")
        self.fit_button.clicked.connect(self.fit_scale)
        controls.addWidget(self.fit_button)
        controls.addStretch(1)
        right.addLayout(controls)
        self.map_view = _ScaledImage("Open a finished job and choose a run.", 320)
        right.addWidget(self.map_view, 3)
        time_row = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.valueChanged.connect(self._slider_moved)
        time_row.addWidget(self.frame_slider, 1)
        self.frame_label = QLabel()
        time_row.addWidget(self.frame_label)
        right.addLayout(time_row)
        self.curves_view = _ScaledImage("", 180)
        right.addWidget(self.curves_view, 2)
        film = QHBoxLayout()
        film.addWidget(QLabel("Film: frames per second"))
        self.fps_spin = QDoubleSpinBox()
        self.fps_spin.setRange(0.25, 60.0)
        self.fps_spin.setDecimals(2)
        self.fps_spin.setSingleStep(0.5)
        self.fps_spin.setValue(DEFAULT_FPS)
        film.addWidget(self.fps_spin)
        self.export_button = QPushButton("Export film…")
        self.export_button.setToolTip(
            "Write an MP4 of the selected run and field into a new folder you choose.")
        self.export_button.clicked.connect(lambda: self.export_film())
        film.addWidget(self.export_button)
        film.addStretch(1)
        right.addLayout(film)
        exports = QHBoxLayout()
        self.export_resolution_combo = QComboBox()
        self.export_resolution_combo.addItem("Current stored display", None)
        for resolution in (1, 2, 5, 10):
            self.export_resolution_combo.addItem(f"{resolution} mm", resolution)
        self.export_resolution_combo.setToolTip(
            "A millimetre choice exports the stored metric fields, at the stored resolution or coarser. "
            "Mass-conserving reduction requires the saved extensive fields.")
        exports.addWidget(QLabel("Export resolution"))
        exports.addWidget(self.export_resolution_combo)
        self.export_scope_combo = QComboBox()
        self.export_scope_combo.addItem("Selected run", "selected")
        self.export_scope_combo.addItem("All finished runs", "series")
        self.export_scope_combo.currentIndexChanged.connect(self._update_export_resolutions)
        self.export_resolution_combo.currentIndexChanged.connect(self._update_export_resolutions)
        exports.addWidget(self.export_scope_combo)
        self.figures_button = QPushButton("Export figures…")
        self.figures_button.clicked.connect(lambda: self.export_media("figures"))
        exports.addWidget(self.figures_button)
        self.series_film_button = QPushButton("Export films for scope…")
        self.series_film_button.clicked.connect(lambda: self.export_media("films"))
        exports.addWidget(self.series_film_button)
        right.addLayout(exports)
        self.export_resolution_note = QLabel()
        self.export_resolution_note.setWordWrap(True)
        right.addWidget(self.export_resolution_note)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._show_current_frame)
        self._set_run_controls(False)

    # ------------------------------------------------------------------ job

    def choose_job(self) -> bool:
        """Ask for a job folder and open it."""
        folder = QFileDialog.getExistingDirectory(self, "Open finished job", self._folder)
        return self.open_job(folder) if folder else False

    @property
    def loading(self) -> bool:
        """True while the job, a run, a frame or the export check is read in the background."""
        return any(reader.busy for reader in self._readers.values())

    def _reader(self, name: str, slot):
        """The page's background reader ``name``, made once and connected to ``slot``."""
        reader = self._readers.get(name)
        if reader is None:
            from darsia.fluidflower.ui_metadata_preview import MetadataPreview

            reader = self._readers[name] = MetadataPreview(self)
            reader.completed.connect(slot)
        return reader

    def _cancel_reads(self, *names: str) -> None:
        """Drop the answers of the named background reads still under way."""
        for name in names:
            if name in self._readers:
                self._readers[name].cancel()

    def open_job(self, root) -> bool:
        """Open a job folder read-only and list all its runs; False when it is no production job.

        The job's record is checked at once. Its finished runs' indexes and
        receipts are read in the background (seconds to minutes on the shared
        drive), and the runs are listed when they are read.
        """
        try:
            read_job_record(root)
        except (JobError, OSError, ValueError, KeyError, TypeError) as error:
            self.status_label.setText(f"Could not open the job: {error}")
            return False
        self._cancel_reads("run", "frame", "exports")
        self.job, self.run, self.image, self.shown, self._scales = None, None, None, None, {}
        self._contents, self._elapsed, self._export_check = None, {}, None
        self._set_run_controls(False)
        self.runs_table.setRowCount(0)
        self.map_view.set_message("")
        self.curves_view.set_message("")
        self.frame_label.setText("")
        self.job_label.setText(str(Path(root)))
        self.status_label.setText("Reading the job's finished runs…")
        self._reader("job", self._job_read).start(lambda _cancelled: (root, ProductionJob(root)))
        return True

    def _job_read(self, value, error) -> None:
        """List the runs of the job read in the background and show its first finished run."""
        if error or value is None:
            self.status_label.setText(f"Could not open the job: {error or 'no answer was returned'}")
            return
        root, job = value
        self.job, self.run, self.image, self._scales = job, None, None, {}
        self._folder = str(Path(root).parent)
        self.job_label.setText(f"{Path(root)}\n{job.facts()}")
        self.status_label.setText("Choose a run. Its products open read-only.")
        self.runs_table.blockSignals(True)
        self.runs_table.setRowCount(len(job.runs))
        for row, state in enumerate(job.runs):
            frames = "–" if state.frames is None else str(state.frames)
            for column, text in enumerate((state.run, state.state, frames)):
                item = QTableWidgetItem(text)
                if state.note:
                    item.setToolTip(state.note)
                self.runs_table.setItem(row, column, item)
        self.runs_table.blockSignals(False)
        first = next((row for row, state in enumerate(job.runs) if state.state == COMPLETE), 0)
        self.runs_table.selectRow(first)
        if self.run is None:
            self._run_selected()

    def _run_selected(self) -> None:
        """Show the run selected in the table."""
        rows = self.runs_table.selectionModel().selectedRows() if self.runs_table.selectionModel() else []
        if self.job is not None and rows:
            self.select_run(self.job.runs[rows[0].row()].run)

    def select_run(self, run: str) -> None:
        """Show one run: its fields, frames and mass curves, or why it cannot be shown.

        A finished run's sources and fields are read in the background; its
        first frame is shown when they are read.
        """
        if self.job is None:
            return
        self._cancel_reads("run", "frame", "exports")
        self.run, self.image, self.shown = run, None, None
        self._contents, self._elapsed, self._export_check = None, {}, None
        state = self.job.state(run)
        if state.state != COMPLETE:
            self._set_run_controls(False)
            if state.note:
                text = f"Run {run} has a done record, but it could not be read: {state.note}"
            else:
                text = f"Run {run} is {state.state.lower()}; its products open when it is complete."
            self.map_view.set_message(text)
            self.curves_view.set_message("")
            self.frame_label.setText("")
            return
        self._set_run_controls(False)
        self.map_view.set_message(f"Reading run {run}…")
        self.curves_view.set_message("")
        self.frame_label.setText("")
        job = self.job
        self._reader("run", self._run_read).start(lambda _cancelled: (job, run, read_run_contents(job, run)))

    def _run_read(self, value, error) -> None:
        """Fill the source, field and frame controls of the run read in the background."""
        if error or value is None:
            self.map_view.set_message(f"Run {self.run} could not be read: {error or 'no answer was returned'}")
            return
        job, run, contents = value
        if job is not self.job or run != self.run:
            return
        self._contents, self._elapsed = contents, mass_table_hours(contents["mass_rows"])
        if contents["problem"]:
            self.status_label.setText(contents["problem"])
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        for source, text in contents["sources"]:
            self.source_combo.addItem(text, source)
        self.source_combo.blockSignals(False)
        visible = len(contents["sources"]) > 1
        self.source_label.setVisible(visible)
        self.source_combo.setVisible(visible)
        frames = contents["frames"]
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, len(frames) - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self._set_run_controls(bool(contents["sources"]))
        if not contents["sources"]:
            self.map_view.set_message(f"Run {run} has no stored maps on this job.")
        self._fill_fields()

    def _fill_fields(self) -> None:
        """List the fields the run stores in the chosen source, keeping the field when it exists."""
        wanted = self.field_combo.currentData()
        source = self.source_combo.currentData()
        fields, problem = [], ""
        if self._contents is not None and source is not None:
            fields = self._contents["fields"].get(source, [])
            problem = self._contents["problems"].get(source, "")
            if problem:
                self.status_label.setText(problem)
        self.field_combo.blockSignals(True)
        self.field_combo.clear()
        for field in fields:
            self.field_combo.addItem(FIELDS[field].label, field)
        index = self.field_combo.findData(wanted)
        self.field_combo.setCurrentIndex(index if index >= 0 else 0)
        self.field_combo.blockSignals(False)
        if not fields and source is not None:
            self.map_view.set_message(problem or f"Run {self.run} stores none of the fields this page shows.")
        self.export_button.setEnabled(bool(fields))
        self._update_export_resolutions()
        self.show_frame(self.frame_slider.value())

    def _media_runs(self):
        """The same finished-run scope used by both media availability and export."""
        if self.job is None or self.run is None:
            return []
        return ([row.run for row in self.job.runs if row.state == COMPLETE]
                if self.export_scope_combo.currentData() == "series" else [self.run])

    def _update_export_resolutions(self, *_args):
        """Keep unavailable metric choices visible with their reason, but unselectable.

        Which resolutions a scope allows is checked in the background (every
        frame's grid receipt) once per scope, field and source.
        """
        field, source = self.field_combo.currentData(), self.source_combo.currentData()
        runs = self._media_runs()
        scope = (self.job, tuple(runs), self.run, field, source)
        if self._export_check is not None and self._export_check[0] == scope:
            self._show_export_problems(*self._export_check[1])
            return
        for button in (self.figures_button, self.series_film_button, self.export_button):
            button.setEnabled(False)
        self.export_resolution_note.setText("Checking which export resolutions are available…")
        job, selected = self.job, [self.run] if self.run else []
        self._reader("exports", self._export_problems_read).start(lambda cancelled: (
            scope, read_export_problems(job, runs, selected, field, source, cancelled)))

    def _export_problems_read(self, value, error) -> None:
        """Show the export check read in the background, unless the scope changed meanwhile."""
        if error or value is None:
            self.export_resolution_note.setText(
                f"The export resolutions could not be checked: {error or 'no answer was returned'}")
            return
        scope, problems = value
        if problems is None:
            return
        self._export_check = (scope, problems)
        self._update_export_resolutions()

    def _show_export_problems(self, problems: dict, selected: dict) -> None:
        """Mark the unavailable export choices for the scope and the selected run."""
        combo = self.export_resolution_combo
        previous_problem = problems.get(combo.currentData(), "")
        combo.blockSignals(True)
        for index in range(combo.count()):
            reason = problems[combo.itemData(index)]
            combo.model().item(index).setEnabled(not reason)
            combo.setItemData(index, reason, Qt.ItemDataRole.ToolTipRole)
        if previous_problem:
            combo.setCurrentIndex(0)
        combo.blockSignals(False)
        available = ", ".join(f"{target} mm" for target in (1, 2, 5, 10) if not problems[target])
        reasons = list(dict.fromkeys(reason for reason in problems.values() if reason))
        note = (f"Metric export uses Stored coarse layer. Available: {available}." if available
                else "Millimetre export is unavailable for this scope.")
        if reasons:
            note += " " + " ".join(reasons)
        self.export_resolution_note.setText(note)
        valid = not problems[combo.currentData()]
        self.figures_button.setEnabled(valid)
        self.series_film_button.setEnabled(valid)
        self.export_button.setEnabled(not selected[combo.currentData()])

    def _set_run_controls(self, enabled: bool) -> None:
        """Enable the frame, field and film controls for a shown run."""
        for widget in (self.field_combo, self.source_combo, self.scale_edit, self.fit_button,
                       self.frame_slider, self.fps_spin, self.export_button,
                       self.export_resolution_combo, self.export_scope_combo,
                       self.figures_button, self.series_film_button):
            widget.setEnabled(enabled)

    # --------------------------------------------------------------- frames

    def _source_changed(self, *_args) -> None:
        """Show the chosen source's fields."""
        self._fill_fields()

    def _field_changed(self, *_args) -> None:
        """Show the chosen field of the current frame."""
        self._update_export_resolutions()
        self.show_frame(self.frame_slider.value())

    def _slider_moved(self, value: int) -> None:
        """Name the frame at once; read it when the slider rests."""
        self._label_frame(value)
        self._timer.start()

    def _label_frame(self, index: int) -> None:
        """The frame line under the map: number, case id and elapsed time."""
        frames = self.job.frames(self.run) if self.job is not None and self.run else []
        if not frames:
            self.frame_label.setText("")
            return
        case = frames[index]
        self.frame_label.setText(f"Frame {index + 1} of {len(frames)} · {case} · "
                                 f"{format_hours(self._elapsed.get(case))}")

    def show_frame(self, index: int) -> None:
        """Show one frame of the selected run now."""
        self._timer.stop()
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(index)
        self.frame_slider.blockSignals(False)
        self._show_current_frame()

    def _scale_key(self):
        """The colour scale's key: one per run, source and field."""
        return (self.run, self.source_combo.currentData(), self.field_combo.currentData())

    def _show_current_frame(self) -> None:
        """Read the selected frame's field in the background and draw it with the run's mass curves."""
        field, source = self.field_combo.currentData(), self.source_combo.currentData()
        if self.job is None or self.run is None or self._contents is None or field is None or source is None:
            return
        index = self.frame_slider.value()
        frames = self._contents["frames"]
        if not frames:
            return
        case = frames[index]
        self._label_frame(index)
        job, run, rows = self.job, self.run, self._contents["mass_rows"]
        wanted, scale_max = (job, run, case, field, source), self._scales.get(self._scale_key())
        self.status_label.setText(f"Reading {case}…")
        self._reader("frame", self._frame_read).start(lambda cancelled: (wanted, read_frame_view(
            job, run, case, field, source, number=index + 1, count=len(frames), scale_max=scale_max,
            mass_rows=rows, cancelled=cancelled)))

    def _frame_read(self, value, error) -> None:
        """Show the frame read in the background, unless the run, field or source changed meanwhile."""
        if error or value is None:
            self.status_label.setText(f"The frame could not be shown: {error or 'no answer was returned'}")
            return
        (job, run, case, field, source), drawn = value
        if (drawn is None or job is not self.job or run != self.run
                or (field, source) != (self.field_combo.currentData(), self.source_combo.currentData())):
            return
        self.image, self.shown = drawn["image"], (run, case, field, source)
        self._elapsed[case] = drawn["elapsed"]
        key = self._scale_key()
        if self.image is not None and not self.image.categorical and key not in self._scales:
            self._scales[key] = drawn["scale_max"]
        self._sync_scale()
        if self._scales.get(key) == drawn["scale_max"]:
            self.map_view.set_pixels(drawn["map"])
        else:
            self._draw_map(drawn["message"])  # The scale was changed meanwhile.
        self.curves_view.set_pixels(drawn["curves"])
        self._label_frame(self.frame_slider.value())
        self.status_label.setText("" if self.image is not None else drawn["message"])

    def _sync_scale(self) -> None:
        """Show the run's colour scale for the field (none for phase labels)."""
        categorical = self.field_combo.currentData() == "phase_labels"
        for widget in (self.scale_label, self.scale_edit, self.fit_button):
            widget.setVisible(not categorical)
        limit = self._scales.get(self._scale_key())
        self.scale_edit.setText("" if limit is None else f"{limit:g}")

    def _draw_map(self, message: str = "") -> None:
        """Render the shown frame's field into the map view."""
        if self.shown is None or self._contents is None:
            return
        run, case, field, _source = self.shown
        frames = self._contents["frames"]
        pixels = render_map(self.image, title=frame_title(run, field, self._elapsed.get(case)),
                            footer=frame_footer(frames.index(case) + 1, len(frames), case, self.image),
                            scale_max=self._scales.get(self._scale_key()), label=FIELDS[field].label,
                            message=message)
        self.map_view.set_pixels(pixels)

    def _scale_edited(self) -> None:
        """Use the typed colour-scale maximum for every frame of the run and its film."""
        try:
            value = float(self.scale_edit.text())
        except ValueError:
            value = math.nan
        if not math.isfinite(value) or value <= 0:
            self._sync_scale()
            return
        self._scales[self._scale_key()] = value
        if self.image is not None:
            self._draw_map()

    def fit_scale(self) -> None:
        """Fit the colour scale to the largest value of the shown frame."""
        if self.image is None or self.image.categorical:
            return
        self._scales[self._scale_key()] = fit_limit(self.image)
        self._sync_scale()
        self._draw_map()

    # ----------------------------------------------------------------- film

    def export_film(self, folder=None) -> FilmResult | None:
        """Write a film of the selected run and field into a new folder in a chosen folder."""
        field, source = self.field_combo.currentData(), self.source_combo.currentData()
        if self.job is None or self.run is None or field is None or source is None:
            self.status_label.setText("Open a finished job and choose a run and a field first.")
            return None
        with self.job.lock:
            problem = export_resolution_problems(self.job, [self.run], field, source)[
                self.export_resolution_combo.currentData()]
        if problem:
            self.status_label.setText(f"The film was not written: {problem}")
            return None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Choose a folder for the film", self._folder)
            if not folder:
                return None
        if any(inside_folder(folder, protected) for protected in self.job.product_folders(self.run)):
            self.status_label.setText("Choose a folder outside the job folder and its product folders. "
                                      "Nothing was written.")
            return None
        frames = self.job.frames(self.run)
        dialog = QProgressDialog("Rendering film frames…", "Cancel", 0, len(frames), self)
        dialog.setWindowTitle("Export film")
        dialog.setWindowModality(Qt.WindowModality.WindowModal)
        dialog.setMinimumDuration(400)

        def progress(done, total):
            """Show the rendered frame count; False when the user cancels."""
            dialog.setValue(done)
            return not dialog.wasCanceled()

        self.export_button.setEnabled(False)
        try:
            with self.job.lock:
                result = write_film(self.job, self.run, field, source, folder, fps=self.fps_spin.value(),
                                    scale_max=self._scales.get(self._scale_key()), progress=progress,
                                    resolution_mm=self.export_resolution_combo.currentData())
        except (JobError, OSError, ValueError) as error:
            self.status_label.setText(f"The film was not written: {error}")
            return None
        finally:
            dialog.reset()
            dialog.deleteLater()
            self.export_button.setEnabled(True)
        if result is None:
            self.status_label.setText("The film export was cancelled. Nothing was written.")
            return None
        self.last_film = result
        self._folder = str(folder)
        self.status_label.setText(result.message)
        return result

    def export_media(self, kind, folder=None):
        """Export actual figures or films for the selected run or all finished runs."""
        field, source = self.field_combo.currentData(), self.source_combo.currentData()
        if self.job is None or self.run is None or field is None or source is None:
            self.status_label.setText("Open a finished job and choose a field first.")
            return None
        runs = self._media_runs()
        with self.job.lock:
            problem = export_resolution_problems(self.job, runs, field, source)[
                self.export_resolution_combo.currentData()]
        if problem:
            self.status_label.setText(f"Export is unavailable: {problem}")
            return None
        if folder is None:
            folder = QFileDialog.getExistingDirectory(self, "Choose an export folder", self._folder)
        if not folder:
            return None
        # Protect every selected run before any output, including a later run.
        if any(inside_folder(folder, protected) for run in runs
               for protected in self.job.product_folders(run)):
            self.status_label.setText("Choose a folder outside the job and every selected run's products.")
            return None
        outputs = []
        self._set_run_controls(False)
        try:
            with self.job.lock:
                for run in runs:
                    options = dict(resolution_mm=self.export_resolution_combo.currentData(),
                                   scale_max=self._scales.get(self._scale_key()))
                    if kind == "figures":
                        outputs.append(write_figures(self.job, run, field, source, folder, **options))
                    elif kind == "films":
                        outputs.append(write_film(self.job, run, field, source, folder,
                                                   fps=self.fps_spin.value(), **options).folder)
                    else:
                        raise ValueError("Choose figures or films")
        except (JobError, OSError, ValueError, KeyError) as error:
            self.status_label.setText(f"Export stopped: {error}. Completed run exports kept: {len(outputs)}.")
            return None
        finally:
            self._set_run_controls(True)
            self._update_export_resolutions()
        self.status_label.setText(f"Exported {kind} for {len(outputs)} finished runs to {folder}.")
        self._folder = str(folder)
        return outputs
