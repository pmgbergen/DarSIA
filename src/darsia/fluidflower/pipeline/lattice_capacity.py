"""Claim-time result-space ESTIMATE of a lattice run (pipeline ``lattice-phase-n1``).

No measured capacity profile exists for lattice runs. Their estimate comes from
the run's own configuration, never from a native profile or measurement:

* products per frame: the lattice shape and the run precision, at documented
  uncompressed bytes per lattice cell of every file a lattice frame writes
  (frame files and stage), plus the stored products when the request stores
  coarser, plus the measured native allowance for JSON files and receipts;
* run constants: the lattice inputs written once per run (``lattice/``);
* in flight: the frame being written, plus that frame's transient native
  preparation (native centre and ring windows in float32 RGB with a valid
  mask, per pixel of the declared native canvas). The runner writes it to a
  local transient folder and removes it after the frame.

No compression gain is assumed. Every figure is an ESTIMATE until a lattice
run has been measured. Native routes never reach this module.
"""
from __future__ import annotations

from dataclasses import dataclass

from .capacity import DEFAULT_RESERVE_BYTES, _integer

ESTIMATE_FORMAT = "darsia-lattice-capacity-estimate-v1"
LATTICE_ESTIMATE_MODEL = (
    "ESTIMATE, not a measurement: lattice products per frame from the lattice shape and precision "
    "at documented uncompressed bytes per cell, plus the transient native preparation of the frame "
    "in flight; no measured lattice profile exists yet and no native profile is reused")
#: Bytes per computing-lattice cell of one frame's files, besides the float products.
FRAME_CELL_BYTES = {
    "products.npz: phase labels and four boolean masks": 5,
    "native_geometry.npz: area, depth, porosity, transport weight (float64) and facies": 33,
    "stage: current, full and temporal log RGB (float32)": 36,
    "stage: current, full and temporal valid masks": 3,
    "stage: phase_area.npz (float64)": 8,
    "stage: phase_classes.npz (six masks)": 6,
    "stage: phase_operator_masks.npz (fourteen masks)": 14,
    "stage: phase labels, photo top and original domain": 3,
}
#: Float products at the run precision per lattice cell: concentration,
#: saturation and the three mass densities.
FRAME_PRECISION_FIELDS = 5
#: Bytes per lattice cell of the run's constant lattice inputs.
RUN_CELL_BYTES = {
    "lattice baseline log RGB (float32) and valid mask": 13,
    "lattice clear-water RGB (float32) and valid mask": 13,
    "lattice physical reference: domain, porosity and depth (float64 bound), facies": 18,
}
#: The lattice floor curve: one float64 per lattice column.
FLOOR_BYTES_PER_COLUMN = 8
#: Stored products per stored cell: an upper bound of 32 arrays of 8 bytes.
STORED_CELL_BYTES = 32 * 8
#: Transient native preparation per native canvas pixel: float32 RGB of the
#: native centre and the ring windows (12 bytes) and the centre valid mask.
NATIVE_TRANSIENT_BYTES_PER_PIXEL = 13


@dataclass(frozen=True)
class LatticeEstimate:
    """The admission ESTIMATE of one lattice run.

    It has the interface of the other capacity estimates
    (``products_estimated_bytes``, ``transient_estimated_bytes``,
    :meth:`needed`), so the capacity ledger reserves it like any run. The
    transient part is the frame in flight with its native preparation, plus
    the run's constant lattice inputs, which stay reserved with the run.
    """

    run: str
    case_count: int
    pipeline: str
    precision: str
    computing_resolution_mm: float
    lattice_shape_hw: tuple
    stored_resolution_mm: float
    stored_shape_hw: tuple
    declared_canvas_pixels: int
    products_per_frame_estimated_bytes: int
    run_constant_estimated_bytes: int
    native_transient_estimated_bytes: int
    inflight_estimated_bytes: int
    estimate: dict
    model: str = LATTICE_ESTIMATE_MODEL

    @property
    def products_estimated_bytes(self):
        """Lattice products of every frame of the run."""
        return self.products_per_frame_estimated_bytes * self.case_count

    @property
    def transient_estimated_bytes(self):
        """The frame in flight with its native preparation, and the run constants."""
        return self.inflight_estimated_bytes + self.run_constant_estimated_bytes

    def needed(self, *, other_remaining_bytes=0, reserve_bytes=DEFAULT_RESERVE_BYTES):
        """Bytes this run needs on the results volume, with other reservations and the reserve."""
        return (self.products_estimated_bytes + self.transient_estimated_bytes
                + _integer(other_remaining_bytes, "Other active remaining bytes")
                + _integer(reserve_bytes, "Reserve bytes"))


def lattice_estimate(task, config):
    """The ESTIMATE of one lattice task from its own run configuration.

    The task must be an explicit lattice-phase-n1 task whose precision,
    computing resolution, run and exact case order match the configuration.
    A configuration without a declared native canvas is refused: nothing is
    counted as zero.
    """
    from ..computing_lattice import computing_lattice
    from ..lattice_storage import stored_layer
    from ..storage_profile import declared_canvas_pixels, estimate_job_bytes
    from .job_queue import name
    from .lattice_runner import PIPELINE, RESOLUTION_KEY, lattice_request
    from .series import PIPELINE as NATIVE
    cases, precision = task.get("ordered_case_ids"), task.get("precision")
    if (task.get("pipeline") != PIPELINE or precision not in {"float32", "float64"}
            or not isinstance(cases, list) or not cases or len(set(cases)) != len(cases)
            or not isinstance(config.get("frames"), list)
            or [frame.get("case_id") for frame in config["frames"]] != cases
            or config.get("run") != task.get("run")):
        raise ValueError("Lattice capacity estimate: task pipeline, precision, run or exact case order differs")
    if "gas_probability_storage" in config:
        raise ValueError("Lattice runs omit the gas-probability storage request")
    request = lattice_request(config, precision)
    if task.get(RESOLUTION_KEY) != request[RESOLUTION_KEY]:
        raise ValueError("Task and configuration computing resolutions differ")
    lattice = computing_lattice(request["extent_m"], request[RESOLUTION_KEY])
    layer = stored_layer(request, lattice)
    pixels = declared_canvas_pixels(config, pipeline=NATIVE)
    if pixels is None:
        raise ValueError("The run configuration declares no native canvas; "
                         "its transient native preparation cannot be estimated")
    rows, columns = lattice.shape
    cells = rows * columns
    # The measured per-frame JSON allowance of the shared native size estimate.
    documented = estimate_job_bytes("full", 1, pipeline=NATIVE)
    json_bytes = documented["files"]["JSON and receipts"]
    files = {key: cells * size for key, size in FRAME_CELL_BYTES.items()}
    files["products.npz: concentration, saturation and three mass densities (" + precision + ")"] = (
        cells * FRAME_PRECISION_FIELDS * (4 if precision == "float32" else 8))
    stored_shape = tuple(lattice.shape) if layer is None else tuple(layer.shape)
    if layer is not None:
        files["stored products: at most 32 arrays of 8 bytes per stored cell"] = (
            stored_shape[0] * stored_shape[1] * STORED_CELL_BYTES)
    files["JSON files and receipts (native allowance per frame)"] = json_bytes
    run_files = {key: cells * size for key, size in RUN_CELL_BYTES.items()}
    run_files["lattice floor curve"] = columns * FLOOR_BYTES_PER_COLUMN
    run_files["JSON files and receipts (native allowance, once per run)"] = json_bytes
    per_frame, per_run = sum(files.values()), sum(run_files.values())
    native = pixels * NATIVE_TRANSIENT_BYTES_PER_PIXEL
    estimate = dict(
        format=ESTIMATE_FORMAT, estimate=True, label=LATTICE_ESTIMATE_MODEL, frames=len(cases),
        lattice_cells=cells, stored_cells=stored_shape[0] * stored_shape[1], declared_canvas_pixels=pixels,
        frame_files=files, run_files=run_files,
        native_transient=dict(bytes=native, bytes_per_canvas_pixel=NATIVE_TRANSIENT_BYTES_PER_PIXEL,
                              location="local transient folder, one frame at a time"),
        compression="none assumed", json_allowance_source=documented["source"])
    return LatticeEstimate(name(task["run"]), len(cases), PIPELINE, precision, request[RESOLUTION_KEY],
                           tuple(lattice.shape), lattice.resolution_mm if layer is None else layer.resolution_mm,
                           stored_shape, pixels, per_frame, per_run, native, per_frame + native, estimate)
