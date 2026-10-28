"""Lattice inputs for the unchanged Phase-map chain (pipeline lattice-phase-n1).

:class:`LatticeRun` coarsens the native inputs that ``series.SeriesInputs``
prepares onto the computing lattice, so that ``phase.infer.infer_labels`` runs
unchanged on lattice arrays:

* every acquisition is mapped to metres by its own frame geometry, exactly as
  N1 maps it (``n1_geometry.frame_coordinates``), so all lattice images are
  registered and every homography passed to the chain is the identity;
* log RGB is averaged through its linear values with area weights; masks take
  the cells with at least half of their area inside; categories take the area
  majority; pixel areas are summed;
* the native centre becomes one fixed lattice box per run;
* pixel-sized parameters come from ``physical_units.parameters_on_grid`` with
  the run's native pixels per cell, and the floor curve and zones reach the
  lattice through the reference-to-lattice homography inside the chain.

No phase function is copied or changed here.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import numpy as np

from ..computing_lattice import CellAssignment, MASK_FRACTION, canvas_pitch_m

LATTICE_INPUT_FORMAT = "darsia-lattice-phase-inputs-v1"


def canvas_to_metric(reference_matrix, geometry, native_origin_xy, *, fields):
    """Matrix from canvas pixel coordinates to metres, as N1's frame coordinates.

    ``frame_coordinates`` evaluates ``reference_matrix @ inv(H)`` at canvas
    columns and rows shifted by the native origin; this is that composition.
    """
    ox, oy = map(float, native_origin_xy)
    shift = np.array([[1.0, 0.0, ox], [0.0, 1.0, oy], [0.0, 0.0, 1.0]])
    homography = np.asarray(geometry[fields["source_to_native"]], np.float64)
    return np.asarray(reference_matrix, np.float64) @ np.linalg.inv(homography) @ shift


def jacobian_area(matrix, shape_hw):
    """Metric area of every pixel of a raster under a projective matrix.

    For (x, y) -> (u/w, v/w) the Jacobian determinant is det(M) / w**3.
    """
    matrix = np.asarray(matrix, np.float64)
    rows, columns = map(int, shape_hw)
    w = (matrix[2, 0] * np.arange(columns, dtype=np.float64)[None, :]
         + matrix[2, 1] * np.arange(rows, dtype=np.float64)[:, None] + matrix[2, 2])
    if np.any(np.abs(w) < 1e-12):
        raise ValueError("The raster maps to infinity")
    return np.abs(np.linalg.det(matrix) / w ** 3)


def acquisition_assignment(lattice, reference_matrix, record, *, fields):
    """Cells of one acquisition's canvas and its canvas-to-metre matrix.

    ``record`` is a native ``full`` record: its ``binding.geometry`` and
    ``native_origin_xy`` place the canvas, ``phase_shape_hw`` sizes it.
    """
    from ..mass.n1_geometry import frame_coordinates
    geometry = record["binding"]["geometry"]
    shape = tuple(map(int, record["phase_shape_hw"]))
    revision = dict(phase_shape_hw=list(shape), native_origin_xy=list(record["native_origin_xy"]))
    x, y = frame_coordinates(reference_matrix, geometry, revision, fields=fields)
    matrix = canvas_to_metric(reference_matrix, geometry, record["native_origin_xy"], fields=fields)
    return CellAssignment(lattice, x, y, jacobian_area(matrix, shape)), matrix


def embed(values, source_box_xywh, canvas_shape_hw, fill):
    """Place a native-centre array (H, W) or (C, H, W) on its canvas."""
    values = np.asarray(values)
    x, y, width, height = map(int, source_box_xywh)
    leading = values.shape[:-2]
    if values.shape[-2:] != (height, width):
        raise ValueError("Native centre array differs from its source box")
    result = np.full((*leading, *map(int, canvas_shape_hw)), fill, values.dtype)
    result[..., y:y + height, x:x + width] = values
    return result


def centre_box(assignment, source_box_xywh, canvas_shape_hw):
    """The lattice box (x, y, width, height) around the native centre.

    It bounds every cell with at least half of its area inside the native
    centre rectangle. Cells of the box outside the centre stay invalid in the
    centre input, because the centre input is NaN there.
    """
    inside = embed(np.ones(tuple(int(v) for v in source_box_xywh[3:1:-1]), bool),
                   source_box_xywh, canvas_shape_hw, False)
    cells = assignment.fraction(inside) >= MASK_FRACTION
    if not cells.any():
        raise ValueError("The native centre covers no lattice cell")
    rows, columns = np.flatnonzero(cells.any(1)), np.flatnonzero(cells.any(0))
    return (int(columns[0]), int(rows[0]), int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1))


def crop(values, box):
    """The lattice box of an (H, W) or (C, H, W) lattice array, copied."""
    x, y, width, height = box
    return np.asarray(values)[..., y:y + height, x:x + width].copy()


def lattice_rules(parameters, *, pixels_per_cell, box):
    """Phase parameters for the lattice: converted rule sizes and box geometry.

    The geometry entries describe the lattice: its centre box, the native
    shape of that box and the matching canvas origin. ``floor_source_origin_xy``
    keeps describing the frozen reference floor curve.
    """
    from ..phase.physical_units import parameters_on_grid
    x, y, width, height = box
    rules = parameters_on_grid(parameters, pixels_per_cell=pixels_per_cell,
                               feature_shape_hw=(height, width))
    geometry = rules["geometry"]
    geometry.update(native_shape_hw=[height, width], source_box_xywh=[x, y, width, height],
                    rig_origin_xy=[-x, -y], source_origin_xy=[0, 0])
    return rules


def reference_to_lattice_native(lattice, reference_matrix, box):
    """Homography from reference native coordinates to lattice-box coordinates.

    The chain transports the frozen floor curve and zones with this matrix;
    the box origin plays the role of the native centre origin.
    """
    x, y = box[:2]
    shift = np.array([[1.0, 0.0, -x], [0.0, 1.0, -y], [0.0, 0.0, 1.0]])
    return shift @ lattice.metric_to_cell() @ np.asarray(reference_matrix, np.float64)


class LatticeAcquisition:
    """One acquisition's lattice images: canvas, masked centre and temporal input."""

    def __init__(self, assignment, box, *, full_log, full_valid, centre, source_box_xywh, provenance):
        """Coarsen the canvas and the masked native centre of one acquisition.

        ``centre`` is the acquisition's ``MaskedNativeInput`` on the native
        centre grid at ``source_box_xywh`` of the canvas.
        """
        from ..phase.inputs import MaskedNativeInput, TemporalNativeInput
        canvas_shape = assignment.source_shape
        self.full_log, self.full_valid, self.full_fraction = assignment.log_rgb(full_log, full_valid)
        centre_log = embed(centre.log, source_box_xywh, canvas_shape, np.float32(np.nan))
        centre_valid = embed(centre.valid, source_box_xywh, canvas_shape, False)
        log, valid, fraction = assignment.log_rgb(centre_log, centre_valid)
        del centre_log, centre_valid
        self.box = tuple(box)
        self.centre_fraction = crop(fraction, box)
        log, valid = crop(log, box), crop(valid, box)
        for array in (self.full_log, self.full_valid, log, valid):
            array.setflags(write=False)
        record = dict(provenance, lattice_box_xywh=list(box), mask_fraction=MASK_FRACTION,
                      coarsening="area-weighted linear mean over valid pixels, then log")
        self.native_input = MaskedNativeInput(log, valid, dict(centre.provenance, lattice=record))
        self.temporal_input = TemporalNativeInput(log, valid, (0, 0), dict(lattice=record))


class LatticeRun:
    """One run's lattice, rules, baseline, predecessor and per-frame coarsening.

    ``inputs`` is the run's ``series.SeriesInputs``. ``reference_matrix`` and
    ``fields`` are N1's reference-to-metre matrix and geometry field names.
    """

    def __init__(self, config, inputs, lattice, *, reference_matrix, fields):
        """Fix the lattice box, pixels per cell, rules and lattice baseline."""
        from ..pipeline import phase_inputs as loaders
        from ..phase import features
        self.config, self.lattice = config, lattice
        self.reference_matrix = np.asarray(reference_matrix, np.float64)
        self.fields = dict(fields)
        parameters = inputs.parameters
        self.source_box = tuple(map(int, parameters["geometry"]["source_box_xywh"]))
        reference = config["phase"]["reference"]["full"]
        assignment, matrix = self.assignment(reference)
        self.reference_canvas_matrix = matrix
        self.canvas_pitch_m = canvas_pitch_m(matrix, tuple(map(int, reference["phase_shape_hw"])))
        self.pixels_per_cell = lattice.spacing_m / self.canvas_pitch_m
        self.box = centre_box(assignment, self.source_box, assignment.source_shape)
        self.rules = lattice_rules(parameters, pixels_per_cell=self.pixels_per_cell, box=self.box)
        self.reference_to_lattice = reference_to_lattice_native(lattice, self.reference_matrix, self.box)
        baseline = inputs.model_bundle["baseline"]
        self.baseline = LatticeAcquisition(
            assignment, self.box, full_log=baseline["full_log"], full_valid=baseline["full_valid"],
            centre=baseline["native_input"], source_box_xywh=self.source_box,
            provenance=dict(role="baseline", source_sha256=config["phase"]["reference"]["source_sha256"]))
        del assignment
        native = dict(inputs.model_bundle["native"])
        p = self.rules
        if native.get("fixed_baseline_decode_cache") is not None:
            # The native cache holds native arrays; the lattice gets its own.
            native["fixed_baseline_decode_cache"] = loaders.FixedBaselineDecodeCache(
                max_bytes=p["fixed_baseline_cache"]["maximum_bytes"], parameters=p["fixed_baseline_cache"])
            native["fixed_baseline_decoder"] = partial(features.decode_native_log, parameters=p["features"])
        native["flatfield_reference"] = features.prepare_flatfield_reference(
            self.baseline.native_input.log, source_valid_mask=self.baseline.native_input.valid,
            tile_rows=p["native"]["block_rows"], parameters=p["features"])
        self.native_bundle = native
        self.previous = self.acquisition_previous(
            inputs.previous, config["phase"]["initial_previous"]["full"], role="initial_previous")

    def assignment(self, record):
        """Cells of an acquisition's canvas under this run's lattice."""
        return acquisition_assignment(self.lattice, self.reference_matrix, record, fields=self.fields)

    def acquisition_previous(self, previous, record, *, role):
        """A lattice predecessor from a native predecessor and its full record."""
        assignment, _ = self.assignment(record)
        temporal = previous["temporal_input"]
        centre = SimpleNamespace(log=temporal.log, valid=temporal.valid, provenance=temporal.provenance)
        if tuple(temporal.origin_xy) != (0, 0):
            pad = -int(temporal.origin_xy[0])
            centre = SimpleNamespace(log=temporal.log[:, pad:-pad, pad:-pad],
                                     valid=temporal.valid[pad:-pad, pad:-pad],
                                     provenance=temporal.provenance)
        acquisition = LatticeAcquisition(
            assignment, self.box, full_log=previous["full_log"], full_valid=previous["full_valid"],
            centre=centre, source_box_xywh=self.source_box,
            provenance=dict(role=role, source=previous["source"]))
        return self.lattice_predecessor(previous["source"], acquisition)

    @staticmethod
    def lattice_predecessor(source, acquisition):
        """The predecessor mapping that infer_labels reads, on the lattice."""
        return dict(source=source, full_log=acquisition.full_log, full_valid=acquisition.full_valid,
                    current_h=np.eye(3), temporal_input=acquisition.temporal_input)

    def frame(self, spec, native, frame):
        """The lattice frame for infer_labels, its assignment and lattice extras.

        ``native`` and ``frame`` are this acquisition's ``SeriesInputs.native``
        and ``SeriesInputs.frame`` results on the native grids.
        """
        from ..phase.inputs import MaskedNativeInput
        if frame.get("prior_cache") is not None or frame.get("encoding_contract") is not None:
            raise ValueError("Lattice computing needs fresh inference; retained native caches are native-only")
        assignment, matrix = self.assignment(native["record"]["full"])
        canvas_shape = assignment.source_shape
        acquisition = LatticeAcquisition(
            assignment, self.box, full_log=frame["full_log"], full_valid=frame["full_valid"],
            centre=frame["native_input"], source_box_xywh=self.source_box,
            provenance=dict(role="current", source=spec["source"]))
        context = native["context"]
        rig = assignment.mask(np.asarray(context.rig_domain, bool))
        hardware = assignment.mask(np.asarray(context.hardware_excluded, bool)) & ~rig
        photo = crop(assignment.mask(embed(frame["photo_sand"], self.source_box, canvas_shape, False)), self.box)
        components = frame["components"]
        open_components = assignment.majority(np.asarray(components["open_components"]))
        barrier = assignment.mask(np.asarray(components["barrier_mask"], bool)) & (open_components == 0)
        scale = self.pixels_per_cell
        lattice_frame = dict(frame)
        lattice_frame.update(
            full_log=acquisition.full_log, full_valid=acquisition.full_valid,
            native_input=acquisition.native_input, temporal_input=acquisition.temporal_input,
            photo_sand=photo, current_h=np.eye(3),
            probability_spacing_m=[float(v) * scale for v in frame["probability_spacing_m"]],
            inherited_spacing_m=[float(v) * scale for v in frame["inherited_spacing_m"]],
            context=SimpleNamespace(rig_domain=rig, hardware_excluded=hardware,
                                    parameters=dict(native=context.parameters, lattice=self.lattice.metadata())),
            observed=assignment.mask(np.asarray(frame["observed"], bool)),
            top_keep=assignment.mask(np.asarray(frame["top_keep"], bool)),
            original_area=positive_cells(assignment.total(frame["original_area"])),
            pixel_area_mm2=positive_cells(assignment.total(frame["pixel_area_mm2"])),
            components=dict(components, open_components=open_components, barrier_mask=barrier),
            reference_to_current_h=self.reference_to_lattice,
            input_provenance=dict(frame["input_provenance"], lattice=self.provenance()),
        )
        if not isinstance(lattice_frame["native_input"], MaskedNativeInput):
            raise TypeError("Lattice centre input must remain a masked native input")
        return lattice_frame, assignment, matrix

    def bundle(self, inputs, assignment):
        """The model bundle with lattice baseline, caches and this frame's envelopes."""
        bundle = dict(inputs.model_bundle)
        envelopes = dict(bundle["envelopes"])
        envelopes.update(env1=assignment.mask(np.asarray(envelopes["env1"], bool)),
                         env2=assignment.mask(np.asarray(envelopes["env2"], bool)))
        bundle.update(native=self.native_bundle, envelopes=envelopes,
                      baseline=dict(native_input=self.baseline.native_input,
                                    temporal_input=self.baseline.temporal_input,
                                    full_log=self.baseline.full_log, full_valid=self.baseline.full_valid))
        return bundle

    def advance(self, lattice_frame):
        """Make a committed selected (or baseline) lattice frame the next predecessor."""
        from ..phase.history import advance_previous
        source = lattice_frame["source"]
        current = dict(source=source, full_log=lattice_frame["full_log"], full_valid=lattice_frame["full_valid"],
                       current_h=np.eye(3), temporal_input=lattice_frame["temporal_input"])
        self.previous = advance_previous(current, self.previous, selected=source["selected"],
                                         baseline=source.get("baseline", False))

    def provenance(self):
        """The run's lattice facts recorded with every lattice frame."""
        return dict(format=LATTICE_INPUT_FORMAT, lattice=self.lattice.metadata(),
                    canvas_pitch_m=self.canvas_pitch_m, native_pixels_per_cell=self.pixels_per_cell,
                    native_source_box_xywh=list(self.source_box), lattice_box_xywh=list(self.box),
                    registration="every acquisition mapped to metres by its own frame geometry; identity homographies",
                    reference_to_lattice_native=self.reference_to_lattice.tolist(),
                    coarsening=dict(log_rgb="area-weighted linear mean over valid pixels, then log",
                                    masks="cells with at least half of their area inside",
                                    categories="area majority, lowest id on ties",
                                    areas="sum of native pixel areas; uncovered cells take the median covered cell",
                                    weights="metric pixel area from each acquisition's projective mapping"))


def positive_cells(areas):
    """Give lattice cells without any source pixel the median covered-cell area.

    Such cells lie outside the photographed rig; the phase and mass grids need
    a positive area in every cell, and no mass is ever assigned to them.
    """
    areas = np.asarray(areas, np.float64).copy()
    covered = areas > 0
    if not covered.any():
        raise ValueError("No lattice cell holds a source pixel")
    areas[~covered] = float(np.median(areas[covered]))
    return areas
