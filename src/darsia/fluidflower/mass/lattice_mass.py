"""Mass on the computing lattice through the unchanged N1 code (lattice-phase-n1).

A lattice mass template replaces each grid-bound shared input of the run's
native template by its lattice coarsening, written once per run:

* the physical reference: facies by area majority over physical-domain pixels,
  the domain mask, porosity and depth means, placed by N1's reference-to-metre
  matrix (facies also through the measured physical correction, as N1 places
  them);
* clear water: area mean over its supported pixels;
* the floor curve, transported to lattice columns;
* an identity physical correction and the lattice cell-to-metre matrix.

Per frame, lattice records hold the current and baseline log RGB, the lattice
phase products, the photographed top, the carried original domain and the
inner rig polygon. ``N1Pipeline.load_frame`` and ``_prepare_statistics`` then
run unchanged on the lattice with identity registration.

The box masks are the native box masks coarsened. The template's ROI scale
the box to the image dimensions, so on a lattice image they would select
another region of the rig than on the native canvas.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from ..computing_lattice import CellAssignment

LATTICE_MASS_FORMAT = "darsia-lattice-n1-template-v1"


def _projected(matrix, shape_hw):
    """Float32 metre coordinates of a raster's pixel centres, as N1 casts them."""
    from .n1_geometry import project_points
    rows, columns = map(int, shape_hw)
    x, y = project_points(matrix, np.arange(columns, dtype=np.float64)[None, :],
                          np.arange(rows, dtype=np.float64)[:, None])
    return x.astype(np.float32), y.astype(np.float32)


def raster_assignment(lattice, matrix, shape_hw):
    """Cells of a raster whose pixel (column, row) maps to metres by ``matrix``."""
    from ..pipeline.lattice_inputs import jacobian_area
    x, y = _projected(matrix, shape_hw)
    return CellAssignment(lattice, x, y, jacobian_area(matrix, shape_hw))


def physical_on_lattice(physical, reference_matrix, correction, lattice):
    """Coarsen N1's physical reference onto the lattice.

    ``physical`` holds facies, domain, porosity and depth_m on the reference native
    grid. Facies are placed through ``correction`` as N1 transports them;
    porosity and depth are means over physical-domain pixels.
    """
    domain = np.asarray(physical["domain"], bool)
    shape = domain.shape
    plain = raster_assignment(lattice, reference_matrix, shape)
    cells = plain.mask(domain)
    result = dict(domain=cells)
    for name in ("porosity", "depth_m"):
        values = np.asarray(physical[name], np.float64)
        supported = domain & np.isfinite(values) & (values > 0)
        mean, _ = plain.mean(np.where(supported, values, 0.0), supported)
        result[name] = np.where(cells & np.isfinite(mean), mean, 0.0).astype(np.asarray(physical[name]).dtype)
    correction = np.asarray(correction, np.float64)
    placed = plain if np.array_equal(correction, np.eye(3)) else raster_assignment(
        lattice, np.asarray(reference_matrix, np.float64) @ correction, shape)
    facies = np.asarray(physical["facies"])
    result["facies"] = placed.majority(facies, where=domain).astype(facies.dtype)
    return result


def clear_water_on_lattice(clear_rgb, clear_valid, reference_to_grid, reference_matrix, lattice):
    """Coarsen clear-water RGB onto the lattice with its supported fraction.

    The clear-water array is indexed in the native coordinates of its own
    frame; ``reference_to_grid`` maps reference native coordinates there.
    """
    rgb = np.asarray(clear_rgb, np.float32)
    valid = np.asarray(clear_valid, bool) & np.isfinite(rgb).all(-1) & (rgb > 0).all(-1)
    matrix = np.asarray(reference_matrix, np.float64) @ np.linalg.inv(np.asarray(reference_to_grid, np.float64))
    assignment = raster_assignment(lattice, matrix, valid.shape)
    result = np.zeros((*lattice.shape, 3), np.float32)
    supported = None
    for channel in range(3):
        mean, _ = assignment.mean(np.where(valid, rgb[..., channel], 0.0).astype(np.float64), valid)
        usable = np.isfinite(mean) & (mean > 0)
        supported = usable if supported is None else supported & usable
        result[..., channel] = np.where(usable, mean, 0.0).astype(np.float32)
    result[~supported] = 0
    return result, supported


def floor_on_lattice(floor, floor_origin_xy, reference_matrix, lattice):
    """The frozen floor curve as a row per lattice column; NaN where unobserved."""
    from .n1_geometry import transport_curve
    return transport_curve(np.asarray(floor, np.float64),
                           H_source_to_target=lattice.metric_to_cell() @ np.asarray(reference_matrix, np.float64),
                           source_origin_xy=floor_origin_xy, target_origin_xy=(0, 0),
                           target_width=lattice.shape[1])


def hydrostatic_dimensions(native_dimensions_m, canvas_shape_hw, canvas_matrix, lattice):
    """Lattice image dimensions that keep N1's hydrostatic depth per metre of rig.

    N1 gives image row r the water depth ``dimensions[0] * r / (rows - 1)``.
    On the native canvas that is a nominal depth per canvas row; the lattice
    gets the same depth per metre of rig, so pressure, gas density and
    solubility change with height as on the native canvas. What remains is a
    constant offset: the native depth at the centre of the lattice's first row,
    returned in metres for the provenance. The second dimension follows the
    same rule; N1 uses it for nominal areas only, which cancel.
    """
    from ..computing_lattice import canvas_pitch_m
    rows, columns = map(int, canvas_shape_hw)
    pitch = canvas_pitch_m(canvas_matrix, canvas_shape_hw)
    native = [float(value) for value in native_dimensions_m]
    per_metre = (native[0] / ((rows - 1) * pitch), native[1] / ((columns - 1) * pitch))
    lattice_rows, lattice_columns = lattice.shape
    dimensions = [per_metre[0] * (lattice_rows - 1) * lattice.spacing_m,
                  per_metre[1] * (lattice_columns - 1) * lattice.spacing_m]
    first = lattice.cell_to_metric() @ np.array([(lattice_columns - 1) / 2, 0.0, 1.0])
    canvas = np.linalg.inv(np.asarray(canvas_matrix, np.float64)) @ first
    offset = native[0] * (canvas[1] / canvas[2]) / (rows - 1)
    return dimensions, dict(depth_per_metre_of_rig=per_metre[0], first_row_native_depth_m=float(offset),
                            rule="dimensions[0] * r / (rows - 1) keeps the native depth per metre of rig")


def native_box_masks(boxes, canvas_shape_hw, dimensions_m):
    """The native box1/box2 masks on a canvas, exactly as N1 evaluates them."""
    import darsia
    image = darsia.ScalarImage(np.zeros(tuple(map(int, canvas_shape_hw)), np.float32),
                               dimensions=list(dimensions_m))
    masks = {}
    for name in ("box1", "box2"):
        roi = boxes[name]
        roi = roi.get("roi", roi) if isinstance(roi, dict) else roi
        masks[name] = np.asarray(darsia.roi_to_mask(darsia.CoordinateArray(roi), image, mode="voxels").img, bool)
    return masks


def lattice_record(directory, name, log, valid, lattice, *, fields, provenance):
    """Write lattice log RGB as a native-format record that compose_raw reads.

    The whole lattice is the record's centre at origin zero, without ring
    windows; its registration is the identity.
    """
    from .n1_pipeline import asset
    from ..native.sidecar import write_log_linear
    directory = Path(directory)
    log_path, valid_path = directory / (name + "_log.npy"), directory / (name + "_valid.npy")
    write_log_linear(log_path, np.asarray(log, np.float32))
    write_log_linear(valid_path, np.asarray(valid, bool))
    shape = list(lattice.shape)
    return dict(phase_shape_hw=shape, native_origin_xy=[0, 0],
                native_centre=dict(native_shape_hw=shape, log_linear_rgb=asset(log_path), valid_mask=asset(valid_path)),
                ring_windows=[], binding=dict(geometry={fields["source_to_native"]: np.eye(3).tolist()}),
                lattice=dict(provenance))


class LatticeMass:
    """A run's lattice N1 template and the frame-level lattice mass records.

    ``native`` is the run's native ``N1Pipeline``; its loaded physical
    reference, clear water, floor, boxes and matrices are coarsened once.
    """

    def __init__(self, native, lattice, directory, *, canvas_shape_hw, canvas_matrix, precision, pixel_math,
                 unc_only):
        """Write the lattice template DATA and load an N1Pipeline on it.

        ``canvas_shape_hw`` and ``canvas_matrix`` describe the run's native
        canvas and its pixel-to-metre mapping; they fix the lattice image
        dimensions (:func:`hydrostatic_dimensions`). Only the native reader,
        boxes and dimensions are kept, so the caller may release the native
        pipeline and its full-resolution arrays.
        """
        from .n1_pipeline import N1Pipeline, _write_json, _write_npz
        self.lattice = lattice
        self.reader, self.boxes = native.reader, native.boxes
        self.native_dimensions_m = list(native.config["dimensions_m"])
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        config = native.config
        self.fields = dict(config["geometry_fields"])
        shared = config["shared"]
        physical = physical_on_lattice(native.physical, native.reference_matrix, native.correction, lattice)
        physical_asset = _write_npz(self.directory / "lattice_physical_reference.npz", physical)
        clear_rgb, clear_valid = clear_water_on_lattice(
            native.clear_rgb, native.clear_valid, shared["clearwater"]["H_reference_to_grid"],
            native.reference_matrix, lattice)
        from .n1_pipeline import asset
        from ..native.sidecar import write_log_linear
        rgb_path, valid_path = self.directory / "lattice_clearwater_rgb.npy", self.directory / "lattice_clearwater_valid.npy"
        write_log_linear(rgb_path, clear_rgb)
        write_log_linear(valid_path, clear_valid)
        floor = floor_on_lattice(native.floor, shared["floor_origin_xy"], native.reference_matrix, lattice)
        floor_path = self.directory / "lattice_floor_curve.npy"
        write_log_linear(floor_path, floor)
        correction = _write_json(self.directory / "lattice_physical_correction.json",
                                 dict(correction_matrix=np.eye(3).tolist(),
                                      native_physical_correction=config["physical_correction"],
                                      role="identity on the lattice; the native correction placed the lattice facies"))
        lattice_shared = dict(deepcopy(shared), physical_reference=physical_asset,
                              reference_matrix=lattice.cell_to_metric().tolist(),
                              floor_curve=asset(floor_path), floor_origin_xy=[0, 0],
                              clearwater=dict(deepcopy(shared["clearwater"]), rgb=asset(rgb_path),
                                              valid=asset(valid_path), H_reference_to_grid=np.eye(3).tolist()))
        dimensions, hydrostatics = hydrostatic_dimensions(config["dimensions_m"], canvas_shape_hw, canvas_matrix,
                                                          lattice)
        self.config = dict(deepcopy(config), shared=lattice_shared, physical_correction=correction,
                           dimensions_m=dimensions,
                           lattice=dict(format=LATTICE_MASS_FORMAT, lattice=lattice.metadata(),
                                        native_dimensions_m=list(config["dimensions_m"]),
                                        native_canvas_shape_hw=list(map(int, canvas_shape_hw)),
                                        hydrostatics=hydrostatics,
                                        native_physical_correction=config["physical_correction"]))
        self.config.pop("pixel_domain", None)
        self.config_asset = _write_json(self.directory / "lattice_mass_template.json",
                                        {key: value for key, value in self.config.items()
                                         if key not in {"frames", "data_aliases"}})
        self._canvas_boxes = {}
        self.pipeline = N1Pipeline(self.config, precision=precision, pixel_math=pixel_math, unc_only=unc_only)

    def box_masks(self, assignment, canvas_shape_hw):
        """The native box masks of a canvas, coarsened with this frame's cells."""
        shape = tuple(map(int, canvas_shape_hw))
        if shape not in self._canvas_boxes:
            self._canvas_boxes[shape] = native_box_masks(self.boxes, shape, self.native_dimensions_m)
        return {name: assignment.mask(mask) for name, mask in self._canvas_boxes[shape].items()}

    def frame_spec(self, spec, directory, *, current, baseline, phase, assignment, canvas_matrix, native_record):
        """The lattice N1 frame spec of one acquisition.

        ``spec`` is the selected frame of the run configuration; ``current``
        and ``baseline`` are lattice records; ``phase`` holds the lattice phase
        descriptors (labels, classes, area). ``assignment`` and
        ``canvas_matrix`` are this frame's cells and canvas-to-metre matrix;
        ``native_record`` is its native full record.
        """
        from PIL import Image
        from .n1_geometry import _carried_domain
        from .n1_pipeline import asset, _write_npz
        from ..pipeline.io_ops import publish_file
        directory = Path(directory)
        mass = deepcopy(spec["mass_spec"])
        reader = self.reader
        canvas_shape = assignment.source_shape
        top = dict(mass["photo_top"])
        top.setdefault("array_key", "keep")
        keep = np.asarray(reader.array(top), bool)
        top_asset = dict(_write_npz(directory / "lattice_photo_top.npz", dict(keep=assignment.mask(keep))),
                         array_key="keep")
        origin = native_record["native_origin_xy"]
        current_matrix = np.array([[1.0, 0.0, -float(origin[0])], [0.0, 1.0, -float(origin[1])], [0.0, 0.0, 1.0]]) @ \
            np.asarray(native_record["binding"]["geometry"][self.fields["source_to_native"]], np.float64)
        carried, carried_proof = _carried_domain(mass["original_domain"], current_matrix, canvas_shape,
                                                 reader, fields=self.fields)
        membership = np.where(assignment.mask(carried), 0, 255).astype(np.uint8)
        domain_path = directory / "lattice_original_domain.png"
        publish_file(domain_path, lambda stream: Image.fromarray(membership).save(stream, format="PNG"))
        replaced = {"parent_labels_map", "native_shape_hw", self.fields["original_domain_to_native"]}
        original_domain = {key: value for key, value in mass["original_domain"].items() if key not in replaced}
        original_domain.update(parent_labels_map=asset(domain_path), native_shape_hw=list(self.lattice.shape),
                               **{self.fields["original_domain_to_native"]: np.eye(3).tolist()},
                               native_carried_domain=carried_proof)
        polygon = np.asarray(mass["inner_polygon_native"], np.float64) - np.asarray(origin, np.float64)
        from .n1_geometry import project_points
        x, y = project_points(self.lattice.metric_to_cell() @ np.asarray(canvas_matrix, np.float64),
                              polygon[:, 0], polygon[:, 1])
        mass.update(photo_top=top_asset, original_domain=original_domain,
                    inner_polygon_native=np.column_stack([x, y]).tolist(),
                    case_id=spec["case_id"], frame_index=spec["frame_index"], elapsed_h=spec["elapsed_h"],
                    source_binding=deepcopy(spec["source"]["source_binding"]),
                    current=current, baseline=baseline,
                    labels=phase["labels"], classes=phase["classes"], area=phase["area"])
        return mass

    def evaluate(self, spec, box_masks, *, output_capture=None):
        """N1Pipeline.evaluate on the lattice, with the coarsened native box masks.

        The same library calls run in the same order; only the box masks come
        from the native canvas instead of the template ROI.
        """
        from darsia._opencv_runtime import scientific_thread, completed_frame
        from .n1_arithmetic import evaluate_main, main_products
        from .n1_statistics import exact_prediction
        from ..storage_profile import stores_colour_coordinate
        pipeline = self.pipeline
        ipp_before = scientific_thread("LatticeMass.evaluate:before")
        capture = dict(colour_coordinate=True) if stores_colour_coordinate(pipeline.config) else {}
        prepared = pipeline._prepare_statistics(spec, **capture)
        frame, response = prepared["frame"], prepared["response"]
        masks = {"full_rig": frame.domain}
        for name in ("box1", "box2"):
            mask = np.asarray(box_masks[name], bool)
            if mask.shape != frame.domain.shape:
                raise ValueError("Lattice box masks differ from the lattice grid")
            masks[name] = mask
        config = pipeline.config
        result = evaluate_main(response, config["anchor"], parameters=pipeline.parameters, pixel_math=pipeline.math,
                               calibration_sha256=config["calibration"]["sha256"], domain_masks=masks)
        if output_capture is not None:
            output_capture.update(frame=frame, x_m=prepared["x_m"], y_m=prepared["y_m"],
                                  volume_m3=response["analysis"].geometry.cached_voxel_volume)
        prediction = exact_prediction(prepared["statistics"], result, pipeline.parameters, config["anchor"])
        evaluated = dict(response=response, result=result, prediction=prediction,
                         export=main_products(result, pipeline.math), statistics=prepared["statistics"],
                         sandmap=prepared["sandmap"])
        self.last_opencv_ipp = completed_frame(ipp_before, "LatticeMass.evaluate:after")
        return evaluated
