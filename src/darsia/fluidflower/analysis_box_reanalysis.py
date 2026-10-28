"""Integrate new physical boxes from checked, already saved mass maps.

No image, phase or calibration producer is invoked. Results are a separate
immutable derivation; the production job and its original products stay intact.
An option (off by default) also integrates each sand layer (facies) of the
saved native maps, into a table of its own next to each box table.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

FORMAT = "darsia-saved-box-mass-analysis-v1"
COMPONENTS = ("total", "aqueous", "gas")
NATIVE_FIELDS = dict(total="mass_total", aqueous="mass_aqueous", gas="mass_free_gas")
CELL_FIELDS = dict(total="cell_mass_total_g", aqueous="cell_mass_aqueous_g", gas="cell_mass_gas_g")
LAYER_COLUMNS = ["run", "case_id", "elapsed_h", "domain_id", "name", "facies_id", "complete", "sample_count",
                 "unknown_sample_count"] + [key + suffix for key in COMPONENTS
                                            for suffix in ("_mass_g", "_quantified_mass_g")]


class MapsUnavailable(ValueError):
    """A saved representation lacks the fields needed for this integration."""


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _pin(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size)


def _boxes(configuration):
    from .metric_analysis_domains import FORMAT as BOX_FORMAT, box
    if (not isinstance(configuration, dict) or configuration.get("format") != BOX_FORMAT
            or configuration.get("units") != "mm"
            or configuration.get("x_axis") != "right" or configuration.get("y_axis") != "up"):
        raise ValueError("Saved measured boxes must declare millimetres, x right and y up")
    rows, ids, names = [], set(), set()
    for value in configuration.get("domains", []):
        checked = box(value["name"], value["bounds_mm"], role=value.get("role", ""),
                      domain_id=value.get("domain_id"), enabled=value["enabled"])
        if checked["domain_id"] in ids or checked["name"] in names:
            raise ValueError("Analysis box names and identities must be unique")
        ids.add(checked["domain_id"])
        names.add(checked["name"])
        if checked["enabled"]:
            rows.append(checked)
    if not rows:
        raise ValueError("At least one enabled named analysis box is required")
    return rows


def integrate_mass_arrays(masses, x_m, y_m, boxes, *, weights_l=None,
                          unknown_mask=None, storage="native", resolution_mm=None):
    """Original float64 pixel integral, or sum of already extensive cell grams.

    Coordinates are saved physical pixel/cell centres. Bounds are half-open.
    Unknown mass is reported separately, never silently replaced by zero.
    """
    x, y = np.asarray(x_m), np.asarray(y_m)
    if (x.ndim != 2 or x.shape != y.shape or not np.isfinite(x).all()
            or not np.isfinite(y).all() or set(masses) != set(COMPONENTS)):
        raise ValueError("Mass arrays need finite physical coordinates on one two-dimensional grid")
    arrays = {key: np.asarray(value) for key, value in masses.items()}
    if any(value.shape != x.shape or value.dtype.kind != "f" or np.isinf(value).any()
           or np.any(value[np.isfinite(value)] < 0) for value in arrays.values()):
        raise ValueError("Saved masses must be nonnegative floating arrays on the coordinate grid")
    if storage == "native":
        weights = np.asarray(weights_l)
        if (weights.shape != x.shape or weights.dtype.kind != "f"
                or not np.isfinite(weights).all() or np.any(weights < 0)):
            raise ValueError("Native mass density requires saved finite pore-volume weights in litres")
    elif storage == "metric_cells":
        weights = np.ones(x.shape, np.float64)
    else:
        raise ValueError("Use native mass density or stored metric-cell mass")
    unknown = np.zeros(x.shape, bool) if unknown_mask is None else np.asarray(unknown_mask)
    if unknown.shape != x.shape or unknown.dtype != bool:
        raise ValueError("Unknown support must be an explicit boolean mask on the mass grid")
    unknown = unknown | np.logical_or.reduce([~np.isfinite(value) for value in arrays.values()])
    rows = []
    for box in boxes:
        x0, x1, y0, y1 = np.asarray(box["bounds_mm"], np.float64) / 1000.
        selected = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        missing = selected & unknown
        valid = selected & ~unknown
        covered = bool(selected.any())
        row = dict(domain_id=box["domain_id"], name=box["name"], bounds_mm=box["bounds_mm"],
                   sample_count=int(selected.sum()), unknown_sample_count=int(missing.sum()),
                   complete=covered and not missing.any(), storage=storage, resolution_mm=resolution_mm)
        for key, value in arrays.items():
            # DarSIA Geometry integrates each spatial axis in order with float64
            # weights. Retain that reduction, excluding other pixels first.
            integral = np.where(valid, value, 0.).astype(np.float64) * weights.astype(np.float64)
            amount = float(integral.sum(axis=0, dtype=np.float64).sum(axis=0, dtype=np.float64))
            row[key + "_quantified_mass_g"] = amount
            row[key + "_mass_g"] = amount if row["complete"] else None
        rows.append(row)
    return rows


def integrate_facies_arrays(masses, facies, *, weights_l, unknown_mask, names=None):
    """Grams of each component in each facies (sand layer) of one native frame.

    ``masses`` holds the density maps in g/L, ``facies`` each pixel's
    non-negative integer facies id and ``weights_l`` the saved pore volumes in
    litres, all on one grid. As for the boxes, unknown support (``unknown_mask``
    or a non-finite mass) leaves a facies' total undefined and its quantified
    subtotal is kept. Facies without pore volume hold no mass and are left out.
    Rows are named from ``names`` (facies id -> layer name), else "Facies <id>".
    """
    ids = np.asarray(facies)
    if ids.ndim != 2 or ids.dtype.kind not in "iu":
        raise ValueError("Saved facies must be an integer map")
    ids = ids.astype(np.int64)
    if ids.size and ids.min() < 0:
        raise ValueError("Saved facies ids must not be negative")
    weights = np.asarray(weights_l, np.float64)
    arrays = {key: np.asarray(masses[key]) for key in COMPONENTS}
    if weights.shape != ids.shape or any(value.shape != ids.shape for value in arrays.values()):
        raise ValueError("Masses, facies and pore volumes must share one native grid")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Native mass density requires saved finite pore-volume weights in litres")
    unknown = np.asarray(unknown_mask, bool) | np.logical_or.reduce([~np.isfinite(value) for value in arrays.values()])
    flat, valid = ids.ravel(), ~unknown.ravel()
    count = int(flat.max()) + 1 if flat.size else 0
    samples = np.bincount(flat, minlength=count)
    missing = np.bincount(flat, weights=unknown.ravel().astype(np.float64), minlength=count)
    volume = np.bincount(flat, weights=weights.ravel(), minlength=count)
    sums = {key: np.bincount(flat[valid], weights=np.asarray(value, np.float64).ravel()[valid]
                             * weights.ravel()[valid], minlength=count) for key, value in arrays.items()}
    rows = []
    for facies_id in np.flatnonzero((samples > 0) & (volume > 0)):
        facies_id = int(facies_id)
        complete = bool(missing[facies_id] == 0)
        row = dict(domain_id=f"facies:{facies_id}", name=str((names or {}).get(facies_id, f"Facies {facies_id}")),
                   facies_id=facies_id, complete=complete, sample_count=int(samples[facies_id]),
                   unknown_sample_count=int(missing[facies_id]))
        for key in COMPONENTS:
            amount = float(sums[key][facies_id])
            row[key + "_mass_g"] = amount if complete else None
            row[key + "_quantified_mass_g"] = amount
        rows.append(row)
    return rows


def layer_names(project):
    """Facies id -> layer name from the series' current geometry revision (its facies table).

    ``project`` is the series' geometry package. Without a revision or table
    there are no names; a facies without a name is called "Facies <id>".
    """
    from .geometry_facies import GeometryRevisionStore
    current = GeometryRevisionStore(project).current()
    path = Path(str((current.get("artifacts") or {}).get("facies_properties_path") or ""))
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {int(row["id"]): str(row.get("name") or "").strip() or f"Facies {int(row['id'])}"
                for row in csv.DictReader(stream) if str(row.get("id") or "").strip()}


class _SavedMaps:
    """Small adapter around the ordinary finished-product and array readers."""

    def __init__(self, root, stores_path, boxes_config):
        from .production_results_view import ProductionJob
        from .pipeline.current_inputs import CurrentInputReader
        self.job = ProductionJob(root)
        self.stores_path, self.configs = stores_path, {}
        self.boxes_config, self.box_models = boxes_config, {}

        class Reader(CurrentInputReader):
            def archive(self, bound, names, *, optional=()):
                return self.reader.archive(self.physical_descriptor(bound), names, optional=optional)

        self.data = Reader(self.job._data())

    def frame_binding(self, run, result):
        """Join box and mass origins through their original measured model.

        Matching the labelled unit alone is insufficient: the acquisition and
        camera-to-metric transform must also be the same. No coordinate refit
        or reinterpretation of an old nominal rectangle is performed here.
        """
        if run not in self.box_models:
            pin = self.boxes_config.get("geometry_index")
            if not pin:
                raise ValueError("Analysis boxes need their saved measured geometry index")
            index = self.data.json(pin)
            entries = index["runs"]
            if isinstance(entries, dict):
                rows = [value for key, value in entries.items() if str(key).casefold() == run.casefold()]
            else:
                rows = [value for value in entries
                        if str(value.get("run_id", value.get("run", ""))).casefold() == run.casefold()]
            if len(rows) != 1 or not isinstance(rows[0].get("reference_model"), dict):
                raise ValueError("The box geometry must bind exactly one measured model for " + run)
            model_pin = deepcopy(rows[0]["reference_model"])
            if not Path(model_pin["path"]).is_absolute():
                model_pin["path"] = str(Path(pin["path"]).parent / model_pin["path"])
            self.box_models[run] = self.data.json(model_pin), model_pin
        box_model, box_pin = self.box_models[run]
        local = (result.get("scalar_result", {}).get("local_build_accounting")
                 or result.get("prediction", {}).get("local_build_accounting"))
        source_pin = (local or {}).get("source_binding", {}).get("metric_model")
        if source_pin is None:
            raise ValueError("Saved mass maps lack a measured coordinate-frame binding; "
                             "their nominal origin cannot be assumed to match these boxes")
        source_model = self.data.json(source_pin)
        for model in (box_model, source_model):
            frame = model.get("metric_frame", {})
            if (str(model.get("run", model.get("run_id", ""))).casefold() != run.casefold()
                    or frame.get("unit") != "mm" or frame.get("x_axis") != "right"
                    or frame.get("y_axis") != "up" or frame.get("origin") != self.boxes_config.get("origin")):
                raise ValueError("Analysis boxes and mass maps use incompatible physical origins or axes")
        if (box_model["reference_image"]["sha256"] != source_model["reference_image"]["sha256"]
                or not np.array_equal(np.asarray(box_model["H_camera_u_to_metric_mm"], np.float64),
                                      np.asarray(source_model["H_camera_u_to_metric_mm"], np.float64))):
            raise ValueError("Analysis boxes and mass maps use different measured coordinate frames")
        return dict(box_model=box_pin, mass_model=deepcopy(source_pin),
                    metric_frame=deepcopy(source_model["metric_frame"]))

    def configuration(self, run):
        if run in self.configs:
            return self.configs[run]
        from .store_map import read_store_map, resolve_input_reference
        task = self.job._reader(run).tasks[run]
        reference = task.get("configuration")
        if not reference:
            raise MapsUnavailable("The saved job has no coordinate configuration binding")
        stores = read_store_map(self.stores_path)
        path = resolve_input_reference(reference, stores, self.job.record["series_id"])
        bound = dict(path=path, sha256=reference["sha256"], size_bytes=reference["bytes"])
        value = self.data.json(bound)
        # A current-Build configuration contains the ordinary mass template.
        if "mass" in value and "template" in value["mass"]:
            value = dict(value["mass"]["template"], frames=value["frames"])
        self.configs[run] = value, deepcopy(reference)
        return self.configs[run]

    def coordinates(self, run, case, result, shape):
        from .mass.n1_geometry import frame_coordinates
        from .storage_profile import load_store
        regeneration = (result.get("static", {}).get("storage") or {}).get("regeneration")
        local = (result.get("scalar_result", {}).get("local_build_accounting")
                 or result.get("prediction", {}).get("local_build_accounting"))
        if regeneration and not local:
            store = load_store(regeneration["store"], reader=self.data)
            if tuple(regeneration["phase_shape_hw"]) != tuple(shape):
                raise ValueError("Saved native coordinates and mass maps have different grids")
            coords = frame_coordinates(store.reference_matrix,
                {"h": regeneration["source_to_native_h"]}, regeneration, fields={"source_to_native": "h"})
            return coords, dict(regeneration=deepcopy(regeneration))
        if regeneration:
            current, config_pin, config = regeneration, None, {}
        else:
            config, config_pin = self.configuration(run)
            frames = [row for row in config["frames"] if row["case_id"] == case]
            if len(frames) != 1 or "current" not in frames[0]:
                raise MapsUnavailable("The saved configuration lacks this frame's native coordinate descriptor")
            current = frames[0]["current"]
        origin = current.get("native_origin_xy")
        if origin is None:
            raise MapsUnavailable("No saved native canvas origin is available")
        revision = dict(phase_shape_hw=list(shape), native_origin_xy=origin)
        declared = current.get("phase_shape_hw")
        if declared is not None and tuple(declared) != tuple(shape):
            raise ValueError("Saved configuration and mass map have different native grids")
        if local:
            binding = local["source_binding"]
            metric = self.data.json(binding["metric_model"])
            receipt = self.data.json(binding["bridge_frame"])
            if metric.get("run") != run:
                raise ValueError("Saved metric model belongs to another run")
            build_to_m = (np.diag([.001, .001, 1.])
                @ np.asarray(metric["H_camera_u_to_metric_mm"], np.float64)
                @ np.asarray(metric["baseline_native_grid"]["H_rig_corrected_native_to_baseline_camera_u"], np.float64))
            matrix = receipt["build_to_frame_native_h"]
            coords = frame_coordinates(build_to_m, {"h": matrix}, revision, fields={"source_to_native": "h"})
            return coords, dict(configuration=config_pin, metric_model=binding["metric_model"],
                                bridge_frame=binding["bridge_frame"], native_origin_xy=origin)
        shared = config.get("shared", {})
        if "reference_matrix" not in shared or "geometry_fields" not in config:
            raise MapsUnavailable("No explicit saved physical reference matrix; use the stored metric layer")
        coords = frame_coordinates(shared["reference_matrix"], current["binding"]["geometry"],
                                   revision, fields=config["geometry_fields"])
        return coords, dict(configuration=config_pin)

    def native(self, run, case, result):
        from .storage_profile import read_products, read_static_fields
        if not self.job._has_products(run, case) or not result.get("static"):
            raise MapsUnavailable("This frame has no saved native mass/pore-volume products")
        units = result.get("units", {})
        if any(units.get(name) != "g/L" for name in NATIVE_FIELDS.values()):
            raise MapsUnavailable("Saved native maps do not declare mass density in g/L")
        folder = self.job._products_folder(run, case)
        names = self.job._product_names(run, case)
        if not set(NATIVE_FIELDS.values()) <= names:
            raise MapsUnavailable("This frame lacks native mass component maps")
        arrays = read_products(result, folder, list(NATIVE_FIELDS.values()) + [name for name in
            ("approved_analysis_gap_mask", "unquantified_facies_mask") if name in names], reader=self.data)
        if "transport_weight_l" not in result["static"]["arrays"]:
            raise MapsUnavailable("No saved native pore-volume weights are available")
        weights = read_static_fields(result, folder, ["transport_weight_l"], reader=self.data)["transport_weight_l"]
        (x, y), geometry = self.coordinates(run, case, result, weights.shape)
        unknown = np.zeros(weights.shape, bool)
        for name in ("approved_analysis_gap_mask", "unquantified_facies_mask"):
            if name in arrays:
                if arrays[name].dtype != bool:
                    raise ValueError("Saved unknown-support masks must be boolean")
                unknown |= arrays[name]
        return dict(masses={key: arrays[name] for key, name in NATIVE_FIELDS.items()}, x_m=x, y_m=y,
                    weights_l=weights, unknown_mask=unknown, storage="native"), dict(
                        products=deepcopy(result["products"]), static=deepcopy(result["static"]), geometry=geometry)

    def facies(self, run, case, result):
        """A frame's saved facies ids on its native grid; MapsUnavailable when none are saved."""
        from .storage_profile import read_static_fields
        if "facies" not in (result.get("static") or {}).get("arrays", {}):
            raise MapsUnavailable("This frame has no saved native facies")
        return read_static_fields(result, self.job._products_folder(run, case), ["facies"],
                                  reader=self.data)["facies"]

    def cells(self, run, case, result):
        reader = self.job._reader(run)
        from .published_dataset.job_reader import LAYER_RECEIPT
        layer, receipt = reader.layer(run, case), reader.layer(run, case, LAYER_RECEIPT)
        if layer is None or receipt is None:
            raise MapsUnavailable("No saved metric mass layer with its grid receipt is available")
        document = json.loads(reader.product_bytes(receipt))
        grid = (document.get("coarse") or {}).get("grid")
        if (not grid or grid.get("format") != "darsia-metric-storage-grid-v1"
                or grid.get("axes") != "row=y increasing, column=x increasing"):
            raise ValueError("The saved metric layer must declare its original physical grid")
        arrays = self.job._layer_arrays(run, case)
        if not set(CELL_FIELDS.values()) <= arrays.keys():
            raise MapsUnavailable("The saved metric layer lacks mass component maps")
        h, w = map(int, grid["shape"])
        x0, x1, y0, y1 = map(float, grid["extent_m"])
        if h <= 0 or w <= 0 or not x1 > x0 or not y1 > y0:
            raise ValueError("The saved metric layer extent is invalid")
        x, y = np.meshgrid(x0 + (np.arange(w) + .5) * (x1-x0)/w,
                           y0 + (np.arange(h) + .5) * (y1-y0)/h)
        unknown = np.zeros((h, w), bool)
        for name in ("mask_approved_analysis_gap_area_fraction", "mask_unquantified_facies_area_fraction"):
            if name in arrays:
                fraction = arrays[name]
                if (fraction.shape != unknown.shape or not np.isfinite(fraction).all()
                        or np.any((fraction < 0) | (fraction > 1))):
                    raise ValueError("Saved unknown-support fractions are invalid")
                unknown |= fraction > 0
        return dict(masses={key: arrays[name] for key, name in CELL_FIELDS.items()}, x_m=x, y_m=y,
                    unknown_mask=unknown, storage="metric_cells", resolution_mm=grid["resolution_mm"]), dict(
                        layer=_product_pin(layer), grid_receipt=_product_pin(receipt), grid=grid)


def _product_pin(product):
    return dict(product_id=product.product_id, path=product.local_path,
                sha256=product.sha256, size_bytes=product.size_bytes)


def _destination(root, output, job, runs):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Box reanalysis needs a new result folder; existing results are immutable")
    if output == root or root in output.parents:
        allowed = root / "analysis" / "boxes"
        if allowed not in output.parents:
            raise ValueError("Derived results inside a job must use analysis/boxes/<new folder>")
    for run in runs:
        for folder in job.product_folders(run) - {job.root}:
            physical = folder.resolve()
            if output == physical or physical in output.parents:
                raise ValueError("Derived results cannot be written into a source product folder")
    return output


def _metadata_pins(job, runs):
    """Only the original completion/control metadata; never the RAW catalogue."""
    paths = [job.root / "job.json"]
    for run in runs:
        from .pipeline.job_queue import name
        reader = job._reader(run)
        paths.extend((job.root / "done" / (name(run) + ".json"),
                      job.root / "results" / name(run) / "task.json",
                      job.root / reader.done[run]["products_index"]["relative_path"],
                      job.root / reader.done[run]["completed"]["relative_path"]))
    return [_pin(path) for path in paths]


def reanalyze_saved_boxes(job_root, boxes_config, output_folder, *, runs=None, source="auto",
                         actor, reason, check_cancelled=None, on_progress=None, stores_path=None,
                         layers=False, layer_names=None):
    """Write new box curves from finished saved maps, preserving all sources.

    ``source='auto'`` prefers native densities and original pore-volume weights;
    it uses saved metric cells only when native data/coordinates are unavailable.
    Corrupt or changed sources always fail rather than selecting a fallback.
    With ``layers`` each run also gets layers-run-NNNN.csv: every frame's mass
    per sand layer (``integrate_facies_arrays``, named by ``layer_names``) from
    its native maps. A frame without native maps or saved facies has no layer
    rows; the manifest counts them. Without ``layers`` the output is unchanged.
    """
    if source not in {"auto", "native", "metric_cells"}:
        raise ValueError("Choose auto, native or metric_cells")
    if not str(actor).strip() or not str(reason).strip():
        raise ValueError("The derived result needs its actor and purpose")
    config = deepcopy(boxes_config)
    boxes = _boxes(config)
    check = check_cancelled or (lambda: None)
    progress = on_progress or (lambda value: None)
    check()
    saved = _SavedMaps(job_root, stores_path, config)
    from .production_results_view import COMPLETE
    available = [row.run for row in saved.job.runs if row.state == COMPLETE]
    selected = available if runs is None else list(runs)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(available):
        raise ValueError("Select at least one finished readable production run")
    destination = _destination(job_root, output_folder, saved.job, selected)
    job_pin = _pin(Path(job_root) / "job.json")
    metadata_pins = _metadata_pins(saved.job, selected)
    total = sum(len(saved.job.frames(run)) for run in selected)
    if not total:
        raise MapsUnavailable("The finished runs contain no saved mass-map frames")
    check()
    destination.mkdir(parents=True, exist_ok=False)
    manifest = dict(format=FORMAT, complete=False, actor=str(actor), reason=str(reason),
        created_utc=datetime.now(timezone.utc).isoformat(), source_job=job_pin,
        boxes_configuration=config, boxes_sha256=hashlib.sha256(_json(config)).hexdigest(),
        boxes=boxes, source_preference=source, source_metadata=metadata_pins, units="g", runs=[], frame_count=0,
        selection="Physical pixel/cell centres within half-open x/y bounds; no resampling",
        unknown_policy="Any selected unknown support makes total mass undefined; quantified subtotal is retained",
        coarse_limitation="Metric cells are selected whole by centre; precision is the recorded storage resolution")
    tables, completed = [], 0
    layer_tables, frames_without_layers = [], 0
    for run_index, run in enumerate(selected):
        records, frames = [], []
        layer_records = []
        reader = saved.job._reader(run)
        for case in saved.job.frames(run):
            check()
            result = saved.job.result(run, case)
            coordinate_frame = saved.frame_binding(run, result)
            try:
                if source == "metric_cells":
                    raise MapsUnavailable("Metric-cell source explicitly selected")
                arrays, evidence = saved.native(run, case, result)
            except MapsUnavailable as unavailable:
                if source == "native":
                    raise
                arrays, evidence = saved.cells(run, case, result)
                evidence["native_unavailable"] = str(unavailable)
            check()
            evidence["coordinate_frame"] = coordinate_frame
            integrated = integrate_mass_arrays(**arrays, boxes=boxes)
            elapsed = saved.job.elapsed_h(run, case)
            records.extend(dict(run=run, case_id=case, elapsed_h=elapsed, **row) for row in integrated)
            if layers:
                try:
                    if arrays["storage"] != "native":
                        raise MapsUnavailable("Layers need native maps")
                    facies = saved.facies(run, case, result)
                except MapsUnavailable:
                    frames_without_layers += 1
                else:
                    layer_records.extend(dict(run=run, case_id=case, elapsed_h=elapsed, **row) for row in
                                         integrate_facies_arrays(arrays["masses"], facies, weights_l=arrays["weights_l"],
                                                                 unknown_mask=arrays["unknown_mask"],
                                                                 names=layer_names))
            result_product = reader.product(run, f"frames/{case}/result.json")
            frames.append(dict(case_id=case, result=_product_pin(result_product), source=evidence))
            completed += 1
            progress(dict(run=run, completed=completed, total=total, frame_count=len(frames),
                          message=f"Integrated {run} {case} ({completed}/{total})"))
        # A single table per run contains every enabled box and frame.
        path = destination / f"run-{run_index + 1:04d}.csv"
        text = io.StringIO(newline="")
        columns = ["run", "case_id", "elapsed_h", "domain_id", "name", "bounds_mm", "storage", "resolution_mm",
                   "complete", "sample_count", "unknown_sample_count"]
        columns += [key + suffix for key in COMPONENTS for suffix in ("_mass_g", "_quantified_mass_g")]
        writer = csv.DictWriter(text, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
        path.write_bytes(text.getvalue().encode("utf-8"))
        bound = _pin(path)
        tables.append(bound)
        manifest["runs"].append(dict(run=run, table=bound, frames=frames,
            completion=deepcopy(reader.done[run])))
        if layers:
            # Each sand layer's masses in a table of their own, next to the box table.
            text = io.StringIO(newline="")
            writer = csv.DictWriter(text, fieldnames=LAYER_COLUMNS)
            writer.writeheader()
            writer.writerows(layer_records)
            layer_path = destination / f"layers-run-{run_index + 1:04d}.csv"
            layer_path.write_bytes(text.getvalue().encode("utf-8"))
            manifest["runs"][-1]["layer_table"] = _pin(layer_path)
            layer_tables.append(manifest["runs"][-1]["layer_table"])
    check()
    if [_pin(row["path"]) for row in metadata_pins] != metadata_pins:
        raise ValueError("The source job or completion records changed during box reanalysis")
    manifest.update(complete=True, frame_count=completed, run_count=len(selected))
    if layers:
        manifest["layers"] = dict(
            names={str(key): value for key, value in (layer_names or {}).items()},
            frames_without_native_facies=frames_without_layers,
            selection="Each pixel's saved facies id on the frame's own native grid")
    path = destination / "manifest.json"
    path.write_bytes(_json(manifest))
    summary = dict(manifest=str(path), tables=tables, boxes=boxes,
                   run_count=len(selected), frame_count=completed)
    if layers:
        summary.update(layer_tables=layer_tables, frames_without_native_facies=frames_without_layers)
    return summary
