"""Read already committed coarse storage products without native field reads.

The caller supplies an original Main-format frame/run completion and an asset
resolver, normally backed by products_index/collect. No search, collection,
native archive reader, model, geometry builder or photograph fallback is used.
Only an explicitly requested BEFORE view opens the saved native-grid LA PNG.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath
import re
import zipfile

import numpy as np

from . import storage_outputs as output
from .storage_resolution import CELL_COUNT_FIELDS, LABEL_LEGEND, metric_grid_shape, resolution_mm


class StoredProductCollectionRequired(FileNotFoundError):
    """The exact immutable product must be made accessible with collect."""

    def __init__(self, descriptor):
        self.descriptor = deepcopy(descriptor)
        super().__init__("Stored product unavailable; collect this exact asset: "
                         + json.dumps(descriptor, sort_keys=True))


def _descriptor(value):
    if (not isinstance(value, dict) or not isinstance(value.get("path"), str)
            or not value["path"] or "\x00" in value["path"]
            or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None
            or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0):
        raise ValueError("Complete immutable stored-product descriptor required")
    return value


def _read_bytes(bound, resolve_asset, loaded):
    _descriptor(bound)
    try:
        path = Path(resolve_asset(deepcopy(bound)))
        before = path.stat()
        if before.st_size != bound["size_bytes"]:
            raise ValueError("Stored product size changed: " + bound["path"])
        data = path.read_bytes()
        after = path.stat()
    except FileNotFoundError as exc:
        raise StoredProductCollectionRequired(bound) from exc
    if ((before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
            or len(data) != bound["size_bytes"]
            or hashlib.sha256(data).hexdigest() != bound["sha256"]):
        raise ValueError("Stored product SHA/size/version changed: " + bound["path"])
    loaded.append(deepcopy(bound))
    return data


def _json(data):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate saved-product metadata key")
            value[key] = item
        return value

    def invalid(_value):
        raise ValueError("Non-finite saved-product JSON metadata")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)


def _record_asset(record, name):
    row = record["files"][name]
    root = record["nas_directory"]
    logical = root.rstrip("/\\") + ("\\" if "\\" in root else "/") + name
    if "path" in row and PureWindowsPath(row["path"]) != PureWindowsPath(logical):
        raise ValueError("Original frame inventory points outside its frame")
    return _descriptor(dict(row, path=logical))


def _finite(value, label):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError("Invalid saved " + label)
    return value


def _shape(value):
    if (not isinstance(value, list) or len(value) != 2
            or any(type(v) is not int or v < 1 for v in value)):
        raise ValueError("Saved grid needs two exact positive integer dimensions")
    return tuple(value)


def _request(document):
    value = document["request"]
    if (not isinstance(value, dict)
            or set(value) != {"format", "resolution_mm", "gas_score_encoding", "extent_m", "precision"}
            or value["format"] != output.REQUEST_FORMAT
            or value["gas_score_encoding"] != output.LA_ENCODING
            or value["precision"] != document["precision"]):
        raise ValueError("Unknown saved coarse request/encoding/precision")
    if resolution_mm(value["resolution_mm"]) is None:
        raise ValueError("Native output is not a coarse saved-product branch")
    extent = value["extent_m"]
    if (not isinstance(extent, list) or len(extent) != 4
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in extent)
            or extent[1] <= extent[0] or extent[3] <= extent[2]):
        raise ValueError("Saved coarse metric extent differs")
    metric_grid_shape(extent, value["resolution_mm"])
    return value


def _identity(record, commit, result, binding, entry, document, expected):
    case = record["frame_id"]
    if (not isinstance(case, str) or re.fullmatch(r"[A-Za-z0-9_-]+", case) is None
            or commit.get("complete") is not True or commit.get("frame_id") != case
            or commit.get("manifest_sha256") != record["manifest_sha256"]
            or result.get("complete") is not True or result.get("case_id") != case
            or binding.get("format") != "darsia-series-frame-v1"
            or binding.get("complete") is not True or binding.get("case_id") != case
            or document.get("format") != output.FRAME_FORMAT
            or document.get("complete") is not True
            or document.get("case_id") != case
            or document.get("run") != result["run"]
            or document.get("job_id") != commit["job_id"]
            or document.get("attempt_directory") != entry["attempt_directory"]
            or document.get("files") != entry["files"]
            or document.get("existing_native_products_unchanged") is not True):
        raise ValueError("Committed saved-product frame/producer identity differs")
    precision = document["precision"]
    code = document["code_commit"]
    source = document["source"]
    if (precision not in {"float32", "float64"}
            or any(v != precision for v in (result["precision"], result["working_pixel_dtype"],
                                            commit["products"]["precision"], commit["settings"]["precision"]))
            or re.fullmatch(r"[0-9a-f]{40}", code) is None
            or result["bindings"]["code_sha256"] != hashlib.sha256(code.encode("ascii")).hexdigest()
            or document["calibration_sha256"] != result["bindings"]["calibration_sha256"]
            or re.fullmatch(r"[0-9a-f]{64}", document["calibration_sha256"]) is None
            or _descriptor(document["configuration"])["sha256"] != binding["configuration_sha256"]
            or source != binding["source"]
            or source["source_sha256"] != result["bindings"]["source_sha256"]
            or source["source_binding"] != result["source_binding"]
            or binding["native"]["source_sha256"] != source["source_sha256"]
            or binding["phase"]["source_sha256"] != source["source_sha256"]
            or binding["phase"]["labels"]["sha256"] != result["bindings"]["phase_sha256"]):
        raise ValueError("Saved source/configuration/freeze/code/precision differs")
    for key, value in (expected or {}).items():
        if key not in {"job_id", "run", "case_id", "configuration", "code_commit",
                       "calibration_sha256", "precision", "source"} or document.get(key) != value:
            raise ValueError("Saved product differs from expected " + key)


def _archive(data, document):
    coarse = document["coarse"]
    if (not isinstance(coarse, dict) or coarse.get("format") != "darsia-coarse-stored-frame-v1"
            or coarse.get("precision") != document["precision"]
            or coarse.get("computing_resolution") != "native"
            or coarse.get("reduction_dtype") != "float64"
            or coarse.get("per_frame_geometry") is not True
            or coarse.get("shared_static_geometry_assumed") is not False
            or coarse.get("label_legend") != LABEL_LEGEND):
        raise ValueError("Unknown saved coarse schema/precision/geometry")
    grid = coarse["grid"]
    shape, native = _shape(grid["shape"]), _shape(grid["native_shape"])
    if math.prod(shape) > 100_000_000:
        raise ValueError("Saved grid exceeds the producer's bounded cell count")
    request = _request(document)
    extent = request["extent_m"]
    if (grid.get("format") != "darsia-metric-storage-grid-v1"
            or grid.get("extent_m") != extent or grid.get("resolution_mm") != request["resolution_mm"]
            or shape != metric_grid_shape(extent, request["resolution_mm"])
            or grid.get("axes") != "row=y increasing, column=x increasing"
            or grid.get("storage_precision") != document["precision"]
            or grid.get("computing_resolution") != "native"
            or grid.get("reduction_dtype") != "float64"
            or grid.get("assignment") != "native pixel centre; half-open cells and extent [min,max); exact upper edge STOP; no clamping"
            or grid.get("cell_count_fields") != CELL_COUNT_FIELDS
            or grid.get("probability_coordinate_counts") != dict(native_pixels=math.prod(native),
                inside_extent=math.prod(native), outside_extent=0, invalid_coordinates=0)):
        raise ValueError("Saved coarse grid/extent/orientation/precision differs")
    required = {"concentration_aqueous", "saturation_free_gas", "concentration_valid_area_fraction",
                "saturation_valid_area_fraction", "phase_water_area_fraction", "phase_dissolved_area_fraction",
                "phase_gas_area_fraction", "phase_excluded_area_fraction", "gas_probability_after_rules",
                "cell_mass_total_g", "cell_mass_aqueous_g", "cell_mass_gas_g", "static_depth_m_mean",
                "static_depth_valid_area_fraction", "static_porosity_mean", "static_porosity_valid_area_fraction",
                "static_pixel_area_m2_sum", "static_facies_majority", "static_facies_majority_area_fraction",
                "static_cell_has_area", *CELL_COUNT_FIELDS}
    before_complete = document["before_rules"]["state"] == "complete"
    if before_complete:
        required |= {"gas_probability_before_rules", "gas_probability_before_rules_valid_pixel_fraction"}
    identities = coarse["arrays"]
    for key in set(identities) - required:
        middle = key.removeprefix("mask_").removesuffix("_area_fraction")
        if (not key.startswith("mask_") or not key.endswith("_area_fraction")
                or not middle or not middle.replace("_", "").isalnum()):
            raise ValueError("Unknown coarse archive array: " + key)
    if not required <= set(identities):
        raise ValueError("Missing coarse archive arrays")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {key + ".npy" for key in identities}:
            raise ValueError("Saved archive keys/members differ from exact array closure")
    arrays = {}
    dtype = np.dtype(document["precision"])
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        for key, identity in identities.items():
            value = archive[key]
            if (value.shape != shape or identity != output._array_identity(value)
                    or value.dtype.hasobject):
                raise ValueError("Saved array shape/dtype/content identity differs: " + key)
            if key == "static_cell_has_area":
                valid_dtype = value.dtype == np.bool_
            elif key in CELL_COUNT_FIELDS:
                valid_dtype = value.dtype == np.dtype("int64")
            elif key == "static_facies_majority":
                valid_dtype = value.dtype.kind in "iu"
            else:
                valid_dtype = value.dtype == dtype
            if not valid_dtype or value.dtype.kind == "f" and np.isinf(value).any():
                raise ValueError("Saved array precision/type/non-finite value differs: " + key)
            arrays[key] = value
    area, has_area = arrays["static_pixel_area_m2_sum"], arrays["static_cell_has_area"]
    if not np.isfinite(area).all() or np.any(area < 0) or not np.array_equal(has_area, area > 0):
        raise ValueError("Saved cell area/coverage differs")
    included_count = arrays["static_included_area_native_pixel_count"]
    canvas_count = arrays["static_full_canvas_probability_pixel_count"]
    if (np.any(included_count < 0) or np.any(canvas_count < 0)
            or np.any(included_count > canvas_count)
            or int(canvas_count.sum(dtype=np.int64)) != math.prod(native)
            or np.any(has_area & (included_count == 0))):
        raise ValueError("Saved included-area/full-canvas pixel counts differ")
    for key, value in arrays.items():
        if key.endswith("_fraction") or key.startswith("gas_probability_"):
            finite = np.isfinite(value)
            if np.any(finite & ((value < 0) | (value > 1))):
                raise ValueError("Saved fraction lies outside [0,1]: " + key)
            if key.endswith("_area_fraction") and not np.array_equal(np.isnan(value), ~has_area):
                raise ValueError("Saved area fraction has different empty-cell support: " + key)
    for mean, fraction in (("concentration_aqueous", "concentration_valid_area_fraction"),
                           ("saturation_free_gas", "saturation_valid_area_fraction"),
                           ("static_depth_m_mean", "static_depth_valid_area_fraction"),
                           ("static_porosity_mean", "static_porosity_valid_area_fraction")):
        if not np.array_equal(np.isnan(arrays[mean]), ~(arrays[fraction] > 0)):
            raise ValueError("Saved mean and finite-area support differ: " + mean)
    if before_complete and not np.array_equal(np.isnan(arrays["gas_probability_before_rules"]),
            ~(arrays["gas_probability_before_rules_valid_pixel_fraction"] > 0)):
        raise ValueError("Saved BEFORE mean/valid-pixel support differs")
    return arrays, grid, native


def read_stored_frame(native_frame_metadata, *, resolve_asset, load_before=False, expected=None):
    """Read a native-commit-anchored coarse frame and independent saved masses.

    ``native_frame_metadata`` is the existing Main-format committed frame record.
    ``resolve_asset(descriptor)`` returns an accessible path or FileNotFoundError;
    it must not collect, search for substitutes or execute a producer.
    """
    record, loaded = native_frame_metadata, []
    commit = _json(_read_bytes(_record_asset(record, "commit.json"), resolve_asset, loaded))
    if ({key: {k: value[k] for k in ("sha256", "size_bytes")}
         for key, value in record["files"].items() if key != "commit.json"} != commit.get("files")):
        raise ValueError("Original committed frame inventory differs")
    result = _json(_read_bytes(_record_asset(record, "result.json"), resolve_asset, loaded))
    binding = result["prediction"]["pipeline"]
    entry = output.validate_publication_binding(binding)
    document = _json(_read_bytes(entry["receipt"], resolve_asset, loaded))
    _identity(record, commit, result, binding, entry, document, expected)
    if document.get("extensive") != entry.get("extensive"):
        raise ValueError("Extensive I/O entry differs from its committed receipt")
    added = {}
    if document.get("extensive") is not None:
        from .storage_extensive_io import declared_files
        added = declared_files(document["extensive"])
    _request(document)
    by_name = {PureWindowsPath(row["path"]).name: row for row in entry["files"]}
    if any(by_name.get(name) != row for name, row in added.items()):
        raise ValueError("Extensive I/O file is outside its committed native entry")
    before = document["before_rules"]
    if before.get("state") == "complete":
        if before.get("encoding") != output.LA_ENCODING or "storage_gas_probability_before_rules.png" not in by_name:
            raise ValueError("Complete BEFORE has no exact LA product")
    elif before.get("state") == "pending":
        if not before.get("reason") or "storage_gas_probability_before_rules.png" in by_name:
            raise ValueError("Pending BEFORE must remain explicitly unavailable")
    else:
        raise ValueError("Unknown BEFORE state")
    after = document["after_rules"]
    if (after["source"] != binding["phase"]["labels"] or after.get("label_legend") != LABEL_LEGEND
            or after.get("support_mask") is not None or after.get("values") != [0, 1]):
        raise ValueError("Saved AFTER does not bind the unchanged final gas indicator")
    data = _read_bytes(by_name["storage_stored_frame.npz"], resolve_asset, loaded)
    arrays, grid, native_shape = _archive(data, document)
    counts = document["final_label_counts"]
    if (set(counts) != set(LABEL_LEGEND) or any(type(v) is not int or v < 0 for v in counts.values())
            or sum(counts.values()) != math.prod(native_shape) or after["gas_pixels"] != counts["2"]):
        raise ValueError("Saved final native label counts differ")
    totals = {}
    document_totals = output.checked_mass_totals(document, result, arrays=arrays, public=True)
    source_totals = output.quantified_totals(result)
    for component, native_name in output.TOTALS.items():
        target = _finite(source_totals[component], component + " native mass")
        cells = arrays["cell_mass_" + component + "_g"]
        if not np.isfinite(cells).all() or np.any(cells < 0):
            raise ValueError("Saved cell masses must remain finite/nonnegative")
        actual = float(cells.sum(dtype=np.float64))
        coarse_check = document["coarse"]["native_mass_checks"][component]
        pixel_check = document["native_pixel_mass_checks"][component]
        if (document_totals[component] != target
                or abs(actual - target) > 1e-6 * abs(target)
                or coarse_check != dict(native_total_g=target, stored_cell_sum_g=actual,
                    absolute_difference_g=abs(actual - target), relative_limit=1e-6, passed=True)
                or pixel_check.get("native_total_g") != target or pixel_check.get("relative_limit") != 1e-6
                or pixel_check.get("passed") is not True
                or abs(_finite(pixel_check["pixel_sum_g"], component + " pixel sum") - target) > 1e-6 * abs(target)):
            raise ValueError("Independent stored/native " + component + " mass closure differs")
        totals[component] = actual
    native_before = None
    if load_before and before["state"] == "complete":
        from PIL import Image
        png = _read_bytes(by_name["storage_gas_probability_before_rules.png"], resolve_asset, loaded)
        with Image.open(io.BytesIO(png)) as image:
            if image.mode != "LA":
                raise ValueError("Saved BEFORE PNG is not literal LA")
            la = np.asarray(image).copy()
        if (la.dtype != np.uint8 or la.shape != (*native_shape, 2)
                or not np.isin(la[..., 1], (0, 255)).all()):
            raise ValueError("Saved BEFORE alpha/grid differs")
        native_before = la[..., 0].astype(document["precision"]) / np.dtype(document["precision"]).type(255)
        native_before[la[..., 1] == 0] = np.nan
        native_before.setflags(write=False)
    for value in arrays.values():
        value.setflags(write=False)
    value = dict(format="darsia-read-coarse-stored-frame-v1", run=document["run"], case_id=record["frame_id"],
                precision=document["precision"], grid=deepcopy(grid), imshow_origin="lower", arrays=arrays,
                before_rules=deepcopy(before), before_native=native_before, native_scalar_reference=deepcopy(result),
                mass_totals_g=totals, source=deepcopy(document["source"]), loaded_assets=loaded,
                storage_receipt=deepcopy(entry["receipt"]), storage_document=deepcopy(document))
    return output.read_mass_scope(value, result, totals)


def read_stored_run(completed_metadata, *, resolve_asset, load_before=False, stream=False):
    """Read one original complete run; never sum mass across acquisition times.

    The default retains the original list result. Explicit ``stream=True``
    retains only metadata and yields one strict, individually verified frame.
    A streaming caller must release a yielded frame before requesting another.
    """
    if type(stream) is not bool:
        raise ValueError("Saved run streaming must be an explicit boolean")
    value, loaded = (deepcopy(completed_metadata) if stream else completed_metadata), []
    records, order = value["records"], value["ordered_case_ids"]
    if (value.get("complete") is not True or value.get("pipeline") != "native-phase-n1"
            or type(value["frames"]) is not int or value["frames"] != len(records)
            or value["case_count"] != len(records) or not records
            or order != [r["frame_id"] for r in records] or len(set(order)) != len(order)):
        raise ValueError("Explicit complete saved native-phase run/order required")
    manifest = _json(_read_bytes(value["manifest"], resolve_asset, loaded))
    if (manifest.get("format") != "darsia-series-run-manifest-v1"
            or manifest["run"] != value["run"] or manifest["configuration"] != value["configuration"]
            or manifest["settings"]["precision"] != value["precision"]
            or set(manifest["frames"]) != set(order)):
        raise ValueError("Saved run manifest/configuration/selection differs")
    def frames():
        for record in records:
            case = record["frame_id"]
            if record["manifest_sha256"] != value["manifest"]["sha256"]:
                raise ValueError("Saved frame belongs to another run manifest")
            yield read_stored_frame(record, resolve_asset=resolve_asset, load_before=load_before,
                expected=dict(job_id=manifest["job_id"], run=value["run"], case_id=case,
                    precision=value["precision"], code_commit=manifest["code"]["commit"],
                    configuration=value["configuration"], calibration_sha256=value["calibration"]["sha256"],
                    source=manifest["frames"][case]["source"]))
    selected_frames = frames() if stream else list(frames())
    return dict(format="darsia-read-coarse-stored-run-v1", run=value["run"], precision=value["precision"],
                ordered_case_ids=list(order), frames=selected_frames, loaded_manifest_assets=loaded,
                original_completion=deepcopy(value))
