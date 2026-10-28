"""Explicit lossless imported raster sources for geometry workspace revisions.

This is a workspace representation, not a map correction. The ordinary painted
and raw region-derived paths do not call these functions. An imported revision
keeps both categorical rasters and the full original CSV independently; edits
require a deliberately created canonical draft and retain their parent identity.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np

IMPORTED = "imported-rasters"
CANONICAL_EDIT = "canonical-from-import"
FORMAT = "darsia-geometry-imported-rasters-v1"
KEYS = {"labels", "facies", "facies_properties"}
PATH_KEYS = {"labels": "labels_path", "facies": "facies_source_path",
             "facies_properties": "facies_properties_source_path"}
SILENT_EDIT = (
    "Imported rasters are independently authoritative. Save cannot silently "
    "derive facies from labels or edit their sources/properties. Create an "
    "explicit canonical edit draft first; it will report the difference to "
    "the imported parent. Registered geometry still requires the project owner's decision."
)
# A registered run is either registered or provisional while a named person reviews it.
PROVISIONAL_STATUS = re.compile(r"provisional - \S(?:.*\S)? to review")


def registered_run_status(status):
    """True for 'registered' or 'provisional - <reviewer> to review'; the reviewer is recorded data."""
    return status == "registered" or (isinstance(status, str) and PROVISIONAL_STATUS.fullmatch(status) is not None)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def _same_grid_metadata(first, second):
    """Exact original image metadata, including Coordinate and class objects."""
    if type(first) is not type(second):
        return False
    if isinstance(first, np.ndarray):
        if first.dtype != second.dtype or first.shape != second.shape:
            return False
        if first.dtype.hasobject and not all(
                _same_grid_metadata(left, right) for left, right in zip(first.flat, second.flat)):
            return False
        return bool(np.array_equal(first, second))
    if isinstance(first, np.generic):
        return first.dtype == second.dtype and bool(np.array_equal(first, second))
    if isinstance(first, dict):
        return ({(type(key), key) for key in first} == {(type(key), key) for key in second}
                and all(_same_grid_metadata(first[key], second[key]) for key in first))
    if isinstance(first, (list, tuple)):
        return len(first) == len(second) and all(
            _same_grid_metadata(left, right) for left, right in zip(first, second))
    if isinstance(first, type):
        return first is second
    if first is None or type(first) in {str, int, float, bool, date, datetime}:
        return bool(first == second)
    return False


def content_digest(config):
    """Permit only the Store's existing depth materialization metadata changes."""
    value = deepcopy(dict(config))
    for key in ("provenance", "depth_path", "depth_artifact_path", "depth_source_reference"):
        value.pop(key, None)
    value.get("imported_rasters", {}).pop("configuration_sha256", None)
    return _digest(value)


def descriptor(path):
    from .preset_wasserstein_inputs import current_scope
    scope = current_scope()
    if scope is not None:
        return scope.descriptor(path)
    from .storage_paths import resolve_storage_path
    path = resolve_storage_path(path).resolve()
    from .setup_source_reads import source_descriptor
    return source_descriptor(path, reader=_read_descriptor)


def _read_descriptor(path):
    """Hash the original bytes without retaining the source contents."""
    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("An imported geometry source changed while being read: " + str(path))
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


def _verify(binding, path=None, *, descriptor_reader=None):
    """Compare the exact binding using the original or an explicit opt-in reader."""
    if (not isinstance(binding, dict) or not Path(str(binding.get("path", ""))).is_absolute()
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("sha256", ""))) is None
            or type(binding.get("size_bytes")) is not int or binding["size_bytes"] < 0):
        raise ValueError("Imported geometry requires explicit path/SHA-256/size bindings.")
    actual = (descriptor if descriptor_reader is None else descriptor_reader)(path or binding["path"])
    if any(actual[key] != binding[key] for key in ("sha256", "size_bytes")):
        raise ValueError("An imported geometry source changed: " + str(path or binding["path"]))
    return actual


def bind_imported_rasters(config, *, sources, diagnosis):
    """Opt in explicitly during preparation; this does not read or write maps."""
    if set(sources) != KEYS or not isinstance(diagnosis, dict):
        raise ValueError("Bind both original rasters, the FULL CSV, and the diagnosis.")
    candidate = deepcopy(dict(config))
    candidate["source_mode"] = IMPORTED
    candidate["imported_rasters"] = {
        "format": FORMAT, "sources": deepcopy(sources), "diagnosis": deepcopy(diagnosis),
        "independent_facies_raster": True, "full_properties_table": True,
        "edit_policy": "explicit canonical draft only; no silent recanonicalisation",
    }
    candidate["imported_rasters"]["configuration_sha256"] = content_digest(candidate)
    return candidate


def _full_csv(path, *, csv_bytes=None, strict=False, max_rows=None):
    """Read the full table, optionally validating bounded already-pinned bytes.

    Ordinary path callers retain the existing permissive CSV and numeric-ID
    rules. The explicit bytes branch shares this reader and preserves every
    string cell while requiring exact headers, row widths and int64 identities.
    """
    import io
    from decimal import Decimal, InvalidOperation

    if csv_bytes is not None:
        if type(csv_bytes) is not bytes:
            raise ValueError("The FULL original facies CSV needs explicit bytes.")
        if strict and b"\x00" in csv_bytes:
            raise ValueError("NUL in CSV.")
        stream = io.StringIO(csv_bytes.decode("utf-8-sig"), newline="")
    else:
        stream = Path(path).open("r", encoding="utf-8-sig", newline="")
    with stream:
        reader = csv.DictReader(stream, strict=strict)
        columns = reader.fieldnames or []
        names = {name.strip().casefold(): name for name in columns}
        identity = names.get("id") or names.get("facies_id")
        if identity is None:
            raise ValueError("The FULL original facies CSV has no id column.")
        if strict and (not columns or any(not name for name in names)
                or len(names) != len(columns) or "name" not in names
                or sum(key in names for key in ("id", "facies_id")) != 1):
            raise ValueError("Blank, duplicate or ambiguous CSV columns.")
        rows = []
        for row in reader:
            if strict and (None in row or any(value is None for value in row.values())):
                raise ValueError("Ragged CSV row.")
            if max_rows is not None and len(rows) >= max_rows:
                raise ValueError("The FULL original facies CSV has too many rows.")
            rows.append(row)
    ids = []
    for row in rows:
        text = str(row.get(identity, "")).strip()
        try:
            if strict:
                value = Decimal(text)
                if (not value.is_finite() or value != value.to_integral_value()
                        or not -(1 << 63) <= value <= (1 << 63) - 1):
                    raise ValueError
            else:
                value = float(text)
                if not np.isfinite(value) or not value.is_integer():
                    raise ValueError
            ids.append(int(value))
        except (ValueError, OverflowError, InvalidOperation):
            raise ValueError("The FULL original facies CSV has a noninteger ID.") from None
    if len(ids) != len(set(ids)):
        raise ValueError("The FULL original facies CSV has duplicate IDs.")
    return columns, rows, ids


def _integer_map(path):
    from .geometry_facies import _load_array
    value = np.asarray(_load_array(Path(path)))
    if value.ndim != 2 or not np.issubdtype(value.dtype, np.integer):
        raise ValueError("Imported labels and facies must be two-dimensional integer rasters.")
    return value


def imported_payload(config):
    """Verify immutable independently authoritative data, including full CSV IDs."""
    contract = config.get("imported_rasters") or {}
    if (config.get("source_mode") != IMPORTED or contract.get("format") != FORMAT
            or contract.get("independent_facies_raster") is not True
            or contract.get("full_properties_table") is not True
            or set(contract.get("sources") or {}) != KEYS
            or contract.get("configuration_sha256") != content_digest(config)):
        raise ValueError(SILENT_EDIT)
    sources = contract["sources"]
    actual = {name: _verify(sources[name], config.get(PATH_KEYS[name])) for name in KEYS}
    _verify(contract["diagnosis"])
    if any(Path(actual[name]["path"]).suffix.casefold() != ".npz"
           or Path(sources[name]["path"]).suffix.casefold() != ".npz"
           for name in ("labels", "facies")):
        raise ValueError("Imported raster authority requires original NPZ archives; PNG/TIFF bytes cannot be named NPZ.")
    if Path(actual["facies_properties"]["path"]).suffix.casefold() != ".csv":
        raise ValueError("Imported raster mode requires the full original CSV, without conversion.")
    labels = _integer_map(actual["labels"]["path"])
    facies = _integer_map(actual["facies"]["path"])
    if labels.shape != facies.shape:
        raise ValueError("Imported labels and facies grids differ; no resampling is allowed.")
    columns, rows, ids = _full_csv(actual["facies_properties"]["path"])
    missing = sorted(set(map(int, np.unique(facies))) - set(ids))
    if missing:
        raise ValueError("The FULL original CSV omits raster facies IDs: " + repr(missing))
    from .geometry_facies import _effective_run_geometry_choice
    run_geometry = config.get("run_geometry") or {}
    if any(_effective_run_geometry_choice(row) == "run_override" for row in run_geometry.get("runs") or []):
        raise ValueError("Imported raster revisions cannot silently add edited run overrides.")
    return dict(labels=labels, facies=facies, columns=columns, rows=rows, ids=ids,
                actual=actual, diagnosis=deepcopy(contract["diagnosis"]))


def difference_to_canonical(config, facies):
    """Compare actual ID arrays while distinguishing an explicit painted change."""
    from .geometry_facies import _canonical_facies, _layer_boundary_mask
    canonical, _ = _canonical_facies(config)
    if canonical.shape != facies.shape:
        raise ValueError("A canonical comparison changed the imported grid.")
    changed = canonical != facies
    labels = _integer_map(config["labels_path"])
    boundary = _layer_boundary_mask(labels, set(map(int, np.unique(labels))))
    count = int(np.count_nonzero(changed))
    total = int(changed.size)
    notice = ("imported rasters; facies raster independent of labels; "
              f"{count:,} pixels ({100 * count / total:.2f} %) differ from a label-derived facies map")
    if "painted_adoption_table" in config:
        notice = ("approved painted map and table compared with the retained parent; "
                  f"{count:,} pixels ({100 * count / total:.2f} %) have a different facies ID")
    return dict(pixel_count=total, different_pixels=count,
                different_fraction=count / total,
                different_pixels_on_label_boundary=int(np.count_nonzero(changed & boundary)),
                difference_rule="array values; no tolerance, splitting or interpolation",
                notice=notice)


def validate_source_mode(config, report):
    """Keep ordinary validation; don't show a label-derived pore-volume as truth."""
    if config.get("source_mode") not in {IMPORTED, CANONICAL_EDIT}:
        return report
    value = deepcopy(report)
    try:
        payload = (imported_payload(config) if config["source_mode"] == IMPORTED
                   else canonical_parent_payload(config))
        if "manual_label_edit" in config:
            manual_label_edit_payload(config)
        if config["source_mode"] == IMPORTED:
            detail = difference_to_canonical(config, payload["facies"])
            value["summary"].update(source_mode=IMPORTED, source_mode_notice=detail["notice"],
                                    imported_raster_difference=detail, facies_count=len(payload["ids"]),
                                    full_properties_ids=payload["ids"],
                                    estimated_pore_volume_m3=None, active_area_fraction=None,
                                    physical_summary_note="Independent raster properties; use the actual rebuilt-rig gate.")
        else:
            value["summary"].update(source_mode=CANONICAL_EDIT,
                                    source_mode_notice="Explicit canonical edit draft; differences to imported parent will be recorded.")
    except (ValueError, OSError, KeyError) as error:
        value["issues"].append(dict(severity="error", code="invalid_geometry_source_mode", message=str(error)))
        value["summary"]["error_count"] += 1
    return value


def prepare_canonical_edit(project, *, actor, reason):
    """Explicit opt-in only. No save, correction, labels, or registration occurs."""
    from .geometry_facies import GeometryRevisionStore
    if not str(actor).strip() or not str(reason).strip():
        raise ValueError("A canonical edit draft needs an explicit actor and reason.")
    store = GeometryRevisionStore(project)
    revision = store.current()
    config = deepcopy(revision.get("configuration") or {})
    imported_payload(config)
    revision_path = store.artifact_root / str(revision.get("revision_id")) / "revision.json"
    record = _verify(descriptor(revision_path))
    if json.loads(revision_path.read_bytes()) != revision:
        raise ValueError("The imported parent revision record differs from current.")
    parent = dict(revision=record, revision_id=revision["revision_id"],
                  imported_configuration=deepcopy(config), actor=str(actor).strip(), reason=str(reason).strip())
    config["source_mode"] = CANONICAL_EDIT
    config.pop("imported_rasters")
    config["canonical_import_parent"] = parent
    config["provenance"] = {"origin": "explicit canonical edit draft from imported rasters",
                            "parent_revision_id": revision["revision_id"],
                            "registered_geometry_authorized": False}
    return config


def canonical_parent_payload(config, *, run_id=None):
    """Read the retained imported parent or its explicit painted-table composition."""
    parent = config.get("canonical_import_parent") or {}
    record = _verify(parent.get("revision"))
    revision = json.loads(Path(record["path"]).read_bytes())
    old = parent.get("imported_configuration") or {}
    if (config.get("source_mode") != CANONICAL_EDIT or not parent.get("actor") or not parent.get("reason")
            or revision.get("revision_id") != parent.get("revision_id")
            or revision.get("configuration") != old):
        raise ValueError("A canonical edit lacks its actual imported parent and explicit opt-in.")
    payload = imported_payload(old)
    if "painted_adoption_table" in config:
        from .painted_adoption_table import canonical_payload
        return canonical_payload(config, parent, payload, run_id=run_id)
    if (str(config.get("facies_properties_source_path")) != str(old.get("facies_properties_source_path"))
            or config.get("grouping_mode") != "shared_facies"):
        raise ValueError("A canonical edit must keep the FULL original CSV and facies IDs.")
    labels = _integer_map(config["labels_path"])
    if labels.shape != payload["labels"].shape:
        raise ValueError("A canonical edit cannot silently change the imported grid.")
    _validate_registered_runs(config, payload, parent, run_id=run_id)
    from .geometry_facies import _load_properties
    props = _load_properties(Path(payload["actual"]["facies_properties"]["path"]))
    for row in config.get("regions") or []:
        actual = props.get(int(row["facies_id"]))
        if actual is None or any(row.get(key) != actual[key] for key in ("porosity", "permeability")):
            raise ValueError("A canonical edit cannot silently change the original physical CSV properties.")
    return payload



def integer_label_translation(labels, *, row_shift, column_shift, fill_label):
    """Translate categorical pixel values; explicit fill, no wrap/interpolation.

    This is a geometry-representation operation only. It never selects a shift,
    a fill category, an epoch, or a scientific acceptance threshold.
    """
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("A label translation needs a two-dimensional integer raster.")
    if any(type(value) is not int for value in (row_shift, column_shift, fill_label)):
        raise ValueError("Label translation offsets and fill must be explicit integers.")
    if fill_label not in set(map(int, np.unique(values))):
        raise ValueError("The explicit fill must be an existing parent label category.")
    height, width = values.shape
    if abs(row_shift) >= height or abs(column_shift) >= width:
        raise ValueError("A label translation must retain overlapping parent pixels.")
    source_r = slice(max(0, -row_shift), min(height, height - row_shift))
    source_c = slice(max(0, -column_shift), min(width, width - column_shift))
    target_r = slice(max(0, row_shift), min(height, height + row_shift))
    target_c = slice(max(0, column_shift), min(width, width + column_shift))
    translated = np.full_like(values, fill_label)
    translated[target_r, target_c] = values[source_r, source_c]
    retained = np.zeros(values.shape, dtype=bool)
    retained[source_r, source_c] = True
    ids, counts = np.unique(values[~retained], return_counts=True)
    metrics = dict(row_shift=row_shift, column_shift=column_shift, fill_label=fill_label,
                   source_shape=list(values.shape), source_dtype=values.dtype.str,
                   clipped_source_pixel_count=int(np.count_nonzero(~retained)),
                   clipped_source_label_counts={str(int(k)): int(v) for k, v in zip(ids, counts)},
                   introduced_fill_pixel_count=int(np.count_nonzero(~retained)),
                   interpolation=False, wrapping=False)
    return translated, metrics


def integer_label_edge_replication(labels, *, row_shift, column_shift, pixel_size_mm):
    """Explicit Q43 categorical shift and nearest covered-edge replication.

    Filled strips are flagged by exact half-open rectangles in revision metrics.
    These flags mean extrapolated geometry, not camera or physical-domain validity.
    Existing values, including 255, retain their categorical meaning.
    """
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("A label translation needs a two-dimensional integer raster.")
    if any(type(value) is not int for value in (row_shift, column_shift)):
        raise ValueError("Label translation offsets must be explicit integers.")
    registration_pixel_shift(dx_mm=0., dy_mm=0., pixel_size_mm=pixel_size_mm)
    height, width = values.shape
    if abs(row_shift) >= height or abs(column_shift) >= width:
        raise ValueError("A label translation must retain overlapping parent pixels.")
    rows = np.clip(np.arange(height) - row_shift, 0, height - 1)
    columns = np.clip(np.arange(width) - column_shift, 0, width - 1)
    translated = values[rows[:, None], columns[None, :]]
    source_r = slice(max(0, -row_shift), min(height, height - row_shift))
    source_c = slice(max(0, -column_shift), min(width, width - column_shift))
    retained = np.zeros(values.shape, dtype=bool)
    retained[source_r, source_c] = True
    ids, counts = np.unique(values[~retained], return_counts=True)
    r0, r1 = max(0, row_shift), min(height, height + row_shift)
    c0, c1 = max(0, column_shift), min(width, width + column_shift)
    rectangles = {"top": [0, r0, 0, width], "bottom": [r1, height, 0, width],
                  "left": [0, height, 0, c0], "right": [0, height, c1, width]}
    pixel_area_mm2 = float(pixel_size_mm) ** 2
    edges = {}
    for edge, bounds in rectangles.items():
        ar0, ar1, ac0, ac1 = bounds
        count = (ar1 - ar0) * (ac1 - ac0)
        edges[edge] = dict(bounds_rc=bounds, pixel_count=count,
                           area_mm2=count * pixel_area_mm2)
    union_count = height * width - (r1 - r0) * (c1 - c0)
    geometry_filled = dict(format="integer-translation-extrapolated-strips-v1",
        shape=list(values.shape), bounds="half-open [row_start,row_stop,column_start,column_stop]",
        pixel_count=union_count, area_mm2=union_count * pixel_area_mm2,
        pixel_size_mm=float(pixel_size_mm), edges=edges,
        corner_overlap_pixel_count=abs(row_shift) * abs(column_shift),
        edge_counts_include_corner_overlap=True,
        flag_rule="geometry_filled is true in the union of the four edge rectangles",
        meaning="extrapolated geometry; not camera or physical-domain validity")
    metrics = dict(row_shift=row_shift, column_shift=column_shift,
        fill_mode="edge-replication", source_shape=list(values.shape),
        source_dtype=values.dtype.str,
        clipped_source_pixel_count=int(np.count_nonzero(~retained)),
        clipped_source_label_counts={str(int(k)): int(v) for k,v in zip(ids,counts)},
        introduced_fill_pixel_count=union_count, geometry_filled=geometry_filled,
        interpolation=False, wrapping=False)
    return translated, metrics


def registration_pixel_shift(*, dx_mm, dy_mm, pixel_size_mm):
    """Explicit metric-to-integer display-grid conversion, with residuals."""
    values = (dx_mm, dy_mm, pixel_size_mm)
    if (any(not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_)) for value in values)
            or not all(np.isfinite(value) for value in values) or pixel_size_mm <= 0):
        raise ValueError("Registration displacement and explicit pixel size must be finite.")
    row, column = int(round(dy_mm / pixel_size_mm)), int(round(dx_mm / pixel_size_mm))
    return dict(row_shift=row, column_shift=column, pixel_size_mm=float(pixel_size_mm),
                candidate_dx_mm=float(dx_mm), candidate_dy_mm=float(dy_mm),
                represented_dx_mm=column * pixel_size_mm, represented_dy_mm=row * pixel_size_mm,
                residual_dx_mm=dx_mm - column * pixel_size_mm,
                residual_dy_mm=dy_mm - row * pixel_size_mm,
                sign_convention="positive dx right; positive dy down", interpolation=False)


CARRIED = "carried-reference-homography-v1"


def _carried_keys(value, keys, name):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("Carried " + name + " has an unknown or incomplete schema.")


def _carried_descriptor(value):
    _carried_keys(value, ("path", "sha256", "size_bytes"), "file binding")
    return _verify(value)


def _carried_data(reference):
    """Read only one explicitly bound JSON member; never discover other files."""
    _carried_keys(reference, ("document", "pointer"), "JSON reference")
    bound = reference["document"]
    actual = _carried_descriptor(bound)
    if bound["size_bytes"] > 1024 * 1024 or type(reference["pointer"]) is not str:
        raise ValueError("Carried evidence needs bounded JSON and an explicit pointer.")
    raw = Path(actual["path"]).read_bytes()
    if len(raw) != bound["size_bytes"] or hashlib.sha256(raw).hexdigest() != bound["sha256"]:
        raise ValueError("Carried JSON changed while being read.")
    value = json.loads(raw)
    pointer = reference["pointer"]
    if pointer and not pointer.startswith("/"):
        raise ValueError("Carried JSON pointers must start at the bound document root.")
    for token in pointer.split("/")[1:] if pointer else ():
        if re.search(r"~(?![01])", token):
            raise ValueError("Carried JSON pointer contains an invalid escape.")
        key = token.replace("~1", "/").replace("~0", "~")
        if type(value) is list:
            if re.fullmatch(r"0|[1-9][0-9]*", key) is None:
                raise ValueError("Carried JSON array pointer is not an explicit index.")
            value = value[int(key)]
        elif type(value) is dict:
            value = value[key]
        else:
            raise ValueError("Carried JSON pointer does not identify a member.")
    _carried_descriptor(bound)
    return value


def _carried_shape(value):
    if (type(value) is not list or len(value) != 2
            or any(type(item) is not int or item <= 0 for item in value)):
        raise ValueError("Carried native shape must contain two positive integers.")
    return tuple(value)


def _carried_matrix(value):
    if (type(value) is not list or len(value) != 3
            or any(type(row) is not list or len(row) != 3 for row in value)
            or any(type(item) not in {int, float} for row in value for item in row)):
        raise ValueError("Carried homography needs an explicit finite numeric 3x3 matrix.")
    matrix = np.asarray(value, np.float64)
    if not np.isfinite(matrix).all() or np.linalg.det(matrix) == 0:
        raise ValueError("Carried homography must be finite and invertible.")
    return matrix


def _carried_u8(labels):
    values = np.asarray(labels)
    if (values.ndim != 2 or not np.issubdtype(values.dtype, np.integer)
            or not values.size or np.min(values) < 0 or np.max(values) > 255):
        raise ValueError("Carried categorical values require a lossless integer 0..255 representation.")
    working = values.astype(np.uint8)
    if not np.array_equal(working.astype(values.dtype), values):
        raise ValueError("Carried categorical uint8 conversion is not lossless.")
    return values, working


def _carried_mask(value, shape):
    mask = np.asarray(value)
    if mask.dtype != np.dtype(bool) or mask.shape != tuple(shape):
        raise ValueError("Carried source flags must be exact boolean native-grid masks.")
    return mask


def _carried_array_identity(value):
    array = np.ascontiguousarray(value)
    return dict(dtype=array.dtype.str, shape=list(array.shape),
                sha256=hashlib.sha256(memoryview(array).cast("B")).hexdigest())


def carried_label_reference(labels, *, reference_shift, source_fill_status,
                            source_filled=None, source_support=None):
    """Apply the original integer primitive once to a DATA-selected reference."""
    values, _ = _carried_u8(labels)
    if type(reference_shift) is not dict:
        raise ValueError("Carried reference shift must be explicit typed metadata.")
    converted = registration_pixel_shift(dx_mm=reference_shift.get("candidate_dx_mm"),
        dy_mm=reference_shift.get("candidate_dy_mm"), pixel_size_mm=reference_shift.get("pixel_size_mm"))
    if not _same_grid_metadata(reference_shift, converted):
        raise ValueError("Carried reference shift differs from its explicit metric conversion.")
    if type(source_fill_status) is not str or source_fill_status not in {"KNOWN", "NOT_RECORDED"}:
        raise ValueError("Carried source fill status must remain explicit.")
    if (source_fill_status == "KNOWN") != (source_filled is not None):
        raise ValueError("Carried known source fill needs its actual mask; unknown fill cannot be invented.")
    arguments = dict(row_shift=converted["row_shift"], column_shift=converted["column_shift"],
                     pixel_size_mm=converted["pixel_size_mm"])
    shifted, metrics = integer_label_edge_replication(values, **arguments)
    filled = np.zeros(values.shape, bool)
    for edge in metrics["geometry_filled"]["edges"].values():
        r0, r1, c0, c1 = edge["bounds_rc"]
        filled[r0:r1, c0:c1] = True
    if source_filled is not None:
        inherited, _ = integer_label_edge_replication(
            _carried_mask(source_filled, values.shape).astype(np.uint8), **arguments)
        filled |= inherited.astype(bool)
    support = None
    if source_support is not None:
        support, _ = integer_label_edge_replication(
            _carried_mask(source_support, values.shape).astype(np.uint8), **arguments)
        support = support.astype(bool)
        # Replicated geometry does not establish previously unobserved support.
        for edge in metrics["geometry_filled"]["edges"].values():
            r0, r1, c0, c1 = edge["bounds_rc"]
            support[r0:r1, c0:c1] = False
    return dict(labels=shifted, geometry_filled=filled, support=support,
        metrics=dict(reference_shift=converted, source_fill_status=source_fill_status,
            input_labels=_carried_array_identity(values), labels=_carried_array_identity(shifted),
            geometry_filled=_carried_array_identity(filled),
            support=None if support is None else _carried_array_identity(support),
            original_edge_metrics=metrics))


def carry_label_reference(reference_labels, *, reference_matrix, target_matrix,
                          target_shape_hw, reference_filled, source_fill_status,
                          reference_support=None):
    """Original forward categorical homography, with explicit edge/flag policy.

    The matrix order is the established target @ inverse(reference) transport.
    The original nearest-neighbour uint8 primitive is retained. Replication is
    explicit here; the older native categorical helpers use a zero border.
    """
    import cv2
    values, working = _carried_u8(reference_labels)
    shape = _carried_shape(target_shape_hw)
    if type(source_fill_status) is not str or source_fill_status not in {"KNOWN", "NOT_RECORDED"}:
        raise ValueError("Carried source fill status must remain explicit.")
    source = _carried_matrix(reference_matrix)
    target = _carried_matrix(target_matrix)
    transform = target @ np.linalg.inv(source)
    if not np.isfinite(transform).all():
        raise ValueError("Carried composition is not finite.")
    dsize = (shape[1], shape[0])
    carried = cv2.warpPerspective(working, transform, dsize,
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE).astype(values.dtype)
    coverage = cv2.warpPerspective(np.ones(values.shape, np.uint8), transform, dsize,
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    inherited = cv2.warpPerspective(_carried_mask(reference_filled, values.shape).astype(np.uint8),
        transform, dsize, flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    filled = inherited | ~coverage
    support = None
    if reference_support is not None:
        support = cv2.warpPerspective(_carried_mask(reference_support, values.shape).astype(np.uint8),
            transform, dsize, flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    return dict(labels=carried, geometry_filled=filled, support=support,
        metrics=dict(composition=transform.tolist(), interpolation="nearest", border="edge-replication",
            source_fill_status=source_fill_status, source_labels=_carried_array_identity(values),
            labels=_carried_array_identity(carried), geometry_filled=_carried_array_identity(filled),
            uncovered_pixel_count=int(np.count_nonzero(~coverage)),
            inherited_filled_pixel_count=int(np.count_nonzero(inherited)),
            geometry_filled_pixel_count=int(np.count_nonzero(filled)),
            support=None if support is None else _carried_array_identity(support)))


def _carried_geometry(value, run, shape):
    _carried_keys(value, ("run_id", "matrix", "run_identity", "shape"), "geometry source")
    if (type(value["run_id"]) is not str or value["run_id"] != run
            or not _same_grid_metadata(value["matrix"]["document"], value["run_identity"]["document"])
            or not _same_grid_metadata(value["matrix"]["document"], value["shape"]["document"])
            or not _same_grid_metadata(_carried_data(value["run_identity"]), run)
            or _carried_shape(_carried_data(value["shape"])) != tuple(shape)):
        raise ValueError("Carried homography belongs to another run or native grid.")
    matrix = _carried_data(value["matrix"])
    _carried_matrix(matrix)
    return matrix


def _carried_chart(reference, run, geometry, source_image, shape, proof):
    value = _carried_data(reference)
    _carried_keys(value, ("format", "run_id", "source_image", "geometry_document",
                         "native_shape_hw", "offset_xy", "status", "proof"), "chart result")
    if (value["format"] != "darsia-native-raster-chart-v1" or value["status"] != "PASS"
            or not _same_grid_metadata(value["run_id"], run)
            or not _same_grid_metadata(value["source_image"], source_image)
            or not _same_grid_metadata(value["geometry_document"], geometry["matrix"]["document"])
            or _carried_shape(value["native_shape_hw"]) != tuple(shape)
            or not _same_grid_metadata(value["offset_xy"], [0, 0])
            or not _same_grid_metadata(value["proof"], proof)
            or _carried_data(value["proof"]) is not True):
        raise ValueError("Carried geometry lacks its positive source-bound raster chart.")
    _carried_descriptor(value["source_image"])


def _carried_load_mask(binding, shape, metadata):
    from .geometry_facies import _load_array, _darsia_metadata
    _carried_descriptor(binding)
    if Path(binding["path"]).suffix.casefold() != ".npz":
        raise ValueError("Carried flags require their original NPZ representation.")
    if not _same_grid_metadata(_darsia_metadata(Path(binding["path"])), metadata):
        raise ValueError("Carried flags changed native grid metadata.")
    return _carried_mask(_load_array(Path(binding["path"])), shape)


def _carried_file_bindings(record):
    """The closed new branch's consumed files, retained for original CAS rechecks."""
    evidence = _carried_data(record["carried_evidence"])
    recipe = _carried_data(evidence["recipe"])
    references = [record["carried_evidence"], evidence["recipe"], recipe["authority"],
                  recipe["chart_proof"], evidence["gate"], evidence["source_fill"]["provenance"],
                  evidence["reference_chart"], evidence["target_chart"]]
    for name in ("reference_geometry", "target_geometry"):
        references.extend(evidence[name][key] for key in ("matrix", "run_identity", "shape"))
    files = [reference["document"] for reference in references]
    files.extend(recipe["evaluators"])
    files.extend(evidence[key] for key in ("reference_labels", "reference_filled", "filled"))
    files.extend(evidence[key] for key in ("reference_support", "support") if evidence[key] is not None)
    files.extend(evidence["source_fill"][key] for key in ("mask", "support")
                 if evidence["source_fill"][key] is not None)
    for name in ("reference_chart", "target_chart"):
        files.append(_carried_data(evidence[name])["source_image"])
    unique = {}
    for bound in files:
        _carried_descriptor(bound)
        unique[(bound["path"], bound["sha256"], bound["size_bytes"])] = deepcopy(bound)
    return list(unique.values())


def _validate_carried_registered_run(config, payload, parent, row):
    from .geometry_facies import _darsia_metadata
    record = row["registration"]
    _carried_keys(record, ("kind", "parent_revision_id", "parent_labels", "status", "authority",
        "measurement", "labels", "source_image", "carried_evidence"), "registration")
    run = str(row["run_id"]).strip().casefold()
    if (record["status"] != "carried" or record["parent_revision_id"] != parent["revision_id"]
            or not _same_grid_metadata(record["parent_labels"], payload["actual"]["labels"])
            or str(row.get("facies_path") or "").strip()):
        raise ValueError("Carried registration needs its genuine imported parent and canonical facies.")
    _carried_descriptor(record["authority"])
    _carried_descriptor(record["measurement"])
    actual = _carried_descriptor(record["labels"])
    if (Path(actual["path"]).suffix.casefold() != ".npz"
            or Path(actual["path"]) != Path(row["labels_path"]).resolve()):
        raise ValueError("Carried registration lacks its actual NPZ labels.")
    metadata = _darsia_metadata(Path(payload["actual"]["labels"]["path"]))
    if not _same_grid_metadata(_darsia_metadata(Path(actual["path"])), metadata):
        raise ValueError("Carried labels changed the imported grid metadata.")
    evidence = _carried_data(record["carried_evidence"])
    _carried_keys(evidence, ("format", "run_id", "reference_run", "recipe", "reference_geometry",
        "target_geometry", "reference_chart", "target_chart", "source_fill", "reference_labels",
        "reference_filled", "reference_support", "filled", "support", "reference_metrics", "metrics", "gate"), "run evidence")
    if evidence["format"] != "darsia-carried-registration-evidence-v1" or evidence["run_id"] != run:
        raise ValueError("Carried evidence belongs to another run.")
    recipe = _carried_data(evidence["recipe"])
    _carried_keys(recipe, ("format", "reference_run", "reference_shift", "authority", "gate_limits",
                          "chart_proof", "evaluators", "source_fill"), "reference recipe")
    reference_run = evidence["reference_run"]
    if (type(reference_run) is not str or not reference_run or recipe["format"] != "darsia-carried-reference-recipe-v1"
            or not _same_grid_metadata(recipe["reference_run"], reference_run)
            or not _same_grid_metadata(recipe["authority"]["document"], record["authority"])):
        raise ValueError("Carried reference or authority differs from its admitted recipe.")
    _carried_data(recipe["authority"])
    shape = payload["labels"].shape
    reference_matrix = _carried_geometry(evidence["reference_geometry"], reference_run, shape)
    target_matrix = _carried_geometry(evidence["target_geometry"], run, shape)
    frame_runs = (config.get("frame_binding") or {}).get("runs") or {}
    photographs = {}
    for selected in (reference_run, run):
        photograph = ((frame_runs.get(selected) or {}).get("fields") or {}).get("shape_corrected_baseline")
        if type(photograph) is dict and set(photograph) == {
                "path", "sha256", "size_bytes", "mtime_ns", "shape", "dtype"}:
            # Keep the parent record intact; only the new file reference is projected.
            photograph = {key: deepcopy(photograph[key]) for key in ("path", "sha256", "size_bytes")}
        photographs[selected] = photograph
        _carried_descriptor(photographs[selected])
    source = _carried_descriptor(record["source_image"])
    if (Path(source["path"]) != Path(row["source_image_path"]).resolve()
            or not _same_grid_metadata(record["source_image"], photographs[run])):
        raise ValueError("Carried run photograph is not its bound corrected baseline.")
    _carried_chart(evidence["reference_chart"], reference_run, evidence["reference_geometry"], photographs[reference_run], shape, recipe["chart_proof"])
    _carried_chart(evidence["target_chart"], run, evidence["target_geometry"], photographs[run], shape, recipe["chart_proof"])
    fill = evidence["source_fill"]
    _carried_keys(fill, ("status", "mask", "support", "provenance"), "source fill lineage")
    if not _same_grid_metadata(fill, recipe["source_fill"]):
        raise ValueError("Carried source fill lineage differs from its admitted recipe.")
    _carried_data(fill["provenance"])
    source_mask = None if fill["mask"] is None else _carried_load_mask(fill["mask"], shape, metadata)
    source_support = None if fill["support"] is None else _carried_load_mask(fill["support"], shape, metadata)
    reference = carried_label_reference(payload["labels"], reference_shift=recipe["reference_shift"],
        source_fill_status=fill["status"], source_filled=source_mask, source_support=source_support)
    _carried_descriptor(evidence["reference_labels"])
    reference_labels = _integer_map(evidence["reference_labels"]["path"])
    if (reference_labels.dtype != reference["labels"].dtype or not np.array_equal(reference_labels, reference["labels"])
            or not _same_grid_metadata(_darsia_metadata(Path(evidence["reference_labels"]["path"])), metadata)
            or not _same_grid_metadata(evidence["reference_metrics"], reference["metrics"])):
        raise ValueError("Carried reference labels differ from their once-registered parent.")
    reference_filled = _carried_load_mask(evidence["reference_filled"], shape, metadata)
    if not np.array_equal(reference_filled, reference["geometry_filled"]):
        raise ValueError("Carried reference lost source or newly filled geometry.")
    reference_support = None if evidence["reference_support"] is None else _carried_load_mask(evidence["reference_support"], shape, metadata)
    if ((reference_support is None) != (reference["support"] is None)
            or (reference_support is not None and not np.array_equal(reference_support, reference["support"]))):
        raise ValueError("Carried reference support differs from its actual source.")
    target = carry_label_reference(reference_labels, reference_matrix=reference_matrix,
        target_matrix=target_matrix, target_shape_hw=list(shape), reference_filled=reference_filled,
        source_fill_status=fill["status"], reference_support=reference_support)
    labels = _integer_map(actual["path"])
    filled = _carried_load_mask(evidence["filled"], shape, metadata)
    support = None if evidence["support"] is None else _carried_load_mask(evidence["support"], shape, metadata)
    if (labels.dtype != target["labels"].dtype or not np.array_equal(labels, target["labels"])
            or not np.array_equal(filled, target["geometry_filled"])
            or (support is None) != (target["support"] is None)
            or (support is not None and not np.array_equal(support, target["support"]))
            or not _same_grid_metadata(evidence["metrics"], target["metrics"])):
        raise ValueError("Carried labels, fill lineage or metrics differ from the exact source transport.")
    gate = _carried_data(evidence["gate"])
    _carried_keys(gate, ("format", "status", "run_id", "reference_run", "labels", "recipe", "target_geometry",
        "reference_geometry", "target_chart", "sharp_boundary_count", "median_mm", "maximum_mm",
        "missing_prior_witnesses", "ambiguous_boundaries", "evaluators"), "boundary gate")
    limits = recipe["gate_limits"]
    _carried_keys(limits, ("median_mm", "maximum_mm"), "gate limits")
    if (gate["format"] != "darsia-carried-boundary-gate-v1" or gate["status"] != "PASS"
            or not _same_grid_metadata(record["measurement"], evidence["gate"]["document"])
            or not _same_grid_metadata(gate["run_id"], run)
            or not _same_grid_metadata(gate["reference_run"], reference_run)
            or any(not _same_grid_metadata(gate[key], value) for key, value in
                (("labels", record["labels"]), ("recipe", evidence["recipe"]),
                 ("target_geometry", evidence["target_geometry"]), ("reference_geometry", evidence["reference_geometry"]),
                 ("target_chart", evidence["target_chart"])))
            or type(gate["sharp_boundary_count"]) is not int or gate["sharp_boundary_count"] <= 0
            or not _same_grid_metadata(gate["missing_prior_witnesses"], [])
            or not _same_grid_metadata(gate["ambiguous_boundaries"], [])
            or type(gate["evaluators"]) is not list or not gate["evaluators"]
            or not _same_grid_metadata(gate["evaluators"], recipe["evaluators"])):
        raise ValueError("Carried labels lack their positive exact source-bound boundary gate.")
    for key in ("median_mm", "maximum_mm"):
        if (type(gate[key]) is not float or type(limits[key]) is not float
                or not np.isfinite(gate[key]) or not np.isfinite(limits[key])
                or gate[key] < 0 or limits[key] < 0 or gate[key] > limits[key]):
            raise ValueError("Carried boundary gate exceeds its explicit finite metric limits.")
    if gate["median_mm"] > gate["maximum_mm"]:
        raise ValueError("Carried boundary gate statistics are inconsistent.")
    for evaluator in gate["evaluators"]:
        _carried_descriptor(evaluator)
    _carried_file_bindings(record)


PAINTED = "painted-map-placement-v1"
PAINTED_V2 = "painted-map-placement-v2"
PAINTED_PREPARATION = "darsia-fluidflower-painted-map-adoption-preparation-v1"
PAINTED_PREPARATION_V2 = "darsia-fluidflower-painted-map-adoption-preparation-v2"


def painted_map_composition(reference_model, target_model, *, geometry_index=None):
    """The matrices that carry a painted map from the reference run's camera
    grid to a target run's rig-corrected native grid, and that grid's shape.

    Both models are measured run geometries (darsia-measured-run-geometry-v1):
    the reference matrix is the reference run's metric-to-camera homography;
    the target matrix is the target's camera-to-native homography times its
    metric-to-camera homography. carry_label_reference composes them as
    target @ inverse(reference).
    """
    grid = target_model.get("baseline_native_grid") or {}
    to_native = _carried_matrix(grid.get("H_baseline_camera_u_to_rig_corrected_native"))
    if geometry_index is None:
        reference = _carried_matrix(reference_model.get("H_metric_mm_to_camera_u"))
        target = _carried_matrix(target_model.get("H_metric_mm_to_camera_u"))
    else:
        from .rig_point_alignment import effective_model_matrix
        reference = effective_model_matrix(reference_model, geometry_index)
        target = effective_model_matrix(target_model, geometry_index)
    return reference, to_native @ target, _carried_shape(grid.get("shape_hw"))


def _validate_painted_run(config, payload, parent, row, *, continuation=False):
    """A run's labels carried from an approved painted map: replayed exactly.

    The record binds the painted map on the reference run's camera grid, both
    measured geometry models, the adoption preparation record (measurement)
    and the map approval (authority). The run's labels must equal the
    existing nearest-neighbour carrier applied to the map with the models'
    composition, on the imported grid with its metadata; facies stay
    derived canonically and the photograph is the run's bound corrected
    baseline.
    """
    from .geometry_facies import _darsia_metadata
    record = row["registration"]
    keys = ("kind", "parent_revision_id", "status", "authority", "measurement",
        "labels", "source_image", "source_map", "reference_model", "target_model")
    _carried_keys(record, keys + (("continuation",) if continuation else ()), "painted registration")
    run = str(row["run_id"]).strip().casefold()
    if (record["status"] != "painted" or record["parent_revision_id"] != parent["revision_id"]
            or str(row.get("facies_path") or "").strip()):
        raise ValueError("A painted run needs its genuine imported parent and canonical facies.")
    local_reuse = (config.get("provenance") or {}).get("local_map_reuse")
    if local_reuse is not None:
        from .painted_adoption_table import _checked, _same
        from .local_map_adoption import PROOF_FORMAT, validate_local_registered_run
        local_preparation = _checked(record["measurement"], metadata=True)
        proof = local_preparation.get("painting_geometry") or {}
        if (not continuation or proof.get("format") != PROOF_FORMAT
                or not _same(local_reuse, dict(format=PROOF_FORMAT, review=proof.get("review")))):
            raise ValueError("Reused reviewed maps require their exact V2 source proof")
        validate_local_registered_run(config, payload, row, local_preparation,
                                      Path(record["measurement"]["path"]).parent)
        return
    authority = _carried_descriptor(record["authority"])
    measurement = _carried_descriptor(record["measurement"])
    actual = _carried_descriptor(record["labels"])
    if (Path(actual["path"]).suffix.casefold() != ".npz"
            or Path(actual["path"]).resolve() != Path(row["labels_path"]).resolve()):
        raise ValueError("A painted run lacks its actual NPZ labels.")
    fields = ((((config.get("frame_binding") or {}).get("runs") or {}).get(run) or {}).get("fields") or {})
    photograph = fields.get("shape_corrected_baseline") or {}
    source = _carried_descriptor(record["source_image"])
    if (any(source[key] != photograph.get(key) for key in ("path", "sha256", "size_bytes"))
            or Path(source["path"]).resolve() != Path(row["source_image_path"]).resolve()):
        raise ValueError("A painted run's photograph is not its bound corrected baseline.")
    models = []
    for name in ("reference_model", "target_model"):
        bound = _carried_descriptor(record[name])
        models.append(json.loads(Path(bound["path"]).read_bytes()))
    preparation = json.loads(Path(measurement["path"]).read_bytes())
    reference_matrix, target_matrix, shape = painted_map_composition(*models,
        geometry_index=preparation.get("geometry_index"))
    if shape != payload["labels"].shape:
        raise ValueError("A painted run's native grid is not the imported grid.")
    source_map = _carried_descriptor(record["source_map"])
    painted = _integer_map(source_map["path"])
    carried = carry_label_reference(painted, reference_matrix=reference_matrix.tolist(),
        target_matrix=target_matrix.tolist(), target_shape_hw=list(shape),
        reference_filled=np.zeros(painted.shape, dtype=bool), source_fill_status="KNOWN",
        reference_support=painted != 0)
    labels = _integer_map(actual["path"])
    entry = next((item for item in preparation.get("runs") or [] if item.get("run_id") == run), {})
    base = labels
    expected_labels = entry.get("labels") or {}
    if "target_corrections" in preparation:
        from .painted_target_integration import effective_product
        from .painted_adoption_table import _local
        effective, effective_bound = effective_product(preparation, Path(measurement["path"]).parent, entry)
        base = _integer_map(_local(Path(measurement["path"]).parent, entry["labels"])["path"])
        expected_labels = effective_bound
        if not np.array_equal(labels, effective["array"]):
            raise ValueError("Painted labels differ from their explicit approved target edit")
    if labels.shape != shape or not np.array_equal(base, carried["labels"].astype(base.dtype)):
        raise ValueError("Painted labels differ from the exact replay of their placement.")
    from .raw_shape_basis_reader import has_shape_basis
    if has_shape_basis(models[-1]):
        from .painted_adoption_geometry import basis_label_fields
        from .painted_adoption_table import _same
        expected = basis_label_fields(preparation["painting_facies"],
            preparation["painting_geometry"], run, Path(measurement["path"]).parent)
        if expected is None or not _same(entry.get("current_labels"), expected["source"]):
            raise ValueError("Painted labels lost this run's exact RAW-basis parent label identity.")
        metadata = expected["metadata"]
    else:
        metadata = _darsia_metadata(Path(payload["actual"]["labels"]["path"]))
    if not _same_grid_metadata(_darsia_metadata(Path(actual["path"])), metadata):
        raise ValueError("Painted labels changed the imported grid metadata.")
    approval = json.loads(Path(authority["path"]).read_bytes())
    expected_format = PAINTED_PREPARATION_V2 if continuation else PAINTED_PREPARATION
    if (preparation.get("format") != expected_format or entry.get("status") != "prepared"
            or expected_labels.get("sha256") != actual["sha256"]
            or (preparation.get("source_map") or {}).get("sha256") != source_map["sha256"]
            or approval.get("sha256") != (preparation.get("map_version") or {}).get("sha256")):
        raise ValueError("A painted run is not the run, map and approval its preparation names.")
    if continuation:
        from .photo_edge_continuation import load_contract, verify_source, verify_product, verify_native_proof, POLICY
        sidecar = _carried_descriptor(record["continuation"])
        expected_sidecar = Path(measurement["path"]).parent / "continuation.json"
        declared = preparation.get("continuation") or {}
        if (Path(sidecar["path"]).resolve() != expected_sidecar.resolve()
                or declared.get("file") != "continuation.json" or declared.get("policy") != POLICY
                or declared.get("adopted") is not False
                or any(declared.get(k) != sidecar[k] for k in ("sha256", "size_bytes"))):
            raise ValueError("The V2 run has another continuation sidecar.")
        contract = load_contract(sidecar["path"], run)
        if (contract["target_grid"] != "native" or contract["source_run"] != preparation.get("reference_run")
                or contract["source_model"]["descriptor"] != record["reference_model"]
                or contract["target_model"]["descriptor"] != record["target_model"]):
            raise ValueError("The V2 continuation is bound to other models, runs or native grid.")
        binding = preparation.get("approved_source")
        verify_source(binding, painted, {run: contract},
            preparation=preparation if "painting_facies" in preparation else None, folder=Path(measurement["path"]).parent)
        if approval != binding["approval"] or preparation.get("approval") != approval:
            raise ValueError("The V2 source approval differs from its immutable approved version.")
        from .painted_adoption_table import _local
        verify_product(_local(Path(measurement["path"]).parent, entry["labels"])["path"],
                       painted, contract, counts=entry.get("continuation"))
        verify_native_proof(preparation, Path(measurement["path"]).parent)
    _verify(actual)
    _verify(source_map)
    if "analysis_exclusions" in config:
        from .painted_analysis_exclusions import configuration_exclusion
        configuration_exclusion(config, run, labels=labels)


def _validate_registered_runs(config, payload, parent, *, run_id=None):
    """Admit explicit original-Store run rows, not unbound arbitrary overrides."""
    from .geometry_facies import _darsia_metadata, _effective_run_geometry_choice
    from .adoption_check_progress import report_check_progress
    rows = (config.get("run_geometry") or {}).get("runs") or []
    overrides = [row for row in rows if _effective_run_geometry_choice(row) == "run_override"]
    if overrides:
        old = parent["imported_configuration"]
        if any(config.get(key) != old.get(key) for key in
               ("frame", "width_m", "height_m", "frame_binding", "baseline_path")):
            raise ValueError("Registered runs cannot change their imported frame or photograph bindings.")
    if run_id is not None:
        key = str(run_id).strip().casefold()
        rows = [row for row in rows if str(row.get("run_id") or "").strip().casefold() == key]
        if len(rows) != 1 or _effective_run_geometry_choice(rows[0]) != "run_override":
            raise ValueError("The selected run has no explicit registered override.")
    seen = set()
    override_count = sum(_effective_run_geometry_choice(row) == "run_override" for row in rows)
    for row in rows:
        if _effective_run_geometry_choice(row) != "run_override":
            continue
        run = str(row.get("run_id") or "").strip().casefold()
        if not run or run in seen or not all(c.isalnum() or c in "-_" for c in run):
            raise ValueError("Registered run identifiers must be explicit and unique.")
        seen.add(run)
        report_check_progress(f"Validating registration for {run.upper()} ({len(seen)} of {override_count} runs)…")
        record = row.get("registration") or {}
        if record.get("kind") == CARRIED:
            _validate_carried_registered_run(config, payload, parent, row)
            continue
        if record.get("kind") == PAINTED:
            _validate_painted_run(config, payload, parent, row)
            continue
        if record.get("kind") == PAINTED_V2:
            _validate_painted_run(config, payload, parent, row, continuation=True)
            continue
        if (record.get("parent_revision_id") != parent["revision_id"]
                or record.get("parent_labels") != payload["actual"]["labels"]
                or not registered_run_status(record.get("status"))):
            raise ValueError("A registered run needs its actual imported parent and explicit status.")
        _verify(record.get("authority"))
        _verify(record.get("measurement"))
        actual = _verify(record.get("labels"), row.get("labels_path"))
        if (Path(actual["path"]).suffix.casefold() != ".npz"
                or str(row.get("facies_path") or "").strip()):
            raise ValueError("Registered run facies must be derived canonically from its NPZ labels.")
        target = _integer_map(actual["path"])
        metric = record.get("candidate") or {}
        converted = registration_pixel_shift(dx_mm=metric.get("candidate_dx_mm"),
            dy_mm=metric.get("candidate_dy_mm"), pixel_size_mm=metric.get("pixel_size_mm"))
        if (metric != converted or any(record.get(key) != converted[key]
                                      for key in ("row_shift", "column_shift"))):
            raise ValueError("Registered candidate metadata disagrees with its integer pixel shift.")
        fill_mode = record.get("fill_mode")
        if fill_mode == "edge-replication":
            if "fill_label" in record:
                raise ValueError("Edge replication cannot substitute a constant fill label.")
            expected, metrics = integer_label_edge_replication(payload["labels"],
                row_shift=record.get("row_shift"), column_shift=record.get("column_shift"),
                pixel_size_mm=converted["pixel_size_mm"])
        elif fill_mode is None:
            expected, metrics = integer_label_translation(payload["labels"], row_shift=record.get("row_shift"),
                column_shift=record.get("column_shift"), fill_label=record.get("fill_label"))
        else:
            raise ValueError("An unknown registered geometry fill mode is forbidden.")
        if record.get("metrics") != metrics:
            raise ValueError("Registered translation metrics disagree with the actual parent pixels.")
        if target.dtype != expected.dtype or not np.array_equal(target, expected):
            raise ValueError("Registered labels differ from the explicit integer parent translation.")
        if not _same_grid_metadata(_darsia_metadata(Path(actual["path"])),
                _darsia_metadata(Path(payload["actual"]["labels"]["path"]))):
            raise ValueError("Registered labels changed the imported grid metadata.")
        source = _verify(record.get("source_image"), row.get("source_image_path"))
        binding = config.get("frame_binding") or {}
        if binding:
            fields = ((binding.get("runs") or {}).get(run) or {}).get("fields") or {}
            declared = fields.get("shape_corrected_baseline") or {}
            if any(source.get(key) != declared.get(key) for key in ("path", "sha256", "size_bytes")):
                raise ValueError("Registered run photograph is not its existing bound corrected baseline.")


def registered_run_sources(project_root, current, run_id):
    """Read one genuine saved Store run; no new map, cache, or fallback.

    Store admission still validates all run rows by default. This explicit
    selected-run read verifies only the selected translation and its sources.
    """
    from .geometry_facies import GEOMETRY_REVISION_FORMAT, _canonical_facies, _darsia_metadata, resolve_run_geometry_artifacts
    root = Path(project_root).resolve()
    current_path = root / "overrides" / "geometry_facies" / "current.json"
    if current_path.stat().st_size > 1024 * 1024:
        raise ValueError("Registered current metadata exceeds 1 MiB.")
    current_file = descriptor(current_path)
    saved = json.loads(current_path.read_bytes())
    revision_id = current.get("revision_id")
    if (saved != current or current.get("schema") != 1
            or current.get("format") != GEOMETRY_REVISION_FORMAT
            or re.fullmatch(r"[0-9a-f]{32}", str(revision_id)) is None):
        raise ValueError("Registered geometry needs its genuine current Store UUID and record.")
    revision_path = root / "results" / "geometry" / revision_id / "revision.json"
    if revision_path.stat().st_size > 1024 * 1024:
        raise ValueError("Registered revision metadata exceeds 1 MiB.")
    revision_file = descriptor(revision_path)
    if json.loads(revision_path.read_bytes()) != current:
        raise ValueError("Registered current differs from the actual immutable Store revision.")
    config = current["configuration"]
    if "painted_adoption_table" in config:
        from .painted_store_replay import registered_run
        return registered_run(current, str(run_id).strip().casefold(),
            current_pin=current_file, revision_pin=revision_file)
    payload = canonical_parent_payload(config, run_id=run_id)
    run = str(run_id).strip().casefold()
    row = next(row for row in config["run_geometry"]["runs"]
               if str(row.get("run_id") or "").strip().casefold() == run)
    artifacts = resolve_run_geometry_artifacts(current, run)
    if artifacts.get("scope") != "run_override" or artifacts.get("run_id") != run:
        raise ValueError("No genuine Store run artifact exists; series fallback is forbidden.")
    destination = root / "results" / "geometry" / revision_id / "runs" / run
    files = {}
    for name, filename in (("labels", "labels.npz"), ("facies", "facies.npz"),
                           ("facies_properties", "facies_properties.csv")):
        path = Path(str(artifacts.get(name + "_path") or "")).resolve()
        if path != destination / filename:
            raise ValueError("Registered artifacts belong to another Store UUID or run.")
        files[name] = descriptor(path)
    labels = _integer_map(files["labels"]["path"])
    expected = _integer_map(row["registration"]["labels"]["path"])
    if labels.dtype != np.dtype(np.int32) or not np.array_equal(labels, expected):
        raise ValueError("Stored registered labels differ numerically from the admitted translation.")
    if not _same_grid_metadata(_darsia_metadata(Path(files["labels"]["path"])),
            _darsia_metadata(Path(row["registration"]["labels"]["path"]))):
        raise ValueError("Stored registered labels changed their grid metadata.")
    run_config = deepcopy(config)
    run_config["labels_path"] = files["labels"]["path"]
    expected_facies, _ = _canonical_facies(run_config, run_id=run)
    facies = _integer_map(files["facies"]["path"])
    if facies.dtype != np.dtype(np.int32) or not np.array_equal(facies, expected_facies):
        raise ValueError("Stored registered facies are not the exact canonical label mapping.")
    if not _same_grid_metadata(_darsia_metadata(Path(files["facies"]["path"])),
            _darsia_metadata(Path(files["labels"]["path"]))):
        raise ValueError("Stored registered facies changed their grid metadata.")
    if "painted_table_bytes" in payload:
        if Path(files["facies_properties"]["path"]).read_bytes() != payload["painted_table_bytes"]:
            raise ValueError("Stored registered run lost its exact derived painting properties CSV.")
    elif Path(files["facies_properties"]["path"]).read_bytes() != Path(payload["actual"]["facies_properties"]["path"]).read_bytes():
        raise ValueError("Stored registered run lost the FULL original properties CSV.")
    guarded = [current_file, revision_file, config["canonical_import_parent"]["revision"],
               payload["diagnosis"], *payload["actual"].values(),
               *[row["registration"][key] for key in ("authority", "measurement", "labels", "source_image")]]
    if row["registration"].get("kind") == CARRIED:
        guarded.extend(_carried_file_bindings(row["registration"]))
    if row["registration"].get("kind") in {PAINTED, PAINTED_V2}:
        guarded.extend(row["registration"][key] for key in ("source_map", "reference_model", "target_model"))
    if "painted_table_guarded_sources" in payload:
        guarded.extend(payload["painted_table_guarded_sources"])
    if row["registration"].get("kind") == PAINTED_V2:
        from .photo_edge_continuation import verify_native_proof
        guarded.append(row["registration"]["continuation"])
        preparation = json.loads(Path(row["registration"]["measurement"]["path"]).read_bytes())
        guarded.extend(verify_native_proof(preparation, Path(row["registration"]["measurement"]["path"]).parent))
        approved = preparation["approved_source"]
        guarded.extend([approved["version"], *approved["files"].values(),
            approved["frame_basis"]["workspace_revision"], approved["frame_basis"]["photograph"]])
        if approved["frame_basis"]["camera_profile"]["kind"] == "camera_u":
            guarded.append(approved["frame_basis"]["camera_profile"]["parameters"])
    result = dict(revision_id=revision_id, run_id=run, current=current_file,
        revision=revision_file, files=files, labels_shape=list(labels.shape),
        registration=deepcopy(row["registration"]), guarded_sources=guarded,
        label_serialization="Original Store int32; exact numeric equality to parent-typed translation",
        provenance="Fresh observed Store product SHA bound to genuine UUID/record and exact arrays; not historical source SHA")
    if row["registration"].get("kind") == CARRIED:
        result["label_serialization"] = "Original Store int32; exact numeric equality to admitted carried labels"
    if row["registration"].get("kind") in {PAINTED, PAINTED_V2}:
        result["label_serialization"] = "Original Store int32; exact numeric equality to the replayed painted placement"
    verify_registered_sources(result)
    return result


def verify_registered_sources(binding):
    """Recheck the exact already-admitted files without decoding them again."""
    if "committed_painted_table" in binding:
        from .painted_store_replay import verify
        return verify(binding)
    for row in [*binding["guarded_sources"], *binding["files"].values()]:
        _verify(row)
    if (binding.get("registration") or {}).get("kind") == PAINTED_V2:
        # Positive V2 also rechecks actual approval history/profile/stat source.
        # Historical registrations retain the original descriptor-only route.
        from .photo_edge_continuation import load_contract, verify_source, verify_product, verify_native_proof
        record = binding["registration"]
        preparation = json.loads(Path(record["measurement"]["path"]).read_bytes())
        run = binding["run_id"]
        from .local_map_adoption import PROOF_FORMAT, verify_local_selected
        if (preparation.get("painting_geometry") or {}).get("format") == PROOF_FORMAT:
            folder = Path(record["measurement"]["path"]).parent
            entry = next(row for row in preparation["runs"] if row["run_id"] == run)
            verify_local_selected(preparation, folder, entry)
            from .painted_target_integration import effective_product
            _, effective = effective_product(preparation, folder, entry)
            if any(binding["files"]["labels"][k] != effective[k] for k in ("sha256", "size_bytes")):
                raise ValueError("Stored labels differ from the exact reused effective map")
            return
        contract = load_contract(record["continuation"]["path"], run)
        source = _integer_map(record["source_map"]["path"])
        verify_source(preparation["approved_source"], source, {run: contract},
            preparation=preparation if "painting_facies" in preparation else None, folder=Path(record["measurement"]["path"]).parent)
        entry = next(row for row in preparation["runs"] if row.get("run_id") == run)
        if "target_corrections" in preparation:
            # Original continuation evidence and the separate effective archive
            # retain their own exact validators; Store copied the latter bytes.
            from .painted_adoption_table import _local
            from .painted_target_integration import effective_product
            folder = Path(record["measurement"]["path"]).parent
            base = _local(folder, entry["labels"])
            verify_product(base["path"], source, contract, counts=entry.get("continuation"))
            _, effective_bound = effective_product(preparation, folder, entry)
            if (Path(record["labels"]["path"]).resolve() != Path(effective_bound["path"]).resolve()
                    or any(record["labels"][key] != effective_bound[key] for key in ("sha256", "size_bytes"))):
                raise ValueError("The registered V2 labels differ from their exact approved effective product.")
            if any(binding["files"]["labels"][key] != effective_bound[key] for key in ("sha256", "size_bytes")):
                raise ValueError("The Store V2 labels differ from the exact effective product archive.")
        else:
            verify_product(record["labels"]["path"], source, contract, counts=entry.get("continuation"))
            verify_product(binding["files"]["labels"]["path"], source, contract, counts=entry.get("continuation"))
        verify_native_proof(preparation, Path(record["measurement"]["path"]).parent)

def require_source_mode_transition(store, config):
    """The simple editor cannot remove the mode and bypass its authority."""
    mode = config.get("source_mode")
    if mode not in {None, "region-derived", IMPORTED, CANONICAL_EDIT}:
        raise ValueError("Unknown geometry source mode.")
    if mode == IMPORTED:
        imported_payload(config)
    elif mode == CANONICAL_EDIT:
        canonical_parent_payload(config)
    if store.current_path.is_file():
        current = store.current()
    else:
        # Ordinary first Save previously did not construct a project-derived
        # default configuration. Existing history still carries authority even
        # if current.json is absent; never bypass an imported historical parent.
        history = store.revisions()
        current = history[-1] if history else {}
    old = current.get("configuration") or {}
    if "painted_adoption_table" in old and "painted_adoption_table" not in config:
        raise ValueError("A painted adoption cannot silently discard its derived table authority.")
    if old.get("source_mode") in {IMPORTED, CANONICAL_EDIT} and mode not in {IMPORTED, CANONICAL_EDIT}:
        raise ValueError(SILENT_EDIT)
    if old.get("source_mode") == IMPORTED and mode != IMPORTED:
        parent = config.get("canonical_import_parent") or {}
        if mode != CANONICAL_EDIT or parent.get("revision_id") != current.get("revision_id"):
            raise ValueError(SILENT_EDIT)


def materialize_source_mode(config, staging, destination):
    """Copy original imported artifacts or materialize an explicit canonical edit."""
    from .geometry_facies import _canonical_facies, _darsia_metadata
    mode = config["source_mode"]
    payload = imported_payload(config) if mode == IMPORTED else canonical_parent_payload(config)
    copied = {}
    for name in ("labels", "facies", "facies_properties"):
        target = Path(staging) / ("facies_properties.csv" if name == "facies_properties" else name + ".npz")
        if name == "facies_properties" and "painted_table_bytes" in payload:
            raw = payload["painted_table_bytes"]
            with target.open("xb") as outgoing:
                outgoing.write(raw)
            expected = {"path": str(target.resolve()), "sha256": hashlib.sha256(raw).hexdigest(), "size_bytes": len(raw)}
            _verify(expected)
        elif mode == CANONICAL_EDIT and name == "facies":
            facies, _ = _canonical_facies(config)
            np.savez_compressed(target, array=facies, metadata=_darsia_metadata(Path(config["labels_path"])))
        else:
            source = (config["labels_path"] if mode == CANONICAL_EDIT and name == "labels"
                      else payload["actual"][name]["path"])
            expected = descriptor(source)
            with Path(source).open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            _verify(expected, target)
            _verify(expected)
        row = descriptor(target)
        row["path"] = str(Path(destination) / target.name)
        copied[name] = row
    detail = difference_to_canonical(config, payload["facies"])
    if mode == CANONICAL_EDIT:
        labels = _integer_map(config["labels_path"])
        detail = {**detail, "labels_changed_from_parent": int(np.count_nonzero(labels != payload["labels"])),
                  "parent_revision": deepcopy(config["canonical_import_parent"]["revision"]),
                  "known_change": "Explicit canonical derivation from edited labels and the full original CSV; not an adopted-v1 identity gate."}
    if "painted_table_bytes" in payload:
        for source in payload["painted_table_guarded_sources"]:
            _verify(source)
        detail["known_change"] = ("Explicit approved-painting table derivation; original CSV preserved, "
                                  "porous extra cells carried by name and nonporous extra cells explicitly blank.")
    return dict(source_mode=mode, labels_path=copied["labels"]["path"],
                source_mode_files=copied, source_mode_difference=detail,
                source_mode_diagnosis=payload["diagnosis"]), payload["rows"]


MANUAL_LABEL_EDIT = "darsia-private-manual-label-edit-provenance-v1"


def _private_manual_context(project_root, manifest):
    root=Path(project_root).resolve()
    value=((manifest.get("tools") or {}).get("native_darsia") or {}).get("manual_label_edit_validation")
    keys={"format","authority_id","selected_run","parent_revision","private_series_root"}
    if (not isinstance(value,dict) or set(value)!=keys or value["format"]!=MANUAL_LABEL_EDIT
            or value["authority_id"]!="Q121" or not str(value["selected_run"])
            or str(value["private_series_root"]).startswith(("\\\\","//"))
            or Path(value["private_series_root"]).resolve()!=root.parent
            or Path(((manifest.get("tools") or {}).get("native_series") or {}).get("series_root", "")).resolve()!=root.parent
            or root.name!="geometry_facies"):
        raise ValueError("Manual validation needs its exact local private series/project binding.")
    _verify(value["parent_revision"])
    return deepcopy(value)


def bind_private_manual_label_edit(project, config, session, materialization):
    """Bind actual original editor outputs only in the explicit private probe.

    No source map, event, correction, registration or pixel value is changed.
    The ordinary editor and all old source modes retain their existing route.
    """
    context=((project.manifest.get("tools") or {}).get("native_darsia") or {}).get("manual_label_edit_validation")
    if context is None:
        return config
    context=_private_manual_context(project.root,project.manifest)
    value=deepcopy(config)
    value["manual_label_edit"]=dict(format=MANUAL_LABEL_EDIT,context=context,
        source_labels=descriptor(session.base_labels_path),session=descriptor(session.manifest_path),
        events=descriptor(session.events_path),materialization=descriptor(materialization["manifest_path"]),
        materialized_labels=descriptor(materialization["labels_path"]))
    manual_label_edit_payload(value)
    return value


def manual_label_edit_payload(config):
    """Verify parent/grid and replay the UNCHANGED original editor events.

    The replay view holds no writer paths or constructor side effects. It calls
    the original _state/_rebuild/_event_mask methods with the original inputs.
    """
    from .geometry_facies import (GeometryLabelEditSession,LABEL_EDIT_FORMAT,
        GEOMETRY_REVISION_FORMAT,_load_array,_darsia_metadata,_file_reference,normalize_run_geometry_config)
    value=config.get("manual_label_edit")
    if (not isinstance(value,dict) or set(value)!={"format","context","source_labels","session",
            "events","materialization","materialized_labels"} or value["format"]!=MANUAL_LABEL_EDIT):
        raise ValueError("Exact typed manual-label-edit provenance required.")
    context=value["context"]
    _private_manual_context(Path(context["private_series_root"])/"geometry_facies",
        {"tools":{"native_darsia":{"manual_label_edit_validation":context},
            "native_series":{"series_root":context["private_series_root"]}}})
    parent_pin=_verify(context["parent_revision"])
    parent=json.loads(Path(parent_pin["path"]).read_bytes())
    old=parent.get("configuration") or {}
    if (parent.get("format")!=GEOMETRY_REVISION_FORMAT or not re.fullmatch(r"[0-9a-f]{32}",str(parent.get("revision_id")))
            or context.get("selected_run")!=(old.get("frame_binding") or {}).get("display_reference_run")
            or config.get("source_mode")!=CANONICAL_EDIT or config.get("facies_source_path")!=""):
        raise ValueError("Private manual edit is only its actual parent display/reference grid.")
    for key in ("frame","width_m","height_m","baseline_path","frame_binding",
            "facies_properties_source_path","grouping_mode","regions","canonical_import_parent"):
        if config.get(key)!=old.get(key):
            raise ValueError("Manual edit changed parent geometry/source/calibration field: "+key)
    if normalize_run_geometry_config(config.get("run_geometry"))!=normalize_run_geometry_config(old.get("run_geometry")):
        raise ValueError("Manual reference edit cannot silently change saved per-run registrations.")
    if config.get("depth_path")!=old.get("depth_path"):
        # The unchanged Store copies depth into its genuine new UUID before
        # its second validation. Accept that byte identity only, not new depth.
        path=Path(config["depth_path"])
        depth_root=Path(context["private_series_root"])/"geometry_facies"/"results/geometry"
        old_depth=descriptor(old["depth_path"]);new_depth=descriptor(path)
        if (path.parent.parent!=depth_root or not re.fullmatch(r"[0-9a-f]{32}",path.parent.name)
                or path.name!="depth"+Path(old["depth_path"]).suffix.casefold()
                or any(new_depth[key]!=old_depth[key] for key in ("sha256","size_bytes"))):
            raise ValueError("Manual Store depth must remain its original exact byte copy.")
    actual={key:_verify(value[key]) for key in ("source_labels","session","events","materialization","materialized_labels")}
    parent_labels=descriptor(old["labels_path"])
    if any(actual["source_labels"][key]!=parent_labels[key] for key in ("sha256","size_bytes")):
        raise ValueError("Manual source labels are not the actual parent label bytes.")
    root=Path(context["private_series_root"]).resolve()/"geometry_facies"
    for row in actual.values():
        if not Path(row["path"]).resolve().is_relative_to(root):
            raise ValueError("Manual session/materialization must stay in the private project.")
    session=json.loads(Path(actual["session"]["path"]).read_bytes())
    materialization=json.loads(Path(actual["materialization"]["path"]).read_bytes())
    reference=config.get("label_edit_session") or {}
    events=[json.loads(line) for line in Path(actual["events"]["path"]).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    session_root=Path(actual["session"]["path"]).parent
    materialization_root=Path(actual["materialization"]["path"]).parent
    if (session.get("format")!=LABEL_EDIT_FORMAT
            or not re.fullmatch(r"[0-9a-f]{32}",str(session.get("session_id")))
            or session["session_id"]!=reference.get("session_id")
            or Path(actual["session"]["path"]).name!="session.json"
            or Path(actual["events"]["path"])!=session_root/"events.jsonl"
            or reference.get("base_labels_path")!=actual["source_labels"]["path"]
            or reference.get("events_path")!=actual["events"]["path"]
            or materialization.get("format")!="darsia-fluidflower-label-materialization-v1"
            or materialization.get("session_id")!=session["session_id"]
            or materialization.get("materialization_id")!=reference.get("materialization_id")
            or not re.fullmatch(r"[0-9a-f]{32}",str(materialization.get("materialization_id")))
            or materialization_root!=session_root/"materializations"/materialization["materialization_id"]
            or Path(actual["materialized_labels"]["path"])!=materialization_root/"labels.npz"
            or materialization.get("base_labels_path")!=actual["source_labels"]["path"]
            or materialization.get("events_path")!=actual["events"]["path"]
            or reference.get("materialization_manifest_path")!=actual["materialization"]["path"]
            or Path(config["labels_path"])!=Path(actual["materialized_labels"]["path"])):
        raise ValueError("Manual parent/session/events/materialization identity differs.")
    base=np.squeeze(_load_array(Path(actual["source_labels"]["path"])))
    labels=np.squeeze(_load_array(Path(actual["materialized_labels"]["path"])))
    if (base.ndim!=2 or not np.issubdtype(base.dtype,np.integer) or labels.dtype!=np.int32
            or labels.shape!=base.shape or materialization.get("shape")!=list(base.shape)
            or not _same_grid_metadata(_darsia_metadata(Path(actual["source_labels"]["path"])),
                _darsia_metadata(Path(actual["materialized_labels"]["path"])))):
        raise ValueError("Manual labels changed their exact parent spatial grid.")
    if (session.get("width_m")!=config["width_m"] or session.get("height_m")!=config["height_m"]
            or session.get("base_labels")!=_file_reference(Path(actual["source_labels"]["path"]))
            or len({event.get("event_id") for event in events})!=len(events)
            or not events or any(event.get("format")!=LABEL_EDIT_FORMAT
                or event.get("kind") not in {"brush","polygon","undo","redo"}
                or not re.fullmatch(r"[0-9a-f]{32}",str(event.get("event_id"))) for event in events)):
        raise ValueError("Real original manual session/event history required.")
    replay=object.__new__(GeometryLabelEditSession)
    replay.width_m,replay.height_m=config["width_m"],config["height_m"]
    replay.base_labels=np.asarray(base,dtype=np.int32);replay.events=events
    replay._rebuild()
    count=replay.active_operation_count()
    if (not count or count!=reference.get("active_operation_count")
            or count!=reference.get("materialized_active_operation_count")
            or count!=materialization.get("active_operation_count") or not np.array_equal(replay.labels,labels)):
        raise ValueError("Manual materialization is not the exact original active-event replay.")
    for row in actual.values():
        _verify(row)
    return dict(parent=parent,parent_binding=parent_pin,files=actual)


def private_manual_label_edit_sources(project_root,current,run):
    """A distinct private source variant; never an old registered-map claim."""
    from .geometry_facies import GEOMETRY_REVISION_FORMAT,_canonical_facies,_load_array
    root=Path(project_root).resolve()
    manifest=json.loads((root/"series.json").read_bytes())
    context=_private_manual_context(root,manifest)
    config=current.get("configuration") or {}
    payload=manual_label_edit_payload(config)
    # Keep the original imported-parent/full-CSV/selected-registration checks.
    # This validates provenance; exported pixels below are the actual new
    # manual materialization, never the old run_override as a fallback.
    canonical_parent_payload(config,run_id=run)
    revision=current.get("revision_id")
    if (current.get("format")!=GEOMETRY_REVISION_FORMAT
            or not re.fullmatch(r"[0-9a-f]{32}",str(revision))
            or revision==payload["parent"]["revision_id"]
            or config["manual_label_edit"]["context"]!=context or run!=context["selected_run"]):
        raise ValueError("Manual export requires its new genuine private Store UUID/reference run.")
    current_path=root/"overrides/geometry_facies/current.json"
    revision_path=root/"results/geometry"/revision/"revision.json"
    if json.loads(current_path.read_bytes())!=current or json.loads(revision_path.read_bytes())!=current:
        raise ValueError("Manual current and immutable Store revision disagree.")
    labels=descriptor(current["artifacts"]["labels_path"])
    facies=descriptor(current["artifacts"]["facies_path"])
    for row in (labels,facies):
        if not Path(row["path"]).resolve().is_relative_to(revision_path.parent):
            raise ValueError("Manual Store artifacts escaped the new UUID.")
    if any(labels[key]!=payload["files"]["materialized_labels"][key] for key in ("sha256","size_bytes")):
        raise ValueError("Manual Store labels are not its actual materialized editor bytes.")
    expected,_properties=_canonical_facies(config)
    if not np.array_equal(_load_array(Path(facies["path"])),expected):
        raise ValueError("Manual Store facies differ from the unchanged original derivation.")
    guarded=[descriptor(current_path),descriptor(revision_path),payload["parent_binding"],*payload["files"].values()]
    return dict(format=MANUAL_LABEL_EDIT,run_id=run,revision_id=revision,
        files=dict(labels=labels,facies=facies),guarded_sources=guarded,
        registration=dict(kind=MANUAL_LABEL_EDIT,not_a_registration_or_H=True,
            parent_revision_id=payload["parent"]["revision_id"],provenance=config["manual_label_edit"]),
        label_serialization="Original editor int32 materialization; exact event replay and source grid",
        provenance="Private Q121 reference-map edit only; not adopted-v1 or carried run registration")
