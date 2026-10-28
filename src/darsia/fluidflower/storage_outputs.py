"""Optional immutable storage outputs; no classifier or mass model is called.

This module is deliberately inert until an explicitly bound storage request is
present. Q28 approves LA only when explicitly requested; it is not a default.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import uuid
import zipfile

import numpy as np

from .storage_resolution import (LABEL_LEGEND, PROBABILITY_FIELDS, aggregate_frame,
                                 aggregate_static, make_grid, metric_grid_shape, resolution_mm)

REQUEST_FORMAT = "darsia-storage-output-request-v1"
FRAME_FORMAT = "darsia-storage-frame-v1"
LA_ENCODING = "uint8-la-q255-v1"
DENSITIES = {"total": "mass_total", "aqueous": "mass_aqueous", "gas": "mass_free_gas"}
TOTALS = {"total": "total_mass_g", "aqueous": "aqueous_mass_g", "gas": "gas_mass_g"}


def quantified_totals(result):
    """Select original finite contribution carriers, never an unknown total."""
    prefix = "quantified_" if result.get("analysis_scope") is not None else ""
    return {key: result[prefix + name] for key, name in TOTALS.items()}


def bind_analysis_scope(document, result):
    """Label finite storage contributions as subtotals with an unknown remainder."""
    scope = result.get("analysis_scope")
    if scope is not None:
        document["analysis_scope"] = deepcopy(scope)
        document["quantified_native_mass_totals_g"] = document["native_mass_totals_g"]
        document["native_mass_totals_g"] = dict.fromkeys(TOTALS)
        document["mass_storage_role"] = "quantified-contribution-carriers-with-unquantified-remainder"
    return document


def checked_mass_totals(document, result, *, arrays=None, public=False):
    """Require saved scope, unknown mask and explicit subtotal scalar closure."""
    scope = result.get("analysis_scope")
    if scope is None:
        if document.get("analysis_scope") is not None:
            raise ValueError("Stored mass scope differs from its native result")
        return document["native_mass_totals_g"]
    if (json.dumps(scope, sort_keys=True, allow_nan=False) !=
            json.dumps(document.get("analysis_scope"), sort_keys=True, allow_nan=False)
            or scope.get("status") != "quantified_subtotal_with_unquantified_remainder"
            or scope.get("whole_rig_mass_complete") is not False
            or scope.get("unknown_mass_g") is not None or scope.get("area_cm2") is not None
            or document.get("native_mass_totals_g") != dict.fromkeys(TOTALS)
            or document.get("mass_storage_role") != "quantified-contribution-carriers-with-unquantified-remainder"
            or public and any(result.get(name) is not None for name in TOTALS.values())):
        raise ValueError("Stored unknown remainder/subtotal scope differs")
    if arrays is not None:
        mask = arrays.get("mask_approved_analysis_gap_area_fraction")
        if mask is None or not np.any(mask > 0):
            raise ValueError("Stored approved unknown mask is missing")
    return document["quantified_native_mass_totals_g"]


def read_mass_scope(value, result, totals):
    """Keep old reader outputs exact; expose scoped carrier sums separately."""
    if result.get("analysis_scope") is not None:
        value.update(analysis_scope=deepcopy(result["analysis_scope"]),
            quantified_mass_totals_g=totals, mass_totals_g=dict.fromkeys(TOTALS), unknown_mass_g=None)
    return value


def parse_request(config, precision):
    """Admit explicit output DATA, never choose an encoding or precision."""
    if "storage" not in config:
        return None
    value = config["storage"]
    if (not isinstance(value, dict) or value.get("format") != REQUEST_FORMAT
            or value.get("gas_score_encoding") != LA_ENCODING
            or precision not in {"float32", "float64"}):
        raise ValueError("Explicit bound storage request and Q28 encoding are required")
    allowed = {"format", "resolution_mm", "gas_score_encoding", "extent_m"}
    if set(value) - allowed or "resolution_mm" not in value:
        raise ValueError("Unknown or missing storage output DATA")
    step = resolution_mm(value["resolution_mm"])
    if step is not None:
        extent = value.get("extent_m")
        if (not isinstance(extent, (tuple, list)) or len(extent) != 4
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) for v in extent)
                or extent[1] <= extent[0] or extent[3] <= extent[2]):
            raise ValueError("Coarse storage needs the bound rig metric extent")
        metric_grid_shape(extent, step)
    return dict(value, precision=precision)


def _asset(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return dict(path=str(path), sha256=digest, size_bytes=path.stat().st_size)


def _array_identity(value):
    value = np.asarray(value)
    digest = hashlib.sha256()
    for chunk in np.nditer(value, flags=["external_loop", "buffered", "zerosize_ok"],
                           op_flags=["readonly"], order="C", buffersize=65536):
        digest.update(chunk.tobytes(order="C"))
    return dict(sha256=digest.hexdigest(), dtype=value.dtype.str, shape=list(value.shape))


def _write_json(path, value):
    """Write one new JSON file through the N111-retried publisher; same bytes as before."""
    from .pipeline.io_ops import check_metadata_bytes, publish_file
    data = (json.dumps(value, ensure_ascii=False, allow_nan=False,
                       sort_keys=True, separators=(",", ":")) + "\n").encode("utf8")
    check_metadata_bytes(path, data)
    publish_file(path, lambda stream: stream.write(data))


def _write_npz(path, arrays):
    """Write one new NPZ through the N111-retried publisher; same bytes as before."""
    from .pipeline.io_ops import publish_file

    def write(stream):
        with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1) as archive:
            for name, array in arrays.items():
                with archive.open(name + ".npy", "w", force_zip64=True) as member:
                    np.lib.format.write_array(member, np.asarray(array), allow_pickle=False)
    publish_file(path, write)


def _bound_path(bound, reader=None):
    if not isinstance(bound, dict) or not {"path", "sha256", "size_bytes"} <= set(bound):
        raise ValueError("Storage product needs a complete immutable descriptor")
    if reader is not None:
        return reader.path(bound)
    path = Path(bound["path"])
    actual = _asset(path)
    if actual != {key: bound[key] for key in actual}:
        raise ValueError("Storage product SHA or size changed: " + str(path))
    return path


def _owned(bound, directory):
    """Check logical ownership before any alias resolves a producer path."""
    path, root = PureWindowsPath(bound["path"]), PureWindowsPath(str(directory))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Storage product escapes its owned frame") from exc
    if len(relative.parts) != 1 or relative.name in {"", ".", ".."}:
        raise ValueError("Storage product is not an owned single frame file")


def _read_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate storage metadata key")
            value[key] = item
        return value
    def invalid(value):
        raise ValueError("Non-finite storage metadata")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs,
                      parse_constant=invalid)


def encode_before(capture, shape):
    """Copy exact native-centre evidence into Q28 LA without resampling.

    Range/absence is a Q23 product-pending state, never a classifier fallback.
    Shape/ownership inconsistencies remain errors rather than missing data.
    """
    if "g3_pixel" not in capture:
        return None, dict(state="pending", reason="native_centre_not_computed",
                          source_array="NativeCentreResult.g3_pixel")
    p, valid = np.asarray(capture["g3_pixel"]), np.asarray(capture["flatfield_valid"])
    box = capture["source_box_xywh"]
    if (not isinstance(box, (tuple, list)) or len(box) != 4
            or any(type(v) is not int for v in box)):
        raise ValueError("Storage score needs its exact integer source_box")
    x, y, width, height = box
    if (p.ndim != 2 or p.dtype != np.float32 or p.shape != (height, width)
            or valid.dtype != bool or valid.shape != p.shape or x < 0 or y < 0
            or min(width, height) < 1 or x + width > shape[1] or y + height > shape[0]):
        raise ValueError("Native-centre score/support/grid differ")
    finite = np.isfinite(p)
    selected = valid & finite
    finite_values = p[selected]
    outside = np.any((finite_values < 0) | (finite_values > 1))
    audit = dict(source_array="NativeCentreResult.g3_pixel",
                 validity_array="NativeCentreResult.flatfield_valid",
                 source_box_xywh=list(box), source_shape_hw=list(p.shape),
                 target_shape_hw=list(shape), finite_valid_pixels=int(selected.sum()),
                 finite_min=(float(finite_values.min()) if finite_values.size else None),
                 finite_max=(float(finite_values.max()) if finite_values.size else None),
                 source_identity=_array_identity(p), validity_identity=_array_identity(valid),
                 existing_probability_encoding=capture.get("probability_encoding"))
    if np.isinf(p[valid]).any() or outside:
        return None, dict(audit, state="pending", reason="pre_rule_score_outside_probability_contract",
                          rescaled=False, clipped=False)
    la = np.zeros((*shape, 2), dtype=np.uint8)
    central = la[y:y + height, x:x + width]
    central[..., 0][selected] = np.rint(255.0 * p[selected].astype(np.float64)).astype(np.uint8)
    central[..., 1][selected] = 255
    return la, dict(audit, state="complete", encoding=LA_ENCODING,
                    quantization="L=round-to-nearest-even(255*p); decode L/255 only where A=255",
                    missing="A=0; L=0 is padding, not evidence",
                    coarse_reduction="arithmetic mean L/255 over A=255 pixels; valid-pixel fraction of all cell pixels; no-valid mean NaN",
                    rescaled=False, clipped=False)


def decode_before(path, shape, precision):
    """Read only the new PNG bytes, respecting the original missing support."""
    from PIL import Image
    with Image.open(path) as image:
        if image.mode != "LA":
            raise ValueError("Q28 score PNG must retain uint8 LA channels")
        la = np.asarray(image).copy()
    if la.dtype != np.uint8 or la.shape != (*shape, 2) or not np.isin(la[..., 1], (0, 255)).all():
        raise ValueError("Q28 score PNG grid or alpha differs")
    result = la[..., 0].astype(np.dtype(precision)) / np.dtype(precision).type(255)
    result[la[..., 1] == 0] = np.nan
    return result


def pixel_masses(export, volume_m3, shape):
    """Expose existing independent density contributions; change no native sum."""
    volume = np.broadcast_to(np.asarray(volume_m3, np.float64), shape)
    if not np.isfinite(volume).all() or np.any(volume < 0):
        raise ValueError("Existing native Geometry volume weights must be finite/nonnegative")
    result, identities = {}, {}
    scoped = export.get("analysis_scope") is not None
    mask = export["products"].get("approved_analysis_gap_mask") if scoped else None
    if scoped and (not isinstance(mask, np.ndarray) or mask.dtype != bool or mask.shape != shape
                   or np.any(volume[mask] != 0)):
        raise ValueError("Quantified storage requires its exact unknown mask and original zero integration weights")
    for component, name in DENSITIES.items():
        public_density = np.asarray(export["products"][name])
        density = np.asarray(export["quantified_products"][name] if scoped else public_density)
        if density.shape != shape or not np.isfinite(density).all() or np.any(density < 0):
            raise ValueError("Existing independent native mass density differs")
        if scoped and (public_density.shape != shape or not np.isnan(public_density[mask]).all()
                       or not np.array_equal(public_density[~mask], density[~mask])):
            raise ValueError("Quantified storage carriers differ from the public known/unknown map")
        # Same Geometry multiply operands, followed by the existing m3-to-L factor.
        result[component] = (volume * density) * 1000.0
        identities[name] = _array_identity(public_density)
    return result, identities, _array_identity(np.asarray(volume_m3, np.float64) * 1000.0)


def _identity(task, spec, request):
    return dict(job_id=task["job_id"], run=task["run"], case_id=spec["case_id"],
                configuration=task["configuration"], code_commit=task["commit"],
                calibration_sha256=task["calibration_sha256"], precision=task["precision"],
                source=spec["source"], request=request)


def write_frame(output, task, spec, phase, inferred, phase_capture,
                mass_capture, evaluated, request, *, extensive_inputs=None):
    """Close new immutable outputs BEFORE the caller's original frame commit.

    Failure preserves this own partial directory. Nothing in old native
    outputs is rewritten. Range/absent BEFORE has an honest final-only receipt.
    """
    from PIL import Image
    if extensive_inputs is not None:
        from .storage_extensive_io import validate_inputs
        validate_inputs(extensive_inputs)
        if request["resolution_mm"] == "native":
            raise ValueError("Extensive I/O requires an explicitly requested coarse grid")
    output = Path(output)
    parent = output / "preparation" / spec["case_id"]
    if not parent.is_dir():
        raise ValueError("Original native/phase preparation must already exist")
    names = {"storage_completed.json", "storage_gas_probability_before_rules.png", "storage_stored_frame.npz"}
    if any((parent / name).exists() for name in names):
        raise FileExistsError("Storage outputs must be new immutable frame files")
    labels = np.asarray(inferred["labels"])
    frame, result = mass_capture["frame"], evaluated["result"]
    if (labels.dtype != np.uint8 or labels.ndim != 2 or not np.isin(labels, (0, 1, 2, 255)).all()
            or not np.array_equal(labels, frame.phase_labels)
            or request["precision"] != result["precision"]):
        raise ValueError("Storage must consume unchanged final phase/precision")
    with Image.open(_bound_path(phase["labels"])) as image:
        if not np.array_equal(np.asarray(image), labels):
            raise ValueError("Saved final phase pixels differ from the storage input")
    shape = labels.shape
    grid = None
    if request["resolution_mm"] != "native":
        grid = make_grid(mass_capture["x_m"], mass_capture["y_m"],
                         np.broadcast_to(np.asarray(frame.pixel_area_m2), shape),
                         include=frame.domain, extent_m=request["extent_m"],
                         storage_resolution_mm=request["resolution_mm"], precision=request["precision"])
    temporary = parent / (".storage." + uuid.uuid4().hex + ".partial")
    temporary.mkdir()
    final_indicator = labels == 2
    la, before = encode_before(phase_capture, shape)
    files = []
    if la is not None:
        png = temporary / "storage_gas_probability_before_rules.png"
        from .pipeline.io_ops import publish_file
        publish_file(png, lambda stream: Image.fromarray(la, mode="LA").save(stream, format="PNG"))
        probabilities = {PROBABILITY_FIELDS[0]: decode_before(png, shape, request["precision"]),
                         PROBABILITY_FIELDS[1]: final_indicator}
        files.append(png)
    else:
        probabilities = {PROBABILITY_FIELDS[1]: final_indicator}
    masses, density_ids, weight_id = pixel_masses(
        evaluated["export"], mass_capture["volume_m3"], shape)
    totals = quantified_totals(result)
    checks = {}
    for component, values in masses.items():
        expected = totals[component]
        if (isinstance(expected, bool) or not isinstance(expected, (int, float))
                or not math.isfinite(expected) or expected < 0):
            raise ValueError("Original native scalar mass must be finite/nonnegative")
        actual = float(values.sum(dtype=np.float64))
        if abs(actual - expected) > 1e-6 * abs(expected):
            raise ValueError("Native pixel contribution sum differs from original " + component + " mass")
        checks[component] = dict(native_total_g=expected, pixel_sum_g=actual,
                                 relative_limit=1e-6, passed=True)
    coarse = None
    if request["resolution_mm"] != "native":
        masks = {name: np.asarray(value) for name, value in inferred["classes"].items()}
        masks["main_mass_eligible"] = result["fields"]["main_mass_eligible_mask"]
        for name, value in (frame.excluded_masks or {}).items():
            masks[name] = value
        reduced = aggregate_frame(grid,
            concentration_aqueous=evaluated["export"]["products"]["concentration_aqueous"],
            saturation_free_gas=evaluated["export"]["products"]["saturation_free_gas"],
            phase_labels=labels, masks=masks, probabilities=probabilities,
            native_pixel_masses_g=masses, native_mass_totals_g=totals, before_status=before)
        static = aggregate_static(grid, depth_m=np.broadcast_to(np.asarray(frame.depth_m), shape),
                                  porosity=np.broadcast_to(np.asarray(frame.porosity), shape),
                                  facies=frame.facies)
        arrays = dict(reduced["arrays"], **{"static_" + key: value for key, value in static.items()})
        archive = temporary / "storage_stored_frame.npz"
        _write_npz(archive, arrays)
        files.append(archive)
        coarse = dict(reduced["metadata"], arrays={key: _array_identity(value)
                                                 for key, value in arrays.items()},
                      per_frame_geometry=True, shared_static_geometry_assumed=False)
    bound_files = [dict(_asset(path), path=str(parent / path.name), kind="map") for path in files]
    document = dict(format=FRAME_FORMAT, complete=True, **_identity(task, spec, request),
        attempt_directory=str(output), before_rules=before,
        after_rules=dict(source_array="final phase_labels == 2", source=phase["labels"],
                         values=[0, 1], gas_pixels=int(final_indicator.sum()),
                         label_legend=dict(LABEL_LEGEND),
                         coarse_reduction="count(final labels==2)/all native cell pixels, irrespective of N1 domain or area",
                         support_mask=None, support_fraction="not inferred; no explicit label support mask supplied"),
        final_label_counts={str(value): int(np.count_nonzero(labels == value)) for value in (0, 1, 2, 255)},
        native_mass_density_identities=density_ids, native_transport_weight_l_identity=weight_id,
        native_mass_totals_g=totals, native_pixel_mass_checks=checks, coarse=coarse,
        files=bound_files, existing_native_products_unchanged=True)
    bind_analysis_scope(document, result)
    if extensive_inputs is not None:
        from .storage_extensive_io import prepare_payload, write_payload, declared_files
        product = prepare_payload(extensive_inputs, grid, labels=labels,
            saturation=evaluated["export"]["products"]["saturation_free_gas"],
            masses=masses, totals=totals)
        document["extensive"] = write_payload(temporary, parent, product, document)
        added = declared_files(document["extensive"])
        bound_files.extend(added.values())
        files.extend(temporary / name for name in added)
    receipt_path = temporary / "storage_completed.json"
    _write_json(receipt_path, document)
    # Only our new flat namespace is moved; old native files are never touched.
    # The original _inventory(stage) will bind every SHA before native commit.
    from .pipeline.job_queue import rename
    for path in [*files, receipt_path]:
        destination = parent / path.name
        if destination.exists():
            raise FileExistsError(destination)
        rename(path, destination)
    entry = dict(format=FRAME_FORMAT, case_id=spec["case_id"], attempt_directory=str(output),
                 receipt=dict(_asset(parent / "storage_completed.json"), kind="operational_receipt"),
                 files=bound_files)
    if extensive_inputs is not None:
        entry["extensive"] = document["extensive"]
    verify_entry(entry, task, spec, request, phase["labels"], result,
                 evaluated["export"]["products"], mass_capture["volume_m3"])
    return entry


def verify_entry(entry, task, spec, request, phase_asset, native_result,
                 native_products, volume_m3=None, transport_weight_l=None, reader=None):
    """Verify the preceding storage closure; never infer or recreate it."""
    if (entry.get("format") != FRAME_FORMAT or entry.get("case_id") != spec["case_id"]
            or not isinstance(entry.get("files"), list)):
        raise ValueError("Storage closure belongs to another frame")
    root = PureWindowsPath(entry["attempt_directory"]) / "preparation" / spec["case_id"]
    _owned(entry["receipt"], root)
    if PureWindowsPath(entry["receipt"]["path"]).name != "storage_completed.json":
        raise ValueError("Storage closure receipt name differs")
    document = _read_json(_bound_path(entry["receipt"], reader))
    if (document.get("format") != FRAME_FORMAT or document.get("complete") is not True
            or document.get("attempt_directory") != entry["attempt_directory"]
            or any(document.get(key) != value for key, value in _identity(task, spec, request).items())
            or document.get("files") != entry["files"]
            or document.get("existing_native_products_unchanged") is not True):
        raise ValueError("Storage closure source/configuration/owner differs")
    seen = set()
    added = {}
    if document.get("extensive") is not None:
        from .storage_extensive_io import declared_files
        added = declared_files(document["extensive"])
    if document.get("extensive") != entry.get("extensive"):
        raise ValueError("Extensive I/O entry differs from its original receipt")
    for bound in entry["files"]:
        _owned(bound, root)
        name = PureWindowsPath(bound["path"]).name
        if name in seen or name not in {"storage_gas_probability_before_rules.png", "storage_stored_frame.npz", *added}:
            raise ValueError("Unexpected or repeated storage output")
        if name in added and bound != added[name]:
            raise ValueError("Extensive I/O asset differs from its declared entry")
        seen.add(name)
        _bound_path(bound, reader)
    if not added.keys() <= seen:
        raise ValueError("Extensive I/O file is missing from its original entry")
    before = document["before_rules"]
    if before.get("state") == "complete":
        if before.get("encoding") != LA_ENCODING or "storage_gas_probability_before_rules.png" not in seen:
            raise ValueError("Complete pre-rule score has no exact encoded product")
    elif before.get("state") == "pending":
        if not before.get("reason") or "storage_gas_probability_before_rules.png" in seen:
            raise ValueError("Pending pre-rule score cannot pretend to have a PNG")
    else:
        raise ValueError("Unknown pre-rule score state")
    if (request["resolution_mm"] == "native") != ("storage_stored_frame.npz" not in seen):
        raise ValueError("Storage closure has a different stored grid")
    if (document["after_rules"]["source"] != phase_asset
            or document["after_rules"].get("label_legend") != LABEL_LEGEND
            or document["after_rules"].get("support_mask") is not None):
        raise ValueError("Final gas indicator does not bind the original phase product/legend/support")
    totals = quantified_totals(native_result)
    if checked_mass_totals(document, native_result) != totals:
        raise ValueError("Storage closure changed an original native scalar mass")
    for component, name in DENSITIES.items():
        if document["native_mass_density_identities"][name] != _array_identity(native_products[name]):
            raise ValueError("Storage closure changed independent native mass density")
        check = document["native_pixel_mass_checks"][component]
        if (check.get("passed") is not True or check.get("native_total_g") != totals[component]
                or check.get("relative_limit") != 1e-6
                or abs(check["pixel_sum_g"] - totals[component]) > 1e-6 * abs(totals[component])):
            raise ValueError("Stored native contribution mass check differs")
    weights = (np.asarray(volume_m3, np.float64) * 1000.0
               if transport_weight_l is None else transport_weight_l)
    if document["native_transport_weight_l_identity"] != _array_identity(weights):
        raise ValueError("Storage closure changed original native geometry weights")
    labels = np.asarray(native_products["phase_labels"])
    counts = {str(value): int(np.count_nonzero(labels == value)) for value in (0, 1, 2, 255)}
    if (document["final_label_counts"] != counts
            or document["after_rules"]["gas_pixels"] != counts["2"]):
        raise ValueError("Storage closure changed literal final phase pixel counts")
    return document


def verify_prefix(prefix, task, config, request):
    """Native commit implies a preceding valid storage closure, or explicit STOP.

    This function only reads the original checkpoint-verified publications.
    Missing/invalid storage never triggers inference replay or false completion.
    """
    from .mass.n1_pipeline import DataReader
    reader = DataReader(unc_only=False)
    records, producers = prefix["records"], prefix["frame_producers"]
    if len(records) != len(producers):
        raise ValueError("Native prefix and its genuine producers differ")
    result = []
    for spec, record, producer in zip(config["frames"], records, producers):
        if spec["case_id"] != record["frame_id"] or producer["case_id"] != spec["case_id"]:
            raise ValueError("Storage resume order differs from native committed prefix")
        directory = Path(record["nas_directory"])
        bound_result = dict(record["files"]["result.json"], path=str(directory / "result.json"))
        native_result = reader.json(bound_result)
        binding = native_result["prediction"]["pipeline"]
        # The original native result SHA/commit inventory already anchors this.
        # Missing closure is an explicit STOP before any next native frame.
        if "storage" not in binding:
            raise ValueError("Committed native frame lacks its required preceding storage closure")
        validate_publication_binding(binding)
        entry = binding["storage"]
        if entry["attempt_directory"] != producer["attempt_directory"]:
            raise ValueError("Storage closure has a different genuine prefix producer")
        if native_result.get("lean_storage") is None:
            product_asset = dict(record["files"]["products.npz"], path=str(directory / "products.npz"))
            native_products = reader.archive(product_asset, ("mass_total", "mass_aqueous", "mass_free_gas", "phase_labels"))
            geometry_asset = dict(record["files"]["native_geometry.npz"], path=str(directory / "native_geometry.npz"))
            weights = reader.archive(geometry_asset, ("transport_weight_l",))["transport_weight_l"]
        else:
            # Lean frames rebuild mass_total and regenerate omitted static
            # fields; both readers verify the producer's recorded SHA-256.
            from .storage_profile import read_products, read_static_fields
            if (native_result["products"]["sha256"] != record["files"]["products.npz"]["sha256"]
                    or native_result["products"]["size_bytes"] != record["files"]["products.npz"]["size_bytes"]):
                raise ValueError("Lean frame products differ from the committed inventory")
            native_products = read_products(native_result, directory, ("mass_total", "mass_aqueous", "mass_free_gas", "phase_labels"), reader=reader)
            weights = read_static_fields(native_result, directory, ("transport_weight_l",), reader=reader)["transport_weight_l"]
        phase_asset = native_result["prediction"]["pipeline"]["phase"]["labels"]
        verify_entry(entry, task, spec, request, phase_asset, native_result,
                     native_products, transport_weight_l=weights, reader=reader)
        result.append(entry)
    return result


def validate_publication_binding(binding):
    """Require every declared storage asset to be in the old anchored inventory."""
    entry = binding["storage"]
    case = binding["case_id"]
    if entry.get("format") != FRAME_FORMAT or entry.get("case_id") != case:
        raise ValueError("Storage publication belongs to another original frame")
    root = PureWindowsPath(entry["attempt_directory"]) / "preparation" / case
    if PureWindowsPath(binding["phase"]["labels"]["path"]).parent != root:
        raise ValueError("Storage namespace differs from the original phase producer")
    inventory = binding["files"]
    expected = {}
    added = {}
    if entry.get("extensive") is not None:
        from .storage_extensive_io import declared_files
        added = declared_files(entry["extensive"])
    for bound in [entry["receipt"], *entry["files"]]:
        _owned(bound, root)
        name = PureWindowsPath(bound["path"]).name
        if name in expected or inventory.get(name) != bound:
            raise ValueError("Storage SHA/size/role is not anchored by the native inventory")
        if name == "storage_completed.json":
            kind = "operational_receipt"
        elif name in {"storage_gas_probability_before_rules.png", "storage_stored_frame.npz"}:
            kind = "map"
        elif name in added and bound == added[name]:
            kind = added[name]["kind"]
        else:
            raise ValueError("Unknown new storage inventory row")
        if bound.get("kind") != kind:
            raise ValueError("Storage output has a different declared product role")
        expected[name] = bound
    if not added.keys() <= expected.keys():
        raise ValueError("Extensive I/O file is outside its original publication inventory")
    if set(name for name in inventory if name.startswith("storage_")) != set(expected):
        raise ValueError("Unlisted storage file in the original native inventory")
    return entry
