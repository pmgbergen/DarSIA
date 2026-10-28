"""Capacity of a phase-two n1 task under a qualified native-phase-n1 profile.

The lean_capacity module also uses these array and storage envelopes.

Phase two (pipeline ``n1``) computes mass from stored phase-one products with the
same N1 evaluation and frame publisher as the qualified series route, and writes
the same per-frame mass files. A full n1 task on the qualified head (commit, CPU,
float32) is admitted by this rule as another run. The estimate is the qualified
profile's measured mass-frame bytes (every file outside the preparation stage),
plus upper bounds for its declared options: the pixel-domain route (N15, with
the hidden-mass rule N16) and the storage layer, with its N16 extensive payload.

Each option counts every array its writer declares at its exact dtype, or the
largest a varying dtype can be, uncompressed on the grid it is written on (the
label grid, or the storage layer's metric grid), with the stated NPZ margins of
option_capacity. The gas probability is a phase-one output and is refused here.
"""
from __future__ import annotations

import math

PIPELINE = "n1"
QUALIFIED_PIPELINE = "native-phase-n1"
STAGE_PART = "preparation"
#: The two archives the pixel-domain route rewrites in full (n1_pipeline.py:1061, 1078-1081).
ARCHIVES = ("products.npz", "native_geometry.npz")
#: Per frame: the n1 frame record and statistics, which may differ from the series route's.
RECORD_BYTES = 1 << 20
#: Per frame: each new storage receipt (storage_completed.json, the photograph and
#: extensive receipts).
RECEIPT_BYTES = 1 << 20
#: Bytes of a scalar NPY member's payload: at most a 64-character string (<U64).
SCALAR_BYTES = 256
#: Bound on one member name, with ".npy", in the archives whose names follow the
#: data (the coarse and extensive frames): every real name is far shorter.
NAME_BYTES = 128
#: "map": the task precision's itemsize (the pixel math dtype); an int: fixed bytes.
#: The pixel-domain products (n1_pipeline.py:1028-1056; pixel_pipeline.py:371-381;
#: mass/pixels.py:262-285): five maps and the labels, 18 boolean masks, five more
#: maps and two int32 donor rows (mass/pixels.py:284; geometry/pixels.py:228).
PIXEL_PRODUCTS = dict(
    concentration_aqueous="map", saturation_free_gas="map", mass_total="map", mass_aqueous="map",
    mass_free_gas="map", phase_labels=1, main_mass_eligible_mask=1, unquantified_facies_mask=1,
    measured_mask=1, concentration_estimated_mask=1, concentration_unknown_mask=1, phase_unknown_mask=1,
    unknown_without_capacity_mask=1, measured_aqueous_mask=1, measured_gas_mask=1, estimated_gas_mask=1,
    column_estimated_aqueous_mask=1, water_mask=1, donors_mask=1, domain_mask=1, visible_mask=1,
    geometry_filled_mask=1, geometry_unknown_mask=1, mass_contribution_mask=1,
    measured_mass_g="map", edge_mass_g="map", unknown_capacity_g="map",
    concentration_donor_row=4, concentration_donor_distance_m="map",
    geometry_donor_row=4, geometry_donor_distance_m="map")
#: The products' scalar members (n1_pipeline.py:1048-1056).
PRODUCT_SCALARS = ("E_lo_g", "E_hi_g", "U_rest_g", "U_rest_defined", "U_rest_amendment_sha256",
                   "source_sha256", "phase_sha256", "calibration_sha256", "code_sha256")
#: The pixel-domain geometry (pixel_pipeline.py:386-388). Area and weights are float64
#: (:259, :310); depth, porosity and facies follow the frame DATA, so they count 8 bytes.
PIXEL_GEOMETRY = dict(pixel_area_m2=8, depth_m=8, porosity=8, facies=8, transport_weight_l=8)
#: The hidden-mass rule's additions (pixel_pipeline.py:556-579): six masks, the int32
#: below-photo donor row (geometry/pixels.py:405) and four maps; the measured component
#: maps (mass/pixels.py:346-368); and six geometry fields at 8 bytes.
N16_PRODUCTS = dict(
    hidden_mask=1, physical_mask=1, physical_properties_valid_mask=1, below_photo_estimated_mask=1,
    below_photo_zero_mask=1, below_photo_unknown_mask=1, below_photo_donor_row=4,
    below_photo_donor_distance_m="map", below_photo_estimated_mass_g="map",
    measured_total_mass_g="map", measured_aqueous_mass_g="map", measured_gas_mass_g="map")
N16_GEOMETRY = dict(n16_solubility_g_per_l=8, full_physical_depth_m=8, full_physical_porosity=8,
                    full_physical_pixel_pore_volume_m3=8, full_physical_facies=8,
                    full_physical_physical_area_m2=8)
#: The n1 storage frame's coarse arrays (storage_resolution.aggregate_frame and
#: aggregate_static, through storage_n1_outputs.py:195-207): 14 frame fields and one
#: area fraction per mask at the grid precision, then the static fields.
COARSE_FRAME_FIELDS = 14
#: The masks the n1 storage layer reduces (storage_n1_outputs.py:174-178): four phase
#: classes, the main mass mask and at most two excluded masks.
COARSE_MASKS = 8  # Includes a possible approved-analysis-gap fraction.
#: The static fields: six at the grid precision, three int64 counts and one boolean.
COARSE_STATIC = dict(precision=6, int64=3, boolean=1)
#: The N16 extensive payload (storage_extensive.aggregate_extensive): four extensive
#: sums, five int64 counts, the hidden area, one boolean, three masses, four intensive
#: fields with their int64 count and boolean, three coverage triples, three channels
#: with their counts, two arrays per facies and the facies majority (at most 8 bytes).
EXTENSIVE = dict(precision=4 + 1 + 3 + 4 + 3 * 2 + 3, int64=5 + 4 + 3 + 3, boolean=1 + 4, majority=8)
WRITERS = ("src/darsia/fluidflower/pipeline/staged_data.py", "src/darsia/fluidflower/pipeline/job_inputs.py")
PIXEL_WRITERS = ("src/darsia/fluidflower/mass/pixel_pipeline.py", "src/darsia/fluidflower/mass/pixels.py",
                 "src/darsia/fluidflower/geometry/pixels.py")
STORAGE_WRITERS = ("src/darsia/fluidflower/storage_n1_outputs.py", "src/darsia/fluidflower/storage_outputs.py",
                   "src/darsia/fluidflower/storage_resolution.py")
EXTENSIVE_WRITERS = ("src/darsia/fluidflower/storage_extensive.py", "src/darsia/fluidflower/storage_extensive_io.py")
N1_MODEL = ("UPPER BOUND, not a measurement: the qualified profile's measured mass-frame bytes "
            "per frame, plus the declared options' arrays at their exact (or largest) dtypes, "
            "uncompressed, with stated margins")
#: Optional outputs refused on this route, each with its reason.
REFUSED = {"gas_probability_storage": "is a phase-one output",
           "probability_storage": "is no output of this route; no writer reads it"}


def _itemsize(kind, precision):
    """Bytes of one value: the task precision for a map, else the fixed size."""
    return {"float32": 4, "float64": 8}[precision] if kind == "map" else kind


def _payloads(table, pixels, precision):
    return {name: pixels * _itemsize(kind, precision) for name, kind in table.items()}


def anonymous_npz_bound(sizes):
    """Upper bound on an NPZ of these member payloads, each name at most NAME_BYTES."""
    from .option_capacity import NPY_HEADER_BYTES, ZIP_END_BYTES, ZIP_MEMBER_BYTES, deflate_bound
    return ZIP_END_BYTES + sum(deflate_bound(NPY_HEADER_BYTES + size) + ZIP_MEMBER_BYTES + 2 * NAME_BYTES
                               for size in sizes)


def coarse_sizes(cells, precision):
    """Member payloads of the n1 storage frame: frame fields, mask fractions and static fields."""
    size = {"float32": 4, "float64": 8}[precision]
    return ([cells * size] * (COARSE_FRAME_FIELDS + COARSE_MASKS + COARSE_STATIC["precision"])
            + [cells * 8] * COARSE_STATIC["int64"] + [cells] * COARSE_STATIC["boolean"])


def extensive_sizes(cells, precision, facies):
    """Member payloads of the N16 extensive frame, for this many facies."""
    size = {"float32": 4, "float64": 8}[precision]
    return ([cells * size] * (EXTENSIVE["precision"] + 2 * facies) + [cells * 8] * EXTENSIVE["int64"]
            + [cells] * EXTENSIVE["boolean"] + [cells * EXTENSIVE["majority"]])


def validate_requests(config):
    """The n1 options a configuration declares, each checked by its writer's own validator.

    This reads only the run configuration, so it runs before any profile read.
    """
    for key, reason in REFUSED.items():
        if config.get(key) is not None or (key == "gas_probability_storage" and key in config):
            raise ValueError("Optional output " + key + " " + reason)
    declared = []
    if config.get("pixel_domain") is not None:
        from ..mass.pixel_pipeline import validate_policy
        validate_policy(config["pixel_domain"])
        declared.append("pixel_domain")
    if config.get("storage") is not None:
        from ..storage_outputs import parse_request
        request = parse_request(config, config.get("precision"))
        if request["resolution_mm"] != "native":  # Native publication is n1's original path.
            declared.append("storage")
    return tuple(declared)


def canvas(config, measured_layout):
    """The label grid every frame declares, which must be the qualified phase canvas."""
    expected = measured_layout.get("phase_shape_hw") if isinstance(measured_layout, dict) else None
    frames = config.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("A run configuration with a nonempty frame list is required")
    for frame in frames:
        shape = (frame.get("current") or {}).get("phase_shape_hw") if isinstance(frame, dict) else None
        if shape != expected:
            raise ValueError("Every n1 frame must declare the qualified phase canvas")
    return tuple(expected)


def mass_frame_bytes(inventory, case_ids, *, excluded=()):
    """Per-frame mean and largest per-case bytes outside the preparation stage.

    ``excluded`` names files (such as the rewritten archives) left out. The
    inventory has been validated by the measured estimate.
    """
    from pathlib import PureWindowsPath
    by_case, total = dict.fromkeys(case_ids, 0), 0
    for row in inventory:
        path = PureWindowsPath(row["relative_path"])
        if STAGE_PART in path.parts or path.name in excluded:
            continue
        total += row["bytes"]
        matches = set(path.parts) & set(by_case)
        if len(matches) == 1:
            by_case[next(iter(matches))] += row["bytes"]
    if not by_case or any(value <= 0 for value in by_case.values()):
        raise ValueError("The qualified profile measures no mass-frame files for every case")
    return math.ceil(total / len(by_case)), max(by_case.values())


def pixel_domain_bytes(config, pixels, precision):
    """Per-frame bound of the pixel-domain products and geometry archives."""
    from .option_capacity import npz_bound
    products, geometry = dict(PIXEL_PRODUCTS), dict(PIXEL_GEOMETRY)
    products["approved_analysis_gap_mask"] = 1
    if "n16" in config["pixel_domain"]:
        products.update(N16_PRODUCTS)
        geometry.update(N16_GEOMETRY)
    scalars = {name: SCALAR_BYTES for name in PRODUCT_SCALARS}
    return (npz_bound(dict(_payloads(products, pixels, precision), **scalars))
            + npz_bound(_payloads(geometry, pixels, precision)))


def _descriptor_bytes(value):
    if not isinstance(value, dict) or type(value.get("size_bytes")) is not int or value["size_bytes"] < 0:
        raise ValueError("Sealed phase-one evidence needs exact descriptor sizes")
    return value["size_bytes"]


def storage_bytes(config, *, canvas_hw, precision, extensive):
    """Per-frame bound of the n1 storage layer: copies, coarse frame, receipts, extensive payload."""
    from ..storage_outputs import parse_request
    from ..storage_resolution import metric_grid_shape
    request = parse_request(config, precision)
    rows, columns = metric_grid_shape(request["extent_m"], request["resolution_mm"])
    cells = rows * columns
    phase1 = config.get("phase1")
    if not isinstance(phase1, dict) or not isinstance(phase1.get("frames"), dict) or not phase1["frames"]:
        raise ValueError("The n1 storage layer needs its sealed phase-one evidence")
    copies = []
    for refs in phase1["frames"].values():
        before = refs.get("before_rules") if isinstance(refs, dict) else None
        if not isinstance(before, dict):
            raise ValueError("Sealed phase-one evidence needs exact descriptor sizes")
        value = _descriptor_bytes(refs["commit"]) + _descriptor_bytes(before.get("receipt"))
        value += _descriptor_bytes(before["png"]) if "png" in before else 0
        photo = refs.get("photograph")
        if isinstance(photo, dict) and photo.get("state") == "complete":
            value += _descriptor_bytes(photo.get("receipt")) + _descriptor_bytes(photo.get("png")) + RECEIPT_BYTES
        copies.append(value)
    frame = (_descriptor_bytes(phase1.get("completion")) + max(copies)
             + anonymous_npz_bound(coarse_sizes(cells, precision)) + RECEIPT_BYTES)
    if extensive:
        shared = (config.get("shared") or {}).get("porosity_by_facies")
        if not isinstance(shared, dict) or not shared:
            raise ValueError("The N16 extensive payload needs the run's explicit facies set")
        frame += anonymous_npz_bound(extensive_sizes(cells, precision, len(shared))) + RECEIPT_BYTES
    return frame


def n1_estimate(task, config, actual, binding, *, declared, repository):
    """The upper-bound estimate of a full n1 task on the qualified head.

    The qualified measured estimate validates the profile and its inventory on
    the measured layout; every frame must declare the qualified phase canvas.
    """
    from .capacity import OptionSpaceEstimate, _pinned_writer_rows, _require_full_profile, estimate_run
    _require_full_profile(task, "An n1 task under a qualified native-phase-n1 profile")
    if config.get("precision") != task.get("precision"):
        raise ValueError("The n1 configuration and task precision differ")
    measured = actual.get("measured_native_layout")
    height, width = canvas(config, measured)
    view = dict(task, pipeline=QUALIFIED_PIPELINE)
    frames = dict(run=config.get("run"), precision=config.get("precision"), frames=config.get("frames"))
    core = estimate_run(view, frames, actual, measurement=actual["measurement"],
                        completion=actual["completion"], requested_native_layout=measured,
                        profile_identity=dict(binding), other_run=True)
    pixel = "pixel_domain" in declared
    per_frame, largest = mass_frame_bytes(actual["measurement"]["inventory"], actual["measured_case_ids"],
                                          excluded=ARCHIVES if pixel else ())
    parts, writers = dict(record=dict(frame_bytes=RECORD_BYTES)), list(WRITERS)
    if pixel:
        parts["pixel_domain"] = dict(frame_bytes=pixel_domain_bytes(config, height * width, task["precision"]),
                                     n16="n16" in config["pixel_domain"])
        writers.extend(PIXEL_WRITERS)
    if "storage" in declared:
        extensive = pixel and "n16" in config["pixel_domain"]
        parts["storage"] = dict(frame_bytes=storage_bytes(config, canvas_hw=(height, width),
                                precision=task["precision"], extensive=extensive), extensive=extensive)
        writers.extend(STORAGE_WRITERS + (EXTENSIVE_WRITERS if extensive else ()))
    extra = sum(part["frame_bytes"] for part in parts.values())
    pins = _pinned_writer_rows(repository, task["commit"], writers)
    count = core.case_count
    from .option_capacity import retained_serialization_bytes, run_file_bytes
    retained = retained_serialization_bytes(count, largest + extra,
        run_bytes=run_file_bytes(actual["measurement"]["inventory"], actual["measured_case_ids"]))
    return OptionSpaceEstimate(run=core.run, case_count=count, measured_native_bytes_per_frame=per_frame,
        native_expected_bytes=per_frame * count, coarse_uncompressed_bytes=0,
        header_metadata_copy_estimated_bytes=0, transient_estimated_bytes=largest + extra + retained,
        products_per_frame_estimated_bytes=per_frame + extra, profile=core.profile,
        measured_commit=core.measured_commit, model=N1_MODEL, option_frame_bytes=extra, option_run_bytes=0,
        retained_serialization_estimated_bytes=retained,
        options=dict(route=PIPELINE, declared=list(declared), parts=parts, writers=pins))
