"""Explicit storage profiles for native N1 frame products: 'full' or 'lean'.

``full`` is today's layout and stays the layout of references, qualification
and the exact gates. Its code path is the unchanged writer.

``lean`` changes storage only. The per-frame computation, including the
per-frame interpolation of the static fields through the frame's registration,
is unchanged and still runs in memory. The profile drops three per-frame writes:

* ``mass_total`` in ``products.npz``. A reader rebuilds it as the elementwise
  sum ``mass_free_gas + mass_aqueous`` in their stored dtype (float32 in
  production), with the producer's operand order. The writer computes this
  sum before it drops the array and keeps ``mass_total`` when the result is
  not bit-identical to the computed array.
* the per-frame static fields of ``native_geometry.npz``. Each run stores its
  constant inputs once (``lean_run_static.npz`` with its index
  ``lean_run_static.json``, exclusive create, SHA-256 bound). Each frame
  stores its registration descriptor, its retained per-frame inputs (bound by
  SHA-256) and the identity of every field it would have written. The writer
  regenerates every field once through :func:`read_static_fields`' own
  regeneration and drops only those that are bit-identical; any other field
  keeps being written per frame.
* optionally, the two colour intermediates ``native_log.npy`` and
  ``temporal_log.npy`` of the native/phase series stage (switch ``keep`` or
  ``drop``, default ``keep``).

The profile is chosen per job (N55). ``prepare_job`` records the explicit job
field ``"storage_profile": "full" | "lean"`` in job.json and in every task.json
(:func:`job_storage_profile`); a lean job and its tasks declare
``format_version`` 2, so a reader that predates the field refuses them instead
of writing the full layout. A job or task without the field is a legacy job and
stays ``full`` (:data:`LEGACY_STORAGE_PROFILE`), whatever the default for new
jobs is. A run writes with :func:`run_setting`: the job's profile, plus the
optional run-configuration entry for the lean sub-options::

    "storage_profile": {"profile": "lean", "colour_intermediates": "keep"}

A profile named by that entry must equal the job's choice; otherwise the run
is refused before any output. Both profiles compute identically.
:func:`estimate_job_bytes` is the one per-profile size estimate that
prepare_job, the Jobs panel and lean capacity admission share.

The same entry takes the optional ``"store_colour_coordinate": true`` (default
:data:`DEFAULT_STORE_COLOUR_COORDINATE` = off). In either profile each frame
then also writes ``colour_path_coordinate.npz``: the per-pixel colour-path
coordinate of the frame's own interpretation (see
:data:`COLOUR_COORDINATE_DEFINITION`), recorded with its SHA-256 in
result.json and in the frame's file inventory.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

#: The profile prepare_job and the Jobs panel choose for a NEW job when the
#: caller does not choose one. References, qualification and the exact gates
#: use "full". After the exact gate, production switches this constant to
#: "lean". Existing jobs never follow it (LEGACY_STORAGE_PROFILE).
DEFAULT_STORAGE_PROFILE = "full"
#: The profile of an existing job or task without the explicit field.
LEGACY_STORAGE_PROFILE = "full"

#: Optional per-pixel colour-path coordinate (``"store_colour_coordinate"`` in
#: the storage_profile entry). Off by default: without the explicit setting no
#: frame writes it and every output stays byte-identical to the writer before.
DEFAULT_STORE_COLOUR_COORDINATE = False
COLOUR_COORDINATE_SETTING = "store_colour_coordinate"
COLOUR_COORDINATE_FORMAT = "darsia-colour-path-coordinate-v1"
COLOUR_COORDINATE_ARCHIVE = "colour_path_coordinate.npz"
COLOUR_COORDINATE_ARRAY = "colour_path_coordinate"
COLOUR_COORDINATE_RECORD = "colour_coordinate"
COLOUR_COORDINATE_DEFINITION = (
    "distance parameter of the pixel's baseline-relative, clear-water-normalised RGB "
    "along its facies' frozen colour path (equidistant parametrisation; "
    "PixelMath.fit_color_path inside n1_models.TypedColor), before the path's "
    "value table, the signal model and the titration flash")

PROFILES = ("full", "lean")
COLOUR_RETENTION = ("keep", "drop")
DEFAULT_COLOUR_INTERMEDIATES = "keep"
SETTING = "storage_profile"
LEAN_FORMAT = "darsia-lean-storage-v1"
STORE_FORMAT = "darsia-lean-run-static-v1"
REGENERATION_FORMAT = "darsia-lean-static-regeneration-v1"
STORE_INDEX = "lean_run_static.json"
STORE_ARCHIVE = "lean_run_static.npz"
COLOUR_INTERMEDIATES = ("native_log.npy", "temporal_log.npy")
MASS_TOTAL_RULE = "mass_free_gas + mass_aqueous, elementwise in their stored dtype"
_STORE_ARRAYS = ("physical_facies", "physical_domain", "physical_porosity", "physical_depth_m",
                 "floor_curve", "depth_points", "reference_matrix", "correction_matrix")
_N1_FIELDS = ("pixel_area_m2", "depth_m", "porosity", "facies", "transport_weight_l")
_N16_FIELDS = ("n16_solubility_g_per_l", "full_physical_depth_m", "full_physical_porosity",
               "full_physical_pixel_pore_volume_m3", "full_physical_facies",
               "full_physical_physical_area_m2")


def run_record_files(config):
    """Name the original run records once for the N1 writer and lean accounting."""
    result = dict(manifest="manifest.json", resume="resume.json", store_archive=STORE_ARCHIVE,
                  store_index=STORE_INDEX)
    if "pixel_domain" in config:
        result["scalar_table"] = "mass_table.csv"
    return result


def resolve(config):
    """Return the storage setting one run configuration's entry requests.

    Without the entry a configuration requests nothing: legacy full. Which
    profile a run writes is the job's choice (:func:`run_setting`).
    """
    value = config.get(SETTING) if isinstance(config, dict) else None
    if value is None:
        return dict(profile=LEGACY_STORAGE_PROFILE,
                    colour_intermediates=DEFAULT_COLOUR_INTERMEDIATES)
    if (not isinstance(value, dict) or "profile" not in value
            or set(value) - {"profile", "colour_intermediates", COLOUR_COORDINATE_SETTING}):
        raise ValueError("storage_profile needs 'profile' and optionally "
                         "'colour_intermediates' and '" + COLOUR_COORDINATE_SETTING + "'")
    if not isinstance(value.get(COLOUR_COORDINATE_SETTING, False), bool):
        raise ValueError(COLOUR_COORDINATE_SETTING + " must be true or false")
    profile = value["profile"]
    colour = value.get("colour_intermediates", DEFAULT_COLOUR_INTERMEDIATES)
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("Unknown storage profile: " + repr(profile))
    if not isinstance(colour, str) or colour not in COLOUR_RETENTION:
        raise ValueError("Colour-intermediate retention must be 'keep' or 'drop'")
    if colour == "drop" and profile != "lean":
        raise ValueError("The colour-intermediate retention switch belongs to the lean profile")
    return dict(profile=profile, colour_intermediates=colour)


def stores_colour_coordinate(config):
    """Whether a run stores the per-pixel colour-path coordinate (default off)."""
    resolve(config)  # The whole entry is validated, the flag's type included.
    value = config.get(SETTING) if isinstance(config, dict) else None
    if value is None:
        return DEFAULT_STORE_COLOUR_COORDINATE
    return value.get(COLOUR_COORDINATE_SETTING, DEFAULT_STORE_COLOUR_COORDINATE)


def require_template_colour_coordinate(series_config, template):
    """A series' mass stage reads the option from its template: they must agree."""
    if stores_colour_coordinate(series_config) != stores_colour_coordinate(template):
        raise ValueError("The series colour-coordinate storage differs from its mass template")


# --------------------------------------------------- per-job choice (N55)

#: The explicit field of job.json and task.json.
JOB_FIELD = "storage_profile"
#: The alias job/task format_version each profile declares. Lean is version 2:
#: a reader that predates the field refuses a lean job instead of writing the
#: full layout. Full and legacy stay version 1.
FORMAT_VERSIONS = {"full": 1, "lean": 2}
_ALIAS_FORMATS = ("darsia-alias-job-v1", "darsia-alias-task-v1")
_PHASE_ONE = "native-phase-statistics"
SERIES_ROUTE = "native-phase-n1"
PHASE_TWO_ROUTE = "n1"


def new_job_profile(choice=None):
    """The profile a new job records: the explicit choice, else the default."""
    value = DEFAULT_STORAGE_PROFILE if choice is None else choice
    if not isinstance(value, str) or value not in PROFILES:
        raise ValueError("A job's storage_profile must be 'full' or 'lean', not " + repr(value))
    return value


def expected_format_version(document):
    """The alias format_version a job/task with this storage field declares."""
    if not isinstance(document, dict):
        raise ValueError("A job or task document is required")
    if JOB_FIELD not in document:
        return FORMAT_VERSIONS[LEGACY_STORAGE_PROFILE]
    value = document[JOB_FIELD]
    if not isinstance(value, str) or value not in PROFILES:
        raise ValueError("Unknown job storage_profile: " + repr(value))
    if document.get("pipeline") == _PHASE_ONE:
        raise ValueError("Phase one writes no N1 frame products; its jobs carry no storage_profile")
    return FORMAT_VERSIONS[value]


def declared_storage_profile(document):
    """The profile a job/task document declares (value validated); absent = legacy full."""
    expected_format_version(document)
    return document.get(JOB_FIELD, LEGACY_STORAGE_PROFILE)


def job_storage_profile(document):
    """The explicit storage profile of a job.json or task.json; absent = legacy full.

    The reader validation: the value and, for an alias job or task, the
    format_version it declares (1 for full and legacy, 2 for lean).
    """
    profile = declared_storage_profile(document)
    if (document.get("format") in _ALIAS_FORMATS
            and document.get("format_version") != FORMAT_VERSIONS[profile]):
        raise ValueError("An alias job/task with storage_profile " + repr(profile)
                         + " must declare format_version " + str(FORMAT_VERSIONS[profile]))
    return profile


def require_task_matches_job(task, job):
    """A claimed task writes with its own job's storage profile."""
    if declared_storage_profile(task) != declared_storage_profile(job):
        raise ValueError("Claimed task storage_profile differs from job.json")


def run_setting(task, config):
    """The storage setting one run writes with: its job's choice.

    The task carries its job's explicit profile; a task without the field is a
    legacy task and writes full. The run configuration's optional entry may add
    the lean sub-options. A profile that entry names must equal the job's
    choice, so neither an old task nor a configuration is reinterpreted.
    """
    chosen = declared_storage_profile(task)
    if not isinstance(config, dict) or config.get(SETTING) is None:
        return dict(profile=chosen, colour_intermediates=DEFAULT_COLOUR_INTERMEDIATES)
    requested = resolve(config)
    if requested["profile"] != chosen:
        raise ValueError(
            "The run configuration's storage_profile entry names '" + requested["profile"]
            + "' but its job chose '" + chosen + "'"
            + (" (a task without the field is a legacy full task)" if JOB_FIELD not in task else "")
            + "; prepare a job with the matching storage_profile")
    return requested


def require_run_choice(profile, config, *, pipeline):
    """prepare_job's check of one run's configuration against the job's choice."""
    setting = run_setting({JOB_FIELD: new_job_profile(profile)}, config)
    template = (config.get("mass") or {}).get("template") or {} if pipeline == SERIES_ROUTE else {}
    if SETTING in template and resolve(template) != setting:
        raise ValueError("The series storage_profile differs from its mass template")
    return setting


# ------------------------------------------------- size estimate (N55)

ESTIMATE_FORMAT = "darsia-storage-profile-estimate-v1"
ESTIMATE_LABEL = ("ESTIMATE, not a measurement: documented per-file sizes; "
                  "no measured lean profile exists yet")
#: Per-frame file sizes measured on AC52 (J4 current1d, native-phase-N1 series
#: route: 32.9 GB for 39 frames, about 843 MB per frame; decimal bytes).
AC52_FRAME_BYTES = {
    "native_log.npy": 196_000_000,       # float32 colour intermediate, 3 x 2979 x 5472
    "temporal_log.npy": 196_000_000,     # float32 colour intermediate
    "native_geometry.npz": 307_000_000,  # static fields, re-warped and written per frame
    "phase_area.npz": 65_600_000,        # per-frame pixel_area_mm2
    "products.npz": 40_000_000,          # average (93.8 MB at f43)
    "valid masks": 33_000_000,           # valid masks of the intermediates (1.3 GB / 39)
    "JSON and receipts": 5_000_000,      # 0.2 GB / 39
}
#: The mass_total part of products.npz, which lean omits: the f43 share
#: (30.8 of 93.8 MB) of the 40 MB average.
AC52_MASS_TOTAL_BYTES = 13_100_000
#: A lean run's constant-input store, once per run (AC52's per-run static.npz).
AC52_RUN_STORE_BYTES = 218_000_000
#: Optional in-run coarse layer at 1 mm (AC52 f43, the largest measured frame),
#: per frame and per run; scaled by (1 mm / resolution)^2.
AC52_COARSE_1MM_FRAME_BYTES = 1_850_000
AC52_COARSE_1MM_RUN_BYTES = 4_700_000
#: Optional colour-path coordinate: raw float32 on the AC52 canvas (written
#: compressed, so this is an upper estimate).
AC52_COLOUR_COORDINATE_BYTES = 4 * 3572 * 6304
#: The frame canvas of the AC52 measurement. A run whose declared canvas is
#: larger has the default sizes scaled up by the pixel ratio (never down).
AC52_CANVAS_HW = (3572, 6304)
#: The files a phase-two (N1) frame writes. phase_area, the colour
#: intermediates and their valid masks are phase one's outputs.
PHASE_TWO_FILES = ("native_geometry.npz", "products.npz", "JSON and receipts")
AC52_SOURCE = ("AC52 J4 current1d, native-phase-N1 series route, 39 frames: 32.9 GB, about "
               "843 MB per frame (native_log and temporal_log 196 MB each, native_geometry "
               "307 MB, phase_area 65.6 MB, products 40 MB average, valid masks 33 MB, JSON "
               "5 MB); mass_total 30.8 of 93.8 MB at f43; per-run static store 218 MB")


def _bytes(value, where):
    if type(value) is not int or value < 0:
        raise ValueError(where + " must be a nonnegative integer byte count")
    return value


def estimate_job_bytes(profile, frames, *, runs=1, pipeline=SERIES_ROUTE,
                       colour_intermediates=DEFAULT_COLOUR_INTERMEDIATES, frame_bytes=None,
                       mass_total_bytes=None, run_store_bytes=None, extra_frame_bytes=0,
                       extra_run_bytes=0, source=None):
    """Bytes one storage profile is expected to retain, per frame and for a job.

    The one estimator of prepare_job, the Jobs panel and lean capacity
    admission. Every number comes from explicit per-file sizes (default: the
    measured AC52 sizes), so the result is labelled an ESTIMATE.

    * full keeps every file of the route (the series route writes all of
      AC52_FRAME_BYTES; phase two writes PHASE_TWO_FILES).
    * lean omits native_geometry.npz (its static fields regenerate) and the
      mass_total part of products.npz, and writes one constant-input store per
      run. With colour_intermediates='drop' (lean series route only) the two
      colour intermediates are not retained.

    ``inflight_bytes`` is the allowance for the frame being written, as in the
    measured capacity model, plus with 'drop' the previous frame's colour
    intermediates, which are removed only after the next commit.
    ``needed_bytes`` = ``job_bytes`` + ``inflight_bytes``, before other active
    reservations and the reserve.
    """
    if not isinstance(profile, str) or profile not in PROFILES:
        raise ValueError("Unknown storage profile: " + repr(profile))
    if pipeline not in (SERIES_ROUTE, PHASE_TWO_ROUTE):
        raise ValueError("Storage profiles apply to the N1 frame routes native-phase-n1 and n1 only")
    if type(frames) is not int or frames < 1 or type(runs) is not int or runs < 1:
        raise ValueError("Positive integer frame and run counts are required")
    if colour_intermediates not in COLOUR_RETENTION:
        raise ValueError("Colour-intermediate retention must be 'keep' or 'drop'")
    if colour_intermediates == "drop" and (profile != "lean" or pipeline != SERIES_ROUTE):
        raise ValueError("Only a lean native-phase-N1 series run drops its colour intermediates")
    sizes = dict(AC52_FRAME_BYTES if frame_bytes is None else frame_bytes)
    if set(sizes) != set(AC52_FRAME_BYTES):
        raise ValueError("Per-file sizes must name exactly: " + ", ".join(AC52_FRAME_BYTES))
    for key, value in sizes.items():
        _bytes(value, key)
    mass_total = _bytes(AC52_MASS_TOTAL_BYTES if mass_total_bytes is None else mass_total_bytes,
                        "mass_total")
    if mass_total > sizes["products.npz"]:
        raise ValueError("mass_total cannot exceed products.npz")
    store = _bytes(AC52_RUN_STORE_BYTES if run_store_bytes is None else run_store_bytes, "Run store")
    extra_frame = _bytes(extra_frame_bytes, "Optional per-frame outputs")
    extra_run = _bytes(extra_run_bytes, "Optional per-run outputs")
    names = AC52_FRAME_BYTES if pipeline == SERIES_ROUTE else PHASE_TWO_FILES
    files = {name: sizes[name] for name in names}
    not_retained, per_run = {}, extra_run
    if profile == "lean":
        not_retained["native_geometry.npz"] = files.pop("native_geometry.npz")
        files["products.npz"] -= mass_total
        not_retained["mass_total in products.npz"] = mass_total
        per_run += store
    colour = 0
    if colour_intermediates == "drop":
        for name in COLOUR_INTERMEDIATES:
            not_retained[name] = files.pop(name)
            colour += not_retained[name]
    if extra_frame:
        files["optional outputs"] = extra_frame
    per_frame = sum(files.values())
    inflight = per_frame + 2 * colour
    job = per_frame * frames + per_run * runs
    return dict(format=ESTIMATE_FORMAT, estimate=True, label=ESTIMATE_LABEL, profile=profile,
                pipeline=pipeline, colour_intermediates=colour_intermediates, frames=frames,
                runs=runs, per_frame_bytes=per_frame, per_run_bytes=per_run,
                inflight_bytes=inflight, job_bytes=job, needed_bytes=job + inflight,
                files=files, not_retained=not_retained,
                parameters=dict(frame_bytes=sizes, mass_total_bytes=mass_total,
                                run_store_bytes=store, extra_frame_bytes=extra_frame,
                                extra_run_bytes=extra_run),
                source=AC52_SOURCE if source is None else str(source))


def declared_canvas_pixels(config, *, pipeline):
    """The largest frame canvas a run configuration declares (metadata), or None.

    Phase two: each frame's ``current.phase_shape_hw`` (the N1 frame ABI).
    Series route: each frame's ``mass_spec.photo_top.shape_hw``.
    """
    largest = None
    for frame in config.get("frames") or ():
        if not isinstance(frame, dict):
            return None
        if pipeline == PHASE_TWO_ROUTE:
            shape = (frame.get("current") or {}).get("phase_shape_hw")
        else:
            shape = ((frame.get("mass_spec") or {}).get("photo_top") or {}).get("shape_hw")
        if (not isinstance(shape, (list, tuple)) or len(shape) != 2
                or any(type(size) is not int or size < 1 for size in shape)):
            return None
        largest = max(largest or 0, shape[0] * shape[1])
    return largest


def run_estimate(profile, config, *, pipeline, **sizes):
    """The shared estimate of one run, from its own configuration DATA only.

    It reads the frame count, the lean colour switch, the optional outputs and
    the declared frame canvas. Without explicit sizes, the AC52 defaults are
    scaled up by the canvas pixel ratio when the run's canvas is larger than
    AC52's (never down). A run without a declared canvas, and an optional
    output without a documented size, are refused, never counted as zero.
    """
    import math
    frames = config.get("frames") if isinstance(config, dict) else None
    if not isinstance(frames, list) or not frames:
        raise ValueError("A run configuration with a nonempty frame list is required")
    setting = require_run_choice(profile, config, pipeline=pipeline)
    if "gas_probability_storage" in config:
        raise ValueError("The gas-probability output has no documented size; it needs a measured profile")
    pixels = declared_canvas_pixels(config, pipeline=pipeline)
    if pixels is None:
        raise ValueError("The run configuration declares no frame canvas shape; "
                         "the documented sizes cannot be applied to it")
    explicit = bool(sizes)  # Explicit caller sizes belong to their own layout: never scaled.
    scale = 1.0 if explicit else max(1.0, pixels / (AC52_CANVAS_HW[0] * AC52_CANVAS_HW[1]))
    if scale > 1.0:
        sizes = dict(frame_bytes={name: math.ceil(value * scale)
                                  for name, value in AC52_FRAME_BYTES.items()},
                     mass_total_bytes=math.ceil(AC52_MASS_TOTAL_BYTES * scale),
                     run_store_bytes=math.ceil(AC52_RUN_STORE_BYTES * scale),
                     source=AC52_SOURCE + "; scaled by %.4f to a declared canvas of %d pixels"
                            % (scale, pixels))
    extra_frame = extra_run = 0
    request = config.get("storage")
    if request is not None:
        if pipeline == SERIES_ROUTE:
            raise ValueError("The series route's storage outputs have no documented size; "
                             "they need a measured profile")
        from .storage_resolution import resolution_mm
        step = resolution_mm(request.get("resolution_mm") if isinstance(request, dict) else None)
        if step is not None:  # Native publication is phase two's original path.
            extra_frame = math.ceil(AC52_COARSE_1MM_FRAME_BYTES / step ** 2)
            extra_run = math.ceil(AC52_COARSE_1MM_RUN_BYTES / step ** 2)
    if stores_colour_coordinate(config):
        extra_frame += max(AC52_COLOUR_COORDINATE_BYTES, 4 * pixels)
    value = estimate_job_bytes(profile, len(frames), pipeline=pipeline,
                               colour_intermediates=setting["colour_intermediates"],
                               extra_frame_bytes=extra_frame, extra_run_bytes=extra_run, **sizes)
    return dict(value, declared_canvas_pixels=pixels, ac52_size_scale=None if explicit else scale)


def estimate_prepared_job(profile, tasks, configs):
    """prepare_job's whole-job estimate for the chosen profile, one row per run."""
    rows = []
    for task, config in zip(tasks, configs, strict=True):
        try:
            value = run_estimate(profile, config, pipeline=task.get("pipeline", PHASE_TWO_ROUTE))
        except (KeyError, TypeError, ValueError) as error:
            rows.append(dict(run=task["run"], available=False, reason=str(error)))
            continue
        rows.append(dict(value, run=task["run"], available=True))
    complete = bool(rows) and all(row["available"] for row in rows)
    total = lambda key: sum(row[key] for row in rows) if complete else None
    # In-flight allowances add up like capacity's whole-job report: one per run.
    return dict(format=ESTIMATE_FORMAT, estimate=True, label=ESTIMATE_LABEL, profile=profile,
                complete=complete, runs=rows, frames=total("frames"), job_bytes=total("job_bytes"),
                inflight_bytes=total("inflight_bytes"), needed_bytes=total("needed_bytes"),
                scope="whole job, before other active reservations and the reserve; "
                      "capacity admission repeats its check against fresh free space")


# ---------------------------------------------------- colour-path coordinate


def write_colour_coordinate(directory, response):
    """Write a frame's captured colour-path coordinate next to products.npz.

    The array is the coordinate n1_models.prepare_response captured during the
    frame's own interpretation (float32 on the native grid, NaN outside the
    colour-quantifiable support). Returns the frame's result.json record.
    """
    import numpy as np
    from .mass.n1_pipeline import _write_npz
    value = response.get(COLOUR_COORDINATE_ARRAY)
    shape = response["frame"].domain.shape
    if value is None:
        raise ValueError("The colour-path coordinate was not captured for this frame")
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.shape != shape:
        raise ValueError("The colour-path coordinate must be float32 on the native grid")
    archive = _write_npz(Path(directory) / COLOUR_COORDINATE_ARCHIVE,
                         {COLOUR_COORDINATE_ARRAY: value})
    return dict(format=COLOUR_COORDINATE_FORMAT, path=COLOUR_COORDINATE_ARCHIVE,
                sha256=archive["sha256"], size_bytes=archive["size_bytes"],
                array=COLOUR_COORDINATE_ARRAY, identity=_identity(value),
                definition=COLOUR_COORDINATE_DEFINITION,
                invalid="NaN outside the colour-quantifiable support "
                        "(prepare_response 'quantifiable': support and not an ignored facies)",
                units="colour-path distance (per facies path)")


def read_colour_coordinate(result, directory, *, reader=None):
    """Return a frame's stored colour-path coordinate, file and array verified.

    Both storage profiles store it the same way. A frame written without the
    option has no coordinate: that raises KeyError, never a substitute.
    """
    record = result.get(COLOUR_COORDINATE_RECORD)
    if record is None:
        raise KeyError("This frame stores no colour-path coordinate ("
                       + COLOUR_COORDINATE_SETTING + " was off)")
    if (not isinstance(record, dict) or record.get("format") != COLOUR_COORDINATE_FORMAT
            or record.get("path") != COLOUR_COORDINATE_ARCHIVE
            or record.get("array") != COLOUR_COORDINATE_ARRAY):
        raise ValueError("Unknown colour-path coordinate record")
    reader = _default_reader(reader)
    bound = dict(path=str(Path(directory) / record["path"]), sha256=record["sha256"],
                 size_bytes=record["size_bytes"])
    value = reader.archive(bound, (record["array"],))[record["array"]]
    if _identity(value) != record["identity"]:
        raise ValueError("Colour-path coordinate differs from its recorded identity")
    return value


def frame_storage(result):
    """Return a frame result's lean record, or None for the full layout."""
    value = result.get("lean_storage")
    if value is None:
        return None
    if (not isinstance(value, dict) or value.get("format") != LEAN_FORMAT
            or value.get("profile") != "lean"):
        raise ValueError("Unknown frame storage record")
    return value


def dropped_after_commit(result):
    """Names of retained-stage files a lean frame removed after its commit."""
    value = frame_storage(result)
    if value is None:
        return frozenset()
    names = frozenset(value.get("dropped_after_commit", ()))
    if not names <= set(COLOUR_INTERMEDIATES):
        raise ValueError("Only the colour intermediates may be dropped after commit")
    return names


def store_assets(record):
    """The per-run constant-input files a lean frame depends on (path/SHA)."""
    store = record["store"]
    return [dict(store["index"]), dict(store["archive"])]


def _default_reader(reader):
    if reader is not None:
        return reader
    from .mass.n1_pipeline import DataReader
    return DataReader(unc_only=False)


def _identity(value):
    from .mass.n1_pipeline import _array_identity
    return _array_identity(value)


# ---------------------------------------------------------------- products


def _rebuild_mass_total(arrays):
    import numpy as np
    gas, aqueous = np.asarray(arrays["mass_free_gas"]), np.asarray(arrays["mass_aqueous"])
    if (gas.dtype != aqueous.dtype or gas.shape != aqueous.shape
            or gas.dtype not in (np.dtype(np.float32), np.dtype(np.float64))):
        raise ValueError("mass_total needs two floating component maps of one dtype and grid")
    # The producers form the total as raw_gas + raw_aqueous; keep that order.
    return np.add(gas, aqueous)


def read_products(result, directory, names=None, *, reader=None):
    """Read products.npz of a frame; rebuild and verify an omitted mass_total."""
    reader = _default_reader(reader)
    bound = dict(result["products"], path=str(Path(directory) / result["products"]["path"]))
    record = frame_storage(result)
    omitted = {} if record is None else dict(record["products"]["omitted"])
    import numpy as np
    with np.load(reader.path(bound), allow_pickle=False) as archive:
        stored = list(archive.files)
    names = list(stored) + [name for name in omitted if name not in stored] if names is None else list(names)
    unknown = [name for name in names if name not in stored and name not in omitted]
    if unknown:
        raise KeyError("Frame products do not contain: " + ", ".join(unknown))
    needed = {name for name in names if name in stored}
    if any(name in omitted for name in names):
        needed |= {"mass_free_gas", "mass_aqueous"}
    arrays = reader.archive(bound, tuple(sorted(needed)))
    for name in names:
        if name in omitted and name not in stored:
            if name != "mass_total" or omitted[name].get("rule") != MASS_TOTAL_RULE:
                raise ValueError("Unknown omitted product rule: " + name)
            value = _rebuild_mass_total(arrays)
            if _identity(value) != omitted[name]["identity"]:
                raise ValueError("Rebuilt mass_total differs from its recorded SHA-256")
            arrays[name] = value
    return {name: arrays[name] for name in names}


# ------------------------------------------------------------- run store


class RunStore:
    """Verified constant inputs of one lean run, read back from their files."""

    def __init__(self, document, arrays, index):
        self.document, self.arrays, self.index = document, arrays, dict(index)
        tables = document["tables"]
        self.tables = tables
        self.physical = {name: arrays["physical_" + name]
                         for name in ("facies", "domain", "porosity", "depth_m")}
        self.floor = arrays["floor_curve"]
        self.depth_points = arrays["depth_points"]
        self.reference_matrix = arrays["reference_matrix"]
        self.correction = arrays["correction_matrix"]
        self.porosity_by_facies = {key: value for key, value in tables["porosity_by_facies"]}
        self.swi_by_facies = {key: value for key, value in tables["swi_by_facies"]}
        self.dimensions_m = tuple(tables["dimensions_m"])


def load_store(index, *, reader=None):
    """Read a run's constant-input store, verifying file and array SHA-256."""
    reader = _default_reader(reader)
    document = reader.json(index)
    if (not isinstance(document, dict) or document.get("format") != STORE_FORMAT
            or document.get("complete") is not True
            or set(document.get("arrays", {})) != set(_STORE_ARRAYS)):
        raise ValueError("A complete lean run-static store is required")
    archive = document["archive"]
    if archive.get("path") != STORE_ARCHIVE:
        raise ValueError("The lean run-static archive must sit next to its index")
    # Resolve each original asset independently. A published metadata copy can
    # live elsewhere than its separately collected archive; its physical parent
    # is not the archive's identity. Keep foreign producer paths purely lexical.
    original = str(index["path"])
    windows = PureWindowsPath(original)
    logical = windows if windows.drive or "\\" in original else PurePosixPath(original)
    location = logical.parent / STORE_ARCHIVE
    arrays = reader.archive(dict(archive, path=str(location)), _STORE_ARRAYS)
    for name, value in arrays.items():
        if _identity(value) != document["arrays"][name]:
            raise ValueError("Lean run-static array differs from its SHA-256: " + name)
    return RunStore(document, arrays, index)


def _pairs(mapping):
    return [[key, value] for key, value in mapping.items()]


# ------------------------------------------------------- regeneration


def _transport_weight(domain, depth_m, porosity, unknown_physical_mask, pixel_area_m2, dimensions_m):
    """FrameData.known_physical_domain and NativeAnalysis Geometry, same operations."""
    import numpy as np
    import darsia
    shape = domain.shape
    depth_view = np.broadcast_to(np.asarray(depth_m), shape)
    pore_view = np.broadcast_to(np.asarray(porosity), shape)
    known = domain & np.isfinite(depth_view) & np.isfinite(pore_view) & (depth_view > 0) & (pore_view > 0)
    known &= ~unknown_physical_mask
    depth = np.broadcast_to(np.asarray(depth_m, np.float64), shape).copy()
    pore = np.broadcast_to(np.asarray(porosity, np.float64), shape).copy()
    depth = np.where(known, depth, 0.)
    pore = np.where(known, pore, 0.)
    geometry = darsia.ExtrudedPorousGeometry(porosity=pore, depth=depth, space_dim=2,
                                             num_voxels=shape, dimensions=list(dimensions_m))
    nominal_volume = geometry.voxel_volume.copy()
    nominal_area = float(np.prod(dimensions_m)) / np.prod(shape)
    geometry.voxel_volume = nominal_volume * (pixel_area_m2 / nominal_area)
    geometry.cached_voxel_volume = geometry.voxel_volume.copy()
    return np.asarray(geometry.cached_voxel_volume, np.float64) * 1000.0


class _Regeneration:
    """Recompute one frame's static fields with the run's own library calls."""

    def __init__(self, store, value, reader):
        if (not isinstance(value, dict) or value.get("format") != REGENERATION_FORMAT
                or value.get("route") not in {"n1", "pixel_domain"}
                or value.get("precision") not in {"float32", "float64"}):
            raise ValueError("Unknown lean static regeneration descriptor")
        if value.get("n16") and value["route"] != "pixel_domain":
            raise ValueError("N16 regeneration requires the pixel-domain route")
        self.store, self.value, self.reader = store, value, reader
        self.fields = {}
        self._done = set()

    def field(self, name):
        if name not in self.fields:
            route = self.value["route"]
            if route == "n1" and name in _N1_FIELDS:
                self._n1()
            elif route == "pixel_domain" and name in _N1_FIELDS:
                self._pixel_domain()
            elif route == "pixel_domain" and self.value.get("n16") and name in _N16_FIELDS:
                self._pixel_domain()
                self._n16(name)
            if name not in self.fields:
                raise KeyError("No lean regeneration rule for static field: " + name)
        return self.fields[name]

    def _common(self):
        if "common" in self._done:
            return
        import numpy as np
        from darsia._opencv_runtime import scientific_thread
        scientific_thread("lean-static-regeneration")
        value = self.value
        shape = tuple(int(size) for size in value["phase_shape_hw"])
        h = np.asarray(value["source_to_native_h"], np.float64)
        if h.shape != (3, 3) or not np.isfinite(h).all() or abs(np.linalg.det(h)) < 1e-12:
            raise ValueError("A registered native homography is required")
        origin = value["native_origin_xy"]
        current_matrix = np.asarray([
            [1.0, 0.0, -float(origin[0])],
            [0.0, 1.0, -float(origin[1])],
            [0.0, 0.0, 1.0],
        ]) @ h
        area = np.asarray(self.reader.array(value["inputs"]["area"]), np.float64) / 1e6
        if area.shape != shape or not np.isfinite(area).all() or np.any(area <= 0):
            raise ValueError("Positive original per-pixel areas required")
        self.shape, self.h, self.current_matrix, self.area = shape, h, current_matrix, area
        self.geometry_row = {self.store.tables["geometry_fields"]["source_to_native"]:
                             value["source_to_native_h"]}
        self._done.add("common")

    def _n1(self):
        """N1Pipeline.load_frame and correct_prepared, physical fields only."""
        if "n1" in self._done:
            return
        import numpy as np
        import cv2
        from .mass import n1_geometry as geometry
        from .mass.production_precision import PixelMath
        self._common()
        store, value, shape = self.store, self.value, self.shape
        tables = store.tables
        math = PixelMath(value["precision"])
        physical = geometry.transport_physical_reference(
            store.physical, self.current_matrix, shape, math, cv2)
        domain = self.reader.archive(value["inputs"]["classes"], ("rig_domain",))["rig_domain"]
        if domain.dtype != bool or domain.shape != shape:
            raise ValueError("Rig classes use a different native grid")
        revision = {"phase_shape_hw": list(shape), "native_origin_xy": value["native_origin_xy"]}
        x, y = geometry.frame_coordinates(store.reference_matrix, self.geometry_row, revision,
                                          fields=tables["geometry_fields"])
        floor = geometry.transport_curve(
            store.floor, H_source_to_target=self.h,
            source_origin_xy=tables["floor_origin_xy"],
            target_origin_xy=revision["native_origin_xy"], target_width=shape[1])
        good = np.flatnonzero(np.isfinite(floor))
        if not len(good):
            raise ValueError("Transported sand floor has no valid columns")
        floor[: good[0]], floor[good[-1] + 1:] = floor[good[0]], floor[good[-1]]
        if not np.isfinite(floor).all():
            raise ValueError("Transported floor has an interior gap")
        facies = None
        if not np.array_equal(store.correction, np.eye(3)):
            facies = cv2.warpPerspective(
                np.asarray(store.physical["facies"], np.uint8),
                self.current_matrix @ store.correction, shape[::-1],
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        floor = np.asarray(floor, np.float64)
        below = domain & (np.arange(shape[0])[:, None] > floor[None, :])
        corrected_domain = domain & ~below
        labels = physical["facies"] if facies is None else facies
        corrected, _fill = geometry.fill_to_edges(labels, corrected_domain,
                                                  sand_labels=store.porosity_by_facies)
        measured = geometry.measured_geometry(
            x, y, corrected, store.depth_points,
            porosity_by_facies=store.porosity_by_facies, swi_by_facies=store.swi_by_facies)
        known = corrected_domain & measured["known"]
        weights = _transport_weight(corrected_domain, measured["depth_m"], measured["porosity"],
                                    corrected_domain & ~known, self.area, store.dimensions_m)
        self.fields.update(
            pixel_area_m2=np.broadcast_to(np.asarray(self.area), shape),
            depth_m=np.broadcast_to(np.asarray(measured["depth_m"]), shape),
            porosity=np.broadcast_to(np.asarray(measured["porosity"]), shape),
            facies=corrected, transport_weight_l=weights)
        self._done.add("n1")

    def _pixel_domain(self):
        """pixel_pipeline.prepare_frame and evaluate, physical fields only."""
        if "pixel_domain" in self._done:
            return
        import numpy as np
        import cv2
        from .geometry.pixels import polygon_pixel_domain, extend_column_geometry
        from .mass import n1_geometry as geometry
        from .mass.production_precision import PixelMath
        self._common()
        store, value, shape = self.store, self.value, self.shape
        tables, declared = store.tables, value["pixel_domain"]
        policy = tables["pixel_domain"]
        domain, inside, fixtures = polygon_pixel_domain(
            declared["inner_polygon_native"],
            {name: points for name, points in declared["fixture_polygons_native"]}, list(shape),
            native_origin_xy=declared["native_origin_xy"],
            raster_scale=policy["raster_scale"], raster_shift=policy["raster_shift"])
        revision = dict(phase_shape_hw=list(shape), native_origin_xy=declared["native_origin_xy"])
        x, y = geometry.frame_coordinates(store.reference_matrix, self.geometry_row, revision,
                                          fields=tables["geometry_fields"])
        math = PixelMath(value["precision"])
        transported = geometry.transport_physical_reference(
            store.physical, self.current_matrix @ store.correction, shape, math, cv2)
        columns = extend_column_geometry(
            transported["facies"], transported["domain"], domain, y,
            porosity_by_facies=store.porosity_by_facies, swi_by_facies=store.swi_by_facies)
        known = domain & ~columns.geometry_unknown
        depth = self.reader.archive(value["inputs"]["basis_geometry"], ("depth_m",))["depth_m"]
        if depth.dtype != np.float64 or depth.shape != shape:
            raise ValueError("Target measured depth must be float64 on the full grid")
        area = np.asarray(self.area, np.float64)
        weights = _transport_weight(domain, depth, columns.porosity, domain & ~known,
                                    self.area, store.dimensions_m)
        self.fields.update(pixel_area_m2=area, depth_m=depth, porosity=columns.porosity,
                           facies=columns.facies, transport_weight_l=weights)
        self.transported, self.inside, self.fixtures, self.y = transported, inside, fixtures, y
        self.depth = depth
        self._done.add("pixel_domain")

    def _n16(self, name):
        """pixel_pipeline._prepare_n16 geometry and prepare_response physics."""
        import numpy as np
        import darsia
        store, value, shape = self.store, self.value, self.shape
        if name == "n16_solubility_g_per_l":
            from .mass.n1_models import TypedPhysics
            from .mass.production_precision import PixelMath
            physics = value["physics"]
            math = PixelMath(value["precision"])
            scalar = darsia.ScalarImage(np.asarray(np.zeros(shape, math.dtype), math.dtype),
                                        dimensions=list(store.dimensions_m))
            scalar.time = float(physics["elapsed_h"]) * 3600
            state = TypedPhysics.__new__(TypedPhysics)
            state.pixel_math = math
            TypedPhysics.__init__(
                state, scalar, atmospheric_pressure=physics["pressure_bar"],
                atmospheric_temperature=physics["temperature_c"],
                atmospheric_pressure_gradient=physics["pressure_gradient_bar_per_m"],
                atmospheric_temperature_gradient=physics["temperature_gradient_c_per_m"])
            self.fields[name] = state.solubility_co2
            return
        if "n16" in self._done:
            return
        from .geometry.pixels import hidden_physical_geometry
        bound = value["inputs"]["not_sand"]
        not_sand = np.zeros(shape, bool) if bound is None else self.reader.array(bound)
        if not_sand.dtype != bool or not_sand.shape != shape:
            raise ValueError("N16 bound not_sand mask differs from the native grid")
        transported = self.transported
        physical = hidden_physical_geometry(
            transported["facies"], transported["domain"], self.inside, self.fixtures, self.y,
            porosity_by_facies=store.porosity_by_facies, swi_by_facies=store.swi_by_facies)
        inside = physical["physical_mask"] & ~not_sand
        columns = physical["geometry"]
        area = np.asarray(self.area, np.float64)
        depth = np.asarray(self.depth, np.float64)
        phi = np.asarray(columns.porosity, np.float64)
        dimensions = store.dimensions_m
        geometry = darsia.ExtrudedPorousGeometry(
            porosity=np.where(inside, phi, 0.), depth=np.where(inside, depth, 0.), space_dim=2,
            num_voxels=inside.shape, dimensions=list(dimensions))
        nominal_volume = geometry.voxel_volume.copy()
        nominal_area = float(np.prod(dimensions)) / np.prod(inside.shape)
        geometry.voxel_volume = nominal_volume * (area / nominal_area)
        geometry.cached_voxel_volume = geometry.voxel_volume.copy()
        self.fields.update(
            full_physical_depth_m=depth, full_physical_porosity=phi,
            full_physical_pixel_pore_volume_m3=np.asarray(geometry.cached_voxel_volume, np.float64),
            full_physical_facies=columns.facies, full_physical_physical_area_m2=area)
        self._done.add("n16")


def regenerate_static(store, regeneration, names, *, reader=None):
    """Recompute named static fields from a run store and a frame descriptor."""
    engine = _Regeneration(store, regeneration, _default_reader(reader))
    return {name: engine.field(name) for name in names}


def read_static_fields(result, directory, names=None, *, reader=None, store=None):
    """Return a frame's static fields, written or regenerated and SHA-verified.

    A full frame reads native_geometry.npz. A lean frame reads its written
    fallback fields and regenerates the others from the run store and its
    registration descriptor. Every returned array must equal the identity the
    producer recorded; a mismatch raises ValueError.
    """
    reader = _default_reader(reader)
    static = result["static"]
    identities = static["arrays"]
    names = list(identities) if names is None else list(names)
    unknown = [name for name in names if name not in identities]
    if unknown:
        raise KeyError("Frame static fields do not contain: " + ", ".join(unknown))
    storage = static.get("storage")
    written = list(identities) if storage is None else list(storage["written"])
    arrays = {}
    selected = [name for name in names if name in written]
    if selected:
        if static["asset"] is None:
            raise ValueError("Written static fields need their native_geometry.npz")
        bound = dict(static["asset"], path=str(Path(directory) / static["asset"]["path"]))
        arrays.update(reader.archive(bound, tuple(selected)))
    rest = [name for name in names if name not in written]
    if rest:
        if storage.get("format") != LEAN_FORMAT or set(rest) - set(storage["regenerated"]):
            raise ValueError("Static field is neither written nor regenerable")
        regeneration = storage["regeneration"]
        if store is None:
            store = load_store(regeneration["store"], reader=reader)
        engine = _Regeneration(store, regeneration, reader)
        for name in rest:
            arrays[name] = engine.field(name)
    for name in names:
        if _identity(arrays[name]) != identities[name]:
            raise ValueError("Static field differs from its recorded SHA-256: " + name)
    return {name: arrays[name] for name in names}


# ------------------------------------------------------------ writer


class LeanRun:
    """One lean run: its per-run constant-input store and write-time gates."""

    def __init__(self, output, config, pipeline, setting, *, colour_intermediates_produced):
        from .mass.n1_pipeline import _write_json, _write_npz
        import numpy as np
        if setting.get("profile") != "lean":
            raise ValueError("LeanRun requires the explicit lean profile")
        if setting["colour_intermediates"] == "drop" and not colour_intermediates_produced:
            raise ValueError("This stage writes no colour intermediates; set their retention "
                             "on the native/phase series run")
        self.setting = dict(setting)
        self.output = Path(output)
        self.reader = pipeline.reader
        shared = config["shared"]
        arrays = dict(
            physical_facies=np.asarray(pipeline.physical["facies"]),
            physical_domain=np.asarray(pipeline.physical["domain"]),
            physical_porosity=np.asarray(pipeline.physical["porosity"]),
            physical_depth_m=np.asarray(pipeline.physical["depth_m"]),
            floor_curve=np.asarray(pipeline.floor),
            depth_points=np.asarray(pipeline.depth_points, np.float64),
            reference_matrix=np.asarray(pipeline.reference_matrix, np.float64),
            correction_matrix=np.asarray(pipeline.correction, np.float64))
        tables = dict(
            porosity_by_facies=_pairs(shared["porosity_by_facies"]),
            swi_by_facies=_pairs(shared["swi_by_facies"]),
            floor_origin_xy=list(shared["floor_origin_xy"]),
            dimensions_m=list(config["dimensions_m"]),
            geometry_fields=dict(config["geometry_fields"]))
        if "pixel_domain" in config:
            policy = config["pixel_domain"]
            tables["pixel_domain"] = dict(raster_scale=policy["raster_scale"],
                                          raster_shift=policy["raster_shift"],
                                          fixture_names=list(policy["fixture_names"]))
        sources = dict(physical_reference=deepcopy(shared["physical_reference"]),
                       floor_curve=deepcopy(shared["floor_curve"]),
                       depth_points=deepcopy(shared["depth_points"]),
                       physical_correction=deepcopy(config["physical_correction"]))
        for key in ("reference_matrix", "physical_coordinate_provenance", "legacy_dimensions_m"):
            if key in shared:
                sources[key] = deepcopy(shared[key])
        archive = _write_npz(self.output / STORE_ARCHIVE, arrays)
        document = dict(
            format=STORE_FORMAT, complete=True, run=config["run"], profile="lean",
            archive=dict(path=STORE_ARCHIVE, sha256=archive["sha256"],
                         size_bytes=archive["size_bytes"]),
            arrays={name: _identity(value) for name, value in arrays.items()},
            tables=tables, sources=sources,
            role="constant inputs of the per-frame static fields, stored once per run")
        from .pipeline.io_ops import LEAN_METADATA_BYTES, metadata_ceiling
        with metadata_ceiling(LEAN_METADATA_BYTES):
            index = _write_json(self.output / STORE_INDEX, document)
        self.store_record = dict(index=dict(index), archive=dict(archive))
        # The write-time gate uses the store exactly as a reader loads it.
        self.store = load_store(dict(index), reader=self.reader)
        self._last_stage = None

    def manifest_record(self):
        return dict(self.setting, format=LEAN_FORMAT, store=deepcopy(self.store_record))

    def outcome_record(self):
        return dict(self.setting, format=LEAN_FORMAT, store=deepcopy(self.store_record))

    def frame(self, spec, evaluated):
        return LeanFrame(self, spec, evaluated)

    # Colour intermediates of the native/phase series stage.
    def committed(self, stage):
        """After a frame commit: drop the previous frame's colour intermediates."""
        if self.setting["colour_intermediates"] != "drop":
            return
        previous, self._last_stage = self._last_stage, Path(stage)
        if previous is not None:
            self._drop(previous)

    def finish(self):
        """After the run: drop the last frame's colour intermediates."""
        if self.setting["colour_intermediates"] == "drop" and self._last_stage is not None:
            self._drop(self._last_stage)
            self._last_stage = None

    def _drop(self, stage):
        root = (self.output / "preparation").resolve()
        stage = Path(stage)
        if stage.resolve().parent != root:
            raise ValueError("Only this attempt's own preparation files may be dropped")
        for name in COLOUR_INTERMEDIATES:
            path = stage / name
            if path.exists():
                path.unlink()


class LeanFrame:
    """Write-time gates of one frame; nothing here changes a computed value."""

    def __init__(self, run, spec, evaluated):
        self.run, self.spec, self.evaluated = run, spec, evaluated
        self.products_record = None
        self.storage = None

    def products(self, products):
        """Return the arrays to write; omit mass_total only if rebuilt bit for bit."""
        written = dict(products)
        omitted, kept = {}, {}
        if "mass_total" in products:
            identity = _identity(products["mass_total"])
            try:
                same = _identity(_rebuild_mass_total(products)) == identity
                reason = None if same else "rebuilt sum is not bit-identical"
            except (KeyError, TypeError, ValueError) as error:
                same, reason = False, type(error).__name__ + ": " + str(error)
            if same:
                del written["mass_total"]
                omitted["mass_total"] = dict(identity=identity, rule=MASS_TOTAL_RULE)
            else:
                kept["mass_total"] = reason
        self.products_record = dict(stored=list(written), omitted=omitted, kept_reasons=kept)
        return written

    def regeneration(self):
        """Registration descriptor and retained inputs of this frame's static fields."""
        spec, evaluated, run = self.spec, self.evaluated, self.run
        frame = evaluated["response"]["frame"]
        current = spec["current"]
        fields = run.store.tables["geometry_fields"]
        area = dict(spec["area"])
        area.setdefault("array_key", "pixel_area_mm2")
        route = "pixel_domain" if "pixel_domain" in evaluated else "n1"
        value = dict(
            format=REGENERATION_FORMAT, route=route, n16="n16_producer" in evaluated,
            precision=evaluated["result"]["precision"], store=deepcopy(run.store_record["index"]),
            source_to_native_h=deepcopy(current["binding"]["geometry"][fields["source_to_native"]]),
            native_origin_xy=deepcopy(current["native_origin_xy"]),
            phase_shape_hw=[int(size) for size in frame.domain.shape],
            inputs=dict(area=area))
        if route == "n1":
            value["inputs"]["classes"] = deepcopy(spec["classes"])
        else:
            declared = spec["pixel_domain_geometry"]
            value["pixel_domain"] = dict(
                inner_polygon_native=deepcopy(declared["inner_polygon_native"]),
                fixture_polygons_native=[[name, deepcopy(points)] for name, points
                                         in declared["fixture_polygons_native"].items()],
                native_origin_xy=deepcopy(declared["native_origin_xy"]))
            value["inputs"]["basis_geometry"] = dict(
                deepcopy(spec["frozen_response_basis"]["geometry"]), array_key="depth_m")
            if value["n16"]:
                value["inputs"]["not_sand"] = deepcopy(declared["n16_masks"]["not_sand"])
                value["physics"] = dict(
                    pressure_bar=frame.pressure_bar, temperature_c=frame.temperature_c,
                    pressure_gradient_bar_per_m=frame.pressure_gradient_bar_per_m,
                    temperature_gradient_c_per_m=frame.temperature_gradient_c_per_m,
                    elapsed_h=frame.elapsed_h)
        return value

    def geometry(self, temporary, fields):
        """Write only the fields the regeneration does not reproduce bit for bit."""
        from .mass.n1_pipeline import _write_npz, _jsonable
        identities = {name: _identity(value) for name, value in fields.items()}
        reasons, regenerated = {}, []
        try:
            # Gate with the descriptor exactly as result.json will return it.
            regeneration = json.loads(json.dumps(_jsonable(self.regeneration()), allow_nan=False))
            engine = _Regeneration(self.run.store, regeneration, self.run.reader)
        except (KeyError, TypeError, ValueError) as error:
            regeneration, engine = None, None
            reasons = {name: "no regeneration descriptor: " + type(error).__name__ + ": " + str(error)
                       for name in fields}
        if engine is not None:
            for name in fields:
                try:
                    same = _identity(engine.field(name)) == identities[name]
                    if not same:
                        reasons[name] = "regenerated field is not bit-identical"
                except Exception as error:  # Any failure keeps the field written.
                    reasons[name] = type(error).__name__ + ": " + str(error)
                else:
                    if same:
                        regenerated.append(name)
            del engine
        written = [name for name in fields if name not in regenerated]
        asset = None
        if written:
            archive = _write_npz(Path(temporary) / "native_geometry.npz",
                                 {name: fields[name] for name in written})
            asset = {"path": "native_geometry.npz", "sha256": archive["sha256"],
                     "size_bytes": archive["size_bytes"]}
        self.storage = dict(format=LEAN_FORMAT, profile="lean", written=written,
                            regenerated=regenerated, kept_reasons=reasons,
                            regeneration=regeneration)
        return asset, identities

    def record(self):
        """The frame's result.json lean record."""
        value = dict(format=LEAN_FORMAT, profile="lean",
                     colour_intermediates=self.run.setting["colour_intermediates"],
                     store=deepcopy(self.run.store_record), products=deepcopy(self.products_record),
                     static=dict(written=list(self.storage["written"]),
                                 regenerated=list(self.storage["regenerated"])))
        if self.run.setting["colour_intermediates"] == "drop":
            value["dropped_after_commit"] = list(COLOUR_INTERMEDIATES)
        return value
