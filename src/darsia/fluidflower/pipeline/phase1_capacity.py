"""Capacity of a phase-one statistics task under a qualified native-phase-n1 profile.

Phase one (pipeline ``native-phase-statistics``) runs the same native and phase
stages as the qualified series route on the same frame DATA, and writes the
same per-frame preparation files, but no mass. Its own additions per frame
are the before-rules score PNG, its receipts and statistics, and the optional
gas probability and original photograph.

A full phase-one task on the qualified head (commit, CPU, float32) is admitted
by this rule as another run. The estimate is the qualified profile's measured
preparation bytes per frame, plus an upper bound for each addition. The
storage layer, the pixel-domain route and the other optional outputs stay
refused, as on the series route.
"""
from __future__ import annotations

import math

PIPELINE = "native-phase-statistics"
QUALIFIED_PIPELINE = "native-phase-n1"
#: The stage directory of the native and phase files in a measured inventory.
STAGE_PART = "preparation"
#: Per frame: phase_before.json, native_statistics.json.gz (at most 0.55 MB on
#: the qualified RAW series route), statistics_geometry_report.json,
#: native_statistics_runtime.json, the photograph receipt or pending record, and
#: phase1_commit.json.
RECEIPTS_BYTES = 8 << 20
#: Per run: manifest.json and phase1_completed.json.
RUN_BYTES = 8 << 20
#: A PNG's signature, IHDR and IEND; then one record per IDAT chunk, which Pillow
#: writes at most 64 KiB at a time, and an allowance for ancillary chunks.
PNG_FIXED_BYTES = 8 + 25 + 12
PNG_CHUNK_BYTES = 12
PNG_CHUNK_PAYLOAD = 1 << 16
PNG_ANCILLARY_BYTES = 1 << 10
#: The zlib header and Adler-32 checksum around the PNG's deflate stream.
ZLIB_WRAPPER_BYTES = 6
PHASE1_MODEL = ("UPPER BOUND, not a measurement: the qualified profile's measured preparation "
                "bytes per frame, plus the before-rules PNG, receipts and declared options as "
                "uncompressed bounds; no assumed compression gain")
#: The phase-one writers this rule pins, beside the qualified profile's own writers.
WRITERS = ("src/darsia/fluidflower/pipeline/phase_statistics.py",
           "src/darsia/fluidflower/storage_outputs.py")
PHOTOGRAPH_WRITERS = ("src/darsia/fluidflower/storage_original_photo.py",)


def pillow_zlib_ng():
    """Whether Pillow's own PNG compressor is zlib-ng; an unknown one counts as zlib-ng."""
    from PIL import features
    version = features.version("zlib")
    return version is None or "ng" in str(version)


def png_bound(height, width, channels):
    """Upper bound on an 8-bit PNG of height x width x channels, as Pillow writes it.

    The deflate bound follows the library Pillow's PNG writer really uses.
    """
    from .option_capacity import deflate_bound
    stream = (deflate_bound(height * (width * channels + 1), zlib_ng=pillow_zlib_ng())
              + ZLIB_WRAPPER_BYTES)
    return (PNG_FIXED_BYTES + stream + PNG_CHUNK_BYTES * (stream // PNG_CHUNK_PAYLOAD + 1)
            + PNG_ANCILLARY_BYTES)


def stage_bytes(inventory, case_ids):
    """Per-frame mean and largest per-case bytes of the measured preparation files.

    The inventory has been validated by the measured estimate. A case without
    preparation files is refused.
    """
    from pathlib import PureWindowsPath
    by_case = dict.fromkeys(case_ids, 0)
    for row in inventory:
        parts = set(PureWindowsPath(row["relative_path"]).parts)
        if STAGE_PART in parts:
            matches = parts & set(by_case)
            if len(matches) == 1:
                by_case[next(iter(matches))] += row["bytes"]
    if not by_case or any(value <= 0 for value in by_case.values()):
        raise ValueError("The qualified profile measures no preparation files for every case")
    return math.ceil(sum(by_case.values()) / len(by_case)), max(by_case.values())


def _photographs(config):
    """The frames whose original photograph the configuration requests."""
    value = config.get("photographs") or {}
    cases = [frame.get("case_id") for frame in config.get("frames") or ()]
    if not isinstance(value, dict) or not set(value) <= set(cases):
        raise ValueError("Photograph requests must name frames of this run")
    return sorted(value)


def validate_requests(config):
    """Validate P1 evidence before selecting its separate optional outputs.

    Phase one requires a native BEFORE evidence request named ``storage``.
    It is part of the validated core, while the generic native-phase-n1
    option selector uses that name for a phase-two mass-output request.
    """
    from . import option_capacity
    from .staged_data import validate_phase1_run
    validate_phase1_run(config)
    return option_capacity.validate_requests({key: value for key, value in config.items()
                                             if key != "storage"})


def phase1_additions(config, *, canvas_hw, precision, declared, include_core=True):
    """Use the phase-one writer's single envelope for core records and declared extras."""
    from . import option_capacity
    height, width = canvas_hw
    photographs = _photographs(config)
    parts = dict(before_rules_png=dict(frame_bytes=png_bound(height, width, 2),
                                       source="LA uint8 on the label grid, uncompressed"),
                 receipts=dict(frame_bytes=RECEIPTS_BYTES, source="receipts and statistics allowance"))
    if not include_core:
        parts = {}
    writers = list(WRITERS) if include_core else []
    for key in declared:
        parts[key] = option_capacity.BUDGETS[key](config, canvas_hw=(height, width), precision=precision)
        writers.extend(option_capacity.WRITERS[key])
    if photographs:
        parts["photographs"] = dict(frame_bytes=png_bound(height, width, 4) + RECEIPTS_BYTES,
                                    source="RGBA uint8 on the label grid, uncompressed; counted for every frame",
                                    frames=photographs)
        writers.extend(PHOTOGRAPH_WRITERS)
    return dict(parts=parts, writers=list(dict.fromkeys(writers)),
        frame_bytes=sum(part["frame_bytes"] for part in parts.values()), run_bytes=RUN_BYTES)


def phase1_estimate(task, config, actual, binding, *, requested, declared, repository):
    """The upper-bound estimate of a full phase-one task on the qualified head.

    The qualified measured estimate validates the profile, the measured
    inventory and the task's native layout (another run of the same stages).
    """
    from . import option_capacity
    from .capacity import (OptionSpaceEstimate, _pinned_writer_rows, _require_full_profile,
                           estimate_run)
    _require_full_profile(task, "A phase-one task under a qualified native-phase-n1 profile")
    view = dict(task, pipeline=QUALIFIED_PIPELINE)
    frames = dict(run=config.get("run"), precision=config.get("precision"), frames=config.get("frames"))
    core = estimate_run(view, frames, actual, measurement=actual["measurement"],
                        completion=actual["completion"], requested_native_layout=requested,
                        profile_identity=dict(binding), other_run=True)
    per_frame, largest = stage_bytes(actual["measurement"]["inventory"], actual["measured_case_ids"])
    additions = phase1_additions(config, canvas_hw=requested["phase_shape_hw"],
        precision=task["precision"], declared=declared)
    parts, writers, extra = additions["parts"], additions["writers"], additions["frame_bytes"]
    pins = _pinned_writer_rows(repository, task["commit"], writers)
    count = core.case_count
    retained = option_capacity.retained_serialization_bytes(count, largest + extra, run_bytes=RUN_BYTES)
    # The per-run files are reserved in the frame in flight, as the ledger
    # reserves only per-frame bytes and the transient.
    return OptionSpaceEstimate(run=core.run, case_count=count, measured_native_bytes_per_frame=per_frame,
        native_expected_bytes=per_frame * count, coarse_uncompressed_bytes=0,
        header_metadata_copy_estimated_bytes=0, transient_estimated_bytes=largest + extra + RUN_BYTES + retained,
        products_per_frame_estimated_bytes=per_frame + extra, profile=core.profile,
        measured_commit=core.measured_commit, model=PHASE1_MODEL, option_frame_bytes=extra,
        option_run_bytes=RUN_BYTES, retained_serialization_estimated_bytes=retained,
        options=dict(route=PIPELINE, declared=list(declared), parts=parts,
                                                 writers=pins))
