"""Source-bound estimates for the original F-off calibration refit.

These are estimates, not measurements or an operating-system memory limit.
Statistics are decoded one file at a time to count the original retained rows
and pair arrays. No RAW/image product, fit, sampling or scientific arithmetic
is performed. The independent accepted service/cache admission still applies.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PureWindowsPath
import re
import zlib

from .job_inputs import MAX_METADATA_BYTES

PIPELINE = "n1-calibration-refit"
MODEL = ("ESTIMATE, not a measurement: bounded decoded original statistics, "
         "declared grids and JSON serialization margins; workspace is an "
         "estimate of all retained rows/compiled pairs, not an OS RAM cap")
# Original capacity._writer_identity admits only the product namespace.
PRODUCT_WRITERS = (
    "src/darsia/fluidflower/pipeline/refit_capacity.py",
    "src/darsia/fluidflower/pipeline/refit_runner.py",
    "src/darsia/fluidflower/pipeline/refit_jobs.py",
    "src/darsia/fluidflower/pipeline/job_inputs.py",
    "src/darsia/fluidflower/pipeline/run_io.py",
    "src/darsia/fluidflower/pipeline/worker.py",
    "src/darsia/fluidflower/pipeline/worker_io.py",
    "src/darsia/fluidflower/pipeline/capacity.py",
    "src/darsia/fluidflower/pipeline/publication_runtime.py",
    "src/darsia/fluidflower/pipeline/common_calibration.py",
    "src/darsia/fluidflower/pipeline/job_queue.py",
    "src/darsia/fluidflower/mass/n1_pipeline.py",
    "src/darsia/fluidflower/mass/n1_refit.py",
    "src/darsia/fluidflower/mass/n1_statistics.py",
    "src/darsia/fluidflower/mass/n1_arithmetic.py",
    "src/darsia/fluidflower/mass/concentration_joint.py",
)
# Cache modules remain mandatory in the independently verified startup closure;
# they are not cast into the original product-writer admission namespace.
REQUIRED_SERVICE_SOURCES = PRODUCT_WRITERS + (
    "src/darsia_cache/control.py",
    "src/darsia_cache/worker_adapter.py",
)
_RULES = {"format", "stage1_window_h", "stage2_start_h", "post_bin_edges_h", "width_limit",
          "anchor_target_h", "anchor_tolerance_h", "baseline", "gain_starts", "beta_starts",
          "gain_grid", "beta_grid", "require_frame_list_evidence"}
_PAIRS = ("own_native_relative", "column_native_relative", "solubility_ratio", "kind", "aqueous_capacity_g")


def read_statistics_bound(path, descriptor):
    """Original strict JSON semantics, with exact pin and bounded gzip expansion."""
    from ..mass.n1_pipeline import _strict_json
    from .job_queue import retry_transient
    path = Path(path)
    if (not isinstance(descriptor, dict) or not {"sha256", "size_bytes"} <= descriptor.keys()
            or re.fullmatch(r"[0-9a-f]{64}", str(descriptor["sha256"])) is None
            or type(descriptor["size_bytes"]) is not int
            or not 0 <= descriptor["size_bytes"] <= MAX_METADATA_BYTES):
        raise ValueError("Mass calibration statistics require a bounded full SHA/size descriptor")
    def read():
        before = path.stat()
        if before.st_size != descriptor["size_bytes"]:
            raise ValueError("Statistics file changed: " + str(path))
        with path.open("rb") as stream:
            raw = stream.read(descriptor["size_bytes"] + 1)
        after = path.stat()
        stamp = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        if (stamp(before) != stamp(after) or len(raw) != descriptor["size_bytes"]
                or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]):
            raise ValueError("Statistics file changed: " + str(path))
        return raw
    raw = retry_transient(read)
    if raw[:2] == b"\x1f\x8b":
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                decoded = stream.read(MAX_METADATA_BYTES + 1)
        except (OSError, EOFError, zlib.error) as error:
            raise ValueError("Mass calibration statistics gzip is invalid") from error
    else:
        decoded = raw
    if len(decoded) > MAX_METADATA_BYTES:
        raise ValueError("Decoded mass calibration statistics exceed the metadata byte limit")
    return _strict_json(decoded), len(decoded)


def _finite(value):
    if isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, list):
        for item in value:
            _finite(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Mass calibration capacity requires finite declared numbers")


def _grid_points(grid):
    if (not isinstance(grid, dict) or not {"spacing", "wide", "local"} <= grid.keys()
            or grid.keys() - {"spacing", "wide", "local", "extra"}
            or grid["spacing"] not in {"linear", "geometric"}
            or not isinstance(grid["wide"], list) or not grid["wide"]):
        raise ValueError("Explicit original mass calibration grid required")
    local_fields = ({"factor_low", "factor_high", "floor", "points"} if grid["spacing"] == "geometric"
                    else {"half_width", "floor", "points"})
    if not isinstance(grid["local"], dict) or set(grid["local"]) != local_fields:
        raise ValueError("Unknown mass calibration grid definition")
    counts = [grid["local"]["points"]]
    for segment in grid["wide"]:
        if not isinstance(segment, list) or len(segment) != 3:
            raise ValueError("Explicit original mass calibration grid segments required")
        counts.append(segment[2])
    if any(type(count) is not int or count < 1 for count in counts):
        raise ValueError("Mass calibration grid point counts must be positive integers")
    extra = grid.get("extra", [])
    if not isinstance(extra, list):
        raise ValueError("Explicit mass calibration extra grid points required")
    return sum(counts) + len(extra) + 1


def _require_task_route(task):
    from ..storage_profile import declared_storage_profile
    if (task.get("pipeline") != PIPELINE or task.get("backend") != "cpu"
            or task.get("precision") != "float32" or type(task.get("threads")) is not int or task["threads"] != 1
            or task.get("format") != "darsia-alias-task-v1" or task.get("format_version") != 1
            or declared_storage_profile(task) != "full"
            or re.fullmatch(r"[0-9a-f]{40}", str(task.get("commit", ""))) is None):
        raise ValueError("Mass calibration capacity requires the explicit CPU/float32/full single-thread alias route")


def require_plan(task, plan):
    """Keep the original F-off rules; refuse unbudgeted routes/options/order."""
    from .refit_runner import validate_plan
    _require_task_route(task)
    fields = {"format", "run", "fold", "rules", "runs", "statistics", "baseline"}
    individual = ({"calibration_scope", "calibrated_run", "source_roles"}
                  if plan.get("calibration_scope") == "individual" else set())
    if (not isinstance(plan, dict) or not fields <= plan.keys()
            or not isinstance(plan["rules"], dict)
            or not isinstance(plan["statistics"], dict)
            or not isinstance(plan["runs"], dict)
            or plan.keys() - (fields | individual | {"variant_f", "colour_paths", "common_phase1", "data_aliases", "chemistry_model", "image_selection"})
            or plan["run"] != task.get("run")
            or plan.get("colour_paths", "as in the phase-one statistics") != "as in the phase-one statistics"
            or plan.get("variant_f", {"enabled": False}) != {"enabled": False}
            or not _RULES <= plan["rules"].keys()
            or plan["rules"].keys() - (_RULES | {"target_times_h"})):
        raise ValueError("Unknown mass calibration numerical/options contract or task/plan mismatch")
    _finite(plan)
    from .job_inputs import InputCatalog
    InputCatalog({}, plan.get("data_aliases", ()))  # Original same-byte alias guard; no file opens.
    rules = validate_plan(plan)
    rules.require_variant_f_off()
    cases = [entry.get("case_id") for run in sorted(plan["statistics"]) for entry in plan["statistics"][run]]
    if (not cases or any(not isinstance(case, str) or not case for case in cases)
            or len(set(cases)) != len(cases) or task.get("ordered_case_ids") != cases
            or type(plan["rules"]["require_frame_list_evidence"]) is not bool):
        raise ValueError("Mass calibration capacity requires the exact original case order and evidence choice")
    for key in ("gain_starts", "beta_starts"):
        if not isinstance(plan["rules"][key], list) or not plan["rules"][key]:
            raise ValueError("Mass calibration capacity requires the original nonempty optimizer starts")
    return rules, sum(_grid_points(plan["rules"][key]) for key in ("gain_grid", "beta_grid"))


@dataclass(frozen=True)
class RefitEstimate:
    run: str
    case_count: int
    products_estimated_bytes: int
    transient_estimated_bytes: int
    products_per_frame_estimated_bytes: int
    workspace_estimated_bytes: int
    estimate: dict
    writer_identity: list
    task_binding: dict
    model: str = MODEL

    def needed(self, *, other_remaining_bytes=0, reserve_bytes=40 * 1024**3):
        from .capacity import _integer
        return (self.products_estimated_bytes + self.transient_estimated_bytes
                + _integer(other_remaining_bytes, "Other active remaining bytes")
                + _integer(reserve_bytes, "Reserve bytes"))


def budget(task, plan, *, path_for_use):
    """Read/count the original bound statistics, never fit or retain all rows."""
    from ..mass.n1_statistics import CONTRACT
    rules, grids = require_plan(task, plan)
    decoded_total = pair_total = histogram_total = largest_decoded = largest_pairs = 0
    inputs = []
    for run in sorted(plan["statistics"]):
        for frame in plan["statistics"][run]:
            pin = frame["statistics"]
            use = "refit." + run + ".frame." + str(frame["frame_index"]) + ".statistics"
            row, decoded = read_statistics_bound(path_for_use(use, pin), pin)
            if (not isinstance(row, dict) or row.get("contract") != CONTRACT or row.get("run") != run
                    or row.get("frame_index") != frame["frame_index"]
                    or row.get("selected_target_h", frame["selected_target_h"]) not in (None, frame["selected_target_h"])):
                raise ValueError("Statistics row differs from the original mass calibration plan")
            paired, histogram = row["paired_estimates"], row["main_aqueous_histogram"]
            if (any(not isinstance(paired[key], list) for key in _PAIRS)
                    or len({len(paired[key]) for key in _PAIRS}) != 1
                    or any(not isinstance(histogram[key], list) for key in ("native_relative", "capacity_g"))
                    or len(histogram["native_relative"]) != len(histogram["capacity_g"])):
                raise ValueError("Original mass calibration statistics array lengths differ")
            pairs = len(paired[_PAIRS[0]])
            decoded_total += decoded
            pair_total += pairs
            histogram_total += len(histogram["native_relative"])
            largest_decoded, largest_pairs = max(largest_decoded, decoded), max(largest_pairs, pairs)
            inputs.append(dict(use=use, sha256=pin["sha256"], size_bytes=pin["size_bytes"], decoded_bytes=decoded))
    # All original frame decisions/source metadata, both grid audits, optimizer
    # records and one frozen/prediction document. Deliberately no gzip saving.
    plan_bytes = len(json.dumps(plan, ensure_ascii=False, allow_nan=False).encode("utf-8"))
    starts = len(rules.gain_starts) + len(rules.beta_starts)
    products = 8 * (decoded_total + plan_bytes) + 8192 * (grids + starts + len(inputs)) + 32 * 1024**2
    per_row = (products + len(inputs) - 1) // len(inputs)
    products = per_row * len(inputs)
    # Original writer retries may retain serialization attempts. Keep the whole
    # run allowance until retirement; there is no committed-frame progress.
    from .job_queue import TRANSIENT_RETRY_DELAYS
    transient = (len(TRANSIENT_RETRY_DELAYS) + 1) * products
    # Python JSON objects + all retained stage-two crossing/prefix arrays,
    # largest compile_frame temporaries, optimizer/grid bookkeeping and decode.
    workspace = (64 * decoded_total + 64 * largest_decoded + 32 * pair_total
                 + 256 * largest_pairs + 32 * histogram_total
                 + 65536 * (grids + starts + len(inputs)) + 128 * 1024**2)
    return dict(products_estimated_bytes=products, products_per_frame_estimated_bytes=per_row,
                accounting_units="original input rows; no per-frame completion progress",
                transient_estimated_bytes=transient,
                workspace_estimated_bytes=workspace, decoded_statistics_bytes=decoded_total,
                paired_rows=pair_total, histogram_rows=histogram_total, grid_points=grids,
                optimizer_starts=starts, statistics=inputs, compute_control=None,
                variant_f=False, numerical_arithmetic_changed=False, RAM_limit_enforced=False,
                cache_bytes_in_result_estimate=0)


def refit_estimate(stores, series_id, task, plan, *, repository, metadata_rows=None):
    """Original rooted catalogue and Git writer identities own every input/code pin."""
    from .capacity import _pinned_writer_rows
    from .store_runtime import resolve_catalog
    rows = metadata_rows if metadata_rows is not None else resolve_catalog(task, stores, series_id)
    def path_for_use(use, pin):
        matches = [row for row in rows if use in row.uses]
        if (len(matches) != 1 or matches[0].reference.get("sha256") != pin["sha256"]
                or matches[0].reference.get("bytes") != pin["size_bytes"]):
            raise ValueError("One exact rooted original mass calibration input role required: " + use)
        return matches[0].path
    value = budget(task, plan, path_for_use=path_for_use)
    writers = _pinned_writer_rows(repository, task["commit"], PRODUCT_WRITERS)
    return RefitEstimate(task["run"], len(task["ordered_case_ids"]), value["products_estimated_bytes"],
        value["transient_estimated_bytes"], value["products_per_frame_estimated_bytes"],
        value["workspace_estimated_bytes"], value, writers,
        {key: deepcopy(task[key]) for key in ("pipeline", "run", "backend", "precision", "threads", "format",
            "format_version", "storage_profile", "commit", "configuration", "ordered_case_ids")})


def require_estimate(task, estimate):
    """Only this task's source-bound refit estimate can extend a cache route."""
    _require_task_route(task)
    if (not isinstance(estimate, RefitEstimate)
            or estimate.task_binding != {key: task.get(key) for key in estimate.task_binding}
            or {row["relative_path"] for row in estimate.writer_identity} != set(PRODUCT_WRITERS)
            or estimate.case_count != len(task["ordered_case_ids"])
            or type(estimate.products_per_frame_estimated_bytes) is not int
            or estimate.products_per_frame_estimated_bytes < 1
            or estimate.products_estimated_bytes != estimate.case_count * estimate.products_per_frame_estimated_bytes
            or estimate.estimate.get("variant_f") is not False
            or estimate.estimate.get("numerical_arithmetic_changed") is not False
            or estimate.estimate.get("cache_bytes_in_result_estimate") != 0):
        raise ValueError("Exact source-bound original mass calibration estimate required")
    return True


def require_service_sources(repository, producer_paths, runtime_paths):
    """The accepted startup closure must include this actual refit route."""
    key = lambda path: str(PureWindowsPath(path)).casefold()
    required = {key(Path(repository) / relative) for relative in REQUIRED_SERVICE_SOURCES}
    if not required <= set(producer_paths):
        raise ValueError("Original mass calibration source boundary is outside the actual startup closure")
    if not any(path.replace("\\", "/").endswith("scipy/optimize/_lsq/least_squares.py")
               for path in runtime_paths):
        raise ValueError("Original mass calibration optimizer is outside the actual startup runtime closure")
    return True
