"""Whole-operation output ceilings for an explicitly typed Setup job.

The RAW limits are the original request's limits, enforced by OwnedRawOutput.
The separate metadata limits must be enforced by the operation's existing
bounded sink. This module reads and verifies inputs only. It does not reserve
space, create an output, qualify a cache profile or authorize a writer.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass


SETUP_FORMAT = "darsia-series-setup-operation-v1"
CAP_MODEL = (
    "ENFORCED OUTPUT CAPS: original RAW request whole/per-file ceilings plus "
    "the explicit bounded metadata/publication allowance; not a measurement "
    "or a common/phase capacity qualification. Keep the whole reservation "
    "until actual operation retirement; frame progress releases no bytes. "
    "Cache/scratch and distinct-volume admission are separate."
)


def _nonnegative(value, where, *, positive=False):
    """Keep literal byte authorities without coercion or estimated defaults."""
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(where + " must be an explicit positive integer" if positive
                         else where + " must be an explicit nonnegative integer")
    return value


def _limits(value, where):
    """Read the original writer's positive whole/per-file ceiling relation."""
    if not isinstance(value, dict):
        raise ValueError(where + " requires explicit output/file byte ceilings")
    whole = _nonnegative(value.get("maximum_output_bytes"), where + " whole bytes", positive=True)
    per_file = _nonnegative(value.get("maximum_file_bytes"), where + " file bytes", positive=True)
    if per_file > whole:
        raise ValueError(where + " file ceiling cannot exceed its whole ceiling")
    return whole, per_file


@dataclass(frozen=True)
class CappedSetupOutput:
    """Declared writer ceilings, with the existing capacity-ledger interface.

    ``products_*_estimated_bytes`` are compatibility properties for that
    interface, not statistical estimates. A ledger integration must recognize
    this positive type and must not subtract committed-frame progress from it.
    No temporary/cache entitlement is inferred from ``transient_estimated_bytes``.
    """
    run: str
    operation: str
    raw_maximum_output_bytes: int
    raw_maximum_file_bytes: int
    metadata_maximum_output_bytes: int
    metadata_maximum_file_bytes: int
    raw_build_request: dict | None
    raw_output_authority: dict | None
    raw_request_inputs: tuple[dict, ...]
    metadata_output_authority: dict
    case_count: int = 1
    transient_estimated_bytes: int = 0
    frame_progress_release_allowed: bool = False
    model: str = CAP_MODEL

    @property
    def maximum_output_bytes(self):
        """The sum of disjoint RAW and metadata/publication output ceilings."""
        return self.raw_maximum_output_bytes + self.metadata_maximum_output_bytes

    @property
    def products_per_frame_estimated_bytes(self):
        """The existing ledger slot holds one entire operation's cap."""
        return self.maximum_output_bytes

    @property
    def products_estimated_bytes(self):
        """Retain the entire declared cap until the owner actually retires."""
        return self.maximum_output_bytes

    def needed(self, *, other_remaining_bytes=0, reserve_bytes=None):
        """Use the existing reserve and explicit concurrent open reservations."""
        if reserve_bytes is None:
            from .capacity import DEFAULT_RESERVE_BYTES
            reserve_bytes = DEFAULT_RESERVE_BYTES
        return (self.maximum_output_bytes
                + _nonnegative(other_remaining_bytes, "Other active remaining bytes")
                + _nonnegative(reserve_bytes, "Reserve bytes"))


def capped_setup_output(task, config):
    """Read exact current Setup output authorities without starting any writer.

    A Build's full RAW request is captured only after Apply's real postimages
    exist. Capacity reads only this exact request's metadata; source payloads
    are verified after the worker owns a claim. Output namespace ownership and
    actual per-write cap enforcement remain with the original producer.
    """
    if (not isinstance(task, dict) or task.get("pipeline") != "series-setup"
            or not isinstance(config, dict) or config.get("format") != SETUP_FORMAT):
        raise ValueError("Only the explicit series-setup operation uses capped Setup output")
    run = config.get("run")
    if not isinstance(run, str) or not run or task.get("run") != run:
        raise ValueError("Setup capacity must name its exact task run")
    operation = config.get("operation")
    if operation not in {"apply", "build", "image-preparation"}:
        raise ValueError("Setup capacity requires the explicit apply or build operation")
    cases = task.get("ordered_case_ids")
    if cases is not None and (not isinstance(cases, list) or len(cases) != 1):
        raise ValueError("One whole Setup operation must retain one reservation")
    metadata = config.get("metadata_output")
    if not isinstance(metadata, dict) or set(metadata) != {"maximum_output_bytes", "maximum_file_bytes"}:
        raise ValueError("Setup metadata output needs its exact whole/per-file ceilings")
    metadata_whole, metadata_file = _limits(metadata, "Setup metadata")
    request_binding, raw_output, pins = None, None, ()
    raw_whole = raw_file = 0
    if operation == "build":
        from ..rig_raw_build import read_build_request_metadata
        request_binding = config.get("raw_build_request")
        request, pins = read_build_request_metadata(request_binding, artifact_kind="raw_rig_baseline")
        if request["run_id"] != run:
            raise ValueError("The full RAW request names another Setup run")
        raw_output = request["output"]
        if not isinstance(raw_output, dict) or set(raw_output) != {
                "destination", "intended_root", "forbidden_raw_roots",
                "maximum_output_bytes", "maximum_file_bytes"}:
            raise ValueError("Setup must retain the exact original RAW output authority")
        raw_whole, raw_file = _limits(raw_output, "RAW output")
    elif config.get("raw_build_request") is not None:
        raise ValueError("Apply cannot claim or consume a future RAW Build request")
    return CappedSetupOutput(
        run=run, operation=operation,
        raw_maximum_output_bytes=raw_whole, raw_maximum_file_bytes=raw_file,
        metadata_maximum_output_bytes=metadata_whole, metadata_maximum_file_bytes=metadata_file,
        raw_build_request=deepcopy(request_binding), raw_output_authority=deepcopy(raw_output),
        raw_request_inputs=tuple(deepcopy(pins)), metadata_output_authority=deepcopy(metadata))
