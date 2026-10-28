"""The optical map used by ordinary Studio review and its disagreement checks.

Archived supported maps remain immutable. Context completion is not part of the
current review basis, and missing optical assets must never silently fall back
to those supported maps.
"""
from collections.abc import Mapping
from pathlib import Path

VERSION = "studio-optical-without-context-v1"
REGIONAL_VERSION = "baseline-relative-optical-regions-v1"
SAVED_NATIVE_VERSION = "saved-native-phase-labels-v1"


def value(candidate, key, default=None):
    return candidate.get(key, default) if isinstance(candidate, Mapping) else getattr(candidate, key, default)


def uses_optical(candidate):
    # Original jobs encode phase-only status in their summary. Collection jobs
    # still carry the unambiguous raw-map binding even without that flag.
    return bool(value(candidate, "phase_model_only", False) or (
        value(candidate, "raw_phase_map_output_dir") and value(candidate, "optical_model_sha256")))


def is_regional(candidate):
    return value(candidate, "phase_support_format") == REGIONAL_VERSION


def reference_layer(candidate, phase):
    return "raw" if uses_optical(candidate) and not is_regional(candidate) and phase in {"all", "gas", "aqueous"} else phase


def optical_path(candidate, frame_index):
    if is_regional(candidate):
        root = value(candidate, "phase_map_output_dir")
        identity = value(candidate, "params_sha256")
        if not root or not identity:
            raise ValueError("Prepare this trained regional model's phase maps before review")
        return Path(root) / "native_phase_maps" / value(candidate, "run") / (
            "A_" + identity[:16]) / f"frame_{frame_index:03d}.png"
    root = value(candidate, "raw_phase_map_output_dir")
    identity = value(candidate, "optical_model_sha256")
    if not root or not identity:
        raise ValueError("Prepare this model's raw optical maps in Phase Model Studio before review")
    return Path(root) / "native_phase_maps" / value(candidate, "run") / (
        "A_" + identity[:16]) / f"frame_{frame_index:03d}.png"
