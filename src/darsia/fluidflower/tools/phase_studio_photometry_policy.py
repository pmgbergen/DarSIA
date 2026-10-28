"""Series-owned eligibility policy for new concentration photometry fits.

The NAS policy records the scientific decision. The code-level FF_UM limit is
deliberately stricter than a configurable default: a missing or edited policy
can never re-enable the four rejected illumination branches.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tomllib

from darsia.fluidflower.series_paths import series_path


POLICY_FORMAT = "studio-concentration-photometry-policy-v1"
ALL_VARIANTS = (
    "none_cc_off", "none_cc_on",
    "historical_cc_off", "historical_cc_on",
    "facies_cc_off", "facies_cc_on",
)
FFUM_ALLOWED_VARIANTS = ("none_cc_off", "none_cc_on")
FFUM_EXCLUDED_VARIANTS = tuple(
    key for key in ALL_VARIANTS if key not in FFUM_ALLOWED_VARIANTS
)


def _policy_path(dataset: str, results_root: str | Path | None) -> Path:
    if results_root is None:
        return series_path(dataset, "config") / "photometry_policy.toml"
    # Studio concentration results live below DarSIA/results/concentration_mass.
    return Path(results_root).resolve().parent.parent / "config" / "photometry_policy.toml"


def load_policy(dataset: str, results_root: str | Path | None = None) -> dict | None:
    """Read a small series policy; retain a safe FF_UM fallback if absent."""
    if str(dataset).casefold() != "ffum":
        return None
    path = _policy_path("ffum", results_root)
    if not path.is_file():
        return dict(format=POLICY_FORMAT, dataset="ffum", source="code_fallback",
                    path=str(path), sha256=None,
                    allowed_variants=list(FFUM_ALLOWED_VARIANTS),
                    excluded_variants=list(FFUM_EXCLUDED_VARIANTS))
    data = path.read_bytes()
    try:
        document = tomllib.loads(data.decode("utf-8-sig"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("Invalid FF_UM photometry policy TOML") from error
    allowed = document.get("allowed_variants")
    excluded = document.get("excluded_variants")
    if (document.get("format") != POLICY_FORMAT
            or str(document.get("dataset", "")).casefold() != "ffum"
            or not isinstance(allowed, list) or not allowed
            or not isinstance(excluded, list)
            or any(not isinstance(value, str) for value in [*allowed, *excluded])
            or len(set(allowed)) != len(allowed)
            or len(set(excluded)) != len(excluded)
            or not set(allowed) <= set(FFUM_ALLOWED_VARIANTS)
            or not set(FFUM_EXCLUDED_VARIANTS) <= set(excluded)
            or set(allowed) & set(excluded)
            or set(allowed) | set(excluded) != set(ALL_VARIANTS)):
        raise ValueError("FF_UM photometry policy must exclude historical/facies branches")
    return dict(format=POLICY_FORMAT, dataset="ffum", source="nas_policy",
                path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                allowed_variants=list(allowed), excluded_variants=list(excluded))


def verify_policy_snapshot(snapshot: dict | None, dataset: str,
                           results_root: str | Path) -> None:
    """Reject a changed policy before a queued comparison creates results."""
    if snapshot is None:
        return  # Legacy CLI requests are still constrained by _selected_variants.
    if str(dataset).casefold() != "ffum" or not isinstance(snapshot, dict):
        raise ValueError("Photometry policy snapshot belongs to another series")
    current = load_policy(dataset, results_root)
    if any(snapshot.get(key) != current.get(key) for key in (
            "format", "dataset", "source", "sha256", "allowed_variants",
            "excluded_variants")):
        raise ValueError("FF_UM photometry policy changed after calibration was queued")
    # A mapped drive and UNC path can name the same NAS file. Its content hash
    # and file identity matter; the spelling of the mount does not.
    if current["source"] == "nas_policy":
        try:
            same_file = Path(snapshot["path"]).samefile(current["path"])
        except (OSError, KeyError, TypeError):
            same_file = False
        if not same_file:
            raise ValueError("FF_UM photometry policy changed after calibration was queued")


def variant_status(dataset: str, settings: dict | None,
                   policy: dict | None = None) -> str:
    """Classify saved results without inventing metadata for legacy methods."""
    if str(dataset).casefold() != "ffum" or not isinstance(settings, dict):
        return "unknown"
    key = settings.get("photometry_variant_id")
    light = settings.get("illumination")
    if not isinstance(key, str):
        key = None
    if not isinstance(light, str):
        light = None
    if light in {"historical", "facies"} or key in FFUM_EXCLUDED_VARIANTS:
        return "excluded"
    allowed = set((policy or {}).get("allowed_variants", FFUM_ALLOWED_VARIANTS))
    if key in allowed and light in {None, "none"}:
        return "allowed"
    if key in ALL_VARIANTS and key not in allowed:
        return "excluded"
    return "unknown"
