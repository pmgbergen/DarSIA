"""Presentation helpers for comparing saved Studio concentration results.

These helpers do not select calibration inputs or change scientific results.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


def _candidate(result):
    return result.get("selected_inputs", {}).get("candidate", {})


def _calibration_label(result):
    fit = result.get("calibration", {})
    scope = str(fit.get("scope", "")).lower()
    if scope == "registered":
        return "Native imported calibration (provisional)"
    if fit.get("applied_calibration"):
        return "Frozen native " + ("shared model" if scope in {"series", "shared"} else "individual model")
    roles = {str(row.get("calibration_role", "")).lower()
             for row in result.get("frames", [])}
    if scope in {"series", "shared"}:
        return "Frozen series model" if fit.get("fit_performed") is False else "Shared series fit"
    if scope == "individual":
        return "Individual fit"
    if any("frozen series" in role for role in roles):
        return "Frozen series model"
    # The original core.fit_series results predate the explicit scope field.
    if (fit.get("format") == "shared-series-mass-fit-v1"
            or any("shared series" in role for role in roles)):
        return "Shared series fit"
    if (any("individual" in role for role in roles)
            or result.get("format") == "studio-concentration-mass-v2"):
        return "Individual fit"
    return "Calibration scope unknown"


def result_variant_label(value, path):
    """Describe a saved model, with a stable suffix for otherwise equal labels."""
    fit = value.get("calibration", {})
    settings = value.get("method", {}).get("settings", {})
    chemistry = {"empirical_power": "power response", "darsia_simple_flash": "SimpleFlash",
                 "darsia_titration_flash": "TitrationFlash (fork)"}.get(
        settings.get("chemistry"), settings.get("chemistry", "response unspecified"))
    restoration = {"darsia_tvd": "TVD", "none": "no restoration"}.get(
        settings.get("restoration"), settings.get("restoration", "restoration unspecified"))
    if settings.get("optical") == "native_color_path":
        chemistry = "Native DarSIA color path + " + str(chemistry)
    if settings.get("response") == "native_preflash_monotone":
        chemistry += " · calibrated color gradient"
    if settings.get("response") == "joint_reference_dilution":
        chemistry = "Joint optical references + dilution"
        restoration = "Fixed reference smoothing"
    has_references = bool(fit.get("overflow_provenance")) or any(
        row.get("gas_reference") is not None
        or row.get("gas_constraint", {}).get("status") == "measured overflow constraint"
        for row in value.get("frames", []))
    mode = value.get("overflow_mode", fit.get("overflow_mode"))
    overflow = ("overflow constrained" if mode == "constrain" else "overflow comparison") if has_references else "no overflow targets"
    candidate = _candidate(value)
    phase = str(candidate.get("candidate_id") or "unspecified")
    if len(phase) > 54:
        phase = phase[:32] + "…" + phase[-20:]
    created = str(value.get("created_utc") or "undated").replace("T", " ")[:19]
    # Path identity distinguishes imported fits with equal settings/timestamps;
    # it is display metadata and never controls numerical cache selection.
    tag = hashlib.sha256(str(Path(path)).encode("utf-8")).hexdigest()[:8]
    photometry = []
    if settings.get("photometry_variant_id"):
        illumination = {
            "none": "No illumination", "historical": "Historical illumination",
            "facies": "Facies illumination",
        }.get(settings.get("illumination"), "Illumination unspecified")
        checker = settings.get("colorchecker_enabled")
        photometry = [illumination, "ColorChecker " + (
            "on" if checker is True else "off" if checker is False else "unspecified")]
    experiment = []
    if settings.get("experiment_family") == "gas_history_sensitivity_v1":
        experiment = ["Experimental: " + str(settings.get("experiment_label") or
                                              settings.get("experiment_variant_id", "gas history"))]
        if settings.get("beta_fixed") == 0:
            chemistry = "Joint optical references; dilution disabled"
    if settings.get("experiment_family") == "local_optical_dilution_v1":
        dilution_mode = settings.get("dilution_mode")
        gas_label = {"fixed75": "fixed 75% gas saturation",
                     "existing_history": "existing gas history",
                     "persistent_core": "persistent gas core",
                     "persistent_full_depth": "causal persistent visible gas at full depth"}.get(
                         settings.get("gas_mode"), "gas policy unspecified")
        experiment = ["Experimental: " + str(settings.get("experiment_label") or
            settings.get("experiment_variant_id") or f"{dilution_mode or 'Unspecified'} dilution + {gas_label}")]
        chemistry = {"local": "Joint optical references + local dilution",
                     "global": "Joint optical references + dilution",
                     "none": "Joint optical references; dilution disabled"}.get(
                         dilution_mode, "Joint optical references; dilution unspecified")
        restoration = "Fixed reference smoothing"
    return " · ".join((_calibration_label(value), *experiment, *photometry, str(chemistry),
                       str(restoration), overflow, "phase " + phase, created + " / " + tag))


def matching_frame_index(old_result, old_index, new_result):
    """Find the same photograph and time in the same experiment, or return None.

    A frame's position in the result list is never its identity. Source names
    take precedence over review indices; elapsed time must always agree.
    """
    old_candidate, new_candidate = _candidate(old_result), _candidate(new_result)
    for key in ("dataset", "run"):
        old_key, new_key = old_candidate.get(key), new_candidate.get(key)
        if not old_key or not new_key or str(old_key).casefold() != str(new_key).casefold():
            return None
    if not isinstance(old_index, int) or not 0 <= old_index < len(old_result.get("frames", [])):
        return None
    old = old_result["frames"][old_index]
    try:
        old_time = float(old["elapsed_h"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(old_time):
        return None
    matches = []
    for index, new in enumerate(new_result.get("frames", [])):
        try:
            new_time = float(new["elapsed_h"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(new_time) or not math.isclose(old_time, new_time, rel_tol=0., abs_tol=1e-7):
            continue
        old_name, new_name = old.get("source_name"), new.get("source_name")
        old_frame, new_frame = old.get("frame_index"), new.get("frame_index")
        if old_name and new_name:
            if old_name == new_name:
                matches.append((0, index))
        elif old_frame is not None and new_frame is not None:
            if old_frame == new_frame:
                matches.append((1, index))
        elif not old_name and not new_name and old_frame is None and new_frame is None:
            matches.append((2, index))
    return min(matches)[1] if matches else None


def ac_compatibility_note(dataset, run):
    """Show documented AC protocol differences without changing eligibility."""
    if str(dataset).casefold().replace("_", "") != "ffac":
        return ""
    # Recipe notes follow build_ac_chemistry_overview.CHEMISTRY_OVERRIDES.
    # AC29 additionally has an uncertain actual injected dose; the nominal
    # protocol curve must not be presented as independent mass validation.
    notes = {
        "ac20": "Phosphate-buffered recipe with NaOH titration to pH 8.00; chemistry differs from the standard NaOH/BTB recipe.",
        "ac21": "Documented acid carryover from the previous experiment, with a slightly lower initial pH.",
        "ac22": "Documented acid carryover: some acid remained at the bottom.",
        "ac24": "Documented BTB dose is about half the dominant series recipe.",
        "ac27": "Documented BTB dose is about half the dominant series recipe.",
        "ac29": "Actual injected dose is uncertain; the nominal injected-mass curve cannot independently validate detected mass. Chemical and initial-state differences are also documented.",
        "ac32": "Ambiguous recipe: the protocol appears to interchange BTB and NaOH quantities, so the literal quantities cannot establish chemistry compatibility.",
        "ac44": "Reduced BTB dose: about 58% of the dominant series recipe.",
        "ac51": "Documented low-pH green/yellow initial state and incomplete preparation metadata.",
    }
    return notes.get(str(run).casefold().replace(" ", ""), "")
