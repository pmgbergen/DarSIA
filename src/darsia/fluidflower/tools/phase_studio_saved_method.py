"""Short, provenance-aware descriptions of immutable concentration results.

This is presentation only. It never infers an optical correction or a gas-depth
rule from a model name, image appearance, or a mass-fit score.
"""

from collections.abc import Mapping
import math

from darsia.fluidflower.photometry_consistency import (
    evaluate_photometry_stage_contracts,
)


def _mapping(value):
    return value if isinstance(value, Mapping) else {}


def explain_ffum_scenario(result: Mapping) -> dict[str, str] | None:
    """Explain recorded FF_UM sensitivity scenarios without claiming hidden 3-D gas.

    Only explicitly published experiment families are decoded. Unknown methods
    keep their recorded names rather than acquiring an invented interpretation.
    """
    settings = _mapping(_mapping(result.get("method")).get("settings"))
    calibration = _mapping(result.get("calibration"))
    family = settings.get("experiment_family")
    variant = settings.get("experiment_variant_id")
    if family == "gas_history_sensitivity_v1":
        if settings.get("gas_mode") == "fixed":
            parameters = _mapping(calibration.get("shared_parameters"))
            if not parameters:
                parameters = _mapping(calibration.get("parameters"))
            saturation = parameters.get("gas_saturation")
            declared = {f"fixed_sg_{round(100 * value):03d}": value
                        for value in (.5, .6, .7, .75, .8, .9, 1.)}.get(variant)
            if saturation is None:
                saturation = declared
            if (isinstance(saturation, bool) or not isinstance(saturation, (int, float))
                    or not math.isfinite(saturation) or not 0 <= saturation <= 1
                    or (declared is not None and not math.isclose(saturation, declared, abs_tol=1e-6))):
                return None
            percent = 100 * saturation
            return {
                "short_label": f"Visible gas: fixed {percent:g}% at every time",
                "detail": ("Every gas-labelled image column uses this fraction of its assumed full-depth "
                           "pore volume from its first appearance. No repeated sighting is required; "
                           "no unseen neighbouring gas or 3-D connection is added."),
            }
        if variant in {"history_tau0p5", "history_tau2p0", "history_tau6p0"}:
            tau = {"history_tau0p5": "0.5", "history_tau2p0": "2", "history_tau6p0": "6"}[variant]
            return {
                "short_label": f"Visible gas: 60–90% with history ({tau} h)",
                "detail": ("Repeated visible gas during injection raises assumed saturation within the "
                           "existing gas mask on the stated time scale. This does not create invisible "
                           "gas or connect separate regions in 3-D."),
            }
        if variant in {"free_sg", "free_sg_no_dilution"}:
            return {
                "short_label": ("Visible gas: fitted constant saturation"
                                + ("; no dilution" if variant == "free_sg_no_dilution" else "")),
                "detail": ("One saturation is fitted across all visible gas columns and times. "
                           "It does not depend on repeated sightings or infer hidden gas."),
            }
    if family == "local_optical_dilution_v1":
        dilution = settings.get("dilution_mode")
        gas = settings.get("gas_mode")
        if dilution not in {"global", "local", "none"}:
            return None
        dilution_label = {"global": "Global dilution", "local": "Local optical dilution",
                          "none": "No dilution"}[dilution]
        gas_labels = {
            "fixed75": "visible gas fixed at 75%",
            "existing_history": "visible gas 60–90% with injection-gated history",
            "persistent_core": "persistent visible-gas core 60–90%",
            "persistent_full_depth": "visible gas 75–100% after repeated sightings",
        }
        details = {
            "fixed75": "Visible gas is assigned 75% saturation from its first appearance.",
            "existing_history": "Repeated visible gas can raise saturation during injection.",
            "persistent_core": ("A stable interior of the visible gas plume can gain saturation "
                                "over time, including after injection stops."),
            "persistent_full_depth": ("A visible column reaches full-depth saturation only after three "
                                      "sightings spanning at least 0.5 h."),
        }
        if gas not in gas_labels:
            return None
        return {
            "short_label": f"{dilution_label}; {gas_labels[gas]}",
            "detail": details[gas] + " Existing phase masks remain fixed; no unseen 3-D corridor is added.",
        }
    return None


def summarize_saved_method(result: Mapping) -> dict[str, str]:
    """Return the actual recorded method choices, marking absent evidence unknown."""
    method = _mapping(result.get("method"))
    settings = _mapping(method.get("settings"))
    calibration = _mapping(result.get("calibration"))
    candidate = _mapping(_mapping(result.get("selected_inputs")).get("candidate"))

    photometry_id = settings.get("photometry_variant_id")
    illumination = settings.get("illumination")
    checker = settings.get("colorchecker_enabled")
    if photometry_id:
        optical = str(photometry_id)
    elif illumination is not None or checker is not None:
        optical = f"illumination={illumination if illumination is not None else 'unknown'}, "
        optical += f"ColorChecker={checker if checker is not None else 'unknown'}"
    elif settings.get("optical") == "native_color_path":
        optical = "native color path on selected phase-model corrected images"
    else:
        optical = "not recorded"

    gas_mode = settings.get("gas_mode") or calibration.get("gas_mode")
    if gas_mode:
        gas = str(gas_mode)
    else:
        parameters = _mapping(calibration.get("parameters"))
        saturation = parameters.get("gas_saturation")
        if isinstance(saturation, (int, float)) and math.isfinite(saturation):
            gas = f"recorded gas saturation {saturation:.3f}"
        elif settings.get("optical") == "native_color_path":
            gas = "native registered gas response"
        else:
            gas = "not recorded"
    depth_rule = settings.get("hidden_gas_depth_mode") or settings.get("gas_depth_mode")
    if depth_rule:
        gas += f"; depth rule {depth_rule}"
    else:
        gas += "; persistent full-depth rule not documented"

    scope = str(calibration.get("scope") or "not recorded")
    overflow_mode = str(result.get("overflow_mode") or calibration.get("overflow_mode") or "not recorded")
    overflow = {
        "off": "off",
        "compare": "comparison only; excluded from fit",
        "constrain": "fit constraint at measured times",
        "hard": "hard feasible interval at training times",
    }.get(overflow_mode, overflow_mode)

    contract = result.get("photometry_stage_contracts")
    stages = evaluate_photometry_stage_contracts(contract)
    if stages["stage_contract_format"] == "recognized":
        correction_consistency = (
            f"phase {stages['phase_chain']['status']}; "
            f"concentration {stages['concentration_chain']['status']}"
        )
    else:
        correction_consistency = "unverified: no complete stage fingerprints"

    identity = str(candidate.get("candidate_id") or candidate.get("run") or "not recorded")
    return {
        "optical": optical,
        "gas": gas,
        "scope": scope,
        "overflow": overflow,
        "correction_consistency": correction_consistency,
        "phase_identity": identity,
    }


def format_saved_method_summary(result: Mapping) -> str:
    """Compact text suitable for the Studio saved-result details panel."""
    item = summarize_saved_method(result)
    scenario = explain_ffum_scenario(result)
    scenario_text = (f"\nScenario: {scenario['short_label']}\nMeaning: {scenario['detail']}"
                     if scenario else "")
    return (f"Optical correction: {item['optical']} · consistency: {item['correction_consistency']}\n"
            f"Gas: {item['gas']}\n"
            f"Calibration: {item['scope']} · overflow: {item['overflow']}\n"
            f"Phase source: {item['phase_identity']}{scenario_text}")
