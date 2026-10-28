"""Pure provenance contract for phase and concentration photometry.

Manual phase labels stay in registered physical coordinates when photometry is
changed, but the learned phase response may not. We therefore compare the two
*internal* optical chains independently:

* phase-model training against phase-map inference;
* concentration clearwater/reference against fit and inference.

Their chains may intentionally differ in a fixed-phase concentration ablation.
That cross-chain difference is reported, not treated as a mismatch. The result
does not certify scientific validity, only consistency of supplied immutable
provenance. Absent or malformed hashes are *unverified*, never equal by default.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import re
from typing import Any


VERSION = "darsia-fluidflower-photometry-consistency-v1"
STAGE_CONTRACT_FORMAT = "darsia-fluidflower-photometry-stages-v1"
STAGES = (
    "phase_training", "phase_inference", "concentration_reference",
    "concentration_fit", "concentration_inference",
)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")


def _digest(value: Any) -> str | None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value.strip()):
        return None
    return value.strip().lower()


def fingerprint_photometry_chain(
    *,
    correction_revision_sha256: str,
    correction_dependency_sha256: str,
    correction_algorithm: str,
    color_path_sha256: str,
    output_encoding: str,
    clipping_policy: str,
) -> str:
    """Hash a *complete* immutable correction policy, excluding image bytes.

    The dependency digest must bind calibration/reference artifacts and their
    execution order. A caller unable to supply any field must leave its stage
    chain unverified rather than hash an incomplete contract. The registered
    pixel-to-physical grid is checked separately from photometry.
    """
    correction = _digest(correction_revision_sha256)
    dependency = _digest(correction_dependency_sha256)
    color_path = _digest(color_path_sha256)
    if any(item is None for item in (correction, dependency, color_path)):
        raise ValueError("Complete SHA-256 correction, dependency and color-path identities required")
    text_fields = (correction_algorithm, output_encoding, clipping_policy)
    if any(not isinstance(item, str) or not item.strip() for item in text_fields):
        raise ValueError("Correction algorithm, output encoding and clipping policy required")
    payload = {
        "format": "photometry-chain-fingerprint-v1",
        "correction_revision_sha256": correction,
        "correction_dependency_sha256": dependency,
        "correction_algorithm": correction_algorithm.strip(),
        "color_path_sha256": color_path,
        "output_encoding": output_encoding.strip(),
        "clipping_policy": clipping_policy.strip(),
    }
    content = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _compare(
    stages: tuple[str, ...],
    records: Mapping[str, Mapping[str, Any] | None],
    key: str,
) -> dict[str, Any]:
    values = {stage: _digest((records[stage] or {}).get(key)) for stage in stages}
    missing = [stage for stage, digest in values.items() if digest is None]
    present = {digest for digest in values.values() if digest is not None}
    if len(present) > 1:
        status = "mismatch"
    elif missing:
        status = "unverified"
    else:
        status = "matched"
    return {
        "status": status,
        "stages": list(stages),
        "missing_stages": missing,
        "sha256_by_stage": values,
    }


def _decision(*, dependencies: tuple[tuple[str, dict[str, Any]], ...],
              fixed_map: str | None = None) -> dict[str, Any]:
    reasons = []
    states = []
    for name, check in dependencies:
        states.append(check["status"])
        if check["status"] != "matched":
            reasons.append(f"{name}:{check['status']}")
    if fixed_map is not None and _digest(fixed_map) is None:
        states.append("unverified")
        reasons.append("fixed_phase_map_sha256:unverified")
    status = ("mismatch" if "mismatch" in states else
              "unverified" if "unverified" in states else "ready")
    return {"status": status, "reasons": reasons}


def evaluate_photometry_consistency(
    *,
    phase_training: Mapping[str, Any] | None = None,
    phase_inference: Mapping[str, Any] | None = None,
    concentration_reference: Mapping[str, Any] | None = None,
    concentration_fit: Mapping[str, Any] | None = None,
    concentration_inference: Mapping[str, Any] | None = None,
    fixed_phase_map_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare five declared stage identities without reading or changing data.

    Each stage mapping needs `chain_sha256` for correction algorithm, frozen
    reference/calibration dependencies, color path, output encoding and clip
    policy; and `registered_grid_sha256` for the pixel-to-physical transform.
    These are *stage identity* hashes, not the per-image source SHA. Missing
    provenance is unverified. A different phase and concentration correction
    is allowed when each internal chain and the registered grid are consistent.

    `fixed_phase_ablation` requires a fixed phase-map digest, a matched
    concentration chain, and a matching phase-inference/concentration grid.
    It does not require phase training/inference chain equality, because a
    fixed phase map can be used to screen concentration methods without
    validating its phase model. `end_to_end_photometry` requires both chains
    and all registered grids to match. Neither status validates local physics,
    concentration fit, mass balance or phase-map correctness.
    """
    records = {
        "phase_training": phase_training,
        "phase_inference": phase_inference,
        "concentration_reference": concentration_reference,
        "concentration_fit": concentration_fit,
        "concentration_inference": concentration_inference,
    }
    for stage, value in records.items():
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{stage} must be a mapping or None")
    phase = _compare(STAGES[:2], records, "chain_sha256")
    concentration = _compare(STAGES[2:], records, "chain_sha256")
    cross = _compare(("phase_inference", "concentration_inference"),
                     records, "chain_sha256")
    cross["status"] = {"matched": "same", "mismatch": "different"}.get(
        cross["status"], "unverified")
    cross["informational_only"] = True
    phase_grid = _compare(STAGES[:2], records, "registered_grid_sha256")
    concentration_grid = _compare(STAGES[2:], records, "registered_grid_sha256")
    cross_grid = _compare(("phase_inference", "concentration_inference"),
                          records, "registered_grid_sha256")
    all_grid = _compare(STAGES, records, "registered_grid_sha256")
    fixed_ablation = _decision(
        dependencies=(("concentration_chain", concentration),
                      ("concentration_grid", concentration_grid),
                      ("phase_to_concentration_grid", cross_grid)),
        fixed_map=fixed_phase_map_sha256 if fixed_phase_map_sha256 is not None else "",
    )
    end_to_end = _decision(
        dependencies=(("phase_chain", phase), ("concentration_chain", concentration),
                      ("registered_grid", all_grid)),
    )
    return {
        "format": VERSION,
        "phase_chain": phase,
        "concentration_chain": concentration,
        "cross_chain": cross,
        "registered_grid": {
            "phase": phase_grid,
            "concentration": concentration_grid,
            "cross_chain": cross_grid,
            "all_stages": all_grid,
        },
        "fixed_phase_ablation": fixed_ablation,
        "end_to_end_photometry": end_to_end,
        "scope_note": (
            "Chain identity and grid provenance only; phase accuracy, optical "
            "calibration response and mass physics require separate validation."
        ),
    }


def evaluate_photometry_stage_contracts(
    envelope: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate a saved result/plan stage envelope, failing closed on legacy data.

    Place the envelope under `photometry_stage_contracts` in a concentration
    result or production task contract. An absent or unknown schema is not
    interpreted by guessing equivalence from old correction-contract strings.
    """
    if not isinstance(envelope, Mapping) or envelope.get("format") != STAGE_CONTRACT_FORMAT:
        result = evaluate_photometry_consistency()
        result["stage_contract_format"] = "unverified"
        return result
    try:
        result = evaluate_photometry_consistency(
            **{stage: envelope.get(stage) for stage in STAGES},
            fixed_phase_map_sha256=envelope.get("fixed_phase_map_sha256"),
        )
    except ValueError:
        result = evaluate_photometry_consistency()
        result["stage_contract_format"] = "unverified"
        result["stage_contract_error"] = "invalid stage mapping"
        return result
    result["stage_contract_format"] = "recognized"
    return result
