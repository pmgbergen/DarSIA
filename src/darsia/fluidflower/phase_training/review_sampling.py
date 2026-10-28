"""Source-bound V5 review sampling policy.

Older FF_AC campaigns saved model-guided ``training_sampling`` coordinates.
Those coordinates describe errors of a different model and must not select
pixels for a newly corrected RAW candidate.  The original review payloads
remain intact; this module only determines their role in a new fit.
"""

from __future__ import annotations
from ..phase_saved_review_metadata import configured_parameters

def _parameters():
    return configured_parameters("training_review_sampling")

def __getattr__(name):
    keys={'POLICY': 'policy'}
    if name not in keys: raise AttributeError(name)
    value=_parameters()[keys[name]]
    return frozenset(value) if name=="CHEMISTRY_EVAL_RUNS" else value

from collections import Counter
from collections.abc import Mapping

from .logistic import (
    CANDIDATE_ONLY_UNSUPPORTED_PHASE_ISSUES,
    PHASE_AQUEOUS,
    PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT,
    PHASE_GAS,
    PHASE_IGNORED,
    PHASE_WATER,
    REGIONAL_EVIDENCE_PHASES,
    MIXED_PHASES,
)


WHOLE_REGION_PIXEL_CLASSES = frozenset(
    (PHASE_WATER, PHASE_AQUEOUS, PHASE_GAS,
     PHASE_CURRENT_EXPERIMENT_CO2_ARTIFACT)
)
REGIONAL_MIL_CLASSES = MIXED_PHASES | REGIONAL_EVIDENCE_PHASES


def supervision_role(phase_class: str, payload: Mapping[str, object]) -> str:
    """Return the training role without consulting an older model's maps.

    Regional MIL regions may supply uniformly selected *candidate* pixels,
    but they must never turn every pixel in the box into a hard phase label.
    Candidate-specific error reports are retained for evaluation only.
    """

    category = str(payload.get("category") or "").strip().lower()
    if category in CANDIDATE_ONLY_UNSUPPORTED_PHASE_ISSUES:
        if phase_class != PHASE_IGNORED:
            raise ValueError("Candidate-specific model issue became a phase label")
        return "candidate_issue_evaluation_only"
    if phase_class == PHASE_IGNORED:
        return "evaluation_only"
    if phase_class in WHOLE_REGION_PIXEL_CLASSES:
        return "whole_region_uniform"
    if phase_class in REGIONAL_MIL_CLASSES:
        return "regional_mil_uniform_candidates"
    raise ValueError(f"Unknown V5 visible-phase supervision class {phase_class!r}")


def sampling_receipt(phase_class: str, payload: Mapping[str, object]) -> dict[str, object]:
    """Small immutable metadata for a source/annotation-bound sample batch."""

    return {
        "contract": _parameters()["policy"],
        "role": supervision_role(phase_class, payload),
        "legacy_model_guided_training_sampling_ignored": "training_sampling" in payload,
        "source_annotation_id": str(payload.get("id") or payload.get("proposal_id") or ""),
    }


def inventory(regions: object) -> dict[str, object]:
    """Count every retained review ID, including regional/evaluation evidence."""

    rows = [(str(region.region_id), sampling_receipt(
        str(region.phase_class), region.payload)) for region in regions]
    if len({identifier for identifier, _ in rows}) != len(rows):
        raise ValueError("V5 review-sampling inventory has duplicate annotation IDs")
    return {
        "sampling_policy": _parameters()["policy"],
        "sampling_regions": len(rows),
        "sampling_roles": dict(sorted(Counter(
            row["role"] for _, row in rows).items())),
        "legacy_model_guided_sampling_ignored_count": sum(
            bool(row["legacy_model_guided_training_sampling_ignored"])
            for _, row in rows),
        "candidate_issue_evaluation_only_count": sum(
            row["role"] == "candidate_issue_evaluation_only"
            for _, row in rows),
        "regional_mil_is_not_hard_pixel_supervision": True,
        "source_annotation_ids_preserved": True,
    }
