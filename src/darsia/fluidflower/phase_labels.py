"""Shared visible-2D phase vocabulary for FluidFlower review clients.

The vocabulary is deliberately independent of Qt and the browser review tool.
Blind labeling, open model correction, training import, and future DarSIA GUIs
can therefore exchange the same stable phase classes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


VISIBLE_PHASE_LABEL_SCHEMA = "fluidflower-visible-phase-v3"


@dataclass(frozen=True)
class VisiblePhaseChoice:
    """One mutually exclusive human interpretation of a visible 2D region."""

    key: str
    label: str
    canonical_class: str
    dominant_phase: str | None
    components: frozenset[str]
    training_eligible: bool = True
    description: str = ""


VISIBLE_PHASE_CHOICES: tuple[VisiblePhaseChoice, ...] = (
    VisiblePhaseChoice(
        "water_or_no_visible_co2",
        "Water / no visible CO2",
        "water",
        "water",
        frozenset({"water"}),
        description="No visible evidence of CO2 introduced during this experiment.",
    ),
    VisiblePhaseChoice(
        "only_aqueous_co2",
        "Only aqueous CO2",
        "aqueous",
        "aqueous",
        frozenset({"aqueous"}),
        description="Aqueous CO2 is visible; no free gas or untouched water is visible.",
    ),
    VisiblePhaseChoice(
        "predominantly_aqueous_co2",
        "Mostly aqueous CO2, some free CO2",
        "mixed_gas_aqueous",
        "aqueous",
        frozenset({"aqueous", "gas"}),
        description="Both CO2 phases are visible, with aqueous CO2 clearly dominant.",
    ),
    VisiblePhaseChoice(
        "mixed_free_and_aqueous_co2",
        "Mixture: free + aqueous CO2",
        "mixed_gas_aqueous",
        None,
        frozenset({"aqueous", "gas"}),
        description="Both CO2 phases are materially visible and neither clearly dominates.",
    ),
    VisiblePhaseChoice(
        "predominantly_free_gas",
        "Mostly free CO2, some aqueous CO2",
        "mixed_gas_aqueous",
        "gas",
        frozenset({"aqueous", "gas"}),
        description="Both CO2 phases are visible, with free CO2 clearly dominant.",
    ),
    VisiblePhaseChoice(
        "free_gas_only",
        "Only free CO2",
        "gas",
        "gas",
        frozenset({"gas"}),
        description="Free CO2 is visible; no aqueous CO2 or untouched water is visible.",
    ),
    VisiblePhaseChoice(
        "mixed_aqueous_co2_and_water",
        "Mixture: aqueous CO2 + untouched water",
        "mixed_aqueous_water",
        "aqueous",
        frozenset({"aqueous", "water"}),
        description="Aqueous CO2 and untouched water are both visible.",
    ),
    VisiblePhaseChoice(
        "mixed_free_aqueous_co2_and_water",
        "Mixture: free + aqueous CO2 + untouched water",
        "mixed_gas_aqueous_water",
        None,
        frozenset({"aqueous", "gas", "water"}),
        description="Free CO2, aqueous CO2, and untouched water are all visible.",
    ),
    VisiblePhaseChoice(
        "co2_present_phase_uncertain",
        "CO2 visible; phase split uncertain",
        "co2_present_region",
        None,
        frozenset({"co2"}),
        description="CO2-related optical change is visible, but its phase split is uncertain.",
    ),
    VisiblePhaseChoice(
        "not_assessable",
        "Uncertain / cannot assess",
        "ignored",
        None,
        frozenset(),
        training_eligible=False,
        description="The image does not support a defensible phase interpretation.",
    ),
    VisiblePhaseChoice(
        "phase_signal_artifact",
        "Artifact / not a valid phase signal",
        "ignored",
        None,
        frozenset(),
        training_eligible=False,
        description="The visible response is an image or experiment artifact, not a phase label.",
    ),
)

VISIBLE_PHASE_CHOICE_BY_KEY: Mapping[str, VisiblePhaseChoice] = {
    choice.key: choice for choice in VISIBLE_PHASE_CHOICES
}
VISIBLE_PHASE_LABELS: Mapping[str, str] = {
    choice.key: choice.label for choice in VISIBLE_PHASE_CHOICES
}
VISIBLE_PHASE_CANONICAL_CLASSES: Mapping[str, str] = {
    choice.key: choice.canonical_class for choice in VISIBLE_PHASE_CHOICES
}

# New manual labels are intentionally more precise than legacy
# ``confirmed_free_gas`` and ``confirmed_aqueous_co2`` records, whose labels
# historically combined pure and dominant regions.
PHASE_CHOICE_TO_MANUAL_CATEGORY: Mapping[str, str] = {
    "water_or_no_visible_co2": "confirmed_no_co2",
    "only_aqueous_co2": "confirmed_only_aqueous_co2",
    "predominantly_aqueous_co2": "confirmed_predominantly_aqueous_co2",
    "mixed_free_and_aqueous_co2": "confirmed_mixed_free_aqueous_co2",
    "predominantly_free_gas": "confirmed_predominantly_free_gas",
    "free_gas_only": "confirmed_only_free_gas",
    "mixed_aqueous_co2_and_water": "confirmed_mixed_aqueous_water",
    "mixed_free_aqueous_co2_and_water": "confirmed_mixed_free_aqueous_water",
    "co2_present_phase_uncertain": "phase_ambiguous_strong_color_response",
    "not_assessable": "phase_not_assessable",
    "phase_signal_artifact": "phase_signal_artifact",
    "ignore_region": "other",
}
MANUAL_CATEGORY_TO_PHASE_CHOICE: Mapping[str, str] = {
    value: key
    for key, value in PHASE_CHOICE_TO_MANUAL_CATEGORY.items()
    if key != "ignore_region"
}

MODEL_PHASE_COMPONENTS: Mapping[str, frozenset[str]] = {
    "water": frozenset({"water"}),
    "aqueous": frozenset({"aqueous"}),
    "gas": frozenset({"gas"}),
    "mixed": frozenset({"aqueous", "gas"}),
}


def visible_phase_payload(choice_key: str) -> dict[str, Any]:
    """Return the stable machine-readable payload for one human choice."""

    if str(choice_key) == "ignore_region":
        return {
            "human_choice": "ignore_region",
            "phase_label_schema": VISIBLE_PHASE_LABEL_SCHEMA,
            "phase_class": "ignored",
            "phase_dominance": "none",
            "phase_components": [],
            "training_eligible": False,
            "interpretation_scope": "visible_2d_slice_only",
        }
    choice = VISIBLE_PHASE_CHOICE_BY_KEY.get(str(choice_key))
    if choice is None:
        raise ValueError(f"Unknown visible phase choice: {choice_key!r}")
    return {
        "human_choice": choice.key,
        "phase_label_schema": VISIBLE_PHASE_LABEL_SCHEMA,
        "phase_class": choice.canonical_class,
        "phase_dominance": choice.dominant_phase or "none",
        "phase_components": sorted(choice.components),
        "training_eligible": choice.training_eligible,
        "interpretation_scope": "visible_2d_slice_only",
    }


def assess_model_phase(model_phase: str, human_choice: str) -> dict[str, Any]:
    """Derive a transparent model-versus-human comparison for one region."""

    choice = VISIBLE_PHASE_CHOICE_BY_KEY.get(str(human_choice))
    predicted = MODEL_PHASE_COMPONENTS.get(str(model_phase).strip().lower())
    if choice is None or predicted is None or not choice.training_eligible:
        return {
            "model_phase_class": str(model_phase or "unscored"),
            "model_assessment": "unscored",
            "missing_phase_components": [],
            "unsupported_phase_components": [],
        }
    observed = choice.components
    missing = sorted(observed - predicted)
    unsupported = sorted(predicted - observed)
    if not missing and not unsupported:
        assessment = "compatible"
    elif missing and unsupported:
        assessment = "phase_substitution"
    elif missing:
        assessment = "model_missed_visible_phase"
    else:
        assessment = "model_added_unsupported_phase"
    return {
        "model_phase_class": str(model_phase),
        "model_assessment": assessment,
        "missing_phase_components": missing,
        "unsupported_phase_components": unsupported,
    }


def phase_choice_from_record(record: Mapping[str, Any]) -> str | None:
    """Recover a shared choice key from new or legacy annotation records."""

    choice = str(record.get("human_choice") or "").strip()
    if choice in VISIBLE_PHASE_CHOICE_BY_KEY:
        return choice
    return MANUAL_CATEGORY_TO_PHASE_CHOICE.get(str(record.get("category") or ""))
