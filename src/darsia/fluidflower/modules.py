"""Stable module registry and quality-gate rules for FluidFlower projects."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

VALID_MODULE_STATUSES = {
    "not_started",
    "ready",
    "in_progress",
    "review_required",
    "approved",
    "complete",
    "stale",
    "blocked",
}

SATISFIED_MODULE_STATUSES = {"approved", "complete"}


@dataclass(frozen=True)
class ModuleDefinition:
    """One independently replaceable application module."""

    key: str
    title: str
    description: str
    dependencies: tuple[str, ...] = ()
    help_text: str = ""


MODULE_DEFINITIONS: tuple[ModuleDefinition, ...] = (
    ModuleDefinition(
        "setup",
        "Series setup",
        "Apparatus class, protocol, chemistry, image sources, and experiments.",
        help_text=(
            "Purpose: establish the experimental series and reference its original "
            "inputs. Define the apparatus class, protocol organization, image folders, "
            "run mapping, and any overflow-source candidates found with the raw data. "
            "The active overflow source is selected after data-integrity review.\n\n"
            "Output: a versioned project manifest and immutable source references. "
            "DarSIA does not move or modify the source files during setup."
        ),
    ),
    ModuleDefinition(
        "data_integrity",
        "Data and integrity",
        "Timestamps, duplicates, missing frames, resolution, and source provenance.",
        ("setup",),
        help_text=(
            "Purpose: determine whether the registered files form a complete and "
            "traceable basis for analysis. Fast scan checks metadata and selected "
            "content; verified scan reads and hashes every file.\n\n"
            "Approval confirms data integrity, not that image correction or physical "
            "interpretation is scientifically correct."
        ),
    ),
    ModuleDefinition(
        "overflow",
        "Overflow measurements",
        "Choose, scan, and review scale-display images or imported overflow tables.",
        ("data_integrity",),
        help_text=(
            "Purpose: establish the overflow-weight evidence used for temporal "
            "alignment and phase constraints. Choose the active source separately "
            "for every experiment: no overflow, scale-display images, or a CSV/Excel "
            "table. Image sources are scanned before the dedicated reader performs "
            "OCR and manual validation.\n\n"
            "The choice remains editable after setup. Replacing image interpretation "
            "with a table creates a new revision and marks dependent results outdated; "
            "raw images and workbooks are never modified."
        ),
    ),
    ModuleDefinition(
        "temporal_sync",
        "Temporal synchronization",
        "Align protocol, injection, experiment images, scale images, overflow, and interventions.",
        ("data_integrity", "overflow"),
        help_text=(
            "Purpose: align every observation and experimental action to one approved "
            "time origin. The registered protocol workbooks are read automatically and "
            "shown per experiment as camera, injection, overflow, chemistry, and other "
            "event streams. Review each experiment or approve the complete series.\n\n"
            "Original timestamps remain unchanged. Local corrections are versioned and "
            "visible beside the raw values."
        ),
    ),
    ModuleDefinition(
        "image_correction",
        "Image correction",
        "Register, align, unfold, crop, and correct lighting and color before geometry or optical interpretation.",
        ("data_integrity",),
        help_text=(
            "Purpose: establish the common pixel grid and remove camera and lighting "
            "variation before facies, color paths, or phase models are built. Use a "
            "ColorChecker when one is available; otherwise record the absence of an "
            "external radiometric target and inspect temporal stability without treating "
            "pH buffers as light references.\n\n"
            "The saved revision records the exact reference images, transforms, settings, "
            "and quality diagnostics used downstream. Changing the active revision marks "
            "geometry, optical calibration, phase interpretation, mass calibration, and "
            "all analyses as outdated."
        ),
    ),
    ModuleDefinition(
        "geometry_facies",
        "Geometry and facies",
        "Define depth, injection points, layer boundaries, porosity, and permeability on the corrected grid.",
        ("image_correction",),
        help_text=(
            "Purpose: define the physical 2D analysis domain and its connection to the "
            "3D rig on the active corrected image grid. Import an existing geometry or "
            "review layer boundaries, facies grouping, depth, porosity, permeability, "
            "excluded regions, and injection-point positions.\n\n"
            "Optical facies proposals are drafts. Physical properties and final layer "
            "assignments require expert review before approval."
        ),
    ),
    ModuleDefinition(
        "optical_calibration",
        "Optical and chemical calibration",
        "Build correction-linked color response paths from measured buffers or other approved optical references.",
        ("image_correction", "geometry_facies", "temporal_sync"),
        help_text=(
            "Purpose: convert corrected optical changes into a versioned response model "
            "before phase interpretation. Known-pH buffer images can fit a monotone, "
            "physically constrained full-color response; projects without buffers retain "
            "the approved generic optical branch.\n\n"
            "Every response model is checksum-linked to its image-correction and geometry "
            "revisions. Optuna may later vary reviewed chemical uncertainty bounds, but "
            "cannot redefine a measured color response."
        ),
    ),
    ModuleDefinition(
        "phase_interpretation",
        "Phase interpretation",
        "Register, validate, and select visible water, aqueous CO2, free-gas, and artifact interpretations.",
        ("optical_calibration",),
        help_text=(
            "Purpose: classify the visible 2D optical signal as unaffected water, "
            "aqueous CO2, free gas, mixtures, or artifacts. Active learning collects "
            "targeted human labels; model validation reviews complete temporal phase maps.\n\n"
            "This module identifies phase evidence. It does not yet assign concentration, "
            "saturation, depth, or mass."
        ),
    ),
    ModuleDefinition(
        "calibration",
        "Mass calibration",
        "Optimize experiment-specific concentration, 2D-to-3D, and mass parameters.",
        ("phase_interpretation",),
        help_text=(
            "Purpose: estimate experiment-specific parameters that convert optical phase "
            "evidence into physical quantities. Select Simple Flash, Titration, or both; "
            "define pH bounds, searchable parameters, objectives, and numerical backend.\n\n"
            "Optimization and final selection use the same recorded objective contract. "
            "CPU float64 remains the numerical reference for accelerator parity. After "
            "optimization, every experiment must pass a separate visual review of its "
            "concentration map and injected-versus-detected mass plot."
        ),
    ),
    ModuleDefinition(
        "mass_analysis",
        "Mass analysis",
        "Convert corrected phase evidence to concentration, saturation, and mass fields.",
        ("calibration", "geometry_facies", "temporal_sync"),
        help_text=(
            "Purpose: apply the approved phase map, calibration, geometry, depth, and "
            "porosity to each selected image. The result contains concentration, gas "
            "saturation, and aqueous/free/total CO2 mass fields.\n\n"
            "Quality and precision settings are part of the result provenance and can be "
            "changed between preparation and production analysis."
        ),
    ),
    ModuleDefinition(
        "wasserstein",
        "Wasserstein analysis",
        "Compare mass-preserving spatial distributions at synchronized experiment times.",
        ("mass_analysis",),
        help_text=(
            "Purpose: compare where CO2 mass is located, not only how much is detected. "
            "Choose runs, synchronized times, analysis boxes, mass components, resolution, "
            "and comparison budget.\n\n"
            "Inputs must be mass preserving and share approved spatial and temporal "
            "references before distances are interpreted."
        ),
    ),
    ModuleDefinition(
        "post_analysis",
        "Post-analysis",
        "Generate phase movies, mass curves, contour products, grids, and diagnostics.",
        ("mass_analysis",),
        help_text=(
            "Purpose: create review and publication products from immutable analysis "
            "results. Select runs, products, domains, frame rate, and output resolution.\n\n"
            "Post-analysis visualizes existing results; it does not recalibrate or alter "
            "the underlying mass fields."
        ),
    ),
)

MODULE_BY_KEY = {definition.key: definition for definition in MODULE_DEFINITIONS}


def initial_module_states() -> dict[str, dict[str, Any]]:
    """Return a new, serializable module-state mapping."""

    return {
        definition.key: {
            "status": "ready" if not definition.dependencies else "not_started",
            "updated_utc": None,
            "updated_by": "",
            "reason": "",
        }
        for definition in MODULE_DEFINITIONS
    }


def normalize_module_states(
    raw_states: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Merge stored states with the current registry without dropping history."""

    normalized = initial_module_states()
    for key, raw in (raw_states or {}).items():
        if key not in normalized or not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status") or normalized[key]["status"])
        if status not in VALID_MODULE_STATUSES:
            status = normalized[key]["status"]
        normalized[key].update(dict(raw))
        normalized[key]["status"] = status
    return normalized


def dependencies_satisfied(
    module_key: str,
    states: Mapping[str, Mapping[str, Any]],
) -> bool:
    definition = MODULE_BY_KEY[module_key]
    return all(
        str(states.get(dependency, {}).get("status") or "") in SATISFIED_MODULE_STATUSES
        for dependency in definition.dependencies
    )


def available_module_keys(
    states: Mapping[str, Mapping[str, Any]],
) -> tuple[str, ...]:
    """Return modules whose dependencies currently pass their quality gates."""

    return tuple(
        definition.key
        for definition in MODULE_DEFINITIONS
        if dependencies_satisfied(definition.key, states)
    )


def downstream_module_keys(module_key: str) -> tuple[str, ...]:
    """Return the transitive set of modules depending on ``module_key``."""

    pending = [module_key]
    found: list[str] = []
    while pending:
        parent = pending.pop(0)
        for definition in MODULE_DEFINITIONS:
            if parent not in definition.dependencies or definition.key in found:
                continue
            found.append(definition.key)
            pending.append(definition.key)
    return tuple(found)


def invalidate_downstream(
    states: MutableMapping[str, MutableMapping[str, Any]],
    module_key: str,
    *,
    updated_utc: str,
    updated_by: str,
    reason: str,
) -> tuple[str, ...]:
    """Mark started descendants stale after an upstream interpretation changes."""

    changed: list[str] = []
    for key in downstream_module_keys(module_key):
        state = states[key]
        if str(state.get("status") or "") not in {
            "ready",
            "in_progress",
            "approved",
            "complete",
            "review_required",
        }:
            continue
        state.update(
            {
                "status": "stale",
                "updated_utc": updated_utc,
                "updated_by": updated_by,
                "reason": reason,
            }
        )
        changed.append(key)
    return tuple(changed)
