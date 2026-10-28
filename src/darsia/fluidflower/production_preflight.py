"""Read-only summary of a saved plan before production execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from darsia.fluidflower.photometry_consistency import (
    evaluate_photometry_stage_contracts,
)


_OPTICAL_BRANCH_KEYS = (
    "correction_revision_id",
    "correction_revision_sha256",
    "correction_dependency_sha256",
    "correction_algorithm",
    "phase_revision_id",
    "phase_revision_sha256",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _positive_int(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def summarize_production_plan(
    module: str,
    plan: Mapping[str, Any],
    *,
    current_optical_branch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe scope and provenance without changing or executing a plan.

    A mass plan records a revision link between visible phase and correction. It
    does not by itself certify that every calibration-fit and application image
    used an identical photometric chain; that distinction is kept explicit.
    """

    saved = _mapping(plan)
    contract = _mapping(saved.get("task_contract") or saved.get("objective_contract"))
    validation = _mapping(saved.get("validation"))
    summary = _mapping(validation.get("summary"))
    upstream = _mapping(saved.get("upstream"))
    branch = _mapping(contract.get("optical_branch") or upstream.get("optical_branch"))
    stage_contracts = contract.get("photometry_stage_contracts") or saved.get(
        "photometry_stage_contracts"
    )
    photometry_evidence = evaluate_photometry_stage_contracts(stage_contracts)
    run_bindings = [
        _mapping(row)
        for row in contract.get("run_bindings") or []
        if isinstance(row, Mapping)
    ]
    run_count = len(run_bindings) or _positive_int(summary.get("run_count"))
    times = _mapping(contract.get("resolved_time_selection"))
    selected_times = _positive_int(
        times.get("selected_time_count") or summary.get("selected_time_count") or 0
    )
    planned_images = run_count * selected_times if run_count and selected_times else 0

    photometry = "not_applicable"
    if module == "mass_analysis":
        expected = _mapping(current_optical_branch)
        if not branch or not expected:
            photometry = "unverified"
        elif not bool(branch.get("selection_confirmed")) or not bool(
            expected.get("compatible")
        ):
            photometry = "mismatch"
        elif any(
            str(branch.get(key) or "") != str(expected.get(key) or "")
            for key in _OPTICAL_BRANCH_KEYS
        ):
            photometry = "mismatch"
        else:
            photometry = "revision_linked"

    stored_ready = (
        bool(validation.get("execution_ready"))
        if module == "mass_analysis"
        else bool(
            _mapping(saved.get("execution")).get("ready")
            or validation.get("execution_ready")
        )
    )
    expected = _mapping(current_optical_branch)
    legacy_contract_ready = (
        module == "mass_analysis" and stored_ready and photometry == "revision_linked"
    )
    # The old branch check assumes that phase and concentration use one
    # correction. A fixed categorical phase map may legitimately use another
    # correction, but only a complete, registered, immutable stage contract
    # can make that exception explicit. Never bypass a changed phase revision
    # or frame-quality blacklist through this route.
    fixed_phase_variant_ready = (
        module == "mass_analysis"
        and stored_ready
        and photometry == "mismatch"
        and bool(branch.get("selection_confirmed"))
        and expected.get("frame_quality_compatible") is True
        and all(
            str(branch.get(key) or "") == str(expected.get(key) or "")
            for key in ("phase_revision_id", "phase_revision_sha256")
        )
        and photometry_evidence["stage_contract_format"] == "recognized"
        and photometry_evidence["cross_chain"]["status"] == "different"
        and photometry_evidence["fixed_phase_ablation"]["status"] == "ready"
        and photometry_evidence["end_to_end_photometry"]["status"] == "ready"
    )
    return {
        "plan_id": str(saved.get("plan_id") or ""),
        "plan_sha256": str(saved.get("content_sha256") or ""),
        "run_count": run_count,
        "selected_time_count": selected_times,
        "planned_images": planned_images,
        "correction_revision_id": str(branch.get("correction_revision_id") or ""),
        "correction_revision_sha256": str(
            branch.get("correction_revision_sha256") or ""
        ),
        "phase_revision_id": str(branch.get("phase_revision_id") or ""),
        "phase_revision_sha256": str(branch.get("phase_revision_sha256") or ""),
        "calibration_plan_id": str(contract.get("calibration_plan_id") or ""),
        "calibration_result_path": str(
            _mapping(contract.get("calibration_result")).get("path") or ""
        ),
        "calibration_result_sha256": str(
            _mapping(contract.get("calibration_result")).get("sha256") or ""
        ),
        "mass_input_path": str(_mapping(contract.get("mass_inputs")).get("path") or ""),
        "mass_input_sha256": str(
            _mapping(contract.get("mass_inputs")).get("sha256") or ""
        ),
        "photometry": photometry,
        "photometry_evidence": photometry_evidence,
        "fit_application_photometry": photometry_evidence["concentration_chain"][
            "status"
        ],
        "phase_training_photometry": photometry_evidence["phase_chain"]["status"],
        "cross_chain_photometry": photometry_evidence["cross_chain"]["status"],
        "registered_grid": photometry_evidence["registered_grid"]["all_stages"][
            "status"
        ],
        "end_to_end_photometry": photometry_evidence["end_to_end_photometry"]["status"],
        "stored_ready": stored_ready,
        "ready_under_existing_contract": legacy_contract_ready,
        "fixed_phase_variant_ready": fixed_phase_variant_ready,
        "ready_for_mass_enqueue": legacy_contract_ready or fixed_phase_variant_ready,
    }


def validate_mass_production_task_photometry(
    task: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    plan_file_sha256: str,
) -> dict[str, Any]:
    """Recheck declared photometry at the worker boundary.

    Legacy plans remain explicitly provisional. This check does not infer
    photometry from image bytes or certify a scientific mass calibration.
    """

    saved = _mapping(plan)
    contract = _mapping(saved.get("task_contract"))
    stage_contracts = contract.get("photometry_stage_contracts") or saved.get(
        "photometry_stage_contracts"
    )
    report = evaluate_photometry_stage_contracts(stage_contracts)
    if report["end_to_end_photometry"]["status"] == "mismatch":
        raise ValueError(
            "Saved mass plan declares conflicting phase/concentration correction "
            "chains or registered grids"
        )

    adapter = _mapping(task.get("adapter"))
    settings = _mapping(adapter.get("settings"))
    preflight = _mapping(settings.get("production_photometry_preflight"))
    if preflight:
        if preflight.get("format") != "darsia-fluidflower-production-preflight-v1":
            raise ValueError("Unsupported production photometry preflight format")
        if str(preflight.get("plan_file_sha256") or "") != plan_file_sha256:
            raise ValueError("Production preflight references a different plan file")
        if preflight.get("stage_evidence") != report:
            raise ValueError("Production photometry evidence changed after queueing")
        expected_status = (
            "matched_declared_stage_hashes"
            if report["end_to_end_photometry"]["status"] == "ready"
            else "provisional_unverified_stage_hashes"
        )
        if preflight.get("status") != expected_status:
            raise ValueError("Production photometry status conflicts with saved plan")
        if preflight.get("basis") == "fixed_categorical_phase_variant":
            if not (
                report["stage_contract_format"] == "recognized"
                and report["end_to_end_photometry"]["status"] == "ready"
                and report["fixed_phase_ablation"]["status"] == "ready"
                and report["cross_chain"]["status"] == "different"
            ):
                raise ValueError("Fixed-phase variant lost its required provenance")
        elif preflight.get("basis") != "legacy_revision_link":
            raise ValueError("Unknown production photometry preflight basis")
    return {
        "status": (
            "matched_declared_stage_hashes"
            if report["end_to_end_photometry"]["status"] == "ready"
            else "provisional_unverified_stage_hashes"
        ),
        "stage_evidence": report,
    }


def validate_mass_input_revision_link(
    plan: Mapping[str, Any], input_manifest: Mapping[str, Any]
) -> str:
    """Reject a prepared optical field from a different saved correction/phase.

    Old manifests without either revision are unverified rather than assumed to
    match. This is only the mass-input side of the concentration provenance;
    legacy calibration-result manifests do not yet identify their fit chain.
    """

    contract = _mapping(_mapping(plan).get("task_contract"))
    branch = _mapping(contract.get("optical_branch"))
    upstream = _mapping(_mapping(input_manifest).get("upstream"))
    pairs = (
        ("correction_revision_sha256", "image_correction_revision_sha256"),
        ("phase_revision_sha256", "phase_revision_sha256"),
    )
    complete = True
    for planned_key, input_key in pairs:
        planned = str(branch.get(planned_key) or "")
        actual = str(upstream.get(input_key) or "")
        if planned and actual and planned != actual:
            raise ValueError(
                f"Prepared mass inputs use a different {input_key} than the "
                "saved production plan"
            )
        complete = complete and bool(planned and actual)
    return "matched" if complete else "unverified"
