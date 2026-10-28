"""Exact, read-only admission of the saved P1-final-2 A–D revision."""
from functools import lru_cache
import json
from pathlib import Path

from .phase_saved_boundary_review import _bound, _checked_sha, _head_assets, _require, _resolved_path, _signature, _stamped_dependencies
from .review_storage import resolve_review_path
from .phase_saved_review_metadata import (canonical_sha256, configured_parameters,
    configured_export, configured_model_identity as _model_identity,
    configured_registration_path as registration_path)

_METADATA_EXPORTS = ('CONTRACT', 'FILES', 'KIND', 'OVERLAYS', 'PLAN', 'QA', 'SEAL', 'SPEC_SHA256', 'STATE_LABEL', 'WARNING', 'ZONE_ARRAY_SHA256')


def __getattr__(name):
    return configured_export('categorical', name, allowed_names=_METADATA_EXPORTS)




def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "P1-final2 metadata must be an object")
    return value


@lru_cache(maxsize=16)
def _saved_dependencies(root_text, plan_stamp):
    PLAN = configured_export('categorical', 'PLAN', allowed_names=_METADATA_EXPORTS)
    plan = _read(Path(root_text) / PLAN)
    inputs = plan["binding"]["inputs"]
    assets = tuple(inputs[key] for key in ("parent_seal", "parent_registration", "spec",
                                           "a_gate_report", "pooled_floor_curve", "zones"))
    return plan, assets, tuple(_head_assets(plan["binding"]["heads"]))


def _parent_admission(parent):
    # Let the parent's own cache check its transitive metadata/head signatures;
    # caching only its immediate plan stamp would miss changes further upstream.
    from .phase_saved_boundary_review import p1_final_saved_review
    return p1_final_saved_review(parent)


def _validate(root, registration, *, check_parent=True):
    CONTRACT = configured_export('categorical', 'CONTRACT', allowed_names=_METADATA_EXPORTS)
    FILES = configured_export('categorical', 'FILES', allowed_names=_METADATA_EXPORTS)
    KIND = configured_export('categorical', 'KIND', allowed_names=_METADATA_EXPORTS)
    PLAN = configured_export('categorical', 'PLAN', allowed_names=_METADATA_EXPORTS)
    QA = configured_export('categorical', 'QA', allowed_names=_METADATA_EXPORTS)
    SEAL = configured_export('categorical', 'SEAL', allowed_names=_METADATA_EXPORTS)
    SPEC_SHA256 = configured_export('categorical', 'SPEC_SHA256', allowed_names=_METADATA_EXPORTS)
    STATE_LABEL = configured_export('categorical', 'STATE_LABEL', allowed_names=_METADATA_EXPORTS)
    ZONE_ARRAY_SHA256 = configured_export('categorical', 'ZONE_ARRAY_SHA256', allowed_names=_METADATA_EXPORTS)
    RULE = configured_parameters('categorical')["rules"]
    root = Path(root)
    identity = list(_model_identity(root))
    _require(registration.get("registration_kind") == KIND
        and _resolved_path(registration.get("model_root", "")) == root.resolve()
        and [registration.get("model_id"), registration.get("model_sha256")] == identity
        and registration.get("publication_scope") == "studio_validation_only"
        and (registration.get("run_count"), registration.get("frame_count")) == (43, 864)
        and registration.get("blind_hold_runs") == []
        and registration.get("physical_scale_approved") is True
        and all(registration.get(key) is False for key in ("mass_eligible", "bottom_mass_allowed",
            "scientific_acceptance", "deployment_allowed", "automatic_activation")),
        "P1-final2 registration changed its identity or review-only restrictions")
    metadata = registration.get("metadata_sha256") or {}
    _require(set(metadata) == set(FILES), "P1-final2 registration lost its complete metadata inventory")
    for relative, expected in metadata.items():
        _checked_sha(str(root / relative), expected, _signature(root / relative))
    _require(metadata[PLAN] == registration.get("source_plan_sha256")
        and metadata[SEAL] == registration.get("production_receipt_sha256"), "P1-final2 plan/seal registration differs")
    plan, seal, qa = _read(root / PLAN), _read(root / SEAL), _read(root / QA)
    unchanged = "source_time_raw_geometry_area_heads_delta_unchanged"
    _require(plan.get("contract") == seal.get("contract") == qa.get("contract") == CONTRACT
        and plan.get("model_identity") == seal.get("model_identity") == identity
        and seal.get("complete") is True and qa.get("technical_all_pass") is True
        and (seal.get("run_count"), seal.get("frame_count")) == (43, 864)
        and (qa.get("run_count"), qa.get("frame_count")) == (43, 864)
        and qa.get("rules") == RULE and qa.get(unchanged) is True and seal.get(unchanged) is True
        and seal.get("domain_revision") == qa.get("domain_revision") == "pooled_sand_floor_v1"
        and all(qa.get(key) is False for key in ("annotations_read", "training", "inference"))
        and seal.get("source_plan_sha256") == metadata[PLAN]
        and seal.get("pipeline_sha256") == plan.get("pipeline_sha256")
        and plan.get("job_sha256") == canonical_sha256(plan["binding"])
        and plan.get("pipeline_sha256") == canonical_sha256(dict(binding=plan["binding"], model_identity=identity)),
        "P1-final2 saved labels differ from their exact complete A–D seal")
    gate = qa.get("a_gate") or {}
    _require(gate.get("all_pass") is True and gate.get("required_passing_runs") == 39
        and gate.get("maximum_median_absolute_residual_mm") == 1.0
        and all(isinstance(gate.get(key), int) and 39 <= gate[key] <= 43
                for key in ("reference_passing_runs", "last_passing_runs")),
        "P1-final2 did not pass the fixed photo-floor gate before postprocessing")
    inputs = plan["binding"]["inputs"]
    _require((inputs.get("spec") or {}).get("sha256") == SPEC_SHA256
        and inputs.get("rules") == RULE and plan.get("placement") == "P1-final2"
        and gate.get("report") == inputs.get("a_gate_report")
        and all(inputs.get(key) is False for key in ("training", "inference", "annotations_read", "mass_eligible", "bottom_mass_allowed"))
        and all(seal.get(key) is False for key in ("training_performed", "inference_performed", "labels_changed",
            "mass_eligible", "bottom_mass_allowed", "scientific_acceptance", "automatic_activation"))
        and seal.get("blind_hold_runs") == [], "P1-final2 changed its postprocessing-only or visibility scope")
    _require(_bound(seal.get("qa"), base=root) == (root / QA).resolve(), "P1-final2 QA binding differs")
    parent = _resolved_path(plan["binding"]["parent"])
    _require(_bound(inputs["parent_seal"], base=root) == (parent / SEAL).resolve()
        and seal.get("parent_seal") == inputs["parent_seal"], "P1-final2 lost its exact P1-final parent seal")
    for key in ("parent_registration", "spec", "a_gate_report", "pooled_floor_curve", "zones"):
        _bound(inputs[key], base=root)
    gate_report = _read(_bound(inputs["a_gate_report"], base=root))
    measured_gate = gate_report.get("gate") or {}
    _require(gate_report.get("complete") is True and measured_gate.get("passes") is True
        and measured_gate.get("required_passing_runs") == 39
        and measured_gate.get("maximum_median_absolute_residual_mm") == 1.0
        and all(measured_gate.get(key) == gate[key] for key in ("reference_passing_runs", "last_passing_runs"))
        and gate_report.get("candidate") == inputs["parent_seal"]
        and (gate_report.get("canonical_curve") or {}).get("asset") == inputs["pooled_floor_curve"]
        and (inputs.get("zones") or {}).get("array_sha256") == ZONE_ARRAY_SHA256,
        "P1-final2 changed its measured floor-gate, pooled curve or locked L1-zone binding")
    parent_seal = _read(parent / SEAL)
    _require(parent_seal.get("placement") == "P1-final" and plan.get("delta") == parent_seal.get("delta")
        and plan["binding"]["heads"] == parent_seal.get("heads") == seal.get("heads"),
        "P1-final2 changed the inherited P1-final heads or lamp shift")
    if check_parent:
        checked = _parent_admission(parent)
        _require(checked["placement"] == "P1-final", "P1-final2 parent is not admitted P1-final")
        _require(_bound(inputs["parent_registration"], base=root) == _resolved_path(str(registration_path(parent.name)))
            and _read(_bound(inputs["parent_registration"], base=root)) == checked["registration"],
            "P1-final2 parent registration binds another candidate")
    frames, inventory, rows = seal.get("frames") or [], plan.get("source_inventory") or [], qa.get("rows") or []
    keys = lambda items: {(f.get("run"), f.get("frame_index")) for f in items}
    _require(len(frames) == len(keys(frames)) == len(inventory) == len(rows) == 864
        and len({f.get("run") for f in frames}) == 43 and keys(frames) == keys(inventory) == keys(rows)
        and all(row.get("all_pass") is True for row in rows)
        and all(isinstance(f.get("source_sha256"), str) and len(f["source_sha256"]) == 64
                and isinstance(f.get("elapsed_h"), (int, float)) for f in inventory),
        "P1-final2 seal lacks 43 runs / 864 unique source-owned frames")
    from .phase_saved_boundary_review import PLAN as PARENT_PLAN
    parent_inventory = {(f["run"], f["frame_index"]): f for f in _read(parent / PARENT_PLAN)["source_inventory"]}
    _require(keys(inventory) == set(parent_inventory)
        and all(all(f.get(key) == parent_inventory[(f["run"], f["frame_index"])].get(key)
                    for key in ("source_sha256", "elapsed_h")) for f in inventory),
        "P1-final2 source/time inventory differs from the admitted P1-final sources")
    profile = _read(root / "phase_rig_camera.json")
    marker = profile.get("p1_final2_revision") or {}
    _require(marker.get("contract") == CONTRACT and marker.get("plan", {}).get("sha256") == metadata[PLAN]
        and not any(key in profile for key in ("p1_final_revision", "g3_t4c_floor_revision", "g3_t4b_floor_revision",
            "g3_t4_floor_revision", "g3_t3_floor_revision", "g3_t2_floor_revision", "g3_t2_l1zone_revision",
            "l1_neighbourhood_floor_revision")), "P1-final2 profile mixed an old active rule or lost its revision")
    return dict(registration=registration, pipeline_sha256=plan["pipeline_sha256"], model_identity=identity,
                placement="P1-final2", delta=plan["delta"], blind_hold_runs=[], state_label=STATE_LABEL)


@lru_cache(maxsize=16)
def _admitted(root_text, registration_text, stamps):
    return _validate(Path(root_text), _read(registration_text))


def p1_final2_saved_review(model_root):
    FILES = configured_export('categorical', 'FILES', allowed_names=_METADATA_EXPORTS)
    PLAN = configured_export('categorical', 'PLAN', allowed_names=_METADATA_EXPORTS)
    root = _resolved_path(str(model_root))
    path = _resolved_path(str(registration_path(root.name)))
    plan_stamp = _signature(root / PLAN)
    plan, assets, heads = _saved_dependencies(str(root), plan_stamp)
    _, dependency_stamps = _stamped_dependencies(root, path, FILES, (*assets, *heads),
        plan_path=root / PLAN, plan_stamp=plan_stamp)
    checked = _parent_admission(_resolved_path(plan["binding"]["parent"]))
    stamps = (tuple(checked["model_identity"]), checked["pipeline_sha256"],
              dependency_stamps)
    return _admitted(str(root), str(path), stamps)


def registered_warning(root, registration):
    WARNING = configured_export('categorical', 'WARNING', allowed_names=_METADATA_EXPORTS)
    try:
        _validate(resolve_review_path(root).resolve(), registration, check_parent=False)
        return WARNING
    except (OSError, KeyError, ValueError, TypeError):
        return "QA FAILED: P1-final2 saved registration changed or is incomplete; do not use for mass."


def require_mass_exclusion(candidate_or_child):
    """Keep the disputed bottom out of readers that cannot apply its mask."""
    PLAN = configured_export('categorical', 'PLAN', allowed_names=_METADATA_EXPORTS)
    path = resolve_review_path(candidate_or_child)
    for root in (path, *tuple(path.parents)[:12]):
        if (root / PLAN).is_file():
            raise ValueError("P1-final2 is a Studio-review candidate; its bottom strip remains blocked for mass. "
                             "This mass reader does not apply the bound bottom-strip exclusion mask")
        profile = root / "phase_rig_camera.json"
        if profile.is_file() and _read(profile).get("p1_final2_revision") is not None:
            raise ValueError("P1-final2 bottom strip remains blocked for mass; no mass eligibility was granted")


__all__ = ["p1_final2_saved_review", "registered_warning", "CONTRACT", "KIND", "PLAN", "SEAL", "QA",
           "FILES", "SPEC_SHA256", "STATE_LABEL", "WARNING", "OVERLAYS", "require_mass_exclusion"]
