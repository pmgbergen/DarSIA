"""Candidate-owned embargo and bottom-mass exclusion; old sessions cannot release it."""
from __future__ import annotations

import json
from pathlib import Path

from .project import sha256_file
from .review_storage import resolve_review_path

POLICY_FILE = "rig_bottom_review_policy.json"
CONTRACT = "ffac-rig-bottom-review-policy-v1"
PURPOSE = "rig-bottom-strip-blind-check-v1"
BLIND_RUNS = ("ac30", "ac49", "ac54")
T3_PURPOSE = "ffac-t3-gas-blind-check-v1"
T3_BLIND_RUNS = ("ac26", "ac55")
PHYSICAL_SCALE_CONTRACT = "ffac-measured-pixel-area-revision-v1"
NOTICE = "New prediction withheld until the source-bound bottom-strip blind labels are sealed. Photographs remain available."


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _bound(value, root):
    if not isinstance(value, dict) or not {"path", "sha256"} <= set(value):
        raise ValueError("Bottom-review policy lacks an exact artifact binding")
    path = resolve_review_path(value["path"], context=root / POLICY_FILE)
    if not path.is_absolute():
        path = root / path
    if sha256_file(path) != value["sha256"]:
        raise ValueError("Bottom-review bound artifact changed")
    return path


def policy_root(candidate_or_child):
    path = resolve_review_path(candidate_or_child).resolve()
    if path.is_file():
        path = path.parent
    for root in (path, *tuple(path.parents)[:12]):
        if (root / POLICY_FILE).is_file():
            return root
        # Old model roots are the terminal scope: do not probe their NAS share
        # ancestors on every mass frame, or inherit another candidate's policy.
        if (root / "summary.json").is_file() and (root / "training_request.json").is_file():
            return None
    return None


def load_policy(candidate_or_child):
    """Return only an explicitly attached new-candidate policy, never old reveal markers."""
    root = policy_root(candidate_or_child)
    if root is None:
        return None
    value = _read(root / POLICY_FILE)
    expected_runs = {PURPOSE: BLIND_RUNS, T3_PURPOSE: T3_BLIND_RUNS}.get(value.get("purpose"))
    if (value.get("contract") != CONTRACT or expected_runs is None
            or value.get("candidate_id") != root.name
            or tuple(value.get("blind_runs") or ()) != expected_runs
            or value.get("no_mass_until_blind_bottom_check") is not True):
        raise ValueError("Bottom-review policy identity or scope changed")
    _bound(value.get("source_plan"), root)
    return {**value, "_candidate_root": str(root), "_policy_sha256": sha256_file(root / POLICY_FILE)}


def write_pending_policy(candidate_base, *, source_plan, purpose=PURPOSE, blind_runs=None):
    """Create one new policy; never overwrite a model, policy, label or session."""
    root = resolve_review_path(candidate_base).resolve()
    _bound(source_plan, root)
    expected_runs = {PURPOSE: BLIND_RUNS, T3_PURPOSE: T3_BLIND_RUNS}.get(purpose)
    if expected_runs is None or (blind_runs is not None and tuple(blind_runs) != expected_runs):
        raise ValueError("Unknown candidate blind-check scope")
    value = dict(contract=CONTRACT, purpose=purpose, candidate_id=root.name,
        source_plan=dict(source_plan), blind_runs=list(expected_runs),
        no_mass_until_blind_bottom_check=True, label_seal=None,
        bottom_mass_exclusion=None, bottom_acceptance=None,
        old_blind_reveal_does_not_release=True, maps_or_labels_changed=False)
    if purpose == T3_PURPOSE:
        value.update(candidate_blind_scoring_required=True, reveal_authorization=None)
    path = root / POLICY_FILE
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return path


def _verified_candidate_seal(value, policy):
    root = Path(policy["_candidate_root"])
    path = _bound(value, root)
    if path.resolve() == (root / "physical_scale_receipt.json").resolve():
        return _verified_physical_scale_seal(_read(path), policy)
    if path.resolve() != (root / "rig_camera_candidate_receipt.json").resolve():
        raise ValueError("Blind release must bind this new candidate's final seal")
    seal = _read(path)
    if (seal.get("complete") is not True or seal.get("run_count") != 43
            or seal.get("frame_count") != 864
            or (seal.get("model_identity") or [None])[0] != policy["candidate_id"]
            or seal.get("source_plan_sha256") != policy["source_plan"]["sha256"]):
        raise ValueError("Blind release candidate seal is incomplete or binds another source plan")
    return seal


def _verified_physical_scale_seal(seal, policy):
    root = Path(policy["_candidate_root"])
    identity = seal.get("model_identity")
    if (seal.get("contract") != PHYSICAL_SCALE_CONTRACT
            or seal.get("complete") is not True or seal.get("run_count") != 43
            or seal.get("frame_count") != 864 or not isinstance(identity, list)
            or len(identity) != 2 or identity[0] != policy["candidate_id"]
            or not isinstance(identity[1], str) or not identity[1]):
        raise ValueError("Blind release physical-scale candidate seal is incomplete or binds another candidate")
    plan_path = _bound(dict(path=str(root / "physical_scale_plan.json"),
                            sha256=seal.get("source_plan_sha256")), root)
    plan = _read(plan_path)
    binding = plan.get("binding")
    if (plan.get("model_identity") != identity or not isinstance(binding, dict)
            or binding.get("contract") != PHYSICAL_SCALE_CONTRACT
            or binding.get("parent_seal") != seal.get("parent_seal")):
        raise ValueError("Blind release physical-scale plan identity or parent binding changed")
    output_root = binding.get("output_root")
    if (not isinstance(output_root, str) or not output_root
            or resolve_review_path(output_root, context=root / POLICY_FILE).resolve() != root.resolve()):
        raise ValueError("Blind release physical-scale plan binds another candidate root")
    parent_path = _bound(seal.get("parent_seal"), root)
    parent_root = binding.get("parent_root")
    if (not isinstance(parent_root, str) or not parent_root
            or parent_path.resolve() != (resolve_review_path(parent_root,
                context=root / POLICY_FILE) / "rig_camera_candidate_receipt.json").resolve()
            or parent_path.parent.resolve() == root.resolve()):
        raise ValueError("Blind release physical-scale plan lacks its exact rig-camera parent seal")
    parent = _read(parent_path)
    parent_identity = parent.get("model_identity")
    if (parent.get("complete") is not True or parent.get("run_count") != 43
            or parent.get("frame_count") != 864 or not isinstance(parent_identity, list)
            or len(parent_identity) != 2 or parent_identity[0] != parent_path.parent.name
            or not isinstance(parent_identity[1], str) or not parent_identity[1]
            or binding.get("parent_identity") != parent_identity
            or parent.get("source_plan_sha256") != policy["source_plan"]["sha256"]):
        raise ValueError("Blind release physical-scale parent seal is incomplete or binds another source plan")
    return seal


def _verified_label_seal(policy):
    bound = policy.get("label_seal")
    if bound is None:
        return None
    root = Path(policy["_candidate_root"])
    seal = _read(_bound(bound, root))
    if (seal.get("purpose") != policy["purpose"] or seal.get("candidate_id") != policy["candidate_id"]
            or seal.get("source_plan_sha256") != policy["source_plan"]["sha256"]
            or tuple(seal.get("runs") or ()) != tuple(policy["blind_runs"])
            or seal.get("sealed_before_new_prediction_reveal") is not True
            or not str(seal.get("reviewer") or "").strip()):
        raise ValueError("A new source-bound bottom blind label seal is required")
    _verified_candidate_seal(seal.get("candidate_seal"), policy)
    _bound(seal.get("label_snapshot"), root)
    if policy["purpose"] == T3_PURPOSE:
        authorization = policy.get("reveal_authorization")
        if authorization is None:
            return None
        release = _read(_bound(authorization, root))
        if (release.get("purpose") != T3_PURPOSE
                or release.get("candidate_id") != policy["candidate_id"]
                or release.get("source_plan_sha256") != policy["source_plan"]["sha256"]
                or release.get("label_seal_sha256") != bound["sha256"]
                or release.get("may_reveal_predictions") is not True
                or not str(release.get("reviewer") or "").strip()):
            raise ValueError("T3 predictions need an exact source-bound blind reveal authorization")
        _verified_candidate_seal(release.get("candidate_seal"), policy)
    return seal


def predictions_withheld(candidate_or_child, run):
    run = str(run).casefold()
    if run not in {*BLIND_RUNS, *T3_BLIND_RUNS}:
        return False
    policy = load_policy(candidate_or_child)
    return policy is not None and run in policy["blind_runs"] and _verified_label_seal(policy) is None


def require_predictions_visible(candidate_or_child, run):
    if predictions_withheld(candidate_or_child, run):
        raise PermissionError(NOTICE)


def withheld_asset_path(candidate_or_child, run):
    """A deliberately absent asset keeps availability checks and photographs working."""
    root = policy_root(candidate_or_child)
    return (root or Path(candidate_or_child)) / "__bottom_blind_prediction_withheld__" / str(run) / "unavailable.png"


def require_bottom_mass_exclusion(candidate_base):
    """Return the mandatory mask manifest, or None only after a verified accepted check.

    An absent mask manifest fails closed. A label seal alone never approves mass.
    The caller must apply the returned per-frame source/geometry-bound masks.
    """
    # T4 releases photographic review for all runs, independently of the mass
    # restriction. Absence of a blind-run policy must not release its floor.
    path = resolve_review_path(candidate_base).resolve()
    if path.is_file():
        path = path.parent
    for candidate in (path, *tuple(path.parents)[:12]):
        if (candidate / "t4c_floor_plan.json").is_file():
            raise ValueError("T4c bottom strip remains blocked for mass; this candidate is for Studio review only")
        if (candidate / "t4b_floor_plan.json").is_file():
            raise ValueError("T4b bottom strip remains blocked for mass; this candidate is for Studio review only")
        if (candidate / "t4_floor_plan.json").is_file():
            raise ValueError("T4 bottom strip remains blocked for mass; this candidate is for Studio review only")
        if (candidate / "summary.json").is_file() and (candidate / "training_request.json").is_file():
            break
    policy = load_policy(candidate_base)
    if policy is None:
        return None
    root = Path(policy["_candidate_root"])
    accepted = policy.get("bottom_acceptance")
    if accepted is not None:
        seal = _verified_label_seal(policy)
        result = _read(_bound(accepted, root))
        if (seal is None or result.get("purpose") != policy["purpose"]
                or result.get("candidate_id") != policy["candidate_id"]
                or result.get("label_seal_sha256") != policy["label_seal"]["sha256"]
                or result.get("source_plan_sha256") != policy["source_plan"]["sha256"]
                or result.get("bottom_mass_accepted") is not True):
            raise ValueError("Bottom mass has no accepted source-bound blind check")
        _verified_candidate_seal(result.get("candidate_seal"), policy)
        return None
    bound = policy.get("bottom_mass_exclusion")
    if bound is None:
        raise ValueError("Bottom mass is excluded until its source-bound exclusion masks are attached")
    manifest = _read(_bound(bound, root))
    if (manifest.get("purpose") != policy["purpose"] or manifest.get("candidate_id") != policy["candidate_id"]
            or manifest.get("source_plan_sha256") != policy["source_plan"]["sha256"]):
        raise ValueError("Bottom mass exclusion binds another candidate or source plan")
    return {**manifest, "_candidate_root": str(root), "_manifest_binding": dict(bound)}
