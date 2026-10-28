"""External Studio publication holds, separate from immutable numerical candidates.

T4c inherits the still-pending BASE blind hold. An explicit, candidate-bound
human authorization can release P1 for display without changing that history.
Old sessions and reveal markers are never read. Other candidates retain their
existing candidate-owned policy semantics.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path

from . import phase_rig_bottom_review_policy as original
from .project import sha256_file
from .review_storage import resolve_review_path

CONTRACT = "ffac-t4c-studio-display-hold-v1"
RELEASE_CONTRACT = "ffac-t4c-p1-studio-display-release-v1"
NOTICE = ("AC30, AC49 and AC54 predictions remain withheld pending an explicit "
          "source-bound blind publication release. Photographs remain available. "
          "The bottom strip remains blocked for mass.")
RELEASE_NOTICE = ("All 43 runs are available for Studio review under an explicit recorded "
                  "display release. The bottom strip remains blocked for mass.")
BLIND_CHECK = "B-T2"
_KNOWN_T4C_ROOTS = {}
_DISPLAY_POLICY_PATHS = {}


def _require(value, message):
    if not value:
        raise ValueError(message)


def explicit_authorization(authorized_by, scored_on, no_blind_check_pending):
    """True when a release names who authorized it, the scored date and no pending check.

    Who authorized the release is recorded in it; it is never compared with a
    fixed name.
    """
    try:
        scored = isinstance(scored_on, str) and date.fromisoformat(scored_on).isoformat() == scored_on
    except ValueError:
        scored = False
    return (isinstance(authorized_by, str) and bool(authorized_by.strip()) and scored
            and no_blind_check_pending is True)


def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "Studio display policy metadata must be an object")
    return value


def _signature(path):
    from .phase_saved_review_metadata import metadata_file_stamp
    return metadata_file_stamp(path)


@lru_cache(maxsize=128)
def _read_cached(path_text, signature):
    return _read(path_text)


@lru_cache(maxsize=256)
def _verified_sha(path_text, expected_sha, signature):
    actual = sha256_file(Path(path_text))
    _require(expected_sha is None or actual == expected_sha, "Studio display hold bound asset changed")
    return actual


def _asset(path, expected_sha=None):
    path = resolve_review_path(path).resolve()
    signature = _signature(path)
    return dict(path=str(path), sha256=_verified_sha(str(path), expected_sha, signature))


def _bound(asset):
    _require(isinstance(asset, dict) and isinstance(asset.get("path"), str)
             and isinstance(asset.get("sha256"), str), "Studio display hold lacks an exact asset")
    path = resolve_review_path(asset["path"]).resolve()
    _asset(path, asset["sha256"])
    return path


def _registration_path(model_id, registry_dir=None):
    from .phase_saved_registration import registration_path
    path = (Path(registry_dir) / f"{model_id}.json" if registry_dir is not None
            else registration_path(model_id))
    return resolve_review_path(path).resolve()


def display_hold_path(model_id, *, registry_dir=None):
    _require(isinstance(model_id, str) and model_id not in {"", ".", ".."}
             and "/" not in model_id and "\\" not in model_id,
             "Studio display hold has an invalid candidate ID")
    return _registration_path(model_id, registry_dir).parent / "studio_display_holds" / f"{model_id}.json"


def display_release_path(model_id, *, registry_dir=None):
    return display_hold_path(model_id, registry_dir=registry_dir).parent.parent / "studio_display_releases" / f"{model_id}.json"


def _policy_paths(root, registry_dir):
    key = str(root), str(registry_dir) if registry_dir is not None else None
    if key not in _DISPLAY_POLICY_PATHS:
        hold = display_hold_path(root.name, registry_dir=registry_dir)
        _DISPLAY_POLICY_PATHS[key] = (hold, hold.parent.parent / "studio_display_releases" / hold.name)
    return _DISPLAY_POLICY_PATHS[key]


def _path_key(path):
    """A lexical cache key: no remote stat or alias-file read is needed."""
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _known_root(path):
    path = Path(path)
    for parent in (path, *tuple(path.parents)[:12]):
        known = _KNOWN_T4C_ROOTS.get(_path_key(parent))
        if known is not None:
            return known
    return None


def _t4c_root(candidate_or_child):
    known = _known_root(candidate_or_child)
    if known is not None:
        return known
    path = resolve_review_path(candidate_or_child).resolve()
    known = _known_root(path)
    if known is not None:
        _KNOWN_T4C_ROOTS[_path_key(candidate_or_child)] = known
        return known
    if path.is_file():
        path = path.parent
    for root in (path, *tuple(path.parents)[:12]):
        if any((root / name).is_file() for name in ("p1_final3_plan.json", "p1_final2_plan.json", "t4c_floor_plan.json")):
            # Positive discoveries cannot disappear during NAS outages and
            # reveal a cached prediction. Unknown paths are never memoized.
            for key_path in (candidate_or_child, path, root):
                _KNOWN_T4C_ROOTS[_path_key(key_path)] = root
            return root
        if (root / "summary.json").is_file() and (root / "training_request.json").is_file():
            break
    return None


def _candidate_graph(root, registry_dir=None):
    """Verify only source-bound metadata; never load labels or prediction assets."""
    from .phase_saved_floor_review import t4c_saved_review
    checked = t4c_saved_review(root)
    identity = checked["model_identity"]
    placement = checked["placement"]
    _require(identity[0] == root.name and placement in ("P1", "P2", "P3"),
             "Studio display hold requires an exact T4c placement")
    registration_path = _registration_path(root.name, registry_dir)
    registration = _read(registration_path)
    _require(registration == checked["registration"], "Studio display hold selects another registration")
    plan = _asset(root / "t4c_floor_plan.json", registration.get("source_plan_sha256"))
    seal_asset = _asset(root / "rig_camera_candidate_receipt.json", registration.get("production_receipt_sha256"))
    seal = _read(seal_asset["path"])
    _require(registration.get("source_plan_sha256") == plan["sha256"]
             and registration.get("production_receipt_sha256") == seal_asset["sha256"]
             and seal.get("model_identity") == identity and seal.get("placement") == placement
             and seal.get("complete") is True and (seal.get("run_count"), seal.get("frame_count")) == (43, 864),
             "Studio display hold candidate seal or plan changed")
    parent_seal_asset = seal.get("parent_seal")
    parent_seal_path = _bound(parent_seal_asset)
    _require(parent_seal_path.name == "rig_camera_candidate_receipt.json",
             "Studio display hold requires the exact BASE seal")
    parent_root = parent_seal_path.parent
    parent_seal = _read(parent_seal_path)
    parent_identity = parent_seal.get("model_identity")
    _require(isinstance(parent_identity, list) and len(parent_identity) == 2
             and parent_identity[0] == parent_root.name and parent_seal.get("complete") is True
             and (parent_seal.get("run_count"), parent_seal.get("frame_count")) == (43, 864)
             and parent_seal.get("bottom_mass_allowed") is False,
             "Studio display hold BASE seal identity changed")
    parent_registration_path = _registration_path(parent_root.name, registry_dir)
    parent_registration = _read(parent_registration_path)
    if parent_registration.get("registration_kind") == "approved_g3_old_domain_seam_revision":
        # The frozen BASE seam registration predates metadata_sha256 and binds
        # this exact policy in its own named field. Never fall back for that kind.
        expected_policy_sha = parent_registration.get("bottom_policy_sha256")
    else:
        expected_policy_sha = (parent_registration.get("metadata_sha256") or {}).get(original.POLICY_FILE)
    _require(isinstance(expected_policy_sha, str) and len(expected_policy_sha) == 64,
             "Studio display hold BASE registration lost its exact policy SHA")
    parent_policy_asset = _asset(parent_root / original.POLICY_FILE, expected_policy_sha)
    _require(parent_registration.get("model_id") == parent_identity[0]
             and parent_registration.get("model_sha256") == parent_identity[1]
             and parent_registration.get("production_receipt_sha256") == parent_seal_asset["sha256"]
             and expected_policy_sha == parent_policy_asset["sha256"],
             "Studio display hold BASE registration or policy binding changed")
    policy = _read(parent_policy_asset["path"])
    _require(policy.get("contract") == original.CONTRACT and policy.get("purpose") == original.PURPOSE
             and policy.get("candidate_id") == parent_root.name
             and tuple(policy.get("blind_runs") or ()) == original.BLIND_RUNS
             and policy.get("label_seal") is None and policy.get("bottom_acceptance") is None
             and policy.get("no_mass_until_blind_bottom_check") is True
             and policy.get("old_blind_reveal_does_not_release") is True,
             "Studio display hold requires the still-pending BASE blind policy")
    return dict(candidate_root=str(root), model_identity=list(identity), placement=placement,
        candidate=dict(plan=plan, seal=seal_asset, registration=_asset(registration_path)),
        base=dict(root=str(parent_root), model_identity=parent_identity, seal=dict(parent_seal_asset),
                  registration=_asset(parent_registration_path), pending_policy=parent_policy_asset))


def _validate_hold_restrictions(value):
    _require(value.get("contract") == CONTRACT and value.get("status") == "pending"
             and tuple(value.get("held_runs") or ()) == original.BLIND_RUNS
             and value.get("release_authorization") is None
             and value.get("photographs_visible") is True
             and value.get("old_release_markers_ignored") is True
             and value.get("mass_analysis_allowed") is False
             and value.get("saved_artifacts_changed") is False
             and value.get("training_performed") is False and value.get("annotations_read") is False,
             "Studio display hold restrictions changed")


def validate_t4c_display_hold(candidate_or_child, *, registry_dir=None):
    root = _t4c_root(candidate_or_child)
    _require(root is not None, "Studio display hold is scoped only to T4c")
    path = display_hold_path(root.name, registry_dir=registry_dir)
    value = _read(path)
    _validate_hold_restrictions(value)
    _require(value.get("binding") == _candidate_graph(root, registry_dir),
             "Studio display hold differs from its exact candidate or pending BASE policy")
    return {**value, "policy_asset": _asset(path)}


def write_t4c_pending_display_hold(candidate_root, *, registry_dir=None):
    """Create an immutable external receipt; never write inside a saved candidate."""
    root = _t4c_root(candidate_root)
    _require(root is not None, "Studio display hold is scoped only to T4c")
    graph = _candidate_graph(root, registry_dir)
    path = display_hold_path(root.name, registry_dir=registry_dir)
    if path.exists():
        validate_t4c_display_hold(root, registry_dir=registry_dir)
        return path
    value = dict(contract=CONTRACT, status="pending", binding=graph,
        held_runs=list(original.BLIND_RUNS), photographs_visible=True,
        release_authorization=None, old_release_markers_ignored=True,
        mass_analysis_allowed=False, saved_artifacts_changed=False,
        training_performed=False, annotations_read=False, notice=NOTICE,
        created_utc=datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    validate_t4c_display_hold(root, registry_dir=registry_dir)
    return path


def _validated_release(root, registry_dir=None, *, full_binding=False):
    """Verify the release and unchanged metadata, without labels or map reads.

    Parsing and SHA work are cached by file signatures. Every access still
    checks those signatures, so deletion, replacement or NAS failure re-holds
    the predictions rather than trusting a stale positive decision.
    """
    hold_path, path = _policy_paths(root, registry_dir)
    value = _read_cached(str(path), _signature(path))
    authorization = value.get("authorization") or {}
    _require(isinstance(authorization, dict), "Studio display release authorization must be an object")
    _require(value.get("contract") == RELEASE_CONTRACT and value.get("status") == "released"
             and tuple(value.get("released_runs") or ()) == original.BLIND_RUNS
             and value.get("display_only") is True and value.get("mass_analysis_allowed") is False
             and value.get("saved_artifacts_changed") is False and value.get("default_candidate_changed") is False
             and value.get("training_performed") is False and value.get("annotations_read") is False
             and authorization.get("kind") == "explicit_user_instruction"
             and authorization.get("blind_check") == BLIND_CHECK
             and explicit_authorization(authorization.get("authorized_by"), authorization.get("scored_on"),
                                        authorization.get("no_blind_check_pending")),
             "Studio display release restrictions or authorization changed")
    binding = value.get("binding") or {}
    _require(isinstance(binding, dict), "Studio display release binding must be an object")
    identity = binding.get("model_identity") or []
    _require(binding.get("candidate_root") == str(root) and binding.get("placement") == "P1"
             and len(identity) == 2 and identity[0] == root.name,
             "Studio display release selects another candidate or placement")
    pending = value.get("pending_hold")
    _require(isinstance(pending, dict) and pending.get("path") == str(hold_path),
             "Studio display release selects another hold")
    _bound(value.get("pending_hold"))
    hold = _read_cached(str(hold_path), _signature(hold_path))
    _validate_hold_restrictions(hold)
    _require(hold.get("binding") == binding, "Studio display release differs from its source hold")
    if full_binding:
        _require(binding == _candidate_graph(root, registry_dir),
                 "Studio display release candidate binding changed")
    else:
        # The immutable hold was fully admitted at release creation. Do not
        # parse the large plan or perform numerical admission on every frame.
        for asset in binding["candidate"].values():
            _bound(asset)
        for key in ("seal", "registration", "pending_policy"):
            _bound(binding["base"][key])
    return {**value, "release_asset": _asset(path)}


def validate_t4c_p1_display_release(candidate_or_child, *, registry_dir=None):
    root = _t4c_root(candidate_or_child)
    _require(root is not None, "Studio display release is scoped only to T4c P1")
    return _validated_release(root, registry_dir, full_binding=True)


def write_t4c_p1_display_release(candidate_root, *, authorized_by, scored_on,
                                no_blind_check_pending, registry_dir=None):
    """Record explicit human release externally; preserve the pending receipt.

    This is publication permission only. It does not read blind labels or
    authorize mass analysis, scientific edits, or a default-candidate change.
    """
    _require(explicit_authorization(authorized_by, scored_on, no_blind_check_pending),
             "Explicit P1 display authorization is required")
    root = _t4c_root(candidate_root)
    _require(root is not None, "Studio display release is scoped only to T4c P1")
    hold = validate_t4c_display_hold(root, registry_dir=registry_dir)
    _require(hold["binding"]["placement"] == "P1", "Only T4c P1 is authorized for this display release")
    path = display_release_path(root.name, registry_dir=registry_dir)
    if path.exists():
        validate_t4c_p1_display_release(root, registry_dir=registry_dir)
        return path
    value = dict(contract=RELEASE_CONTRACT, status="released", binding=hold["binding"],
        pending_hold=hold["policy_asset"], released_runs=list(original.BLIND_RUNS),
        authorization=dict(kind="explicit_user_instruction", authorized_by=authorized_by,
                           blind_check=BLIND_CHECK, scored_on=scored_on,
                           no_blind_check_pending=no_blind_check_pending),
        display_only=True, mass_analysis_allowed=False, saved_artifacts_changed=False,
        default_candidate_changed=False, training_performed=False, annotations_read=False,
        notice=RELEASE_NOTICE, created_utc=datetime.now(timezone.utc).isoformat())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    validate_t4c_p1_display_release(root, registry_dir=registry_dir)
    return path


_KNOWN_P1_FINAL_ROOTS = {}


def _p1_final(root):
    if root is None:
        return False
    key = _path_key(root)
    if key in _KNOWN_P1_FINAL_ROOTS:
        return _KNOWN_P1_FINAL_ROOTS[key]
    # This is a sealed candidate's immutable revision kind, not an absent
    # candidate discovery result. Publication always creates a new root.
    final = (root / "p1_final_plan.json").is_file()
    _KNOWN_P1_FINAL_ROOTS[key] = final
    return final


def predictions_withheld(candidate_or_child, run):
    root = _t4c_root(candidate_or_child)
    if root is None:
        return original.predictions_withheld(candidate_or_child, run)
    if str(run).casefold() not in original.BLIND_RUNS:
        return False
    try:
        if (root / "p1_final3_plan.json").is_file():
            from .phase_saved_contact_review import p1_final3_saved_review
            p1_final3_saved_review(root)
        elif (root / "p1_final2_plan.json").is_file():
            from .phase_saved_categorical_review import p1_final2_saved_review
            p1_final2_saved_review(root)
        elif _p1_final(root):
            from .phase_saved_boundary_review import p1_final_saved_review
            p1_final_saved_review(root)
        else:
            _validated_release(root)
    except (OSError, ValueError, KeyError, TypeError):
        return True
    return False


def require_predictions_visible(candidate_or_child, run):
    if predictions_withheld(candidate_or_child, run):
        raise PermissionError(NOTICE if _t4c_root(candidate_or_child) else original.NOTICE)


def withheld_asset_path(candidate_or_child, run):
    root = _t4c_root(candidate_or_child)
    if root is None:
        return original.withheld_asset_path(candidate_or_child, run)
    return root / "__studio_display_prediction_withheld__" / str(run).casefold() / "unavailable.png"


def effective_publication_warning(candidate_or_child, producer_warning):
    """Correct display wording while retaining the immutable producer warning."""
    root = _t4c_root(candidate_or_child)
    if root is None:
        return producer_warning
    if (root / "p1_final3_plan.json").is_file():
        from .phase_saved_contact_review import p1_final3_saved_review
        try:
            p1_final3_saved_review(root)
            return producer_warning
        except (OSError, ValueError, KeyError, TypeError):
            return f"{producer_warning} {NOTICE}".strip()
    if (root / "p1_final2_plan.json").is_file():
        from .phase_saved_categorical_review import p1_final2_saved_review
        try:
            p1_final2_saved_review(root)
            return producer_warning
        except (OSError, ValueError, KeyError, TypeError):
            return f"{producer_warning} {NOTICE}".strip()
    if _p1_final(root):
        from .phase_saved_boundary_review import p1_final_saved_review
        try:
            p1_final_saved_review(root)
            return producer_warning
        except (OSError, ValueError, KeyError, TypeError):
            return f"{producer_warning} {NOTICE}".strip()
    warning = str(producer_warning).replace("All runs visible;", "").strip()
    released = all(not predictions_withheld(root, run) for run in original.BLIND_RUNS)
    return f"{warning} {RELEASE_NOTICE if released else NOTICE}".strip()


__all__ = ["CONTRACT", "RELEASE_CONTRACT", "NOTICE", "RELEASE_NOTICE", "BLIND_CHECK",
           "explicit_authorization", "display_hold_path", "display_release_path", "validate_t4c_display_hold",
           "validate_t4c_p1_display_release", "write_t4c_p1_display_release",
           "write_t4c_pending_display_hold", "predictions_withheld", "require_predictions_visible",
           "withheld_asset_path", "effective_publication_warning"]
