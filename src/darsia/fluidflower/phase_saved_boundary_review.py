"""Read-only, exact admission of the P1-final saved post-processing revision."""
from functools import lru_cache
import json
from pathlib import Path

from .project import sha256_file
from .review_storage import resolve_review_path
from .phase_saved_review_metadata import (canonical_sha256, configured_parameters,
    configured_export, configured_model_identity as _model_identity,
    configured_registration_path as registration_path)

_METADATA_EXPORTS = ('CONTRACT', 'FILES', 'KIND', 'PLAN', 'QA', 'SEAL', 'SPEC_SHA256', 'STATE_LABEL', 'WARNING')


def __getattr__(name):
    return configured_export('boundary', name, allowed_names=_METADATA_EXPORTS)




def _read(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    _require(isinstance(value, dict), "P1-final metadata must be an object")
    return value


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _signature(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


@lru_cache(maxsize=1024)
def _resolved_path(path_text, base_text=None):
    """Resolve an immutable descriptor once, without caching its current stamp.

    Network Path.resolve() walks the remote ancestors. The descriptor belongs
    to a frozen candidate, while every subsequent access still stats the final
    file and invalidates the SHA cache if its metadata changes or it disappears.
    """
    path = resolve_review_path(path_text)
    if not path.is_absolute() and base_text is not None:
        path = Path(base_text) / path
    return path.resolve()


@lru_cache(maxsize=1024)
def _checked_sha(path_text, expected, stamp):
    _require(isinstance(expected, str) and len(expected) == 64 and sha256_file(path_text) == expected,
             "P1-final bound metadata changed")
    return path_text


def _bound(asset, *, base=None):
    _require(isinstance(asset, dict) and isinstance(asset.get("path"), str), "P1-final bound asset is absent")
    path = _resolved_path(asset["path"], str(base) if base is not None else None)
    _checked_sha(str(path), asset.get("sha256"), _signature(path))
    return path


def _stamped_dependencies(root, registration, files, assets, *, plan_path, plan_stamp):
    """Stat each unique admitted dependency once per check.

    Bound descriptors still require their SHA before admission. Reusing the
    same fresh signature for both that check and the admission key avoids a
    second network stat of every head and input. Model identity heads belong
    to the key even when they are not repeated in the plan's head inventory.
    """
    dependencies = []
    expected = {}
    for asset in assets:
        _require(isinstance(asset, dict) and isinstance(asset.get("path"), str),
                 "P1-final bound asset is absent")
        path = _resolved_path(asset["path"], str(root))
        dependencies.append(path)
        expected.setdefault(path, []).append(asset.get("sha256"))
    paths = dict.fromkeys((registration, *(root / name for name in files),
                          root / "models/generic/co2_presence_signal.json",
                          root / "models/generic/gas_within_co2_signal.json", *dependencies))
    stamps = {path: plan_stamp if path == plan_path else _signature(path) for path in paths}
    for path, digests in expected.items():
        for digest in digests:
            _checked_sha(str(path), digest, stamps[path])
    return dependencies, tuple((str(path), stamp) for path, stamp in stamps.items())


@lru_cache(maxsize=8)
def _parent_admission(parent_text, registration_sha, seal_sha, stamps):
    from .phase_saved_floor_review import t4c_saved_review
    return t4c_saved_review(Path(parent_text))


def _head_assets(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            yield value
        else:
            for child in value.values():
                yield from _head_assets(child)


@lru_cache(maxsize=16)
def _saved_dependencies(root_text, plan_stamp):
    PLAN = configured_export('boundary', 'PLAN', allowed_names=_METADATA_EXPORTS)
    plan = _read(Path(root_text) / PLAN)
    inputs = plan["binding"]["inputs"]
    return (plan, tuple(inputs[key] for key in ("parent_seal", "parent_registration", "base_seal", "spec")),
            tuple(_head_assets(plan["binding"]["heads"])))


def _validate(root, registration, *, check_parent=True):
    CONTRACT = configured_export('boundary', 'CONTRACT', allowed_names=_METADATA_EXPORTS)
    FILES = configured_export('boundary', 'FILES', allowed_names=_METADATA_EXPORTS)
    KIND = configured_export('boundary', 'KIND', allowed_names=_METADATA_EXPORTS)
    PLAN = configured_export('boundary', 'PLAN', allowed_names=_METADATA_EXPORTS)
    QA = configured_export('boundary', 'QA', allowed_names=_METADATA_EXPORTS)
    SEAL = configured_export('boundary', 'SEAL', allowed_names=_METADATA_EXPORTS)
    SPEC_SHA256 = configured_export('boundary', 'SPEC_SHA256', allowed_names=_METADATA_EXPORTS)
    STATE_LABEL = configured_export('boundary', 'STATE_LABEL', allowed_names=_METADATA_EXPORTS)
    RULE = configured_parameters('boundary')["rules"]
    root = Path(root)
    identity = list(_model_identity(root))
    _require(registration.get("registration_kind") == KIND
        and resolve_review_path(registration.get("model_root", "")).resolve() == root.resolve()
        and [registration.get("model_id"), registration.get("model_sha256")] == identity
        and registration.get("publication_scope") == "studio_validation_only"
        and (registration.get("run_count"), registration.get("frame_count")) == (43, 864)
        and all(registration.get(key) is False for key in ("mass_eligible", "bottom_mass_allowed",
            "scientific_acceptance", "deployment_allowed", "automatic_activation")),
        "P1-final registration changed its identity or review-only restrictions")
    metadata = registration.get("metadata_sha256") or {}
    _require(set(metadata) == set(FILES), "P1-final registration lost its complete metadata inventory")
    for relative, expected in metadata.items():
        _checked_sha(str(root / relative), expected, _signature(root / relative))
    _require(metadata[PLAN] == registration.get("source_plan_sha256")
        and metadata[SEAL] == registration.get("production_receipt_sha256"), "P1-final plan/seal registration differs")
    plan, seal, qa = _read(root / PLAN), _read(root / SEAL), _read(root / QA)
    _require(plan.get("contract") == seal.get("contract") == qa.get("contract") == CONTRACT
        and plan.get("model_identity") == seal.get("model_identity") == identity
        and seal.get("complete") is True and qa.get("technical_all_pass") is True
        and (seal.get("run_count"), seal.get("frame_count")) == (43, 864)
        and (qa.get("run_count"), qa.get("frame_count")) == (43, 864)
        and qa.get("rules") == RULE
        and qa.get("source_time_geometry_area_heads_delta_unchanged") is True
        and seal.get("source_time_geometry_area_heads_delta_unchanged") is True
        and all(qa.get(key) is False for key in ("annotations_read", "training", "inference"))
        and seal.get("source_plan_sha256") == metadata[PLAN]
        and seal.get("pipeline_sha256") == plan.get("pipeline_sha256")
        and plan.get("job_sha256") == canonical_sha256(plan["binding"])
        and plan.get("pipeline_sha256") == canonical_sha256(dict(binding=plan["binding"], model_identity=identity)),
        "P1-final saved labels differ from their exact complete seal")
    inputs = plan["binding"]["inputs"]
    _require((inputs.get("spec") or {}).get("sha256") == SPEC_SHA256
        and inputs.get("rules") == RULE
        and plan.get("placement") == "P1-final"
        and all(inputs.get(key) is False for key in ("training", "inference", "annotations_read", "mass_eligible", "bottom_mass_allowed"))
        and all(seal.get(key) is False for key in ("training_performed", "inference_performed", "labels_changed",
            "mass_eligible", "bottom_mass_allowed", "scientific_acceptance", "automatic_activation"))
        and seal.get("blind_hold_runs") == [], "P1-final changed its post-processing-only or visibility scope")
    _require(_bound(seal.get("qa"), base=root) == (root / QA).resolve(), "P1-final QA binding differs")
    parent = resolve_review_path(plan["binding"]["parent"]).resolve()
    _require(_bound(inputs["parent_seal"], base=root) == (parent / SEAL).resolve()
        and seal.get("parent_seal") == inputs["parent_seal"], "P1-final lost its exact P1 parent seal")
    _bound(inputs["parent_registration"], base=root)
    _bound(inputs["spec"], base=root)
    _bound(inputs["base_seal"], base=root)
    parent_seal = _read(parent / SEAL)
    parent_registration = _read(_bound(inputs["parent_registration"], base=root))
    _require(parent_seal.get("placement") == "P1" and plan.get("delta") == parent_seal.get("delta")
        and plan["binding"]["heads"] == parent_seal.get("heads") == seal.get("heads")
        and metadata["t4c_floor_plan.json"] == parent_registration.get("source_plan_sha256"),
        "P1-final changed P1 heads, shift, sources or inherited bottom-mass marker")
    if check_parent:
        stamps = tuple(_signature(p) for p in (parent / SEAL, parent / "t4c_floor_plan.json",
            _bound(inputs["parent_registration"], base=root)))
        admitted = _parent_admission(str(parent), inputs["parent_registration"]["sha256"], inputs["parent_seal"]["sha256"], stamps)
        _require(admitted["placement"] == "P1", "P1-final parent is not the admitted optical P1")
        _require(_bound(inputs["parent_registration"], base=root) == resolve_review_path(registration_path(parent.name)).resolve(),
            "P1-final parent registration binds another candidate")
    frames = seal.get("frames") or []
    inventory = plan.get("source_inventory") or []
    qa_rows = qa.get("rows") or []
    _require(len(frames) == len({(f.get("run"), f.get("frame_index")) for f in frames}) == 864
        and len({f.get("run") for f in frames}) == 43
        and len(inventory) == 864
        and len(qa_rows) == 864 and all(row.get("all_pass") is True for row in qa_rows)
        and {(f.get("run"), f.get("frame_index")) for f in qa_rows}
            == {(f.get("run"), f.get("frame_index")) for f in inventory}
        and {(f.get("run"), f.get("frame_index")) for f in frames}
            == {(f.get("run"), f.get("frame_index")) for f in inventory}
        and all(isinstance(f.get("source_sha256"), str) and len(f["source_sha256"]) == 64
            and isinstance(f.get("elapsed_h"), (int, float)) for f in inventory),
        "P1-final seal lacks 43 runs / 864 unique source-owned frames")
    profile = _read(root / "phase_rig_camera.json")
    marker = profile.get("p1_final_revision") or {}
    _require(marker.get("contract") == CONTRACT and marker.get("plan", {}).get("sha256") == metadata[PLAN]
        and not any(key in profile for key in ("g3_t4c_floor_revision", "g3_t4b_floor_revision", "g3_t4_floor_revision",
            "g3_t3_floor_revision", "g3_t2_floor_revision", "g3_t2_l1zone_revision", "l1_neighbourhood_floor_revision")),
        "P1-final profile mixed an active floor rule or lost its saved revision")
    return dict(registration=registration, pipeline_sha256=plan["pipeline_sha256"], model_identity=identity,
                placement="P1-final", delta=plan["delta"], blind_hold_runs=[], state_label=STATE_LABEL)


@lru_cache(maxsize=16)
def _admitted(root_text, registration_text, stamps):
    return _validate(Path(root_text), _read(registration_text))


def p1_final_saved_review(model_root):
    FILES = configured_export('boundary', 'FILES', allowed_names=_METADATA_EXPORTS)
    PLAN = configured_export('boundary', 'PLAN', allowed_names=_METADATA_EXPORTS)
    SEAL = configured_export('boundary', 'SEAL', allowed_names=_METADATA_EXPORTS)
    root = _resolved_path(str(model_root))
    path = _resolved_path(str(registration_path(root.name)))
    plan_stamp = _signature(root / PLAN)
    plan, metadata_assets, heads = _saved_dependencies(str(root), plan_stamp)
    dependencies, dependency_stamps = _stamped_dependencies(root, path, FILES,
        (*metadata_assets, *heads), plan_path=root / PLAN, plan_stamp=plan_stamp)
    parent = _resolved_path(plan["binding"]["parent"])
    # Parent plan is large. Its immutable sealed admission runs once, and is
    # invalidated by the parent's plan/seal/registration stamps.
    known_stamps = dict(dependency_stamps)
    parent_stamps = tuple(known_stamps[str(p)] if str(p) in known_stamps else _signature(p)
        for p in (parent / SEAL, parent / "t4c_floor_plan.json", dependencies[1]))
    parent_admission = _parent_admission(str(parent), plan["binding"]["inputs"]["parent_registration"]["sha256"],
        plan["binding"]["inputs"]["parent_seal"]["sha256"], parent_stamps)
    stamps = (tuple(parent_admission["model_identity"]), parent_admission["pipeline_sha256"],
        dependency_stamps)
    return _admitted(str(root), str(path), stamps)


def registered_warning(root, registration):
    WARNING = configured_export('boundary', 'WARNING', allowed_names=_METADATA_EXPORTS)
    try:
        _validate(resolve_review_path(root).resolve(), registration, check_parent=False)
        return WARNING
    except (OSError, KeyError, ValueError, TypeError):
        return "QA FAILED: P1-final saved registration changed or is incomplete; do not use for mass."


__all__ = ["p1_final_saved_review", "registered_warning", "CONTRACT", "KIND", "PLAN", "SEAL", "QA", "FILES", "SPEC_SHA256", "STATE_LABEL", "WARNING"]
