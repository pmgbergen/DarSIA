"""Read-only source/grid admission for the explicit painting-table branch.

Measured homographies are used only after joining their actual photograph,
profile, saved model and renderer certificates. Existing corrected NPZ payloads
have no renderer-output certificate in the admitted schema and remain refused.
This module does not infer that missing link or add a new certificate format.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

FORMAT = "darsia-painted-adoption-geometry-proof-v1"


from .corrected_setup import _identity


def _require_corrected_payload_link(config, run, model, renderer):
    """The deliberately closed admission seam for an old corrected payload.

    RAW, a renderer homography and equal dimensions do not establish identity
    with the saved corrected-baseline NPZ. Route 2 may supply an independently
    reviewed build certificate later; no such format is admitted here today.
    """
    raise ValueError(f"{run}: no admitted renderer-to-saved corrected-baseline payload link; "
                     "a matching grid or RAW photograph is insufficient")


def _saved_frame(document, run):
    """Read the one exact saved source-model frame for this run."""
    if isinstance(document, dict) and document.get("run") == run and "conditional_frame" in document:
        if isinstance(document["conditional_frame"], dict):
            return document["conditional_frame"]
        raise ValueError(f"{run}: the saved source-model frame is not a document")
    rows = document.get("runs") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{run}: the measured geometry lacks its exact saved source-model frame")
    found = [row for row in rows if isinstance(row, dict) and row.get("run") == run]
    if len(found) != 1 or not isinstance(found[0].get("conditional_frame"), dict):
        raise ValueError(f"{run}: the measured geometry lacks its exact saved source-model frame")
    return found[0]["conditional_frame"]


def _profile(model, run, context, checked, *, source_run):
    """Join the measured camera profile to its original configured run assignment."""
    from .image_formats import image_family
    profile = model.get("camera_profile")
    if (not isinstance(profile, dict) or not isinstance(profile.get("camera_wh"), list)
            or any(type(value) is not int or value <= 0 for value in profile["camera_wh"])
            or profile["camera_wh"] != list(reversed(model["shape_hw"]))):
        raise ValueError(f"{run}: the measured camera profile has another grid")
    family = image_family(Path(model["reference_image"]["path"]))
    if family not in {"raw", "processed"}:
        raise ValueError(f"{run}: a camera-U model needs an actual RAW or processed photograph")
    raw = family == "raw"
    if not raw:
        if set(profile) != {"camera_wh"}:
            raise ValueError(f"{run}: plain pixels cannot borrow an unproved camera profile")
        return []
    required = {"path", "sha256", "size_bytes", "profile_id", "camera_wh"}
    if (set(profile) != required or context.get("kind") != "camera_u"
            or not isinstance(profile["profile_id"], str) or not profile["profile_id"]):
        raise ValueError(f"{run}: RAW needs an exact configured camera-U profile proof")
    if run == source_run and context.get("profile_id") != profile["profile_id"]:
        raise ValueError(f"{run}: the measured profile differs from the approved session's configured profile")
    declared = context["parameters"]
    actual = checked(_identity(profile))
    if any(actual[key] != declared[key] for key in ("sha256", "size_bytes")):
        raise ValueError(f"{run}: measured and configured camera-U parameter identities differ")
    document = checked(declared, metadata=True)
    if not isinstance(document, dict):
        raise ValueError(f"{run}: camera-U parameters must be their bound document")
    registration = document.get("registration", document)
    source = document.get("source") or {}
    if not isinstance(registration, dict) or not isinstance(source, dict):
        raise ValueError(f"{run}: camera-U parameter assignments are malformed")
    common = registration.get("common") or {}
    profiles = source.get("profiles") or {}
    if (not isinstance(common, dict) or not isinstance(profiles, dict)
            or not isinstance(common.get("profiles"), dict)
            or not isinstance(common.get("sensor"), dict)):
        raise ValueError(f"{run}: camera-U parameters lack explicit sensor and profile assignments")
    assignments = [value for key, value in profiles.items() if str(key).casefold() == run]
    if len(assignments) != 1:
        raise ValueError(f"{run}: camera-U parameters need one unambiguous run assignment")
    selected = assignments[0]
    if (selected != profile["profile_id"] or selected not in (common.get("profiles") or {})
            or (common.get("sensor") or {}).get("camera_wh") != profile["camera_wh"]):
        raise ValueError(f"{run}: the measured camera profile differs from its bound run assignment")
    return [_identity(profile), declared]


def validate_models(binding, model_bindings, *, configuration_copies=None, folder=None,
                    basis_sources=None, basis_checks=None, basis_metadata_replay=None,
                    basis_photo_relocations=None):
    """Return consumed file guards after exact compositional source admission.

    The explicit RAW-basis branch reads saved NPZ and parent-grid metadata; it
    does not decode RAW or refit registration. The caller first verifies the
    saved map closure and explicit preparation-frame basis.
    """
    from .painted_adoption_table import _checked, _same, _local
    from . import geometry_workspace_frame as workspace
    from . import round5_geometry as geometry
    from .map_placement import GEOMETRY_FORMAT

    from .raw_shape_basis_reader import has_shape_basis
    if not isinstance(model_bindings, dict):
        raise ValueError("The painting preparation needs its exact measured run models")
    documents = {run: _checked(bound, metadata=True) for run, bound in model_bindings.items()}
    if any(has_shape_basis(model) for model in documents.values()):
        return _validate_basis_models(binding, model_bindings, documents,
            configuration_copies=configuration_copies, folder=folder,
            basis_sources=basis_sources, basis_checks=basis_checks,
            basis_metadata_replay=basis_metadata_replay, basis_photo_relocations=basis_photo_relocations)

    basis = binding["frame_basis"]
    current = _checked(basis["workspace_revision"], metadata=True)
    config = current.get("configuration") or {}
    run = str(binding["record"]["run_id"]).casefold()
    if not isinstance(model_bindings, dict) or run not in model_bindings:
        raise ValueError("The approved session lacks its exact measured reference model")
    photograph = _checked(basis["photograph"])
    if workspace.frame_kind(config) == "rig-corrected":
        fields = workspace.selected_fields(config, run, check_files=True)
        if Path(fields["shape_corrected_baseline"]["path"]).resolve() == Path(photograph["path"]).resolve():
            raise ValueError("The saved corrected-baseline SOURCE has no admitted renderer payload link; "
                             "it cannot be treated as camera-U pixels")
    from .image_formats import image_family
    if image_family(photograph["path"]) not in {"raw", "processed"}:
        raise ValueError("The preparation SOURCE needs an actual RAW or processed photograph, not an unbound raster grid")
    guarded = [basis["workspace_revision"], basis["photograph"]]
    for target_run, bound in model_bindings.items():
        if (not isinstance(target_run, str) or not target_run
                or target_run != target_run.casefold()
                or not all(char.isalnum() or char in "-_" for char in target_run)):
            raise ValueError("Measured run identities must be explicit safe names")
        model = _checked(bound, metadata=True)
        if (not isinstance(model, dict) or model.get("format") != GEOMETRY_FORMAT
                or model.get("run_id") != target_run or model.get("run") != target_run
                or model.get("grid") != "camera_u"
                or not isinstance(model.get("shape_hw"), list) or len(model["shape_hw"]) != 2
                or any(type(value) is not int or value <= 0 for value in model["shape_hw"])):
            raise ValueError(f"{target_run}: explicit measured camera-grid model identity required")
        reference = _checked(model["reference_image"])
        model_photo = _checked(model["model_photograph"])
        if target_run == run:
            if (not _same(reference, photograph)
                    or model["shape_hw"] != list(reversed(binding["frame"]["image_size"]))):
                raise ValueError("The measured reference model belongs to another approved-session photograph or grid")
        guarded.extend(_profile(model, target_run, basis["camera_profile"], _checked, source_run=run))
        saved = _checked(model["source_model"], metadata=True)
        frame = _saved_frame(saved, target_run)
        _checked(model["current_edges"])
        baseline = _identity(model["baseline_configuration"])
        row = ((config.get("frame_binding") or {}).get("runs") or {}).get(target_run)
        if not isinstance(row, dict) or not _same(baseline, _identity(row["configuration"])):
            raise ValueError(f"{target_run}: the measured baseline configuration is not the workspace's bound run")
        if configuration_copies is None:
            _checked(baseline)
            configuration_guard = baseline
        else:
            copied = configuration_copies.get(target_run)
            if (not isinstance(copied, dict) or set(copied) != {"original", "copy"}
                    or not _same(copied["original"], baseline)):
                raise ValueError(f"{target_run}: the original configuration closure changed")
            configuration_guard = _local(folder, copied["copy"])
            if any(configuration_guard[key] != baseline[key] for key in ("sha256", "size_bytes")):
                raise ValueError(f"{target_run}: the immutable configuration copy changed its source bytes")
        grid = model.get("baseline_native_grid")
        if not isinstance(grid, dict) or not isinstance(grid.get("renderer"), dict):
            raise ValueError(f"{target_run}: no exact baseline-native renderer certificate")
        renderer = _checked(grid["renderer"], metadata=True)
        bridge = model.get("baseline_to_model_registration")
        if bridge is not None:
            _checked(bridge)
        # Reuse the existing builder's bridge/source/grid algebra and guards.
        # It is a pure reconstruction, not a write or a refit.
        try:
            rebuilt = geometry.reference_model(target_run, frame,
                reference_image=reference, camera_profile=deepcopy(model["camera_profile"]),
                source_model=model["source_model"], edge_review=model["current_edges"],
                baseline_configuration=baseline, model_photograph=model_photo,
                baseline_to_model=bridge, baseline_native_renderer=grid["renderer"])
        except (TypeError, AttributeError, IndexError) as error:
            raise ValueError(f"{target_run}: malformed source-model, bridge or renderer proof") from error
        for key in ("H_model_photograph_u_to_metric_mm", "H_camera_u_to_metric_mm",
                    "H_metric_mm_to_camera_u", "baseline_native_grid", "area_factor",
                    "saved_model_certificate"):
            if not _same(model.get(key), rebuilt.get(key)):
                raise ValueError(f"{target_run}: measured model/bridge/renderer composition changed ({key})")
        fields = workspace.selected_fields(config, target_run, check_files=True)
        if grid.get("shape_hw") != fields["labels"]["shape"]:
            raise ValueError(f"{target_run}: renderer and registered native grid differ")
        guarded.extend(_require_corrected_payload_link(config, target_run, model, renderer))
        guarded.extend([bound, model["reference_image"], model["model_photograph"], model["source_model"],
                        model["current_edges"], configuration_guard, grid["renderer"]])
        if bridge is not None:
            guarded.append(bridge)
    for row in guarded:
        _checked(row)
    return guarded


def make_proof(binding, model_bindings, *, basis_checks=None):
    """Freeze the actual measured-model and optional explicit RAW-basis closure."""
    checks = {}
    validate_models(binding, model_bindings, basis_checks=checks)
    if basis_checks is not None:
        basis_checks.update(checks)
    proof = dict(format=FORMAT, models=deepcopy(model_bindings), baseline_configurations={})
    if checks:
        from .painted_adoption_table import _checked
        from .raw_shape_basis_reader import _key
        pool, positions, members = [], {}, {}
        for run, checked in checks.items():
            members[run] = []
            for row in checked.sources_to_copy():
                key = (row["path"], row["sha256"], row["size_bytes"])
                if key not in positions:
                    positions[key] = len(pool)
                    pool.append(dict(original=row, copy=None))
                members[run].append(positions[key])
        proof["raw_shape_basis"] = dict(
            authorities={run: _basis_authority_summary(
                _checked(model_bindings[run], metadata=True), model_bindings[run]) for run in checks},
            source_copies=pool, source_indices=members)
        replay = {run: [row["path"] for row in checked.sources_to_copy()
                        if _key(row["path"]) in checked.guards.metadata_replay_keys]
                  for run, checked in checks.items()}
        if any(replay.values()):
            proof["raw_shape_basis"]["metadata_replay"] = replay
        relocations = {run: deepcopy(list(checked.guards.photo_relocations.values()))
                       for run, checked in checks.items()}
        if any(relocations.values()):
            proof["raw_shape_basis"]["photo_relocations"] = relocations
    return proof


def received_source_rows(proof, *, basis_checks=None, folder=None):
    """Recheck CURRENT receipted sources for NEW preparation, never its own copies.

    Received bases retain their genuine Root-bound pool. An ordinary basis may
    substitute ONLY the historical parent TOML that its strict, Root-released
    configuration-lineage audit already proved byte-identical. RAW, parent NPZ,
    producer/current configuration and ownership checks remain physical.
    Caller must revalidate the returned readers after its last source/output IO.
    """
    from .painted_adoption_table import _checked
    from .raw_shape_basis_reader import (has_shape_basis, read_model_basis, ReadShapeBasis,
                                        has_retained_source_context, _key, _same)
    rows, checks = {}, []
    if 'raw_shape_basis' not in proof:
        return rows, checks
    replay = proof['raw_shape_basis'].get('metadata_replay') or {}
    relocations = proof['raw_shape_basis'].get('photo_relocations') or {}
    replay_sources = (_basis_sources(proof, proof['models'], folder)
                      if replay and folder is not None and not has_retained_source_context() else None)
    for run, bound in proof['models'].items():
        model = _checked(bound, metadata=True)
        if not has_shape_basis(model):
            continue
        grid = model['baseline_native_grid']
        received = 'received_basis' in grid['raw_shape_basis_authority']
        lineage = (grid.get('parent_grid_compatibility') or {}).get('configuration_lineage')
        checked = None if basis_checks is None else basis_checks.get(run)
        retained = (has_retained_source_context() or bool(replay.get(run))
                    or (checked is not None and bool(checked.guards.aliases)))
        if not received and lineage is None and not retained:
            continue
        if checked is None:
            checked = (read_model_basis(model, aliases=replay_sources[run], folder=folder,
                                       metadata_replay=replay.get(run, ()),
                                       photo_relocations=relocations.get(run, ()))
                       if replay_sources is not None else read_model_basis(model))
        elif (not isinstance(checked, ReadShapeBasis)
                or not _same(checked.certificate, grid['renderer'])
                or not _same(checked.compatibility, grid['parent_grid_compatibility'])):
            raise ValueError('The preparation retained another checked RAW basis')
        if not _same(checked.guards.add(_identity(bound), metadata=True, copyable=True), model):
            raise ValueError('The checked RAW model changed before source retention')
        checked.revalidate()
        if received or retained:
            for row in checked.copy_source_rows():
                original, physical = _identity(row['original']), _identity(row['physical'])
                if not received and _key(original['path']) not in checked.guards.aliases:
                    continue
                key = (original['path'], original['sha256'], original['size_bytes'])
                if any(original[name] != physical[name] for name in ('sha256', 'size_bytes')):
                    raise ValueError('Receipted source changed its exact original byte identity')
                rows.setdefault(key, physical)
        audit = (checked.compatibility or {}).get('configuration_lineage')
        if audit is not None:
            original = _identity(audit['parent_configuration'])
            archived = _identity(audit['verified_original_configuration_copy'])
            if (not _same(original, _identity(model['baseline_configuration']))
                    or not _same(original, checked.compatibility['configuration'])
                    or Path(original['path']).suffix.casefold() != '.toml'
                    or Path(archived['path']).suffix.casefold() != '.toml'
                    or any(original[name] != archived[name] for name in ('sha256', 'size_bytes'))):
                raise ValueError('The checked lineage lost its exact historical parent TOML')
            checked.guards.add(archived, copyable=True)
            physical = _identity(checked.guards.rows[_key(archived['path'])]['physical'])
            if any(original[name] != physical[name] for name in ('sha256', 'size_bytes')):
                raise ValueError('The checked historical TOML changed its physical bytes')
            key = (original['path'], original['sha256'], original['size_bytes'])
            rows.setdefault(key, physical)
        checks.append(checked)
    return rows, checks


def write_configuration_copies(folder, proof, *, basis_checks=None):
    """Keep actual before-Apply bytes, never rewrite an original run file."""
    from .painted_adoption_table import _checked, _read, _relative
    received, checks = received_source_rows(proof, basis_checks=basis_checks)
    target = Path(folder) / "configuration_sources"
    target.mkdir(exist_ok=False)
    rows = {}
    for run, bound in proof["models"].items():
        model = _checked(bound, metadata=True)
        source = _identity(model["baseline_configuration"])
        physical = received.get((source["path"], source["sha256"], source["size_bytes"]), source)
        _checked(physical)
        raw = _read(physical["path"], 1024 * 1024)
        destination = target / (run + ".toml")
        if not destination.resolve().is_relative_to(target.resolve()):
            raise ValueError("Configuration copy leaves its new preparation")
        with destination.open("xb") as stream:
            stream.write(raw)
        copied = _relative(folder, destination)
        if any(copied[key] != source[key] for key in ("sha256", "size_bytes")):
            raise ValueError("The preparation configuration copy differs from its actual source")
        _checked(physical)
        rows[run] = dict(original=source, copy=copied)
    proof["baseline_configurations"] = rows
    if "raw_shape_basis" in proof:
        _write_basis_copies(folder, proof, received_sources=received)
    for checked in checks:
        checked.revalidate()


def verify_proof(binding, proof, expected_models, folder, *, basis_checks=None):
    """Replay the exact preparation proof, using retained metadata bytes only."""
    from .painted_adoption_table import _same
    from .local_map_adoption import PROOF_FORMAT, verify_local_geometry
    if isinstance(proof, dict) and proof.get("format") == PROOF_FORMAT:
        return verify_local_geometry(binding, proof, expected_models, folder)
    if (not isinstance(proof, dict) or set(proof) not in (
            {"format", "models", "baseline_configurations"},
            {"format", "models", "baseline_configurations", "raw_shape_basis"})
            or proof["format"] != FORMAT or not _same(proof["models"], expected_models)
            or not isinstance(proof["baseline_configurations"], dict)
            or set(proof["baseline_configurations"]) != set(expected_models)):
        raise ValueError("The prepared transport lacks its exact source/grid/model proof")
    sources = _basis_sources(proof, expected_models, folder)
    checks = {}
    guards = validate_models(binding, proof["models"],
        configuration_copies=proof["baseline_configurations"], folder=folder,
        basis_sources=sources, basis_checks=checks,
        basis_metadata_replay=(proof.get("raw_shape_basis") or {}).get("metadata_replay"),
        basis_photo_relocations=(proof.get("raw_shape_basis") or {}).get("photo_relocations"))
    if bool(checks) != ("raw_shape_basis" in proof):
        raise ValueError("The preparation changed its explicit RAW-basis branch")
    if basis_checks is not None:
        basis_checks.update(checks)
    return guards


def _basis_sources(proof, expected_models, folder):
    """Require a complete per-run historical source-copy list, never a fallback."""
    from .raw_shape_basis_reader import _Pins
    from .painted_adoption_table import _checked, _same
    value = proof.get("raw_shape_basis")
    if value is None:
        return None
    required = {"authorities", "source_copies", "source_indices"}
    if (not isinstance(value, dict) or not required <= set(value)
            or set(value) - required - {"metadata_replay", "photo_relocations"}
            or not isinstance(value["authorities"], dict)
            or not isinstance(value["source_copies"], list) or not value["source_copies"]
            or not isinstance(value["source_indices"], dict)
            or set(value["authorities"]) != set(expected_models)
            or set(value["source_indices"]) != set(expected_models)):
        raise ValueError("The preparation lost its per-run RAW-basis source copies")
    sources, used = {}, set()
    for run, bound in expected_models.items():
        model = _checked(bound, metadata=True)
        indices = value["source_indices"][run]
        if (not isinstance(indices, list) or not indices
                or any(type(index) is not int or not 0 <= index < len(value["source_copies"]) for index in indices)
                or len(set(indices)) != len(indices)
                or not _same(value["authorities"][run], _basis_authority_summary(model, bound))):
            raise ValueError("The preparation changed its exact RAW-basis operation authority")
        rows = [value["source_copies"][index] for index in indices]
        _Pins(rows, folder=folder)
        sources[run] = rows
        used.update(indices)
    if used != set(range(len(value["source_copies"]))):
        raise ValueError("The RAW-basis source pool contains an unconsumed input")
    replay = value.get("metadata_replay")
    if replay is not None:
        if not isinstance(replay, dict) or set(replay) != set(expected_models):
            raise ValueError("Historical metadata replay needs its exact per-run source selection")
        for run, paths in replay.items():
            if (not isinstance(paths, list) or any(not isinstance(path, str) for path in paths)
                    or len(paths) != len(set(paths))
                    or any(Path(path).suffix.casefold() not in {".toml", ".csv"} for path in paths)
                    or not set(paths) <= {row["original"]["path"] for row in sources[run]}):
                raise ValueError("Historical metadata replay changed its exact source copies")
    relocations = value.get("photo_relocations")
    if relocations is not None:
        from .rig_point_review import _photo_descriptor
        if not isinstance(relocations, dict) or set(relocations) != set(expected_models):
            raise ValueError("Photograph relocations need their exact per-run model bindings")
        for run, rows in relocations.items():
            model = _checked(expected_models[run], metadata=True)
            originals = [_photo_descriptor(model[key]) for key in ("reference_image", "model_photograph")]
            if (not isinstance(rows, list) or any(not isinstance(row, dict) or
                    _photo_descriptor(row.get("original", {})) not in originals for row in rows)):
                raise ValueError("Photograph relocation differs from the bound model acquisition")
    return sources


def _basis_authority_summary(model, bound):
    """Retain authority locators once without duplicating the full source list."""
    authority = model["baseline_native_grid"]["raw_shape_basis_authority"]
    summary = dict(model=deepcopy(bound), operation_request=deepcopy(authority["operation_request"]),
                   root_release=deepcopy(authority["root_release"]))
    if "full_review" in authority:
        summary["full_review"] = deepcopy(authority["full_review"])
    return summary


def _write_basis_copies(folder, proof, *, received_sources=None):
    """Exclusively retain consumed metadata/source bytes before any Apply."""
    import hashlib
    import os
    from .painted_adoption_table import _checked, _relative
    target = Path(folder) / "raw_basis_sources"
    target.mkdir(exist_ok=False)
    copied = {}
    received = {} if received_sources is None else received_sources
    for row in proof["raw_shape_basis"]["source_copies"]:
        original = _identity(row["original"])
        key = (original["path"], original["sha256"], original["size_bytes"])
        if key not in copied:
            physical = received.get(key, original)
            _checked(physical)
            destination = target / f"{len(copied):06d}.bin"
            digest, size = hashlib.sha256(), 0
            with Path(physical["path"]).open("rb") as source, destination.open("xb") as output:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    size += len(block)
                output.flush()
                os.fsync(output.fileno())
            if digest.hexdigest() != original["sha256"] or size != original["size_bytes"]:
                raise ValueError("A RAW-basis source changed while its historical copy was written")
            _checked(physical)
            copied[key] = _relative(folder, destination)
        row["copy"] = deepcopy(copied[key])


def _validate_basis_models(binding, model_bindings, documents, *, configuration_copies,
                           folder, basis_sources, basis_checks, basis_metadata_replay=None,
                           basis_photo_relocations=None):
    """Join a NEW RAW basis to this painting's exact frame and measured models."""
    from . import geometry_workspace_frame as workspace
    from .painted_adoption_table import _checked, _same, _local
    from .raw_shape_basis_reader import has_shape_basis, read_model_basis
    from .map_placement import GEOMETRY_FORMAT
    from .image_formats import image_family
    basis = binding["frame_basis"]
    source_run = str(binding["record"]["run_id"]).casefold()
    if source_run not in documents or any(not has_shape_basis(value) for value in documents.values()):
        raise ValueError("Every measured run needs its own explicit RAW basis; no legacy-grid fallback")
    if image_family(basis["photograph"]["path"]) not in {"raw", "processed"}:
        raise ValueError("The painting SOURCE needs its actual camera photograph; an old corrected NPZ is not admitted")
    guarded = []
    for run, model in documents.items():
        if (not isinstance(run, str) or not run or run != run.casefold()
                or not all(char.isalnum() or char in "-_" for char in run)
                or not isinstance(model, dict) or model.get("format") != GEOMETRY_FORMAT
                or model.get("run") != run or model.get("run_id") != run or model.get("grid") != "camera_u"):
            raise ValueError("The RAW-basis measured model needs its exact run and camera grid")
        grid = model["baseline_native_grid"]
        if not _same(grid.get("parent_workspace_revision"), basis["workspace_revision"]):
            raise ValueError(f"{run}: RAW basis and painting preparation belong to different workspace revisions")
        aliases = () if basis_sources is None else basis_sources[run]
        checked = read_model_basis(model, aliases=aliases, folder=folder,
            metadata_replay=(basis_metadata_replay or {}).get(run, ()),
            photo_relocations=(basis_photo_relocations or {}).get(run, ()))

        def read(bound, *, metadata=False):
            return checked.guards.add(_identity(bound), metadata=metadata, copyable=True)

        current = read(basis["workspace_revision"], metadata=True)
        config = current["configuration"]
        photograph = checked.guards.add(_identity(basis["photograph"]))
        reference = checked.guards.add(_identity(model["reference_image"]))
        model_photo = checked.guards.add(_identity(model["model_photograph"]))
        if run == source_run and (not _same(reference, photograph)
                or model.get("shape_hw") != list(reversed(binding["frame"]["image_size"]))):
            raise ValueError("The RAW-basis measured reference belongs to another painting photograph or grid")
        _profile(model, run, basis["camera_profile"], read, source_run=source_run)
        read(model["source_model"], metadata=True)
        read(model["current_edges"])
        baseline = _identity(model["baseline_configuration"])
        selected = ((config.get("frame_binding") or {}).get("runs") or {}).get(run) or {}
        if not _same(baseline, _identity(selected.get("configuration") or {})):
            raise ValueError(f"{run}: the RAW-basis model belongs to another bound original configuration")
        if configuration_copies is not None:
            copied = configuration_copies.get(run)
            if (not isinstance(copied, dict) or set(copied) != {"original", "copy"}
                    or not _same(copied["original"], baseline)):
                raise ValueError("The preparation changed its original configuration closure")
            actual = _local(folder, copied["copy"])
            if any(actual[key] != baseline[key] for key in ("sha256", "size_bytes")):
                raise ValueError("The configuration copy differs from its actual original bytes")
            guarded.append(actual)
        bridge = model.get("baseline_to_model_registration")
        if bridge is not None:
            read(bridge, metadata=True)
        # read_model_basis already rebuilt the unchanged saved-frame/bridge
        # algebra and checked all native matrices against its decoded basis.
        fields = workspace.selected_fields(config, run, check_files=True)
        if grid.get("shape_hw") != fields["labels"]["shape"]:
            raise ValueError(f"{run}: RAW basis and this run's original labels differ")
        read(model_bindings[run], metadata=True)
        checked.revalidate()
        if basis_sources is not None:
            wanted = {row["original"]["path"] for row in aliases}
            consumed = {row["path"] for row in checked.sources_to_copy()}
            if wanted != consumed:
                raise ValueError("The historical RAW-basis source list differs from actual consumed inputs")
        guarded.extend(row["physical"] for row in checked.guards.rows.values())
        if basis_checks is not None:
            basis_checks[run] = checked
    for row in guarded:
        _checked(row)
    return guarded


def basis_label_fields(binding, proof, run, folder):
    """Return verified per-run label metadata; ``run=None`` reads the set once."""
    if "raw_shape_basis" not in proof:
        return None
    sources = _basis_sources(proof, proof["models"], folder)
    if run is not None:
        from .painted_adoption_table import _checked, _same
        from .raw_shape_basis_reader import read_model_basis
        model = _checked(proof["models"][run], metadata=True)
        if not _same(model["baseline_native_grid"]["parent_workspace_revision"],
                     binding["frame_basis"]["workspace_revision"]):
            raise ValueError("The run label grid belongs to another painting workspace")
        checked = read_model_basis(model, aliases=sources[run], folder=folder,
            metadata_replay=proof["raw_shape_basis"].get("metadata_replay", {}).get(run, ()),
            photo_relocations=proof["raw_shape_basis"].get("photo_relocations", {}).get(run, ()))
        checked.revalidate()
        return deepcopy(checked.parent_labels)
    checks = {}
    validate_models(binding, proof["models"],
        configuration_copies=proof["baseline_configurations"], folder=folder,
        basis_sources=sources, basis_checks=checks,
        basis_metadata_replay=proof["raw_shape_basis"].get("metadata_replay"),
        basis_photo_relocations=proof["raw_shape_basis"].get("photo_relocations"))
    return {key: deepcopy(checked.parent_labels) for key, checked in checks.items()}
