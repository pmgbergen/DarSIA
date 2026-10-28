"""Renamed folders preserve photo identity and never replace human reviews."""
from copy import deepcopy
from pathlib import Path
import json
import toml

import numpy as np
import pytest

from darsia.fluidflower import rig_point_review as review
from darsia.fluidflower import rig_point_alignment as alignment
from darsia.fluidflower import map_placement
from .test_rig_point_review import context, approve  # noqa: F401
from .test_rig_point_alignment import case, descriptor, capture  # noqa: F401


def test_explicit_storage_move_preserves_approved_points_and_frame(context, tmp_path, monkeypatch):
    document = review.prepare_review(context.frame, context.runs, context.models)
    document = approve(approve(document, 'source'), 'target')
    before = json.dumps(document, sort_keys=True)
    destinations = {}
    for row in context.runs:
        old = Path(row['path'])
        new = tmp_path / 'moved' / old.name
        new.parent.mkdir(exist_ok=True)
        old.rename(new)
        destinations[str(old)] = str(new)
    registry = tmp_path / 'private-registry.toml'
    registry.write_text(toml.dumps(dict(schema_version=1,
        storage=dict(path_aliases=destinations))), encoding='utf-8')
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY', str(registry))
    from darsia.fluidflower.storage_paths import refresh_storage_paths
    refresh_storage_paths()
    current_frame = deepcopy(context.frame)
    current_frame.image = destinations[current_frame.image]
    bindings = {row['run_id']: review.photo_binding(destinations[row['path']], row['image_size'])
                for row in context.runs}
    assert review.reference_matches(document, current_frame)
    assert review.review_blocker(document, current_frame, ['source', 'target'], bindings) is None
    assert review.reviewed_frame(document, 'target').run_id == 'target'
    assert json.dumps(document, sort_keys=True) == before
    changed = deepcopy(current_frame)
    changed.width_m += .001
    assert review.review_blocker(document, changed, ['source', 'target'], bindings)
    photo = Path(destinations[context.runs[1]['path']])
    photo.write_bytes(photo.read_bytes() + b'changed')
    bindings['target'] = review.photo_binding(photo, context.runs[1]['image_size'])
    assert review.review_blocker(document, current_frame, ['source', 'target'], bindings)


def test_moved_review_capsule_reuses_its_original_index_and_approvals(case, tmp_path, monkeypatch):
    from dataclasses import replace
    from darsia.fluidflower.storage_paths import refresh_storage_paths
    derived = capture(case)
    index_bytes = Path(derived.index_path).read_bytes()
    review_bytes = json.dumps(case.document, sort_keys=True)
    old_project, old_models = case.project.root, case.index.parent
    new_project, new_models = tmp_path/'moved-project', tmp_path/'moved-models'
    old_project.rename(new_project); old_models.rename(new_models)
    aliases = {str(old_project): str(new_project), str(old_models): str(new_models)}
    for row in case.context.runs:
        old = Path(row['path']);new = tmp_path/'moved-photos'/old.name
        new.parent.mkdir(exist_ok=True);old.rename(new);aliases[str(old)] = str(new)
    registry = tmp_path/'private-registry.toml'
    registry.write_text(toml.dumps(dict(schema_version=1,storage=dict(path_aliases=aliases))),encoding='utf-8')
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY', str(registry));refresh_storage_paths()
    case.project.root = new_project
    case.index = new_models/case.index.name
    case.selection = replace(case.selection,index_path=str(case.index),project_root=str(new_project))
    case.context.frame.image = aliases[case.context.frame.image]
    again = capture(case)
    assert again.sha256 == derived.sha256 and again.size_bytes == derived.size_bytes
    assert Path(again.index_path).read_bytes() == index_bytes
    assert Path(again.index_path).is_relative_to(new_project)
    assert len(list(Path(again.index_path).parent.parent.iterdir())) == 1
    models = map_placement.read_run_geometry(again.index_path)
    assert json.dumps(case.document, sort_keys=True) == review_bytes
    for run, model in models.items():
        np.testing.assert_allclose(model.metric_to_grid,review.metric_to_grid(case.document,run))
        assert alignment.model_photo_path(model).is_file()
        assert Path(model.source).read_bytes() == case.original[run]


def relocated(context):
    runs = deepcopy(context.runs)
    target = next(row for row in runs if row['run_id'] == 'target')
    old = Path(target['path'])
    expected = descriptor(old)
    new = old.with_name('corrected-date.ARW')
    old.rename(new)
    target['path'] = str(new)
    target['photograph_relocation'] = review.verify_photo_relocation(expected, new)
    return runs, expected, new


def test_same_bytes_allow_renamed_path_but_same_size_different_bytes_do_not(context):
    runs, expected, new = relocated(context)
    witness = runs[1]['photograph_relocation']
    current = review.photo_binding(new, [1000, 700])
    review.check_photo_relocation(witness, current, expected)
    assert witness['original']['sha256'] == witness['current']['sha256']
    before = new.read_bytes()
    new.write_bytes(b'x' * len(before))
    with pytest.raises(ValueError, match='SHA-256'):
        review.verify_photo_relocation(expected, new)
    with pytest.raises(ValueError, match='binding'):
        review.check_photo_relocation(witness, review.photo_binding(new), expected)


def test_only_empty_pending_proposals_are_recovered_and_other_approvals_remain_exact(context):
    runs, _, _ = relocated(context)
    existing = review.prepare_review(context.frame, runs)
    existing = approve(existing, 'source')
    prepared = review.prepare_review(context.frame, runs, context.models)
    recovered = review.recover_missing_proposals(existing, prepared)
    assert recovered['runs']['source'] == existing['runs']['source']
    assert recovered['runs']['manual'] == existing['runs']['manual']
    assert recovered['runs']['target']['current_rc'] == prepared['runs']['target']['proposed_rc']
    assert recovered['runs']['target']['status'] == 'pending'
    assert recovered['runs']['target']['approval'] is None
    edited = review.set_point(existing, 'target', 'top_left', 20., 30.)
    assert review.recover_missing_proposals(edited, prepared) == edited


def test_approved_relocated_photo_replays_against_unchanged_original_model(case):
    runs, original_photo, new = relocated(case.context)
    document = review.prepare_review(case.context.frame, runs, case.context.models)
    document = approve(approve(document, 'source'), 'target')
    document['required_runs'] = ['source', 'target']
    case.document = document
    case.context.runs = runs
    derived = capture(case)
    models = map_placement.read_run_geometry(derived.index_path)
    np.testing.assert_allclose(models['target'].metric_to_grid, review.metric_to_grid(document, 'target'))
    assert models['target'].reference_image == original_photo
    assert alignment.model_photo_path(models['target']) == new.resolve()
    assert alignment.model_photo_path(models['source']) == Path(models['source'].reference_image['path'])
    assert Path(models['target'].source).read_bytes() == case.original['target']
    record = models['target'].entry[alignment.ENTRY_KEY]
    wrong = dict(original_photo, sha256='0' * 64)
    with pytest.raises(ValueError, match='binding'):
        alignment.effective_matrix(record, 'target', wrong)
    with pytest.raises(ValueError, match='binding'):
        alignment.model_photo_path(models['target'], wrong)
    import os
    stamp = new.stat()
    new.write_bytes(b'x' * new.stat().st_size)
    os.utime(new, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    with pytest.raises(ValueError, match='SHA-256'):
        alignment.model_photo_path(models['target'])
    new.touch()
    with pytest.raises(ValueError):
        alignment.effective_matrix(record, 'target', original_photo)
