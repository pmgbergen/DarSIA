from pathlib import Path
import json
import tomllib

import pytest

from darsia.fluidflower import calibration_source_locations as subject
from darsia.fluidflower.mass.n1_pipeline import asset, _write_json


def _case(tmp_path, monkeypatch):
    from darsia.fluidflower import refit_job_panel
    from darsia.fluidflower.pipeline import local_overflow_refit
    root, old, moved = tmp_path/'series', tmp_path/'old', tmp_path/'moved'
    root.mkdir()
    moved.mkdir()
    (root/'series.toml').write_text('schema_version=1\nseries_id="example"\n[runs]\nids=["sample"]\n')
    review = _write_json(moved/'review.json', dict(measurement=3.25))
    logical = dict(review, path=str(old/'review.json'))
    method = _write_json(tmp_path/'method.json', dict(
        protocol=logical, addendum=logical, overflow_review=logical))
    request = _write_json(tmp_path/'inputs.json', dict(
        format='darsia-local-overflow-refit-inputs-v1', method=method,
        scale_reviews=dict(sample=logical)))
    monkeypatch.setattr(refit_job_panel, 'series_refit_roles', lambda *args: ({}, ''))
    calls = []
    def checked(bound, *, roles, reader):
        assert reader.json(bound)['scale_reviews']['sample'] == logical
        assert reader.json(logical) == dict(measurement=3.25)
        calls.append(reader.path(logical))
    monkeypatch.setattr(local_overflow_refit, 'read_request', checked)
    return root, old, moved, request, review, calls


def test_explicit_relocation_verifies_unchanged_bytes_and_saves_only_shared_alias(tmp_path, monkeypatch):
    root, old, moved, request, review, calls = _case(tmp_path, monkeypatch)
    before = tomllib.loads((root/'series.toml').read_text())
    originals = {Path(pin['path']): Path(pin['path']).read_bytes() for pin in (request, review)}
    result = subject.relocate_overflow_sources(root, request, old, moved)
    after = tomllib.loads((root/'series.toml').read_text())
    aliases = after.pop('storage')['path_aliases']
    assert after == before
    assert aliases == {str(old).replace('\\', '/'): str(moved)}
    assert result['request'] == request
    assert calls == [Path(review['path'])]
    assert all(path.read_bytes() == raw for path, raw in originals.items())


def test_changed_destination_is_refused_before_any_alias_save(tmp_path, monkeypatch):
    root, old, moved, request, review, calls = _case(tmp_path, monkeypatch)
    before = (root/'series.toml').read_bytes()
    Path(review['path']).write_text(json.dumps(dict(measurement=8.75)))
    with pytest.raises(ValueError, match='Data SHA|Data size'):
        subject.relocate_overflow_sources(root, request, old, moved)
    assert (root/'series.toml').read_bytes() == before
    assert not calls


def test_unrelated_folder_is_not_saved_as_a_source_alias(tmp_path, monkeypatch):
    root, old, moved, request, review, calls = _case(tmp_path, monkeypatch)
    before = (root/'series.toml').read_bytes()
    with pytest.raises(ValueError, match='names no sources'):
        subject.relocate_overflow_sources(root, request, tmp_path/'unrelated', moved)
    assert (root/'series.toml').read_bytes() == before


def test_shared_alias_is_readable_without_this_machines_start_registration(tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths
    from darsia.fluidflower.pipeline.current_inputs import CurrentInputReader
    root, old, moved, request, review, calls = _case(tmp_path, monkeypatch)
    subject.relocate_overflow_sources(root, request, old, moved)
    monkeypatch.setattr(storage_paths, '_settings',
                        lambda: (tmp_path/'empty-local-registry.toml', dict(schema_version=1, series=[])))
    monkeypatch.setattr(storage_paths, '_shared_alias_roots', {})
    reader = CurrentInputReader(series_roots=(root,))
    logical = dict(review, path=str(old/'review.json'))
    assert reader.path(logical) == Path(review['path'])
    assert reader.json(logical) == dict(measurement=3.25)
