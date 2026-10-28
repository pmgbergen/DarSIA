"""A moved exact saved adoption remains ready; changed content remains blocked."""
import json

import pytest
import toml

from darsia.fluidflower.painted_adoption_table import _bound
from darsia.fluidflower.series_setup_jobs_ui import SeriesSetupJobsController
from .test_findings_before_apply_jobs import owner  # noqa: F401
from .test_series_setup_jobs_ui import _Status


@pytest.mark.parametrize('change', ['none', 'sha256', 'size_bytes', 'path'])
def test_setup_recognizes_only_exact_adopted_preparation_after_move(owner, tmp_path, monkeypatch, change):
    original = tmp_path / 'old preparation'
    original.mkdir()
    (original / 'index.json').write_text('{"map":"approved"}', encoding='utf-8')
    approved = _bound(original / 'index.json')
    moved = tmp_path / 'new preparation'
    original.rename(moved)
    registry = tmp_path / 'private-registry.toml'
    registry.write_text(toml.dumps(dict(schema_version=1,
        storage=dict(path_aliases={str(original): str(moved)}))), encoding='utf-8')
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY', str(registry))
    if change == 'sha256':
        approved['sha256'] = '0' * 64
    elif change == 'size_bytes':
        approved['size_bytes'] += 1
    elif change == 'path':
        approved['path'] = str(tmp_path / 'unrelated/index.json')
    current = owner.workspace.project.root / 'overrides/geometry_facies/current.json'
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(dict(configuration=dict(painted_adoption_table=dict(
        preparation=approved)))), encoding='utf-8')
    before = current.read_bytes()
    status = _Status(owner.view)
    status.result = dict(preparation=dict(folder=str(moved), runs=[dict(run_id='r1', status='prepared')],
        chart_review=dict(runs=dict(r1={}))))
    owner.view.series_use_status = status
    controller = SeriesSetupJobsController(owner.workspace, owner.view,
        context_provider=owner.context, preparer=owner.forbidden)
    assert owner.view.apply_build_button.isEnabled() is (change == 'none')
    assert current.read_bytes() == before
    assert owner.effects == []
    if change != 'none':
        assert 'Save the checked approved map' in owner.view.apply_build_reason.text()
