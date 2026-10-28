"""Normal Results moves preserve products and legacy native path consumers."""
from pathlib import Path
import tomllib

import pytest
import toml

from darsia.fluidflower import series_creation as creation
from darsia.fluidflower import series_storage_migration as migration


@pytest.fixture
def existing(tmp_path, monkeypatch):
    registry = tmp_path / 'registry.toml'
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY', str(registry))
    photographs = tmp_path / 'photographs'
    photographs.mkdir()
    (photographs / 'baseline.jpg').write_bytes(b'ordinary RGB source, no RAW required')
    project = tmp_path / 'project'
    project.mkdir()
    root = project / 'DarSIA'
    creation.create_series_folder(creation.SeriesDraft('Study', 'Study', 'medium', root,
        [creation.RunDraft('r1', [photographs], baseline='baseline.jpg')]))
    # The new-series default creates this empty queue; this fixture models an
    # older layout and needs a genuinely empty destination for Results relocation.
    (project / 'Results/jobs').rmdir()
    path = root / 'series.toml'
    document = tomllib.loads(path.read_text())
    document['paths']['results'] = 'results'
    document['roots']['results'] = 'results'
    path.write_text(toml.dumps(document))
    run_path = root / 'config/runs/r1.toml'
    run = tomllib.loads(run_path.read_text())
    run['data']['results'] = str(root / 'results/r1')
    run_path.write_text(toml.dumps(run))
    source = root / 'results'
    (source / 'r1').mkdir(parents=True)
    payload = source / 'r1/field.npz'
    payload.write_bytes(b'immutable saved product')
    frozen = source / 'frozen.toml'
    frozen.write_text(toml.dumps(dict(inputs=dict(field=str(payload)))))
    registry.write_text(toml.dumps(dict(schema_version=1,
        series=[dict(id='study', name='Study', root=str(root))])))
    return root, project, source, project / 'Results', frozen.read_bytes()


def test_move_round_trips_existing_native_consumer_and_same_product(existing):
    from darsia.presets.workflows.config.utils import _get_section_from_toml
    root, project, source, target, frozen_bytes = existing
    inode = (source.stat().st_dev, source.stat().st_ino)
    original = (root / 'series.toml').read_bytes()
    plan = migration.preview_results_migration(root, project, target, expected_bytes=original)
    assert source.exists() and not any(target.iterdir())
    assert (root / 'series.toml').read_bytes() == original
    result = migration.apply_results_migration(plan, writers_quiescent=True)
    assert not source.exists() and (target.stat().st_dev, target.stat().st_ino) == inode
    assert (target / 'r1/field.npz').read_bytes() == b'immutable saved product'
    assert (target / 'frozen.toml').read_bytes() == frozen_bytes
    assert Path(_get_section_from_toml(target / 'frozen.toml', 'inputs')['field']) == target / 'r1/field.npz'
    document = tomllib.loads((root / 'series.toml').read_text())
    assert document['storage']['path_aliases'][str(source)] == str(target)
    run = tomllib.loads((root / 'config/runs/r1.toml').read_text())
    assert run['data']['results'] == str(target / 'r1')
    assert result['payload_reads'] == result['payload_copies'] == 0
    assert not (root / '.results-location.lock').exists()


def test_metadata_save_failure_restores_original_directory(existing, monkeypatch):
    root, project, source, target, _ = existing
    before = (root / 'series.toml').read_bytes()
    plan = migration.preview_results_migration(root, project, target)
    def fail(*args, **kwargs):
        raise OSError('metadata save failed')
    monkeypatch.setattr(migration, 'save_series_locations', fail)
    with pytest.raises(OSError, match='metadata save failed'):
        migration.apply_results_migration(plan, writers_quiescent=True)
    assert (source / 'r1/field.npz').read_bytes() == b'immutable saved product'
    assert target.is_dir() and not any(target.iterdir())
    assert (root / 'series.toml').read_bytes() == before
    assert not (root / '.results-location.lock').exists()


def test_move_requires_reviewed_empty_target_and_closed_writers(existing):
    root, project, source, target, _ = existing
    plan = migration.preview_results_migration(root, project, target)
    with pytest.raises(ValueError, match='Close jobs'):
        migration.apply_results_migration(plan, writers_quiescent=False)
    (target / 'another result.txt').write_text('keep me')
    with pytest.raises(ValueError, match='contains files'):
        migration.apply_results_migration(plan, writers_quiescent=True)
    assert source.is_dir() and (target / 'another result.txt').read_text() == 'keep me'
