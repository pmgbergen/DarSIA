"""Ordinary replay keeps real typed-loader inputs despite retained source copies.

The inherited fixtures use a synthetic 2x3 producer and complete tiny lineage;
the owned saves, public reader, typed loaders and source hashes remain real.
"""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest

from darsia.fluidflower import raw_shape_basis_reader as reader
from .test_raw_shape_basis_reader import basis, _descriptor, _changed_byte  # noqa: F401
from .test_raw_shape_basis_copy import lineage_transport  # noqa: F401
from .test_raw_shape_configuration_lineage import synthetic_lineage  # noqa: F401


pytestmark = pytest.mark.parametrize('basis', ['owner-release'], indirect=True)


@pytest.fixture
def ordinary(lineage_transport, tmp_path):
    """Restore only the fixture's original owner, preserving its empty receiver."""
    value = lineage_transport
    original = value.basis.context.destination.parent
    empty = tmp_path / 'preserved-empty-receiver'
    for path in (original, value.retained, empty):
        assert path.resolve().is_relative_to(tmp_path.resolve())
    original.rename(empty)
    value.retained.rename(original)
    return value


def _read(value):
    return reader.read_shape_basis(value.basis.certificate, run_id='sample',
        authority=value.basis.authority, parent_workspace=value.basis.parent_workspace,
        aliases=value.pool['files'], folder=value.pool_root)


def _critical(value):
    case = value.lineage
    return dict(accepted=case['proof']['accepted_configurations'][0],
        current=case['request']['configuration_files'][0],
        series=case['request']['series_context']['manifest'])


def test_ordinary_replay_keeps_typed_inputs_original_and_historical_copy_allowed(ordinary):
    value = ordinary
    archived = value.lineage['proof']['original_configuration_copy']
    checked = _read(value)
    audit = checked.compatibility['configuration_lineage']
    assert audit['parent_configuration'] == value.lineage['parent']
    assert audit['consumed_configuration_files'] == value.lineage['request']['configuration_files']
    assert not Path(value.lineage['parent']['path']).exists()
    for binding in [*value.lineage['proof']['accepted_configurations'],
                    *value.lineage['request']['configuration_files'],
                    value.lineage['request']['series_context']['manifest']]:
        key = reader._key(binding['path'])
        assert key not in checked.guards.aliases
        assert checked.guards.rows[key]['physical'] == binding
        copied = next(row for row in value.pool['files']
                      if reader._key(row['original']['path']) == key)['copy']
        physical_copy = dict(copied, path=str((value.pool_root / copied['path']).resolve()))
        copy_key = reader._key(physical_copy['path'])
        assert checked.guards.rows[copy_key]['physical'] == physical_copy
        assert copy_key not in checked.guards.copyable
    by_path = lambda rows: {reader._key(row['path']): row for row in rows}
    assert by_path(checked.sources_to_copy()) == by_path(value.observed['copyable_sources'])
    historical = checked.guards.rows[reader._key(archived['path'])]
    assert historical['physical']['path'] != archived['path']
    assert reader._key(archived['path']) in checked.guards.aliases
    checked.revalidate()


@pytest.mark.parametrize('role', ['accepted', 'current', 'series'])
@pytest.mark.parametrize('fault', ['changed', 'missing'])
@pytest.mark.parametrize('when', ['before_read', 'after_read'])
def test_valid_pool_never_hides_changed_or_missing_typed_input(ordinary, role, fault, when):
    value = ordinary
    binding = _critical(value)[role]
    checked = _read(value) if when == 'after_read' else None
    copied = next(row for row in value.pool['files']
                  if reader._key(row['original']['path']) == reader._key(binding['path']))
    path = Path(binding['path'])
    if fault == 'changed':
        _changed_byte(path)
    else:
        path.rename(path.with_name(path.name + '.retained-missing'))
    assert _descriptor(value.pool_root / copied['copy']['path'])['sha256'] == binding['sha256']
    with pytest.raises((ValueError, OSError)):
        checked.revalidate() if checked is not None else _read(value)


@pytest.mark.parametrize('role', ['accepted', 'current', 'series'])
@pytest.mark.parametrize('when', ['before_read', 'after_read'])
def test_tampered_critical_copy_refuses_despite_exact_original(ordinary, role, when):
    value = ordinary
    binding = _critical(value)[role]
    checked = _read(value) if when == 'after_read' else None
    copied = next(row for row in value.pool['files']
                  if reader._key(row['original']['path']) == reader._key(binding['path']))
    _changed_byte(value.pool_root / copied['copy']['path'])
    assert _descriptor(binding['path']) == binding
    with pytest.raises(ValueError, match='changed'):
        checked.revalidate() if checked is not None else _read(value)


def test_exact_membership_keeps_existing_alias_semantics(basis, tmp_path):
    initial = reader.read_shape_basis(basis.certificate, run_id='sample',
        authority=basis.authority, parent_workspace=basis.parent_workspace)
    folder = tmp_path / 'membership-pool'
    folder.mkdir()
    aliases = []
    for index, binding in enumerate(initial.sources_to_copy()):
        target = folder / f'{index:04d}.bin'
        shutil.copyfile(binding['path'], target)
        aliases.append(dict(original=deepcopy(binding), copy=dict(_descriptor(target), path=target.name)))
    checked = reader.read_shape_basis(basis.certificate, run_id='sample',
        authority=basis.authority, parent_workspace=basis.parent_workspace,
        aliases=aliases, folder=folder, parent_configuration_lineage={'unused': 'exact membership'})
    assert 'configuration_lineage' not in checked.compatibility
    for binding in [*basis.document['inputs']['configuration_files'],
                    basis.document['inputs']['series_context']['manifest']]:
        key = reader._key(binding['path'])
        assert key in checked.guards.aliases
        assert checked.guards.rows[key]['physical']['path'] != binding['path']
    checked.revalidate()
