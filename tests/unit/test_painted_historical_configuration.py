"""Historical TOML retention using the actual strict five-input lineage audit.

Only the surrounding full RAW/model reader is isolated: its genuine output
context is represented by ReadShapeBasis. The strict lineage verifier and its
current physical TOML/protocol checks execute unchanged on real tiny files.
No fixture claims RAW decoder, 39-run or production qualification.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower import painted_adoption_geometry as geometry
from darsia.fluidflower import raw_shape_basis_reader as reader
from darsia.fluidflower import raw_shape_configuration_lineage as lineage
from darsia.fluidflower.rig_raw_write import _identity as directory_identity
from .test_raw_shape_configuration_lineage import synthetic_lineage  # noqa: F401


def _bound(path):
    data = Path(path).read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))


def _json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return _bound(path)


@pytest.fixture
def retained(synthetic_lineage, tmp_path, monkeypatch):
    case = synthetic_lineage
    pins = reader._Pins()
    audit = lineage._require_shape_configuration_lineage(pins,
        run=case['run'], parent_configuration=case['parent'],
        producer_configuration_files=case['request']['configuration_files'],
        request=case['request'], proof=case['proof'])
    # The historical locator now holds distinct current bytes, as after a
    # migration; only the explicitly proved archive preserves the old bytes.
    current = Path(case['parent']['path'])
    current.parent.mkdir()
    current.write_bytes(b'# later current configuration; never rolled back\n')
    products = tmp_path / 'owned-model-context'
    products.mkdir()
    certificate = _json(products / 'certificate.json', {'unit': 'no RAW execution'})
    compatibility = dict(configuration=deepcopy(case['parent']), configuration_lineage=audit)
    checked = reader.ReadShapeBasis(certificate, dict(outputs=[], output_ownership=dict(
        authority=dict(destination=str(products), intended_root=str(tmp_path)),
        directory_identity=list(directory_identity(products)),
        intended_root_identity=list(directory_identity(tmp_path)))), pins, compatibility=compatibility)
    model = dict(baseline_configuration=deepcopy(case['parent']), baseline_native_grid=dict(
        renderer=certificate, raw_shape_basis_authority={},
        parent_grid_compatibility=deepcopy(compatibility)))
    path = tmp_path / 'model.json'
    binding = _json(path, model)
    proof = dict(models={case['run']: binding}, raw_shape_basis=dict(source_copies=[]))
    calls = []
    def actual_checked_context(value):
        assert value == model
        calls.append(value)
        return checked
    monkeypatch.setattr(reader, 'read_model_basis', actual_checked_context)
    return SimpleNamespace(case=case, checked=checked, model=model, model_path=path,
        proof=proof, current=current, calls=calls)


def test_checked_lineage_supplies_only_exact_historical_parent_toml(retained):
    f = retained
    rows, checks = geometry.received_source_rows(f.proof)
    parent = f.case['parent']
    key = (parent['path'], parent['sha256'], parent['size_bytes'])
    assert rows == {key: f.case['proof']['original_configuration_copy']}
    assert checks == [f.checked] and len(f.calls) == 1
    assert f.checked.guards.aliases == {}  # no general alias capability added
    assert _bound(f.current)['sha256'] != parent['sha256']
    f.checked.revalidate()


def test_preparation_reuses_its_checked_context_and_copies_original_bytes(retained, tmp_path, monkeypatch):
    f = retained
    def forbid_new_model_decode(*args, **kwargs):
        raise AssertionError('The already checked model must not be decoded again')
    monkeypatch.setattr(reader, 'read_model_basis', forbid_new_model_decode)
    folder = tmp_path / 'new-preparation'
    folder.mkdir()
    current = f.current.read_bytes()
    geometry.write_configuration_copies(folder, f.proof,
        basis_checks={f.case['run']: f.checked})
    copy = f.proof['baseline_configurations'][f.case['run']]
    expected = Path(f.case['proof']['original_configuration_copy']['path']).read_bytes()
    assert (folder / copy['copy']['path']).read_bytes() == expected
    assert copy['original'] == f.case['parent']
    assert f.current.read_bytes() == current
    assert f.checked.guards.aliases == {}


@pytest.mark.parametrize('role', ['archive', 'current', 'lineage'])
def test_retained_context_revalidates_archive_current_consumption_and_proof(retained, role):
    f = retained
    _, checks = geometry.received_source_rows(f.proof)
    paths = dict(archive=Path(f.case['proof']['original_configuration_copy']['path']),
                 current=f.case['actual_path'], lineage=f.case['source_lineage_path'])
    path = paths[role]
    path.write_bytes(path.read_bytes() + b'\n# changed after first source lookup\n')
    with pytest.raises(ValueError, match='changed'):
        checks[0].revalidate()


def test_changed_archive_refuses_before_any_preparation_copy(retained, tmp_path):
    f = retained
    archive = Path(f.case['proof']['original_configuration_copy']['path'])
    archive.write_bytes(archive.read_bytes() + b'changed')
    folder = tmp_path / 'new-preparation'
    folder.mkdir()
    with pytest.raises(ValueError, match='changed'):
        geometry.write_configuration_copies(folder, f.proof)
    assert not (folder / 'configuration_sources').exists()


def test_an_ordinary_basis_without_lineage_keeps_original_path_guard(retained, tmp_path, monkeypatch):
    f = retained
    f.model['baseline_native_grid']['parent_grid_compatibility'].pop('configuration_lineage')
    f.proof['models'][f.case['run']] = _json(f.model_path, f.model)
    def forbid_lineage_fallback(*args, **kwargs):
        raise AssertionError('No new ordinary source route without an explicit lineage audit')
    monkeypatch.setattr(reader, 'read_model_basis', forbid_lineage_fallback)
    assert geometry.received_source_rows(f.proof) == ({}, [])
    folder = tmp_path / 'legacy-preparation'
    folder.mkdir()
    with pytest.raises(ValueError, match='imported geometry source changed'):
        geometry.write_configuration_copies(folder, f.proof)


def test_retained_context_for_another_basis_refuses(retained):
    f = retained
    f.checked.certificate = dict(f.checked.certificate, sha256='0' * 64)
    with pytest.raises(ValueError, match='another checked RAW basis'):
        geometry.received_source_rows(f.proof, basis_checks={f.case['run']: f.checked})


def test_lineage_cannot_supply_another_parent_configuration(retained):
    f = retained
    f.model['baseline_configuration'] = dict(f.case['parent'], path=str(f.current.parent / 'other.toml'))
    f.proof['models'][f.case['run']] = _json(f.model_path, f.model)
    with pytest.raises(ValueError, match='historical parent TOML'):
        geometry.received_source_rows(f.proof)


def test_lineage_mapping_cannot_become_a_general_non_toml_source_alias(retained):
    f = retained
    archive = Path(f.case['proof']['original_configuration_copy']['path'])
    other = archive.with_suffix('.bin')
    other.write_bytes(archive.read_bytes())
    copied = _bound(other)
    f.checked.compatibility['configuration_lineage']['verified_original_configuration_copy'] = copied
    f.model['baseline_native_grid']['parent_grid_compatibility'] = deepcopy(f.checked.compatibility)
    f.proof['models'][f.case['run']] = _json(f.model_path, f.model)
    with pytest.raises(ValueError, match='historical parent TOML'):
        geometry.received_source_rows(f.proof)
