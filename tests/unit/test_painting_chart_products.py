"""Approved chart products feed the ordinary effective-map reader."""
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest

from darsia.fluidflower import painting_chart_transfer as chart
from darsia.fluidflower import painting_chart_products as products
from darsia.fluidflower import painting_chart_transfer_reader as reader
from darsia.fluidflower.painted_target_integration import effective_product
from darsia.fluidflower.painted_adoption_table import _bound
from darsia.fluidflower.painted_adoption_table import _native_flags
from .test_painting_chart_transfer import inputs


@pytest.fixture
def saved_chart_maps(tmp_path, monkeypatch, request):
    data = inputs()
    data['base_labels'] = data['base_labels'].astype(np.int32)
    scenario = getattr(request, 'param', {})
    if scenario.get('border_crossing'):
        data['inside'][8:10, 20:30] = False
    preparation = dict(format='darsia-fluidflower-painted-map-adoption-preparation-v2',
        painting_geometry=dict(review={'fixture': 'review'}), painting_facies={'fixture': 'painting'}, runs=[])
    base = dict(array=data['base_labels'], inside=data['inside'],
        filled=np.zeros(data['inside'].shape, bool), support=data['inside'].copy(),
        continued=np.zeros(data['inside'].shape, bool), effective_coverage=data['inside'].copy(),
        gaps=np.zeros(data['inside'].shape, bool))
    for run in chart.RUNS:
        folder = tmp_path / run
        folder.mkdir()
        path = folder / 'labels.npz'
        np.savez_compressed(path, **base, metadata={'dimensions': [60., 80.]})
        preparation['runs'].append(dict(run_id=run, labels=dict(_bound(path), path=f'{run}/labels.npz'),
            target_model={'fixture': run}))
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def proposal(self, run, *, correction_mm=None):
            values = deepcopy(data)
            values['source_binding'] = dict(review=preparation['painting_geometry']['review'],
                painting_facies=preparation['painting_facies'], target_model={'fixture': run})
            return chart.propose_chart_transfer(run, **values, correction_mm=correction_mm)
    monkeypatch.setattr(reader, 'ChartTransferSession', Session)
    corrections = scenario
    approvals = {run: chart.confirm_chart_transfer(Session().proposal(run, correction_mm=corrections.get(run)), actor='Reviewer',
        reason='Checked the chart outline') for run in chart.RUNS}
    originals = {run: (tmp_path/run/'labels.npz').read_bytes() for run in chart.RUNS}
    preparation['chart_transfers'] = products.prepare_chart_products(preparation, tmp_path, approvals)
    return preparation, tmp_path, base, originals


def test_all_fourteen_effective_maps_use_approved_chart_before_fields_and_25_keep_bytes(saved_chart_maps):
    preparation, folder, base, originals = saved_chart_maps
    for row in preparation['runs']:
        run = row['run_id']
        actual, bound = effective_product(preparation, folder, row)
        labels, filled, support, proof_bound = _native_flags(folder, row, preparation)
        assert proof_bound == bound
        for value, name in ((labels, 'array'), (filled, 'filled'), (support, 'support')):
            np.testing.assert_array_equal(value, actual[name])
        assert (folder/run/'labels.npz').read_bytes() == originals[run]
        if run in chart.MOVED_RUNS:
            assert Path(bound['path']).name == 'chart_labels.npz'
            assert np.all(actual['array'][8:12, 20:30] == 2)
            assert np.all(actual['array'][23:31, 20:30] == 17)
            assert not np.array_equal(actual['array'], base['array'])
        else:
            assert Path(bound['path']).name == 'labels.npz'
            for key in base:
                assert np.array_equal(actual[key], base[key])
        for key in ('inside', 'support', 'continued', 'filled'):
            assert np.array_equal(actual[key], base[key])


@pytest.mark.parametrize('fault', ['approval', 'run', 'source', 'artifact'])
def test_changed_chart_source_or_approval_cannot_feed_fields(saved_chart_maps, fault):
    preparation, folder, _, _ = saved_chart_maps
    row = next(item for item in preparation['runs'] if item['run_id'] == 'ac39')
    record = preparation['chart_transfers']['ac39']
    if fault == 'approval':
        record['approval']['correction_mm'][0] += 1.
    elif fault == 'run':
        record['proposal']['run_id'] = 'ac40'
    elif fault == 'source':
        record['proposal']['source_binding']['target_model'] = {'fixture': 'changed'}
    else:
        path = folder / record['artifact']['path']
        path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError):
        effective_product(preparation, folder, row)
    with pytest.raises(ValueError):
        _native_flags(folder, row, preparation)


def test_incomplete_approvals_do_not_write_products(tmp_path):
    with pytest.raises(ValueError, match='every adoption run'):
        products.prepare_chart_products({}, tmp_path, {})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('fault', ['correction', 'reason', 'source', 'missing'])
def test_unchanged_run_also_requires_its_exact_approval(saved_chart_maps, fault):
    preparation, folder, _, _ = saved_chart_maps
    row = next(item for item in preparation['runs'] if item['run_id'] == 'ac17')
    record = preparation['chart_review']['runs']['ac17']
    if fault == 'correction':
        record['approval']['correction_mm'][0] = 1.
    elif fault == 'reason':
        record['approval']['reason'] = ''
    elif fault == 'source':
        record['proposal']['source_binding']['target_model'] = {'fixture': 'changed'}
    else:
        del preparation['chart_review']['runs']['ac17']
    with pytest.raises(ValueError):
        effective_product(preparation, folder, row)


@pytest.mark.parametrize('saved_chart_maps', [{'ac17': [4., 2.]}], indirect=True)
def test_manual_change_on_default_unchanged_run_reaches_effective_product(saved_chart_maps):
    preparation, folder, base, originals = saved_chart_maps
    row = next(item for item in preparation['runs'] if item['run_id'] == 'ac17')
    actual, bound = effective_product(preparation, folder, row)
    record = preparation['chart_transfers']['ac17']
    assert record['approval']['correction_mm'] == [4., 2.]
    assert record['proposal']['margin_mm'] == 5.
    assert Path(bound['path']).name == 'chart_labels.npz'
    assert not np.array_equal(actual['array'], base['array'])
    assert (folder / 'ac17' / 'labels.npz').read_bytes() == originals['ac17']


@pytest.mark.parametrize('saved_chart_maps', [{'border_crossing': True}], indirect=True)
def test_products_preserve_unchanged_border_mask_and_separate_analysis_domain(saved_chart_maps):
    preparation, folder, base, originals = saved_chart_maps
    for run in ('ac17', 'ac39'):
        row = next(row for row in preparation['runs'] if row['run_id'] == run)
        actual, bound = effective_product(preparation, folder, row)
        assert np.array_equal(actual['inside'], base['inside'])
        assert not np.any(actual['gaps'] & ~actual['inside'])
        assert (folder / run / 'labels.npz').read_bytes() == originals[run]
        if run == 'ac17':
            assert Path(bound['path']).name == 'labels.npz'
            assert all(np.array_equal(actual[key], value) for key, value in base.items())


def test_chart_follows_the_existing_effective_map_reader(tmp_path, monkeypatch):
    """Finite reader-order seam; existing corrected labels remain material input."""
    from darsia.fluidflower import painted_target_integration as target
    data = inputs()
    data['base_labels'] = data['base_labels'].astype(np.int32)
    data['base_labels'][40:45, 40:45] = 2
    data['source_material'][40:45, 40:45] = 2
    run = 'another_run'
    folder = tmp_path / run
    folder.mkdir()
    path = folder / 'effective_target.npz'
    mask = data['inside']
    base = dict(array=data['base_labels'], inside=mask, support=mask,
        filled=~mask, continued=~mask, effective_coverage=mask, gaps=~mask)
    np.savez_compressed(path, **base, metadata={'dimensions': [60., 80.]})
    preparation = dict(format='darsia-fluidflower-painted-map-adoption-preparation-v2',
        painting_geometry=dict(review={'fixture': 'review'}), painting_facies={'fixture': 'painting'},
        runs=[dict(run_id=run, target_model={'fixture': run})])
    data['source_binding'] = dict(review=preparation['painting_geometry']['review'],
        painting_facies=preparation['painting_facies'], target_model={'fixture': run})
    class Session:
        def __init__(self, *args, **kwargs):
            pass
        def proposal(self, run_id, *, correction_mm=None):
            return chart.propose_chart_transfer(run_id, **data, correction_mm=correction_mm)
    monkeypatch.setattr(reader, 'ChartTransferSession', Session)
    monkeypatch.setattr(target, '_effective_without_chart', lambda *args: (base, _bound(path)))
    approval = chart.confirm_chart_transfer(Session().proposal(run, correction_mm=[2., 0.]),
        actor='Reviewer', reason='Checked this saved corrected map')
    preparation['chart_transfers'] = products.prepare_chart_products(preparation, tmp_path, {run: approval})
    actual, _ = effective_product(preparation, tmp_path, preparation['runs'][0])
    assert np.all(actual['array'][40:45, 40:45] == 2)
    assert not np.array_equal(actual['array'], base['array'])
