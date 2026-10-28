"""Bound metadata, independent RGB transport and display-only refusal contracts."""
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import painted_review_comparison as display


def _descriptor(path, raw):
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))


@pytest.fixture
def source(tmp_path, monkeypatch):
    from darsia.fluidflower import painted_analysis_exclusions as core
    from darsia.fluidflower import series_geometry

    documents, reads, native_reads = {}, [], []

    def bind(name, value):
        raw = json.dumps(value, sort_keys=True).encode()
        descriptor = _descriptor(tmp_path / name, raw)
        documents[descriptor['path']] = (descriptor, deepcopy(value))
        return descriptor

    def checked(binding, metadata=False):
        assert Path(binding['path']).suffix.casefold() not in ('.arw', '.npz'), 'No binary source reads'
        reads.append(binding['path'])
        descriptor, value = documents[binding['path']]
        if binding != descriptor:
            raise ValueError('Synthetic metadata source changed')
        return deepcopy(value if metadata else descriptor)

    manifest = dict(schema_version=1, runs=dict(ids=['ac17', 'ac18']))
    manifest_binding = bind('series.toml', manifest)
    monkeypatch.setattr(series_geometry, 'read_series_manifest', lambda root: deepcopy(manifest))
    monkeypatch.setattr(display, '_checked', checked)
    monkeypatch.setattr(display, '_bound', lambda path: deepcopy(documents[str(path)][0]))
    context = dict(format=display.FORMAT, series_manifest=manifest_binding,
        declared_run_order=['ac17', 'ac18'], declared_runs=deepcopy(manifest['runs']),
        image_roles={}, limitations={}, anchor_metric_mm=[0., 0.])
    profile = bind('profile.json', dict(common=dict(sensor=dict(camera_wh=[40, 30]))))
    rows, packets, models, certificates = [], {}, {}, {}
    for run, shape, c, n in (
            ('ac17', [12, 16], [[2., 0., 4.], [0., 3., 5.], [0., 0., 1.]],
             [[1., 0., -1.], [0., 1., -2.], [0., 0., 1.]]),
            ('ac18', [12, 16], [[2., 0., 6.], [0., 3., 8.], [0., 0., 1.]],
             [[1., 0., 1.], [0., 1., 2.], [0., 0., 1.]])):
        raw = _descriptor(tmp_path / (run + '_injection') / 'DSC00002.ARW', run.encode())
        photo = _descriptor(tmp_path / (run + '.npz'), (run + 'photo').encode())
        pixels = dict(shape=shape + [3], dtype='uint8', sha256=hashlib.sha256(run.encode()).hexdigest())
        certificate = dict(renderer=dict(raw_acquisition=deepcopy(raw), camera_u_to_native_h=n,
            native_shape_hw=shape, profile_id='same', camera_u_wh=[40, 30],
            sensor_metadata=dict(orientation=1, white_level=4095, black_level=0)),
            inputs=dict(raw=deepcopy(raw), profile=deepcopy(profile), protocol_inputs=[],
                camera_binding=dict(run_id=run), uncorrected_grid=dict(num_voxels=[30, 40],
                    homography_coordinates=['column', 'row', 'one'], pixel_centres='integer indices',
                    indexing='ij', array_row_direction='down', array_column_direction='right',
                    metadata=dict(date=run))),
            corrected_baseline=dict(archive=photo, pixels=pixels),
            homography_convention=dict(source='saved corrected-baseline pixel centres (column,row)',
                target='camera-U pixel centres (column,row)', camera_u_axes='x right, y down', raw_orientation=1))
        renderer = bind(run + '_renderer.json', certificate)
        model = dict(run_id=run, grid='camera_u', shape_hw=[30, 40],
            H_metric_mm_to_camera_u=c, reference_image=raw,
            camera_profile=dict(profile, profile_id='same', camera_wh=[40, 30]),
            metric_frame=dict(unit='mm', origin='inner left-wall and bottom intersection', x_axis='right', y_axis='up'),
            baseline_native_grid=dict(renderer=renderer, H_baseline_camera_u_to_rig_corrected_native=n, shape_hw=shape))
        model_binding = bind(run + '_model.json', model)
        row = dict(run_id=run, target_model=model_binding, native_photo=photo,
            native_photo_pixels=pixels, gap=dict(shape_hw=shape),
            own_frame_native_xy=[[0., 0.], [15., 0.], [15., 11.], [0., 11.]],
            transported_frame_native_xy=[[0., 1.], [15., 1.], [15., 10.], [0., 10.]])
        rows.append(row)
        rgb = np.arange(np.prod(shape) * 3, dtype=np.uint8).reshape(*shape, 3)
        packets[run] = dict(run_id=run, image_rgb=rgb, image_encoding='rgb_uint8',
            native_to_photo_h=np.eye(3), gap_mask=np.zeros(shape, bool))
        models[run], certificates[run] = model, certificate
    review = dict(display_context=context, source=dict(reference_run='ac17'), runs=rows,
        review_binding=_descriptor(tmp_path / 'review.json', b'review'))

    def read_review(value):
        if isinstance(value, (str, Path)):
            assert str(value) == review['review_binding']['path']
            return deepcopy(review)
        return deepcopy(value)

    def native_preview(value, run, component=None):
        native_reads.append((run, component))
        return dict(packets[run], image_rgb=packets[run]['image_rgb'].copy())

    monkeypatch.setattr(display, '_review', read_review)
    monkeypatch.setattr(core, 'read_gap_preview', native_preview)
    return SimpleNamespace(review=review, context=context, manifest=manifest, bind=bind,
        documents=documents, reads=reads, native_reads=native_reads, packets=packets,
        models=models, certificates=certificates, checked=checked)


def _reseal_model(source, run):
    row = next(row for row in source.review['runs'] if row['run_id'] == run)
    source.models[run]['baseline_native_grid']['renderer'] = source.bind(run + '_renderer.json', source.certificates[run])
    row['target_model'] = source.bind(run + '_model.json', source.models[run])


def test_previous_rgb_uses_independent_native_camera_metric_composition_without_source_mutation(source):
    before = {run: packet['image_rgb'].copy() for run, packet in source.packets.items()}
    result = display.read_comparison_preview(source.review, 'ac18', None, 'previous-run')
    # Independent coordinate oracle: old native (x,y) -> camera (x+1,y+2)
    # -> metric -> new camera (+2,+3) -> new native (+1,+2), total (+4,+7).
    expected = np.zeros_like(before['ac18'])
    expected[7:, 4:] = before['ac17'][:-7, :-4]
    support = np.zeros(expected.shape[:2], bool)
    support[7:, 4:] = True
    np.testing.assert_array_equal(result['image_rgb'], expected)
    np.testing.assert_array_equal(result['valid_support'], support)
    np.testing.assert_array_equal(result['display_transform'], [[1., 0., 4.], [0., 1., 7.], [0., 0., 1.]])
    assert result['source_run_id'] == 'ac17' and result['display_grid']['model'] == source.review['runs'][1]['target_model']
    for run in before:
        np.testing.assert_array_equal(source.packets[run]['image_rgb'], before[run])


def test_immediate_declared_previous_missing_or_excluded_is_not_skipped(source):
    source.context['declared_run_order'] = ['ac17', 'ac20', 'ac18']
    source.manifest['runs']['ids'] = ['ac17', 'ac20', 'ac18']
    source.context['declared_runs'] = deepcopy(source.manifest['runs'])
    packet = display.read_gap_preview(source.review, 'ac18')
    previous = next(choice for choice in packet['image_choices'] if choice['image_id'] == 'previous-run')
    assert previous['available'] is False and 'AC20' in previous['reason'] and 'no run is skipped' in previous['reason']
    assert packet['position_readout']['previous_run_id'] == 'ac20' and packet['position_readout']['previous'] is None
    with pytest.raises(ValueError, match='no run is skipped'):
        display.read_comparison_preview(source.review, 'ac18', None, 'previous-run')
    assert all(run == 'ac18' for run, _ in source.native_reads)


@pytest.mark.parametrize('tamper', ['profile', 'pixel-centres', 'axes'])
def test_explicit_profile_or_spatial_context_mismatch_refuses_comparison(source, tamper):
    if tamper == 'profile':
        other = source.bind('other_profile.json', dict(common=dict(sensor=dict(camera_wh=[40, 30]))))
        source.models['ac17']['camera_profile'].update(other)
        source.certificates['ac17']['inputs']['profile'] = deepcopy(other)
    elif tamper == 'pixel-centres':
        source.certificates['ac17']['inputs']['uncorrected_grid']['pixel_centres'] = 'half-integer indices'
    else:
        source.certificates['ac17']['homography_convention']['camera_u_axes'] = 'x left, y down'
    _reseal_model(source, 'ac17')
    with pytest.raises(ValueError):
        display.read_comparison_preview(source.review, 'ac18', None, 'previous-run')


def test_position_is_metadata_only_and_ignores_run_dates_in_shared_camera_context(source):
    readout = display.read_position_metadata(source.review, 'ac18')
    assert source.native_reads == []
    assert readout['method'] == 'camera registration (map placement)' and readout['anchor_metric_mm'] == [0., 0.]
    assert readout['previous']['dx_mm'] == pytest.approx(1.)
    assert readout['previous']['dy_mm'] == pytest.approx(1.)
    assert readout['previous']['camera_u_context']['raw_orientation'] == 1
    assert 'run_id' not in readout['previous']['camera_u_context']


def _first_role(source, tmp_path, *, selected_baseline='True', earlier_name=False):
    run, row = 'ac18', source.review['runs'][1]
    raw = source.models[run]['reference_image']
    directory = Path(raw['path']).parent
    directory.mkdir()
    Path(raw['path']).touch()
    (directory / ('DSC00001.ARW' if earlier_name else 'DSC00003.ARW')).touch()
    record = dict(path=raw['path'], source_sha256=raw['sha256'], baseline=selected_baseline, datetime='2026-10-01T01:00:00')
    path = tmp_path / 'imaging.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record))
        writer.writeheader(); writer.writerow(record)
    selection = _descriptor(path, path.read_bytes())
    source.documents[str(path)] = (selection, None)
    source.certificates[run]['inputs']['protocol_inputs'] = [dict(role='imaging', folder=str(directory), source=selection)]
    _reseal_model(source, run)
    names = sorted([item.name for item in directory.iterdir()], key=str.casefold)
    role = dict(format='darsia-first-experiment-review-role-v1', run_id=run, role='first-experiment', available=True,
        selected_raw=deepcopy(raw), selected_model=deepcopy(row['target_model']),
        renderer=deepcopy(source.models[run]['baseline_native_grid']['renderer']),
        native_photo=deepcopy(row['native_photo']), native_photo_pixels=deepcopy(row['native_photo_pixels']),
        selection=selection, selected_row=record, directory_observation=dict(path=str(directory), raw_count=len(names),
            first_raw_name=names[0], names_sha256=hashlib.sha256(json.dumps(names, separators=(',', ':')).encode()).hexdigest(),
            directory_mtime_ns=directory.stat().st_mtime_ns))
    receipt = dict(format='darsia-first-experiment-review-roles-v1', runs={run: role}, created_utc='first observation')
    source.context['image_roles']['first-experiment'] = source.bind('first_role.json', receipt)
    return receipt


@pytest.mark.parametrize('selected_baseline,earlier_name,accepted', [('True', False, True), ('False', False, False), ('True', True, False)])
def test_first_experiment_requires_original_protocol_and_first_extant_raw_without_reading_raw_bytes(source, tmp_path, monkeypatch,
        selected_baseline, earlier_name, accepted):
    _first_role(source, tmp_path, selected_baseline=selected_baseline, earlier_name=earlier_name)
    original = Path.read_bytes
    def no_raw(path):
        assert path.suffix.casefold() != '.arw', 'RAW bytes must never be read for this role'
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', no_raw)
    packet = display.read_gap_preview(source.review, 'ac18')
    first = next(choice for choice in packet['image_choices'] if choice['image_id'] == 'first-experiment')
    assert first['available'] is accepted
    if accepted:
        result = display.read_comparison_preview(source.review, 'ac18', None, 'first-experiment')
        np.testing.assert_array_equal(result['image_rgb'], source.packets['ac18']['image_rgb'])
    else:
        with pytest.raises(ValueError):
            display.read_comparison_preview(source.review, 'ac18', None, 'first-experiment')


def test_missing_first_role_and_unrendered_clearwater_remain_unavailable(source):
    packet = display.read_gap_preview(source.review, 'ac18')
    assert not next(choice for choice in packet['image_choices'] if choice['image_id'] == 'first-experiment')['available']
    clear = next(choice for choice in packet['image_choices'] if choice['image_id'] == 'clearwater')
    assert not clear['available'] and 'current-route' in clear['reason']
    with pytest.raises(ValueError, match='current-route'):
        display.read_comparison_preview(source.review, 'ac18', None, 'clearwater')


def test_capture_binds_full_registered_order_and_copies_mutable_callers(source, monkeypatch):
    from darsia.fluidflower import series_geometry
    root = Path(source.context['series_manifest']['path']).parent
    monkeypatch.setattr(series_geometry, 'native_series_link', lambda project: dict(series_root=str(root)))
    order = ['ac17', 'ac18']
    captured = display.capture_display_context(object(), declared_order=order)
    order.append('ac19')
    assert captured['declared_run_order'] == ['ac17', 'ac18']
    with pytest.raises(ValueError, match='registered series order changed'):
        display.capture_display_context(object(), declared_order=['ac18'])
    source.manifest['runs']['ids'].append('ac19')
    with pytest.raises(ValueError, match='declarations changed'):
        display.validate_display_context(captured)


def test_context_source_identity_excludes_receipt_capture_locator_and_time_but_retains_actual_source(source, tmp_path):
    receipt = _first_role(source, tmp_path)
    before = display.context_source_sha256(source.context)
    receipt['created_utc'] = 'later observation'
    receipt['runs']['ac18']['directory_observation']['directory_mtime_ns'] += 1
    source.context['image_roles']['first-experiment'] = source.bind('new_receipt_folder.json', receipt)
    assert display.context_source_sha256(source.context) == before
    receipt['runs']['ac18']['selected_raw']['sha256'] = 'f' * 64
    source.context['image_roles']['first-experiment'] = source.bind('new_receipt_folder.json', receipt)
    assert display.context_source_sha256(source.context) != before


@pytest.mark.parametrize('value', [{1: 'coerced key'}, {'number': float('nan')}, {'array': np.array([1])}, {'tuple': (1, 2)}])
def test_public_canonical_digest_rejects_nonliteral_or_nonfinite_metadata(value):
    with pytest.raises(ValueError, match='finite literal JSON'):
        display.display_context_digest(value)


def test_compact_check_digest_binds_preferences_even_with_identical_empty_issue_list(source):
    first = display.display_findings(source.review)
    changed = deepcopy(first['display_attachments'])
    changed['water']['height_tolerance_mm'] = 4.
    second = display.display_findings(source.review, changed)
    assert first['issues'] == second['issues'] == []
    assert first['display_findings_sha256'] != second['display_findings_sha256']
    assert first['display_attachments']['display_context_sha256'] == display.context_source_sha256(source.context)
    assert len(json.dumps(first['display_attachments']).encode()) < 1500
    assert 'bases' not in first['display_attachments']['water'] and 'readouts' not in first['display_attachments']['position']
    assert source.native_reads == []


def test_changed_advisory_tolerance_exposes_original_measurement_in_declared_order(source):
    attachment = display.make_display_attachments(source.review)
    attachment['position']['shift_tolerance_mm'] = 0.
    result = display.display_findings(source.review, attachment)
    assert [issue['details']['comparison'] for issue in result['issues']] == ['reference', 'previous']
    assert all(issue['severity'] == 'warning' and issue['details']['run_id'] == 'ac18' for issue in result['issues'])
    assert result['issues'][0]['details']['measurement']['dx_mm'] == pytest.approx(1.)


@pytest.mark.parametrize('tamper', ['stale-context', 'water-action', 'boolean-tolerance', 'supplied-basis'])
def test_compact_attachments_reject_stale_context_unbound_water_and_hidden_inputs(source, tamper):
    attachment = display.make_display_attachments(source.review)
    if tamper == 'stale-context':
        attachment['display_context_sha256'] = 'f' * 64
    elif tamper == 'water-action':
        attachment['water']['actions']['ac18'] = dict(run_id='ac18', basis_sha256='f' * 64, actor='human')
    elif tamper == 'boolean-tolerance':
        attachment['position']['shift_tolerance_mm'] = True
    else:
        attachment['water']['bases'] = {'ac18': {'invented': 'basis'}}
    with pytest.raises(ValueError):
        display.display_findings(source.review, attachment)


def test_historical_context_absence_preserves_no_attachment_contract(source):
    source.review.pop('display_context')
    result = display.display_findings(source.review)
    assert result['display_attachments'] is None and result['issues'] == []
    with pytest.raises(ValueError, match='bound display context'):
        display.display_findings(source.review, {'format': display.ATTACHMENTS_FORMAT})


def _clear_selection(source, tmp_path):
    raw_path = tmp_path / 'explicit_heldout_flush' / 'DSC00123.ARW'
    raw_path.parent.mkdir()
    raw_path.write_bytes(b'synthetic-not-decoded')
    raw = _descriptor(raw_path, b'synthetic-not-decoded')
    processing = source.bind('old_processing.json', dict(format='ffac-v5-normalized-clearwater-processing-v1',
        dataset='ffac', run='ac18', raw_source_path=str(raw_path), raw_source_sha256=raw['sha256'], selected_rule='explicit original rule',
        historical_profile='old profile deliberately incompatible', historical_H=[[9, 0, 100], [0, 9, 100], [0, 0, 1]]))
    reference = source.bind('old_reference.json', dict(format='ffac-v5-normalized-clearwater-reference-v1',
        dataset='ffac', run='ac18', raw_source_sha256=raw['sha256'], correction_receipt_path=processing['path'],
        correction_receipt_sha256=processing['sha256'], rgb_path='forbidden-old-rgb.npy', valid_mask_path='forbidden-old-mask.npy'))
    merged = source.bind('explicit_special_row.json', dict(run='ac18', complete=True, reference=deepcopy(reference)))
    merge = source.bind('old_merge.json', dict(complete=True, sources=[dict(run='ac18', merged=merged, reference=reference)]))
    selection = dict(format='darsia-clearwater-display-selection-v1', run_id='ac18', merged_row=merged,
        reference=reference, processing=processing, selected_raw=raw, selected_rule='explicit original rule',
        selection_only=True, historical_render_not_reused=True)
    catalog = dict(format='darsia-clearwater-display-selections-v1', merge=merge, runs=dict(ac18=selection), utc='capture one')
    source.context['image_roles']['clearwater'] = source.bind('clear_catalog.json', catalog)
    # Existing model coordinate convention: +metric y is up, native image y down.
    source.models['ac18']['H_metric_mm_to_camera_u'] = [[1., 0., 2.], [0., -1., 10.], [0., 0., 1.]]
    _reseal_model(source, 'ac18')
    return catalog


def test_clearwater_uses_original_selected_raw_with_current_profile_and_fixed_native_h_only(source, tmp_path, monkeypatch):
    catalog = _clear_selection(source, tmp_path)
    selection = catalog['runs']['ac18']
    camera = np.arange(30 * 40 * 3, dtype=np.uint8).reshape(30, 40, 3)
    before = camera.copy()
    calls = []
    def original_reader(path, **options):
        calls.append((str(path), deepcopy(options)))
        assert str(path) == selection['selected_raw']['path']
        assert options['expected_sha256'] == selection['selected_raw']['sha256'] and options['factor'] == 1
        assert str(options['profile_path']) == source.models['ac18']['camera_profile']['path']
        assert options['profile_id'] == 'same'
        return camera
    original_bytes = Path.read_bytes
    def no_raw(path):
        assert path.suffix.casefold() != '.arw', 'The injected original loader is the only RAW reader'
        return original_bytes(path)
    monkeypatch.setattr(Path, 'read_bytes', no_raw)
    packet = display.read_gap_preview(source.review, 'ac18')
    assert next(row for row in packet['image_choices'] if row['image_id'] == 'clearwater')['available']
    assert calls == []  # Merely listing the role performs metadata admission only.
    result = display.read_comparison_preview(source.review, 'ac18', None, 'clearwater', photo_reader=original_reader)
    expected = np.zeros((12, 16, 3), np.uint8)
    expected[2:, 1:] = camera[:10, :15]
    np.testing.assert_array_equal(result['image_rgb'], expected)
    np.testing.assert_array_equal(result['display_transform'], [[1., 0., 1.], [0., 1., 2.], [0., 0., 1.]])
    np.testing.assert_array_equal(camera, before)
    assert len(calls) == 1 and result['water_surface']['action'] is None
    assert result['water_surface']['basis'] == display.expected_water_surface_basis(source.review, 'ac18')
    assert not any('forbidden-old-' in path for path in source.reads)


def test_clearwater_water_basis_uses_own_geometry_without_borrowing_baseline_bolts(source, tmp_path):
    from darsia.fluidflower import painted_water_surface as water
    _clear_selection(source, tmp_path)
    basis = display.expected_water_surface_basis(source.review, 'ac18')
    assert basis['left_bolts_photo_xy'] is None and basis['port_photo_xy'] is None
    assert basis['own_geometry_origin_metric_mm'] == [0., 0.] and basis['own_geometry_up_metric_xy'] == [0., 1.]
    np.testing.assert_array_equal(basis['photo_to_metric_h'], [[1., 0., -3.], [0., -1., 12.], [0., 0., 1.]])
    action = water.make_water_surface_action(basis, [[4., 4.], [12., 4.]], actor='synthetic human', confirmed_utc='2026-10-22T00:00:00Z')
    measured = water.water_surface_measurements(action, expected_basis=basis)
    assert measured['height_mm'] == pytest.approx(8.) and measured['bolt_height_mm'] is None
    attachment = display.make_display_attachments(source.review, water_actions={'ac18': action})
    assert display.validate_display_attachments(attachment)['water']['actions']['ac18'] == action
    attachment['water']['height_tolerance_mm'] = 4.
    assert display.validate_display_attachments(attachment)['water']['actions']['ac18'] == action
    assert source.native_reads == []


@pytest.mark.parametrize('tamper', ['processing-run', 'raw-sha', 'merge-row', 'rule', 'renderer-shape'])
def test_clearwater_original_selection_chain_or_current_renderer_grid_mismatch_refuses(source, tmp_path, tamper):
    catalog = _clear_selection(source, tmp_path)
    row = catalog['runs']['ac18']
    if tamper in ('processing-run', 'raw-sha', 'rule'):
        processing = source.documents[row['processing']['path']][1]
        processing[{'processing-run': 'run', 'raw-sha': 'raw_source_sha256', 'rule': 'selected_rule'}[tamper]] = 'changed'
        # Deliberately do not reseal old reference: both direct predicates and original SHA joins matter.
        source.documents[row['processing']['path']] = (row['processing'], processing)
    elif tamper == 'merge-row':
        catalog['runs']['ac18']['merged_row'] = source.bind('unselected_row.json', dict(run='ac18', complete=True))
        source.context['image_roles']['clearwater'] = source.bind('clear_catalog.json', catalog)
    reader = lambda *_args, **_kwargs: np.zeros((2, 2, 3) if tamper == 'renderer-shape' else (30, 40, 3), np.uint8)
    with pytest.raises(ValueError):
        display.read_comparison_preview(source.review, 'ac18', None, 'clearwater', photo_reader=reader)


def test_unrelated_manifest_and_raw_directory_metadata_changes_do_not_change_stable_sources(source, tmp_path):
    _first_role(source, tmp_path)
    old = display.context_source_sha256(source.context)
    directory = Path(source.models['ac18']['reference_image']['path']).parent
    (directory / 'unrelated-note.txt').touch()
    info = directory.stat()
    os.utime(directory, ns=(info.st_atime_ns, info.st_mtime_ns + 2_000_000_000))
    source.manifest['native'] = dict(unrelated_current_apply='new configuration pointer')
    source.bind('series.toml', source.manifest)
    assert display.context_source_sha256(source.context) == old
    packet = display.read_gap_preview(source.review, 'ac18')
    assert next(row for row in packet['image_choices'] if row['image_id'] == 'first-experiment')['available']
    (directory / 'DSC00001.ARW').touch()
    packet = display.read_gap_preview(source.review, 'ac18')
    assert not next(row for row in packet['image_choices'] if row['image_id'] == 'first-experiment')['available']
