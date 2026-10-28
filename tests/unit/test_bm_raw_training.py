import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.phase import prepare_bm_raw_training as prepare
from darsia.fluidflower.phase.train_bm_raw_model import content_input
from darsia.fluidflower.phase.bm_raw_model import prepare_new_raw_frame
from darsia.fluidflower.phase import bm_raw_model
from darsia.fluidflower.phase.package_bm_raw_model import _copy_runtime_sources


def test_projected_probability_roundoff_is_bounded_and_recorded():
    eps = np.spacing(np.float32(1))
    values = np.array([[[1+4*eps, -4*eps, .4]], [[np.nan]*3]], np.float32)
    report = bm_raw_model.bound_projected_scores(values, np.array([[True], [False]]))
    np.testing.assert_array_equal(values[0, 0], [1., 0., np.float32(.4)])
    assert np.isnan(values[1]).all()
    assert report['clipped_score_elements'] == 2
    assert report['maximum_range_excess'] == 4*eps


@pytest.mark.parametrize('invalid', [np.nan, np.inf, -np.inf,
    np.float32(1)+5*np.spacing(np.float32(1)), -5*np.spacing(np.float32(1))])
def test_projected_probability_errors_are_rejected_before_mutation(invalid):
    values = np.array([[[1+np.spacing(np.float32(1))]], [[invalid]]], np.float32)
    original = values.copy()
    with pytest.raises(ValueError, match='Projected BM probabilities'):
        bm_raw_model.bound_projected_scores(values, np.ones((2, 1), bool), chunk_rows=1)
    np.testing.assert_array_equal(values, original)


def test_bundled_override_imports_from_its_actual_fluidflower_namespace(tmp_path):
    source = tmp_path/'original/darsia'
    for relative in ('', 'fluidflower', 'fluidflower/phase'):
        folder = source/relative; folder.mkdir(parents=True, exist_ok=True)
        (folder/'__init__.py').write_text('')
    original = source/'fluidflower/phase/bm_probe.py'
    original.write_text('VALUE = "old"\n')
    replacement = tmp_path/'replacement.py'
    replacement.write_text('VALUE = "bound RAW runtime"\n')
    destination = tmp_path/'bundle'
    _copy_runtime_sources(source, {'phase/bm_probe.py': replacement}, destination)
    script = ('import sys;sys.path.insert(0,sys.argv[1]);'
              'from darsia.fluidflower.phase.bm_probe import VALUE;'
              'assert VALUE == "bound RAW runtime"')
    subprocess.run([sys.executable, '-I', '-c', script,
                    str(destination/'runtime/source')], check=True)
    assert original.read_text() == 'VALUE = "old"\n'


def test_sony_tables_are_bound_to_each_actual_exposure(monkeypatch):
    tables = [4, 11, 12, 21, 22]
    sub = SimpleNamespace(tags={99: SimpleNamespace(value=tables)})
    class Tiff:
        pages = [SimpleNamespace(pages=[sub])]
        def __init__(self, path):
            assert path == 'frame2.ARW'
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    monkeypatch.setattr(prepare.tifffile, 'TiffFile', Tiff)
    parameters = {'common': {'lens': dict(main_page=0, sub_page=0, ca_tag=99,
        ca_min_length=5, ca_header=4, red_slice=[1, 3], blue_slice=[3, 5],
        ca_red=[1, 2], ca_blue=[3, 4]), 'profiles': {'camera-u': {'distortion': [0, 0]}}}}
    def verify(path, *, parameters):
        assert parameters['ca_red'] == [11, 12]
        assert parameters['ca_blue'] == [21, 22]
        return (3, 7), parameters['ca_red'], parameters['ca_blue']
    monkeypatch.setattr(prepare.registration, 'read_lens_tables', verify)
    prepare.bind_file_lens_tables('frame2.ARW', parameters)
    assert parameters['common']['profiles']['camera-u']['distortion'] == [3, 7]


def frame(root, support):
    folder = root/'frame_000'; folder.mkdir()
    np.save(folder/'content_rgb.npy', np.full((*support.shape, 3), .3, np.float32))
    np.save(folder/'content_valid_support.npy', support)
    receipt = {'arrays': {name: {'path': 'C:/deleted_cache/'+name,
                                'sha256': prepare.sha256(folder/name)}
                          for name in ('content_rgb.npy', 'content_valid_support.npy')}}
    (folder/'receipt.json').write_text(json.dumps(receipt))
    return folder


def test_durable_feature_input_resolves_relative_to_bundle(tmp_path):
    frame(tmp_path, np.ones((3, 4), bool))
    values, _ = content_input(tmp_path, 0)
    assert values.shape == (3, 4, 3)


def test_unsupported_feature_pixels_do_not_turn_into_black_evidence(tmp_path):
    support = np.ones((3, 4), bool); support[1, 1] = False
    frame(tmp_path, support)
    with pytest.raises(ValueError, match='Incomplete finite'):
        content_input(tmp_path, 0)


def test_changed_feature_input_fails_checksum(tmp_path):
    folder = frame(tmp_path, np.ones((3, 4), bool))
    np.save(folder/'content_rgb.npy', np.zeros((3, 4, 3), np.float32))
    with pytest.raises(ValueError, match='checksum'):
        content_input(tmp_path, 0)


def test_new_raw_time_requires_explicit_temporal_context(tmp_path):
    with pytest.raises(ValueError, match='preceding prepared RAW'):
        prepare_new_raw_frame(tmp_path, 'new.ARW', 'new.TIF', tmp_path/'result',
            expected_raw_sha256='raw', expected_photo_sha256='photo', elapsed_h=2.,
            previous_prepared_frame=None)
    assert not (tmp_path/'result').exists()


def test_new_raw_time_requires_bound_previous_receipt(tmp_path):
    with pytest.raises(ValueError, match='receipt must be explicitly SHA-bound'):
        prepare_new_raw_frame(tmp_path, 'new.ARW', 'new.TIF', tmp_path/'result',
            expected_raw_sha256='raw', expected_photo_sha256='photo', elapsed_h=2.,
            previous_prepared_frame=tmp_path/'previous')


def test_support_frame_without_earlier_context_has_clear_error(monkeypatch, tmp_path):
    manifest = {'prepared_frames': [0, 17, 20, 21], 'inferable_frames': [0, 21]}
    monkeypatch.setattr(bm_raw_model, 'load_bundle', lambda root: (manifest, None, None, None, None))
    with pytest.raises(ValueError, match='lacks its preceding RAW context'):
        bm_raw_model.predict_prepared_frame(tmp_path, 20, tmp_path/'out')


def test_new_frame_checks_baseline_receipt_before_using_its_arrays(monkeypatch, tmp_path):
    root = tmp_path/'model'; root.mkdir()
    (root/'model.json').write_text('{}')
    baseline = root/'inputs/frame_000'; baseline.mkdir(parents=True)
    (baseline/'receipt.json').write_text('changed receipt')
    context = tmp_path/'context'; (context/'frame_001').mkdir(parents=True)
    (context/'frame_001/receipt.json').write_text('{}')
    (context/'inference_context.json').write_text(json.dumps({
        'model_sha256': prepare.sha256(root/'model.json'),
        'current_receipt_sha256': prepare.sha256(context/'frame_001/receipt.json')}))
    monkeypatch.setattr(bm_raw_model, 'load_bundle', lambda root: ({'frame_receipts': {'0': 'wrong'}}, None, None, None, None))
    calls = []
    def input_data(parent, index):
        calls.append((parent, index))
        return None, None
    monkeypatch.setattr(bm_raw_model, 'content_input', input_data)
    with pytest.raises(ValueError, match='checksum'):
        bm_raw_model.predict_new_raw_frame(root, context, tmp_path/'out')
    assert calls == [(context, 1)]
