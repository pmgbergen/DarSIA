"""Full sensor FOV and reversible display coordinates without scientific edits."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower.native import raw_full_display as display
from darsia.fluidflower.native.io import sha256_file


def parameters():
    return {'common': {
        'sensor': dict(raw_wh=[8, 6], camera_wh=[6, 4], cfa_pattern=[[0, 1], [3, 2]],
                       color_description='RGBG', black_levels=[800]*4, tile_rows=2),
        'lens': dict(raw_offset_xy=[1., 1.], knots=[0., .5, 1.], distortion_denominator=100.),
        'profiles': {'test': dict(coefficients=dict(R=5., amp=1., cu=3., cv=2., tx=0., ty=0.),
                                  distortion=[0., 0., 0.])}}}


def test_nonlinear_sensor_mapping_roundtrip_and_complete_boundary_extent():
    data = parameters()
    data['common']['profiles']['test']['distortion'] = [4., 1., -8.]
    mapping = display.mapping_from_parameters(data, 'test')
    points = np.vstack((mapping.sensor_boundary_xy(), [[3., 3.], [100., -80.]]))
    camera_u = mapping.sensor_to_camera_u_xy(points)
    np.testing.assert_allclose(mapping.camera_u_to_sensor_xy(camera_u), points, atol=2e-5)
    left, top, width, height = display.full_raw_display_extent([mapping])
    shown = camera_u[:-2] - [left, top]
    assert (shown > 0).all() and (shown < [width-1, height-1]).all()
    # Every sensor edge is covered, including curved edge midpoints, at one U scale.
    other = deepcopy(data)
    other['common']['lens']['raw_offset_xy'] = [1.1, .9]
    other_mapping = display.mapping_from_parameters(other, 'test')
    extent = display.full_raw_display_extent([mapping, other_mapping])
    assert extent == display.full_raw_display_extent([other_mapping, mapping])
    for value in (mapping, other_mapping):
        shown = value.sensor_to_camera_u_xy(value.sensor_boundary_xy()) - extent[:2]
        assert (shown >= 0).all() and (shown < extent[2:]).all()


@pytest.fixture
def decoder(tmp_path, monkeypatch):
    import sys
    from darsia.fluidflower.native import raw_photograph, raw_camera
    data = parameters()
    raw = tmp_path / 'original.ARW'
    raw.write_bytes(b'original bound RAW')
    profile_path = tmp_path / 'bound.json'
    profile_path.write_bytes(b'bound camera profile')
    pixels = np.arange(8*6*3, dtype=np.uint8).reshape(6, 8, 3)
    calls = []

    class Source:
        raw_image_visible = np.zeros((6, 8), np.uint16)
        raw_pattern = np.array([[0, 1], [3, 2]])
        color_desc = b'RGBG'
        black_level_per_channel = [800]*4
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def postprocess(self, **options):
            calls.append(options)
            return pixels.copy()

    source = Source()
    monkeypatch.setitem(sys.modules, 'rawpy', SimpleNamespace(imread=lambda path: source))
    monkeypatch.setattr(raw_photograph, '_parameters', lambda *args: data)
    monkeypatch.setattr(raw_camera, 'read_camera_identity', lambda path: SimpleNamespace(orientation=1))
    monkeypatch.setattr(display.registration, 'read_lens_tables', lambda *args, **kwargs: ((0., 0., 0.), (), ()))
    options = dict(profile_path=profile_path, profile_sha256=sha256_file(profile_path), profile_id='test',
                   expected_sha256=sha256_file(raw), expected_orientation=1)
    return raw, pixels, source, calls, options


def test_layers_and_bound_preview_share_one_immutable_display_decode(decoder):
    raw, pixels, source, calls, options = decoder
    unbound = {key: value for key, value in options.items() if key not in ('expected_sha256', 'expected_orientation')}
    first = display.read_full_raw_display(raw, **unbound)
    second = display.read_full_raw_display(raw, **options)
    assert len(calls) == 1
    assert first['image_rgb'] is second['image_rgb']
    assert not second['image_rgb'].flags.writeable
    assert first['display_cache_reused'] is False and second['display_cache_reused'] is True
    assert second['raw_orientation'] == 1 and second['display_conversion']['use_camera_wb'] is True
    second['display_conversion']['use_camera_wb'] = False
    assert display.read_full_raw_display(raw, **options)['display_conversion']['use_camera_wb'] is True
    with pytest.raises(ValueError, match='display binding'):
        display.read_full_raw_display(raw, **dict(options, expected_sha256='0'*64))
    with pytest.raises(ValueError, match='orientation'):
        display.read_full_raw_display(raw, **dict(options, expected_orientation=6))
    raw.write_bytes(b'changed source bytes')
    with pytest.raises(ValueError, match='display binding'):
        display.read_full_raw_display(raw, **options)


def test_display_cache_is_bounded_and_invalidates_changed_profile(decoder, monkeypatch):
    raw, pixels, source, calls, options = decoder
    display.read_full_raw_display(raw, **options)
    options['profile_path'].write_bytes(b'changed profile fixture')
    display.read_full_raw_display(raw, **options)
    assert len(calls) == 2
    display._DISPLAY_CACHE.clear()
    monkeypatch.setattr(display, '_DISPLAY_CACHE_BYTES', 1)
    display.read_full_raw_display(raw, **options)
    display.read_full_raw_display(raw, **options)
    assert len(calls) == 4 and not display._DISPLAY_CACHE


def test_full_raw_decoder_keeps_sensor_pixels_outside_original_camera_u_and_inverse_clicks(decoder):
    raw, pixels, source, calls, options = decoder
    before = raw.read_bytes()
    packet = display.read_full_raw_display(raw, **options)
    mapping = packet['mapping']
    # Complete sensor corners appear, including rows/columns outside the old 6x4 U.
    sensor = np.array([[0., 0.], [7., 0.], [7., 5.], [0., 5.]])
    camera_u = mapping.sensor_to_camera_u_xy(sensor)
    shown = mapping.camera_u_to_display_xy(camera_u)
    np.testing.assert_array_equal(packet['image_rgb'][shown[:, 1].astype(int), shown[:, 0].astype(int)],
                                  pixels[sensor[:, 1].astype(int), sensor[:, 0].astype(int)])
    np.testing.assert_array_equal(mapping.display_to_camera_u_xy(shown), camera_u)
    assert camera_u.min() < 0 and camera_u[:, 0].max() >= mapping.camera_u_wh[0]
    assert packet['full_raw_sensor_field'] is True and packet['display_only'] is True
    assert calls == [dict(use_camera_wb=True, output_bps=8, half_size=False, user_flip=0)]
    assert raw.read_bytes() == before


@pytest.mark.parametrize('damage', ['raw_hash', 'sensor', 'orientation', 'cropped_extent'])
def test_full_raw_display_refuses_wrong_identity_sensor_orientation_or_cropped_field(decoder, damage):
    raw, pixels, source, calls, options = decoder
    if damage == 'raw_hash':
        options['expected_sha256'] = '0'*64
    elif damage == 'sensor':
        source.raw_pattern = np.array([[1, 0], [2, 3]])
    elif damage == 'orientation':
        options['expected_orientation'] = 6
    else:
        options['extent'] = (0, 0, 6, 4)
    with pytest.raises(ValueError):
        display.read_full_raw_display(raw, **options)
