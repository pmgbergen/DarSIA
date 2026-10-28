"""Complete photograph context on a common display plane; synthetic sensor RGB."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import painted_camera_jpg_display as display


def models():
    geometry = SimpleNamespace(model={'camera_profile': {'camera_wh': [8, 6]}})
    row = {'gap': {'shape_hw': [4, 5]}}
    return [(geometry, np.array([[1., 0., -2.], [0., 1., -1.], [0., 0., 1.]]), row),
            (geometry, np.array([[1., 0., 1.], [0., 1., 2.], [0., 0., 1.]]), row)]


def test_common_extent_contains_full_photos_beyond_original_native_crop():
    entries = models()
    shape, offset = display._transfer_canvas(entries)
    assert shape[0] > 6 and shape[1] > 8
    for _, native, _ in entries:
        corners = np.array([[0., 0., 1.], [7., 0., 1.], [7., 5., 1.], [0., 5., 1.]])
        actual = corners @ (offset @ native).T
        assert (actual[:, :2] >= 0).all()
        assert (actual[:, 0] < shape[1]).all() and (actual[:, 1] < shape[0]).all()
    np.testing.assert_array_equal(offset[:2, :2], np.eye(2))
    reverse_shape, reverse_offset = display._transfer_canvas(list(reversed(entries)))
    assert reverse_shape == shape
    np.testing.assert_array_equal(reverse_offset, offset)


def test_full_photo_reader_preserves_native_masks_and_rig_and_uses_one_common_scale(monkeypatch):
    from darsia.fluidflower.native import raw_full_display, raw_photograph
    from darsia.fluidflower import painted_analysis_exclusions
    from tests.unit.test_raw_full_display import parameters
    entries = models()
    rows = [dict(entry[2], run_id=str(i)) for i, entry in enumerate(entries)]
    review = {'runs': rows}
    raw = {'path': 'source.ARW', 'sha256': 'a'*64, 'size_bytes': 1}
    profile = dict(path='bound.json', sha256='b'*64, profile_id='test')
    geometries = [SimpleNamespace(model=dict(camera_profile=profile, baseline_native_grid={'renderer': {}}),
                                  reference_image=raw) for entry in entries]
    original_mask = np.ones((4, 5), bool)
    original_frame = [[0., 0.], [4., 0.], [4., 3.], [0., 3.]]
    for index, row in enumerate(rows):
        row['own_frame_native_xy'] = (np.asarray(original_frame) + [index*2., -index]).tolist()
    packet = dict(gap_mask=original_mask, own_frame_native_xy=original_frame)
    before = deepcopy(review)
    monkeypatch.setattr(display.comparison, '_review', lambda value: value)
    monkeypatch.setattr(display.comparison, '_model', lambda row:
        (geometries[int(row['run_id'])], entries[int(row['run_id'])][1], {}))
    monkeypatch.setattr(display.comparison, '_checked', lambda value, **kwargs: value)
    native_options = []
    def admitted(*args, **kwargs):
        native_options.append(kwargs)
        return dict(packet)
    monkeypatch.setattr(painted_analysis_exclusions, '_read_admitted_gap_preview', admitted)
    monkeypatch.setattr(display.comparison, '_camera_u_context', lambda *args: dict(raw_orientation=1))
    monkeypatch.setattr(raw_photograph, '_parameters', lambda *args: parameters())
    pixels = np.full((12, 14, 3), 123, np.uint8)
    translation = [[1., 0., 4.], [0., 1., 4.], [0., 0., 1.]]
    calls = []
    def full_raw(path, **options):
        calls.append((path, options))
        return dict(image_rgb=pixels, valid_support=np.ones(pixels.shape[:2], bool),
                    camera_u_to_photo_h=translation, sensor_wh=[8, 6], camera_u_wh=[6, 4],
                    extent_u_xywh=[-4, -4, 14, 12])
    monkeypatch.setattr(raw_full_display, 'read_full_raw_display', full_raw)
    first, second = [display.read_transfer_preview(review, str(i)) for i in range(2)]
    assert first['image_rgb'].shape == second['image_rgb'].shape
    assert native_options == [dict(include_native_rgb=False)] * 2
    assert first['camera_u_to_photo_h'] == second['camera_u_to_photo_h']
    assert calls[0][1]['extent'] == calls[1][1]['extent']
    assert first['review_crop_xywh'] == second['review_crop_xywh']
    assert first['review_crop_xywh'] == first['display_grid']['review_crop_xywh']
    left, top, width, height = first['review_crop_xywh']
    for index, row in enumerate(rows):
        frame = np.asarray(row['own_frame_native_xy'])
        shown = np.column_stack((frame, np.ones(4))) @ np.asarray((first, second)[index]['native_to_photo_h']).T
        xy = shown[:, :2] / shown[:, 2, None]
        assert (xy > [left, top]).all() and (xy < [left+width-1, top+height-1]).all()
    for index, result in enumerate((first, second)):
        np.testing.assert_allclose(result['native_to_photo_h'], np.asarray(translation) @ np.linalg.inv(entries[index][1]))
        assert result['gap_mask'] is original_mask
        assert result['own_frame_native_xy'] is original_frame
        assert result['full_camera_photograph'] is True
        assert result['full_raw_sensor_field'] is True and result['camera_jpg_display_only'] is False
        assert int(result['valid_support'].sum()) == 168
        assert np.all(result['image_rgb'][result['valid_support']] == 123)
    assert review == before


def test_full_photo_layout_rejects_a_projection_horizon():
    entries = models()
    entries[0][1][2] = [1., 0., -3.]
    with pytest.raises(ValueError, match='horizon'):
        display._transfer_canvas(entries)


def test_common_extent_also_shows_a_saved_rig_frame_outside_camera_coverage():
    entries = models()
    frame = np.array([[-6., -3.], [9., -3.], [9., 8.], [-6., 8.]])
    entries[0][2]['own_frame_native_xy'] = frame.tolist()
    shape, offset = display._transfer_canvas(entries)
    shown = np.column_stack((frame, np.ones(4))) @ offset.T
    assert (shown[:, :2] > 0).all()
    assert (shown[:, 0] < shape[1]-1).all() and (shown[:, 1] < shape[0]-1).all()


def test_layout_cache_reuses_exact_cohort_but_rejects_changed_unselected_model(tmp_path, monkeypatch):
    from darsia.fluidflower.native import raw_photograph
    from tests.unit.test_raw_full_display import parameters
    entries = models()
    rows, calls = [], []
    def bound(path):
        raw = path.read_bytes()
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    profile_path = tmp_path / 'profile.json'
    profile_path.write_text(json.dumps(parameters()))
    profile = dict(bound(profile_path), profile_id='test')
    for index, (_, _, template) in enumerate(entries):
        path = tmp_path / f'model-{index}.json'
        path.write_text(json.dumps(dict(run=index)))
        rows.append(dict(template, run_id=str(index), target_model=bound(path),
            own_frame_native_xy=[[0., 0.], [4., 0.], [4., 3.], [0., 3.]]))
    review_path = tmp_path / 'review.json'
    review_path.write_text(json.dumps(dict(runs=rows)))
    review = dict(runs=rows, review_binding=bound(review_path))
    def model(row):
        display.comparison._checked(row['target_model'], metadata=True)
        calls.append(row['run_id'])
        return SimpleNamespace(model=dict(camera_profile=profile), entry={}), entries[int(row['run_id'])][1], {}
    monkeypatch.setattr(display.comparison, '_model', model)
    monkeypatch.setattr(raw_photograph, '_parameters', lambda *args:
        display.comparison._checked(profile, metadata=True))
    first = display._transfer_layout(review)
    second = display._transfer_layout(review)
    assert calls == ['0', '1']
    assert first[1] == second[1]
    for left, right in zip(first[0], second[0]):
        np.testing.assert_array_equal(left[1], right[1])
    first[0][0][1][0, 0] = 99.
    assert display._transfer_layout(review)[0][0][1][0, 0] == 1.
    # A different run's model affects the fixed cohort viewport too.
    (tmp_path / 'model-1.json').write_text(json.dumps(dict(run=9)))
    with pytest.raises(ValueError):
        display._transfer_layout(review)


@pytest.mark.parametrize('damage', [None, 'pixels', 'nonfinite'])
def test_transfer_authenticates_linear_native_rgb_without_guessing_display_encoding(tmp_path, damage):
    from darsia.fluidflower import painted_analysis_exclusions as gaps
    from darsia.fluidflower.series_geometry import RigFrame
    image = np.full((2, 3, 3), 1.25, np.float32)
    image[0, 0] = -.01
    if damage == 'nonfinite':
        image[0, 0, 0] = np.nan
    photo = tmp_path / 'native.npz'
    np.savez(photo, array=image)
    mask = np.zeros((2, 3), bool)
    product = tmp_path / 'labels.npz'
    np.savez(product, array=np.ones((2, 3), np.uint8), filled=mask, support=~mask,
        inside=~mask, continued=mask, effective_coverage=~mask, gaps=mask)
    frame = RigFrame(corners=dict(top_left=(0., 0.), top_right=(0., 2.),
        bottom_right=(1., 2.), bottom_left=(1., 0.)), width_m=.9, height_m=.49,
        run_id='sample', source='synthetic measured frame', image=str(photo), image_size=(3, 2))
    row = dict(run_id='sample', product=gaps._bound(product), native_photo=gaps._bound(photo),
        native_photo_pixels=gaps._identity(image), gap=dict(identity=gaps._identity(mask)),
        own_frame_native_xy=[[0., 0.], [2., 0.], [2., 1.], [0., 1.]],
        transported_frame_native_xy=[[0., 0.], [2., 0.], [2., 1.], [0., 1.]],
        native_to_painting_h=np.eye(3).tolist(), contract=dict(source_model=dict(shape_hw=[2, 3])))
    review = dict(runs=[row], source=dict(reference_run='sample', painting_facies=dict(frame=frame.to_dict())),
        painting_photograph=row['native_photo'], map_version=dict(name='synthetic'))
    if damage == 'pixels':
        row['native_photo_pixels']['sha256'] = '0' * 64
    if damage is not None:
        with pytest.raises(ValueError, match='native RGB'):
            gaps._read_admitted_gap_preview(review, 'sample', include_native_rgb=False)
        return
    packet = gaps._read_admitted_gap_preview(review, 'sample', include_native_rgb=False)
    assert 'image_rgb' not in packet and 'image_encoding' not in packet
    np.testing.assert_array_equal(packet['gap_mask'], mask)
    with np.load(photo) as saved:
        np.testing.assert_array_equal(saved['array'], image)
    # The ordinary display reader still refuses an undeclared 0..1 encoding.
    with pytest.raises(ValueError, match='display encoding'):
        gaps._read_admitted_gap_preview(review, 'sample')
