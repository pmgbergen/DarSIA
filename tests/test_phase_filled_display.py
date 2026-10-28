"""Saved display flags retain their coordinates and strict source bindings."""
import hashlib
import json

import numpy as np
import pytest

from darsia.fluidflower.phase_filled_display import resolve_filled_display_mask
from darsia.fluidflower.phase_rig_display_canvas import pad_rig_display_mask


@pytest.fixture
def saved_view(tmp_path):
    from darsia.fluidflower.native.config import configure_parameters, reset_parameters

    directory = tmp_path / "native_frames" / "sample"
    directory.mkdir(parents=True)
    shape = (10, 12)
    observed = np.zeros(shape, bool)
    observed[1:4, 2:6] = True
    unobserved = np.zeros(shape, bool)
    unobserved[8:, 3:9] = True
    domain = np.ones(shape, bool)
    fields = dict(modelled=domain & ~(observed | unobserved),
                  filled_observed=observed, filled_unobserved=unobserved,
                  rig_domain=domain)
    classes = directory / "classes.npz"
    np.savez_compressed(classes, **fields)
    revision = dict(contract="ffac-rig-domain-revision-v1", canvas_wh=[12, 8],
                    phase_shape_hw=[10, 12], native_origin_xy=[-2, -1],
                    classes=dict(path=str(classes),
                                 sha256=hashlib.sha256(classes.read_bytes()).hexdigest()))
    native = dict(canvas_wh=[12, 10], phase_shape_hw=[10, 12],
                  analysis_image_shape_hw=[4, 8], content_box_xywh=[2, 1, 8, 4],
                  frame_sources=[dict(frame_index=0, rig_domain_revision=revision)])
    manifest = directory / "manifest.json"

    def write():
        manifest.write_text(json.dumps(native), encoding="utf-8")

    write()
    token = configure_parameters({
        "saved_review_metadata": {"display_views": {"rig_contract": revision["contract"]}}
    })
    try:
        yield tmp_path, native, fields, classes, manifest, write
    finally:
        reset_parameters(token)


def test_declared_bottom_extension_preserves_every_saved_flag(saved_view):
    root, _native, fields, classes, manifest, _write = saved_view
    before = (classes.read_bytes(), manifest.read_bytes())
    expected = fields["filled_observed"] | fields["filled_unobserved"]
    result = resolve_filled_display_mask(root, "sample", 0)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == bool and result.shape == (10, 12)
    assert result[8:, 3:9].all()  # Saved flags in appended rows remain present.
    assert (classes.read_bytes(), manifest.read_bytes()) == before


def test_existing_photographic_canvas_behavior_is_unchanged(saved_view):
    root, native, fields, _classes, _manifest, write = saved_view
    native["canvas_wh"] = [12, 8]
    write()
    np.testing.assert_array_equal(resolve_filled_display_mask(root, "sample", 0),
                                  fields["filled_observed"] | fields["filled_unobserved"])


def test_analysis_crop_keeps_exact_original_coordinates(saved_view):
    root, _native, fields, _classes, _manifest, _write = saved_view
    expected = (fields["filled_observed"] | fields["filled_unobserved"])[1:5, 2:10]
    np.testing.assert_array_equal(resolve_filled_display_mask(root, "sample", 0, canvas=False),
                                  expected)


@pytest.mark.parametrize("canvas_wh,phase_shape", [
    ([13, 10], [10, 12]), ([12, 11], [11, 12]),
    ([12, 7], [7, 12]), ([12, 10], None), ([12, 10], [9, 12]),
])
def test_unknown_canvas_changes_are_rejected(saved_view, canvas_wh, phase_shape):
    root, native, _fields, _classes, _manifest, write = saved_view
    native["canvas_wh"] = canvas_wh
    if phase_shape is None:
        native.pop("phase_shape_hw")
    else:
        native["phase_shape_hw"] = phase_shape
    write()
    with pytest.raises(ValueError, match="another review canvas"):
        resolve_filled_display_mask(root, "sample", 0)


def test_source_sha_failure_is_not_hidden_by_display_mapping(saved_view):
    root, _native, _fields, classes, _manifest, _write = saved_view
    with classes.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        resolve_filled_display_mask(root, "sample", 0)


def test_invalid_class_partition_is_still_rejected(saved_view):
    root, native, fields, classes, _manifest, write = saved_view
    fields["filled_observed"][0, 0] = True  # Overlaps the saved modelled class.
    np.savez_compressed(classes, **fields)
    native["frame_sources"][0]["rig_domain_revision"]["classes"]["sha256"] = hashlib.sha256(classes.read_bytes()).hexdigest()
    write()
    with pytest.raises(ValueError, match="disjoint complete partition"):
        resolve_filled_display_mask(root, "sample", 0)


def test_original_padding_helper_adds_only_false_bottom_rows():
    source = np.zeros((8, 12), bool)
    source[0, 0] = source[7, 11] = True
    original = source.copy()
    result = pad_rig_display_mask(source, dict(canvas_wh=[12, 8], phase_shape_hw=[10, 12]))
    np.testing.assert_array_equal(result[:8], original)
    assert not result[8:].any()
    np.testing.assert_array_equal(source, original)
