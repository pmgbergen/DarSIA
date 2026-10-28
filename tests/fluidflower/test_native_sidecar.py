"""Lossless writers keep exact scientific values and reject changed publications."""

import numpy as np
import pytest
from PIL import Image

from darsia.fluidflower.native import sidecar
from darsia.fluidflower.native.io import sha256_file


@pytest.fixture
def parameters():
    return dict(
        raw_detail_floor=2.0,
        raw_detail_ceiling=65535.0,
        raw_detail_exponent=0.75,
        raw_detail_invalid=1.0,
        maximum_invalid_fraction=0.001,
        unit_denominator=65535,
        nonclipping_contract="nonclipping",
        exact_valid_mask_policy="exact",
        legacy_fraction_policy="fraction",
        jpeg_format="JPEG",
        jpeg_quality=100,
        jpeg_subsampling=0,
        mask_archive_key="invalid_mask",
        base_mode="base",
        extra_modes=["lift"],
        field_keys={
            name: "stored_" + name
            for name in (
                "photo_sand_mask_path",
                "photo_sand_mask_sha256",
                "filled_invalid_mask_path",
                "filled_invalid_mask_sha256",
                "display_invalid_mask_path",
                "display_invalid_mask_sha256",
                "display_modes",
            )
        },
    )


def test_array_writer_preserves_nan_and_reconciles_only_equal_content(tmp_path):
    values = np.array([[1, np.nan], [3, 4]], np.float32)
    path = tmp_path / "signal.npy"
    digest = sidecar.write_log_linear(path, values)
    assert digest == sha256_file(path)
    assert np.array_equal(np.load(path), values, equal_nan=True)
    assert sidecar.write_log_linear(path, values) == digest
    with pytest.raises(ValueError, match="differs"):
        sidecar.write_log_linear(path, np.zeros_like(values))
    assert sha256_file(path) == digest
    copied = tmp_path / "copied.npy"
    assert sidecar.copy_verified(path, copied, digest) == digest
    assert copied.read_bytes() == path.read_bytes()


def test_float_qa_respects_required_support_without_clipping(parameters):
    signal = np.full((3, 2, 3), np.log(2), np.float32)
    signal[:, 0, 0] = np.nan
    required = np.ones((2, 3), bool)
    required[0, 0] = False
    qa = sidecar.fixed_float_input_qa(
        signal, required_valid_mask=required, parameters=parameters
    )
    assert qa["required_valid_pixels"] == 5 and qa["invalid_policy"] == "exact"
    signal[:, 1, 1] = np.float32(np.log(65535) + 1)
    with pytest.raises(ValueError, match="denominator"):
        sidecar.fixed_float_input_qa(
            signal, required_valid_mask=required, parameters=parameters
        )


def test_detail_view_and_mask_archive_are_separate_from_input(tmp_path, parameters):
    signal = np.zeros((3, 2, 3), np.float32)
    signal[:, 0, 0] = np.nan
    saved = signal.copy()
    rgb = sidecar.raw_detail_display(signal, parameters=parameters)
    assert np.all(rgb[0, 0] == 1) and np.all(rgb[1, 1] == 0)
    assert np.array_equal(saved, signal, equal_nan=True)
    mask = ~np.isfinite(signal).all(0)
    path = tmp_path / "mask.npz"
    sidecar.write_display_invalid_mask(path, mask, parameters=parameters)
    with np.load(path) as archive:
        assert np.array_equal(archive["invalid_mask"], mask)


def test_exact_ring_assembly_and_changed_centre_rejection(tmp_path):
    parameters = dict(
        contract="source-ring",
        maximum_center_delta_log=1e-5,
        unit_denominator=65535,
        field_keys={
            key: key
            for key in (
                "extended_ring_path",
                "extended_ring_sha256",
                "extended_pad_px",
                "extended_origin_xy",
            )
        },
    )
    padded = np.arange(3 * 6 * 7, dtype=np.float32).reshape(3, 6, 7) / 100
    core = padded[:, 1:-1, 1:-1].copy()
    strips, difference = sidecar.split_extended_ring(
        padded, core, pad_px=1, maximum_center_delta_log=1e-5, parameters=parameters
    )
    assert difference == 0 and strips["top"].shape == (3, 1, 7)
    path = tmp_path / "ring.npz"
    digest, _ = sidecar.write_extended_ring(
        path, padded, core, pad_px=1, parameters=parameters
    )
    restored = sidecar.assemble_extended_log(
        core, path, ring_sha256=digest, expected_pad_px=1, parameters=parameters
    )
    assert np.array_equal(restored, padded)
    frame = sidecar.CorrectedFrame(
        np.zeros((4, 5, 3), np.float32), core, {}, extended_log_linear_rgb=padded
    )
    metadata = sidecar.write_frame_ring(
        tmp_path / "frame_ring.npz", frame, core, pad_px=1, parameters=parameters
    )
    assert metadata["extended_valid_pixels"] == 42
    changed = core.copy()
    changed[0, 0, 0] += 0.1
    with pytest.raises(ValueError, match="centre"):
        sidecar.split_extended_ring(
            padded,
            changed,
            pad_px=1,
            maximum_center_delta_log=1e-5,
            parameters=parameters,
        )


def test_display_writer_and_optional_receipts_use_explicit_fields(tmp_path, parameters):
    def renderer(rgb, **kwargs):
        return np.rint(rgb * 255).astype(np.uint8)

    rgb = np.zeros((3, 4, 3), np.float32)
    rgb[..., 0] = 1
    target = tmp_path / "view.jpg"
    digest = sidecar.write_display_jpeg(
        rgb,
        run="sample",
        index=0,
        count=1,
        elapsed_h=0,
        canvas_wh=(4, 3),
        destination=target,
        renderer=renderer,
        parameters=parameters,
    )
    with Image.open(target) as image:
        assert image.size == (4, 3)
    frame = sidecar.CorrectedFrame(
        rgb,
        np.zeros((3, 3, 4), np.float32),
        {},
        display_modes={"base": rgb, "lift": rgb},
        photo_sand_mask=np.ones((3, 4), bool),
        display_invalid_mask=np.zeros((3, 4), bool),
    )
    fields = sidecar.write_auxiliary_outputs(
        frame,
        target,
        digest,
        run="sample",
        index=0,
        count=1,
        elapsed_h=0,
        canvas_wh=(4, 3),
        renderer=renderer,
        parameters=parameters,
    )
    assert fields["stored_display_modes"]["base"]["jpeg_sha256"] == digest
    assert set(
        sidecar.auxiliary_output_fields(
            dict(fields, unrelated=1), parameters=parameters
        )
    ) == set(fields)
    with pytest.raises(ValueError, match="geometry"):
        sidecar.write_display_jpeg(
            rgb,
            run="sample",
            index=0,
            count=1,
            elapsed_h=0,
            canvas_wh=(5, 3),
            destination=tmp_path / "wrong.jpg",
            renderer=renderer,
            parameters=parameters,
        )
