"""Synthetic parity and producer/current binding tests for saved presentation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pytest

from darsia.fluidflower.native import display_modes as core

_ORIGINAL_COMMIT = "f0d338aa2be5e6ee8be6a2db0bd8f6fb70e9bd5a"


def _original(path, name, **externals):
    repo = Path(__file__).resolve().parents[2]
    source = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{_ORIGINAL_COMMIT}:{path}"]
    ).decode("utf-8")
    tree = ast.parse(source)
    class PassiveImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None if node.level else node
    tree = PassiveImports().visit(tree)
    module = types.ModuleType(name)
    module.__file__ = __file__
    module.__dict__.update(externals)
    sys.modules[name] = module
    exec(compile(ast.fix_missing_locations(tree), path, "exec"), module.__dict__)
    return module


@pytest.fixture
def original():
    display = _original(
        "src/darsia/fluidflower/tools/ffac_track1_display_modes.py",
        "_original_saved_display",
    )
    temporal = _original(
        "src/darsia/fluidflower/tools/ffac_v5_temporal_alignment.py",
        "_original_saved_temporal",
    )
    change = _original(
        "src/darsia/fluidflower/tools/ffac_track1_change_flatfield.py",
        "_original_saved_change", display=display, _h=temporal._h,
    )
    return display, change


@pytest.fixture
def parameters(original):
    display, change = original
    # Values are taken from the exact original source object, not retuned.
    dp = dict(
        modes=display.DISPLAY_MODES, gains=display.GAINS, base_mode="natural",
        matrix=display.M_NATURAL.tolist(),
        luminance_weights=display.LUMINANCE_WEIGHTS.tolist(),
        white_multiplier=0.95,
        fill_connectivity=8, fill_max_area_px=64, fill_radius_px=2,
        fill_max_iterations=64, highlight_threshold=0.55, highlight_width=0.25,
        chroma_median_size_px=5, unfilled_rgb8=40,
        srgb=dict(threshold=0.0031308, linear_gain=12.92, power_gain=1.055,
                  power_denominator=2.4, power_offset=0.055),
        contract_metadata=display.display_contract(),
    )
    cp = dict(
        contract=change.CONTRACT, box_size=change.BOX_SIZE,
        minimum_reference_luminance=change.MIN_REFERENCE_LUMINANCE,
        reference_luminance=display.Y_REFERENCE, lift_mode="lift1",
        tile_rows=5, box_radius=2, validity_tolerance=1e-6,
        homography_floor=1e-9,
        homography_error="V5 temporal alignment requires an invertible 3x3 H",
    )
    return dict(display_modes=dp, change_display=cp)


def test_lift_mode_uses_canonical_pixels_and_original_chroma(original, parameters):
    display, _ = original
    rng = np.random.default_rng(817)
    signal = rng.normal(4.1, 0.35, (3, 19, 23)).astype(np.float32)
    signal[:, 8, 11] = np.nan
    sand = np.indices((19, 23))[1] % 3 != 0
    saved = signal.copy()
    expected, _ = display.render_display_modes(signal, sand, tile_rows=5)
    calls = []
    def chroma(image):
        calls.append(image.shape)
        return display._median_chroma_rgb8(image)
    actual, _ = core.render_display_modes(
        signal, sand, tile_rows=5, parameters=parameters["display_modes"],
        selected_modes=("lift1",), chroma_filter=chroma,
    )
    assert list(actual) == ["lift1"]
    np.testing.assert_array_equal(actual["lift1"], expected["lift1"])
    np.testing.assert_array_equal(signal, saved)
    assert calls == [(19, 23, 3)]


def test_change_reference_warp_then_blur_and_outside_pixels(original, parameters):
    display, change = original
    rng = np.random.default_rng(1819)
    reference = rng.normal(4.2, 0.3, (3, 25, 27)).astype(np.float32)
    current = rng.normal(4.1, 0.4, (3, 25, 27)).astype(np.float32)
    reference[:, 7, 9] = np.nan
    reference[:, 13:23, 14:24] = np.nan
    support = np.ones((25, 27), bool)
    support[0:3] = False
    sand = np.indices((25, 27))[1] > 4
    h = np.array([[1., .005, .4], [-.003, 1., -.2], [.0002, -.0001, 1.]])
    old = change.prepare_flatfield_reference(reference, source_valid_mask=support, tile_rows=5)
    new = core.prepare_display_flatfield_reference(
        reference, source_valid_mask=support, tile_rows=5, parameters=parameters)
    np.testing.assert_array_equal(new.luminance, old.luminance)
    np.testing.assert_array_equal(new.valid, old.valid)
    kwargs = dict(target_shape_hw=(25, 27), source_origin_xy=(-2, 3), target_origin_xy=(1, -1))
    old_warp = change.register_flatfield_reference(old, h, **kwargs)
    new_warp = core.register_display_flatfield_reference(new, h, parameters=parameters, **kwargs)
    np.testing.assert_array_equal(new_warp.luminance_box5, old_warp.luminance_box5)
    np.testing.assert_array_equal(new_warp.valid, old_warp.valid)
    assert new_warp.alignment == old_warp.alignment
    modes, _ = display.render_display_modes(current, sand, tile_rows=5)
    lift = np.rint(modes["lift1"] * 255).astype(np.uint8)
    expected, old_meta = change.render_change_flatfield(current, old_warp, sand, lift1_rgb=lift, tile_rows=5)
    actual, new_meta = core.render_change_flatfield(
        current, new_warp, sand, lift1_rgb=lift, tile_rows=5, parameters=parameters)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual[~sand], lift[~sand])
    assert new_meta == old_meta


def test_reference_invalid_homography_preserves_exact_guard(parameters):
    ref = core.DisplayFlatfieldReference(
        np.ones((5, 6), np.float32), np.ones((5, 6), bool), {})
    for h in (np.zeros((3, 3)), np.eye(2), np.full((3, 3), np.nan)):
        with pytest.raises(ValueError, match="V5 temporal alignment requires an invertible 3x3 H"):
            core.register_display_flatfield_reference(
                ref, h, target_shape_hw=(5, 6), source_origin_xy=(0, 0),
                target_origin_xy=(0, 0), parameters=parameters)


def test_passive_producer_is_not_relabelled_current_code(tmp_path, monkeypatch):
    from darsia.fluidflower import phase_rig_display_prerender as cache
    from darsia.fluidflower import phase_saved_review_metadata as metadata
    current = tmp_path / "phase_rig_display_prerender.py"
    producer = tmp_path / "saved_producer.py"
    current.write_bytes(b"current renderer\\n")
    producer.write_bytes(b"saved original renderer\\n")
    def asset(path):
        return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    size_bytes=path.stat().st_size)
    p = dict(current_execution_sources={"current": asset(current)},
             producer_code_sources={"original": asset(producer)})
    monkeypatch.setattr(cache, "__file__", str(current))
    monkeypatch.setattr(metadata, "configured_parameters", lambda section: p)
    def bound(a, *, base=None, verify=True):
        path = Path(a["path"]).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != a["sha256"]:
            raise ValueError("Rig/camera bound source artifact changed")
        return path
    monkeypatch.setattr(metadata, "bound_path", bound)
    assert cache._renderer_code() == {"original": p["producer_code_sources"]["original"]["sha256"]}
    current.write_bytes(b"changed current renderer\\n")
    with pytest.raises(ValueError, match="bound source artifact changed"):
        cache._renderer_code()
