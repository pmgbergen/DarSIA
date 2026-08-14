import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import darsia
from darsia.presets.workflows.helper.helper_roi import (
    _box_from_rectangle_selection,
    _box_from_zoom_limits,
    _corners_from_box,
    _scalar_image_for_mode,
    format_roi_template,
    helper_roi,
)

helper_roi_module = importlib.import_module(
    "darsia.presets.workflows.helper.helper_roi"
)


def test_format_roi_template_contains_corner_values() -> None:
    template = format_roi_template(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert "[roi.roi_name]" in template
    assert 'name = "roi_name"' in template
    assert "corner_1 = [1, 2]" in template
    assert "corner_2 = [3, 4]" in template


def test_box_from_zoom_limits_clips_to_shape() -> None:
    box = _box_from_zoom_limits(xlim=(-5.0, 8.2), ylim=(2.1, 100.0), shape=(10, 20))
    assert box is not None
    rows, cols = box
    assert rows.start == 2
    assert rows.stop == 10
    assert cols.start == 0
    assert cols.stop == 9


def test_box_from_rectangle_selection_handles_none_event_data() -> None:
    box = _box_from_rectangle_selection(
        x_press=None,
        y_press=1.0,
        x_release=2.0,
        y_release=3.0,
        shape=(10, 20),
    )
    assert box is None


def test_box_from_rectangle_selection_clips_to_shape() -> None:
    box = _box_from_rectangle_selection(
        x_press=-2.0,
        y_press=1.1,
        x_release=8.5,
        y_release=100.0,
        shape=(10, 20),
    )
    assert box is not None
    rows, cols = box
    assert rows.start == 1
    assert rows.stop == 10
    assert cols.start == 0
    assert cols.stop == 9


def test_corners_from_box_returns_two_coordinates() -> None:
    image = darsia.ScalarImage(np.ones((10, 20), dtype=float), dimensions=[1.0, 2.0])
    corner_1, corner_2 = _corners_from_box(image, (slice(1, 4), slice(2, 6)))
    assert corner_1.shape == (2,)
    assert corner_2.shape == (2,)
    assert not np.allclose(corner_1, corner_2)


def test_scalar_image_for_mode_maps_mass_to_mass_total(monkeypatch) -> None:
    mass_image = darsia.ScalarImage(np.ones((2, 2), dtype=float), dimensions=[1.0, 1.0])
    called = {"requested_modes": None}

    def _fake_analysis_scalar_products(
        *, mass_analysis_result, requested_modes, **kwargs
    ):
        del mass_analysis_result, kwargs
        called["requested_modes"] = set(requested_modes)
        return {"mass_total": mass_image}, None

    monkeypatch.setattr(
        helper_roi_module, "analysis_scalar_products", _fake_analysis_scalar_products
    )

    result = _scalar_image_for_mode(
        mode="mass",
        result=SimpleNamespace(),
        fluidflower=SimpleNamespace(geometry=SimpleNamespace()),
        experiment=SimpleNamespace(injection_protocol=SimpleNamespace()),
        color_to_mass_analysis=SimpleNamespace(co2_mass_analysis=SimpleNamespace()),
        img_date=None,
    )
    assert called["requested_modes"] == {"mass_total"}
    assert result is mass_image


def test_helper_roi_mode_none_dispatches_viewer_with_source_images(
    monkeypatch,
) -> None:
    img = darsia.ScalarImage(np.ones((4, 4), dtype=float), dimensions=[1.0, 1.0])
    captured: dict[str, object] = {}

    class _FakeRigInstance:
        baseline = img
        geometry = SimpleNamespace()

        def load_experiment(self, experiment):
            del experiment

    class _FakeRig:
        @classmethod
        def load(cls, rig_path, corrections):
            del cls, rig_path, corrections
            return _FakeRigInstance()

    class _FakeConfig:
        def __init__(self):
            self.data = SimpleNamespace(
                use_cache=False,
                cache=None,
                registry=SimpleNamespace(),
            )
            self.rig = SimpleNamespace(path=Path("/tmp/rig.npz"))
            self.helper = SimpleNamespace(
                roi=SimpleNamespace(mode="none", data=SimpleNamespace())
            )
            self.corrections = None
            self.color_to_mass = None

        def check(self, *args):
            del args

    monkeypatch.setattr(
        helper_roi_module,
        "FluidFlowerConfig",
        lambda path, require_data, require_results: _FakeConfig(),
    )
    monkeypatch.setattr(
        helper_roi_module.darsia.ProtocolledExperiment,
        "init_from_config",
        lambda config: SimpleNamespace(injection_protocol=SimpleNamespace()),
    )
    monkeypatch.setattr(
        helper_roi_module,
        "select_image_paths",
        lambda *args, **kwargs: [Path("/tmp/img_001.jpg")],
    )
    monkeypatch.setattr(
        helper_roi_module, "load_images_with_cache", lambda **kwargs: [img]
    )
    monkeypatch.setattr(
        helper_roi_module,
        "launch_roi_helper_viewer",
        lambda images, mode: captured.update({"images": images, "mode": mode}),
    )

    helper_roi(_FakeRig, Path("/tmp/config.toml"))
    assert captured["mode"] == "none"
    assert captured["images"] == [img]
