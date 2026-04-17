from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import darsia

from darsia.presets.workflows.analysis.analysis_thresholding import (
    analysis_thresholding_from_context,
)
from darsia.presets.workflows.analysis.streaming import (
    encode_low_resolution_png,
    publish_stream_images,
)
from darsia.presets.workflows.config.analysis import AnalysisThresholdingConfig
from darsia.presets.workflows.user_interface_analysis import run_analysis


def test_encode_low_resolution_png_handles_scalar_and_color_arrays() -> None:
    scalar = np.random.rand(32, 48).astype(np.float32)
    color = np.random.rand(32, 48, 3).astype(np.float32)

    scalar_png = encode_low_resolution_png(scalar)
    color_png = encode_low_resolution_png(color)

    assert isinstance(scalar_png, bytes)
    assert isinstance(color_png, bytes)
    assert len(scalar_png) > 0
    assert len(color_png) > 0


def test_publish_stream_images_encodes_images() -> None:
    received = []

    def _callback(payload):
        received.append(payload)

    publish_stream_images(
        stream_callback=_callback,
        image_payload={
            "a": np.random.rand(20, 20),
            "b": np.random.rand(20, 20, 3),
        },
        logger=logging.getLogger(__name__),
        error_message="stream failed",
    )

    assert len(received) == 1
    assert set(received[0].keys()) == {"a", "b"}
    assert isinstance(received[0]["a"], bytes)
    assert isinstance(received[0]["b"], bytes)


def test_run_analysis_forwards_stream_callback_to_all_modes(monkeypatch) -> None:
    recorded: dict[str, object] = {}
    stream_callback = lambda payload: payload  # noqa: E731
    fake_ctx = object()

    def _capture(name):
        def _inner(ctx, **kwargs):
            recorded[name] = kwargs.get("stream_callback")
            assert ctx is fake_ctx

        return _inner

    monkeypatch.setattr(
        "darsia.presets.workflows.user_interface_analysis.prepare_analysis_context",
        lambda **kwargs: fake_ctx,
    )
    monkeypatch.setattr(
        "darsia.presets.workflows.user_interface_analysis.analysis_cropping_from_context",
        _capture("cropping"),
    )
    monkeypatch.setattr(
        "darsia.presets.workflows.user_interface_analysis.analysis_mass_from_context",
        _capture("mass"),
    )
    monkeypatch.setattr(
        "darsia.presets.workflows.user_interface_analysis.analysis_volume_from_context",
        _capture("volume"),
    )
    monkeypatch.setattr(
        "darsia.presets.workflows.user_interface_analysis.analysis_segmentation_from_context",
        _capture("segmentation"),
    )
    monkeypatch.setattr(
        "darsia.presets.workflows.user_interface_analysis.analysis_fingers_from_context",
        _capture("fingers"),
    )
    monkeypatch.setattr(
        "darsia.presets.workflows.user_interface_analysis.analysis_thresholding_from_context",
        _capture("thresholding"),
    )

    args = SimpleNamespace(
        config=["/tmp/config.toml"],
        all=False,
        cropping=True,
        mass=True,
        volume=True,
        segmentation=True,
        fingers=True,
        thresholding=True,
        show=False,
        save_jpg=False,
        save_npz=False,
    )
    run_analysis(rig_cls=object, args=args, stream_callback=stream_callback)

    assert recorded == {
        "cropping": stream_callback,
        "mass": stream_callback,
        "volume": stream_callback,
        "segmentation": stream_callback,
        "fingers": stream_callback,
        "thresholding": stream_callback,
    }


def test_thresholding_writes_separated_formats_and_streams_layer_keys(
    tmp_path: Path,
) -> None:
    thresholding_config = AnalysisThresholdingConfig().load(
        sec={
            "thresholding": {
                "formats": ["jpg", "npz"],
                "layers": {
                    "gas": {
                        "mode": "saturation_g",
                        "threshold_min": 0.2,
                        "label": "Gas plume",
                        "fill": [255, 0, 0],
                        "stroke": [255, 255, 255],
                    }
                },
            }
        },
        results=tmp_path,
    )

    image_path = tmp_path / "img001.png"
    stream_payloads: list[dict[str, bytes]] = []

    class _FakeImage:
        def __init__(self, img: np.ndarray) -> None:
            self.img = img

    class _FakeFluidFlower:
        def read_image(self, path: Path) -> _FakeImage:
            del path
            return _FakeImage(np.zeros((16, 24, 3), dtype=np.uint8))

    def _fake_color_to_mass(img: _FakeImage):
        del img
        scalar = np.zeros((16, 24), dtype=np.float32)
        scalar[2:8, 4:12] = 0.25
        scalar[10:14, 14:20] = 0.9
        scalar = _FakeImage(scalar)
        return SimpleNamespace(
            concentration_aq=scalar,
            saturation_g=scalar,
            mass=scalar,
            mass_g=scalar,
            mass_aq=scalar,
        )

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            data=SimpleNamespace(results=tmp_path),
            analysis=SimpleNamespace(thresholding=thresholding_config),
        ),
        experiment=SimpleNamespace(
            injection_protocol=SimpleNamespace(
                injected_mass=lambda date=None, **_: 1.0
            )
        ),
        fluidflower=_FakeFluidFlower(),
        image_paths=[image_path],
        color_to_mass_analysis=_fake_color_to_mass,
    )

    def _stream_callback(payload):
        if payload is not None:
            stream_payloads.append(payload)

    analysis_thresholding_from_context(ctx, stream_callback=_stream_callback)

    assert (
        tmp_path / "thresholding" / "jpg" / "gas" / "img001.jpg"
    ).exists(), "JPG output should be written to format-specific subfolder."
    assert (
        tmp_path / "thresholding" / "jpg" / "all" / "img001.jpg"
    ).exists(), "Combined JPG output should be written to jpg/all subfolder."
    assert (
        tmp_path / "thresholding" / "npz" / "gas" / "img001.npz"
    ).exists(), "NPZ output should be written to format-specific subfolder."
    assert len(stream_payloads) == 1
    assert "thresholding_source_image" in stream_payloads[0]
    assert "thresholding_gas" in stream_payloads[0]
    assert "thresholding_all" in stream_payloads[0]


def test_thresholding_supports_rescaled_layer_modes(tmp_path: Path) -> None:
    thresholding_config = AnalysisThresholdingConfig().load(
        sec={
            "thresholding": {
                "formats": ["npz"],
                "layers": {
                    "rescaled": {
                        "mode": "rescaled_mass",
                        "threshold_min": 0.2,
                        "label": "Rescaled mass plume",
                    }
                },
            }
        },
        results=tmp_path,
    )

    image_path = tmp_path / "img001.png"

    class _FakeImage:
        def __init__(self) -> None:
            self.img = np.zeros((16, 24, 3), dtype=np.uint8)
            self.date = None

    class _FakeFluidFlower:
        def __init__(self) -> None:
            self.geometry = darsia.Geometry(
                space_dim=2, num_voxels=(16, 24), dimensions=[1.0, 1.0]
            )

        def read_image(self, path: Path) -> _FakeImage:
            del path
            return _FakeImage()

    scalar = darsia.ScalarImage(
        np.full((16, 24), 0.5, dtype=float), dimensions=[1.0, 1.0]
    )

    class _FakeCo2Mass:
        def inverse_mass_analysis(self, mass):
            return SimpleNamespace(
                mass=mass,
                saturation_g=mass,
                concentration_aq=mass,
            )

    class _FakeColorToMass:
        def __init__(self) -> None:
            self.co2_mass_analysis = _FakeCo2Mass()

        def __call__(self, img):
            del img
            return SimpleNamespace(
                concentration_aq=scalar,
                saturation_g=scalar,
                mass=scalar,
                mass_g=scalar,
                mass_aq=scalar,
            )

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            data=SimpleNamespace(results=tmp_path),
            analysis=SimpleNamespace(thresholding=thresholding_config),
        ),
        experiment=SimpleNamespace(
            injection_protocol=SimpleNamespace(
                injected_mass=lambda date=None, **_: 1.0
            )
        ),
        fluidflower=_FakeFluidFlower(),
        image_paths=[image_path],
        color_to_mass_analysis=_FakeColorToMass(),
    )

    analysis_thresholding_from_context(ctx)

    assert (tmp_path / "thresholding" / "npz" / "rescaled" / "img001.npz").exists()
