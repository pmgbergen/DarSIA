from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

import darsia
from darsia import make_coordinate
from darsia.multiphase.mass_analysis import CO2MassAnalysis
from darsia.presets.workflows.analysis.analysis_mass import analysis_mass_from_context
from darsia.presets.workflows.analysis.expert_knowledge import ExpertKnowledgeAdapter
from darsia.presets.workflows.config.roi import RoiConfig


def _make_co2_mass_analysis(shape: tuple[int, int]) -> CO2MassAnalysis:
    baseline = darsia.ScalarImage(np.zeros(shape, dtype=float), dimensions=[1.0, 1.0])
    return CO2MassAnalysis(baseline=baseline)


def test_inverse_mass_analysis_reconstructs_and_clips() -> None:
    co2 = _make_co2_mass_analysis((2, 2))
    co2.solubility_co2 = np.full((2, 2), 2.0)
    co2.density_gaseous_co2 = np.full((2, 2), 10.0)

    mass = darsia.ScalarImage(
        np.array([[0.5, 2.0], [3.0, 5.0]], dtype=float),
        dimensions=[1.0, 1.0],
    )
    result = co2.inverse_mass_analysis(mass)

    expected_c_aq = np.array([[0.25, 1.0], [1.0, 1.0]])
    expected_s_g = np.array([[0.0, 0.0], [0.125, 0.375]])
    expected_mass_g = 10.0 * expected_s_g
    expected_mass_aq = 2.0 * expected_c_aq * (1.0 - expected_s_g)
    expected_mass = expected_mass_g + expected_mass_aq

    assert np.allclose(result.concentration_aq.img, expected_c_aq)
    assert np.allclose(result.saturation_g.img, expected_s_g)
    assert np.allclose(result.mass_g.img, expected_mass_g)
    assert np.allclose(result.mass_aq.img, expected_mass_aq)
    assert np.allclose(result.mass.img, expected_mass)


def test_inverse_mass_analysis_handles_zero_denominator() -> None:
    co2 = _make_co2_mass_analysis((2, 2))
    co2.solubility_co2 = np.full((2, 2), 2.0)
    co2.density_gaseous_co2 = np.full((2, 2), 2.0)

    mass = darsia.ScalarImage(np.full((2, 2), 10.0), dimensions=[1.0, 1.0])
    result = co2.inverse_mass_analysis(mass)

    assert np.isfinite(result.saturation_g.img).all()
    assert np.isfinite(result.concentration_aq.img).all()
    assert np.allclose(result.saturation_g.img, 0.0)
    assert np.allclose(result.concentration_aq.img, 1.0)


def test_analysis_mass_writes_default_artifacts_only(
    tmp_path: Path,
) -> None:
    injected_mass = 8.0
    stream_payloads: list[dict[str, bytes]] = []
    image_path = tmp_path / "img001.png"

    class _FakeInjectionProtocol:
        def injected_mass(self, *, date: datetime | None, roi: Any = None) -> float:
            _ = (date, roi)
            return injected_mass

    class _FakeFluidFlower:
        def __init__(self) -> None:
            self.geometry = darsia.Geometry(
                space_dim=2, num_voxels=(4, 4), dimensions=[1.0, 1.0]
            )

        def read_image(self, _path: Path) -> darsia.Image:
            return darsia.Image(
                np.zeros((4, 4, 3), dtype=np.uint8),
                dimensions=[1.0, 1.0],
                scalar=False,
                time=1.0,
                date=datetime(2025, 1, 1, 0, 0, 0),
                name="img001",
            )

    class _FakeColorToMass:
        def __init__(self, co2_mass_analysis: CO2MassAnalysis) -> None:
            self.co2_mass_analysis = co2_mass_analysis

        def __call__(self, image: darsia.Image) -> darsia.SimpleMassAnalysisResults:
            c_aq = darsia.ScalarImage(
                np.full((4, 4), 0.4, dtype=float),
                dimensions=[1.0, 1.0],
                time=image.time,
                date=image.date,
                name=image.name,
            )
            s_g = darsia.ScalarImage(
                np.full((4, 4), 0.2, dtype=float),
                dimensions=[1.0, 1.0],
                time=image.time,
                date=image.date,
                name=image.name,
            )
            return self.co2_mass_analysis.mass_analysis(c_aq=c_aq, s_g=s_g)

    co2 = _make_co2_mass_analysis((4, 4))
    co2.solubility_co2 = np.full((4, 4), 2.0)
    co2.density_gaseous_co2 = np.full((4, 4), 10.0)

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            data=SimpleNamespace(results=tmp_path),
            analysis=SimpleNamespace(mass=SimpleNamespace(roi={}, roi_and_label={})),
        ),
        experiment=SimpleNamespace(injection_protocol=_FakeInjectionProtocol()),
        fluidflower=_FakeFluidFlower(),
        image_paths=[image_path],
        color_to_mass_analysis=_FakeColorToMass(co2),
        analysis_labels=darsia.ScalarImage(np.zeros((4, 4), dtype=np.uint8)),
        expert_knowledge_adapter=None,
    )

    def _stream_callback(payload):
        if payload is not None:
            stream_payloads.append(payload)

    analysis_mass_from_context(ctx, stream_callback=_stream_callback)

    assert (tmp_path / "mass" / "npz" / "img001.npz").exists()
    assert (tmp_path / "mass" / "jpg" / "img001.jpg").exists()
    non_default_products = [
        "rescaled_mass",
        "extensive_mass",
        "extensive_rescaled_mass",
        "saturation_g",
        "rescaled_saturation_g",
        "concentration_aq",
        "rescaled_concentration_aq",
    ]
    for product in non_default_products:
        assert not (tmp_path / product).exists()

    assert (tmp_path / "sparse_data" / "integrated_mass.csv").exists()
    assert len(stream_payloads) == 1
    assert "mass_total" in stream_payloads[0]
    assert "rescaled_mass" in stream_payloads[0]
    assert "rescaled_saturation_g" in stream_payloads[0]
    assert "rescaled_concentration_aq" in stream_payloads[0]


def test_analysis_mass_writes_configured_export_subset_with_extensive_modes(
    tmp_path: Path,
) -> None:
    injected_mass = 8.0
    image_path = tmp_path / "img001.png"
    configured_export = [
        "rescaled_mass",
        "extensive_mass",
        "extensive_rescaled_mass",
    ]

    class _FakeInjectionProtocol:
        def injected_mass(self, *, date: datetime | None, roi: Any = None) -> float:
            _ = (date, roi)
            return injected_mass

    class _FakeFluidFlower:
        def __init__(self) -> None:
            self.geometry = darsia.Geometry(
                space_dim=2, num_voxels=(4, 4), dimensions=[1.0, 1.0]
            )

        def read_image(self, _path: Path) -> darsia.Image:
            return darsia.Image(
                np.zeros((4, 4, 3), dtype=np.uint8),
                dimensions=[1.0, 1.0],
                scalar=False,
                time=1.0,
                date=datetime(2025, 1, 1, 0, 0, 0),
                name="img001",
            )

    class _FakeColorToMass:
        def __init__(self, co2_mass_analysis: CO2MassAnalysis) -> None:
            self.co2_mass_analysis = co2_mass_analysis

        def __call__(self, image: darsia.Image) -> darsia.SimpleMassAnalysisResults:
            c_aq = darsia.ScalarImage(
                np.full((4, 4), 0.4, dtype=float),
                dimensions=[1.0, 1.0],
                time=image.time,
                date=image.date,
                name=image.name,
            )
            s_g = darsia.ScalarImage(
                np.full((4, 4), 0.2, dtype=float),
                dimensions=[1.0, 1.0],
                time=image.time,
                date=image.date,
                name=image.name,
            )
            return self.co2_mass_analysis.mass_analysis(c_aq=c_aq, s_g=s_g)

    co2 = _make_co2_mass_analysis((4, 4))
    co2.solubility_co2 = np.full((4, 4), 2.0)
    co2.density_gaseous_co2 = np.full((4, 4), 10.0)

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            data=SimpleNamespace(results=tmp_path),
            analysis=SimpleNamespace(
                mass=SimpleNamespace(roi={}, roi_and_label={}, export=configured_export)
            ),
        ),
        experiment=SimpleNamespace(injection_protocol=_FakeInjectionProtocol()),
        fluidflower=_FakeFluidFlower(),
        image_paths=[image_path],
        color_to_mass_analysis=_FakeColorToMass(co2),
        analysis_labels=darsia.ScalarImage(np.zeros((4, 4), dtype=np.uint8)),
        expert_knowledge_adapter=None,
    )

    analysis_mass_from_context(ctx)

    selected_products = [
        "rescaled_mass",
        "extensive_mass",
        "extensive_rescaled_mass",
    ]
    for product in selected_products:
        assert (tmp_path / product / "npz" / "img001.npz").exists()
        assert (tmp_path / product / "jpg" / "img001.jpg").exists()
    for product in [
        "mass",
        "saturation_g",
        "rescaled_saturation_g",
        "concentration_aq",
        "rescaled_concentration_aq",
    ]:
        assert not (tmp_path / product).exists()


def test_analysis_mass_applies_expert_knowledge_to_rescaled_fields(
    tmp_path: Path,
) -> None:
    injected_mass = 8.0
    image_path = tmp_path / "img001.png"
    roi = RoiConfig()
    roi.roi = make_coordinate([[0.0, 0.0], [0.5, 1.0]])
    roi.name = "left_half"
    adapter = ExpertKnowledgeAdapter(
        saturation_g_rois={"left_half": roi},
        concentration_aq_rois={"left_half": roi},
    )

    class _FakeInjectionProtocol:
        def injected_mass(self, *, date: datetime | None, roi: Any = None) -> float:
            _ = (date, roi)
            return injected_mass

    class _FakeFluidFlower:
        def __init__(self) -> None:
            self.geometry = darsia.Geometry(
                space_dim=2, num_voxels=(4, 4), dimensions=[1.0, 1.0]
            )

        def read_image(self, _path: Path) -> darsia.Image:
            return darsia.Image(
                np.zeros((4, 4, 3), dtype=np.uint8),
                dimensions=[1.0, 1.0],
                scalar=False,
                time=1.0,
                date=datetime(2025, 1, 1, 0, 0, 0),
                name="img001",
            )

    class _FakeColorToMass:
        def __init__(self, co2_mass_analysis: CO2MassAnalysis) -> None:
            self.co2_mass_analysis = co2_mass_analysis

        def __call__(self, image: darsia.Image) -> darsia.SimpleMassAnalysisResults:
            c_aq = darsia.ScalarImage(
                np.full((4, 4), 0.4, dtype=float),
                dimensions=[1.0, 1.0],
                time=image.time,
                date=image.date,
                name=image.name,
            )
            s_g = darsia.ScalarImage(
                np.full((4, 4), 0.2, dtype=float),
                dimensions=[1.0, 1.0],
                time=image.time,
                date=image.date,
                name=image.name,
            )
            return self.co2_mass_analysis.mass_analysis(c_aq=c_aq, s_g=s_g)

    co2 = _make_co2_mass_analysis((4, 4))
    co2.solubility_co2 = np.full((4, 4), 2.0)
    co2.density_gaseous_co2 = np.full((4, 4), 10.0)

    ctx = SimpleNamespace(
        config=SimpleNamespace(
            data=SimpleNamespace(results=tmp_path),
            analysis=SimpleNamespace(mass=SimpleNamespace(roi={}, roi_and_label={})),
        ),
        experiment=SimpleNamespace(injection_protocol=_FakeInjectionProtocol()),
        fluidflower=_FakeFluidFlower(),
        image_paths=[image_path],
        color_to_mass_analysis=_FakeColorToMass(co2),
        analysis_labels=darsia.ScalarImage(np.zeros((4, 4), dtype=np.uint8)),
        expert_knowledge_adapter=adapter,
    )

    analysis_mass_from_context(ctx)

    sat = darsia.imread(tmp_path / "rescaled_saturation_g" / "npz" / "img001.npz")
    caq = darsia.imread(tmp_path / "rescaled_concentration_aq" / "npz" / "img001.npz")
    assert np.any(sat.img == 0.0)
    assert np.any(caq.img == 0.0)
