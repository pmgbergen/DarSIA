from pathlib import Path

import pytest

from darsia.presets.workflows.config.analysis import AnalysisConfig
from darsia.presets.workflows.config.color_embedding_registry import (
    ColorEmbeddingRegistry,
)
from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.format_registry import FormatRegistry
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def _load_color_registry(config_path: Path, tmp_path: Path) -> ColorEmbeddingRegistry:
    return ColorEmbeddingRegistry().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
    )


def test_analysis_cropping_formats_are_loaded(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.cropping]
formats = ["npz", "jpg"]
""".strip(),
    )

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
    )

    assert config.cropping is not None
    assert config.cropping.formats == ["npz", "jpg"]


def test_analysis_cropping_formats_reject_invalid_entries(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.cropping]
formats = ["npz", "png"]
""".strip(),
    )

    with pytest.raises(
        ValueError, match=r"Unsupported \[analysis\.cropping\]\.formats entries"
    ):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )


def test_analysis_thresholding_layers_and_formats_are_loaded(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
formats = ["jpg", "npz"]
[analysis.thresholding.layers.gas]
mode = "saturation_g"
threshold_min = 0.15
label = "Gas plume"
fill = [255, 0, 0]
stroke = [255, 255, 255]
[analysis.thresholding.layers.aq]
mode = "concentration_aq"
threshold_max = 0.05
label = "Aqueous plume"
fill = [0, 0, 255]
stroke = [255, 255, 255]
[analysis.thresholding.legend]
show = true
font_scale = 0.8
text_color = [255, 255, 255]
position = [10, 20]
box_enabled = true
box_color = [0, 0, 0]
box_alpha = 0.5
box_padding = 8
""".strip(),
    )

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
    )

    assert config.thresholding is not None
    assert config.thresholding.formats == ["jpg", "npz"]
    assert set(config.thresholding.layers.keys()) == {"gas", "aq"}
    assert config.thresholding.layers["gas"].mode == "saturation_g"
    assert config.thresholding.layers["gas"].threshold_min == 0.15
    assert config.thresholding.layers["gas"].label == "Gas plume"
    assert config.thresholding.layers["aq"].mode == "concentration_aq"
    assert config.thresholding.layers["aq"].threshold_max == 0.05


def test_analysis_thresholding_rejects_invalid_layer_mode(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
[analysis.thresholding.layers.bad]
mode = "not_supported"
threshold_min = 0.1
""".strip(),
    )

    with pytest.raises(ValueError, match=r"Unsupported analysis\.thresholding\.layers"):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )


def test_analysis_thresholding_rejects_invalid_formats(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
formats = ["jpg", "png"]
[analysis.thresholding.layers.gas]
mode = "saturation_g"
threshold_min = 0.1
""".strip(),
    )

    with pytest.raises(
        ValueError, match=r"Unsupported \[analysis\.thresholding\]\.formats entries"
    ):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )


def test_analysis_thresholding_accepts_extended_modes(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]

[analysis.thresholding]
[analysis.thresholding.layers.mass_rescaled]
mode = "rescaled_mass"
threshold_min = 0.1
[analysis.thresholding.layers.gas_rescaled]
mode = "rescaled_saturation_g"
threshold_min = 0.1
[analysis.thresholding.layers.aq_rescaled]
mode = "rescaled_concentration_aq"
threshold_min = 0.1
[analysis.thresholding.layers.red]
mode = "color.red_channel"
threshold_min = 0.2
[analysis.thresholding.layers.green_band]
mode = "color.green_band"
threshold_min = 0.5
""".strip(),
    )

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
    )

    assert config.thresholding is not None
    assert config.thresholding.layers["mass_rescaled"].mode == "rescaled_mass"
    assert config.thresholding.layers["gas_rescaled"].mode == "rescaled_saturation_g"
    assert config.thresholding.layers["aq_rescaled"].mode == "rescaled_concentration_aq"
    assert config.thresholding.layers["red"].mode == "color.red_channel"
    assert config.thresholding.layers["green_band"].mode == "color.green_band"


def test_analysis_thresholding_rejects_invalid_color_mode_token(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.thresholding]
[analysis.thresholding.layers.bad]
mode = "color.rgb.r"
threshold_min = 0.1
""".strip(),
    )

    with pytest.raises(
        ValueError,
        match=r"Unsupported analysis\.thresholding\.layers\.bad\.mode 'color\.rgb\.r'",
    ):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )


def test_analysis_segmentation_accepts_rescaled_mode(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.segmentation]
label = "Gas contour"
mode = "rescaled_saturation_g"
thresholds = [0.1]
color = [255, 0, 0]
""".strip(),
    )

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
    )

    assert config.segmentation is not None
    assert config.segmentation.config.mode == "rescaled_saturation_g"


def test_analysis_segmentation_rejects_invalid_mode(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.segmentation]
label = "Bad contour"
mode = "invalid_mode"
thresholds = [0.1]
color = [255, 0, 0]
""".strip(),
    )

    with pytest.raises(ValueError, match=r"Unsupported analysis\.segmentation\.mode"):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )


def test_analysis_segmentation_rejects_invalid_color_mode_token(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.segmentation]
label = "Bad contour"
mode = "color.rgb.r"
thresholds = [0.1]
color = [255, 0, 0]
""".strip(),
    )

    with pytest.raises(
        ValueError,
        match=r"Unsupported analysis\.segmentation\.mode 'color\.rgb\.r'",
    ):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )


def test_analysis_data_can_be_resolved_from_data_registry(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
data = "analysis_set"
""".strip(),
    )
    data_registry = DataRegistry().load(
        {
            "time": {
                "analysis_set": {
                    "times": ["01:00:00", "02:00:00"],
                    "tol": "00:05:00",
                }
            }
        }
    )

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
        data_registry=data_registry,
    )

    assert config.data is not None
    assert config.data.image_times == pytest.approx([1.0, 2.0])


def test_analysis_formats_load_from_registry_identifiers(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
formats = ["my_npy", "4k"]

[format.npy.my_npy]
dtype = "np.float32"

[format.jpg.4k]
resolution = [2160, 4096]
""".strip(),
    )
    format_registry = FormatRegistry().load(config_path)

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
        format_registry=format_registry,
    )

    assert config.formats == ["my_npy", "4k"]


def test_analysis_mass_export_defaults_to_none(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[color.channel.my_colorpath]
mode = "absolute"
basis = "global"
color_space = "RGB"
channel = "r"

[analysis]
[analysis.mass]
color = "my_colorpath"
""".strip(),
    )
    color_registry = _load_color_registry(config_path, tmp_path)

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
        color_embedding_registry=color_registry,
    )

    assert config.mass is not None
    assert config.mass.export is None


def test_analysis_mass_export_accepts_supported_modes(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[color.channel.my_colorpath]
mode = "absolute"
basis = "global"
color_space = "RGB"
channel = "r"

[analysis]
[analysis.mass]
color = "my_colorpath"
export = ["mass", "extensive_mass", "rescaled_concentration_aq"]
""".strip(),
    )
    color_registry = _load_color_registry(config_path, tmp_path)

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
        color_embedding_registry=color_registry,
    )

    assert config.mass is not None
    assert config.mass.export == [
        "mass",
        "extensive_mass",
        "rescaled_concentration_aq",
    ]


def test_analysis_mass_export_rejects_unsupported_modes(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[color.channel.my_colorpath]
mode = "absolute"
basis = "global"
color_space = "RGB"
channel = "r"

[analysis]
[analysis.mass]
color = "my_colorpath"
export = ["mass", "not_supported"]
""".strip(),
    )
    color_registry = _load_color_registry(config_path, tmp_path)

    with pytest.raises(
        ValueError, match=r"Unsupported \[analysis\.mass\]\.export entries"
    ):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
            color_embedding_registry=color_registry,
        )


def test_analysis_inline_data_selector_is_deprecated(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.data.time.analysis_set]
times = ["01:00:00", "02:00:00"]
tol = "00:05:00"
""".strip(),
    )

    with pytest.warns(DeprecationWarning, match=r"\[analysis\.data\]"):
        config = AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )

    assert config.data is not None
    assert config.data.image_times == pytest.approx([1.0, 2.0])


def test_analysis_expert_knowledge_defaults_to_empty_lists(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
""".strip(),
    )

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
    )

    assert config.expert_knowledge.saturation_g == []
    assert config.expert_knowledge.concentration_aq == []


def test_analysis_expert_knowledge_resolves_roi_registry_keys(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.expert_knowledge]
saturation_g = ["storage"]
concentration_aq = ["storage"]
""".strip(),
    )
    roi = RoiConfig().load(
        {"name": "storage", "corner_1": [0.0, 0.0], "corner_2": [1.0, 1.0]}
    )
    roi_registry = RoiRegistry()
    roi_registry.register("storage", roi)

    config = AnalysisConfig().load(
        path=config_path,
        data=tmp_path,
        results=tmp_path,
        roi_registry=roi_registry,
    )

    assert config.expert_knowledge.saturation_g == ["storage"]
    assert config.expert_knowledge.concentration_aq == ["storage"]


def test_analysis_expert_knowledge_rejects_non_list_values(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[analysis]
[analysis.expert_knowledge]
saturation_g = "storage"
""".strip(),
    )

    with pytest.raises(ValueError, match="analysis\\.expert_knowledge\\.saturation_g"):
        AnalysisConfig().load(
            path=config_path,
            data=tmp_path,
            results=tmp_path,
        )
