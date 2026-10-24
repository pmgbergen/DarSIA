"""Unit tests for the top-level ExpertKnowledgeConfig (relocated from analysis.py)."""

from pathlib import Path

import pytest

from darsia.presets.workflows.config.expert_knowledge import ExpertKnowledgeConfig
from darsia.presets.workflows.config.roi import RoiConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry


def _write(path: Path, content: str) -> Path:
    path.write_text(content)
    return path


def test_expert_knowledge_dataclass_defaults_to_empty_lists() -> None:
    config = ExpertKnowledgeConfig()

    assert config.saturation_g == []
    assert config.concentration_aq == []
    assert config.no_saturation_g == []
    assert config.no_concentration_aq == []


def test_expert_knowledge_load_raises_when_section_missing(tmp_path: Path) -> None:
    # Matches sibling top-level configs (CorrectionsConfig, RoiRegistry): load()
    # raises when its own section is absent; FluidFlowerConfig's _load_section
    # wrapper is what turns this into a silent None/default at the top level.
    config_path = _write(tmp_path / "config.toml", "")

    with pytest.raises(KeyError):
        ExpertKnowledgeConfig().load(path=config_path)


def test_expert_knowledge_empty_section_defaults_to_empty_lists(tmp_path: Path) -> None:
    config_path = _write(tmp_path / "config.toml", "[expert_knowledge]")

    config = ExpertKnowledgeConfig().load(path=config_path)

    assert config.saturation_g == []
    assert config.concentration_aq == []
    assert config.no_saturation_g == []
    assert config.no_concentration_aq == []


def test_expert_knowledge_resolves_roi_registry_keys(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[expert_knowledge]
saturation_g = ["storage"]
concentration_aq = ["storage"]
no_saturation_g = ["calibration"]
no_concentration_aq = ["calibration"]
""".strip(),
    )
    roi_registry = RoiRegistry()
    roi_registry.register(
        "storage",
        RoiConfig().load(
            {"name": "storage", "corner_1": [0.0, 0.0], "corner_2": [1.0, 1.0]}
        ),
    )
    roi_registry.register(
        "calibration",
        RoiConfig().load(
            {"name": "calibration", "corner_1": [0.0, 0.0], "corner_2": [1.0, 1.0]}
        ),
    )

    config = ExpertKnowledgeConfig().load(path=config_path, roi_registry=roi_registry)

    assert config.saturation_g == ["storage"]
    assert config.concentration_aq == ["storage"]
    assert config.no_saturation_g == ["calibration"]
    assert config.no_concentration_aq == ["calibration"]


def test_expert_knowledge_rejects_non_list_values(tmp_path: Path) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[expert_knowledge]
saturation_g = "storage"
""".strip(),
    )

    with pytest.raises(ValueError, match="expert_knowledge\\.saturation_g"):
        ExpertKnowledgeConfig().load(path=config_path)


def test_expert_knowledge_rejects_non_list_values_for_negative_fields(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[expert_knowledge]
no_saturation_g = "storage"
""".strip(),
    )

    with pytest.raises(ValueError, match="expert_knowledge\\.no_saturation_g"):
        ExpertKnowledgeConfig().load(path=config_path)


def test_expert_knowledge_requires_roi_registry_when_keys_present(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[expert_knowledge]
saturation_g = ["storage"]
""".strip(),
    )

    with pytest.raises(ValueError, match="requires a loaded ROI registry"):
        ExpertKnowledgeConfig().load(path=config_path)


def test_expert_knowledge_requires_roi_registry_when_negative_keys_present(
    tmp_path: Path,
) -> None:
    config_path = _write(
        tmp_path / "config.toml",
        """
[expert_knowledge]
no_concentration_aq = ["storage"]
""".strip(),
    )

    with pytest.raises(ValueError, match="requires a loaded ROI registry"):
        ExpertKnowledgeConfig().load(path=config_path)
