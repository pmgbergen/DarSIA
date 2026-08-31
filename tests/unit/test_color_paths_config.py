"""Unit tests for color path embedding configuration via ColorEmbeddingRegistry."""

import textwrap
from pathlib import Path

import pytest

from darsia.presets.workflows.config.color_embedding_registry import (
    ColorEmbeddingRegistry,
)
from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.roi import RoiAndLabelConfig, RoiConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_toml(tmp_path: Path, content: str) -> Path:
    """Write *content* to a ``config.toml`` file and return its path."""
    p = tmp_path / "config.toml"
    p.write_text(textwrap.dedent(content))
    return p


def _make_registry_with_roi(name: str = "my_roi") -> RoiRegistry:
    """Return a :class:`RoiRegistry` with a single plain :class:`RoiConfig`."""
    reg = RoiRegistry()
    roi = RoiConfig()
    roi.load({"name": name, "corner_1": [0.1, 0.2], "corner_2": [0.8, 0.9]})
    reg.register(name, roi)
    return reg


def _make_data_registry(tmp_path: Path) -> DataRegistry:
    """Return a DataRegistry with dummy baseline and calibration path entries.

    Creates one dummy image file so PathData validation passes.
    """
    dummy = tmp_path / "dummy.jpg"
    dummy.touch()
    sec = {
        "data_path": [
            {"name": "baseline_imgs", "paths": ["dummy.jpg"]},
            {"name": "cal_imgs", "paths": ["dummy.jpg"]},
        ]
    }
    return DataRegistry().load(sec, data_folder=tmp_path)


def _minimal_color_path_embedding_toml(extra: str = "") -> str:
    """Return a minimal [[color_path]] array-of-tables TOML using registry references.

    Args:
        extra: Additional TOML lines to insert inside [[color_path]].
    """
    return textwrap.dedent(
        f"""\
[[color_path]]
name = "default"
baseline = "baseline_imgs"
data     = "cal_imgs"
{extra}
"""
    )


# ---------------------------------------------------------------------------
# Inline ROI sub-sections
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Selector resolution and validation
# ---------------------------------------------------------------------------
