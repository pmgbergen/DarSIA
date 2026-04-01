"""Labeling configuration."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class LabelingConfig:
    colored_image: Path = field(default_factory=Path)
    """Path to the segmented file."""
    unite_labels: list[list[int]] = field(default_factory=list)
    """List of lists of labels to be united."""
    water_label: int | None = None
    """Label corresponding to water (if any)."""
    colorchecker_label: int | None = None
    """Label corresponding to colorchecker (if any)."""
    labels: Path = field(default_factory=Path)
    """Path to the labels file."""
    # Numerical options for labeling.
    rtol: float = 0.001
    """Minimum significance of labels to be kept (default: 0.1% of the image area)."""
    ensure_connectivity: bool = True
    """Whether to ensure that labels are connected regions (default: True)."""

    def load(self, path: Path, results: Path | None = None) -> "LabelingConfig":
        """Load labeling config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "labeling")
        self.colored_image = _get_key(sec, "colored_image", required=True, type_=Path)
        self.unite_labels = _get_key(sec, "unite_labels", required=False, type_=list)
        self.water_label = _get_key(sec, "water_label", required=False, type_=int)
        self.colorchecker_label = _get_key(
            sec, "colorchecker_label", required=False, type_=int
        )
        self.labels = _get_key(sec, "labels", required=False, type_=Path)
        if not self.labels:
            assert results is not None
            self.labels = results / "setup" / "labels" / "labels.npz"
        self.rtol = _get_key(
            sec, "rtol", default=self.rtol, required=False, type_=float
        )
        self.ensure_connectivity = _get_key(
            sec,
            "ensure_connectivity",
            default=self.ensure_connectivity,
            required=False,
            type_=bool,
        )
        return self

    def error(self):
        raise ValueError("Use [labeling] in the config file to load labeling.")
