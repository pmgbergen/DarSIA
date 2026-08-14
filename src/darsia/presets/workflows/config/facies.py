"""Facies configuration for the setup."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_key, _get_section_from_toml

logger = logging.getLogger(__name__)


@dataclass
class FaciesConfig:
    """Facies configuration for the setup."""

    # id: list[int] = field(default_factory=list)
    # """List of facies IDs."""
    props: Path = field(default_factory=Path)
    """Path to the facies properties file."""
    path: Path = field(default_factory=Path)
    """Path to the facies file."""
    facies_to_labels_map: dict[int, list[int]] = field(default_factory=dict)
    """Mapping from facies ID to list of segment labels."""
    label_to_facies_map: dict[int, list[int]] = field(default_factory=dict)
    """Mapping from segment label to list of facies IDs."""

    def load(self, path: Path, results: Path | None = None) -> "FaciesConfig":
        """Load facies config from a toml file from [section]."""
        sec = _get_section_from_toml(path, "facies")

        # Input output paths.
        self.props = _get_key(sec, "props", required=True, type_=Path)
        self.path = _get_key(sec, "path", required=False, type_=Path)
        if not self.path:
            assert results is not None
            self.path = results / "setup" / "facies" / "facies.npz"

        # Allow for manually grouping facies, e.g., for heterogeneous analysis
        # using the same label for multiple facies. If not provided, each facies
        # is its own group with canonical labels.
        ids = sec.keys() - {"props", "path"}
        for i in ids:
            self.facies_to_labels_map[int(i)] = [int(s) for s in sec[str(i)]["labels"]]

        # Check that all labels are unique across facies, i.e., that no label is shared
        # by multiple facies. This is not strictly necessary, but it is a common use
        # case to allow for multiple facies to share the same label, e.g., for
        # heterogeneous analysis. If this is the case, we will log a warning.
        all_labels = [
            label for labels in self.facies_to_labels_map.values() for label in labels
        ]
        if len(all_labels) != len(set(all_labels)):
            raise ValueError(
                "Some labels are shared by multiple facies. This is not allowed."
            )

        # Reverse mapping for convenience. Note that this allows for multiple facies to
        # share the same label, which is useful for heterogeneous analysis.
        self.label_to_facies_map = {
            label: facies_id
            for facies_id, labels in self.facies_to_labels_map.items()
            for label in labels
        }

        return self
