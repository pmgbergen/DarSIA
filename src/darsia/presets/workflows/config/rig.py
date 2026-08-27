from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_key, _get_section_from_toml


@dataclass
class RigConfig:
    """Specifications for the rig."""

    width: float | None = field(
        default=None,
        metadata={
            "name": "Width",
            "help": "Width of the rig in meters.",
            "placeholder": "Width in meters (e.g., 1.0)",
        },
    )
    """Width of the rig in meters."""
    height: float | None = field(
        default=None,
        metadata={
            "name": "Height",
            "help": "Height of the rig in meters.",
            "placeholder": "Height in meters (e.g., 0.5)",
        },
    )
    """Height of the rig in meters."""
    dim: int = field(
        default=2,
        metadata={
            "name": "Dimension",
            "help": "Spatial dimension of the rig setup.",
            "options": [2, 3],
        },
    )
    """Spatial dimension (2 or 3)."""
    path: Path = field(
        default_factory=Path,
        metadata={"hidden": True},
    )
    """Path to the rig cache file. Computed under `results` if not given."""

    def load(self, path: Path, results: Path | None = None) -> "RigConfig":
        sec = _get_section_from_toml(path, "rig")
        self.width = _get_key(sec, "width", required=True, type_=float)
        self.height = _get_key(sec, "height", required=True, type_=float)
        self.dim = _get_key(sec, "dim", required=True, type_=int)
        default_path = results / "setup" / "rig" if results else None
        self.path = _get_key(
            sec, "path", default=default_path, required=False, type_=Path
        )
        assert self.path is not None, "results is required if path is not set"
        return self

    def to_dict(self) -> dict:
        """Serialize rig config to TOML-compatible dict for load/save round-trips."""
        return {
            "width": self.width,
            "height": self.height,
            "dim": self.dim,
        }

    def error(self):
        raise ValueError("Use [specs] in the config file to load specs.")
