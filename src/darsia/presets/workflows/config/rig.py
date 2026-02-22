import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

from .utils import _get_key, _get_section_from_toml


@dataclass
class RigConfig:
    """Specifications for the rig."""

    width: float = 0  # in meters
    """Width of the rig in meters."""
    height: float = 0  # in meters
    """Height of the rig in meters."""
    dim: int = 2  # spatial dimension (2 or 3)
    """Spatial dimension (2 or 3)."""
    resolution: tuple[int, int] = (500, 1000)
    """Default resolution for images (height, width)."""
    path: Path = field(default_factory=Path)
    """Path to the rig cache file."""

    def load(self, path: Path, results: Path | None = None) -> "FluidFlowerRigConfig":
        sec = _get_section_from_toml(path, "rig")
        self.width = _get_key(sec, "width", required=True, type_=float)
        self.height = _get_key(sec, "height", required=True, type_=float)
        self.dim = _get_key(sec, "dim", required=True, type_=int)
        self.resolution = _get_key(
            sec, "resolution", default=(500, 1000), required=False, type_=tuple
        )
        self.path = _get_key(sec, "path", required=False, type_=Path)
        if not self.path:
            assert results is not None
            self.path = results / "setup" / "rig"

        return self

    def error(self):
        raise ValueError("Use [specs] in the config file to load specs.")
