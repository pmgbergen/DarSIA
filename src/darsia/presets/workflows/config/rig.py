from dataclasses import dataclass, field
import json
from pathlib import Path

from .utils import _config_path, _get_key, _get_section_from_toml


def _source_record(value):
    """Decode finite object JSON without dropping nulls or accepting duplicate keys."""
    def pairs(items):
        """Preserve each original key exactly once, including in nested metadata."""
        result = {}
        for key, item in items:
            if key in result:
                raise ValueError("Duplicate native source record key")
            result[key] = item
        return result

    def invalid(value):
        """Reject nonfinite JSON spellings rather than changing their meaning."""
        raise ValueError("Nonfinite native source record value: " + value)

    if not isinstance(value, str):
        raise ValueError("A native source record needs its JSON string in TOML")
    result = json.loads(value, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError("A native source record must decode to an object")
    json.dumps(result, allow_nan=False)
    return result


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
    analysis_exclusions: dict | None = field(default=None, metadata={"hidden": True})
    """Optional exact approved native analysis-exclusion source record."""
    native_product: dict | None = field(default=None, metadata={"hidden": True})
    """Optional exact adopted native-label source record, independent of gaps."""

    def load(self, path: Path, results: Path | None = None) -> "RigConfig":
        sec = _get_section_from_toml(path, "rig")
        self.width = _get_key(sec, "width", required=True, type_=float)
        self.height = _get_key(sec, "height", required=True, type_=float)
        self.dim = _get_key(sec, "dim", required=True, type_=int)
        default_path = results / "setup" / "rig" if results else None
        rig_path = _get_key(sec, "path", required=False)
        self.path = default_path if rig_path is None else _config_path(rig_path, path)
        assert self.path is not None, "results is required if path is not set"
        for name in ("analysis_exclusions", "native_product"):
            value = sec.get(name)
            try:
                record = None if value is None else _source_record(value)
            except (TypeError, ValueError) as error:
                raise ValueError("Rig " + name + " requires its original source record JSON") from error
            setattr(self, name, record)
        return self

    def to_dict(self) -> dict:
        """Serialize rig config to TOML-compatible dict for load/save round-trips."""
        result = {
            "width": self.width,
            "height": self.height,
            "dim": self.dim,
        }
        for name in ("analysis_exclusions", "native_product"):
            if getattr(self, name) is not None:
                if not isinstance(getattr(self, name), dict):
                    raise ValueError("Rig " + name + " requires its original source record")
                result[name] = json.dumps(getattr(self, name), ensure_ascii=True,
                    sort_keys=True, separators=(",", ":"), allow_nan=False)
        return result

    def error(self):
        raise ValueError("Use [specs] in the config file to load specs.")
