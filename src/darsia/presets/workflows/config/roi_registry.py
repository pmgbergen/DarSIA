"""Registry for ROI configurations loaded from a top-level [[roi]] TOML array-of-tables."""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .roi import RoiAndSubroiConfig, RoiConfig

logger = logging.getLogger(__name__)


@dataclass
class RoiRegistry:
    """A registry of named ROI entries loaded from a top-level [[roi]] TOML array-of-tables.

    Entries are auto-typed on load:
    - If the entry has a ``subroi`` sub-section → :class:`RoiAndSubroiConfig`.
    - Otherwise → :class:`RoiConfig` (with optional ``label`` field).
    """

    def __init__(self) -> None:
        self._registry: dict[str, RoiConfig | RoiAndSubroiConfig] = {}

    def load(self, path: Path | list[Path]) -> "RoiRegistry":
        """Load all ROI entries from the top-level ``[[roi]]`` array-of-tables in TOML.

        Hand-parses TOML (like FormatRegistry) since array-of-tables is not supported
        by the generic _get_section_from_toml helper.

        Args:
            path: Path or list of Paths to TOML config file(s).

        Returns:
            self

        Raises:
            ValueError: If the [roi] section is not an array-of-tables (strict format
                enforcement).
            ValueError: If any ROI entry has a duplicate name (checked during load).
        """
        paths = [path] if isinstance(path, Path) else path
        self._registry = {}
        seen_names: set[str] = set()

        for p in paths:
            if not p.exists():
                continue
            with open(p, "rb") as f:
                data = tomllib.load(f)

            if "roi" not in data:
                continue

            roi_data = data["roi"]
            # Strict format: [[roi]] is a list, not nested dicts like [roi.*]
            if not isinstance(roi_data, list):
                raise ValueError(
                    "The [roi] section must be an array-of-tables format (use [[roi]]), not"
                    "nested tables."
                )

            for entry in roi_data:
                name = entry.get("name")
                if name is None:
                    raise ValueError(
                        "Each [[roi]] entry must have a required 'name' field."
                    )
                name = str(name).strip()

                if name in seen_names:
                    raise ValueError(
                        f"ROI name '{name}' is duplicated. ROI names must be globally unique."
                    )
                seen_names.add(name)

                if "subroi" in entry:
                    self._registry[name] = RoiAndSubroiConfig().load(entry)
                else:
                    self._registry[name] = RoiConfig().load(entry)

        return self

    def register(self, key: str, roi: "RoiConfig | RoiAndSubroiConfig") -> None:
        """Add a single ROI entry to the registry without overwriting existing entries.

        This is useful when inline ROI definitions (e.g. from a
        ``[color.path.<id>.roi.*]`` TOML sub-section) need to be injected into the
        shared registry so that they can later be resolved by key name.

        Args:
            key: The name to register the entry under.
            roi: The ROI config object to register.

        Raises:
            KeyError: If *key* is already present in the registry.
        """
        if key in self._registry:
            raise KeyError(
                f"ROI key '{key}' is already registered. "
                f"Use a different key or remove the existing entry first."
            )
        self._registry[key] = roi

    def keys(self) -> list[str]:
        """Return all registered key names."""
        return list(self._registry.keys())

    def resolve(
        self, keys: str | list[str]
    ) -> dict[str, RoiConfig | RoiAndSubroiConfig]:
        """Return a dict of the requested entries keyed by their registry name.

        Args:
            keys: A single key string or a list of key strings.

        Returns:
            Dict mapping each requested key to its loaded ROI config object.

        Raises:
            KeyError: If any requested key is not present in the registry.
        """
        if isinstance(keys, str):
            keys = [keys]
        result = {}
        for key in keys:
            if key not in self._registry:
                raise KeyError(
                    f"ROI key '{key}' not found in registry. "
                    f"Available keys: {list(self._registry.keys())}"
                )
            result[key] = self._registry[key]
        return result

    def resolve_rois(self, keys: str | list[str]) -> dict[str, RoiConfig]:
        """Return ROI entries with no label restriction for the given keys.

        Args:
            keys: A single key string or a list of key strings.

        Returns:
            Dict containing only the entries with ``label is None`` (plain RoiConfig
            instances or RoiAndSubroiConfig with no label, but not label-restricted entries).
        """
        resolved = self.resolve(keys)
        return {
            k: v
            for k, v in resolved.items()
            if isinstance(v, RoiConfig) and v.label is None
        }

    def resolve_roi_and_labels(self, keys: str | list[str]) -> dict[str, RoiConfig]:
        """Return ROI entries with label restriction for the given keys.

        Args:
            keys: A single key string or a list of key strings.

        Returns:
            Dict containing only the entries with ``label is not None`` (label-restricted
            RoiConfig instances, which may be RoiConfig or RoiAndSubroiConfig subclasses).
        """
        resolved = self.resolve(keys)
        return {
            k: v
            for k, v in resolved.items()
            if isinstance(v, RoiConfig) and v.label is not None
        }


@dataclass
class RoiRegistryConfig:
    """GUI-editable view of the [[roi]] TOML array-of-tables (excludes subroi entries).

    This is purely for GUI schema introspection and does not replace RoiRegistry,
    which remains the canonical runtime registry. RoiRegistryConfig.load() delegates
    to RoiRegistry.load() internally to parse the [[roi]] array-of-tables format.
    """

    rois: dict[str, RoiConfig] = field(
        default_factory=dict,
        metadata={
            "name": "ROIs",
            "help": "Named ROI definitions with optional per-label restrictions.",
            "group": "ROIs",
            "widget": "roi_map",
        },
    )
    """Dict of named ROI configurations (excludes RoiAndSubroiConfig entries)."""

    def load(self, path: Path | list[Path]) -> "RoiRegistryConfig":
        """Load ROI entries from [[roi]] array-of-tables via RoiRegistry.

        Filters out RoiAndSubroiConfig entries (nested subroi) and returns only
        plain RoiConfig entries for GUI editing.

        Args:
            path: Path or list of Paths to TOML config file(s).

        Returns:
            self

        Raises:
            ValueError: If the [roi] section is not an array-of-tables.
            ValueError: If any ROI name is duplicated.
        """
        registry = RoiRegistry().load(path)
        self.rois = {
            k: v
            for k, v in registry._registry.items()
            if isinstance(v, RoiConfig) and not isinstance(v, RoiAndSubroiConfig)
        }
        return self
