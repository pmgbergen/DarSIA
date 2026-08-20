"""Registry for ROI configurations loaded from a top-level [[roi]] TOML array-of-tables."""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .roi import RoiAndSubroiConfig, RoiConfig
from .utils import _convert_none

logger = logging.getLogger(__name__)


def _load_roi_key_list(
    sub_sec: dict,
    key: str,
    *,
    context: str,
    roi_registry: "RoiRegistry | None",
    restricted: bool,
    allow_str: bool = False,
    none_if_absent: bool = False,
) -> list[str] | None:
    """Validate and normalize a ROI-registry-key-reference field.

    Reads `sub_sec[key]`, validates it's a list[str] (or a bare str if `allow_str`),
    and validates each entry resolves in `roi_registry` — using `resolve_roi_and_labels`
    if `restricted` else `resolve_rois`. Returns the validated list[str] unresolved
    (resolution to RoiConfig objects happens at point of use, not here).

    Args:
        sub_sec: Section dict to read from.
        key: Key name within the section.
        context: Human-readable context for error messages (e.g. "analysis.mass.roi").
        roi_registry: ROI registry to validate keys against (required if keys are present).
        restricted: If True, use resolve_roi_and_labels (label-restricted). If False,
            use resolve_rois (plain ROIs only).
        allow_str: If True, accept a bare string and wrap as single-entry list.
        none_if_absent: If True, return None when the key is absent. If False, return [].

    Returns:
        Validated list[str] (or None if `none_if_absent=True` and key is absent).

    Raises:
        ValueError: If the value is not a valid list[str]/str, if the registry is
            missing when keys are present, or if any key doesn't resolve or has
            the wrong type (label-restricted vs. plain).
    """
    raw_value = _convert_none(sub_sec.get(key))

    if raw_value is None:
        return None if none_if_absent else []

    # Normalize to list[str]
    if isinstance(raw_value, str):
        if not allow_str:
            raise ValueError(f"{context} must be a list of ROI registry keys.")
        roi_keys = [raw_value]
    elif isinstance(raw_value, list):
        roi_keys = [str(k).strip() for k in raw_value]
    else:
        msg = (
            f"{context} must be None, a string, or a list of strings."
            if allow_str
            else f"{context} must be a list of ROI registry keys."
        )
        raise ValueError(msg)

    if not roi_keys:
        return None if none_if_absent else []

    # Validate registry is available
    if roi_registry is None:
        raise ValueError(
            f"{context} references ROI keys, but no ROI registry is available. "
            "Define top-level [[roi]] entries."
        )

    # Validate each key resolves and has the correct type
    if restricted:
        resolved = roi_registry.resolve_roi_and_labels(roi_keys)
        unrestricted_keys = [k for k in roi_keys if k not in resolved]
        if unrestricted_keys:
            raise ValueError(
                f"{context} contains unknown or unrestricted ROI keys: "
                f"{unrestricted_keys}"
            )
    else:
        resolved = roi_registry.resolve_rois(roi_keys)
        missing_keys = [k for k in roi_keys if k not in resolved]
        if missing_keys:
            raise ValueError(f"{context} contains unknown ROI keys: {missing_keys}")

    return roi_keys


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
