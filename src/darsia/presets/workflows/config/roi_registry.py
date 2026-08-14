"""Registry for ROI configurations loaded from a top-level [roi.*] TOML section."""

import logging
from pathlib import Path

from .roi import RoiAndLabelConfig, RoiAndSubroiConfig, RoiConfig
from .utils import _get_section_from_toml

logger = logging.getLogger(__name__)


class RoiRegistry:
    """A registry of named ROI entries loaded from a top-level [roi.*] TOML section.

    Entries are auto-typed on load:
    - If the entry has a ``label`` key → :class:`RoiAndLabelConfig`.
    - If the entry has a ``subroi`` sub-section → :class:`RoiAndSubroiConfig`.
    - Otherwise → :class:`RoiConfig`.
    """

    def __init__(self) -> None:
        self._registry: dict[
            str, RoiConfig | RoiAndLabelConfig | RoiAndSubroiConfig
        ] = {}

    def load(self, path: Path | list[Path]) -> "RoiRegistry":
        """Load all ROI entries from the top-level ``[roi]`` section of a TOML file.

        Args:
            path: Path or list of Paths to TOML config file(s).

        Returns:
            self
        """
        roi_sec = _get_section_from_toml(path, "roi")
        self._registry = {}
        for key, entry in roi_sec.items():
            if "label" in entry:
                self._registry[key] = RoiAndLabelConfig().load(entry)
            elif "subroi" in entry:
                self._registry[key] = RoiAndSubroiConfig().load(entry)
            else:
                self._registry[key] = RoiConfig().load(entry)
        return self

    def register(
        self, key: str, roi: "RoiConfig | RoiAndLabelConfig | RoiAndSubroiConfig"
    ) -> None:
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
    ) -> dict[str, RoiConfig | RoiAndLabelConfig | RoiAndSubroiConfig]:
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
        """Return only plain :class:`RoiConfig` entries for the given keys.

        Args:
            keys: A single key string or a list of key strings.

        Returns:
            Dict containing only the entries that are plain :class:`RoiConfig`
            instances (i.e. not subclasses).
        """
        resolved = self.resolve(keys)
        return {k: v for k, v in resolved.items() if type(v) is RoiConfig}

    def resolve_roi_and_labels(
        self, keys: str | list[str]
    ) -> dict[str, RoiAndLabelConfig]:
        """Return only :class:`RoiAndLabelConfig` entries for the given keys.

        Args:
            keys: A single key string or a list of key strings.

        Returns:
            Dict containing only the entries that are :class:`RoiAndLabelConfig`
            instances.
        """
        resolved = self.resolve(keys)
        return {k: v for k, v in resolved.items() if isinstance(v, RoiAndLabelConfig)}
