"""Unit tests for RoiRegistry register behavior."""

import pytest

from darsia.presets.workflows.config.roi import RoiConfig
from darsia.presets.workflows.config.roi_registry import RoiRegistry


class TestRoiRegistryRegister:
    def test_register_new_key(self):
        reg = RoiRegistry()
        roi = RoiConfig()
        roi.load({"name": "a", "corner_1": [0.0, 0.0], "corner_2": [1.0, 1.0]})
        reg.register("a", roi)
        assert "a" in reg.keys()
        assert reg.resolve("a")["a"] is roi

    def test_register_duplicate_key_raises(self):
        reg = RoiRegistry()
        roi = RoiConfig()
        roi.load({"name": "a", "corner_1": [0.0, 0.0], "corner_2": [1.0, 1.0]})
        reg.register("a", roi)
        with pytest.raises(KeyError, match="already registered"):
            reg.register("a", roi)

    def test_register_does_not_affect_loaded_entries(self):
        """Entries added via register() coexist with entries from load()."""
        reg = RoiRegistry()
        # Manually seed the registry as if load() had been called
        roi_loaded = RoiConfig()
        roi_loaded.load(
            {"name": "loaded", "corner_1": [0.0, 0.0], "corner_2": [0.5, 0.5]}
        )
        reg._registry["loaded"] = roi_loaded

        roi_new = RoiConfig()
        roi_new.load({"name": "new", "corner_1": [0.5, 0.5], "corner_2": [1.0, 1.0]})
        reg.register("new", roi_new)
        assert set(reg.keys()) == {"loaded", "new"}
