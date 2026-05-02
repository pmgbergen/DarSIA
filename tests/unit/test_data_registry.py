"""Unit tests for DataRegistry and the updated TimeData section keys."""

import pytest

from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.time_data import TimeData

# ---------------------------------------------------------------------------
# DataRegistry tests
# ---------------------------------------------------------------------------


def _make_registry_sec():
    """Return a sample [data] section dict with all three sub-registries."""
    return {
        "interval": {
            "calibration1": {
                "start": "01:00:00",
                "end": "23:00:00",
                "num": 5,
                "tol": "00:10:00",
            },
            "calibration2": {
                "start": "24:00:00",
                "end": "140:00:00",
                "num": 5,
                "tol": "00:10:00",
            },
            "phase_1": {
                "start": "00:00:00",
                "end": "01:00:00",
                "num": 13,
                "tol": "00:01:00",
            },
        },
        "time": {
            "manual_snap": {
                "times": ["00:30:00", "01:00:00"],
                "tol": "00:05:00",
            },
        },
    }


class TestDataRegistryLoad:
    def test_keys_populated(self):
        reg = DataRegistry().load(_make_registry_sec())
        assert set(reg.keys()) == {
            "calibration1",
            "calibration2",
            "phase_1",
            "manual_snap",
        }

    def test_interval_entry_has_times(self):
        reg = DataRegistry().load(_make_registry_sec())
        td = reg.resolve("phase_1")
        assert len(td.image_times) == 13
        assert td.image_times[0] == pytest.approx(0.0)
        assert td.image_times[-1] == pytest.approx(1.0)

    def test_time_entry_has_times(self):
        reg = DataRegistry().load(_make_registry_sec())
        td = reg.resolve("manual_snap")
        assert len(td.image_times) == 2
        assert td.image_times[0] == pytest.approx(0.5)  # 00:30:00 = 0.5 h
        assert td.image_times[1] == pytest.approx(1.0)  # 01:00:00 = 1.0 h

    def test_empty_registry_sections(self):
        reg = DataRegistry().load({})
        assert reg.keys() == []

    def test_only_path_sub_section(self, tmp_path):
        """Path sub-registry creates entries with image_paths populated."""
        # Create dummy files so validation doesn't warn
        dummy = tmp_path / "img.jpg"
        dummy.touch()
        sec = {"path": {"imgs": {"paths": ["img.jpg"]}}}
        reg = DataRegistry().load(sec, data_folder=tmp_path)
        td = reg.resolve("imgs")
        assert len(td.image_paths) == 1
        assert td.image_paths[0] == dummy


class TestDataRegistryDuplicateCheck:
    def test_duplicate_interval_and_time(self):
        sec = {
            "interval": {"dup": {"start": "01:00:00", "end": "02:00:00", "num": 2}},
            "time": {"dup": {"times": ["01:30:00"], "tol": "00:05:00"}},
        }
        with pytest.raises(ValueError, match="duplicate"):
            DataRegistry().load(sec)

    def test_duplicate_interval_and_path(self, tmp_path):
        (tmp_path / "x.jpg").touch()
        sec = {
            "interval": {"dup": {"start": "01:00:00", "end": "02:00:00", "num": 2}},
            "path": {"dup": {"paths": ["x.jpg"]}},
        }
        with pytest.raises(ValueError, match="duplicate"):
            DataRegistry().load(sec, data_folder=tmp_path)

    def test_duplicate_time_and_path(self, tmp_path):
        (tmp_path / "x.jpg").touch()
        sec = {
            "time": {"dup": {"times": ["01:00:00"], "tol": "00:05:00"}},
            "path": {"dup": {"paths": ["x.jpg"]}},
        }
        with pytest.raises(ValueError, match="duplicate"):
            DataRegistry().load(sec, data_folder=tmp_path)

    def test_no_duplicate_distinct_keys(self):
        sec = {
            "interval": {"cal": {"start": "01:00:00", "end": "02:00:00", "num": 2}},
            "time": {"snap": {"times": ["01:30:00"], "tol": "00:05:00"}},
        }
        reg = DataRegistry().load(sec)  # must not raise
        assert set(reg.keys()) == {"cal", "snap"}


class TestDataRegistryResolve:
    def test_resolve_single_string(self):
        reg = DataRegistry().load(_make_registry_sec())
        td = reg.resolve("calibration1")
        assert isinstance(td, TimeData)
        assert len(td.image_times) == 5

    def test_resolve_list(self):
        reg = DataRegistry().load(_make_registry_sec())
        td = reg.resolve(["calibration1", "calibration2"])
        assert len(td.image_times) == 10

    def test_resolve_deduplicates(self):
        """Resolving the same key twice should not duplicate times."""
        reg = DataRegistry().load(_make_registry_sec())
        td = reg.resolve(["calibration1", "calibration1"])
        # After dedup, same as resolving once
        td_once = reg.resolve("calibration1")
        assert td.image_times == td_once.image_times

    def test_resolve_missing_key_raises(self):
        reg = DataRegistry().load(_make_registry_sec())
        with pytest.raises(KeyError, match="nonexistent"):
            reg.resolve("nonexistent")

    def test_resolve_helpful_error_message(self):
        reg = DataRegistry().load(_make_registry_sec())
        with pytest.raises(KeyError) as exc_info:
            reg.resolve("missing_key")
        assert "Available keys" in str(exc_info.value)


# ---------------------------------------------------------------------------
# TimeData new section keys (interval / time / path)
# ---------------------------------------------------------------------------


class TestTimeDataNewSectionKeys:
    def test_interval_key(self):
        sec = {"interval": {"cal": {"start": "01:00:00", "end": "05:00:00", "num": 5}}}
        td = TimeData().load(sec)
        assert len(td.image_times) == 5
        assert td.mode == "intervals"

    def test_time_key(self):
        sec = {"time": {"snap": {"times": ["01:00:00", "02:00:00"], "tol": "00:05:00"}}}
        td = TimeData().load(sec)
        assert td.image_times == pytest.approx([1.0, 2.0])
        assert td.mode == "times"

    def test_mixed_interval_and_time(self):
        sec = {
            "interval": {"cal": {"start": "01:00:00", "end": "03:00:00", "num": 3}},
            "time": {"snap": {"times": ["00:30:00"], "tol": "00:05:00"}},
        }
        td = TimeData().load(sec)
        assert td.mode == "mixed"
        # 3 from interval + 1 from time, all deduped and sorted
        assert 0.5 in td.image_times
        assert 1.0 in td.image_times
