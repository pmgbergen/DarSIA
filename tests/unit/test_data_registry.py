"""Unit tests for DataRegistry and the updated TimeData section keys."""

import pytest

from darsia.presets.workflows.config.data_registry import DataRegistry
from darsia.presets.workflows.config.time_data import TimeData


def test_common_and_merged_run_keep_one_consistent_named_definition(tmp_path):
    import toml
    from darsia.presets.workflows.config.roi_registry import RoiRegistry
    common = _make_registry_sec()
    common["roi"] = [dict(name="whole", corner_1=[0.0, 0.0], corner_2=[1.0, 1.0])]
    merged = dict(data_interval=list(common["data_interval"]),
                  data_time=[dict(name="manual_snap")], roi=list(common["roi"]))
    paths = [tmp_path / "common.toml", tmp_path / "merged-run.toml"]
    for path, value in zip(paths, (common, merged)):
        path.write_text(toml.dumps(value), encoding="utf-8")
    before = [path.read_bytes() for path in paths]
    registry = DataRegistry().load(paths)
    assert set(registry.keys()) == set(DataRegistry().load(common).keys())
    assert registry.resolve("manual_snap").image_times == [0.5, 1.0]
    assert len(registry.resolve("calibration1").image_times) == 5
    rois = RoiRegistry().load(paths)
    assert rois.keys() == ["whole"]
    assert [path.read_bytes() for path in paths] == before


@pytest.mark.parametrize("category,first,second", [
    ("data_interval", dict(name="window", start="00:00:00", end="01:00:00", num=2),
        dict(name="window", start="00:00:00", end="01:00:00", num=3)),
    ("data_time", dict(name="window", times=["00:00:00"]), dict(name="window", times=["01:00:00"])),
    ("roi", dict(name="whole", corner_1=[0.0, 0.0], corner_2=[1.0, 1.0]),
        dict(name="whole", corner_1=[0.0, 0.0], corner_2=[2.0, 1.0])),
])
def test_conflicting_named_values_across_sources_remain_errors(tmp_path, category, first, second):
    import toml
    from darsia.presets.workflows.config.roi_registry import RoiRegistry
    paths = [tmp_path / "first.toml", tmp_path / "second.toml"]
    for path, row in zip(paths, (first, second)):
        path.write_text(toml.dumps({category: [row]}), encoding="utf-8")
    registry = RoiRegistry() if category == "roi" else DataRegistry()
    with pytest.raises(ValueError, match="duplicate.*conflicting"):
        registry.load(paths)


@pytest.mark.parametrize("category", ["data_interval", "roi"])
def test_duplicate_names_in_one_source_still_fail(tmp_path, category):
    import toml
    from darsia.presets.workflows.config.roi_registry import RoiRegistry
    row = (dict(name="window", start="00:00:00", end="01:00:00", num=2) if category == "data_interval"
           else dict(name="whole", corner_1=[0.0, 0.0], corner_2=[1.0, 1.0]))
    path = tmp_path / "duplicate.toml"
    path.write_text(toml.dumps({category: [row, row]}), encoding="utf-8")
    registry = RoiRegistry() if category == "roi" else DataRegistry()
    with pytest.raises(ValueError, match="duplicate.*one configuration source"):
        registry.load([path])

# ---------------------------------------------------------------------------
# DataRegistry tests
# ---------------------------------------------------------------------------


def _make_registry_sec():
    """Return a sample [data] section dict with all three sub-registries."""
    return {
        "data_interval": [
            {
                "name": "calibration1",
                "start": "01:00:00",
                "end": "23:00:00",
                "num": 5,
                "tol": "00:10:00",
            },
            {
                "name": "calibration2",
                "start": "24:00:00",
                "end": "140:00:00",
                "num": 5,
                "tol": "00:10:00",
            },
            {
                "name": "phase_1",
                "start": "00:00:00",
                "end": "01:00:00",
                "num": 13,
                "tol": "00:01:00",
            },
        ],
        "data_time": [
            {
                "name": "manual_snap",
                "times": ["00:30:00", "01:00:00"],
                "tol": "00:05:00",
            },
        ],
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
        sec = {"data_path": [{"name": "imgs", "paths": ["img.jpg"]}]}
        reg = DataRegistry().load(sec, data_folder=tmp_path)
        td = reg.resolve("imgs")
        assert len(td.image_paths) == 1
        assert td.image_paths[0] == dummy


class TestDataRegistryDuplicateCheck:
    def test_duplicate_interval_and_time(self):
        sec = {
            "data_interval": [{"name": "dup", "start": "01:00:00", "end": "02:00:00", "num": 2}],
            "data_time": [{"name": "dup", "times": ["01:30:00"], "tol": "00:05:00"}],
        }
        with pytest.raises(ValueError, match="duplicate"):
            DataRegistry().load(sec)

    def test_duplicate_interval_and_path(self, tmp_path):
        (tmp_path / "x.jpg").touch()
        sec = {
            "data_interval": [{"name": "dup", "start": "01:00:00", "end": "02:00:00", "num": 2}],
            "data_path": [{"name": "dup", "paths": ["x.jpg"]}],
        }
        with pytest.raises(ValueError, match="duplicate"):
            DataRegistry().load(sec, data_folder=tmp_path)

    def test_duplicate_time_and_path(self, tmp_path):
        (tmp_path / "x.jpg").touch()
        sec = {
            "data_time": [{"name": "dup", "times": ["01:00:00"], "tol": "00:05:00"}],
            "data_path": [{"name": "dup", "paths": ["x.jpg"]}],
        }
        with pytest.raises(ValueError, match="duplicate"):
            DataRegistry().load(sec, data_folder=tmp_path)

    def test_no_duplicate_distinct_keys(self):
        sec = {
            "data_interval": [{"name": "cal", "start": "01:00:00", "end": "02:00:00", "num": 2}],
            "data_time": [{"name": "snap", "times": ["01:30:00"], "tol": "00:05:00"}],
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
