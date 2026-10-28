"""Original table assembly from synthetic shared scalar records; no solver calls."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from darsia.fluidflower.preset_wasserstein_export import export_preset_wasserstein_csv
from darsia.presets.workflows.comparison.comparison_wasserstein import (
    WassersteinDistanceResult, _assemble_all_wasserstein_results,
)
from .test_preset_wasserstein_results import _published, _read, _put


def test_shared_export_is_byte_identical_to_original_csv_and_never_reads_producer(tmp_path, monkeypatch):
    case = _published(tmp_path, nonfinite=True, single_time=True)
    baseline = tmp_path / "original-local-results"
    baseline.mkdir()
    (baseline / case.filename).write_bytes((case.output / case.filename).read_bytes())
    original_config = SimpleNamespace(wasserstein=SimpleNamespace(
        times=[(2.5, 0.1)], runs=["run-a", "run-b"], results=baseline,
        roi={"whole rig": SimpleNamespace(name="whole rig")}))
    _assemble_all_wasserstein_results(original_config)
    expected = (baseline / "tables" / "wasserstein_distances_all.csv").read_bytes()
    original_stat, original_open = Path.stat, Path.open
    def guarded_stat(path, *args, **kwargs):
        if path.is_relative_to(case.output):
            pytest.fail("CSV export inspected producer-local output")
        return original_stat(path, *args, **kwargs)
    def guarded_open(path, *args, **kwargs):
        if path.is_relative_to(case.output):
            pytest.fail("CSV export opened producer-local output")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(Path, "open", guarded_open)
    result = export_preset_wasserstein_csv(case.root, tmp_path / "export", reader=_read)
    assert Path(result["csv"]["path"]).read_bytes() == expected
    assert b"NaN" in expected and b"time,roi_name,run_1,run_2,normalization,distance" in expected
    assert result["comparison_count"] == 1 and result["nonfinite_distances"] == 1
    receipt = _read(result["receipt"]["path"])
    assert receipt["job"]["path"] == str(case.root / "job.json")
    assert receipt["publications"][0]["comparisons"] == case.result["comparisons"]
    assert receipt["validation"]["state"] == "not_scored"


@pytest.mark.parametrize("problem", ["incomplete", "scalar", "owner", "existing"])
def test_export_refuses_incomplete_or_tampered_shared_sources_before_creating_destination(tmp_path, problem):
    case = _published(tmp_path, single_time=problem != "incomplete")
    destination = tmp_path / "export"
    if problem == "scalar":
        _put(case.lease / "artifacts" / "metadata" / case.filename, {"distance": 1})
    elif problem == "owner":
        _put(case.lease / "owner.json", dict(case.owner, actual_owner={"machine": "another-cpu"}))
    elif problem == "existing":
        destination.mkdir()
        (destination / "kept.txt").write_text("keep", encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        export_preset_wasserstein_csv(case.root, destination, reader=_read)
    if problem == "existing":
        assert list(destination.iterdir()) == [destination / "kept.txt"]
    else:
        assert not destination.exists()


def test_original_assembler_seam_preserves_time_pair_roi_and_normalization_order(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    runs, rois, times = ["c", "a", "b"], ["right", "left"], [(2.0, 0.1), (1.0, 0.2)]
    records = []
    for time, _ in times:
        for left in runs:
            for right in runs:
                if left >= right:
                    continue
                for roi in rois:
                    record = WassersteinDistanceResult(run_1=left, run_2=right, time=time,
                        time_1=time, time_2=time, roi_name=roi, roi_exact_mass=1,
                        roi_detected_mass_1=1, roi_detected_mass_2=1, distance=0.12345678901234567,
                        normalized_distance={"mass_based": 0.3, "rig_based": float("nan"), "plume_based": 0.1},
                        computation_time=0.2, timestamp="2026-10-23T12:00:00", status="success")
                    record.save_to_dir(original)
                    records.append(record)
    config = SimpleNamespace(wasserstein=SimpleNamespace(times=times, runs=runs, results=original,
        roi={key: SimpleNamespace(name=key) for key in rois}))
    _assemble_all_wasserstein_results(config)
    _assemble_all_wasserstein_results(None, result_records=iter(records), output_dir=tmp_path / "table-export")
    actual = (tmp_path / "table-export" / "wasserstein_distances_all.csv").read_bytes()
    assert actual == (original / "tables" / "wasserstein_distances_all.csv").read_bytes()
    lines = actual.decode().splitlines()
    assert lines[1].startswith("2.0,right,a,c,None,")
    assert lines[2].startswith("2.0,right,a,c,mass_based,")
    assert lines[3].startswith("2.0,right,a,c,rig_based,NaN")
