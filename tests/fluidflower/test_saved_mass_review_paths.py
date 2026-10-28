"""Saved mass review resolves metadata paths without computing products."""
from pathlib import Path
from darsia.fluidflower import saved_mass_rendering


def test_saved_result_path_resolution_preserves_non_paths(tmp_path, monkeypatch):
    absolute = str(tmp_path / "retained" / "fields.npz")
    result = {"precision": "float32", "complete": True, "frames": [
        {"preview": "frame.png", "fields": absolute, "mass": 1.25},
        {"fields": "arrays.npz"}], "mass_curve": "curve.png",
        "overflow_curve": absolute, "series_calibration": "frozen.json",
        "result_source_manifest": "sources.json", "method": {"gain": 23.859}}
    monkeypatch.setattr(saved_mass_rendering, "read", lambda path: result)
    path = tmp_path / "run" / "result.json"
    out = saved_mass_rendering.read_result(path)
    assert out is result
    assert out["frames"][0]["preview"] == str(path.parent / "frame.png")
    assert out["frames"][0]["fields"] == absolute
    assert out["frames"][1]["fields"] == str(path.parent / "arrays.npz")
    assert out["mass_curve"] == str(path.parent / "curve.png")
    assert out["overflow_curve"] == absolute
    assert out["series_calibration"] == str(path.parent / "frozen.json")
    assert out["result_source_manifest"] == str(path.parent / "sources.json")
    assert out["precision"] == "float32" and out["frames"][0]["mass"] == 1.25
    assert out["method"] == {"gain": 23.859}
