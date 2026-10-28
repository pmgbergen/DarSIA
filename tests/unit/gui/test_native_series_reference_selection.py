"""Local metadata fixtures; no Qt/application, image or numerical imports."""

import json
from pathlib import Path

import pytest

from darsia.fluidflower import series_reference_selection as s


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def series(tmp_path):
    put(tmp_path / "series.toml", 'schema_version=1\n[runs]\nids=["ac28","ac52"]\n')
    return [
        (
            run,
            put(tmp_path / f"config/merged/{run}.toml", f'run="{run}"\n'),
            "Merged configuration",
        )
        for run in ("ac28", "ac52")
    ]


def saved_reference(root, run):
    put(
        root / "geometry_facies/series.json",
        json.dumps({"tools": {"native_series": {"reference_run": run}}}),
    )


def test_manifest_order_is_used_without_saved_reference(tmp_path):
    choices = series(tmp_path)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    value = s.reference_choice(tmp_path, choices)
    assert value.run_id == "ac28" and value.config_path == str(choices[0][1])
    assert "manifest order" in value.source and not value.issues
    assert all(Path(path).read_bytes() == raw for path, raw in before.items())


def test_saved_native_reference_wins(tmp_path):
    choices = series(tmp_path)
    saved_reference(tmp_path, "ac52")
    value = s.reference_choice(tmp_path, choices)
    assert value.run_id == "ac52" and value.config_path == str(choices[1][1])
    assert value.source == "saved native geometry reference"


def test_invalid_saved_reference_never_skips_to_first(tmp_path):
    choices = series(tmp_path)
    saved_reference(tmp_path, "ac99")
    value = s.reference_choice(tmp_path, choices)
    assert value.run_id == "ac99" and not value.config_path and value.issues


def test_missing_reference_single_file_does_not_skip(tmp_path):
    choices = series(tmp_path)
    choices[0] = ("ac28", None, "Split common/run settings; merged file missing")
    value = s.reference_choice(tmp_path, choices)
    assert value.run_id == "ac28" and not value.config_path
    assert "Split common/run" in value.issues[0]


def test_saved_reference_config_removed_after_inventory_is_missing(tmp_path):
    choices = series(tmp_path)
    saved_reference(tmp_path, "ac52")
    choices[1][1].unlink()
    value = s.reference_choice(tmp_path, choices)
    assert value.run_id == "ac52" and not value.config_path and value.issues


def test_unreadable_saved_geometry_does_not_fall_back(tmp_path):
    choices = series(tmp_path)
    put(tmp_path / "geometry_facies/series.json", "{malformed")
    value = s.reference_choice(tmp_path, choices)
    assert not value.config_path and value.issues


def test_different_series_do_not_share_reference_state(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    choices_a = series(a)
    choices_b = series(b)
    saved_reference(a, "ac52")
    assert s.reference_choice(a, choices_a).run_id == "ac52"
    assert s.reference_choice(b, choices_b).run_id == "ac28"


def test_explicit_geometry_folder_is_respected(tmp_path):
    choices = series(tmp_path)
    put(
        tmp_path / "series.toml",
        'schema_version=1\n[paths]\ngeometry_facies="custom"\n[runs]\nids=["ac28","ac52"]\n',
    )
    put(
        tmp_path / "custom/series.json",
        json.dumps({"tools": {"native_series": {"reference_run": "ac52"}}}),
    )
    assert s.reference_choice(tmp_path, choices).run_id == "ac52"


def test_no_recursive_discovery(tmp_path, monkeypatch):
    choices = series(tmp_path)
    monkeypatch.setattr(
        Path, "rglob", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No rglob"))
    )
    assert s.reference_choice(tmp_path, choices).config_path


@pytest.mark.parametrize("value", [0, False, [], {}, True, 1])
def test_present_invalid_reference_types_never_mean_absent(tmp_path, value):
    choices = series(tmp_path)
    saved_reference(tmp_path, value)
    result = s.reference_choice(tmp_path, choices)
    assert not result.config_path and result.issues
    assert "must be a string" in result.issues[0]
