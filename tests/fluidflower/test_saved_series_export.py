"""Saved exports keep recorded scalars and refuse to replace existing files."""
import json
import pytest
from darsia.fluidflower.tools import phase_studio_series_export as saved_export


def test_saved_csv_scalars_and_existing_destination(tmp_path):
    source = tmp_path / "result.json"
    source.write_text(json.dumps({"frames": [{"elapsed_h": 1.25,
        "aqueous_mass_g": 2.375, "gas_mass_g": 0.125,
        "total_mass_g": 2.5, "calibration_role": "frozen"}]}), encoding="utf-8")
    output = tmp_path / "mass.csv"
    saved_export.main(["--result", str(source), "--destination", str(output),
                       "--export", "csv"])
    lines = output.read_text(encoding="utf-8").splitlines()
    assert lines[1] == "1.25,,2.375,0.125,2.5,,,,,frozen"
    before = output.read_bytes()
    with pytest.raises(FileExistsError, match="Choose a new export path"):
        saved_export.export(source, output, "csv")
    assert output.read_bytes() == before


def test_retired_request_branch_is_unavailable(tmp_path):
    with pytest.raises(SystemExit) as exc:
        saved_export.main(["--request", str(tmp_path / "request.json")])
    assert exc.value.code == 2
