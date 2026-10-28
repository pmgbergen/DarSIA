"""Overflow shares protocol file/sheet handling without inferring gas volume."""
from types import SimpleNamespace

import pandas as pd
import pytest
import toml

from darsia.experiment.experiment import ProtocolledExperiment
from darsia.experiment.protocols import OverflowProtocol
from darsia.fluidflower.overflow_data_page import read_overflow_data
from darsia.presets.workflows.config.protocols import ProtocolsConfig


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch, tmp_path):
    monkeypatch.setenv("DARSIA_SERIES_REGISTRY", str(tmp_path / "no-registry.toml"))


def test_csv_and_selected_excel_sheet_keep_measurements_and_clock_text(tmp_path):
    values = pd.DataFrame({
        "datetime": ["2023-01-02T10:00:00.123456+01:00", "2023-01-02T11:00:00.123456+01:00"],
        "weight_g": [-0.7, 1.25], "status": ["Corrected", "Verified"],
        "source": ["scale-a.jpg", "scale-b.jpg"],
    })
    csv = tmp_path / "overflow.csv"
    values.to_csv(csv, index=False)
    workbook = tmp_path / "overflow.xlsx"
    with pd.ExcelWriter(workbook) as writer:
        pd.DataFrame({"unrelated": [999]}).to_excel(writer, sheet_name="other", index=False)
        values.to_excel(writer, sheet_name="run-b", index=False)
    before = csv.read_bytes(), workbook.read_bytes()
    for reference in (csv, (workbook, "run-b")):
        actual = OverflowProtocol(reference)
        pd.testing.assert_frame_equal(actual.df, values)
        assert "gas_volume_ml" not in actual.df
    assert before == (csv.read_bytes(), workbook.read_bytes())


def test_experiment_optional_overflow_is_passed_from_original_config(tmp_path):
    folder = tmp_path / "images"
    imaging = tmp_path / "imaging.csv"
    imaging.write_text("path,datetime\na.jpg,2023-01-02T10:00:00+01:00\n")
    injection = tmp_path / "injection.csv"
    injection.write_text("id,location_x,location_y,start,end,rate_kg/s\n1,0,0,2023-01-02T10:00:00+01:00,2023-01-02T11:00:00+01:00,0.001\n")
    pressure = tmp_path / "pressure.csv"
    pressure.write_text("datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,temperature_gradient_celsius\n2023-01-02T10:00:00+01:00,1,20,0,0\n")
    overflow = tmp_path / "overflow.csv"
    overflow.write_text("time_h,weight_g\n-0.1,391.2\n1,400.5\n")
    config = tmp_path / "run.toml"
    section = dict(imaging={str(folder): str(imaging)}, injection=str(injection),
                   pressure_temperature=str(pressure))
    for reference in (None, str(overflow)):
        section["overflow"] = "" if reference is None else reference
        config.write_text(toml.dumps({"protocols": section}))
        protocols = ProtocolsConfig().load(config)
        experiment = ProtocolledExperiment.init_from_config(
            SimpleNamespace(data=SimpleNamespace(data=[], folders=[folder]), protocols=protocols))
        if reference is None:
            assert experiment.overflow_protocol is None
        else:
            assert experiment.overflow_protocol.df["weight_g"].tolist() == [391.2, 400.5]
        assert experiment.injection_protocol.injected_mass(time=0.5) == pytest.approx(1.8)


def test_direct_protocol_is_displayed_without_reading_table_or_old_project(tmp_path):
    root = tmp_path / "series"
    (root / "config/runs").mkdir(parents=True)
    (root / "series.toml").write_text(toml.dumps({
        "schema_version": 1, "series_id": "example", "runs": {"ids": ["r1", "r2"]},
        "legacy": {"studio_project": "missing/series.json"},
    }))
    # An unreadable table still means Configured, not reviewed/valid measurements.
    selected = root / "overflow.xlsx"
    selected.write_bytes(b"not a workbook")
    (root / "config/runs/r1.toml").write_text(toml.dumps({
        "protocols": {"overflow": [str(selected), "r1"]}}))
    (root / "config/runs/r2.toml").write_text("[protocols]\noverflow = ''\n")
    before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
    data = read_overflow_data(root)
    assert "1 of 2" in data.summary
    assert data.runs[0].file.text == "overflow.xlsx [r1]"
    assert data.runs[0].status.text == "Configured"
    assert data.runs[1].status.text == "Not configured"
    selected.unlink()
    assert read_overflow_data(root).runs[0].status.text == "Unavailable"
    assert {p: p.read_bytes() for p in before if p != selected} == {
        p: content for p, content in before.items() if p != selected}
