"""The original Protocols action must reuse workbook file/worksheet inputs."""
from pathlib import Path

import openpyxl
import pytest
import toml

from darsia.presets.workflows.setup import setup_protocols as module


def workbook(path, columns, rows):
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "r1"
    sheet.append(columns)
    for row in rows:
        sheet.append(row)
    book.save(path)
    return [str(path), "r1"]


@pytest.fixture
def selected(tmp_path):
    imaging = workbook(tmp_path / "imaging.xlsx", ["path", "datetime"],
        [["first.ARW", "2023-01-01T00:00:00+01:00"]])
    injection = workbook(tmp_path / "injection.xlsx",
        ["id", "location_x", "location_y", "start", "end", "rate_sccm", "density kg/m3"],
        [[1, 0.5, 0.02, "2023-01-01T00:00:00+01:00", "2023-01-01T01:00:00+01:00", 10, 1.872]])
    pressure = workbook(tmp_path / "pressure.xlsx",
        ["datetime", "pressure_bar", "temperature_celsius", "pressure_gradient_bar", "temperature_gradient_celsius"],
        [["2023-01-01T00:00:00+01:00", 0.998798, 23.0, 0.0, 0.0]])
    blacklist = workbook(tmp_path / "blacklist.xlsx", ["path"], [["excluded.ARW"]])
    overflow = workbook(tmp_path / "overflow.xlsx", ["datetime", "weight_g"],
        [["2023-01-01T00:00:00+01:00", 12.5]])
    document = dict(data=dict(folders=[str(tmp_path / "RAW_not_present")],
        baseline="first.ARW", results=str(tmp_path / "results_not_created")),
        protocols=dict(imaging={str(tmp_path / "RAW_not_present"): imaging},
            injection=injection, pressure_temperature=pressure,
            blacklist=blacklist, overflow=overflow))
    path = tmp_path / "run.toml"
    path.write_text(toml.dumps(document), encoding="utf-8")
    return path, document


def test_run_and_preview_read_selected_sheets_without_raw_or_output_io(selected, monkeypatch, caplog):
    path, document = selected
    original = {file: file.read_bytes() for file in path.parent.iterdir()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Workbook-backed protocol setup cannot enumerate images or generate output")
    monkeypatch.setattr(module.os, "listdir", forbidden)
    monkeypatch.setattr(module.Image, "open", forbidden)
    monkeypatch.setattr(module, "_write_csv", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    with caplog.at_level("INFO"):
        assert module.preview_protocol_setup_conflicts(path) == []
        module.setup_imaging_protocol(path, force=True)
    assert "Generation options apply to CSV templates only" in caplog.text
    assert all(file.read_bytes() == raw for file, raw in original.items())
    assert not Path(document["data"]["results"]).exists()


@pytest.mark.parametrize("kind", ["imaging", "injection", "pressure_temperature", "blacklist", "overflow"])
def test_missing_selected_worksheet_is_reported_without_replacing_inputs(selected, kind):
    path, document = selected
    reference = (next(iter(document["protocols"][kind].values())) if kind == "imaging"
                 else document["protocols"][kind])
    reference[1] = "missing"
    path.write_text(toml.dumps(document), encoding="utf-8")
    originals = {file: file.read_bytes() for file in path.parent.iterdir()}
    with pytest.raises(ValueError, match="Worksheet named 'missing' not found"):
        module.setup_imaging_protocol(path)
    assert all(file.read_bytes() == raw for file, raw in originals.items())


def test_one_primary_workbook_keeps_the_rest_of_selected_csv_set_read_only(selected, monkeypatch):
    path, document = selected
    pressure = path.parent / "pressure.csv"
    pressure.write_text("datetime,pressure_bar,temperature_celsius,pressure_gradient_bar,temperature_gradient_celsius\n"
        "2023-01-01T00:00:00+01:00,0.998798,23,0,0\n", encoding="utf-8")
    document["protocols"]["pressure_temperature"] = str(pressure)
    path.write_text(toml.dumps(document), encoding="utf-8")
    def forbidden(*args, **kwargs):
        raise AssertionError("Existing mixed CSV/Excel protocol sets must not be regenerated")
    monkeypatch.setattr(module, "_write_pressure_temperature_template", forbidden)
    before = pressure.read_bytes()
    module.setup_imaging_protocol(path)
    assert pressure.read_bytes() == before
