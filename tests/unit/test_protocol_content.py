"""A corrected injection position keeps a completed Setup; other protocol changes do not."""
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from darsia.fluidflower import protocol_content as subject


def _workbook(path, *, x=0.72, start="2023-02-09 13:00:00", rate=1e-6):
    """Write a two-sheet injection workbook with the reader's columns."""
    rows = lambda: pd.DataFrame(dict(id=[1, 2], location_x=[x, 0.30], location_y=[0.18, 0.10],
                                     start=[start, "2023-02-09 15:00:00"],
                                     end=["2023-02-09 14:00:00", "2023-02-09 16:00:00"],
                                     **{"rate_kg/s": [rate, 2e-6]}))
    with pd.ExcelWriter(path) as writer:
        rows().to_excel(writer, sheet_name="ac17", index=False)
        rows().to_excel(writer, sheet_name="ac18", index=False)
    return path


def test_the_schedule_digest_leaves_out_the_injection_positions(tmp_path):
    first = subject.injection_schedule_digest(_workbook(tmp_path / "a.xlsx"), "ac17")
    assert subject.injection_schedule_digest(_workbook(tmp_path / "b.xlsx", x=0.74), "ac17") == first
    assert subject.injection_schedule_digest(_workbook(tmp_path / "c.xlsx", start="2023-02-09 13:00:05"), "ac17") != first
    assert subject.injection_schedule_digest(_workbook(tmp_path / "d.xlsx", rate=2e-6), "ac17") != first


def _row(path, role="injection", sheet="ac17"):
    from darsia.fluidflower.rig_raw_build import _descriptor
    return dict(role=role, folder=None, sheet=sheet, source=_descriptor(path))


def test_only_injection_rows_get_the_schedule_digest(tmp_path):
    book = _workbook(tmp_path / "injection.xlsx")
    other = tmp_path / "pressure.csv"
    other.write_text("time,pressure\n0,1\n")
    rows = subject.with_content([_row(book), _row(other, role="pressure_temperature", sheet=None)])
    assert rows[0]["content"] == dict(format=subject.FORMAT,
                                      sha256=subject.injection_schedule_digest(book, "ac17"))
    assert "content" not in rows[1]
    assert subject.without_content(rows) == [_row(book), _row(other, role="pressure_temperature", sheet=None)]
    # A file the injection reader cannot parse keeps only its exact byte pin.
    assert "content" not in subject.with_content([_row(other, sheet=None)])[0]


def test_a_changed_position_reads_the_current_file_and_other_changes_refuse(tmp_path):
    book = _workbook(tmp_path / "injection.xlsx")
    row = subject.with_content([_row(book)])[0]
    assert subject.current_source(row) == row["source"]
    _workbook(book, x=0.74)
    current = subject.current_source(row)
    assert current["path"] == row["source"]["path"] and current["sha256"] != row["source"]["sha256"]
    assert current == _row(book)["source"]
    # A Setup made before the digest keeps its exact check: its consumer gets the recorded pin.
    legacy = subject.without_content([row])[0]
    assert subject.current_source(legacy) is legacy["source"]
    _workbook(book, x=0.74, start="2023-02-09 13:00:05")
    with pytest.raises(ValueError, match="changed beyond"):
        subject.current_source(row)


def test_the_consumers_own_check_decides_first(tmp_path):
    book = _workbook(tmp_path / "injection.xlsx")
    row = subject.with_content([_row(book)])[0]

    def unreadable(path):
        raise AssertionError("A passing check must not read the file again")

    assert subject.current_source(row, check=lambda pin: None, physical=unreadable) is row["source"]

    def refuses(pin):
        raise ValueError("size differs")

    _workbook(book, x=0.74)
    assert subject.current_source(row, check=refuses)["sha256"] == _row(book)["source"]["sha256"]


def test_the_build_check_keeps_a_build_after_a_corrected_position(tmp_path):
    from darsia.fluidflower.rig_raw_basis import protocol_sources, require_protocol_sources
    book = _workbook(tmp_path / "injection.xlsx")
    imaging = tmp_path / "imaging.csv"
    imaging.write_text("path,datetime\nDSC00001.ARW,2023-02-09 12:00:00\n")
    pressure = tmp_path / "pressure.csv"
    pressure.write_text("time,pressure\n0,1\n")
    protocols = SimpleNamespace(imaging=imaging, injection=(book, "ac17"),
                                pressure_temperature=pressure, blacklist=None)
    request = dict(protocol_inputs=subject.with_content(protocol_sources(protocols)))
    require_protocol_sources(request, protocols)
    _workbook(book, x=0.74)
    require_protocol_sources(request, protocols)
    legacy = dict(protocol_inputs=subject.without_content(request["protocol_inputs"]))
    with pytest.raises(ValueError, match="differ from the exact bound files"):
        require_protocol_sources(legacy, protocols)
    _workbook(book, x=0.74, rate=3e-6)
    with pytest.raises(ValueError, match="differ from the exact bound files"):
        require_protocol_sources(request, protocols)
