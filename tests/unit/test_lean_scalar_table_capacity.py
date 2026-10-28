"""Bound the canonical numeric table with the original scalar-table writer.

These deliberately long labels and extreme finite numbers are synthetic.
The actual sealed run must still report its exact selected fields and case IDs.
"""
import csv
import hashlib
import json
import sys

import pytest

from darsia.fluidflower.pipeline import io_ops, runner


@pytest.mark.parametrize("with_hidden_mass", (False, True))
def test_eighty_numeric_pixel_rows_fit_the_lean_run_record_ceiling(tmp_path, with_hidden_mass):
    """Exercise real JSON reads, UTF-8 quoting and the bounded CSV publisher."""
    fields = ["case_id", "M_measured_g", "M_edge_g", "U_g",
              "phase_unknown_no_capacity_area_cm2"]
    numbers = [-sys.float_info.max, sys.float_info.max, float.fromhex("0x0.0000000000001p-1022"), -0.]
    records, cases = [], []
    for index in range(80):
        # Much longer than ordinary frame IDs, with multibyte and CSV escaping.
        case = "case-" + "\u03c7" * 128 + ',"\r\n' + str(index)
        cases.append(case)
        directory = tmp_path / ("frame-" + str(index))
        directory.mkdir()
        result = dict(zip(fields, [case, *numbers], strict=True), complete=True)
        result["pixel_domain"] = dict(contract="darsia-pixel-domain-v1",
                                      diagnostic="synthetic unselected metadata" * 100)
        if with_hidden_mass:
            result["pixel_domain"]["n16"] = dict(contract="darsia-n16-q85-a-v1",
                                                U_total_g=numbers[2])
        raw = json.dumps(result, ensure_ascii=False, allow_nan=False).encode("utf-8")
        path = directory / "result.json"
        path.write_bytes(raw)
        records.append(dict(frame_id=case, nas_directory=str(directory), files={
            "result.json": dict(sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))}))
    # Numeric fields are finite binary64 values. Reserve 32 ASCII bytes per
    # float, plus worst-case UTF-8/doubled quotes and delimiters for every row.
    escaped_case = max(len(case.encode("utf-8")) + case.count('"') for case in cases)
    header_bytes = sum(len(field.encode("utf-8")) for field in fields) + len(fields) - 1 + 2
    row_bytes = escaped_case + 2 + 4 * 32 + len(fields) - 1 + 2
    bound = header_bytes + len(records) * row_bytes
    assert bound < io_ops.LEAN_METADATA_BYTES // 100
    destination = tmp_path / "mass_table.csv"
    with io_ops.metadata_ceiling(io_ops.LEAN_METADATA_BYTES):
        receipt = runner.scalar_table(records, fields, destination)
    raw = destination.read_bytes()
    assert len(raw) <= bound
    assert receipt["size_bytes"] == len(raw)
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    with destination.open(encoding="utf-8", newline="") as stream:
        decoded = list(csv.DictReader(stream))
    assert [row["case_id"] for row in decoded] == cases
    assert all([float(row[field]) for field in fields[1:]] == numbers for row in decoded)
    assert not list(tmp_path.glob(".mass_table.csv.*.partial"))
