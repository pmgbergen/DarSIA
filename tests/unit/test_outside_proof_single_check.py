"""One outside-proof operation authenticates all runs once, with no stale reuse."""
from pathlib import Path

import pytest

from darsia.fluidflower import local_map_adoption as local
from darsia.fluidflower import painted_adoption_table as table
from .test_local_map_adoption import _prepare
from .test_painted_analysis_exclusions import (
    gap_case, registered_setup, imported, synthetic_payload_admission,
)  # noqa: F401 -- genuine finite fixture chain


def test_outside_proof_checks_full_closure_once_and_rejects_later_changed_product(gap_case, monkeypatch):
    prepared, _ = _prepare(gap_case, gap_case.root / "single_outside_check")
    _, approved, frame = table.verify_version(prepared["painting_facies"])
    actual = local.verify_local_preparation
    calls = []

    def checked(*args, **kwargs):
        calls.append(1)
        return actual(*args, **kwargs)

    monkeypatch.setattr(local, "verify_local_preparation", checked)
    table.verify_outside_proof(prepared, prepared["folder"], approved, frame)
    assert len(prepared["runs"]) > 1
    assert len(calls) == 1

    # A later operation must authenticate again; a previous success is not a
    # process-wide cache or permission to accept changed consumed bytes.
    product = Path(prepared["folder"]) / prepared["runs"][-1]["labels"]["path"]
    product.write_bytes(product.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        table.verify_outside_proof(prepared, prepared["folder"], approved, frame)
    assert len(calls) == 2
