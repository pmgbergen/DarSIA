"""Selected Store validation replays only that run's admitted gap exclusion."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_adoption_table as table
from darsia.fluidflower import painted_analysis_exclusions as exclusions

from .test_painted_analysis_exclusions import (  # noqa: F401
    ACTION, _approve, _fresh, gap_case, imported, registered_setup,
    synthetic_payload_admission,
)


@pytest.fixture
def admitted(gap_case):
    preparation = _fresh(gap_case, _approve(gap_case.review))
    declaration = exclusions.make_facies_source_acquisition(
        preparation, "indicator", **ACTION
    )
    config, _ = adoption.adoption_candidate(
        gap_case.data.revision, preparation["folder"],
        facies_source_acquisition=declaration,
    )
    # Retain the real complete admission and its existing dependency memo.
    sources.canonical_parent_payload(config)
    return SimpleNamespace(config=config, preparation=preparation)


def observe_exclusions(monkeypatch):
    original, observed = exclusions.prepared_exclusion, []

    def checked(preparation, folder, run):
        observed.append(run)
        return original(preparation, folder, run)

    monkeypatch.setattr(exclusions, "prepared_exclusion", checked)
    return observed


@pytest.mark.parametrize("run", ["one", "two"])
def test_selected_validation_replays_real_exclusions_only_for_that_run(
        admitted, monkeypatch, run):
    observed = observe_exclusions(monkeypatch)
    result = sources.canonical_parent_payload(admitted.config, run_id=" " + run.upper() + " ")
    assert result["painted_table_bytes"]
    # Both the canonical contract and selected registration keep their actual
    # exclusion checks; neither may replay the other run's native arrays.
    assert observed == [run, run]


def test_full_admission_still_replays_every_run_exclusion(admitted, monkeypatch):
    observed = observe_exclusions(monkeypatch)
    sources.canonical_parent_payload(admitted.config)
    assert observed == ["one", "two", "one", "two"]


def test_selected_validation_rejects_corrupt_actual_mask_even_with_a_colliding_stamp(
        admitted, monkeypatch):
    selected = "two"
    mask_path = Path(admitted.preparation[exclusions.KEY]["runs"][selected]["artifact"]["path"])
    with np.load(mask_path, allow_pickle=False) as archive:
        mask = archive["approved_analysis_gap"].copy()
    assert mask.any()
    old_stamp, original_stamp = table._FILES.stamp(mask_path), table._FILES.stamp
    # Keep the already admitted common derivation warm, then exercise the
    # selected run's real numeric mask validation against changed source bytes.
    monkeypatch.setattr(table._FILES, "stamp", lambda path:
        old_stamp if Path(path) == mask_path else original_stamp(path))
    np.savez_compressed(mask_path, approved_analysis_gap=np.zeros_like(mask))
    observed = observe_exclusions(monkeypatch)
    with pytest.raises(ValueError, match="stored approved-analysis-gap mask changed"):
        sources.canonical_parent_payload(admitted.config, run_id=selected)
    assert observed == [selected]


@pytest.mark.parametrize("damage", ["unknown_selection", "missing", "duplicate", "unknown_row"])
def test_invalid_run_inventory_rejects_before_any_exclusion_replay(
        admitted, monkeypatch, damage):
    config = deepcopy(admitted.config)
    rows = config["run_geometry"]["runs"]
    selected = "two"
    if damage == "unknown_selection":
        selected = "unknown"
    elif damage == "missing":
        rows[:] = [row for row in rows if row["run_id"] != selected]
    elif damage == "duplicate":
        rows.append(deepcopy(next(row for row in rows if row["run_id"] == selected)))
    else:
        next(row for row in rows if row["run_id"] == selected)["run_id"] = "unknown"
    observed = observe_exclusions(monkeypatch)
    with pytest.raises(ValueError, match="selected run|run rows differ"):
        sources.canonical_parent_payload(config, run_id=selected)
    assert observed == []
