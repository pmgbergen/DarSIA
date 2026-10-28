"""Explicit grouping of ordinary runs into injection cycles.

This is provenance metadata, not a baseline, time-zero, dose or background-DIC
model. Each member still uses its own ordinary run configuration. No existing
run is split, renamed or migrated by reading this optional manifest section.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re


FORMAT = "darsia-cycle-runs-v1"
_ID = re.compile(r"[a-z0-9][a-z0-9_-]*")


def _identifier(value: object) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


@dataclass(frozen=True)
class CycleRun:
    """One explicitly identified ordinary run and its previous-cycle link."""

    run_id: str
    experiment: str
    number: int
    previous_run_id: str | None = None

    def __post_init__(self):
        if not _identifier(self.run_id) or not _identifier(self.experiment):
            raise ValueError("Cycle run and experiment IDs must be explicit lower-case IDs")
        if type(self.number) is not int or self.number < 1:
            raise ValueError("Cycle number must be a positive integer")
        if self.number == 1:
            if self.previous_run_id is not None:
                raise ValueError("A first cycle cannot name a previous cycle")
        elif not _identifier(self.previous_run_id):
            raise ValueError("A later cycle requires its explicit previous run ID")


def cycle_runs_table(cycles: Sequence[CycleRun], ordered_run_ids: Sequence[str]) -> dict:
    """Validate complete, grouped cycle provenance in the manifest run order."""
    rows, ids = tuple(cycles), tuple(ordered_run_ids)
    if (not rows or not all(isinstance(row, CycleRun) for row in rows)
            or tuple(row.run_id for row in rows) != ids or len(set(ids)) != len(ids)):
        raise ValueError("Cycle metadata must cover every run exactly in manifest order")
    previous: dict[str, CycleRun] = {}
    current_group = None
    for row in rows:
        last = previous.get(row.experiment)
        if last is not None and current_group != row.experiment:
            raise ValueError("Cycles of an experiment must be grouped together")
        expected_number = 1 if last is None else last.number + 1
        expected_previous = None if last is None else last.run_id
        if row.number != expected_number or row.previous_run_id != expected_previous:
            raise ValueError("Cycle order and previous run must match the same experiment")
        previous[row.experiment] = row
        current_group = row.experiment
    records = {}
    for row in rows:
        value = {"experiment": row.experiment, "number": row.number}
        if row.previous_run_id is not None:
            value["previous_run_id"] = row.previous_run_id
        records[row.run_id] = value
    return {"format": FORMAT, "runs": records}


def parse_cycle_runs(manifest: Mapping) -> tuple[CycleRun, ...]:
    """Read only an explicitly present cycle section; legacy manifests are unchanged."""
    if "cycle_runs" not in manifest:
        return ()
    value = manifest["cycle_runs"]
    if (not isinstance(value, Mapping) or set(value) != {"format", "runs"}
            or value["format"] != FORMAT or not isinstance(value["runs"], Mapping)):
        raise ValueError("Unknown or malformed cycle-run manifest section")
    runs = manifest.get("runs", {})
    ids = runs.get("ids") if isinstance(runs, Mapping) else None
    if not isinstance(ids, list) or any(not _identifier(run) for run in ids):
        raise ValueError("Cycle metadata requires explicit ordered manifest run IDs")
    records = value["runs"]
    if set(records) != set(ids):
        raise ValueError("Cycle metadata must cover every manifest run, without extra runs")
    cycles = []
    for run_id in ids:
        row = records[run_id]
        if (not isinstance(row, Mapping) or not {"experiment", "number"} <= set(row)
                or set(row) - {"experiment", "number", "previous_run_id"}):
            raise ValueError("Only explicit experiment, number and previous run are cycle metadata")
        cycles.append(CycleRun(run_id, row["experiment"], row["number"], row.get("previous_run_id")))
    cycle_runs_table(cycles, ids)
    return tuple(cycles)
