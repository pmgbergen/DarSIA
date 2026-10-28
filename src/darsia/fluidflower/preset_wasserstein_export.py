"""Explicit local CSV export through the original Comparison table assembler."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path, PureWindowsPath


def export_preset_wasserstein_csv(job_root, destination, *, reader=None):
    """Assemble verified shared records in original order into one fresh folder.

    The original scientific JSON is read only from hash-bound shared copies.
    The original assembler retains its scalar ordering, precision and NaN CSV
    representation. No input store, producer-local output or solver is opened.
    """
    from .jobs_metadata import JobSnapshotReader
    from .mass.n1_pipeline import _write_json
    from .pipeline.job_queue import descriptor, name, _own_asset
    from .pipeline.products_publication import _read_index, _json
    from .preset_wasserstein_protocol import validate_preset_wasserstein_job
    from .preset_wasserstein_results import preset_wasserstein_terminal, _scientific_row, _filename, _check_receipt, RECEIPT
    from .published_jobs_snapshot import _SnapshotMetadata
    from darsia.presets.workflows.comparison.comparison_wasserstein import (
        WassersteinDistanceResult, _assemble_all_wasserstein_results,
    )

    root, output = Path(job_root).resolve(), Path(destination)
    if (not output.is_absolute() or PureWindowsPath(str(output)).drive.startswith("\\\\")
            or str(output).startswith("//") or output.exists() or output.is_symlink()
            or output.resolve() != output or output.is_relative_to(root)
            or not output.parent.is_dir()):
        raise ValueError("Choose an explicit fresh local export folder outside the shared job, with an existing parent")
    reader = reader or JobSnapshotReader().read
    sources = []

    def read(path, expected=None):
        bound = descriptor(path)
        if expected is not None and bound != expected:
            raise ValueError("Shared export metadata descriptor changed")
        value = reader(path, bound)
        sources.append(bound)
        return value

    job = read(root / "job.json")
    validate_preset_wasserstein_job(job)
    records, snapshots, publications = [], [], []
    statuses, nonfinite, normalized_nonfinite = Counter(), 0, 0
    for run in job["expected_runs"]:
        task = read(root / "results" / name(run) / "task.json")
        done = read(root / "done" / (name(run) + ".json"))
        if task["run"] != run:
            raise ValueError("Export requested-time task differs from its job")
        completed, _, _ = preset_wasserstein_terminal(root, task, job, done, read)
        index = descriptor(_own_asset(root, done["products_index"]))
        folder, _document, products = _read_index(index)
        snapshot = _SnapshotMetadata(index, folder, products)
        snapshots.append(snapshot)
        if _json(snapshot.read("completed.json")[1]) != completed:
            raise ValueError("Shared export completion changed after terminal verification")
        _check_receipt(completed, _json(snapshot.read(RECEIPT)[1]))
        publications.append(dict(run=run, comparison_time=completed["comparison_time"],
            products_index=index, execution_receipt=completed["execution_receipt"],
            comparisons=completed["comparisons"]))
        for declaration in completed["comparisons"]:
            _, raw = snapshot.read(_filename(declaration))
            _scientific_row(raw, declaration)
            records.append(WassersteinDistanceResult(**json.loads(raw.decode("utf-8-sig"))))
        snapshot.finish()
        statuses.update(completed["result_statuses"])
        nonfinite += completed["nonfinite_distances"]
        normalized_nonfinite += completed["nonfinite_normalized_distances"]

    def finish():
        for source in sources:
            if descriptor(Path(source["path"])) != source:
                raise ValueError("Shared export source changed during verification")
        for snapshot in snapshots:
            snapshot.finish()

    finish()
    output.mkdir()  # Exclusive destination, after every requested time was verified.
    _assemble_all_wasserstein_results(None, result_records=records, output_dir=output / "tables")
    csv = descriptor(output / "tables" / "wasserstein_distances_all.csv")
    finish()
    evidence = dict(format="darsia-preset-wasserstein-csv-export-v1", pipeline="preset-wasserstein",
        complete=True, job=sources[0], source_metadata=sources, publications=publications,
        comparison_count=len(records), nonfinite_distances=nonfinite,
        nonfinite_normalized_distances=normalized_nonfinite, result_statuses=dict(statuses),
        csv=csv, validation=dict(state="not_scored", reason="Original result table export is not scientific acceptance"))
    receipt = _write_json(output / "export_receipt.json", evidence)
    return dict(root=str(output), csv=csv, receipt=receipt, comparison_count=len(records),
        nonfinite_distances=nonfinite, nonfinite_normalized_distances=normalized_nonfinite,
        result_statuses=dict(statuses))
