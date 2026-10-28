"""The re-export gate: each run's built rig against its adopted labels.

The synthetic two-run registered workspace of the registered export tests.
Run one is exported and applied; a stand-in rig folder plays the built rig
(the fixture's run configurations name no rig.path). Reads only, apart from
the receipt. Temporary folders only.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from darsia.fluidflower import corrected_setup as export
from darsia.fluidflower import corrected_setup_all_runs as all_runs
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import series_geometry

from .test_geometry_imported_rasters import imported  # noqa: F401
from .test_geometry_registered_export_apply import registered_setup  # noqa: F401


def test_the_rebuilt_check_compares_each_built_rig_with_its_adopted_labels(
    registered_setup, tmp_path, monkeypatch
):
    data = registered_setup
    output = export.export_corrected_setup(
        data.project, reference=SimpleNamespace(run_id="one"), actor="synthetic owner"
    )
    export.apply_corrected_selected_run(data.project, output)
    rig = tmp_path / "built-rig"
    # The fixture's configurations name no rig.path: point the check at a stand-in.
    monkeypatch.setattr(series_geometry, "_storage_path", lambda value, base: rig)
    assert all_runs.rebuilt_run_check(data.project, "one")["status"] == "not built"
    missing = all_runs.rebuilt_run_check(data.project, "two")
    assert missing["status"] == "not applied" and "FileNotFoundError" in missing["detail"]
    rig.mkdir()
    for name in ("labels", "facies"):
        adopted = geometry._load_array(Path(output.receipt["binding"]["sources"][name]["path"]))
        np.savez(rig / f"{name}.npz", array=adopted)
    row = all_runs.rebuilt_run_check(data.project, "one")
    assert row["status"] == "identical"
    assert row["comparisons"]["labels"] == {
        "identical": True, "different_pixels": 0, "built_shape": [2, 3], "adopted_shape": [2, 3],
    }
    changed = geometry._load_array(rig / "labels.npz").copy()
    changed[0, 0] += 1
    np.savez(rig / "labels.npz", array=changed)
    row = all_runs.rebuilt_run_check(data.project, "one")
    assert row["status"] == "differs" and row["comparisons"]["labels"]["different_pixels"] == 1
    assert row["comparisons"]["facies"]["identical"]
    path, result = all_runs.check_rebuilt_runs(
        data.project, [SimpleNamespace(run_id="two"), SimpleNamespace(run_id="one")]
    )
    assert [(item["run_id"], item["status"]) for item in result["runs"]] == [
        ("one", "differs"), ("two", "not applied"),
    ]
    assert not result["all_identical"] and result["identical"] == []
    assert json.loads(path.read_text(encoding="utf-8"))["revision_id"] == data.revision["revision_id"]
    assert path.parent == output.destination
