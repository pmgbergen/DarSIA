"""The provenance of a placed painted map, what its adoption still lacks, and
the preparation of its adoption on each run's rig-corrected native grid.

A synthetic placement and synthetic geometry models in a temporary folder
(the geometry index form of the placement tests); nothing is adopted.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import map_adoption as ma
from darsia.fluidflower import map_placement as mp

from .test_map_placement import REFERENCE_H, SHIFTED_H, _index, _layers


def _placement(tmp_path, version):
    index = _index(tmp_path / "geometry", {"run1": REFERENCE_H, "run2": SHIFTED_H})
    return mp.place_map_in_runs(
        tmp_path / "reviews",
        _layers(),
        reference_run="run2",
        geometry_index=index,
        runs=["run1", "run2", "run3"],
        photo_loader=lambda run, geometry: np.full(geometry.shape_hw + (3,), 50, dtype=np.uint8),
        roles={1: "free_water", 2: "porous_medium", 3: "porous_medium"},
        facies={1: "water", 2: "C", 3: "ESF"},
        version=version,
        measure_offsets=False,
    )


def test_the_provenance_binds_every_placed_run_by_sha256(tmp_path):
    version = {"name": "Map A", "sha256": "a" * 64, "approval": {"sha256": "a" * 64}}
    result = _placement(tmp_path, version)
    record = ma.placement_provenance(result["folder"])
    assert record["format"] == ma.PROVENANCE_FORMAT
    assert record["reference_run"] == "run2" and record["approval"] == {"sha256": "a" * 64}
    index = record["placement"]["index"]
    with open(index["path"], "rb") as stream:
        assert index["sha256"] == hashlib.sha256(stream.read()).hexdigest()
    assert [row["run_id"] for row in record["runs"]] == ["run2", "run1"]
    for row in record["runs"]:
        with open(row["layers"]["path"], "rb") as stream:
            assert row["layers"]["sha256"] == hashlib.sha256(stream.read()).hexdigest()
    assert record["incomplete_runs"] == [{"run_id": "run3", "status": "no geometry"}]
    # A placed run whose layer map went missing is refused.
    (Path(result["folder"]) / "run1" / "layers.npz").unlink()
    with pytest.raises(FileNotFoundError, match="run1: the placed layer map is missing"):
        ma.placement_provenance(result["folder"])


def test_the_blockers_name_what_an_adoption_still_lacks(tmp_path):
    record = ma.placement_provenance(
        _placement(tmp_path, {"name": "Map A", "sha256": "a" * 64, "approval": {"sha256": "a" * 64}})["folder"]
    )
    assert ma.map_facies(record) == ["C", "ESF", "water"]
    assert ma.adoption_blockers(record, facies=["ESF", "C"]) == [
        "Not placed: RUN3.",
        "No transport from the camera grid to the run's label grid: RUN2, RUN1.",
        "No facies id for: C, ESF.",
    ]
    assert ma.adoption_blockers(
        record, transports={"run1": {}, "run2": {}}, facies_ids={"ESF": 1, "C": 2}, facies=["ESF", "C"]
    ) == ["Not placed: RUN3."]
    record["approval"] = {"sha256": "b" * 64}
    assert ma.adoption_blockers(record)[0] == (
        "The placed map version has no approval of its own SHA-256."
    )


# The native grid is the camera grid without its top 5 rows: 30 x 60.
NATIVE = {
    "shape_hw": [30, 60],
    "H_baseline_camera_u_to_rig_corrected_native": [[1.0, 0.0, 0.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]],
}


def _native_index(folder):
    path = _index(folder, {"run1": REFERENCE_H, "run2": SHIFTED_H, "run4": REFERENCE_H})
    index = json.loads(path.read_text(encoding="utf-8"))
    for run in ("run1", "run2"):
        model_path = Path(index["runs"][run]["reference_model"]["path"])
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["baseline_native_grid"] = NATIVE
        model_path.write_text(json.dumps(model), encoding="utf-8")
        index["runs"][run]["reference_model"] = {"path": str(model_path)}
    path.write_text(json.dumps(index), encoding="utf-8")
    return path


def test_the_map_is_carried_to_a_runs_native_label_grid(tmp_path):
    geometry = mp.read_run_geometry(_native_index(tmp_path / "geometry"))
    matrix, shape = ma.native_transport(geometry["run2"])
    assert shape == (30, 60) and np.allclose(matrix, NATIVE["H_baseline_camera_u_to_rig_corrected_native"])
    labels, filled, support, composition = ma.native_labels(_layers(), geometry["run1"], geometry["run2"])
    # Native (row, col) is the target's camera (row + 5, col), the reference's (row + 5, col - 3).
    assert np.allclose(composition, [[1.0, 0.0, 3.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]])
    assert labels.shape == (30, 60)
    assert labels[0, 32] == 2 and labels[0, 33] == 3
    assert labels[14, 10] == 2 and labels[15, 10] == 1
    assert filled[:, :3].all() and not filled[:, 3:].any()
    assert not support[:, :3].any() and support[:, 3:].all()
    with pytest.raises(ValueError, match="run4: the geometry has no baseline native grid"):
        ma.native_transport(geometry["run4"])


def test_a_preparation_writes_each_runs_native_map_and_adopts_nothing(tmp_path):
    index = _native_index(tmp_path / "geometry")
    seen = []

    def current(run):
        seen.append(run)
        shape = [30, 60] if run == "run1" else [31, 60]
        return {"metadata": {"dimensions": [0.49, 0.9], "run": run}, "shape_hw": shape, "source": {"path": f"{run}/labels.npz"}}

    result = ma.prepare_adoption(
        tmp_path / "adoption",
        _layers(),
        reference_run="run1",
        geometry_index=index,
        runs=["run1", "run2", "run3", "run4"],
        roles={1: "free_water", 2: "porous_medium", 3: "porous_medium"},
        names={1: "Water", 2: "Fine", 3: "Coarse"},
        facies={1: "water", 2: "C", 3: "ESF"},
        facies_ids={"C": 2},
        version={"name": "Map A", "sha256": "a" * 64},
        current_labels=current,
    )
    folder = Path(result["folder"])
    assert [(row["run_id"], row["status"]) for row in result["runs"]] == [
        ("run1", "prepared"), ("run2", "grid differs"), ("run3", "no geometry"), ("run4", "no native grid"),
    ]
    assert seen == ["run1", "run2"]
    with np.load(folder / "run1" / "labels.npz", allow_pickle=True) as saved:
        assert saved["array"].dtype == np.int32 and saved["array"].shape == (30, 60)
        assert np.array_equal(saved["array"], _layers()[5:35].astype(np.int32))
        assert saved["metadata"].item() == {"dimensions": [0.49, 0.9], "run": "run1"}
        assert not saved["filled"].any() and saved["support"].all()
    row = result["runs"][0]
    assert row["labels"]["sha256"] == hashlib.sha256((folder / "run1" / "labels.npz").read_bytes()).hexdigest()
    assert row["current_labels"] == {"path": "run1/labels.npz"}
    assert [(region["facies"], region["facies_id"]) for region in result["regions"]] == [
        ("water", None), ("C", 2), ("ESF", None),
    ]
    assert result["blockers"] == [
        "Not prepared: RUN2, RUN3, RUN4.",
        f"No facies id for: ESF, water ({ma.FACIES_IDS_PENDING}).",
    ]
    assert result["adopted"] is False
    assert json.loads((folder / "index.json").read_text(encoding="utf-8"))["runs"][0]["status"] == "prepared"
    assert sorted(path.name for path in folder.iterdir()) == [
        "approval.json", "index.json", "run1", "source_map.npz",
    ]
    assert result["runs"][0]["label_values"] == [1, 2, 3]
    with np.load(folder / "source_map.npz") as saved:
        assert np.array_equal(saved["array"], _layers())
