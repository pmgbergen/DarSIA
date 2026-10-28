"""Adopting a prepared painted map as a genuine Store revision, on a copy.

The synthetic two-run corrected workspace of the registered export tests
(a canonical edit of an imported parent, runs 'one' and 'two' on a 2 x 3
grid). A painted map on run one's camera grid is prepared for both runs
through a synthetic measured geometry, adopted with the facies table of the
map, saved by the real Store and read back, exported and applied by the
existing per-run route. Temporary folders only.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import corrected_setup as export
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import map_adoption as ma
from darsia.fluidflower import map_placement as mp

from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_geometry_imported_rasters import imported  # noqa: F401

MAP = np.array([[5, 5, 6], [5, 6, 6]], dtype=np.uint8)
IDENTITY = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
# Run two's camera sees the rig one pixel further right.
SHIFTED = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def _geometry(folder):
    folder.mkdir()
    runs = {}
    for run, matrix in (("one", IDENTITY), ("two", SHIFTED)):
        model = {
            "format": mp.GEOMETRY_FORMAT,
            "run_id": run,
            "shape_hw": [2, 3],
            "H_metric_mm_to_camera_u": matrix,
            "reference_image": {},
            "baseline_native_grid": {
                "shape_hw": [2, 3],
                "H_baseline_camera_u_to_rig_corrected_native": IDENTITY,
            },
        }
        path = folder / f"{run}.json"
        path.write_text(json.dumps(model), encoding="utf-8")
        runs[run] = {"reference_model": {"path": str(path)}}
    index = folder / "index.json"
    index.write_text(json.dumps({"format": mp.GEOMETRY_FORMAT, "runs": runs}), encoding="utf-8")
    return index


def _prepare(data, tmp_path, approval="a" * 64):
    parent_labels = data.config["canonical_import_parent"]["imported_configuration"]["labels_path"]
    grid = {
        "metadata": geometry._darsia_metadata(Path(parent_labels)),
        "shape_hw": [2, 3],
        "source": sources.descriptor(parent_labels),
    }
    return ma.prepare_adoption(
        tmp_path / "adoption",
        MAP,
        reference_run="one",
        geometry_index=_geometry(tmp_path / "geometry"),
        runs=["one", "two"],
        roles={5: "porous_medium", 6: "porous_medium"},
        names={5: "Fine", 6: "Coarse"},
        facies={5: "Sand1", 6: "Sand2"},
        version={"name": "Map A", "sha256": "a" * 64, "approval": {"sha256": approval}},
        current_labels=lambda run: grid,
    )


def test_a_prepared_painted_map_is_adopted_as_a_new_store_revision(registered_setup, tmp_path):
    data = registered_setup
    prepared = _prepare(data, tmp_path)
    assert [row["status"] for row in prepared["runs"]] == ["prepared", "prepared"]
    assert ma.label_values(prepared) == [5, 6]
    with pytest.raises(ValueError, match="No facies id for label"):
        ma.adoption_candidate(data.revision, prepared["folder"], {5: 1})
    with pytest.raises(ValueError, match="not in the series' facies table"):
        ma.adoption_candidate(data.revision, prepared["folder"], {5: 1, 6: 5})
    candidate, summary = ma.adoption_candidate(data.revision, prepared["folder"], {5: 1, 6: 2})
    assert summary["runs"] == ["one", "two"] and summary["not_prepared"] == []
    assert [(row["label_id"], row["facies_id"], row["porosity"]) for row in candidate["regions"]] == [
        (5, 1, 0.4), (6, 2, 0.3),
    ]
    store = geometry.GeometryRevisionStore(data.project)
    revision = store.save_revision(candidate, actor="synthetic owner", reason="adopt the painted map")
    assert revision["revision_id"] != data.revision["revision_id"]
    assert store.current()["revision_id"] == revision["revision_id"]
    assert revision["configuration"]["provenance"]["map_version"] == {"name": "Map A", "sha256": "a" * 64}
    expected = {"one": MAP.astype(np.int32), "two": np.array([[5, 5, 5], [5, 5, 6]], np.int32)}
    for run in ("one", "two"):
        resolved = sources.registered_run_sources(data.project.root, revision, run)
        labels = geometry._load_array(Path(resolved["files"]["labels"]["path"]))
        assert labels.dtype == np.int32 and np.array_equal(labels, expected[run])
        assert resolved["registration"]["kind"] == sources.PAINTED
        assert "replayed painted placement" in resolved["label_serialization"]
    # The existing per-run export and apply take the adopted labels.
    reference = export.export_corrected_setup(
        data.project, reference=SimpleNamespace(run_id="one"), actor="synthetic owner"
    )
    child = export.export_corrected_setup(
        data.project, reference=SimpleNamespace(run_id="two"), actor="synthetic owner"
    )
    for output, run in ((reference, "one"), (child, "two")):
        labels = geometry._load_array(output.destination / "native_labels.npz")
        assert np.array_equal(labels, expected[run])
        export.apply_corrected_selected_run(data.project, output)
        assert (data.series / f"config/merged/{run}.toml").read_bytes() != data.before[("merged", run)]


def test_the_store_refuses_painted_labels_that_are_not_their_replay(registered_setup, tmp_path):
    data = registered_setup
    prepared = _prepare(data, tmp_path)
    candidate, _summary = ma.adoption_candidate(data.revision, prepared["folder"], {5: 1, 6: 2})
    row = candidate["run_geometry"]["runs"][1]
    path = Path(row["labels_path"])
    with np.load(path, allow_pickle=True) as saved:
        metadata = saved["metadata"].item()
    np.savez_compressed(path, array=np.full((2, 3), 6, np.int32), metadata=metadata)
    row["registration"]["labels"] = sources.descriptor(path)
    report = geometry.validate_geometry_config(candidate)
    messages = " ".join(issue["message"] for issue in report["issues"])
    assert "exact replay of their placement" in messages
    with pytest.raises(ValueError, match="exact replay of their placement"):
        geometry.GeometryRevisionStore(data.project).save_revision(
            candidate, actor="synthetic owner", reason="tampered"
        )
    assert geometry.GeometryRevisionStore(data.project).current()["revision_id"] == data.revision["revision_id"]


def test_the_store_refuses_a_map_its_approval_does_not_name(registered_setup, tmp_path):
    data = registered_setup
    prepared = _prepare(data, tmp_path, approval="b" * 64)
    candidate, _summary = ma.adoption_candidate(data.revision, prepared["folder"], {5: 1, 6: 2})
    report = geometry.validate_geometry_config(candidate)
    messages = " ".join(issue["message"] for issue in report["issues"])
    assert "map and approval its preparation names" in messages


def test_painted_map_composition_carries_reference_camera_to_target_native():
    reference = {"H_metric_mm_to_camera_u": IDENTITY}
    target = {
        "H_metric_mm_to_camera_u": SHIFTED,
        "baseline_native_grid": {
            "shape_hw": [2, 3],
            "H_baseline_camera_u_to_rig_corrected_native": [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]],
        },
    }
    reference_matrix, target_matrix, shape = sources.painted_map_composition(reference, target)
    assert shape == (2, 3)
    assert np.allclose(target_matrix @ np.linalg.inv(reference_matrix), [[1, 0, 1], [0, 1, -1], [0, 0, 1]])
    with pytest.raises(ValueError, match="native shape"):
        sources.painted_map_composition(reference, {**target, "baseline_native_grid": {
            "H_baseline_camera_u_to_rig_corrected_native": IDENTITY}})
