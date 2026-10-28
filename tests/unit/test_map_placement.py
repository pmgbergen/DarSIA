"""Placing an approved painted map in every run through each run's measured
geometry, for review.

A synthetic geometry index of the darsia-measured-run-geometry-v1 form: each
run maps metric rig millimetres to its camera grid by a homography. The map
goes to a run as inverse(H_target_grid_to_mm) @ H_reference_grid_to_mm.
Temporary folders and synthetic arrays only.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from darsia.fluidflower import map_placement as mp

SHAPE = (40, 60)
# Metric millimetres to grid pixels: 2 px per mm, and the second run's
# camera sees the rig 3 px further right.
REFERENCE_H = [[2.0, 0.0, 10.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]]
SHIFTED_H = [[2.0, 0.0, 13.0], [0.0, 2.0, 4.0], [0.0, 0.0, 1.0]]


def _index(folder, runs):
    """An index in the builder's form: each model bound by path, SHA-256 and size."""

    folder.mkdir(parents=True, exist_ok=True)
    entries = {}
    for run, matrix in runs.items():
        (folder / run).mkdir(exist_ok=True)
        model = {
            "format": mp.GEOMETRY_FORMAT,
            "run": run,
            "run_id": run,
            "reference_image": {"path": f"{run}.arw", "sha256": "0" * 64, "size_bytes": 1},
            "grid": "camera_u",
            "shape_hw": list(SHAPE),
        }
        if run.endswith("inv"):
            model["H_camera_u_to_metric_mm"] = np.linalg.inv(matrix).tolist()
        else:
            model["H_metric_mm_to_camera_u"] = matrix
        path = folder / run / "reference_model.json"
        path.write_text(json.dumps(model), encoding="utf-8")
        entries[run] = {
            "run": run,
            "reference_model": {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "size_bytes": path.stat().st_size,
            },
        }
    path = folder / "index.json"
    path.write_text(json.dumps({"format": mp.GEOMETRY_FORMAT, "runs": entries}), encoding="utf-8")
    return path


def _layers():
    layers = np.full(SHAPE, 2, dtype=np.uint8)
    layers[:, 30:] = 3
    layers[20:, :] = 1  # Free water below.
    return layers


def test_the_geometry_index_is_read_with_either_matrix(tmp_path):
    path = _index(tmp_path / "geometry", {"run1": REFERENCE_H, "run2inv": SHIFTED_H})
    geometry = mp.read_run_geometry(path)
    assert sorted(geometry) == ["run1", "run2inv"]
    assert np.allclose(geometry["run1"].metric_to_grid, REFERENCE_H)
    assert np.allclose(geometry["run2inv"].metric_to_grid, SHIFTED_H)
    assert geometry["run1"].shape_hw == SHAPE
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "other", "runs": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="is not a darsia-measured-run-geometry-v1 index"):
        mp.read_run_geometry(bad)
    # A model that changed after the index bound it is refused.
    model = tmp_path / "geometry" / "run1" / "reference_model.json"
    model.write_text(model.read_text(encoding="utf-8").replace("run1", "run9"), encoding="utf-8")
    with pytest.raises(ValueError, match="is not the reference model the index bound"):
        mp.read_run_geometry(path)
    # A model given by a relative path is taken from the index's folder.
    model.write_text(json.dumps({"format": mp.GEOMETRY_FORMAT}), encoding="utf-8")
    path.write_text(
        json.dumps(
            {"format": mp.GEOMETRY_FORMAT, "runs": {"run1": {"reference_model": "run1/reference_model.json"}}}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no metric-to-camera homography"):
        mp.read_run_geometry(path)


def test_a_map_is_carried_through_the_rig_to_another_run(tmp_path):
    geometry = mp.read_run_geometry(
        _index(tmp_path / "geometry", {"run1": REFERENCE_H, "run2": SHIFTED_H})
    )
    layers = _layers()
    same, none = mp.place_layer_map(layers, geometry["run1"], geometry["run1"], SHAPE)
    assert np.array_equal(same, layers) and not none.any()
    placed, uncovered = mp.place_layer_map(layers, geometry["run1"], geometry["run2"], SHAPE)
    # Everything moves 3 px right; the 3 columns the map does not reach are flagged.
    assert np.array_equal(placed[:, 3:], layers[:, :-3])
    assert uncovered[:, :3].all() and not uncovered[:, 3:].any()
    assert mp.map_offset(layers, geometry["run1"], geometry["run2"]) == {
        "median_px": 3.0, "max_px": 3.0
    }
    assert mp.map_offset(layers, geometry["run1"], geometry["run1"]) == {
        "median_px": 0.0, "max_px": 0.0
    }


def test_the_overlay_has_two_labelled_line_types():
    photo = np.full(SHAPE + (3,), 100, dtype=np.uint8)
    layers = _layers()
    uncovered = np.zeros(SHAPE, dtype=bool)
    uncovered[:, :3] = True
    overlay = mp.boundary_overlay(photo, layers, uncovered)
    boundary, gap = mp.BOUNDARY_COLOR, mp.GAP_COLOR
    assert tuple(overlay[19, 10]) == boundary and tuple(overlay[20, 10]) == boundary
    assert tuple(overlay[5, 29]) == boundary and tuple(overlay[5, 30]) == boundary
    assert tuple(overlay[5, 2]) == gap and tuple(overlay[5, 3]) == gap
    assert tuple(overlay[5, 10]) == (100, 100, 100)  # Away from the lines.
    assert [color for _label, color in mp.OVERLAY_LEGEND] == [boundary, gap]
    assert np.array_equal(photo, np.full(SHAPE + (3,), 100, dtype=np.uint8))
    with pytest.raises(ValueError, match="differ in size"):
        mp.boundary_overlay(photo[:10], layers)


def test_layer_offsets_use_the_existing_band_measurement():
    shape = (120, 200)
    layers = np.full(shape, 2, dtype=np.uint8)
    layers[:, 100:] = 3
    photo = np.full(shape + (3,), 40, dtype=np.uint8)
    photo[:, 106:] = 200  # The visible edge lies 6 px (3 mm) right of the boundary.
    geometry = mp.RunGeometry("run1", np.asarray(REFERENCE_H), {}, shape, "")
    assert mp.grid_spacing_mm(geometry, shape) == pytest.approx((0.5, 0.5))
    roles = {2: "porous_medium", 3: "porous_medium"}
    result = mp.layer_offsets(photo, layers, geometry, roles)
    assert result["band_mm"] == 25.0 and result["spacing_mm"] == [0.5, 0.5]
    (boundary,) = result["boundaries"]
    assert boundary["labels"] == [2, 3] and boundary["status"] == "candidate"
    assert boundary["median_mm"] == 3.0 and boundary["p10_mm"] == boundary["p90_mm"] == 3.0
    assert boundary["measured_sample_count"] == boundary["sample_count"] == 120
    # The change from a reference run's own offset for the same pair of layers.
    shifted = {"boundaries": [dict(boundary, median_mm=5.5), {"labels": [2, 9], "median_mm": 1.0}]}
    mp.add_changes_from_reference(shifted, result)
    assert [row["change_from_reference_mm"] for row in shifted["boundaries"]] == [2.5, None]
    mp.add_changes_from_reference(result, result)
    assert result["boundaries"][0]["change_from_reference_mm"] == 0.0
    # On a reduced grid the same edge is found, to within the coarser spacing.
    coarse = mp.layer_offsets(photo, layers, geometry, roles, spacing_mm=1.0)
    assert coarse["spacing_mm"] == [1.0, 1.0]
    assert abs(coarse["boundaries"][0]["median_mm"] - 3.0) <= 0.5


def test_the_checks_and_the_review_order():
    layers = _layers()
    layers[0, 0] = 9
    layers[10, 10] = 1  # Free water enclosed in the sand.
    uncovered = np.zeros(SHAPE, dtype=bool)
    uncovered[:, :2] = True
    checks = mp.run_checks(layers, uncovered, {1: "free_water"}, {1, 2, 3})
    assert checks["free_water_pixels"] == 20 * 60 + 1 and checks["inside_pixels"] == 40 * 60
    assert checks["water_pocket_pixels"] == 1
    assert checks["uncovered_pixels"] == 2 * 40 and checks["gap_pixels"] == 2 * 40
    layers[:, :2] = 0  # Pixels outside the map are not gaps.
    assert mp.run_checks(layers, uncovered, {1: "free_water"}, {1, 2, 3})["gap_pixels"] == 0
    assert checks["unknown_layer_ids"] == [9] and checks["outside_pixels"] == 0
    assert mp.review_order(["run1", "run2", "run3", "run4"], "RUN3", ["run4", "nope"]) == [
        "run3", "run4", "run1", "run2"
    ]


def test_a_placement_writes_a_new_review_folder_only(tmp_path):
    index = _index(tmp_path / "geometry", {"run1": REFERENCE_H, "run2": SHIFTED_H})
    seen = []
    progress = []

    def photo(run, geometry):
        seen.append(run)
        return np.full(geometry.shape_hw + (3,), 50, dtype=np.uint8)

    result = mp.place_map_in_runs(
        tmp_path / "reviews",
        _layers(),
        reference_run="RUN2",
        geometry_index=index,
        runs=["run1", "run2", "run3"],
        photo_loader=photo,
        roles={1: "free_water", 2: "porous_medium", 3: "porous_medium"},
        names={1: "Free water", 2: "Fine sand", 3: "Coarse sand"},
        version={"name": "Map A", "sha256": "a" * 64},
        run_progress=progress.append,
    )
    folder = Path(result["folder"])
    assert folder.parent == tmp_path / "reviews"
    assert [row["run_id"] for row in result["runs"]] == ["run2", "run1", "run3"]
    assert [row["status"] for row in result["runs"]] == ["placed", "placed", "no geometry"]
    assert seen == ["run2", "run1"]
    assert progress == [dict(phase="transfer", completed=completed, total=3, run_id=run)
        for n, run in enumerate(("run2", "run1", "run3")) for completed in (n, n + 1)]
    assert mp.read_placement(folder)["map_version"]["name"] == "Map A"
    with np.load(folder / "run1" / "layers.npz") as saved:
        assert saved["layers"].shape == SHAPE and saved["uncovered"][:, -3:].all()
    overlay = np.asarray(Image.open(folder / "run1" / "overlay.jpg"))
    assert overlay.shape == SHAPE + (3,)
    assert result["runs"][1]["checks"]["uncovered_pixels"] == 3 * 40
    assert result["runs"][0]["offset_px"]["max_px"] == 0.0
    assert result["runs"][1]["offset_px"] == {"median_px": 3.0, "max_px": 3.0}
    assert result["layers"]["3"] == {"name": "Coarse sand", "role": "porous_medium", "facies": ""}
    assert [row["label"] for row in result["overlay_legend"]] == [
        label for label, _color in mp.OVERLAY_LEGEND
    ]
    # A uniform photograph shows no edge to measure an offset against.
    offsets = result["runs"][1]["layer_offsets"]
    assert offsets["band_mm"] == 25.0 and offsets["boundaries"]
    assert all(row["median_mm"] is None for row in offsets["boundaries"])
    assert sorted(path.name for path in folder.iterdir()) == ["index.json", "run1", "run2"]
    with pytest.raises(ValueError, match="no model for the reference run"):
        mp.place_map_in_runs(
            tmp_path / "reviews", _layers(), reference_run="run9", geometry_index=index,
            runs=["run1"], photo_loader=photo, roles={}, version={},
        )
    with pytest.raises(ValueError, match="is not on the reference run's grid"):
        mp.place_map_in_runs(
            tmp_path / "reviews", _layers()[:20], reference_run="run1", geometry_index=index,
            runs=["run1"], photo_loader=photo, roles={}, version={},
        )
    assert len(list((tmp_path / "reviews").iterdir())) == 1
