"""Synthetic adapter boundaries; no real series, palette fit, NAS or rig job."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

from darsia.fluidflower import geometry_import as subject
from darsia.fluidflower import layer_painting as painting
from darsia.fluidflower.series_geometry import SeriesRunInputs


def test_mapping_preserves_label_and_facies_ids_including_duplicate_names():
    config = {"facies": {"props": "facies.csv", "0": {"labels": [1]},
                          "7": {"labels": [0]}, "8": {"labels": [13]}}}
    props = {0: {"name": "water", "porosity": None, "permeability": None},
             7: {"name": "F", "porosity": .44, "permeability": 4295},
             8: {"name": "F", "porosity": .44, "permeability": 4295}}
    rows = subject._region_rows(config, props, [0, 1, 13],
                                {0: "free_water", 7: "porous_medium", 8: "porous_medium"})
    assert [(r["label_id"], r["facies_id"]) for r in rows] == [(0, 7), (1, 0), (13, 8)]
    assert rows[1]["porosity"] is None
    assert rows[1]["permeability"] is None


def test_original_facies_path_is_metadata_not_a_numeric_mapping():
    config = {"facies": {"props": "facies.csv", "path": "setup/facies/facies.npz",
                          "8": {"labels": [0]}}}
    rows = subject._region_rows(config, {8: {"name": "F", "porosity": .44, "permeability": 4295}},
                                [0], {8: "porous_medium"})
    assert [(row["label_id"], row["facies_id"]) for row in rows] == [(0, 8)]


@pytest.mark.parametrize("role", ["free_water", "excluded", "porous_medium"])
@pytest.mark.parametrize("values", [(None, None), (.37, 12.5)])
def test_roles_never_replace_recorded_or_missing_physical_properties(role, values):
    porosity, permeability = values
    rows = subject._region_rows({"facies": {"0": {"labels": [1]}}},
                                {0: {"name": "source row", "porosity": porosity,
                                     "permeability": permeability}}, [1], {0: role})
    assert rows[0]["porosity"] == porosity
    assert rows[0]["permeability"] == permeability


@pytest.mark.parametrize("config,roles,match", [
    ({"facies": {"1": {"labels": [2]}, "2": {"labels": [2]}}}, {1: "porous_medium"}, "more than one"),
    ({"facies": {"1": {"labels": [2]}}}, {1: "porous_medium"}, "no configured facies"),
    ({"facies": {"1": {"labels": [3]}}}, {}, "explicit existing analysis role"),
])
def test_ambiguous_or_missing_inputs_are_not_guessed(config, roles, match):
    with pytest.raises(ValueError, match=match):
        subject._region_rows(config, {1: {"name": "sand", "porosity": .4, "permeability": 10}}, [3], roles)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "series"
    root.mkdir()
    (root / "series.toml").write_text('schema_version=1\nseries_id="TEST"\n[runs]\nids=["display"]\n')
    original = root / "original-map.png"
    source_photo = root / "historical-source.JPG"
    display_photo = root / "display-baseline.JPG"
    depth = root / "existing-same-grid-depth.npz"
    table = root / "facies.csv"
    protocol = root / "injection.csv"
    for path in (original, source_photo, display_photo):
        path.write_bytes(path.name.encode())
    np.savez(depth, array=np.ones((2, 2)))
    table.write_bytes(b'id,name,porosity,permeability,swi CO2\n1,F,.4,10,.12\n2,F,.5,20,.23\n')
    protocol.write_text('location_x,location_y\n0.2,0.1\n0.2,0.1\n')
    crop = dict(top_left=[0, 0], bottom_left=[1, 0], bottom_right=[1, 1], top_right=[0, 1], width=.9, height=.49, in_meters=True)
    configuration = root / "display.toml"
    config = {"rig": {"width": .9, "height": .49, "dim": 2},
              "labeling": {"colored_image": str(original)},
              "facies": {"props": str(table), "1": {"labels": [0, 3]}, "2": {"labels": [4]}},
              "corrections": {"curvature": {"crop": crop}}}
    configuration.write_text(toml.dumps(config))
    project = SimpleNamespace(root=root / "geometry_facies", manifest={})
    project.root.mkdir()
    calls = []
    monkeypatch.setattr(subject, "native_series_link", lambda _project: {"series_root": str(root)})
    class Store:
        def __init__(self, project):
            self.current_path = project.root / "overrides/geometry_facies/current.json"
            self.artifact_root = project.root / "results/geometry"
        def revisions(self):
            return []
        def save_revision(self, candidate, *, actor, reason):
            calls.append((deepcopy(candidate), actor, reason))
            target = self.artifact_root / ("a" * 32)
            target.mkdir(parents=True)
            revision = {"format": "darsia-fluidflower-geometry-facies-v1", "revision_id": "a" * 32,
                        "configuration": deepcopy(candidate)}
            (target / "revision.json").write_text(json.dumps(revision))
            return revision
    monkeypatch.setattr(subject, "GeometryRevisionStore", Store)
    def palette(source, reference, cfg, *, input_grid, work_root, frame):
        assert source == original and reference.run_id == "historical"
        assert reference.baseline_image == source_photo
        assert reference.injection_protocol is None
        assert input_grid == "raw"
        target = Path(work_root) / "original-palette"
        target.mkdir()
        labels = target / "labels.npz"
        np.savez(labels, array=np.array([[0, 3], [4, 4]], dtype=np.uint8))
        receipt = target / "import.json"
        receipt.write_text('{"extra_trees_fit_run":false}')
        session = painting.PaintingSession(run_id="historical", image_path=str(source_photo), image_size=(2, 2))
        session.layers = [painting.PaintedLayer(i + 1, f"Imported layer {native}", "#ff0000", "old") for i, native in enumerate((0, 3, 4))]
        return SimpleNamespace(native_labels=labels, receipt=receipt, session=session)
    monkeypatch.setattr(subject, "import_coloured_map", palette)
    state = {}
    monkeypatch.setattr(subject, "set_reference_run", lambda _p, run, **_kw: state.update(display=run))
    monkeypatch.setattr(subject, "save_rig_frame", lambda _p, frame, **_kw: state.update(frame=frame))
    monkeypatch.setattr(subject, "save_injection_points", lambda _p, points, **_kw: state.update(points=points))
    reference = SeriesRunInputs(run_id="display", config_path=configuration, baseline_image=display_photo,
                                injection_protocol=protocol, crop=crop, width_m=.9, height_m=.49)
    args = dict(series_root=root, source_map=original, source_image=source_photo,
                source_run="historical", display_reference=reference, configuration=configuration,
                facies_table=table, workspace_depth_path=depth, facies_roles={1: "porous_medium", 2: "porous_medium"},
                permeability_unit="darcy", actor="test importer")
    return project, args, calls, state


def test_authoritative_raster_store_uuid_and_historical_photo_are_not_rewritten(prepared):
    project, args, calls, state = prepared
    before = {key: Path(args[key]).read_bytes() for key in ("source_map", "source_image", "configuration", "facies_table", "workspace_depth_path")}
    result = subject.adopt_pipeline_map(project, **args)
    candidate, actor, reason = calls[0]
    assert result.revision_id == "a" * 32 and result.revision_name == "adopted-v1"
    assert candidate["provenance"]["revision_name"] == "adopted-v1"
    assert candidate["baseline_path"] == str(args["source_image"])
    assert candidate["display_image_path"] == str(args["display_reference"].baseline_image)
    assert candidate["provenance"]["reference_run"] == "historical"
    assert candidate["provenance"]["display_reference_run"] == "display"
    assert candidate["provenance"]["source_run_registered"] is False
    assert [(r["label_id"], r["facies_id"]) for r in candidate["regions"]] == [(0, 1), (3, 1), (4, 2)]
    with np.load(candidate["labels_path"]) as archive:
        assert np.array_equal(archive["array"], [[0, 3], [4, 4]])
    assert candidate["depth_path"] == str(args["workspace_depth_path"])
    assert "facies_generation" not in candidate and "depth_note" not in candidate
    assert Path(candidate["facies_properties_source_path"]).read_bytes() == before["facies_table"]
    assert b"swi CO2" in Path(candidate["facies_properties_source_path"]).read_bytes()
    assert state["display"] == "display" and state["frame"].run_id == "display"
    assert len(state["points"]) == 1
    session = json.loads(result.editor_session.read_bytes())
    assert session["run_id"] == "historical" and session["image"]["path"] == str(args["source_image"])
    assert len(session["facies"]) == 2  # Duplicate F names stay separate by numeric facies ID.
    receipt = json.loads(result.receipt.read_bytes())
    assert receipt["workspace_depth_generated"] is False
    assert receipt["classifier_fit_run"] is False and receipt["editor_proposal_saved"] is False
    assert receipt["lossless_export_gate"].startswith("NOT RUN")
    assert all(Path(args[key]).read_bytes() == value for key, value in before.items())


def test_missing_real_depth_stops_before_palette_or_revision(prepared, monkeypatch):
    project, args, calls, state = prepared
    args["workspace_depth_path"].unlink()
    monkeypatch.setattr(subject, "import_coloured_map", lambda *_a, **_kw: pytest.fail("palette workflow must not run"))
    with pytest.raises(FileNotFoundError):
        subject.adopt_pipeline_map(project, **args)
    assert not calls and not state


def test_original_store_shape_failure_is_visible_without_placeholder_or_success(prepared, monkeypatch):
    project, args, calls, state = prepared
    class RefusingStore(subject.GeometryRevisionStore):
        def save_revision(self, *_a, **_kw):
            raise ValueError("Baseline, depth, labels, and facies must share the same spatial shape")
    monkeypatch.setattr(subject, "GeometryRevisionStore", RefusingStore)
    before = args["workspace_depth_path"].read_bytes()
    with pytest.raises(ValueError, match="same spatial shape"):
        subject.adopt_pipeline_map(project, **args)
    assert args["workspace_depth_path"].read_bytes() == before
    assert not calls and not state
    assert not list(project.root.rglob("adoption.json"))


def test_preexisting_editor_draft_is_never_overwritten(prepared, monkeypatch):
    project, args, calls, state = prepared
    draft = painting.session_path(project, "historical")
    draft.parent.mkdir(parents=True)
    draft.write_bytes(b"existing human draft")
    monkeypatch.setattr(subject, "import_coloured_map", lambda *_a, **_kw: pytest.fail("palette workflow must not run"))
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        subject.adopt_pipeline_map(project, **args)
    assert draft.read_bytes() == b"existing human draft"
    assert not calls and not state


def test_existing_source_unit_reaches_store_with_original_values_and_no_conversion(prepared):
    project, args, calls, state = prepared
    args["permeability_unit"] = "source_unit"
    result = subject.adopt_pipeline_map(project, **args)
    candidate = calls[0][0]
    assert candidate["permeability_unit"] == "source_unit"
    assert [(r["label_id"], r["facies_id"], r["porosity"], r["permeability"])
            for r in candidate["regions"]] == [(0, 1, .4, 10), (3, 1, .4, 10), (4, 2, .5, 20)]
    assert candidate["provenance"]["explicit_adoption_request"]["permeability_unit"] == "source_unit"
    assert json.loads(result.editor_session.read_bytes())["permeability_unit"] == "source_unit"


def test_unknown_source_unit_is_still_refused_before_raw_import(prepared):
    project, args, calls, state = prepared
    args["permeability_unit"] = "unknown_unit"
    with pytest.raises(ValueError, match="explicit Darcy"):
        subject.adopt_pipeline_map(project, **args)
    assert not calls and not state
