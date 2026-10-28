"""Synthetic source-admission seams, without approval or adoption authority.

The producer/NPZ fixture is real and temporary; RAW decoding, registration and
runtime identities retain the explicit limitations of the reader unit fixture.
The tiny painting frame context here is not an approved saved-version capture
or Store qualification. These tests exercise only the new compositional reader
and its transfer/preparation boundaries.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import map_adoption, map_placement, round5_geometry
from darsia.fluidflower import painted_adoption_geometry as admission
from darsia.fluidflower import raw_shape_basis_reader as reader

from .test_raw_shape_basis_reader import basis, _descriptor, _json, _read


def _model(f, tmp_path):
    """Bind a genuine fixture basis to an explicit synthetic saved metric frame."""
    frame = dict(H_camera_to_mm=[[1., 0., 0.], [0., -1., 2.], [0., 0., 1.]],
                 sand_area_factor14=1.0, explicit_certificate={"synthetic": True})
    saved = _json(tmp_path / "saved-metric-model.json",
                  dict(run="sample", conditional_frame=frame))
    edges = _json(tmp_path / "saved-edges.json", {"synthetic": True})
    profile = dict(_descriptor(f.profile), profile_id="camera", camera_wh=[3, 2])
    model = round5_geometry.reference_model("sample", frame,
        reference_image=_descriptor(f.source), camera_profile=profile,
        source_model=saved, edge_review=edges,
        baseline_configuration=_descriptor(f.configuration),
        baseline_native_renderer=f.certificate, raw_shape_basis_authority=f.authority,
        raw_shape_basis_workspace=f.parent_workspace)
    bound = _json(tmp_path / "reference-model.json", model)
    context = dict(kind="camera_u", profile_id="camera", parameters=_descriptor(f.profile))
    binding = dict(record={"run_id": "sample"}, frame={"image_size": [3, 2]},
        frame_basis=dict(workspace_revision=f.parent_workspace,
                         photograph=_descriptor(f.source), camera_profile=context))
    return model, bound, binding


def test_new_model_reconstructs_basis_and_uses_original_transport(basis, tmp_path):
    model, bound, binding = _model(basis, tmp_path)
    checked = reader.read_model_basis(model)
    assert checked.parent_labels["source"] == {k: basis.fields["labels"][k]
        for k in ("path", "sha256", "size_bytes")}
    proof = admission.make_proof(binding, {"sample": bound})
    assert proof["raw_shape_basis"]["authorities"]["sample"]["root_release"] == basis.authority["root_release"]
    assert proof["raw_shape_basis"]["authorities"]["sample"]["model"] == bound
    geometry = map_placement.RunGeometry(run_id="sample",
        metric_to_grid=np.asarray(model["H_metric_mm_to_camera_u"]),
        reference_image=model["reference_image"], shape_hw=(2, 3),
        source=bound["path"], model=model, entry={"reference_model": bound})
    source = np.array([[1, 1, 2], [1, 2, 2]], np.uint8)
    labels, filled, support, matrix = map_adoption.native_labels(source, geometry, geometry)
    np.testing.assert_array_equal(labels, source)
    np.testing.assert_array_equal(np.asarray(matrix), np.eye(3))
    assert not filled.any() and support.all()


def test_independently_valid_other_workspace_cannot_replace_painting_parent(basis, tmp_path):
    _, bound, binding = _model(basis, tmp_path)
    other = deepcopy(basis.parent)
    other["revision_id"] = "another-same-grid-workspace"
    binding["frame_basis"]["workspace_revision"] = _json(tmp_path / "other-parent.json", other)
    with pytest.raises(ValueError, match="different workspace"):
        admission.make_proof(binding, {"sample": bound})


def test_preparation_metadata_copies_replay_exact_original_authorities(basis, tmp_path):
    model, bound, binding = _model(basis, tmp_path)
    proof = admission.make_proof(binding, {"sample": bound})
    folder = tmp_path / "preparation"
    folder.mkdir()
    admission.write_configuration_copies(folder, proof)
    admission.verify_proof(binding, proof, {"sample": bound}, folder)
    expected = admission.basis_label_fields(binding, proof, "sample", folder)
    assert expected["source"]["path"] == basis.fields["labels"]["path"]
    for path in (basis.configuration, basis.review_path, basis.release_path,
                 basis.profile, basis.parent_path):
        path.write_bytes(path.read_bytes() + b"\nchanged original fixture metadata\n")
    admission.verify_proof(binding, proof, {"sample": bound}, folder)
    with pytest.raises(ValueError):
        reader.read_model_basis(model)
    row = next(row for row in proof["raw_shape_basis"]["source_copies"]
               if row["original"]["path"] == str(basis.configuration))
    copied = folder / row["copy"]["path"]
    copied.write_bytes(copied.read_bytes() + b"changed copy")
    with pytest.raises(ValueError, match="size|changed"):
        admission.verify_proof(binding, proof, {"sample": bound}, folder)


def test_missing_basis_refuses_before_version_capture_or_any_destination(basis, tmp_path, monkeypatch):
    model, _, _ = _model(basis, tmp_path)
    model["baseline_native_grid"].pop("raw_shape_basis_authority")
    bound = _json(tmp_path / "old-model.json", model)
    index = _json(tmp_path / "legacy-index.json", dict(format=map_placement.GEOMETRY_FORMAT,
                  runs={"sample": {"reference_model": bound}}))
    from darsia.fluidflower import painted_adoption_table
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing basis must refuse before version/photo/model hashing")
    monkeypatch.setattr(painted_adoption_table, "capture_version", forbidden)
    destination = tmp_path / "uncreated-preparation"
    with pytest.raises(ValueError, match="The rigs need their RAW basis"):
        map_adoption.prepare_adoption(destination, np.ones((2, 3), np.uint8),
            reference_run="sample", geometry_index=index["path"], runs=["sample"],
            roles={1: "porous_medium"}, version={}, painting_version={})
    assert not destination.exists()


def test_basis_transfer_closes_model_before_photograph_or_output(basis, tmp_path):
    model, _, _ = _model(basis, tmp_path)
    model["baseline_native_grid"]["H_baseline_camera_u_to_rig_corrected_native"][0][2] = .25
    bound = _json(tmp_path / "wrong-model.json", model)
    index = _json(tmp_path / "wrong-index.json", dict(format=map_placement.GEOMETRY_FORMAT,
                  runs={"sample": {"reference_model": bound}}))
    def forbidden(*args, **kwargs):
        raise AssertionError("A changed model must refuse before a photograph or writer")
    destination = tmp_path / "uncreated-transfer"
    with pytest.raises(ValueError, match="changed its RAW basis"):
        map_placement.place_map_in_runs(destination, np.ones((2, 3), np.uint8),
            reference_run="sample", geometry_index=index["path"], runs=["sample"],
            photo_loader=forbidden, roles={1: "porous_medium"}, version={})
    assert not destination.exists()


def test_new_output_boundary_refuses_original_owned_or_native_folders(basis, tmp_path):
    model, _, _ = _model(basis, tmp_path)
    for root in (basis.context.destination, basis.source.parent, basis.parent_path.parent):
        with pytest.raises(ValueError, match="outside their original"):
            reader.guard_output_root(root / "new-output", [model])
    reader.guard_output_root(tmp_path / "independent-output", [model])


def test_transfer_rejects_changed_saved_metric_algebra_even_with_genuine_basis(basis, tmp_path):
    model, _, _ = _model(basis, tmp_path)
    model["H_metric_mm_to_camera_u"][0][2] += .25
    with pytest.raises(ValueError, match="saved frame or baseline bridge changed"):
        reader.read_model_basis(model)


def test_transfer_retains_incomplete_outputs_if_consumed_configuration_changes(basis, tmp_path):
    _, bound, _ = _model(basis, tmp_path)
    index = _json(tmp_path / "index.json", dict(format=map_placement.GEOMETRY_FORMAT,
                  runs={"sample": {"reference_model": bound}}))
    def photograph(*args):
        basis.configuration.write_bytes(basis.configuration.read_bytes() + b"late drift\n")
        return np.zeros((2, 3, 3), np.uint8)
    destination = tmp_path / "transfer"
    with pytest.raises(ValueError, match="changed|size"):
        map_placement.place_map_in_runs(destination, np.ones((2, 3), np.uint8),
            reference_run="sample", geometry_index=index["path"], runs=["sample"],
            photo_loader=photograph, roles={1: "porous_medium"}, version={}, measure_offsets=False)
    attempts = list(destination.iterdir())
    assert len(attempts) == 1
    assert (attempts[0] / "sample" / "layers.npz").is_file()
    assert not (attempts[0] / "index.json").exists()
