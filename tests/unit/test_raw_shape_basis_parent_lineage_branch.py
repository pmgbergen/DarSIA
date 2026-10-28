"""Exercise the optional join while retaining original parent-grid refusals."""
from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

import darsia
from darsia.fluidflower import geometry_workspace_frame as workspace
from darsia.fluidflower import raw_shape_basis_reader as reader
from darsia.fluidflower import raw_shape_configuration_lineage as lineage


@pytest.fixture
def parent_case(tmp_path, monkeypatch):
    """Keep real spatial metadata while isolating archive transport."""
    def descriptor(name):
        """Bind one generic synthetic source identity."""
        return dict(path=str(tmp_path / name), sha256=hashlib.sha256(name.encode()).hexdigest(), size_bytes=1)

    original_configuration = descriptor("original.toml")
    actual_configuration = descriptor("actual.toml")
    workspace_binding = descriptor("revision.json")
    revision = dict(configuration=dict(frame_binding=dict(runs={
        "sample": dict(configuration=original_configuration)})))
    rgb = darsia.OpticalImage(np.zeros((2, 3, 3), dtype=np.float32),
                             color_space="RGB", dimensions=[1.0, 1.5])
    scalar = darsia.Image(np.zeros((2, 3), dtype=np.uint8), scalar=True, dimensions=[1.0, 1.5])
    images = {"shape_corrected_baseline": rgb,
              "labels": scalar, "facies": scalar, "depth": scalar}
    fields = {name: dict(**descriptor(name + ".npz"), shape=list(image.img.shape))
              for name, image in images.items()}
    spatial = reader._spatial_grid(rgb)
    grid = dict(metadata={key: spatial[key] for key in ("space_dim", "dimensions", "origin")},
                indexing=spatial["indexing"], voxel_size=spatial["voxel_size"],
                num_voxels=spatial["num_voxels"], array_row_direction=spatial["row_direction"],
                array_column_direction=spatial["column_direction"], pixel_centres=spatial["pixel_centres"])
    document = dict(inputs=dict(configuration_files=[actual_configuration]),
                    corrected_baseline=dict(grid=grid))
    reads = []

    class Pins:
        """Record field reads after the configuration decision."""
        def add(self, binding, **kwargs):
            """Return the immutable revision or record a selected field."""
            if binding == workspace_binding:
                return revision
            reads.append(deepcopy(binding))
            return None

    monkeypatch.setattr(workspace, "selected_fields", lambda *a, **kw: fields)
    monkeypatch.setattr(reader, "_npz_header", lambda *a, **kw: None)
    monkeypatch.setattr(darsia, "imread", lambda path: next(
        image for name, image in images.items() if fields[name]["path"] == str(path)))
    return dict(pins=Pins(), document=document, binding=workspace_binding,
                original=original_configuration, actual=actual_configuration,
                images=images, fields=fields, reads=reads)


def invoke(case, proof=None):
    """Call the original parent-grid check with an optional explicit proof."""
    return reader._parent_grid(case["pins"], case["document"], case["binding"], "sample",
                               request={"run_id": "sample"}, parent_configuration_lineage=proof)


def forbid_join(*args, **kwargs):
    """Fail if an unchanged original path reaches the optional verifier."""
    raise AssertionError("The optional lineage verifier must not run")


def test_missing_configuration_still_refuses_without_explicit_lineage(parent_case, monkeypatch):
    """Preserve the original missing-membership refusal."""
    monkeypatch.setattr(lineage, "_require_shape_configuration_lineage", forbid_join)
    with pytest.raises(ValueError, match="original config chain"):
        invoke(parent_case)
    assert parent_case["reads"] == []


def test_exact_original_configuration_keeps_existing_path(parent_case, monkeypatch):
    """Avoid the fallback when the original exact binding is present."""
    parent_case["document"]["inputs"]["configuration_files"] = [parent_case["original"]]
    monkeypatch.setattr(lineage, "_require_shape_configuration_lineage", forbid_join)
    labels, compatibility = invoke(parent_case, {"unused": "explicit proof"})
    assert labels["shape_hw"] == [2, 3]
    assert compatibility["configuration"] == parent_case["original"]
    assert "configuration_lineage" not in compatibility
    assert len(parent_case["reads"]) == 4


def test_duplicate_original_configuration_keeps_original_refusal(parent_case, monkeypatch):
    """Do not use lineage to waive duplicate original bindings."""
    parent_case["document"]["inputs"]["configuration_files"] = [
        parent_case["original"], deepcopy(parent_case["original"])]
    monkeypatch.setattr(lineage, "_require_shape_configuration_lineage", forbid_join)
    with pytest.raises(ValueError, match="original config chain"):
        invoke(parent_case, {"unused": "explicit proof"})
    assert parent_case["reads"] == []


def test_positive_join_preserves_original_identity_and_all_four_grids(parent_case, monkeypatch):
    """Keep historical identity separate and read all original fields."""
    received = {}
    audit = {"historical_configuration": parent_case["original"],
             "consumed_configuration_files": [parent_case["actual"]]}

    def verify(pins, **kwargs):
        """Capture the explicit join inputs without asserting real lineage."""
        received.update(kwargs)
        assert pins is parent_case["pins"]
        return audit

    proof = {"explicit": "bound proof DATA"}
    monkeypatch.setattr(lineage, "_require_shape_configuration_lineage", verify)
    labels, compatibility = invoke(parent_case, proof)
    assert received["parent_configuration"] == parent_case["original"]
    assert received["producer_configuration_files"] == [parent_case["actual"]]
    assert received["proof"] is proof
    assert received["request"] == {"run_id": "sample"}
    assert labels["shape_hw"] == [2, 3]
    assert compatibility["configuration"] == parent_case["original"]
    assert compatibility["configuration_lineage"] == audit
    assert compatibility["old_new_pixel_equality_claimed"] is False
    assert len(parent_case["reads"]) == 4


def test_positive_join_cannot_skip_later_full_grid_check(parent_case, monkeypatch):
    """Retain the original spatial-grid refusal after a positive join."""
    monkeypatch.setattr(lineage, "_require_shape_configuration_lineage", lambda *a, **kw: {"verified": True})
    parent_case["document"]["corrected_baseline"]["grid"]["metadata"]["origin"][0] += 1.0
    with pytest.raises(ValueError, match="full grid differ"):
        invoke(parent_case, {"explicit": "bound proof DATA"})


def test_failed_lineage_cannot_reach_field_reads(parent_case, monkeypatch):
    """Stop before field reads when the verifier rejects its inputs."""
    def refuse(*args, **kwargs):
        """Represent a rejected bound historical source."""
        raise ValueError("Bound historical source differs")

    monkeypatch.setattr(lineage, "_require_shape_configuration_lineage", refuse)
    with pytest.raises(ValueError, match="historical source differs"):
        invoke(parent_case, {"explicit": "wrong bound proof"})
    assert parent_case["reads"] == []


@pytest.mark.parametrize("audit", [None, {}, False, []])
def test_missing_verifier_result_refuses_before_grid_reads(parent_case, monkeypatch, audit):
    """Require a concrete audit from the positive verifier."""
    monkeypatch.setattr(lineage, "_require_shape_configuration_lineage", lambda *a, **kw: audit)
    with pytest.raises(ValueError, match="verified bindings"):
        invoke(parent_case, {"explicit": "bound proof DATA"})
    assert parent_case["reads"] == []


def test_existing_release_without_lineage_keeps_legacy_reader_input(tmp_path):
    """Keep existing model and preflight callers idle for older releases."""
    pins = reader._Pins()
    release_path = str(tmp_path / "release.json")
    authority = {"root_release": {"path": release_path}}
    pins.documents[reader._key(release_path)] = {}
    assert reader._released_parent_lineage(pins, authority, None) is None
    explicit = {"diagnostic": "explicit retained-input proof"}
    assert reader._released_parent_lineage(pins, authority, explicit) is explicit


def test_new_release_forwards_pinned_lineage_through_existing_reader_callers(tmp_path):
    """Read the proof from the already validated release, with final rehash."""
    proof = {"existing": "five-key proof is validated by the positive verifier"}
    path = tmp_path / "proof.json"
    raw = json.dumps(proof).encode()
    path.write_bytes(raw)
    binding = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    release_path = str(tmp_path / "release.json")
    authority = {"root_release": {"path": release_path}}
    pins = reader._Pins()
    pins.documents[reader._key(release_path)] = {"parent_configuration_lineage": binding}
    assert reader._released_parent_lineage(pins, authority, None) == proof
    assert reader._released_parent_lineage(pins, authority, deepcopy(proof)) == proof
    pins.final()
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="source size changed"):
        pins.final()


def test_explicit_proof_cannot_override_new_release_binding(tmp_path):
    """Refuse a caller attempting to replace the pre-build proof identity."""
    path = tmp_path / "proof.json"
    raw = b'{"bound":"proof"}'
    path.write_bytes(raw)
    binding = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
    release_path = str(tmp_path / "release.json")
    authority = {"root_release": {"path": release_path}}
    pins = reader._Pins()
    pins.documents[reader._key(release_path)] = {"parent_configuration_lineage": binding}
    with pytest.raises(ValueError, match="differs from the bound release"):
        reader._released_parent_lineage(pins, authority, {"unbound": "override"})


@pytest.fixture
def real_parent_grid(tmp_path):
    """Save all four original image archives with genuine historical metadata."""
    def descriptor(path):
        """Bind the actual bytes written by the original image serializer."""
        raw = path.read_bytes()
        return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))

    dimensions = [1.0, 1.5]
    images = dict(
        shape_corrected_baseline=darsia.OpticalImage(
            np.zeros((2, 3, 3), dtype=np.float32), color_space="RGB", dimensions=dimensions),
        labels=darsia.Image(np.arange(6, dtype=np.uint8).reshape(2, 3), dimensions=dimensions),
        facies=darsia.Image(np.zeros((2, 3), dtype=np.uint8), dimensions=dimensions),
        depth=darsia.Image(np.ones((2, 3), dtype=np.float64), scalar=True, dimensions=dimensions))
    fields = {}
    for name, image in images.items():
        path = tmp_path / (name + ".npz")
        image.save(path)
        fields[name] = dict(**descriptor(path), shape=list(image.img.shape))
    config_path = tmp_path / "run.toml"
    config_path.write_bytes(b"# original configuration binding\n")
    configuration = descriptor(config_path)
    correction_path = tmp_path / "saved_correction.npz"
    correction_path.write_bytes(b"synthetic correction binding; no correction is executed")
    revision = dict(configuration=dict(frame="rig-corrected", frame_binding=dict(
        format=workspace.FRAME_BINDING_FORMAT, source_reference_run="sample",
        display_reference_run="sample", reference_shape_corrections=[descriptor(correction_path)],
        runs={"sample": dict(configuration=configuration, fields=fields,
                             configured_depth_map=fields["depth"])})))
    revision_path = tmp_path / "revision.json"
    revision_path.write_text(json.dumps(revision), encoding="utf-8")
    spatial = reader._spatial_grid(images["shape_corrected_baseline"])
    grid = dict(metadata={k: spatial[k] for k in ("space_dim", "dimensions", "origin")},
                indexing=spatial["indexing"], voxel_size=spatial["voxel_size"],
                num_voxels=spatial["num_voxels"], array_row_direction=spatial["row_direction"],
                array_column_direction=spatial["column_direction"], pixel_centres=spatial["pixel_centres"])
    return dict(pins=reader._Pins(), images=images, fields=fields, descriptor=descriptor,
                revision=revision, revision_path=revision_path, binding=descriptor(revision_path),
                document=dict(inputs=dict(configuration_files=[configuration]),
                              corrected_baseline=dict(grid=grid)))


def test_parent_grid_preserves_generic_scalar_false_archive_metadata(real_parent_grid):
    """Generic rank-two labels and facies keep scalar=False through real save/load."""
    case = real_parent_grid
    labels, compatibility = reader._parent_grid(
        case["pins"], case["document"], case["binding"], "sample")
    case["pins"].final()
    assert labels["metadata"]["scalar"] is False
    assert labels["metadata"]["series"] is False
    assert labels["shape_hw"] == [2, 3]
    assert set(compatibility["fields"]) == set(workspace.FIELDS)
    for name in ("labels", "facies"):
        loaded = darsia.imread(case["fields"][name]["path"])
        assert loaded.scalar is False
        assert loaded.series is False
        assert loaded.range_dim == 0
        assert loaded.range_num == 1
        np.testing.assert_array_equal(loaded.img, case["images"][name].img)
        assert loaded.img.dtype == np.uint8


@pytest.mark.parametrize("kind", ["rgb", "series"])
def test_parent_grid_rejects_rgb_or_series_categorical_archives(real_parent_grid, kind):
    """The unchanged original shape contract rejects vector or time axes."""
    case = real_parent_grid
    if kind == "rgb":
        image = darsia.OpticalImage(np.zeros((2, 3, 3), dtype=np.float32),
                                   color_space="RGB", dimensions=[1.0, 1.5])
    else:
        image = darsia.Image(np.zeros((2, 3, 2), dtype=np.uint8),
                             scalar=True, series=True, time=[0., 1.], dimensions=[1.0, 1.5])
    path = case["revision_path"].parent / "labels.npz"
    image.save(path)
    case["fields"]["labels"] = dict(**case["descriptor"](path), shape=list(image.img.shape))
    case["revision_path"].write_text(json.dumps(case["revision"]), encoding="utf-8")
    case["binding"] = case["descriptor"](case["revision_path"])
    with pytest.raises(ValueError, match="grids differ"):
        reader._parent_grid(case["pins"], case["document"], case["binding"], "sample")
