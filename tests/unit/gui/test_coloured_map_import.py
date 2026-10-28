"""Local fixture checks for raw-grid import; no classifier or NAS access."""

from pathlib import Path
import hashlib
import json
import tomllib

import numpy as np
from PIL import Image
import pytest

from darsia.fluidflower import coloured_map_import as maps
from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower.series_geometry import RigFrame, SeriesRunInputs


def _inputs(tmp_path):
    baseline = tmp_path / "baseline.png"
    colored = tmp_path / "coloured.png"
    Image.new("RGB", (20, 16), (70, 80, 90)).save(baseline)
    image = Image.new("RGB", (20, 16), (255, 0, 0))
    image.paste((0, 255, 0), (10, 0, 20, 16))
    image.save(colored)
    reference = SeriesRunInputs(
        "ac52",
        tmp_path / "run.toml",
        baseline_image=baseline,
        width_m=0.9,
        height_m=0.49,
    )
    config = {
        "rig": {"width": 0.9, "height": 0.49, "dim": 2},
        "labeling": {"rtol": 0.001, "ensure_connectivity": True, "unite_labels": []},
        "do_not_touch": ["original config"],
    }
    return reference, colored, config


def _fake_original(path, show=False):
    assert show is False
    settings = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    assert set(settings) == {"rig", "labeling"}
    assert settings["labeling"]["rtol"] == 0.001
    assert Path(settings["labeling"]["colored_image"]).is_file()
    labels = np.ones((16, 20), dtype=np.uint8)
    labels[:, 10:] = 2
    np.savez_compressed(settings["labeling"]["labels"], array=labels)


def test_original_workflow_called_once_in_isolated_folder_and_provenance_honest(
    tmp_path,
):
    reference, colored, config = _inputs(tmp_path)
    before = colored.read_bytes()
    calls = []

    def segment(*args, **kwargs):
        calls.append((args, kwargs))
        return _fake_original(*args, **kwargs)

    imported = maps.import_coloured_map(
        colored,
        reference,
        config,
        input_grid="raw",
        work_root=tmp_path / "imports",
        frame=None,
        segment=segment,
    )
    assert len(calls) == 1
    assert colored.read_bytes() == before
    assert config["do_not_touch"] == ["original config"]
    assert "labels" not in config["labeling"]
    assert imported.session.image_path == str(reference.baseline_image)
    assert imported.session.polygons and imported.proposal.image_size == (20, 16)
    assert imported.proposal.digest == lp.proposal_digest(imported.session, None)
    assert all(
        entry.porosity is None and entry.permeability is None
        for entry in imported.session.facies.values()
    )
    rebuilt = lp.rasterize_polygons(imported.session.polygons, (16, 20), (1.0, 1.0))
    assert np.array_equal(rebuilt, imported.proposal.classes)
    receipt = json.loads(imported.receipt.read_bytes())
    assert receipt["extra_trees_fit_run"] is False
    assert "no ExtraTrees fit" in receipt["interpretation"]
    assert receipt["source_map_sha256"] == hashlib.sha256(before).hexdigest()
    assert receipt["frozen_input_sha256"] == receipt["source_map_sha256"]
    assert (
        receipt["resized"] is False and receipt["facies_properties_inferred"] is False
    )


def test_corrected_grid_rejected_even_when_dimensions_match(tmp_path):
    reference, colored, config = _inputs(tmp_path)
    with pytest.raises(ValueError, match="raw reference photograph"):
        maps.import_coloured_map(
            colored,
            reference,
            config,
            input_grid="corrected",
            work_root=tmp_path / "imports",
            frame=None,
            segment=lambda *args: pytest.fail("must not run"),
        )
    assert not (tmp_path / "imports").exists()


def test_no_automatic_resize_or_workflow_for_wrong_dimensions(tmp_path):
    reference, colored, config = _inputs(tmp_path)
    Image.new("RGB", (19, 16), (255, 0, 0)).save(colored)
    with pytest.raises(ValueError, match="without resizing"):
        maps.import_coloured_map(
            colored,
            reference,
            config,
            input_grid="raw",
            work_root=tmp_path / "imports",
            frame=None,
            segment=lambda *args: pytest.fail("must not run"),
        )
    assert not (tmp_path / "imports").exists()


def test_polygons_keep_nested_region_exact_without_contour_simplification(tmp_path):
    reference, _, _ = _inputs(tmp_path)
    labels = np.zeros((16, 20), dtype=np.uint8)
    labels[4:12, 5:15] = 6
    session, proposal = maps.imported_layer_state(
        labels, reference, raw_size=(20, 16), frame=None
    )
    rebuilt = lp.rasterize_polygons(session.polygons, labels.shape, (1.0, 1.0))
    assert np.array_equal(rebuilt, proposal.classes)
    assert len(session.layers) == 2


def test_degenerate_component_is_reported_without_approximation(tmp_path):
    reference, _, _ = _inputs(tmp_path)
    labels = np.ones((16, 20), dtype=np.uint8)
    labels[5, 5] = 2
    with pytest.raises(ValueError, match="cannot represent exactly"):
        maps.imported_layer_state(labels, reference, raw_size=(20, 16), frame=None)


def test_original_workflow_error_is_not_retried(tmp_path):
    reference, colored, config = _inputs(tmp_path)
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("original labeling failure")

    with pytest.raises(RuntimeError, match="original labeling failure"):
        maps.import_coloured_map(
            colored,
            reference,
            config,
            input_grid="raw",
            work_root=tmp_path / "imports",
            frame=None,
            segment=broken,
        )
    assert calls == [1]


def test_other_photo_frame_cannot_mask_the_raw_reference_map(tmp_path):
    reference, colored, _ = _inputs(tmp_path)
    labels = np.ones((16, 20), dtype=np.uint8)
    frame = RigFrame(
        {
            "top_left": (0.0, 0.0),
            "bottom_left": (15.0, 0.0),
            "bottom_right": (15.0, 19.0),
            "top_right": (0.0, 19.0),
        },
        width_m=0.9,
        height_m=0.49,
        run_id=reference.run_id,
        image=str(colored),
        image_size=(20, 16),
    )
    with pytest.raises(ValueError, match="marked on another photograph"):
        maps.imported_layer_state(labels, reference, raw_size=(20, 16), frame=frame)


def test_real_original_palette_workflow_on_local_two_colour_fixture(tmp_path):
    reference, colored, config = _inputs(tmp_path)
    original = colored.read_bytes()
    imported = maps.import_coloured_map(
        colored,
        reference,
        config,
        input_grid="raw",
        work_root=tmp_path / "imports",
        frame=None,
    )
    assert len(imported.session.layers) == 2
    assert np.array_equal(
        imported.proposal.classes[:, :10], np.ones((16, 10), np.uint8)
    )
    assert np.array_equal(
        imported.proposal.classes[:, 10:], np.full((16, 10), 2, np.uint8)
    )
    assert imported.native_labels.with_suffix(".jpg").is_file()
    assert colored.read_bytes() == original
