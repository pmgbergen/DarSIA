"""A separate crop area through export, Apply and the Build crop (local fixtures)."""

import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

import darsia
from darsia.fluidflower import setup_crop_area as crop
from darsia.fluidflower import setup_geometry_apply as apply
from darsia.fluidflower.series_geometry import (
    RigFrame,
    SeriesRunInputs,
    export_native_setup,
    rig_frame_from_inputs,
    series_run_inputs,
)
from darsia.presets.workflows.config.corrections import CurvatureCorrectionConfig

from .test_setup_crop_default_bytes import (  # noqa: F401
    GOLDEN,
    _export_values,
    _geometry_export,
    _plan_values,
    _registration,
    no_aliases,
)

# Points marked on the reference photograph ac1.png (80 x 50 px).
MARKED = {
    "top_left": (8.0, 12.0),
    "top_right": (9.0, 64.0),
    "bottom_right": (40.0, 63.0),
    "bottom_left": (39.0, 13.0),
}


def _separate(marked=MARKED, image="ac1.png"):
    def prepare(project, images):
        crop.save_crop_area(
            project,
            crop.CropArea(marked, "ac1", image=str(images / image), image_size=(80, 50)),
        )
        crop.save_crop_choice(project, crop.CROP_SEPARATE_AREA)

    return prepare


def _plan(root, export):
    return apply.prepare_crop_plan(
        root,
        export.destination / "export.json",
        expected_revision_id="revision-1",
        register=_registration,
    )


def _crop_table(data: bytes) -> dict:
    return tomllib.loads(data.decode("utf-8"))["corrections"]["curvature"]["crop"]


def test_a_stored_area_changes_nothing_while_the_reference_quad_is_chosen(
    tmp_path, no_aliases
):
    def prepare(project, images):
        _separate()(project, images)
        crop.save_crop_choice(project, crop.CROP_REFERENCE_QUAD)

    folder = tmp_path / "fixture"
    folder.mkdir()
    root, _images, _project, export = _geometry_export(folder, prepare_project=prepare)
    assert _export_values(folder, export) == GOLDEN["export"]
    assert _plan_values(folder, root, export) == GOLDEN["apply"]


def test_the_export_records_the_area_next_to_the_rig_frame(tmp_path, no_aliases):
    root, images, project, export = _geometry_export(tmp_path, prepare_project=_separate())
    manifest = json.loads((export.destination / "export.json").read_text(encoding="utf-8"))
    keys = list(manifest)
    assert keys[keys.index("rig_frame") + 1] == "crop_area"
    area = crop.saved_crop_areas(project)["ac1"]
    assert manifest["crop_area"] == area.to_dict()
    expected_digest = hashlib.sha256(
        json.dumps(
            {
                "reference_run": "ac1",
                "frame": manifest["rig_frame"],
                "points": manifest["injection_points"],
                "crop_area": manifest["crop_area"],
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert manifest["inputs_digest"] == expected_digest
    text = export.setup_toml.read_text(encoding="utf-8")
    assert "# The crop is the series' separate crop area" in text
    frame = RigFrame.from_dict(manifest["rig_frame"])
    extent = crop.crop_extent(frame, area)
    assert f"# origin = [{extent.left_m}, {extent.top_m}]" in text
    # Changing the area makes a new export of the same revision.
    moved = dict(MARKED, top_left=(10.0, 14.0))
    crop.save_crop_area(
        project, crop.CropArea(moved, "ac1", image=str(images / "ac1.png"), image_size=(80, 50))
    )
    reference = next(row for row in series_run_inputs(root) if row.run_id == "ac1")
    again = export_native_setup(project, reference=reference)
    assert again.reused is False
    assert json.loads((again.destination / "export.json").read_text())["crop_area"][
        "corners"]["top_left"] == [10.0, 14.0]


def test_apply_crops_every_run_to_the_area_in_reference_coordinates(tmp_path, no_aliases):
    root, _images, project, export = _geometry_export(tmp_path, prepare_project=_separate())
    manifest = json.loads((export.destination / "export.json").read_text(encoding="utf-8"))
    frame = RigFrame.from_dict(manifest["rig_frame"])
    extent = crop.crop_extent(frame, crop.saved_crop_areas(project)["ac1"])
    plan = _plan(root, export)
    by_run = {row["run_id"]: row["crop"] for row in plan.runs}
    assert by_run["ac1"] == extent.native_crop()
    # The stub registration moves ac2 by +2.5 rows and -3.5 columns.
    moved = {name: (row + 2.5, col - 3.5) for name, (row, col) in extent.corners.items()}
    assert by_run["ac2"] == extent.native_crop(moved)
    assert by_run["ac2"]["origin"] == [extent.left_m, extent.top_m]
    assert by_run["ac2"]["width"] == extent.width_m and by_run["ac2"]["height"] == extent.height_m
    for change in plan.changes:
        assert _crop_table(change.after) == by_run[change.run_id]
    receipt = apply.apply_crop_plan(plan, receipt_directory=tmp_path / "receipts")
    assert json.loads(receipt.read_bytes())["state"] == "complete"
    merged = tomllib.loads((root / "config/merged/ac2.toml").read_text(encoding="utf-8"))
    settings = CurvatureCorrectionConfig().load(merged["corrections"]["curvature"])
    assert settings.crop.origin == (extent.left_m, extent.top_m)
    # A run file with a separate crop is no longer read as the rig frame.
    rows = {row.run_id: row for row in series_run_inputs(root)}
    assert rows["ac2"].crop["origin"] == [extent.left_m, extent.top_m]
    assert rig_frame_from_inputs(rows["ac2"]) is None


def test_switching_back_writes_exactly_the_default_crop(tmp_path, no_aliases):
    root, images, project, export = _geometry_export(tmp_path, prepare_project=_separate())
    default_folder = tmp_path / "default"
    default_folder.mkdir()
    default_root, _, _, default_export = _geometry_export(default_folder)
    expected = {
        change.path.relative_to(default_root).as_posix(): change.after
        for change in _plan(default_root, default_export).changes
    }
    apply.apply_crop_plan(_plan(root, export), receipt_directory=tmp_path / "receipts")
    crop.save_crop_choice(project, crop.CROP_REFERENCE_QUAD)
    reference = next(row for row in series_run_inputs(root) if row.run_id == "ac1")
    back = export_native_setup(project, reference=reference)
    assert "crop_area" not in json.loads((back.destination / "export.json").read_text())
    plan = _plan(root, back)
    assert len(plan.changes) == 4
    for change in plan.changes:
        assert "origin" in _crop_table(change.before)
        text = change.after.replace(str(images).replace("\\", "\\\\").encode(), b"<IMAGES>")
        reference_text = expected[change.path.relative_to(root).as_posix()].replace(
            str(default_folder / "photographs").replace("\\", "\\\\").encode(), b"<IMAGES>"
        )
        assert text.replace(str(root).replace("\\", "\\\\").encode(), b"<ROOT>") == (
            reference_text.replace(str(default_root).replace("\\", "\\\\").encode(), b"<ROOT>")
        )


def test_apply_refuses_an_area_marked_on_another_photograph(tmp_path, no_aliases):
    root, images, project, export = _geometry_export(tmp_path, prepare_project=_separate())
    path = export.destination / "export.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["crop_area"]["image"] = str(images / "ac2.png")
    path.write_text(json.dumps(manifest), encoding="utf-8")
    before = {item: item.read_bytes() for item in root.rglob("*.toml")}
    with pytest.raises(ValueError, match="not marked on the exported rig frame's photograph"):
        _plan(root, export)
    assert all(item.read_bytes() == data for item, data in before.items())


def test_the_export_refuses_a_separate_area_it_cannot_apply(tmp_path, no_aliases):
    def without_area(project, images):
        crop.save_crop_choice(project, crop.CROP_SEPARATE_AREA)

    with pytest.raises(ValueError, match="Mark the separate crop area"):
        _geometry_export(tmp_path / "missing", prepare_project=without_area)

    def corrected(project, images):
        _separate()(project, images)
        path = project.root / "overrides" / "geometry_facies" / "current.json"
        current = json.loads(path.read_text(encoding="utf-8"))
        current["configuration"]["frame"] = "rig-corrected"
        path.write_text(json.dumps(current), encoding="utf-8")

    with pytest.raises(ValueError, match="already cropped"):
        _geometry_export(tmp_path / "corrected", prepare_project=corrected)


def _blob_coordinates(image):
    """Physical (x, y) of the bright blob's centre in a corrected image."""
    values = np.asarray(image.img, dtype=float)
    weights = np.where(values > 0.5 * values.max(), values, 0.0)
    rows, cols = np.indices(values.shape)
    row = (weights * rows).sum() / weights.sum()
    col = (weights * cols).sum() / weights.sum()
    dy, dx = (float(value) for value in image.voxel_size)
    x0, y0 = (float(value) for value in image.origin)
    return x0 + (col + 0.5) * dx, y0 - (row + 0.5) * dy


def test_the_built_crop_keeps_the_reference_quad_coordinates():
    frame = RigFrame(
        corners={"top_left": (20.0, 30.0), "bottom_left": (230.0, 25.0),
                 "bottom_right": (225.0, 380.0), "top_right": (15.0, 370.0)},
        width_m=1.0, height_m=0.6, run_id="ac1", image="ac1.png", image_size=(400, 250),
    )
    marked = {name: frame.rig_to_pixel(x, y) for name, (x, y) in {
        "top_left": (0.2, 0.5), "bottom_left": (0.2, 0.1),
        "bottom_right": (0.8, 0.1), "top_right": (0.8, 0.5)}.items()}
    extent = crop.crop_extent(frame, crop.CropArea(marked, "ac1", image_size=(400, 250)))
    assert (extent.left_m, extent.right_m, extent.bottom_m, extent.top_m) == (0.2, 0.8, 0.1, 0.5)
    row, col = frame.rig_to_pixel(0.55, 0.32)
    rows, cols = np.indices((250, 400))
    photo = np.exp(-((rows - row) ** 2 + (cols - col) ** 2) / (2 * 3.0**2)).astype(np.float32)

    def built(table):
        settings = CurvatureCorrectionConfig().load({"active": ["crop"], "crop": table})
        correction = darsia.CurvatureCorrection(config=settings.to_dict())
        return correction(darsia.ScalarImage(photo, dimensions=[0.25, 0.4]))

    reference = built(frame.native_crop())
    separate = built(extent.native_crop())
    assert np.allclose(separate.origin, [0.2, 0.5])
    assert np.allclose(separate.dimensions, [0.4, 0.6])
    for image in (reference, separate):
        assert _blob_coordinates(image) == pytest.approx((0.55, 0.32), abs=0.005)
    # Without the origin the same pixels would sit 0.2 m left and 0.1 m low.
    shifted = dict(extent.native_crop())
    del shifted["origin"]
    x, y = _blob_coordinates(built(shifted))
    assert x == pytest.approx(0.35, abs=0.005) and y == pytest.approx(0.22, abs=0.005)


def test_a_run_crop_with_an_origin_is_not_a_rig_frame():
    table = {"top_left": [3, 4], "bottom_left": [44, 6], "bottom_right": [43, 73],
             "top_right": [4, 72], "width": 0.6, "height": 0.3, "in_meters": True}
    row = SeriesRunInputs(run_id="ac1", config_path=Path("ac1.toml"), crop=table)
    assert rig_frame_from_inputs(row) is not None
    with_origin = SeriesRunInputs(
        run_id="ac1", config_path=Path("ac1.toml"), crop=dict(table, origin=[0.1, 0.4])
    )
    assert rig_frame_from_inputs(with_origin) is None
