"""Static physical targets follow appearance across corrected RAW views."""
from copy import deepcopy
import hashlib
import json

import cv2
import numpy as np
import pytest

from darsia.fluidflower import rig_static_references as refs


def packet(tmp_path, image, *, run="source", origin=(-100., -60.), profile="first"):
    raw = tmp_path / f"{run}.RAW"
    raw.write_bytes(run.encode())
    calibration = tmp_path / f"{profile}.json"
    calibration.write_bytes(profile.encode())
    return dict(image_rgb=image, full_raw_sensor_field=True, image_encoding="rgb_uint8",
        camera_u_to_photo_h=[[1, 0, -origin[0]], [0, 1, -origin[1]], [0, 0, 1]],
        extent_u_xywh=[*origin, image.shape[1], image.shape[0]],
        source_raw=dict(path=str(raw), sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
        profile=dict(path=str(calibration), sha256=hashlib.sha256(calibration.read_bytes()).hexdigest(), profile_id=profile),
        valid_support=np.ones(image.shape[:2], bool), run_id=run)


def texture(seed, size=112):
    rng = np.random.default_rng(seed)
    image = rng.integers(40, 215, (size, size, 3), np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .6)
    for _ in range(24):
        centre = tuple(rng.integers(5, size-5, 2).tolist())
        colour = tuple(rng.integers(0, 256, 3).tolist())
        cv2.circle(image, centre, int(rng.integers(2, 9)), colour, -1)
    return image


def scene(tmp_path):
    rng = np.random.default_rng(913)
    image = rng.integers(0, 100, (600, 800, 3), np.uint8)
    positions = [(45, 45), (620, 65), (65, 435), (625, 425)]
    regions = []
    for index, (x, y) in enumerate(positions):
        image[y:y+112, x:x+112] = texture(index)
    source = packet(tmp_path, image)
    for index, (x, y) in enumerate(positions):
        regions.append(refs.capture_region(source, region_id=f"point-{index}", name=f"Fixed mark {index}",
            rect_u_xywh=[x-100, y-60, 111, 111], anchor_u_xy=[x-100+55, y-60+55],
            metric_xy_mm=[x*2., (600-y)*2.]))
    return source, regions, positions


def transformed_scene(source, positions, h, shape=(720, 1024)):
    image = np.random.default_rng(811).integers(110, 220, (*shape, 3), np.uint8)
    for x, y in positions:
        mask = np.zeros(source["image_rgb"].shape[:2], np.uint8)
        mask[y:y+112, x:x+112] = 255
        warped = cv2.warpPerspective(source["image_rgb"], h, shape[::-1])
        available = cv2.warpPerspective(mask, h, shape[::-1]) != 0
        image[available] = warped[available]
    return image


def document(regions):
    value = refs.empty_document()
    value["regions"] = regions
    return value


def test_projective_recognition_uses_static_patches_with_other_profile_and_moving_background(tmp_path):
    source, regions, positions = scene(tmp_path)
    h = np.array([[1.13, .04, 42.], [.02, 1.08, -9.], [.00008, -.0001, 1.]])
    target = packet(tmp_path, transformed_scene(source, positions, h), run="target", origin=(-160, -80), profile="second")
    result = refs.recognize_regions(document(regions), target)
    assert result["status"] == "proposal", result
    assert result["model"]["status"] == "valid"
    assert result["photo"]["profile"]["profile_id"] == "second"
    expected = []
    for region in regions:
        source_xy = np.array(region["anchor_u_xy"])+[100, 60]
        photo_xy = refs._project(h, [source_xy])[0]
        expected.append(photo_xy+[-160, -80])
        observation = result["observations"][region["id"]]
        assert observation["status"] == "recognized", observation
        assert observation["method"] == "sift_homography"
        np.testing.assert_allclose(observation["anchor_u_xy"], expected[-1], atol=1.7)
    projected = refs._project(result["model"]["H_metric_mm_to_camera_u"], [r["metric_xy_mm"] for r in regions])
    np.testing.assert_allclose(projected, expected, atol=1.7)
    assert result["proposal_only"] is True


def test_partial_reference_at_image_edge_has_only_available_anchor_and_no_model(tmp_path):
    source, regions, positions = scene(tmp_path)
    h = np.array([[1., 0, -80], [0, 1., 0], [0, 0, 1.]])
    target = packet(tmp_path, transformed_scene(source, positions, h, (600, 800)), run="edge")
    result = refs.recognize_regions(document(regions[:1]), target)
    observed = result["observations"][regions[0]["id"]]
    assert observed["status"] == "partial", observed
    assert .4 < observed["coverage"] < .9
    np.testing.assert_allclose(observed["anchor_u_xy"], [-80, 40], atol=1.5)
    assert result["model"] is None


def test_masked_raw_support_makes_partial_explicit(tmp_path):
    source, regions, _ = scene(tmp_path)
    target = packet(tmp_path, source["image_rgb"].copy(), run="masked")
    target["valid_support"][45:80, 45:157] = False
    result = refs.recognize_regions(document(regions[:1]), target)
    observed = result["observations"]["point-0"]
    assert observed["status"] == "partial", observed
    assert observed["coverage"] < .8


def test_repeated_or_absent_patches_do_not_fabricate_an_anchor(tmp_path):
    image = np.zeros((320, 600, 3), np.uint8)
    image[80:192, 65:177] = texture(80)
    source = packet(tmp_path, image)
    region = refs.capture_region(source, region_id="mark", name="Fixed mark", rect_u_xywh=[-35, 20, 111, 111],
                                 anchor_u_xy=[20, 75], metric_xy_mm=[0, 0])
    target_image = np.zeros_like(image)
    target_image[80:192, 65:177] = texture(80)
    target_image[80:192, 365:477] = texture(80)
    target = packet(tmp_path, target_image, run="duplicate")
    result = refs.recognize_regions(document([region]), target)
    observed = result["observations"]["mark"]
    assert observed["status"] == "ambiguous", observed
    assert observed["anchor_u_xy"] is None
    missing = packet(tmp_path, np.full_like(image, 90), run="missing")
    observed = refs.recognize_regions(document([region]), missing)["observations"]["mark"]
    assert observed["status"] == "missing"
    assert observed["anchor_u_xy"] is None


def test_bounded_ncc_fallback_recognizes_scale(tmp_path, monkeypatch):
    source, regions, positions = scene(tmp_path)
    h = np.array([[1.25, 0, 10], [0, 1.25, 20], [0, 0, 1.]])
    target = packet(tmp_path, transformed_scene(source, positions, h, (800, 1100)), run="scaled")
    monkeypatch.setattr(refs, "_feature_candidate", lambda *args, **kwargs: None)
    observed = refs.recognize_regions(document(regions[:1]), target)["observations"]["point-0"]
    assert observed["status"] == "recognized", observed
    assert observed["method"] == "multiscale_ncc"
    expected = refs._project(h, [[100, 100]])[0]-[100, 60]
    np.testing.assert_allclose(observed["anchor_u_xy"], expected, atol=2.)


def manual_data():
    metric = [[0., 0.], [1000., 0.], [1000., 500.], [0., 500.], [500., 250.]]
    h = np.array([[.7, .04, -40.], [.02, -.65, 375.], [.00008, .00004, 1.]])
    camera = refs._project(h, metric)
    regions = [dict(id=f"mark-{index}", metric_xy_mm=point, plane_id="rig") for index, point in enumerate(metric)]
    observations = {r["id"]: dict(anchor_u_xy=xy.tolist(), score=1., status="manual", method="manual")
                    for r, xy in zip(regions, camera)}
    return regions, observations, h


def test_manual_metric_model_projects_known_points_and_keeps_depth_planes_separate():
    regions, observations, h = manual_data()
    model = refs.fit_reference_model(regions, observations)
    assert model["status"] == "valid"
    probes = [[-150, 90], [825, 255]]
    np.testing.assert_allclose(refs._project(model["H_metric_mm_to_camera_u"], probes), refs._project(h, probes), atol=.0001)
    regions[-1]["plane_id"] = "rear"
    observations["mark-4"]["anchor_u_xy"] = [10000., 20000.]
    assert refs.fit_reference_model(regions, observations)["status"] == "valid"
    regions[-2]["plane_id"] = "rear"
    missing = refs.fit_reference_model(regions, observations)
    assert missing["status"] == "insufficient"
    assert missing["H_metric_mm_to_camera_u"] is None
    assert "four" in missing["reason"]


def test_explicit_metric_uncertainty_accepts_two_mm_measurement_error_but_not_large_error(tmp_path):
    regions, observations, _ = manual_data()
    h = np.array([[6., .05, -40.], [.02, -6., 3500.], [0., 0., 1.]])
    camera = refs._project(h, [region["metric_xy_mm"] for region in regions])
    for region, xy in zip(regions, camera):
        observations[region["id"]]["anchor_u_xy"] = xy.tolist()
    regions[-1]["metric_xy_mm"][0] += 2.
    exact = refs.fit_reference_model(regions, observations, measurement_uncertainty_mm=0.)
    uncertain = refs.fit_reference_model(regions, observations, measurement_uncertainty_mm=2.)
    assert exact["status"] == "inconsistent", exact
    assert uncertain["status"] == "valid", uncertain
    assert uncertain["measurement_uncertainty_mm"] == 2.
    assert uncertain["image_measurement_budget_u_px"] == 5.
    assert uncertain["max_u_px"] == pytest.approx(exact["max_u_px"])
    assert 5. < uncertain["max_u_px"] < 17.1
    assert uncertain["max_mm"] > 0.
    for residual in uncertain["residuals"]:
        assert residual["allowed_u_px"] == pytest.approx(5.+2.*residual["local_scale_u_px_per_mm"])
        assert residual["error_mm"] >= 0.
    regions[-1]["metric_xy_mm"][0] += 18.
    large = refs.fit_reference_model(regions, observations, measurement_uncertainty_mm=2.)
    assert large["status"] == "inconsistent", large
    assert large["max_u_px"] > max(r["allowed_u_px"] for r in large["residuals"])
    value = refs.empty_document()
    value["measurement_uncertainty_mm"] = 2.
    root = tmp_path / "measurement-project"
    refs.save_references(root, value, reason="Measured geometry proposal", actor="Reviewer")
    assert refs.load_references(root)["measurement_uncertainty_mm"] == 2.
    for invalid in (-1., float("nan"), float("inf"), 101.):
        value["measurement_uncertainty_mm"] = invalid
        with pytest.raises(ValueError, match="Measurement uncertainty"):
            refs.save_references(root, value, actor="Reviewer")


@pytest.mark.parametrize("damage,expected", [("collinear", "degenerate"), ("mirror", "invalid_orientation"),
                                            ("outlier", "inconsistent"), ("stale", "stale")])
def test_metric_model_refuses_degenerate_mirrored_inconsistent_or_stale_anchors(damage, expected):
    regions, observations, _ = manual_data()
    if damage == "collinear":
        for index, region in enumerate(regions):
            region["metric_xy_mm"] = [index*10., index*20.]
    elif damage == "mirror":
        for observation in observations.values():
            observation["anchor_u_xy"][0] *= -1
    elif damage == "outlier":
        observations["mark-4"]["anchor_u_xy"][0] += 70
    else:
        regions[0]["identity"] = "new"
        observations["mark-0"]["region_identity"] = "old"
    model = refs.fit_reference_model(regions, observations)
    assert model["status"] == expected, model
    assert model["H_metric_mm_to_camera_u"] is None


def test_persistence_is_explicit_atomic_and_manual_bindings_survive(tmp_path):
    source, regions, _ = scene(tmp_path)
    value = document(regions)
    photo = refs.photo_binding(source)
    value["runs"]["source"] = dict(run_id="source", photo=photo, model=None,
        observations={r["id"]: dict(anchor_u_xy=r["anchor_u_xy"], score=1., status="manual", method="manual") for r in regions})
    root = tmp_path / "project"
    assert refs.load_references(root) == refs.empty_document()
    assert not root.exists()
    before = deepcopy(value)
    stored = refs.save_references(root, value, reason="Measured fixed targets", actor="Reviewer")
    assert value == before
    files = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()]
    assert files == ["annotations/rig_static_references/references.json"]
    loaded = refs.load_references(root)
    assert loaded == stored
    assert loaded["runs"]["source"]["observations"]["point-0"]["status"] == "manual"
    assert loaded["history"][-1]["reason"] == "Measured fixed targets"
    changed = refs.update_region(regions[0], name="Measured mark", metric_xy_mm=[40, 70], plane_id="front")
    assert changed["identity"] != regions[0]["identity"]
    assert changed["template"] == regions[0]["template"]
    assert changed["source"] == regions[0]["source"]


@pytest.mark.parametrize("damage", ["raw", "profile", "shape", "measurement", "removed_id"])
def test_saved_observations_and_geometry_are_invalidated_when_bindings_change(tmp_path, damage):
    source, regions, _ = scene(tmp_path)
    value = document(regions)
    run = refs.recognize_regions(value, source)
    assert run["model"] is not None
    value["runs"]["source"] = run
    root = tmp_path / "project"
    refs.save_references(root, value, actor="Reviewer")
    path = root / "annotations" / "rig_static_references" / "references.json"
    if damage in ("raw", "profile"):
        key = "source_raw" if damage == "raw" else "profile"
        from pathlib import Path
        Path(source[key]["path"]).write_bytes(b"changed binding")
    else:
        edited = json.loads(path.read_text())
        if damage == "shape":
            edited["runs"]["source"]["photo"]["image_shape"][0] += 10
        elif damage == "measurement":
            edited["regions"][0]["metric_xy_mm"][0] += 10
        else:
            edited["regions"].pop(0)
        path.write_text(json.dumps(edited))
    loaded = refs.load_references(root)
    assert loaded["runs"]["source"]["model"] is None
    assert loaded["runs"]["source"]["status"] == "stale"
    assert loaded["runs"]["source"]["observations"]["point-0"]["status"] == "stale"
    assert loaded["runs"]["source"]["observations"]["point-0"]["anchor_u_xy"] is None


def test_capture_and_recognition_reject_invalid_source_coordinates_and_changed_shape(tmp_path):
    source, regions, _ = scene(tmp_path)
    bad = dict(source, full_raw_sensor_field=False)
    with pytest.raises(ValueError, match="complete original RAW"):
        refs.capture_region(bad, region_id="mark", name="Mark", rect_u_xywh=[0, 0, 30, 30],
                            anchor_u_xy=[5, 5], metric_xy_mm=[0, 0])
    with pytest.raises(ValueError, match="beyond available"):
        refs.capture_region(source, region_id="mark", name="Mark", rect_u_xywh=[-120, -80, 100, 100],
                            anchor_u_xy=[-100, -60], metric_xy_mm=[0, 0])
    edited = deepcopy(regions[0])
    edited["rect_u_xywh"][0] += 1
    result = refs.recognize_regions(document([edited]), source)
    assert result["observations"]["point-0"]["status"] == "invalid"
    with pytest.raises(ValueError, match="Duplicate"):
        refs.save_references(tmp_path / "invalid", document([regions[0], regions[0]]), actor="Reviewer")
    assert not (tmp_path / "invalid").exists()


def test_extent_mapping_keeps_negative_camera_u_and_bounds_encoded_template(tmp_path):
    source, regions, _ = scene(tmp_path)
    del source["camera_u_to_photo_h"]
    binding = refs.photo_binding(source)
    np.testing.assert_allclose(binding["camera_u_to_photo_h"], [[1, 0, 100], [0, 1, 60], [0, 0, 1]])
    assert regions[0]["rect_u_xywh"][0] < 0
    edited = deepcopy(regions[0])
    edited["template"]["shape_hw"] = [50000, 50000]
    with pytest.raises(ValueError, match="shape is invalid"):
        refs.update_region(edited, name="Changed")
