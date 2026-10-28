"""Sand-type equalisation: one light field per run, from same-facies sand outside the frame band."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from darsia.fluidflower import image_normalisation as choices
from darsia.fluidflower.mass.n1_pipeline import asset
from darsia.fluidflower.native import full_photometry as subject
from darsia.fluidflower.optical_reference import facies_light_field, polynomial_terms
from darsia.fluidflower.pipeline.current_image_preparation import facies_support, stable_support
from .test_current_image_preparation import inputs
from .test_independent_full_photometry import photo_parameters  # noqa: F401

LEVELS = {1: np.log(.2), 2: np.log(.35), 5: np.log(.1)}


def layers(shape):
    """Three horizontal facies layers, as in a rig's sand map."""
    rows = np.mgrid[:shape[0], :shape[1]][0]
    return np.where(rows < shape[0] // 3, 1, np.where(rows < 2 * shape[0] // 3, 2, 5))


def field(shape, coefficients, native_shape=None):
    """A light field at the producers' coordinates of a (working) grid, or of a native grid."""
    gh, gw = shape
    nh, nw = native_shape or shape
    rows, cols = np.mgrid[:nh, :nw]
    x = (((cols + .5) * gw / nw - .5) - gw / 2) / (gw / 2)
    y = (((rows + .5) * gh / nh - .5) - gh / 2) / (gh / 2)
    return polynomial_terms(x, y, 2) @ np.asarray(coefficients, float)


def test_same_facies_sand_separates_light_from_sand_and_ignores_outliers_and_the_band():
    shape = (64, 96)
    facies = layers(shape)
    light = [-.1, .2, 0., 0., .05]  # y, x, y*y, x*y, x*x
    log_green = np.vectorize(LEVELS.get)(facies) + field(shape, light)
    log_green[8:16, 40:48] += 1.0  # residual dye in one patch
    facies = facies.copy()
    facies[:, :4] = -1  # the band along the frame
    log_green[:, :4] = 5.0
    fitted = facies_light_field(log_green, facies, degree=2, block=4, minimum_blocks=5)
    np.testing.assert_allclose(fitted["coefficients"], light, atol=1e-2)
    for value, level in LEVELS.items():
        assert fitted["facies_levels"][value] == pytest.approx(level, abs=1e-2)
    spread = fitted["interquartile_log_spread"][1]
    assert spread["after"] < .2 * spread["before"]


def test_too_little_sand_outside_the_band_is_refused():
    with pytest.raises(ValueError, match="Too little same-facies sand"):
        facies_light_field(np.zeros((8, 8)), np.ones((8, 8), int), degree=2, block=4)


def test_the_applied_field_flattens_each_facies_and_keeps_the_chart_level():
    native, shape = (128, 192), (64, 96)
    facies = layers(native)
    green = np.exp(np.vectorize(LEVELS.get)(facies) + field(shape, [-.1, .2, 0., 0., .05], native))
    rgb = np.stack([green * .7, green, green * 1.3], axis=-1).astype(np.float32)
    chart = [[4., 176.], [4., 184.], [12., 184.], [12., 176.]]  # (row, column) corners
    light, usable = subject._facies_light_fit(rgb, dict(mask=np.ones(native, bool), facies=facies, width_mm=0.),
                                              shape, degree=2, native_to_u=np.eye(3), chart_roi_rc=chart)
    corrected = subject._apply_light(np.moveaxis(rgb, -1, 0), light, np.eye(3), native)
    for value in LEVELS:
        inside = (facies == value)
        inside[:, :4] = inside[:, -4:] = False
        values = corrected[1][inside]
        assert values.std() / values.mean() < 5e-3
    np.testing.assert_allclose(corrected[:, 8, 180], rgb[8, 180], rtol=5e-3)
    np.testing.assert_allclose(corrected[0] / corrected[1], rgb[..., 0] / rgb[..., 1], rtol=1e-5)
    assert light["anchor"] == "colorchecker" and usable.any()


def test_a_run_keeps_its_start_photo_field_and_its_chemistry(tmp_path, photo_parameters, monkeypatch):
    native = (128, 192)
    parameters = dict(photo_parameters, native_area_log_int_8=2)
    facies = layers(native)
    sand = np.vectorize(LEVELS.get)(facies)
    def photo(coefficients):
        green = np.exp(sand + field((64, 96), coefficients, native))
        return np.stack([green * .7, green, green * 1.3], axis=-1).astype(np.float32)
    frames = {"one": photo([0., .1, 0., 0., 0.]), "two": photo([-.1, .2, 0., 0., .05])}
    changed = frames["two"].copy()
    changed[48:80, 60:120] *= np.array([1.6, 1.3, .5], np.float32)  # CO2 changes the indicator colour
    def load(run):
        pin_file = tmp_path / (run + ".json")
        if not pin_file.exists():
            pin_file.write_text(json.dumps({"run": run}))
        pin = asset(pin_file)
        return dict(run=run, rgb_full=np.moveaxis(frames[run], -1, 0) * 100, rgb_native=frames[run],
            white_range_dn=100., native_to_camera_u_h=np.eye(3), chart_rgb=None, chart_roi_rc=None,
            shape_assets=[pin], image_metadata={},
            source_pins=dict(raw_acquisition=pin, camera_binding=dict(profile_id="camera"),
                             native_configuration=pin, baseline_renderer=pin),
            parameters=dict(photometry=parameters, support=dict(coverage_tolerance=1e-6)))
    choice = choices.choice_record("rig", ["one", "two"], choices.EQUALISE_SAND, "one",
                                   color_method="off", exclusion_band_mm=0)
    choice_file = tmp_path / "series.json"
    choice_file.write_text("{}")
    supports = {run: dict(mask=np.ones(native, bool), facies=facies, width_mm=0.) for run in frames}
    result = subject.prepare_independent_photometry({run: (lambda run=run: load(run)) for run in frames},
        choice, supports, tmp_path / "prepared", choice_authority=asset(choice_file))
    model = json.loads(Path(result["runs"]["two"]["fit"]["path"]).read_bytes())
    assert model["baseline_light_fit"]["producer"].endswith("facies_light_field")
    assert "facies" not in model["stable_support"]
    assert result["reference_proposal"]["uses_lighting"] is True
    binding = json.loads(Path(result["runs"]["two"]["source_binding"]["path"]).read_bytes())
    from darsia.fluidflower.native.raw_rig_source import RawRigSourceProcessor
    source = dict(asset_aliases={}, profiles={"two": "camera"}, native_shape_hw=list(native),
        source_id_pattern="[a-z]+", homography_min_scale=1e-12, homography_match_atol=1e-10,
        symbols={"reference_to_frame_h_key": "reference_h"})
    def sensor(path, **kw):
        return SimpleNamespace(source=binding["baseline_raw"], profile_id="camera",
            metadata=dict(white_level=100, black_level=0, raw_visible_wh=[192, 128]),
            distortion=(), ca_red=(), ca_blue=())
    monkeypatch.setattr(subject.raw_positioning, "read_sensor", sensor)
    monkeypatch.setattr(subject.raw_positioning, "camera_u", lambda *a, **k: np.moveaxis(changed, -1, 0) * 100)
    monkeypatch.setattr(subject.raw_positioning, "load_shape_chain", lambda *a, **k: [])
    monkeypatch.setattr(subject.raw_positioning, "render_native", lambda *a, **k: (None, dict(
        native_to_camera_u_h=np.eye(3).tolist(), camera_u_wh=[192, 128])))
    monkeypatch.setattr(subject, "_light_fit", lambda *a, **k: pytest.fail("refitted light on a later photo"))
    processor = RawRigSourceProcessor(result["runs"]["two"]["source_binding"]["path"],
        parameters=dict(source=source, photometry=parameters, registration={},
                        support=dict(coverage_tolerance=1e-6)))
    actual, info = processor.correct(dict(contract="darsia-raw-native-frame-v1", run="two",
        acquisition=binding["baseline_raw"], reference_to_native_h=np.eye(3).tolist(),
        baseline_to_native_h=np.eye(3).tolist()))
    corrected = np.exp(actual) / 100
    assert info["photometry"]["lighting_application"] == "start-photo-field"
    unchanged = facies == 5
    unchanged[:, :4] = unchanged[:, -4:] = False
    assert corrected[1][unchanged].std() / corrected[1][unchanged].mean() < 5e-3
    # The colour change keeps its full size against unchanged sand of its facies.
    inside, beside = corrected[:, 60, 90], corrected[:, 60, 150]
    np.testing.assert_allclose(inside / beside, [1.6, 1.3, .5], rtol=1e-2)


def test_equalise_sand_choice_records_its_band_and_field_and_old_choices_keep_their_value():
    record = choices.choice_record("series", ["r1", "r2"], choices.EQUALISE_SAND, "r2",
                                   color_method="uniform24", exclusion_band_mm=40)
    assert record["exclusion_band"] == {"kind": "frame-facing-sand-band", "width_mm": 40.0}
    assert record["light_field"] == {"kind": "start-photo-same-facies", "degree": 2}
    assert "stable_area" not in record
    assert choices.selected_corrections(record) == dict(
        lighting=True, color_method="uniform24", light_field="start-photo-same-facies")
    none = choices.choice_record("series", ["r1", "r2"], choices.EQUALISE_SAND, "r2",
                                 color_method="uniform24", exclusion_band_mm=0)
    assert none["exclusion_band"]["width_mm"] == 0.0
    document = choices.with_choice({"series_id": "series"}, "series", ["r1", "r2"], choices.EQUALISE_SAND,
                                   "r2", color_method="uniform24", exclusion_band_mm=40)
    assert choices.saved_choice(document, "series", ("r1", "r2")) == record
    for wrong in (dict(color_method="uniform24"), dict(color_method="uniform24", exclusion_band_mm=-1),
                  dict(color_method="uniform24", exclusion_band_mm=40, stable_band_mm=40),
                  dict(exclusion_band_mm=40), dict(color_method="saved_native", exclusion_band_mm=40)):
        with pytest.raises(ValueError):
            choices.choice_record("series", ["r1", "r2"], choices.EQUALISE_SAND, "r2", **wrong)
    with pytest.raises(ValueError, match="Only sand-type equalisation"):
        choices.choice_record("series", ["r1", "r2"], choices.MATCH_REFERENCE, "r2",
                              color_method="uniform24", stable_band_mm=40, exclusion_band_mm=40)
    old = choices.choice_record("series", ["r1", "r2"], choices.MATCH_REFERENCE, "r2",
                                color_method="uniform24", stable_band_mm=40)
    assert choices.selected_corrections(old) == dict(lighting=True, color_method="uniform24")


def test_facies_support_leaves_out_the_frame_band_and_keeps_every_facies(tmp_path):
    reader, context = inputs(tmp_path)
    band = stable_support(context("r1"), 2., reader=reader)["mask"]
    support = facies_support(context("r1"), 2., reader=reader)
    sand = context("r1").facies == 12
    np.testing.assert_array_equal(support["mask"], sand & ~band)
    assert support["mask"][10, 4:-4].all() and not support["mask"][10, :2].any()
    np.testing.assert_array_equal(support["facies"], context("r1").facies)
    np.testing.assert_array_equal(facies_support(context("r1"), 0., reader=reader)["mask"], sand)
    with pytest.raises(ValueError, match="millimetres"):
        facies_support(context("r1"), -1., reader=reader)
