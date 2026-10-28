"""Current image choices and Build grids admit the actual trained basis only."""
import json
import re

import pytest
import toml

from darsia.fluidflower.image_normalisation import choice_record, MATCH_REFERENCE, COLOR_UNIFORM24
from darsia.fluidflower.mass.n1_pipeline import DataReader, asset
from darsia.fluidflower.pipeline.current_model_basis import require_current_model_basis, UNBOUND_MODEL_REASON
from .test_current_selected_model import _selected, _write


def _choice(**changes):
    values = dict(color_method=COLOR_UNIFORM24, stable_band_mm=20.)
    values.update(changes)
    return choice_record("synthetic", ["sample"], MATCH_REFERENCE, "sample", **values)


def _manifest(root, **changes):
    import tomllib
    path = root / "series.toml"
    value = tomllib.loads(path.read_text())
    value.update(changes)
    path.write_text(toml.dumps(value))


def _snapshot(root, *, change=None, fit_change=None, data_change=False):
    original = json.loads((root / "training_source/operation_result.json").read_text())
    if change is not None:
        certificate = json.loads((root / "training_source/renderer.json").read_text())
        change(certificate)
        original["baseline_renderer"] = _write(root / "current_build/renderer.json", certificate)
        operation = _write(root / "current_build/operation_result.json", original)
    else:
        operation = asset(root / "training_source/operation_result.json")
    runs = [dict(run="sample", operation="build", state="Complete",
                 result=dict(files={"operation_result.json": operation}))]
    import tomllib
    choice = tomllib.loads((root / "series.toml").read_text()).get("image_normalisation")
    if choice is not None:
        from darsia.fluidflower.pipeline.setup_image_preparation import FORMAT, OPERATION
        binding = asset(root / "training_source/raw_binding.json")
        if fit_change is not None or data_change:
            bound = json.loads((root / "training_source/raw_binding.json").read_text())
            for role, model in bound["models"].items():
                sidecar = json.loads((root / "training_source/fit.json").read_text())
                if fit_change is not None:
                    fit_change(sidecar)
                if data_change:
                    sidecar["data"]["sha256"] = "c" * 64
                model["data"] = sidecar["data"]
                model["sidecar"] = _write(root / "current_build" / (role + ".json"), sidecar)
            binding = _write(root / "current_build/raw_binding.json", bound)
        receipt = _write(root / "current_build/image_preparation.json", dict(format=FORMAT, complete=True,
            choice=choice, operations=dict(sample=operation), runs=dict(sample=dict(source_binding=binding))))
        runs.append(dict(run="setup-image-preparation", operation=OPERATION, state="Complete",
                         result=dict(image_preparation=receipt)))
    return dict(runs=runs)


class MetadataReader(DataReader):
    def path(self, descriptor):
        assert descriptor["path"].endswith((".json", ".toml")), "No RAW/image/array reads during model admission"
        return super().path(descriptor)

    def array(self, *_args, **_kwargs):
        pytest.fail("Model admission must not read image arrays")


def test_unchanged_current_choice_and_real_build_grid_are_admitted_without_image_reads(tmp_path):
    _selected(tmp_path, choice=_choice())
    result = require_current_model_basis(tmp_path, build_snapshot=_snapshot(tmp_path), reader=MetadataReader())
    assert result["family"] == "fresh-native-logistic"


@pytest.mark.parametrize("changed", [_choice(color_method="off"), _choice(stable_band_mm=5.)])
def test_saved_optical_change_requires_retraining_before_build_is_considered(tmp_path, changed):
    _selected(tmp_path, choice=_choice())
    _manifest(tmp_path, image_normalisation=changed)
    with pytest.raises(ValueError, match=re.escape(UNBOUND_MODEL_REASON)):
        require_current_model_basis(tmp_path, reader=MetadataReader())


def test_save_audit_protocols_and_analysis_boxes_do_not_change_the_image_basis(tmp_path):
    choice = _choice()
    _selected(tmp_path, choice=choice)
    choice["decision"] = dict(actor="another user", approved_utc="2026-10-26T10:00:00+00:00",
                              reason="Saved the same image choices", decision_ref="image-normalisation-method-choice")
    _manifest(tmp_path, image_normalisation=choice,
              protocols=dict(pressure_temperature=["new_conditions.xlsx", "sample"]),
              analysis_boxes=dict(regions=[dict(name="new mass region", x_min=1, x_max=2)]))
    require_current_model_basis(tmp_path, build_snapshot=_snapshot(tmp_path), reader=MetadataReader())


@pytest.mark.parametrize("change", [
    lambda value: value["renderer"].update(native_shape_hw=[9, 8]),
    lambda value: value["renderer"]["native_to_camera_u_h"][0].__setitem__(2, 1.),
    lambda value: value["renderer"]["source_grid"]["metadata"].update(dimensions=[.16, .08]),
    lambda value: value["inputs"]["raw"].update(sha256="b" * 64),
    lambda value: value["outputs"][0].update(sha256="c" * 64),
    lambda value: value["outputs"][1].update(sha256="c" * 64),
    lambda value: value["renderer"].update(shape_corrections=[dict(path="unopened-curvature.npz", sha256="c" * 64, size_bytes=20)]),
])
def test_new_crop_registration_physical_grid_or_raw_baseline_requires_retraining(tmp_path, change):
    _selected(tmp_path, choice=_choice())
    with pytest.raises(ValueError, match=re.escape(UNBOUND_MODEL_REASON)):
        require_current_model_basis(tmp_path, build_snapshot=_snapshot(tmp_path, change=change), reader=MetadataReader())


def test_protocol_time_and_nonspatial_renderer_metadata_do_not_require_retraining(tmp_path):
    _selected(tmp_path, choice=_choice())
    snapshot = _snapshot(tmp_path, change=lambda value: value["renderer"]["source_grid"]["metadata"].update(
        date="2026-10-26", time=42., analysis_boxes=[1, 2]))
    require_current_model_basis(tmp_path, build_snapshot=snapshot, reader=MetadataReader())


def test_new_native_parameter_data_requires_retraining(tmp_path):
    _selected(tmp_path, choice=_choice())
    current = _write(tmp_path / "new_native.json", dict(source=dict(changed_image_parameter=1)))
    _manifest(tmp_path, native=dict(configuration=current["path"], sha256=current["sha256"]))
    with pytest.raises(ValueError, match=re.escape(UNBOUND_MODEL_REASON)):
        require_current_model_basis(tmp_path, reader=MetadataReader())


@pytest.mark.parametrize("change", [
    lambda value: value["baseline_chart_roi_rc"][0].__setitem__(0, 2),
    lambda value: value.update(baseline_color_fit=dict(gain=[1.1, 1., 1.])),
    lambda value: value.update(baseline_light_fit=dict(scale=1.2)),
])
def test_changed_actual_colorchecker_or_stable_light_fit_requires_retraining(tmp_path, change):
    _selected(tmp_path, choice=_choice())
    with pytest.raises(ValueError, match=re.escape(UNBOUND_MODEL_REASON)):
        require_current_model_basis(tmp_path, build_snapshot=_snapshot(tmp_path, fit_change=change), reader=MetadataReader())


def test_changed_reference_pixels_or_stable_mask_requires_retraining_without_reading_arrays(tmp_path):
    _selected(tmp_path, choice=_choice())
    with pytest.raises(ValueError, match=re.escape(UNBOUND_MODEL_REASON)):
        require_current_model_basis(tmp_path, build_snapshot=_snapshot(tmp_path, data_change=True), reader=MetadataReader())


def test_same_fit_saved_under_new_geometry_audit_and_source_locators_is_reused(tmp_path):
    _selected(tmp_path, choice=_choice())
    snapshot = _snapshot(tmp_path, fit_change=lambda value: value.update(
        source=dict(adopted_revision=dict(sha256="b" * 64), pressure_protocol="new.xlsx"),
        stable_support=dict(analysis_region="another mass region")))
    require_current_model_basis(tmp_path, build_snapshot=snapshot, reader=MetadataReader())


def test_changed_sampling_raster_requires_retraining_but_material_properties_alone_do_not(tmp_path):
    _selected(tmp_path, choice=_choice())
    change = lambda value: value.update(stable_support=dict(source_labels=dict(sha256="a" * 64)))
    with pytest.raises(ValueError, match=re.escape(UNBOUND_MODEL_REASON)):
        require_current_model_basis(tmp_path, build_snapshot=_snapshot(tmp_path, fit_change=change), reader=MetadataReader())
    snapshot = _snapshot(tmp_path, fit_change=lambda value: value.update(
        stable_support=dict(material_table=dict(sha256="b" * 64), porosity=.3, permeability=42)))
    require_current_model_basis(tmp_path, build_snapshot=snapshot, reader=MetadataReader())


def test_missing_current_build_run_requires_retraining(tmp_path):
    _selected(tmp_path, choice=_choice())
    with pytest.raises(ValueError, match=re.escape(UNBOUND_MODEL_REASON)):
        require_current_model_basis(tmp_path, build_snapshot=dict(runs=[]), reader=MetadataReader())


def test_missing_preparation_provenance_is_not_promoted_to_a_current_model(tmp_path):
    _selected(tmp_path, choice=_choice(), prepared=False)
    # The historical model, heads and summary remain internally consistent.
    with pytest.raises(ValueError, match=re.escape(UNBOUND_MODEL_REASON)):
        require_current_model_basis(tmp_path, build_snapshot=_snapshot(tmp_path), reader=MetadataReader())
