from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader, asset
from darsia.fluidflower.phase_training.model import DualPhaseSignalModel, FEATURE_NAMES
from darsia.fluidflower.pipeline.current_model_basis import require_current_model_basis, _digest
from darsia.fluidflower.phase.selected_model import load_bundle, predict_scores
from darsia.fluidflower.phase_training.raw_frame_context import TEMPORAL_CONTRACT


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return asset(path)


def _selected(root, gas=4., choice=None, *, prepared=True):
    model_root = root / "model"
    photo = dict(format="source-supported-test", unit_denominator=1.)
    native = _write(root / "native.json", {})
    raw = dict(path=str(root / "unopened.ARW"), sha256="a" * 64, size_bytes=100)
    source_root = root / "training_source"
    grid = dict(num_voxels=[8, 8], voxel_size=[.01, .01],
        metadata=dict(space_dim=2, indexing="ij", dimensions=[.08, .08], origin=[0., .08]))
    outputs = [dict(path=str(source_root / name), sha256=digest * 64, size_bytes=20)
               for name, digest in (("labels.npz", "d"), ("facies.npz", "e"))]
    certificate = _write(source_root / "renderer.json", dict(inputs=dict(raw=raw, profile=native), outputs=outputs, renderer=dict(
        source_grid=grid, native_shape_hw=[8, 8], profile_id="camera", camera_u_wh=[8, 8], shape_corrections=[],
        native_to_camera_u_h=np.eye(3).tolist())))
    operation = _write(source_root / "operation_result.json", dict(run="sample", operation="build", complete=True,
        baseline_renderer=certificate))
    bound = dict(run="sample", native_shape_hw=[8, 8], baseline_raw=raw)
    if choice is not None:
        from darsia.fluidflower.native.full_photometry import INDEPENDENT, ORDER
        data = dict(path=str(source_root / "unopened-fit.npz"), sha256="f" * 64, size_bytes=20)
        sidecar = _write(source_root / "fit.json", dict(format=INDEPENDENT, data=data,
            native_shape_hw=[8, 8], native_to_camera_u_h=np.eye(3).tolist(),
            baseline_chart_roi_rc=[[1, 1], [1, 3], [3, 3], [3, 1]],
            baseline_color_fit=dict(gain=[1., 1., 1.]), baseline_light_fit=dict(scale=1.),
            correction_order=ORDER, color_interface="sRGB transfer of camera-linear RGB"))
        bound.update(normalisation=dict(choice=deepcopy(choice)), photometric_method=INDEPENDENT,
            correction_order=ORDER, models={key: dict(data=data, sidecar=sidecar) for key in ("fit", "reference")})
    raw_binding = _write(source_root / "raw_binding.json", bound)
    preparation = _write(source_root / "prepared_sources.json", dict(run="sample", operation_result=operation, raw_binding=raw_binding))
    manifest = _write(source_root / "manifest.json", dict(run="sample", native_configuration=native,
        raw_binding=raw_binding, native_shape_hw=[8, 8],
        **(dict(current_build_preparation=preparation) if prepared else {})))
    basis = dict(format="darsia-trained-image-basis-v1", native_configuration=native,
        input_photometry=photo, input_photometry_sha256=_digest(photo),
        encoding="float32-camera-linear-unit-rgb", temporal_contract=TEMPORAL_CONTRACT,
        runs=[dict(run="sample", native_shape_hw=[8, 8], native_configuration=native,
                   raw_binding=raw_binding, source_manifest=manifest)])
    basis_pin = _write(model_root / "image_basis.json", basis)
    _write(model_root / "summary.json", dict(image_basis=basis_pin))
    decisions = {}
    for role, intercept in (("co2_presence", 4.), ("gas_within_co2", gas)):
        value = DualPhaseSignalModel(tuple(np.zeros(len(FEATURE_NAMES))),
            tuple(np.ones(len(FEATURE_NAMES))), tuple(np.zeros(len(FEATURE_NAMES))),
            intercept, training_provenance=dict(input_photometry=photo, image_basis=basis_pin)).to_dict()
        _write(model_root / "models" / "generic" / (role + "_signal.json"), value)
        decisions[role] = dict(model_sha256=value["model_sha256"], presence_threshold=.5)
    _write(model_root / "phase_decision_policy.json", dict(models=dict(generic=decisions)))
    import toml
    series = dict(series_id="synthetic", runs=dict(ids=["sample"]),
                  method=dict(active="phase_model", phase_model=dict(model_version="model")))
    if choice is not None:
        series["image_normalisation"] = deepcopy(choice)
    (root / "series.toml").write_text(toml.dumps(series), encoding="utf-8")
    return model_root


def test_the_selected_trained_heads_change_real_predictions_without_legacy_models(tmp_path):
    _selected(tmp_path, gas=4.)
    binding = require_current_model_basis(tmp_path, dict(runs=[dict(run="sample")]))
    bundle = load_bundle(binding, DataReader())
    image = np.full((8, 8, 3), .3, np.float32)
    labels, co2, gas, _ = predict_scores(image, image, None, bundle)
    assert (labels == 2).all() and np.all(gas > .5) and np.all(co2 > .5)
    assert not {"edge", "placement", "original_model"} & bundle.keys()
    _selected(tmp_path, gas=-4.)
    replacement = require_current_model_basis(tmp_path)
    labels, _, gas, _ = predict_scores(image, image, None, load_bundle(replacement, DataReader()))
    assert (labels == 1).all() and np.all(gas < .5)
    assert replacement["heads"] != binding["heads"]


def test_a_changed_training_basis_or_policy_is_refused(tmp_path):
    model = _selected(tmp_path)
    basis_path = model / "image_basis.json"
    value = json.loads(basis_path.read_text())
    value["input_photometry"]["unit_denominator"] = 2.
    _write(basis_path, value)
    with pytest.raises(ValueError, match="coherent producer-recorded"):
        require_current_model_basis(tmp_path)
    _selected(tmp_path)
    policy_path = model / "phase_decision_policy.json"
    value = json.loads(policy_path.read_text())
    value["models"]["generic"]["co2_presence"]["model_sha256"] = "a" * 64
    _write(policy_path, value)
    with pytest.raises(ValueError, match="another image basis"):
        require_current_model_basis(tmp_path)


def test_selected_model_does_not_grant_missing_run_coverage(tmp_path):
    _selected(tmp_path)
    with pytest.raises(ValueError, match="cover every"):
        require_current_model_basis(tmp_path, dict(runs=[dict(run="other")]))


def test_fresh_probability_product_keeps_actual_head_roles_and_decisions(tmp_path):
    from darsia.fluidflower.phase.probability_storage import capture_selected, spatial_proof, write_product, read_product
    from PIL import Image
    p = np.array([[.9, .9], [.1, .8]], np.float32)
    r = np.array([[.9, .1], [.8, .7]], np.float32)
    labels = np.array([[2, 1], [0, 2]], np.uint8)
    support = np.ones((2, 2), bool)
    capture = {}
    capture_selected(capture, p, r, support, thresholds=(.5, .5), labels=labels,
        classes=dict(modelled=support), source_box=(0, 0, 2, 2))
    product = capture["product"]
    assert product["evidence"]["decisions"]["selected"]["regional_role"] == "gas_within_co2"
    assert product["arrays"]["regional_probability"][0, 0] == r[0, 0]
    assert spatial_proof(product["arrays"], labels,
        product["evidence"]["decision_thresholds"])["unexplained_pixels"] == 0
    _selected(tmp_path)
    binding = require_current_model_basis(tmp_path)
    phase = dict(family="fresh-native-logistic", selected=binding)
    phase_pin = _write(tmp_path / "phase.json", dict(model_bundles=phase))
    label_path = tmp_path / "labels.png"
    Image.fromarray(labels).save(label_path)
    spec = dict(case_id="actual-selected", source=dict(run="sample", source_sha256="a" * 64))
    roles = write_product(tmp_path, spec, capture, model_bindings=phase,
        phase_configuration=phase_pin, labels=labels, label_descriptor=asset(label_path))
    actual = read_product(roles, reader=DataReader(), source=spec["source"], case_id=spec["case_id"], labels=labels)
    assert set(actual["receipt"]["head_bindings"]) == {"selected"}
    for name, values in product["arrays"].items():
        assert actual["arrays"][name].tobytes() == values.tobytes()


def test_selected_inference_dispatch_runs_real_heads_and_explicit_fill(tmp_path):
    from types import SimpleNamespace
    from darsia.fluidflower.phase.infer import infer_labels
    _selected(tmp_path)
    binding = require_current_model_basis(tmp_path)
    bundle = load_bundle(binding, DataReader())
    valid = np.ones((8, 8), bool)
    native = SimpleNamespace(valid=valid, log=np.full((3, 8, 8), np.log(.3), np.float32))
    bundle["baseline"] = dict(native_input=native)
    context = SimpleNamespace(rig_domain=np.ones((10, 10), bool),
        hardware_excluded=np.zeros((10, 10), bool), parameters={})
    source = dict(source_sha256="c" * 64, previous_source_sha256="b" * 64)
    frame = dict(native_input=native, source=source, current_h=np.eye(3),
        photo_sand=valid, context=context, observed=np.ones((10, 10), bool),
        physical_scale_revision={}, history_policy="original", pixel_area_mm2=1., scientific={})
    previous = dict(source=dict(source_sha256="b" * 64), current_h=np.eye(3),
        temporal_input=SimpleNamespace(log=native.log, valid=valid, origin_xy=(0, 0)))
    parameters = dict(features=dict(native_decoder=dict(denominator=1.),
        alignment=dict(homography_floor=1e-12, coverage_floor=1e-6)),
        feature_support=dict(homography_epsilon=1e-12, coverage_tolerance=1e-6, texture_footprint_hw=[1, 1]),
        geometry=dict(source_box_xywh=[1, 1, 8, 8]), domain=dict(tie_fill_contract="nearest"))
    result = infer_labels(frame, previous, bundle, parameters)
    assert (result["labels"] == 2).all()
    assert result["classes"]["modelled"].sum() == 64
    assert result["classes"]["filled_observed"].sum() == 36
    assert result["qa"]["legacy_models_consumed"] is False
    previous["source"]["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="predecessor"):
        infer_labels(frame, previous, bundle, parameters)


def test_transferable_model_tiled_predictions_match_the_original_producer(monkeypatch):
    from darsia.fluidflower import phase_features
    from darsia.fluidflower.phase_training.model import TRANSFERABLE_MODEL_SCHEMA, TRANSFERABLE_MODEL_ALGORITHM
    contract = phase_features.OpticalFeatureContract(
        chemistry_adapter="none", reference_adapter="none", include_dynamic=True,
        generic_version=phase_features.RELATIVE_BRIGHT_GRAIN_FEATURE_VERSION)
    count = len(contract.feature_names)
    generator = np.random.default_rng(17)
    current, baseline, previous = (generator.uniform(.1, .8, (70, 9, 3)).astype(np.float32) for _ in range(3))
    models = {role: DualPhaseSignalModel(tuple(np.zeros(count)), tuple(np.ones(count)),
        tuple(generator.normal(0, .3, count)), .2, schema=TRANSFERABLE_MODEL_SCHEMA,
        algorithm=TRANSFERABLE_MODEL_ALGORITHM, feature_names=contract.feature_names,
        feature_contract=contract.to_dict()) for role in ("co2_presence", "gas_within_co2")}
    expected = [model.predict_score(current, baseline, previous_rgb=previous) for model in models.values()]
    policy = {role: dict(presence_threshold=.5) for role in models}
    prepare = phase_features.prepare_optical_feature_workspace
    calls = []
    def workspace(*args, **kwargs):
        calls.append(True)
        return prepare(*args, **kwargs)
    monkeypatch.setattr(phase_features, "prepare_optical_feature_workspace", workspace)
    labels, co2, gas, _ = predict_scores(current, baseline, previous,
        dict(selected_models=models, selected_policy=policy))
    assert calls == [True]
    assert np.array_equal(co2, expected[0]) and np.array_equal(gas, expected[1])
    assert np.array_equal(labels == 2, (expected[0] >= .5) & (expected[1] >= .5))
