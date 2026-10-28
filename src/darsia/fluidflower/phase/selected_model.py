"""Current trained optical heads on their actual source-supported image grid.

This classifier family uses the saved CO2-presence and conditional-gas heads
directly. It does not borrow legacy edge, placement, G3 or regional models.
Categorical fill retains the original explicit physical-domain accounting.
"""
from copy import deepcopy

import numpy as np

FAMILY = "fresh-native-logistic"


def load_bundle(binding, reader):
    from ..phase_training.model import DualPhaseSignalModel
    policy = reader.json(binding["decision_policy"])["models"]["generic"]
    models = {}
    for role, pin in binding["heads"].items():
        value = reader.json(pin)
        model = DualPhaseSignalModel.from_dict(value)
        if policy[role]["model_sha256"] != value["model_sha256"]:
            raise ValueError("The selected decision policy names another trained head")
        models[role] = model
    contracts = [model.feature_contract for model in models.values()]
    if (len(models) != 2 or contracts[0] != contracts[1]
            or len({(model.schema, model.texture_sigma) for model in models.values()}) != 1):
        raise ValueError("Selected phase heads must share the exact feature contract")
    return dict(family=FAMILY, selected_models=models, selected_policy=policy,
                selected_binding=deepcopy(binding), native={})


def predict_scores(current, baseline, previous, bundle):
    """Run the two actual saved classifiers and their recorded decision policy."""
    from ..phase_training import model as trained
    models, policy = bundle["selected_models"], bundle["selected_policy"]
    first = models["co2_presence"]
    if first.schema == trained.TRANSFERABLE_MODEL_SCHEMA:
        from ..phase_features import (OpticalFeatureContract, prepare_optical_feature_workspace,
                                      optical_feature_matrix)
        contract = OpticalFeatureContract.from_dict(first.feature_contract)
        workspace = prepare_optical_feature_workspace(current, baseline, previous_rgb=previous,
            texture_sigma=first.texture_sigma, contract=contract)
        shape = workspace.current.shape[:2]
        matrix_for = lambda selection: optical_feature_matrix(workspace, selection)
    else:
        image, reference, gray, smooth = trained._feature_workspace(current, baseline,
            texture_sigma=first.texture_sigma)
        shape = image.shape[:2]
        matrix_for = lambda selection: trained._feature_matrix(image, reference, gray, smooth, selection)
    co2, gas = (np.empty(shape, np.float32) for _ in range(2))
    # One shared feature tile feeds both actual heads; never retain a full
    # native-image-by-feature matrix or construct it again for the gas head.
    for start in range(0, shape[0], 64):
        stop = min(shape[0], start + 64)
        matrix = matrix_for(np.s_[start:stop, :])
        co2[start:stop] = models["co2_presence"].predict_features(matrix).reshape(stop-start, shape[1])
        gas[start:stop] = models["gas_within_co2"].predict_features(matrix).reshape(stop-start, shape[1])
    co2_threshold = float(policy["co2_presence"]["presence_threshold"])
    gas_threshold = float(policy["gas_within_co2"]["presence_threshold"])
    labels = np.zeros(co2.shape, np.uint8)
    present = co2 >= co2_threshold
    labels[present] = 1
    labels[present & (gas >= gas_threshold)] = 2
    return labels, co2, gas, (co2_threshold, gas_threshold)


def _centre(temporal, shape):
    x, y = (int(value) for value in temporal.origin_xy)
    height, width = shape
    row, column = -y, -x
    log = temporal.log[:, row:row + height, column:column + width]
    valid = temporal.valid[row:row + height, column:column + width]
    if log.shape != (3, height, width) or valid.shape != tuple(shape):
        raise ValueError("Selected classifier history lacks its native source centre")
    return log, valid


def infer_labels(frame, previous, bundle, parameters, *, output_capture=None):
    from darsia._opencv_runtime import scientific_thread, completed_frame
    from ..native.support import compute_feature_support
    from . import domain, features
    ipp = scientific_thread("selected-model:before")
    native, base = frame["native_input"], bundle["baseline"]["native_input"]
    shape = native.valid.shape
    previous_log = previous_valid = None
    previous_h = None
    if previous is not None:
        if previous["source"]["source_sha256"] != frame["source"]["previous_source_sha256"]:
            raise ValueError("Selected classifier received another acquisition predecessor")
        previous_log, previous_valid = _centre(previous["temporal_input"], shape)
        previous_h = previous["current_h"]
    fp = parameters["features"]
    current_rgb = features.decode_native_log(native.log, native.valid, parameters=fp)
    baseline_rgb = features.decode_native_log(base.log, base.valid, parameters=fp)
    previous_rgb = (None if previous_log is None else
                    features.decode_native_log(previous_log, previous_valid, parameters=fp))
    aligned, prior, alignment = features.align_temporal_inputs(
        baseline=baseline_rgb, current=current_rgb, previous=previous_rgb,
        h_baseline_to_current=frame["current_h"], h_baseline_to_previous=previous_h,
        baseline_origin_xy=(0, 0), previous_origin_xy=(0, 0),
        baseline_source_valid_mask=base.valid, previous_source_valid_mask=previous_valid,
        return_support_mask=True, parameters=fp)
    safe, _raw = compute_feature_support(current_valid=native.valid,
        baseline_valid=base.valid, previous_valid=previous_valid,
        h_baseline_to_current=frame["current_h"], h_baseline_to_previous=previous_h,
        baseline_origin_xy=(0, 0), previous_origin_xy=(0, 0),
        parameters=parameters["feature_support"])
    safe &= alignment.pop("joint_valid_mask") & frame["photo_sand"]
    labels, co2, gas, thresholds = predict_scores(current_rgb, aligned, prior, bundle)
    labels[~safe] = 255
    box = parameters["geometry"]["source_box_xywh"]
    full_shape = frame["context"].rig_domain.shape
    embedded = domain.embed_native_canvas(labels, source_box_xywh=box,
        phase_shape_hw=full_shape, fill=255)
    supported = domain.embed_native_canvas(safe, source_box_xywh=box,
        phase_shape_hw=full_shape, fill=False) & frame["context"].rig_domain
    embedded[~supported] = 255
    fill = {}
    result, classes = domain.fill_rig_labels(embedded, frame["context"].rig_domain,
        frame["observed"] | supported, hidden=frame["context"].hardware_excluded,
        audit=fill, parameters=parameters["domain"])
    classes.update(rig_domain=frame["context"].rig_domain,
        hardware_excluded=frame["context"].hardware_excluded,
        below_sand_floor=np.zeros(full_shape, bool))
    qa = dict(classifier_family=FAMILY, selected_model=deepcopy(bundle["selected_binding"]),
        legacy_models_consumed=False, alignment=alignment, categorical_fill=fill,
        physical_scale_revision=deepcopy(frame["physical_scale_revision"]),
        source_history=dict(previous_source_sha256=frame["source"]["previous_source_sha256"],
            policy=frame["history_policy"]),
        opencv_ipp=completed_frame(ipp, "selected-model:after"))
    if output_capture is not None and "gas_probability_storage" in output_capture:
        from .probability_storage import capture_selected
        capture_selected(output_capture["gas_probability_storage"], co2, gas, safe,
            thresholds=thresholds, labels=result, classes=classes, source_box=box)
    return dict(labels=result, classes=classes, pixel_area_mm2=frame["pixel_area_mm2"],
        qa=qa, source=frame["source"], scientific=frame["scientific"],
        seam_masks={}, post_masks={}, context_parameters=frame["context"].parameters)
