"""Assemble frozen lamp-context scores without model or campaign I/O."""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from . import features, heads, regional
from .rules import PlacementContext, _cached_fraction, _placement_labels


def prepare_shift_context(
    original_model, base_pixel_model, base_regional_model,
    current_log, aligned_baseline_log, context_domain, *,
    light_epoch_id, spacing_m, feature_shape_hw, metadata,
    parameters: Mapping, feature_parameters: Mapping,
    heads_parameters: Mapping, regional_parameters: Mapping,
    rules_parameters: Mapping,
    aligned_previous_log=None, source_baseline_log=None,
    h_baseline_to_current=None, source_baseline_valid=None,
    base_labels=None, joint_pixel_model=None, projected_regional_model=None,
):
    """Prepare immutable optical evidence in the original lamp-rule order.

    The identity epoch returns before decoding, scoring, regional preparation
    or optional model inspection. The active epoch uses the separate presence
    multiplication and source-normalized filters. A projected head is admitted
    only as an explicit pair; the epoch itself never selects another model.
    Bound model identities, contracts and descriptive audit fields are supplied
    in metadata. No model fitting, file loading or source resampling occurs.
    """
    if light_epoch_id not in parameters['light_epoch_values']:
        raise ValueError('Unknown explicitly bound light epoch')
    valid = np.asarray(context_domain)
    if valid.dtype != bool or valid.ndim != 2:
        raise ValueError('Placement context requires the exact boolean native grid')
    audit = dict(metadata)
    if light_epoch_id == parameters['identity_epoch']:
        return PlacementContext(light_epoch_id, valid, metadata=audit)
    models = {parameters['base_model_key']: base_regional_model}
    projected = None
    supplied = (joint_pixel_model, projected_regional_model)
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise ValueError('Projected placement requires both explicit frozen heads')
        projected = heads.epoch_pixel_model(
            joint_pixel_model, light_epoch_id, parameters=heads_parameters)
        models[parameters['projected_model_key']] = projected_regional_model
        if projected_regional_model['settings'] != base_regional_model['settings']:
            raise ValueError('Projected placement changed frozen regional context settings')
    for values in (current_log, aligned_baseline_log, aligned_previous_log):
        if values is not None and (values.dtype != np.float32 or values.shape != (3, *valid.shape)):
            raise ValueError('Placement requires unchanged CHW float32 source pixels')
    valid = valid & np.isfinite(current_log).all(0) & np.isfinite(aligned_baseline_log).all(0)
    if aligned_previous_log is not None:
        valid &= np.isfinite(aligned_previous_log).all(0)
    current, _ = features.log_to_unit_rgb(current_log, valid, parameters=feature_parameters)
    baseline, _ = features.log_to_unit_rgb(aligned_baseline_log, valid, parameters=feature_parameters)
    previous = None if aligned_previous_log is None else features.log_to_unit_rgb(
        aligned_previous_log, valid, parameters=feature_parameters)[0]
    predictor = heads.CpuFeaturePredictor(
        original_model, baseline, parameters=heads_parameters,
        feature_parameters=feature_parameters, valid=valid,
        feature_shape_hw=feature_shape_hw)
    current_features = predictor.prepare_rgb(current)
    previous_features = predictor.baseline if previous is None else predictor.prepare_rgb(previous)
    presence = np.empty(valid.shape, np.float32)
    for first, last, values in predictor.feature_blocks(
        current_features, previous_features, dynamic_available=previous is not None):
        presence[first:last] = heads.score_packed_presence(
            values, predictor.weights, predictor.bias).reshape(last-first, valid.shape[1])
    flat, flat_valid = features.masked_flatfield_features(
        current_log, aligned_baseline_log, valid,
        source_baseline_log=source_baseline_log,
        h_baseline_to_current=h_baseline_to_current,
        source_baseline_valid=source_baseline_valid,
        parameters=feature_parameters)
    flat[~flat_valid] = 0
    logits = {parameters['base_model_key']: heads.pixel_logits(
        base_pixel_model, flat, parameters=heads_parameters)}
    if projected is not None:
        logits[parameters['projected_model_key']] = heads.pixel_logits(
            projected, flat, parameters=heads_parameters)
    base_probability = heads.shifted_probability(
        logits[parameters['base_model_key']], 0., flat_valid)
    shared = regional.build_shared_regional_context(
        current, baseline, presence, valid, spacing_m,
        regional.Settings(**base_regional_model['settings'], parameters=regional_parameters),
        parameters=regional_parameters)
    audit.update(
        **{parameters["projected_applied_audit_key"]: projected is not None},
        context_columns_unchanged=[i for i in range(parameters['regional_feature_count'])
                                   if i not in regional_parameters['conditional_columns']],
        gas_dependent_columns=list(regional_parameters['conditional_columns']),
        regional_context_shape_hw=list(shared.prepared['domain'].shape),
        optical_context_native_shape_hw=list(feature_shape_hw),
        optical_spatial_sigma_px=float(predictor.spatial_sigma))
    result = PlacementContext(light_epoch_id, valid, flat_valid, base_probability,
                              logits, models, shared, audit)
    if base_labels is not None:
        labels = _placement_labels(base_labels, parameters=rules_parameters)
        if labels.shape != valid.shape:
            raise ValueError('Frozen base labels do not match the prepared native grid')
        _cached_fraction(result, labels, parameters=rules_parameters)
    return result
