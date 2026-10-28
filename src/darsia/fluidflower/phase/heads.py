"""Frozen optical and gas heads with their original scoring arithmetic."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from copy import deepcopy
from typing import Any
import cv2
import numpy as np
from scipy.special import expit
from .features import OpticalFeatureContract, FeatureBuilder

@dataclass(frozen=True)
class SerializedPhaseHead:
    """One frozen standardized head, without training or radiometry substitution."""
    name: str
    feature_contract: OpticalFeatureContract
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    threshold: float
    radiometry_profile: Mapping | None

    @classmethod
    def from_mapping(cls, payload, *, name, feature_parameters, parameters):
        """Parse the original model JSON values with its float64 coefficient storage."""
        contract = OpticalFeatureContract.from_dict(dict(payload.get('feature_contract') or {}), parameters=feature_parameters)
        mean = np.asarray(payload.get('feature_mean') or [], dtype=np.float64)
        scale = np.asarray(payload.get('feature_scale') or [], dtype=np.float64)
        coefficients = np.asarray(payload.get('coefficients') or [], dtype=np.float64)
        if not mean.size == scale.size == coefficients.size == len(contract.feature_names):
            raise ValueError('Invalid serialized head dimensions')
        threshold = float(((payload.get('training_provenance') or {}).get('cross_validation') or {}).get('oof_threshold', parameters['default_threshold']))
        return cls(name=name, feature_contract=contract, mean=mean, scale=np.maximum(scale, np.finfo(np.float64).eps), coefficients=coefficients, intercept=float(payload.get('intercept') or 0.0), threshold=threshold, radiometry_profile=(payload.get('training_provenance') or {}).get('radiometry_profile'))

    def probability(self, features: np.ndarray) -> np.ndarray:
        'Apply original standardized logistic evidence without inferring concentration or mass.'
        values = np.asarray(features, dtype=np.float64)
        if values.shape[-1] != self.coefficients.size:
            raise ValueError(f'Feature width {values.shape[-1]} differs from {self.coefficients.size}')
        logits = (values - self.mean) / self.scale @ self.coefficients + self.intercept
        output = np.empty_like(logits, dtype=np.float64)
        positive = logits >= 0
        output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
        exp_values = np.exp(logits[~positive])
        output[~positive] = exp_values / (1.0 + exp_values)
        return output.astype(np.float32)

@dataclass(frozen=True)
class HierarchicalPhaseModel:
    """Source-bound presence and conditional-gas heads on one feature contract."""
    co2_presence: SerializedPhaseHead
    gas_within_co2: SerializedPhaseHead
    adapter: str

    @classmethod
    def from_mappings(cls, documents, *, adapter, feature_parameters, parameters):
        """Load already content-verified documents without inspecting campaigns."""
        co2 = SerializedPhaseHead.from_mapping(documents['co2_presence'], name='co2_presence', feature_parameters=feature_parameters, parameters=parameters)
        gas = SerializedPhaseHead.from_mapping(documents['gas_within_co2'], name='gas_within_co2', feature_parameters=feature_parameters, parameters=parameters)
        if co2.feature_contract != gas.feature_contract:
            raise ValueError('Hierarchical heads use different contracts')
        return cls(co2, gas, str(adapter))

    @property
    def feature_contract(self):
        """The original common column order of both frozen heads."""
        return self.co2_presence.feature_contract

    def predict_features(self, features: np.ndarray) -> dict[str, np.ndarray]:
        'Validate and apply explicit frozen optical data without fitting or changing pixels.'
        co2 = self.co2_presence.probability(features)
        conditional_gas = self.gas_within_co2.probability(features)
        free_gas = co2 * conditional_gas
        aqueous = co2 * (np.float32(1.0) - conditional_gas)
        water = np.float32(1.0) - co2
        return {'co2_presence_probability': co2, 'gas_probability_conditional_on_co2': conditional_gas, 'untouched_water_probability': water, 'aqueous_co2_probability': aqueous, 'free_co2_gas_probability': free_gas}

def packed_parameters(model):
    'Fold original optical normalization into two linear heads before its float32 cast.'
    heads = (model.co2_presence, model.gas_within_co2)
    weights = np.column_stack([h.coefficients / h.scale for h in heads])
    bias = np.array([h.intercept - h.mean @ weights[:, i] for i, h in enumerate(heads)])
    return (weights, bias)

def score_packed_heads(features, weights, bias):
    """Original float32 two-head multiplication; retain it separate from presence."""
    return expit(features @ weights + bias)

def score_packed_presence(features, weights, bias):
    """Original single-head presence multiplication for regional context."""
    return expit(features @ weights[:, 0] + bias[0])

class CpuFeaturePredictor(FeatureBuilder):
    """Original packed CPU predictor composed with the unchanged feature builder."""

    def __init__(self, model, baseline, *, parameters, feature_parameters, block_rows=None, texture_sigma=None, input_encoding=None, feature_block_rows=None, valid=None, feature_shape_hw=None):
        """Validate frozen optical contracts and bind the two CPU heads to baseline features."""
        self.model = model
        d = feature_parameters['contract']['defaults']
        supported = tuple((OpticalFeatureContract(**dict(d, generic_version=version, include_dynamic=dynamic), parameters=feature_parameters) for version in feature_parameters['contract']['packed_versions'] for dynamic in (False, True)))
        if model.feature_contract not in supported or any((h.radiometry_profile is not None for h in (model.co2_presence, model.gas_within_co2))):
            raise ValueError('Packed optical heads require the bound generic contract')
        block_rows = feature_parameters['optical']['block_rows'] if block_rows is None else block_rows
        texture_sigma = feature_parameters['optical']['texture_sigma'] if texture_sigma is None else texture_sigma
        input_encoding = feature_parameters['optical']['unit_encoding'] if input_encoding is None else input_encoding
        if feature_block_rows is not None:
            if not isinstance(feature_block_rows, int) or isinstance(feature_block_rows, bool) or feature_block_rows < 1:
                raise ValueError('feature_block_rows must be a positive integer or None')
            block_rows = min(block_rows, feature_block_rows)
        self.feature_block_rows = feature_block_rows
        weights, bias = packed_parameters(model)
        self.weights, self.bias = (weights.astype(np.float32), bias.astype(np.float32))
        super().__init__(model.feature_contract, baseline, parameters=feature_parameters, input_encoding=input_encoding, block_rows=block_rows, texture_sigma=texture_sigma, valid=valid, feature_shape_hw=feature_shape_hw)
        self.head_parameters = parameters

    def predict(self, current_rgb, previous_rgb, *, current_key=None, previous_key=None):
        'Apply original packed optical scoring to prepared columns without changing the current grid.'
        current = self.prepare_rgb(current_rgb)
        previous = self.baseline if previous_rgb is None else self.previous if previous_key is not None and self.previous_key == previous_key else self.prepare_rgb(previous_rgb)
        height, width = current_rgb.shape[:2]
        output = np.empty((height, width, 2), dtype=np.float32)
        for start, stop, features in self.feature_blocks(current, previous, dynamic_available=previous_rgb is not None):
            output[start:stop] = expit(features @ self.weights + self.bias).reshape(stop - start, width, 2)
        self.previous, self.previous_key = (current, current_key)
        return {'co2_presence_probability': output[..., 0], 'gas_probability_conditional_on_co2': output[..., 1]}

def gas_probability(model: Mapping[str, Any], current: np.ndarray, flatfield: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    'Score frozen gas evidence with float64 standardization and float32 output.'
    variant = model['variant']
    if variant not in (*parameters['variants']['current'], parameters['variants']['flatfield_only'], parameters['variants']['combined']):
        raise ValueError(f'Unknown gas feature variant {variant!r}')
    features = current if variant in parameters['variants']['current'] else flatfield if variant == parameters['variants']['flatfield_only'] else np.concatenate((current, flatfield), axis=-1)
    fit = model['fit']
    values = np.asarray(features, np.float64)
    mean, scale, coefficients = (np.asarray(fit[key], np.float64) for key in ('mean', 'scale', 'coefficients'))
    if values.shape[-1] != len(model['feature_names']) or scale.shape != coefficients.shape or (scale <= 0).any():
        raise ValueError('Gas-head feature dimensions or normalization differ')
    from scipy.special import expit
    return expit((values - mean) / scale @ coefficients + float(fit['intercept'])).astype(np.float32)

def regional_probability(features, fit, *, parameters: Mapping):
    'Score pooled context in the original float32 folded-weight order.'
    values = np.asarray(features, np.float32)
    mean, scale, coef = (np.asarray(fit[k], np.float32) for k in ('mean', 'scale', 'coefficients'))
    if any((a.shape != (len(parameters['regional_feature_names']),) for a in (mean, scale, coef))) or (scale <= 0).any():
        raise ValueError('Region head differs from its feature contract')
    return expit(values @ (coef / scale) + float(fit['intercept']) - mean @ (coef / scale)).astype(np.float32)

def encode_decode_probabilities(g3, regional, flat_valid, *, labelled, parameters: Mapping):
    """Literal export/decoding order; do not substitute float32 probabilities."""
    pixel, scores, valid = map(np.asarray, (g3, regional, flat_valid))
    if pixel.dtype != np.float32 or pixel.ndim != 2 or scores.dtype != np.float32 or (scores.ndim != 2) or (valid.dtype != bool) or (valid.shape != pixel.shape) or (not np.isfinite(pixel[valid]).all()) or (not np.isfinite(scores).all()) or np.any((pixel[valid] < 0) | (pixel[valid] > 1)) or np.any((scores < 0) | (scores > 1)) or (type(labelled) is not bool):
        raise ValueError('Native-centre encoding requires valid float32 probability grids')
    if labelled:
        stored_g3 = pixel.astype(np.float16)
        stored_regional = cv2.resize(scores, pixel.shape[::-1], interpolation=cv2.INTER_LINEAR).astype(np.float16)
        decoded_g3 = stored_g3.astype(np.float32)
        decoded_regional = stored_regional.astype(np.float32)
        layout = 'native-float16'
    else:
        stored_g3 = np.full(pixel.shape, parameters['encoding']['invalid_code'], np.uint8)
        stored_g3[valid] = np.rint(parameters['encoding']['compact_scale'] * pixel[valid]).astype(np.uint8)
        decoded_g3 = stored_g3.astype(np.float32) / np.float32(parameters['encoding']['compact_scale'])
        decoded_g3[stored_g3 == parameters['encoding']['invalid_code']] = np.nan
        stored_regional = scores.astype(np.float16)
        decoded_regional = stored_regional.astype(np.float32)
        layout = 'compact-g3-u8-regional-own-grid-float16'
    return (decoded_g3, decoded_regional, dict(storage_layout=layout, native_shape_hw=list(pixel.shape), regional_working_shape_hw=list(scores.shape), g3_compact_rule='round-to-nearest-even(254*p); 255 invalid; decode code/254', regional_resize='float32 cv2.INTER_LINEAR to native grid before threshold', labelled=labelled, encoding_performed=True))

def _bool_grid(value, shape, name):
    array = np.asarray(value)
    if array.dtype != bool or array.shape != shape:
        raise ValueError(f'optical {name} must be a boolean mask on the exact base grid')
    return array

def pixel_logits(pixel_model, features4, *, chunk_rows=None, parameters: Mapping):
    """Original float64 G3 standardization, without a probability/logit round trip.

    Row chunks avoid allocating a native-grid four-column float64 duplicate.
    The optical indicator must already have been folded into this four-feature head.
    """
    if chunk_rows is None:
        chunk_rows = parameters['score_chunk_rows']
    x = np.asarray(features4)
    fit = pixel_model['fit']
    mean, scale, coefficients = (np.asarray(fit[k], np.float64) for k in ('mean', 'scale', 'coefficients'))
    if x.ndim != 3 or x.shape[2] != parameters['flat_feature_count'] or mean.shape != (parameters['flat_feature_count'],) or (scale.shape != (parameters['flat_feature_count'],)) or (coefficients.shape != (parameters['flat_feature_count'],)) or np.any(scale <= 0) or (not np.isfinite(x).all()) or (not all((np.isfinite(v).all() for v in (mean, scale, coefficients)))) or (not np.isfinite(fit['intercept'])) or (int(chunk_rows) != chunk_rows) or (chunk_rows < 1):
        raise ValueError('optical logit scoring needs finite four-feature inputs and a bounded positive row chunk')
    result = np.empty(x.shape[:2], np.float64)
    for first in range(0, x.shape[0], int(chunk_rows)):
        last = min(first + int(chunk_rows), x.shape[0])
        values = np.asarray(x[first:last], np.float64)
        result[first:last] = (values - mean) / scale @ coefficients + fit['intercept']
    return result

def shifted_probability(logits, delta, valid=None):
    'Validate and apply explicit frozen optical data without fitting or changing pixels.'
    if not np.isfinite(delta):
        raise ValueError('optical shift must be finite')
    values = np.asarray(logits)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError('optical needs finite cached native logits')
    probability = expit(values + float(delta)).astype(np.float32)
    if valid is not None:
        probability[~_bool_grid(valid, values.shape, 'flat-field support')] = np.nan
    return probability

def epoch_pixel_model(model, light_epoch_id, *, parameters: Mapping):
    """Fold a known indicator into the intercept, without fitting another head."""
    if light_epoch_id not in parameters['epoch']['values']:
        raise ValueError('optical needs a project-declared light epoch')
    result = deepcopy(model)
    fit = result['fit']
    fit['intercept'] += fit['coefficients'][4] * (float(light_epoch_id == parameters['epoch']['positive_value']) - fit['mean'][4]) / fit['scale'][4]
    for key in ('mean', 'scale', 'coefficients'):
        fit[key] = fit[key][:4]
    result['feature_names'] = list(parameters['flat_feature_names'])
    return result


def validate_refit(frozen, refit, *, parameters: Mapping):
    'Require identical context and non-gas heads while admitting only the declared gas replacement.'
    if frozen['settings'] != refit['settings']:
        raise ValueError('Regional refit changed frozen settings')
    if frozen.get('feature_names') != refit.get('feature_names'):
        raise ValueError('Regional refit changed feature names')
    if set(frozen['heads']) != set(refit['heads']):
        raise ValueError('Regional refit changed the set of heads')
    for name in frozen['heads']:
        if name != parameters["gas_head_name"] and frozen['heads'][name] != refit['heads'][name]:
            raise ValueError('Regional refit changed a frozen non-gas head')
