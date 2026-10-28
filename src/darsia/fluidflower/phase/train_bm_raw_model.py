"""Refit the existing BM optical and regional methods on bound full-RAW inputs.

The historical physical representation is internal to feature calculation.
Each source RAW, transferred mask and per-file Sony profile remains full-frame.
No production selection or shared job runner is modified by this entrypoint.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .bm_raw_numerics import load_numerics
from .prepare_bm_raw_training import bound, data_root, sha256, write


HEADS = ('co2_presence', 'gas_within_co2')
REGIONAL_COLUMNS = np.array([4, 5, 6, 7, 8, 14, 15, 16, 17, 18])
DOMAIN_SHA = 'bece7b3511a13dfa6ba969566efd4a2a0615d4d3169a34d137da2968f79b5eca'
# Relative to the configured shared data folder (prepare_bm_raw_training.data_root).
DOMAIN = Path('Albus/Autokalibrering_log/phase_models/iterative_v10_20260908/ffbm_specialist/phase-model-20260908T192122Z-fca39f0a/plume_context/ffbm/c1.npz')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), allow_nan=False), flush=True)


def packed(fit, *, regional=False):
    result = asdict(fit)
    for key in ('mean', 'scale', 'coefficients'):
        value = result[key]
        if regional:
            full = np.ones(20) if key == 'scale' else np.zeros(20)
            full[REGIONAL_COLUMNS] = value
            value = full
        result[key] = value.tolist()
    return result


def unpack(logistic, fit):
    return logistic.LogisticFit(**{k: np.asarray(v) if k in ('mean', 'scale', 'coefficients') else v
                                  for k, v in fit.items()})


def content_input(dataset, index):
    """Resolve self-contained frame arrays, never stale absolute cache paths."""
    folder = Path(dataset)/f'frame_{index:03d}'
    receipt = read(folder/'receipt.json')
    values = {}
    for name in ('content_rgb.npy', 'content_valid_support.npy'):
        path = bound(folder/name, receipt['arrays'][name]['sha256'])
        values[name] = np.load(path, mmap_mode='r')
    rgb, support = values['content_rgb.npy'], values['content_valid_support.npy']
    if rgb.shape != (*support.shape, 3) or support.dtype != bool:
        raise ValueError('BM feature input and support dimensions differ')
    # The pilot establishes complete calibrated content coverage. Fail closed
    # if a later frame cannot preserve it; never silently label black padding.
    if not support.all() or not np.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError(f'Incomplete finite BM physical feature input: frame {index}')
    return rgb, receipt


def contract_for(method):
    return method.optical.OpticalFeatureContract(generic_version='baseline-relative-bright-grain-v5')


def optical_predictions(method, current, baseline, previous, fits, *, chunk_rows=32):
    workspace = method.optical.prepare_optical_feature_workspace(
        current, baseline, previous_rgb=previous, texture_sigma=2., contract=contract_for(method))
    outputs = {head: np.empty(current.shape[:2], np.float32) for head in HEADS}
    predictors = {head: method.learning.SerializedPhaseHead(head, contract_for(method),
        fit.mean, fit.scale, fit.coefficients, fit.intercept, .5) for head, fit in fits.items()}
    for start in range(0, current.shape[0], chunk_rows):
        stop = min(start+chunk_rows, current.shape[0])
        features = method.optical.optical_feature_matrix(workspace, (slice(start, stop), slice(None)))
        for head, predictor in predictors.items():
            outputs[head][start:stop] = predictor.probability(features).reshape(stop-start, current.shape[1])
    return outputs


def extract_optical(dataset, output, method):
    dataset, output = Path(dataset), Path(output)
    plan = read(dataset/'plan.json')
    records = read(dataset/'evidence.json')['regions']
    baseline, baseline_receipt = content_input(dataset, 0)
    owner_by_id = {r['region_id']: i for i, r in enumerate(records)}
    all_features, all_owners, coverage = [], [], []
    for row in plan['frames']:
        index = row['frame_index']
        if not row['region_ids']:
            continue
        started = time.monotonic()
        current, receipt = content_input(dataset, index)
        previous = content_input(dataset, index-1)[0] if index > 0 else None
        workspace = method.optical.prepare_optical_feature_workspace(
            current, baseline, previous_rgb=previous, texture_sigma=2., contract=contract_for(method))
        for region_id in row['region_ids']:
            owner = owner_by_id[region_id]; record = records[owner]
            if record['payload'].get('training_sampling'):
                raise ValueError('BM special sampling requires its original explicit policy')
            shape = current.shape[:2]
            bounds, mask = method.annotation.annotation_window(record['payload'],
                canvas_wh=shape[::-1], content_box=(0, 0, shape[1], shape[0]))
            rr, cc = np.nonzero(mask)
            if len(rr) > 1024:
                seed = int.from_bytes(hashlib.sha256(region_id.encode()).digest()[:8], 'little')
                selected = np.sort(np.random.default_rng(seed).choice(len(rr), 1024, replace=False))
                rr, cc = rr[selected], cc[selected]
            rr += bounds[1]; cc += bounds[0]
            values = method.optical.optical_feature_matrix(workspace, (rr, cc))
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f'Invalid BM optical region features: {region_id}')
            all_features.append(values)
            all_owners.append(np.full(len(values), owner, np.int32))
            coverage.append(dict(region_id=region_id, frame_index=index, training_rows=len(values)))
        del workspace, current, previous
        emit('optical_frame', frame=index, seconds=time.monotonic()-started)
    matrix, owners = np.concatenate(all_features), np.concatenate(all_owners)
    np.savez_compressed(output/'optical_samples.npz', features=matrix, owners=owners)
    write(output/'optical_extraction.json', dict(coverage=coverage, rows=len(matrix),
        samples_sha256=sha256(output/'optical_samples.npz'), feature_contract=contract_for(method).to_dict(),
        texture_sigma=2., maximum_samples_per_region=1024, source_plan_sha256=sha256(dataset/'plan.json'),
        baseline_source=baseline_receipt['source'], previous_policy='preceding original source timeline frame'))
    return matrix, owners, records


def fit_heads(matrix, owners, records, method, *, regional):
    regions = [method.logistic.ReviewedRegion(**r) for r in records]
    seeds, seed_reports = {}, {}
    for head in HEADS:
        emit('seed_fit', kind='regional' if regional else 'optical', head=head, rows=len(matrix))
        seeds[head], seed_reports[head] = method.logistic.fit_weighted_logistic_mil(
            matrix, owners, regions, head=head, runs=None, regularization=.001,
            backend='cpu', max_iter=1500)
    emit('joint_fit', kind='regional' if regional else 'optical', rows=len(matrix))
    fits, report = method.joint.fit_joint_aqueous_mil(matrix, owners, regions,
        seed_fits=seeds, regularization={head: .001 for head in HEADS}, backend='cpu',
        max_iter=2000, tolerance=1e-6 if regional else 1e-7)
    return fits, report, seed_reports


def extract_regional(dataset, output, method, optical_fits):
    dataset, output = Path(dataset), Path(output)
    plan = read(dataset/'plan.json'); records = read(dataset/'evidence.json')['regions']
    owner_by_id = {r['region_id']: i for i, r in enumerate(records)}
    baseline, _ = content_input(dataset, 0)
    domain_path = data_root()/DOMAIN
    with np.load(bound(domain_path, DOMAIN_SHA)) as source:
        domain = source['domain'].astype(bool)
        spacing = (abs(float(source['y_m'][1]-source['y_m'][0])), abs(float(source['x_m'][1]-source['x_m'][0])))
    native = dict(canvas_wh=list(domain.shape[::-1]), content_box_xywh=[0, 0, *domain.shape[::-1]],
                  analysis_image_shape_hw=list(domain.shape))
    all_features, all_owners, coverage = [], [], []
    for row in plan['frames']:
        index = row['frame_index']
        if not row['region_ids']:
            continue
        started = time.monotonic()
        current, _ = content_input(dataset, index)
        previous = content_input(dataset, index-1)[0] if index > 0 else None
        scores = optical_predictions(method, current, baseline, previous, optical_fits)
        prepared = method.regional.prepare(current, baseline, scores[HEADS[0]], scores[HEADS[1]], domain, spacing)
        selected = [(owner_by_id[rid], records[owner_by_id[rid]]) for rid in row['region_ids']]
        features, owners, rows = method.sampling.region_samples(prepared, selected, native, native_domain=domain)
        all_features.append(features); all_owners.append(owners); coverage.extend(rows)
        del scores, prepared, current, previous
        emit('regional_frame', frame=index, rows=len(features), seconds=time.monotonic()-started)
    matrix, owners = np.concatenate(all_features), np.concatenate(all_owners)
    np.savez_compressed(output/'regional_samples.npz', features=matrix, owners=owners)
    write(output/'regional_extraction.json', dict(coverage=coverage, rows=len(matrix),
        samples_sha256=sha256(output/'regional_samples.npz'), spacing_m=list(spacing),
        settings=asdict(method.regional.Settings()), active_columns=REGIONAL_COLUMNS.tolist(),
        domain=dict(path=str(domain_path), sha256=DOMAIN_SHA), maximum_samples_per_region=512,
        small_region_policy='native-points-with-interpolated-regional-features'))
    return matrix[:, REGIONAL_COLUMNS], owners, records


def train(dataset, output, snapshot, *, stage='all'):
    """Train fresh optical/regional heads; retain failed convergence checkpoints."""
    dataset, output = Path(dataset), Path(output)
    if stage not in ('all', 'optical', 'regional'):
        raise ValueError('Unknown BM training stage')
    output.mkdir(parents=True, exist_ok=True)
    ready = read(dataset/'prepared.json')
    bound(dataset/'plan.json', ready['plan_sha256'])
    method = load_numerics(snapshot)
    if stage in ('all', 'optical'):
        matrix, owners, records = extract_optical(dataset, output, method)
        fits, report, seeds = fit_heads(matrix, owners, records, method, regional=False)
        write(output/'optical_model.json', dict(feature_contract=contract_for(method).to_dict(), texture_sigma=2.,
            heads={h: packed(f) for h, f in fits.items()}, fit_report=report, seed_reports=seeds,
            frozen_heads=[], input_contract=read(dataset/'plan.json')['input_contract'],
            inference_backend='source-bound SerializedPhaseHead CPU64'))
        if not report['converged']:
            raise ValueError('BM optical fit did not meet the original gradient requirement')
        del matrix, owners
    if stage in ('all', 'regional'):
        optical = read(output/'optical_model.json')
        if not optical['fit_report']['converged']:
            raise ValueError('A converged fresh BM optical fit is required')
        fits = {h: unpack(method.logistic, v) for h, v in optical['heads'].items()}
        matrix, owners, records = extract_regional(dataset, output, method, fits)
        fits, report, seeds = fit_heads(matrix, owners, records, method, regional=True)
        write(output/'regional_model.json', dict(format=method.regional.VERSION,
            feature_names=list(method.regional.FEATURE_NAMES), settings=asdict(method.regional.Settings()),
            heads={h: packed(f, regional=True) for h, f in fits.items()}, fit_report=report, seed_reports=seeds,
            active_columns=REGIONAL_COLUMNS.tolist(), frozen_heads=[], scientific_acceptance=False))
        if not report['converged']:
            raise ValueError('BM regional fit did not meet the original gradient requirement')
        write(output/'training_complete.json', dict(series='FF_BM', active_regions=713,
            refitted_heads=['optical/'+h for h in HEADS]+['regional/'+h for h in HEADS], frozen_heads=[],
            evaluation_scope='training-only; no independent generalisation claim',
            inference_entrypoint='darsia.fluidflower.phase.bm_raw_model',
            optical_sha256=sha256(output/'optical_model.json'), regional_sha256=sha256(output/'regional_model.json'),
            numerical_source_binding=method.source_binding))
    emit('training_stage_complete', stage_name=stage)
