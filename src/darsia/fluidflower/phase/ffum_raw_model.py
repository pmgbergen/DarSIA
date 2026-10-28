"""Inference for the bound FF_UM full-sensor RAW model and its physical features."""
import argparse
from dataclasses import fields
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .ffum_raw_input import read_um_full_frame
from .ffum_raw_numerics import load_numerics
from .ffum_review_mapping import UmReviewMapping
from .train_ffum_raw_model import (HEADS, bound, content_input, optical_predictions,
                                   read, sha, unpack, write)
from .um_bm_review_regions import capture_legacy_drift
from .um_bm_sampling import project_to_full_frame, sample_full_frame


def load_bundle(root):
    root = Path(root).resolve(); manifest = read(root/'model.json')
    if (manifest.get('format') != 'ffum-full-raw-phase-v1' or manifest.get('series') != 'FF_UM'
            or manifest.get('frozen_heads') != []):
        raise ValueError('A refitted FF_UM RAW model is required')
    if Path(__file__).resolve() != root/'runtime/source/darsia/fluidflower/phase/ffum_raw_model.py':
        raise ValueError('Use the UM model-bound runtime/launcher, not a mutable installed reader')
    def asset(key):
        descriptor = manifest['assets'][key]; path = (root/descriptor['path']).resolve()
        if not path.is_relative_to(root): raise ValueError('UM model asset escaped its bundle')
        return bound(path, descriptor['sha256'])
    optical, regional = read(asset('optical_generic')), read(asset('regional'))
    if not all(m['fit_report']['converged'] for m in (optical, regional)):
        raise ValueError('UM inference requires converged optical and regional fits')
    method = load_numerics(asset('numerics_regional').parent)
    prediction_sha = method.prediction_source_binding['source_learning.py']['sha256']
    if (any(m['optical_prediction_source_sha256'] != prediction_sha for m in (optical, regional))
            or regional['numerical_source_sha256'] != manifest['assets']['numerics_regional']['sha256']
            or optical['numerical_source_sha256'] != manifest['assets']['numerics_optical']['sha256']):
        raise ValueError('UM inference numerical sources differ from the fitted model')
    return manifest, method, optical, regional, asset


def _settings(module, specification):
    names = {field.name for field in fields(module.Settings)}
    if specification.get('format') != module.VERSION:
        raise ValueError('UM original output policy differs from bound runtime')
    return module.Settings(**{k: v for k, v in specification.items() if k in names})


def predict_content(method, current, baseline, previous, domain, spacing, optical, regional, policy):
    fits = {h: unpack(method.logistic, optical['heads'][h]) for h in HEADS}
    scores = optical_predictions(method, current, baseline, previous, fits)
    p, g = scores[HEADS[0]], scores[HEADS[1]]
    raw = np.where(p >= .5, np.where(g >= .6, 2, 1), 0).astype(np.uint8)
    prepared = method.regional.prepare(current, baseline, p, g, domain, spacing,
                                       method.regional.Settings(**regional['settings']))
    labels, report = method.regional.predict(prepared, regional, raw, domain)
    labels, gas_rejected = method.gas.confirm_gas(labels,
        method.regional.probability(prepared['features'], regional['heads'][HEADS[1]]), domain,
        _settings(method.gas, policy['regional_gas']))
    co2 = np.where(labels > 0, np.maximum(p, .5), np.minimum(p, .49)).astype(np.float32)
    gas = np.where(labels == 1, np.minimum(g, .4), g).astype(np.float32)
    guarded, guard_report = method.guards.guard_labels(labels, domain, spacing,
        _settings(method.guards, policy['output_guards']))
    rejected = ~domain | ((labels > 0) & (guarded == 0))
    co2[rejected] = 0.; gas[rejected] = 0.
    cleaned, clean_report = method.freegas.clean_labels(guarded, spacing,
        _settings(method.freegas, policy['free_gas_cleanup']))
    gas[(guarded == 2) & (cleaned == 1)] = 0.
    return np.stack((co2, gas), axis=-1), cleaned, dict(regional=report,
        gas_rejected_pixels=int(gas_rejected.sum()), output_guards=guard_report, free_gas_cleanup=clean_report)


def mapper_from_receipt(receipt, curvature):
    with np.load(curvature) as source:
        mapper = UmReviewMapping(source['grid'], tuple(map(int, source['shape'])),
            np.asarray(receipt['drift_matrix']), tuple(receipt['photo_wh']), tuple(receipt['resized_wh']),
            tuple(receipt['full_raw_canvas_origin_jpg_xy']))
    mapper.validate_all_cells()
    return mapper


def bound_output_scores(values, support):
    """Reject invalid predictions and remove only float32 interpolation roundoff."""
    tolerance = 4 * np.finfo(np.float32).eps
    count, low, high = 0, 1., 0.
    for start in range(0, values.shape[0], 128):
        block = values[start:start+128]
        selected = block[support[start:start+128]]
        if not selected.size:
            continue
        if not np.isfinite(selected).all():
            raise ValueError('Non-finite supported UM classification score')
        minimum, maximum = float(selected.min()), float(selected.max())
        if minimum < -tolerance or maximum > 1. + tolerance:
            raise ValueError('UM classification score exceeds probability bounds')
        count += int(((selected < 0.) | (selected > 1.)).sum())
        low, high = min(low, minimum), max(high, maximum)
        np.clip(block, 0., 1., out=block)
    return dict(minimum_before=low, maximum_before=high,
                corrected_float32_roundoff_elements=count, tolerance=float(tolerance),
                score_semantics='CO2 presence and conditional gas, not additive phase fractions')


def render_result(root, context_key, current, previous, receipt, raw_rgb, raw_support, output):
    root, output = Path(root), Path(output)
    manifest, method, optical, regional, asset = load_bundle(root)
    context = manifest['contexts'][context_key]
    bound(root/'inputs/metric'/context['baseline_job']/'receipt.json', manifest['frame_receipts'][context['baseline_job']])
    baseline, domain, base_receipt = content_input(root/'inputs', context['baseline_job'])
    if (receipt['profile_sha256'] != manifest['assets']['profile']['sha256']
            or receipt['full_raw_canvas_origin_jpg_xy'] != manifest['canvas_origin_jpg_xy']
            or receipt['source_job']['photometry_sha256'] != manifest['assets'][context['photometry']]['sha256']
            or receipt['correction_sources'] != base_receipt['correction_sources']):
        raise ValueError('UM inference profile, camera table or geometry differs from training')
    if (raw_rgb.shape != (*manifest['canvas_shape_hw'], 3)
            or raw_support.shape != tuple(manifest['canvas_shape_hw']) or raw_support.dtype != bool):
        raise ValueError('UM inference no longer has its bound full RAW canvas')
    scores, labels, report = predict_content(method, current, baseline, previous, domain,
        receipt['content_spacing_yx_m'], optical, regional, manifest['output_policy'])
    mapper = mapper_from_receipt(receipt, asset(context['curvature']))
    coordinates = {}
    def mapped_once(points):
        points = np.ascontiguousarray(points)
        key = (points.shape, points.dtype.str, hashlib.sha256(memoryview(points)).digest())
        if key not in coordinates:
            coordinates[key] = mapper.display_to_content(points)
        return coordinates[key]
    # Scores and categorical labels use identical output pixel centres.
    # Reuse their exact float64 inverse coordinates; interpolation still differs.
    projected = project_to_full_frame(scores, domain, mapped_once, domain.shape, raw_support)
    score_bounds = bound_output_scores(projected.values, projected.valid_support)
    categories = project_to_full_frame(labels, domain, mapped_once, domain.shape,
                                       raw_support, interpolation='nearest')
    coordinates.clear()
    valid = categories.valid_support
    full_labels = np.full(raw_support.shape, 255, np.uint8)
    full_labels[valid] = categories.values[valid].astype(np.uint8)
    if not np.isin(full_labels, (0, 1, 2, 255)).all(): raise ValueError('Invalid UM phase code')
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/'classification_scores.npy', projected.values)
    np.save(output/'score_support.npy', projected.valid_support)
    np.save(output/'labels.npy', full_labels); np.save(output/'valid_support.npy', valid)
    # Downsample the already-classified overlay only for viewing; unknown
    # pixels keep the original full-field photograph, with no phase colour.
    overlay = raw_rgb.astype(np.float32)
    colours = {1: np.array([0, 190, 255]), 2: np.array([255, 170, 20])}
    for code, colour in colours.items():
        selected = full_labels == code
        overlay[selected] = .58*overlay[selected]+.42*colour
    preview = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
    preview.thumbnail((1440, 1000), Image.Resampling.LANCZOS)
    preview.save(output/'raw_phase_overlay.png')
    report.update(model_sha256=sha(root/'model.json'), context=context_key,
        source=receipt['source_job'], full_shape_hw=list(full_labels.shape),
        phase_codes={'0': 'water', '1': 'aqueous', '2': 'gas', '255': 'unknown'},
        pixels_by_code={str(k): int((full_labels == k).sum()) for k in (0, 1, 2, 255)},
        supported_pixels=int(valid.sum()), score_support_pixels=int(projected.valid_support.sum()),
        labels_interpolation='nearest', scores_interpolation='linear',
        classification_score_bounds=score_bounds,
        classification_scores_not_concentration=True, crop_applied=False,
        feature_metric='historical physical coordinates, each temporal frame separately mapped',
        optical_geometry='relative RAW-to-JPG R-only alignment, not absolute optical calibration',
        files={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    write(output/'inference.json', report)
    return report


def predict_prepared_example(model_root, output):
    root = Path(model_root); manifest, _, _, _, asset = load_bundle(root)
    example = manifest['example']; dataset = root/'inputs'
    for job in (example['current'], example['previous']):
        if job: bound(dataset/'metric'/job/'receipt.json', manifest['frame_receipts'][job])
    current, _, receipt = content_input(dataset, example['current'])
    previous = content_input(dataset, example['previous'])[0] if example['previous'] else None
    return render_result(root, example['context'], current, previous, receipt,
        np.asarray(Image.open(asset('example_rgb')).convert('RGB')),
        np.load(asset('example_support')), output)


def validate_temporal_request(previous_preparation, previous_receipt_sha256,
                              first_temporal_frame, temporal_context_only):
    if temporal_context_only:
        if previous_preparation is not None or previous_receipt_sha256 or first_temporal_frame:
            raise ValueError('A context-only UM input has no prediction history or first-frame claim')
    else:
        if (previous_preparation is None) != bool(first_temporal_frame):
            raise ValueError('Supply the previous prepared frame or explicitly mark the first temporal frame')
        if previous_preparation is not None and not previous_receipt_sha256:
            raise ValueError('The previous UM temporal receipt must be SHA-bound')


def prepare_new_raw_frame(model_root, raw_path, photo_path, output, *, run,
                          raw_sha256, photo_sha256, previous_preparation=None,
                          previous_receipt_sha256=None, first_temporal_frame=False,
                          temporal_context_only=False):
    """Prepare a real new RAW/JPG pair using the series-bound camera/rig setup.

    Original JPG supplies only measured historical drift. Pixel values come
    solely from the complete RAW field. No per-image camera-profile choice.
    """
    import darsia
    validate_temporal_request(previous_preparation, previous_receipt_sha256,
                              first_temporal_frame, temporal_context_only)
    root, output = Path(model_root), Path(output)
    manifest, _, _, _, asset = load_bundle(root)
    context_key = manifest['run_contexts'][run]; context = manifest['contexts'][context_key]
    raw = bound(raw_path, raw_sha256); photo = bound(photo_path, photo_sha256)
    if raw.stem != photo.stem or raw.suffix.upper() != '.NEF' or photo.suffix.upper() != '.JPG':
        raise ValueError('UM requires the original matching Nikon NEF/JPG exposure pair')
    previous_binding = None
    if previous_preparation is not None:
        previous = read(bound(Path(previous_preparation)/'preparation.json', previous_receipt_sha256))
        if previous['model_sha256'] != sha(root/'model.json') or previous['context'] != context_key:
            raise ValueError('UM previous frame belongs to another model or physical context')
        previous_binding = dict(path=str(Path(previous_preparation).resolve()), sha256=previous_receipt_sha256)
    full = read_um_full_frame(raw, raw_sha256=raw_sha256, profile_path=asset('profile'),
        profile_sha256=manifest['assets']['profile']['sha256'],
        photometry_path=asset(context['photometry']),
        photometry_sha256=manifest['assets'][context['photometry']]['sha256'])
    resize = darsia.ResizeCorrection(); resize.load(asset(context['resize']))
    drift = darsia.DriftCorrection(); drift.load(asset(context['drift']))
    old = darsia.imread(photo).img; resized = resize.correct_array(old)
    cv2.setRNGSeed(0); _, matrix = capture_legacy_drift(drift, resized)
    bound(root/'inputs/metric'/context['baseline_job']/'receipt.json', manifest['frame_receipts'][context['baseline_job']])
    _, domain, baseline_receipt = content_input(root/'inputs', context['baseline_job'])
    receipt = dict(baseline_receipt, source_job=dict(run=run, photo=str(photo), photo_sha256=photo_sha256,
        raw=str(raw), raw_sha256=raw_sha256, photometry_sha256=manifest['assets'][context['photometry']]['sha256']),
        drift_matrix=matrix.tolist(), photo_wh=[old.shape[1], old.shape[0]],
        resized_wh=[resized.shape[1], resized.shape[0]], profile_sha256=full.metadata['profile_sha256'],
        full_raw_canvas_origin_jpg_xy=full.metadata['canvas_origin_jpg_xy'])
    mapper = mapper_from_receipt(receipt, asset(context['curvature']))
    sampled = sample_full_frame(full.rgb, full.valid_support, mapper.content_to_display, domain.shape)
    output.mkdir(parents=True, exist_ok=False); folder = output/'metric'/'current'; folder.mkdir(parents=True)
    for name, value in (('rgb_unit', sampled.values/np.float32(255)), ('source_support', sampled.valid_support),
                        ('training_domain', domain & sampled.valid_support)):
        np.save(folder/(name+'.npy'), value); receipt[name+'_sha256'] = sha(folder/(name+'.npy'))
    write(folder/'receipt.json', receipt)
    Image.fromarray(full.rgb).save(output/'full_sensor.png'); np.save(output/'raw_support.npy', full.valid_support)
    write(output/'preparation.json', dict(model_sha256=sha(root/'model.json'), context=context_key,
        previous=previous_binding, first_temporal_frame=bool(first_temporal_frame),
        temporal_context_only=bool(temporal_context_only),
        receipt_sha256=sha(folder/'receipt.json'), full_sensor_sha256=sha(output/'full_sensor.png'),
        raw_support_sha256=sha(output/'raw_support.npy')))
    return output/'preparation.json'


def predict_new_raw_frame(model_root, preparation, output):
    root, preparation = Path(model_root), Path(preparation)
    record = read(preparation/'preparation.json')
    if record.get('temporal_context_only', False):
        raise ValueError('A context-only UM record cannot be predicted without its own temporal history')
    if record['model_sha256'] != sha(root/'model.json'): raise ValueError('UM model revision changed')
    bound(preparation/'metric/current/receipt.json', record['receipt_sha256'])
    current, _, receipt = content_input(preparation, 'current'); previous = None
    if record['previous']:
        previous_root = Path(record['previous']['path'])
        p = read(bound(previous_root/'preparation.json', record['previous']['sha256']))
        bound(previous_root/'metric/current/receipt.json', p['receipt_sha256'])
        previous = content_input(previous_root, 'current')[0]
    return render_result(root, record['context'], current, previous, receipt,
        np.asarray(Image.open(bound(preparation/'full_sensor.png', record['full_sensor_sha256'])).convert('RGB')),
        np.load(bound(preparation/'raw_support.npy', record['raw_support_sha256'])), output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument('--preparation', type=Path)
    source.add_argument('--raw', type=Path)
    p.add_argument('--photo', type=Path); p.add_argument('--run')
    p.add_argument('--raw-sha256'); p.add_argument('--photo-sha256')
    p.add_argument('--previous-preparation', type=Path); p.add_argument('--previous-receipt-sha256')
    p.add_argument('--first-temporal-frame', action='store_true')
    args = p.parse_args()
    preparation = args.preparation
    if args.raw:
        if not all((args.photo, args.run, args.raw_sha256, args.photo_sha256)):
            p.error('New UM input requires --photo, --run and both original source SHA256 values')
        preparation = args.output.with_name(args.output.name+'_preparation')
        prepare_new_raw_frame(args.model, args.raw, args.photo, preparation, run=args.run,
            raw_sha256=args.raw_sha256, photo_sha256=args.photo_sha256,
            previous_preparation=args.previous_preparation,
            previous_receipt_sha256=args.previous_receipt_sha256,
            first_temporal_frame=args.first_temporal_frame)
    result = (predict_new_raw_frame(args.model, preparation, args.output) if preparation
              else predict_prepared_example(args.model, args.output))
    print(result['full_shape_hw'], result['pixels_by_code'], flush=True)


if __name__ == '__main__': main()
