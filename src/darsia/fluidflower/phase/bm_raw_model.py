"""Explicit FF_BM RAW inference with the same inputs used by its four fitted heads.

This entrypoint accepts the model's bound prepared RAW timeline. It never opens
historical review JPEGs or selects the generic cropped-image inference route.
The calibrated rig representation is internal; output arrays use the full RAW
canvas and carry explicit support outside the historical physical domain.
"""
import json
import importlib.util
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

from ..native.raw_full_display import RawDisplayMapping
from .bm_raw_numerics import load_numerics
from .bm_raw_regions import BMReviewMapping, CurvatureInverse
from .prepare_bm_raw_training import bound, write, sha256, prepare_frame
from .train_bm_raw_model import HEADS, content_input, optical_predictions, read, unpack
from .um_bm_sampling import project_to_full_frame


def bound_projected_scores(values, valid_support, *, chunk_rows=128):
    """Bound only float32 interpolation roundoff, rejecting larger errors.

    The allowance is four float32 ULPs at one, the probability range's upper
    bound. Validate every supported sample before changing any output value.
    Unsupported NaNs and categorical labels are left unchanged.
    """
    values, support = np.asarray(values), np.asarray(valid_support)
    if (values.dtype != np.float32 or values.ndim != 3 or support.dtype != bool
            or support.shape != values.shape[:2] or chunk_rows < 1):
        raise ValueError('Projected BM scores require float32 HWC and boolean support')
    limit = 4*float(np.spacing(np.float32(1)))
    count, maximum = 0, 0.
    for y in range(0, len(values), chunk_rows):
        block = values[y:y+chunk_rows][support[y:y+chunk_rows]]
        if not np.isfinite(block).all():
            raise ValueError('Projected BM probabilities contain non-finite supported values')
        if block.size:
            excess = max(0., -float(block.min()), float(block.max())-1.)
            if excess > limit:
                raise ValueError('Projected BM probabilities exceed float32 roundoff allowance')
            maximum = max(maximum, excess)
            count += int(np.count_nonzero((block < 0) | (block > 1)))
    if count:
        for y in range(0, len(values), chunk_rows):
            block, valid = values[y:y+chunk_rows], support[y:y+chunk_rows]
            block[valid] = np.clip(block[valid], 0., 1.)
    return dict(policy='float32-interpolation-roundoff-only', clipped_score_elements=count,
                maximum_range_excess=maximum, permitted_range_excess=limit)


def predict_content(method, current, baseline, previous, optical, regional, domain, spacing):
    """Apply the preserved optical, regional, gas and island operations in order."""
    fits = {h: unpack(method.logistic, f) for h, f in optical['heads'].items()}
    scores = optical_predictions(method, current, baseline, previous, fits)
    p, g = scores[HEADS[0]], scores[HEADS[1]]
    raw = np.where(p >= .5, np.where(g >= .6, 2, 1), 0).astype(np.uint8)
    prepared = method.regional.prepare(current, baseline, p, g, domain, spacing)
    labels, report = method.regional.predict(prepared, regional, raw, domain)
    labels, gas_rejected = method.gas.confirm_gas(labels,
        method.regional.probability(prepared['features'], regional['heads'][HEADS[1]]), domain)
    co2 = np.where(labels > 0, np.maximum(p, .5), np.minimum(p, .49)).astype(np.float32)
    gas = np.where(labels == 1, np.minimum(g, .4), g).astype(np.float32)
    guarded, guard_report = method.guards.guard_labels(labels, domain, spacing)
    rejected = ~domain | ((labels > 0) & (guarded == 0))
    co2[rejected] = 0.; gas[rejected] = 0.
    cleaned, cleanup_report = method.freegas.clean_labels(guarded, spacing)
    gas[(guarded == 2) & (cleaned == 1)] = 0.
    fields = np.stack((co2, gas, 1-co2, co2*(1-gas), co2*gas), axis=-1)
    return fields, cleaned, dict(regional=report, regional_gas_rejected_pixels=int(gas_rejected.sum()),
        output_guards=guard_report, free_gas_cleanup=cleanup_report)


def load_bundle(root):
    root = Path(root)
    manifest = read(root/'model.json')
    if manifest.get('format') != 'ffbm-full-raw-phase-v1' or manifest.get('frozen_heads') != []:
        raise ValueError('A four-head refitted FF_BM full-RAW model is required')
    def asset(name):
        descriptor = manifest['assets'][name]
        path = (root/descriptor['path']).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError('BM model asset escapes its bundle')
        return bound(path, descriptor['sha256'])
    optical, regional = read(asset('optical')), read(asset('regional'))
    if not all(m['fit_report']['converged'] for m in (optical, regional)):
        raise ValueError('BM inference requires converged optical and regional fits')
    # The path is relative to the durable artifact, not its former local cache.
    bound(root/'runtime/numerics/source_binding.json', manifest['numerics_binding_sha256'])
    method = load_numerics(root/'runtime/numerics')
    return manifest, method, optical, regional, asset


def _legacy_drift_factory(root):
    """Bind preserved legacy feature/drift code without changing global DarSIA."""
    import darsia
    root = Path(root)
    manifest = read(root/'model.json')
    bindings = read(bound(root/'runtime_binding.json', manifest['runtime_binding_sha256']))
    modules = {}
    for name in ('legacy_features', 'legacy_translation', 'legacy_drift'):
        relative = 'runtime/legacy/'+name+'.py'
        path = bound(root/relative, bindings[relative])
        spec = importlib.util.spec_from_file_location('_bm_bound_'+name, path)
        module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
        spec.loader.exec_module(module); modules[name] = module
    translation = modules['legacy_translation']
    translation.FeatureDetection = modules['legacy_features'].FeatureDetection
    namespace = dict(vars(darsia)); namespace['TranslationEstimator'] = translation.TranslationEstimator
    modules['legacy_drift'].darsia = SimpleNamespace(**namespace)
    return modules['legacy_drift'].DriftCorrection


def prepare_new_raw_frame(model_root, raw_path, photo_path, output, *, expected_raw_sha256,
                          expected_photo_sha256, elapsed_h, previous_prepared_frame,
                          previous_receipt_sha256=None):
    """Prepare a new C1 exposure with explicit source and temporal registration.

    The paired original TIFF supplies the actual historical drift measurement;
    the new RAW supplies its own Sony tables. ``previous_prepared_frame`` is the
    directory containing the preceding frame's receipt and registered arrays.
    It is required after time zero. No timestamp/image matching is guessed.
    """
    if not np.isfinite(elapsed_h) or elapsed_h < 0:
        raise ValueError('An explicit nonnegative C1 elapsed time is required')
    if elapsed_h > 0 and previous_prepared_frame is None:
        raise ValueError('Supply the preceding prepared RAW frame for dynamic features')
    if previous_prepared_frame is not None and not previous_receipt_sha256:
        raise ValueError('The preceding frame receipt must be explicitly SHA-bound')
    root, output = Path(model_root), Path(output)
    manifest, _, _, _, asset = load_bundle(root)
    raw = bound(raw_path, expected_raw_sha256); photo = bound(photo_path, expected_photo_sha256)
    if photo.stem.split('_')[-1] != raw.stem or raw.suffix.upper() != '.ARW':
        raise ValueError('An original BM TIFF/ARW pair with the same camera frame ID is required')
    plan = read(bound(root/'inputs/plan.json', manifest['input_plan_sha256']))
    baseline = next(r for r in plan['frames'] if r['frame_index'] == 0)
    row = dict(frame_index=1, raw=dict(path=str(raw), sha256=expected_raw_sha256),
        photo=dict(path=str(photo), sha256=expected_photo_sha256), region_ids=[], elapsed_h=float(elapsed_h))
    plan['frames'] = [baseline, row]
    for key in ('config', 'curvature', 'registration'):
        path = asset(key)
        plan[key] = {**plan[key], 'path': str(path), 'sha256': sha256(path)}
    path = asset('profile_template')
    plan['profile'] = dict(path=str(path), sha256=sha256(path))
    output.mkdir(parents=True, exist_ok=False)
    write(output/'plan.json', plan)
    previous = None
    if previous_prepared_frame is not None:
        folder = Path(previous_prepared_frame)
        record = read(bound(folder/'receipt.json', previous_receipt_sha256))
        if record['photometry'] != manifest['photometry'] or record['curvature']['sha256'] != plan['curvature']['sha256']:
            raise ValueError('Previous RAW frame does not share this BM feature representation')
        target = output/'previous/frame_000'; target.mkdir(parents=True)
        for name in ('content_rgb.npy', 'content_valid_support.npy'):
            source = bound(folder/name, record['arrays'][name]['sha256'])
            shutil.copy2(source, target/name)
            bound(target/name, record['arrays'][name]['sha256'])
        shutil.copy2(folder/'receipt.json', target/'receipt.json')
        previous = dict(path='previous', receipt_sha256=previous_receipt_sha256, source=record['source'])
    receipt = prepare_frame(output, 1, drift_factory=_legacy_drift_factory(root))
    write(output/'inference_context.json', dict(model_sha256=sha256(root/'model.json'),
        current_frame_index=1, current_receipt_sha256=sha256(output/'frame_001/receipt.json'),
        elapsed_h=float(elapsed_h), previous=previous,
        baseline_frame_index=0, dynamic_policy='explicit preceding prepared RAW frame' if previous else 'first frame: unavailable'))
    return receipt


def predict_prepared_frame(model_root, frame_index, output):
    """Run a bound RAW frame and return full-canvas fields with supported pixels.

    A new source frame must first receive its own measured drift, Sony tables,
    RAW sampling and registered feature input through the BM preparation API.
    Existing source timeline provenance remains immutable.
    """
    root, output = Path(model_root), Path(output)
    manifest, method, optical, regional, asset = load_bundle(root)
    if frame_index not in manifest['prepared_frames']:
        raise ValueError('Frame has no bound full-RAW preparation in this model')
    if frame_index not in manifest['inferable_frames']:
        raise ValueError('This support frame lacks its preceding RAW context; use explicit new-frame preparation')
    dataset = root/'inputs'
    for index in {0, frame_index, max(0, frame_index-1)}:
        relative = f'inputs/frame_{index:03d}/receipt.json'
        bound(root/relative, manifest['frame_receipts'][str(index)])
    current, receipt = content_input(dataset, frame_index)
    baseline, _ = content_input(dataset, 0)
    previous = content_input(dataset, frame_index-1)[0] if frame_index > 0 else None
    folder = dataset/f'frame_{frame_index:03d}'
    bound(folder/'profile.json', receipt['full_raw']['profile_document']['sha256'])
    if receipt['photometry'] != manifest['photometry']:
        raise ValueError('BM inference photometry differs from training')
    raw_support = np.load(bound(folder/'raw_valid_support.npy', receipt['arrays']['raw_valid_support.npy']['sha256']), mmap_mode='r')
    return _render_prediction(root, output, method, optical, regional, asset, current,
                              baseline, previous, receipt, raw_support, frame_index)


def predict_new_raw_frame(model_root, prepared_context, output):
    """Infer a context produced by prepare_new_raw_frame, including new times."""
    root, context_root = Path(model_root), Path(prepared_context)
    manifest, method, optical, regional, asset = load_bundle(root)
    context = read(context_root/'inference_context.json')
    if context['model_sha256'] != sha256(root/'model.json'):
        raise ValueError('New RAW preparation belongs to a different BM model')
    bound(context_root/'frame_001/receipt.json', context['current_receipt_sha256'])
    current, receipt = content_input(context_root, 1)
    bound(root/'inputs/frame_000/receipt.json', manifest['frame_receipts']['0'])
    baseline, _ = content_input(root/'inputs', 0)
    previous = None
    if context['previous']:
        previous_root = context_root/'previous'
        bound(previous_root/'frame_000/receipt.json', context['previous']['receipt_sha256'])
        previous, _ = content_input(previous_root, 0)
    if receipt['photometry'] != manifest['photometry']:
        raise ValueError('New RAW photometry differs from training')
    bound(context_root/'frame_001/profile.json', receipt['full_raw']['profile_document']['sha256'])
    raw_support = np.load(bound(context_root/'frame_001/raw_valid_support.npy',
        receipt['arrays']['raw_valid_support.npy']['sha256']), mmap_mode='r')
    return _render_prediction(root, Path(output), method, optical, regional, asset, current,
                              baseline, previous, receipt, raw_support, receipt['frame_index'])


def _render_prediction(root, output, method, optical, regional, asset, current,
                       baseline, previous, receipt, raw_support, frame_index):
    with np.load(asset('domain')) as source:
        domain = source['domain'].astype(bool)
        spacing = (abs(float(source['y_m'][1]-source['y_m'][0])), abs(float(source['x_m'][1]-source['x_m'][0])))
    fields, content_labels, report = predict_content(method, current, baseline, previous, optical, regional, domain, spacing)
    inverse = CurvatureInverse(np.load(asset('curvature'), allow_pickle=True).item())
    mapping = BMReviewMapping(RawDisplayMapping(**receipt['raw_mapping']), inverse,
        np.asarray(receipt['drift_matrix']), tuple(receipt['registration']['tiff_to_sensor_xy']),
        tuple(receipt['photo_wh']), tuple(receipt['resized_wh']))
    projected = project_to_full_frame(fields, domain, mapping, domain.shape, raw_support)
    # Categorical phase IDs use nearest sampling. Linear mixing of codes would
    # invent aqueous labels between gas and water. Scores remain continuous.
    projected_labels = project_to_full_frame(content_labels, domain, mapping,
        domain.shape, raw_support, interpolation='nearest')
    projected.valid_support[:] &= projected_labels.valid_support
    projected.values[~projected.valid_support] = np.nan
    roundoff = bound_projected_scores(projected.values, projected.valid_support)
    # Unsupported outer RAW pixels remain 255, never asserted to be water.
    labels = np.full(raw_support.shape, 255, np.uint8)
    labels[projected.valid_support] = projected_labels.values[projected.valid_support].astype(np.uint8)
    output.mkdir(parents=True, exist_ok=False)
    np.save(output/'classification_scores.npy', projected.values)
    np.save(output/'valid_support.npy', projected.valid_support)
    np.save(output/'labels.npy', labels)
    np.save(output/'content_labels.npy', content_labels)
    report.update(frame_index=frame_index, model_sha256=sha256(root/'model.json'),
        projection_roundoff=roundoff,
        source=receipt['source'], raw_mapping=receipt['raw_mapping'], full_shape_hw=list(labels.shape),
        supported_pixels=int(projected.valid_support.sum()), unknown_label=255,
        field_names=['co2_presence_probability', 'gas_probability_conditional_on_co2',
                     'untouched_water_probability', 'aqueous_co2_probability', 'free_co2_gas_probability'],
        classification_scores_not_concentration=True,
        physical_support='historically calibrated domain projected onto uncropped RAW',
        masks_and_images_retained_full_frame=True,
        outputs={p.name: sha256(p) for p in output.glob('*.npy')})
    write(output/'receipt.json', report)
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--frame', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(predict_prepared_frame(args.model, args.frame, args.output)), flush=True)
