"""Extract and refit the existing UM heads from full-field RAW preparations.

Historical physical coordinates are an internal feature grid, not a crop of
the source product. Every temporal input has its own measured registration.
"""
import argparse
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .ffum_raw_numerics import load_numerics, sha

HEADS = ('co2_presence', 'gas_within_co2')
ADAPTERS = ('generic', 'btb-ratiometric-relative-v2')
REGIONAL_COLUMNS = np.array([4, 5, 6, 7, 8, 14, 15, 16, 17, 18])


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    temporary.replace(path)


def bound(path, expected):
    path = Path(path)
    if sha(path) != expected:
        raise ValueError('UM asset checksum changed: '+str(path))
    return path


def emit(stage, **data):
    print(json.dumps(dict(stage=stage, **data)), flush=True)


def optical_record(row):
    source = row['original_inventory']
    return dict(region_id=row['region_id'], run=source['run'], dataset=source['dataset'],
        frame_index=int(source['frame_index']), elapsed_h=float(source['elapsed_h']),
        phase_class=row['phase_class'], confidence=source['confidence'], source=source['source'],
        # The original solver schema predates a top-level reviewer_id field.
        # Keep that provenance in the unmodified annotation payload.
        reviewer=source['reviewer'], payload=row['origin']['original_annotation'])


def training_records(dataset, kind):
    plan = read(Path(dataset)/'dataset_plan.json')
    if kind == 'optical':
        return [optical_record(r) for r in read(bound(plan['optical_cohort'], plan['optical_cohort_sha256']))['regions']]
    model = Path(plan['model']); protocol = read(model/'regional_training_protocol.json')
    return read(bound(model/'evidence.json', protocol['evidence_sha256']))['regions']


def content_input(dataset, job_id, *, plan=None):
    """Resolve prepared arrays relative to dataset and verify every saved array."""
    folder = Path(dataset)/'metric'/job_id
    receipt = read(folder/'receipt.json')
    if plan is not None:
        job = next(j for j in plan['jobs'] if j['job_id'] == job_id)
        if (receipt['source_job'] != job or receipt['profile_sha256'] != plan['profile_sha256']
                or receipt['full_raw_canvas_origin_jpg_xy'] != plan['full_raw_canvas_origin_jpg_xy']):
            raise ValueError('UM prepared input differs from the bound source/profile')
    arrays = {}
    for name in ('rgb_unit', 'source_support', 'training_domain'):
        arrays[name] = np.load(bound(folder/(name+'.npy'), receipt[name+'_sha256']), mmap_mode='r')
    rgb, support, domain = (arrays[n] for n in ('rgb_unit', 'source_support', 'training_domain'))
    tolerance = 4*np.finfo(np.float32).eps
    if (rgb.shape != (*support.shape, 3) or support.dtype != bool or domain.dtype != bool
            or domain.shape != support.shape or not support.all() or not np.isfinite(rgb).all()
            or rgb.min() < -tolerance or rgb.max() > 1+tolerance or np.any(domain & ~support)):
        raise ValueError('Incomplete finite UM physical feature input; never fill invalid pixels as evidence')
    # Convex float32 interpolation of bounded uint8 can overshoot by one ULP.
    # Enforce only that rounding tolerance, not a tone curve or exposure change.
    if rgb.min() < 0 or rgb.max() > 1:
        rgb = np.clip(rgb, 0, 1)
    return rgb, domain, receipt


def temporal_input(dataset, group, plan):
    current, domain, receipt = content_input(dataset, group['current'], plan=plan)
    baseline, baseline_domain, baseline_receipt = content_input(dataset, group['baseline'], plan=plan)
    previous = None
    if group['previous'] is not None:
        previous, previous_domain, previous_receipt = content_input(dataset, group['previous'], plan=plan)
        if (previous.shape != current.shape or not np.array_equal(previous_domain, domain)
                or previous_receipt['correction_sources'] != receipt['correction_sources']):
            raise ValueError('UM previous frame uses a different physical coordinate representation')
    if (current.shape != baseline.shape or not np.array_equal(domain, baseline_domain)
            or baseline_receipt['correction_sources'] != receipt['correction_sources']):
        raise ValueError('UM baseline uses a different physical coordinate representation')
    return current, baseline, previous, domain, receipt


def contract_for(method, adapter):
    return method.optical.OpticalFeatureContract(generic_version='baseline-relative-bright-grain-v5',
        chemistry_adapter='none' if adapter == 'generic' else adapter)


def optical_predictions(method, current, baseline, previous, fits, *, adapter='generic', chunk_rows=32):
    workspace = method.optical.prepare_optical_feature_workspace(current, baseline,
        previous_rgb=previous, texture_sigma=2., contract=contract_for(method, adapter))
    predictors = {h: method.learning.SerializedPhaseHead(h, contract_for(method, adapter),
        fits[h].mean, fits[h].scale, fits[h].coefficients, fits[h].intercept, .5) for h in HEADS}
    scores = {h: np.empty(current.shape[:2], np.float32) for h in HEADS}
    for start in range(0, current.shape[0], chunk_rows):
        stop = min(start+chunk_rows, current.shape[0])
        matrix = method.optical.optical_feature_matrix(workspace, (slice(start, stop), slice(None)))
        for head in HEADS:
            scores[head][start:stop] = predictors[head].probability(matrix).reshape(stop-start, current.shape[1])
    return scores


def extract(dataset, output, snapshot, kind, *, run=None, limit=None, optical_model=None, ready_only=False):
    """Per-frame outputs permit bounded workers and exact checksum-based resume."""
    dataset, output = Path(dataset), Path(output)
    plan = read(dataset/'dataset_plan.json'); method = load_numerics(snapshot)
    records = training_records(dataset, kind); by_id = {r['region_id']: i for i, r in enumerate(records)}
    groups = [g for g in plan['groups'] if g[kind] and (run is None or g['run'] == run)]
    if limit: groups = groups[:limit]
    contract = contract_for(method, 'btb-ratiometric-relative-v2')
    optical_fits = None
    if kind == 'regional':
        model = read(optical_model)
        if not model['fit_report']['converged'] or model['adapter'] != 'generic':
            raise ValueError('Fresh converged generic UM optical heads are required')
        optical_fits = {h: unpack(method.logistic, model['heads'][h]) for h in HEADS}
    for group in groups:
        started = time.monotonic(); job_id = group['current']
        if ready_only and any(not (dataset/'metric'/j/'receipt.json').is_file()
                              for j in (group['current'], group['baseline'], group['previous']) if j is not None):
            continue
        destination = output/kind/job_id
        binding = dict(plan_sha256=sha(dataset/'dataset_plan.json'), kind=kind, group=group,
            numerical_source_sha256=sha(Path(snapshot)/'source_binding.json'),
            optical_model_sha256=sha(optical_model) if optical_model else None)
        if kind == 'regional':
            binding['optical_prediction_source_sha256'] = method.prediction_source_binding['source_learning.py']['sha256']
        receipt_path = destination/'receipt.json'
        if receipt_path.exists():
            receipt = read(receipt_path)
            if receipt['binding'] != binding:
                raise ValueError('Existing UM extraction has a different binding')
            bound(destination/'samples.npz', receipt['samples_sha256'])
            emit('extraction_cached', kind=kind, job=job_id); continue
        current, baseline, previous, domain, frame = temporal_input(dataset, group, plan)
        native = read(bound(group['manifest'], frame['source_job']['manifest_sha256']))
        selected = [(by_id[r['region_id']], records[by_id[r['region_id']]]) for r in group[kind]]
        if kind == 'optical':
            workspace = method.optical.prepare_optical_feature_workspace(current, baseline,
                previous_rgb=previous, texture_sigma=2., contract=contract)
            matrices, ownership, coverage = [], [], []
            for owner, record in selected:
                if record['payload'].get('training_sampling'):
                    raise ValueError('An active legacy special sampling policy requires explicit source binding')
                (x0, y0, x1, y1), mask = method.annotation.annotation_window(record['payload'],
                    canvas_wh=native['canvas_wh'], content_box=native['content_box_xywh'])
                original_pixels = int(mask.sum()); mask &= domain[y0:y1, x0:x1]
                rr, cc = np.nonzero(mask)
                if len(rr) > 1024:
                    seed = int.from_bytes(hashlib.sha256(record['region_id'].encode()).digest()[:8], 'little')
                    take = np.sort(np.random.default_rng(seed).choice(len(rr), 1024, replace=False))
                    rr, cc = rr[take], cc[take]
                values = method.optical.optical_feature_matrix(workspace, (rr+y0, cc+x0)) if len(rr) else np.empty((0, 37), np.float32)
                if not np.isfinite(values).all():
                    raise ValueError('Invalid UM optical features: '+record['region_id'])
                matrices.append(values); ownership.append(np.full(len(values), owner, np.int32))
                coverage.append(dict(region_id=record['region_id'], original_annotation_pixels=original_pixels,
                    pixels_in_original_domain=int(mask.sum()), training_rows=len(values),
                    status='included' if len(values) else 'no-supported-original-domain-pixels'))
            matrix, owners = np.concatenate(matrices), np.concatenate(ownership)
            del workspace
        else:
            scores = optical_predictions(method, current, baseline, previous, optical_fits)
            prepared = method.regional.prepare(current, baseline, scores[HEADS[0]], scores[HEADS[1]],
                domain, frame['content_spacing_yx_m'])
            matrix, owners, coverage = method.sampling.region_samples(prepared, selected, native, native_domain=domain)
            del scores, prepared
        destination.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(destination/'samples.npz', features=matrix, owners=owners)
        write(receipt_path, dict(binding=binding, rows=len(matrix), coverage=coverage,
            samples_sha256=sha(destination/'samples.npz'),
            frame_receipt_sha256=sha(dataset/'metric'/job_id/'receipt.json'),
            contract=contract.to_dict() if kind == 'optical' else list(method.regional.FEATURE_NAMES)))
        emit('extracted', kind=kind, run=group['run'], index=group['index'], job=job_id,
             rows=len(matrix), seconds=round(time.monotonic()-started, 1))
        del current, baseline, previous, domain, matrix, owners


def pack(fit, *, regional=False):
    result = asdict(fit)
    for name in ('mean', 'scale', 'coefficients'):
        value = result[name]
        if regional:
            full = np.ones(20) if name == 'scale' else np.zeros(20)
            full[REGIONAL_COLUMNS] = value; value = full
        result[name] = value.tolist()
    return result


def unpack(logistic, fit):
    names = {f.name for f in fields(logistic.LogisticFit)}
    return logistic.LogisticFit(**{k: np.asarray(v) if k in ('mean', 'scale', 'coefficients') else v
        for k, v in fit.items() if k in names})


def fit(dataset, output, snapshot, kind, *, adapter='generic', omitted_raw=None):
    dataset, output = Path(dataset), Path(output)
    plan = read(dataset/'dataset_plan.json'); method = load_numerics(snapshot)
    records = training_records(dataset, kind); matrices, ownership, bindings, omitted = [], [], [], []
    jobs = {j['job_id']: j for j in plan['jobs']}
    if omitted_raw is not None:
        approval = read(omitted_raw)
        if approval.get('decision') != 'omit_missing_raw_from_regional_fit':
            raise ValueError('Explicit missing-RAW disposition required')
        allowed_missing = set(approval['raw_sha_unavailable_paths'])
    else:
        allowed_missing = set()
    for group in plan['groups']:
        if not group[kind]: continue
        job_id = group['current']
        if not jobs[job_id]['raw_available']:
            if kind != 'regional' or jobs[job_id]['raw'] not in allowed_missing:
                raise ValueError('Missing RAW is unresolved: '+jobs[job_id]['raw'])
            omitted.extend(r['region_id'] for r in group[kind]); continue
        folder = output/kind/job_id; receipt = read(folder/'receipt.json')
        if receipt['binding']['plan_sha256'] != sha(dataset/'dataset_plan.json'):
            raise ValueError('Extraction plan changed')
        with np.load(bound(folder/'samples.npz', receipt['samples_sha256'])) as archive:
            matrices.append(archive['features']); ownership.append(archive['owners'])
        bindings.append(dict(job=job_id, receipt_sha256=sha(folder/'receipt.json')))
    matrix, owners = np.concatenate(matrices), np.concatenate(ownership)
    if kind == 'regional':
        matrix = matrix[:, REGIONAL_COLUMNS]; regularization = {h: .001 for h in HEADS}
    else:
        cohort = read(bound(plan['optical_cohort'], plan['optical_cohort_sha256']))
        source = Path(cohort['source_model'])
        regularization = {}
        for head in HEADS:
            relative = 'models\\'+adapter+'\\'+head+'_signal.json'
            source_head = read(bound(source/'models'/adapter/(head+'_signal.json'), cohort['source_head_sha256'][relative]))
            expected = contract_for(method, adapter).to_dict()
            if source_head['feature_contract'] != expected or source_head['texture_sigma'] != 2:
                raise ValueError('UM original optical feature contract differs')
            regularization[head] = source_head['training_provenance']['cross_validation']['selected_regularization']
        if adapter == 'generic': matrix = matrix[:, :36]
    regions = [method.logistic.ReviewedRegion(**r) for r in records]
    seeds, seed_reports = {}, {}
    for head in HEADS:
        emit('seed_fit', kind=kind, adapter=adapter, head=head, rows=len(matrix), regularization=regularization[head])
        seeds[head], seed_reports[head] = method.logistic.fit_weighted_logistic_mil(matrix, owners, regions,
            head=head, runs=None, regularization=regularization[head], backend='cpu', max_iter=1500)
    emit('joint_fit', kind=kind, adapter=adapter)
    fits, report = method.joint.fit_joint_aqueous_mil(matrix, owners, regions, seed_fits=seeds,
        regularization=regularization, backend='cpu', max_iter=2000, tolerance=1e-6 if kind == 'regional' else 1e-7)
    result = dict(format=method.regional.VERSION if kind == 'regional' else 'ffum-raw-optical-heads-v1',
        adapter=adapter if kind == 'optical' else 'generic optical input',
        feature_contract=contract_for(method, adapter).to_dict() if kind == 'optical' else None,
        feature_names=list(method.regional.FEATURE_NAMES) if kind == 'regional' else contract_for(method, adapter).to_dict()['feature_names'],
        settings=asdict(method.regional.Settings()) if kind == 'regional' else None, texture_sigma=2.,
        heads={h: pack(f, regional=kind == 'regional') for h, f in fits.items()}, fit_report=report,
        seed_reports=seed_reports, regularization=regularization, training_rows=len(matrix),
        original_region_count=len(records), omitted_missing_raw_region_ids=omitted,
        source_extractions=bindings, numerical_source_sha256=sha(Path(snapshot)/'source_binding.json'),
        optical_prediction_source_sha256=method.prediction_source_binding['source_learning.py']['sha256'],
        frozen_heads=[], independent_validation=False, evaluation_scope='training only',
        source_products='uncropped full-sensor RAW', feature_metric='internal original physical representation',
        optical_geometry='relative JPG R-only alignment, not absolute lens calibration')
    path = output/'models'/(kind+'_'+adapter+'.json')
    write(path, result)
    if not report['converged']:
        raise ValueError('UM fit did not meet the original gradient requirement; checkpoint preserved')
    emit('fit_complete', kind=kind, adapter=adapter, model=str(path), rows=len(matrix))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('extract', 'fit'))
    for name in ('dataset', 'output', 'snapshot'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--kind', choices=('optical', 'regional'), required=True)
    parser.add_argument('--run'); parser.add_argument('--limit', type=int)
    parser.add_argument('--ready-only', action='store_true')
    parser.add_argument('--adapter', choices=ADAPTERS, default='generic')
    parser.add_argument('--optical-model', type=Path); parser.add_argument('--omitted-raw', type=Path)
    args = parser.parse_args()
    if args.action == 'extract':
        extract(args.dataset, args.output, args.snapshot, args.kind, run=args.run, limit=args.limit,
                optical_model=args.optical_model, ready_only=args.ready_only)
    else:
        fit(args.dataset, args.output, args.snapshot, args.kind, adapter=args.adapter, omitted_raw=args.omitted_raw)


if __name__ == '__main__': main()
