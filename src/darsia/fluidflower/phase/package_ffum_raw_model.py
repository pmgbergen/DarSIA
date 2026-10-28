"""Persist a complete UM RAW model without mutable cache/runtime dependencies."""
import importlib.metadata
from pathlib import Path
import shutil
import sys

from .train_ffum_raw_model import bound, read, sha, write


def snapshot_native_support(native_runtime, target_fluidflower):
    """Keep the approved reader's complete import support beside its snapshot."""
    source, target = Path(native_runtime).parent, Path(target_fluidflower)
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo')
    shutil.copytree(source/'pipeline', target/'pipeline', ignore=ignore, dirs_exist_ok=True)
    for name in ('store_map.py', 'series_paths.py', 'job_filesystem.py'):
        shutil.copy2(source/name, target/name)


def package_model(dataset, training, geometry_masks, numerical_sources, destination,
                  *, legacy_runtime, native_runtime, overrides, missing_raw_decision):
    dataset, training, destination = map(Path, (dataset, training, destination))
    plan = read(dataset/'dataset_plan.json'); complete = read(training/'training_complete.json')
    if (complete['regional_active'] != 2518 or complete['optical_active'] != 1947
            or complete['frozen_heads'] or complete['independent_validation']):
        raise ValueError('Unexpected UM final training disposition')
    for filename, digest in complete['models'].items(): bound(training/'models'/filename, digest)
    destination.mkdir(parents=True, exist_ok=False)
    assets = {}
    def copy(source, relative, *, expected=None, key=None):
        source = Path(source); digest = expected or sha(source); bound(source, digest)
        target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            bound(target, digest)
        else:
            shutil.copy2(source, target); bound(target, digest)
        if key: assets[key] = dict(path=relative, sha256=digest)
        return digest
    for filename in complete['models']:
        key = 'regional' if filename.startswith('regional_') else filename.removesuffix('.json')
        copy(training/'models'/filename, 'models/'+filename, expected=complete['models'][filename], key=key)
    copy(plan['profile'], 'camera/experimental_relative_profile.json', expected=plan['profile_sha256'], key='profile')
    copy(dataset/'dataset_plan.json', 'training/dataset_plan.json')
    copy(plan['optical_cohort'], 'training/optical_cohort.json', expected=plan['optical_cohort_sha256'])
    copy(missing_raw_decision, 'training/missing_raw_decision.json', expected=complete['missing_raw_decision_sha256'])
    copy(training/'training_complete.json', 'training/training_complete.json')
    copy(training/'prepared_input_completion.json', 'training/prepared_input_completion.json',
         key='prepared_input_completion')
    copy(dataset.parent/'executed_preparation_code_binding.json',
         'training/executed_preparation_code_binding.json', key='executed_preparation_code_binding')
    source_model = Path(plan['model'])
    for name in ('evidence.json', 'extraction.json', 'regional_training_protocol.json', 'phase_optical_regions.json'):
        copy(source_model/name, 'training/original_'+name)
    # The preserved feature samples and owner receipts make the actual fit
    # inspectable without re-reading transient image preparation directories.
    sampled_regions = {}
    for kind in ('optical', 'regional'):
        sampled_regions[kind] = set()
        for folder in (training/kind).iterdir():
            if not folder.is_dir() or not (folder/'receipt.json').is_file(): continue
            receipt = read(folder/'receipt.json')
            sampled_regions[kind].update(r['region_id'] for r in receipt['coverage'] if r['training_rows'] > 0)
            copy(folder/'receipt.json', f'training/{kind}/{folder.name}/receipt.json')
            copy(folder/'samples.npz', f'training/{kind}/{folder.name}/samples.npz', expected=receipt['samples_sha256'])
    mask_count = 0
    for receipt_path in Path(geometry_masks).glob('*_frame_*.json'):
        receipt = read(receipt_path); mask_count += len(receipt['records'])
        copy(receipt_path, 'regions/'+receipt_path.name)
        copy(receipt_path.with_suffix('.npz'), 'regions/'+receipt_path.with_suffix('.npz').name,
             expected=receipt['masks_sha256'])
    if mask_count != 2527: raise ValueError('UM bundle requires all2527 original region masks and dispositions')
    # Preserve whether each historical photograph checksum was already present
    # or was newly bound from today's original file; do not upgrade provenance.
    for source in (Path(geometry_masks).parent/'geometry_full').glob('*_frame_*.json'):
        copy(source, 'regions/original_geometry/'+source.name)
    contexts, by_run, copied_jobs, receipt_hashes = {}, {}, set(), {}
    def copy_job(job_id):
        folder = dataset/'metric'/job_id; receipt = read(folder/'receipt.json')
        if job_id not in copied_jobs:
            for name in ('rgb_unit', 'source_support', 'training_domain'):
                copy(folder/(name+'.npy'), f'inputs/metric/{job_id}/{name}.npy', expected=receipt[name+'_sha256'])
            copy(folder/'receipt.json', f'inputs/metric/{job_id}/receipt.json')
            receipt_hashes[job_id] = sha(folder/'receipt.json'); copied_jobs.add(job_id)
        return receipt
    jobs = {j['job_id']: j for j in plan['jobs']}
    for group in plan['groups']:
        job = jobs[group['baseline']]; context_key = job['manifest_sha256'][:20]
        if context_key in contexts: continue
        receipt = copy_job(group['baseline'])
        table_key = 'photometry_'+job['photometry_sha256'][:16]
        copy(job['photometry'], 'camera/'+table_key+'.json', expected=job['photometry_sha256'], key=table_key)
        keys = {}
        for i, role in ((0, 'resize'), (1, 'drift')):
            source = receipt['correction_sources'][i]; key = role+'_'+source['sha256'][:16]
            copy(source['path'], 'geometry/'+key+'.npz', expected=source['sha256'], key=key); keys[role] = key
        grid_sha = sha(receipt['curvature_cache']); key = 'curvature_'+grid_sha[:16]
        copy(receipt['curvature_cache'], 'geometry/'+key+'.npz', expected=grid_sha, key=key); keys['curvature'] = key
        copy(job['manifest'], 'geometry/native_'+job['manifest_sha256']+'.json', expected=job['manifest_sha256'])
        signature = dict(baseline_raw_sha256=job['raw_sha256'], photometry_sha256=job['photometry_sha256'],
            corrections=[r['sha256'] for r in receipt['correction_sources']],
            baseline_rgb_sha256=receipt['rgb_unit_sha256'], domain_sha256=receipt['training_domain_sha256'])
        contexts[context_key] = dict(run=group['run'], baseline_job=group['baseline'],
            photometry=table_key, signature=signature, **keys)
        if group['run'] in by_run and contexts[by_run[group['run']]]['signature'] != signature:
            raise ValueError('UM run has non-equivalent historical contexts; an explicit source choice is required')
        by_run.setdefault(group['run'], context_key)
    example = next(g for g in plan['groups'] if g['run'] == 'run16' and g['index'] == 0)
    current_receipt = copy_job(example['current'])
    if example['previous']: copy_job(example['previous'])
    full_dir = Path(current_receipt['full_raw']); full = read(full_dir/'receipt.json')
    copy(full_dir/'full_sensor.png', 'inputs/example_full_sensor.png', expected=full['full_sensor_png_sha256'], key='example_rgb')
    copy(full_dir/'valid_support.npy', 'inputs/example_raw_support.npy', expected=full['valid_support_sha256'], key='example_support')
    for kind, folder in numerical_sources.items():
        for path in Path(folder).glob('*'):
            if path.is_file(): copy(path, f'runtime/numerics_{kind}/'+path.name)
        assets['numerics_'+kind] = dict(path=f'runtime/numerics_{kind}/source_binding.json', sha256=sha(Path(folder)/'source_binding.json'))
    # Preserve the original geometric classes and the approved fixed-table
    # native implementation in this model, never in a shared installation.
    ignore = shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo')
    shutil.copytree(legacy_runtime, destination/'runtime/source/darsia', ignore=ignore)
    native_target = destination/'runtime/source/darsia/fluidflower/native'
    shutil.copytree(native_runtime, native_target, ignore=ignore, dirs_exist_ok=True)
    # The native reader's existing I/O retry helper is an import dependency.
    # Snapshot it inside this model without editing the shared pipeline.
    snapshot_native_support(native_runtime, destination/'runtime/source/darsia/fluidflower')
    for relative, source in overrides.items():
        target = destination/'runtime/source/darsia'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
    bindings = {str(p.relative_to(destination)).replace('\\', '/'): sha(p)
                for p in (destination/'runtime').rglob('*') if p.is_file()}
    write(destination/'runtime_binding.json', bindings)
    original_profile = read(source_model/'phase_optical_regions.json')
    import cv2
    dependencies = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'Pillow', 'tifffile', 'rawpy')}
    dependencies['opencv'] = cv2.__version__
    write(destination/'model.json', dict(format='ffum-full-raw-phase-v1', series='FF_UM', name='FF_UM',
        original_model=str(source_model), refitted_heads=complete['refitted_heads'], frozen_heads=[],
        scientific_acceptance=False, selected_for_production=False, independent_validation=False,
        regional_original_regions=2527, regional_active_regions=2518, optical_original_regions=2848,
        optical_active_regions=1947, transferred_original_regions=mask_count,
        sampled_region_counts={kind: len(values) for kind, values in sampled_regions.items()},
        active_region_count_meaning='source-available eligible cohort; individual unsampled statuses retained in extraction receipts',
        source_pixels_retained_full_frame=True, canvas_shape_hw=plan['full_raw_canvas_shape_hw'],
        canvas_origin_jpg_xy=plan['full_raw_canvas_origin_jpg_xy'],
        optical_geometry='relative RAW-to-camera-JPG R-only alignment; not absolute lens calibration',
        photometry='original fixed period camera table; uint8/255 with4ULP convex interpolation tolerance',
        feature_metric='internal original physical coordinates; baseline/current/previous mapped separately',
        assets=assets, contexts=contexts, run_contexts=by_run, frame_receipts=receipt_hashes,
        example=dict(current=example['current'], previous=example['previous'],
                     context=jobs[example['baseline']]['manifest_sha256'][:20]),
        output_policy={k: original_profile[k] for k in ('regional_gas', 'output_guards', 'free_gas_cleanup')},
        runtime_binding_sha256=sha(destination/'runtime_binding.json'), dependencies=dependencies, python=sys.version,
        runtime_validation_required='fresh source decode and physical input compared against the training products',
        entrypoint='darsia.fluidflower.phase.ffum_raw_model.predict_prepared_example',
        new_frame_entrypoint='darsia.fluidflower.phase.ffum_raw_model.prepare_new_raw_frame',
        temporal_requirement='explicit preceding prepared frame, or first temporal frame marker',
        gui_connection='separate integration required; no production selector changed'))
    launcher = '''from pathlib import Path
import hashlib,json,os,runpy,sys
root=Path(__file__).resolve().parent
model=json.loads((root/'model.json').read_bytes())
binding=root/'runtime_binding.json'
if hashlib.sha256(binding.read_bytes()).hexdigest()!=model['runtime_binding_sha256']:
 raise ValueError('UM runtime binding changed')
for name,digest in json.loads(binding.read_bytes()).items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
  raise ValueError('UM runtime changed: '+name)
sys.path.insert(0,str(root/'runtime/source'))
os.environ.setdefault('DARSIA_FFUM_PROJECT_ROOT',str(root.parents[1]))
sys.argv=[sys.argv[0],'--model',str(root),*sys.argv[1:]]
runpy.run_module('darsia.fluidflower.phase.ffum_raw_model',run_name='__main__')
'''
    (destination/'run_inference.py').write_text(launcher, encoding='utf8')
    return destination
