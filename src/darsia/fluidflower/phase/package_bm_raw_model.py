"""Persist the BM RAW candidate together with its complete bound inference inputs."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

from .prepare_bm_raw_training import bound, sha256, write
from .train_bm_raw_model import DOMAIN, DOMAIN_SHA, read


def _copy_runtime_sources(runtime_source, overrides, destination):
    """Snapshot DarSIA, then place BM overrides in their FluidFlower package."""
    target_root = Path(destination)/'runtime/source/darsia'
    shutil.copytree(runtime_source, target_root,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '*.pyo'))
    for relative, source in overrides.items():
        target = target_root/'fluidflower'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def package_model(dataset, fits, numerical_source, destination, *, runtime_source, overrides, legacy_source):
    """Create a durable, self-contained candidate without changing selection.

    Source paths retained inside preparation receipts are provenance only.
    Executable assets and array lookups are relative to this new model root.
    """
    dataset, fits, destination = map(Path, (dataset, fits, destination))
    complete = read(fits/'training_complete.json')
    plan = read(dataset/'plan.json')
    ready = read(dataset/'prepared.json')
    bound(dataset/'plan.json', ready['plan_sha256'])
    for kind in ('optical', 'regional'):
        bound(fits/f'{kind}_model.json', complete[f'{kind}_sha256'])
    destination.mkdir(parents=True, exist_ok=False)
    assets = {}
    def copy(source, relative, *, expected=None, key=None):
        source = Path(source)
        digest = expected or sha256(source)
        bound(source, digest)
        target = destination/relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        bound(target, digest)
        if key:
            assets[key] = dict(path=relative, sha256=digest)
    for kind in ('optical', 'regional'):
        copy(fits/f'{kind}_model.json', f'models/{kind}_model.json', key=kind)
    for path in fits.glob('*.json'):
        if not path.name.endswith('_model.json'):
            copy(path, 'training/'+path.name)
    for name in ('plan.json', 'prepared.json', 'cohort.json', 'evidence.json', 'code_binding.json'):
        copy(dataset/name, 'inputs/'+name)
    if (dataset/'sampling_compatibility.json').is_file():
        copy(dataset/'sampling_compatibility.json', 'inputs/sampling_compatibility.json')
    for name, digest in read(dataset/'code_binding.json').items():
        copy(Path(legacy_source)/name, 'runtime/preparation/'+name, expected=digest)
    compatibility = dataset/'sampling_compatibility.json'
    if compatibility.is_file():
        copy(Path(legacy_source)/'um_bm_sampling_inference.py',
             'runtime/preparation/um_bm_sampling_nearest_extension.py',
             expected=read(compatibility)['nearest_extension_sha256'])
    receipts = {}
    photometry = None
    for index in ready['frames']:
        folder = dataset/f'frame_{index:03d}'; record = read(folder/'receipt.json')
        base = 'inputs/'+folder.name+'/'
        copy(folder/'receipt.json', base+'receipt.json')
        receipts[str(index)] = sha256(folder/'receipt.json')
        copy(folder/'profile.json', base+'profile.json', expected=record['full_raw']['profile_document']['sha256'])
        for name, descriptor in record['arrays'].items():
            copy(folder/name, base+name, expected=descriptor['sha256'])
        for mask in record['region_masks']:
            name = Path(mask['path']).name
            copy(folder/name, base+name, expected=mask['sha256'])
        if photometry is None:
            photometry = record['photometry']
        elif photometry != record['photometry']:
            raise ValueError('BM training frames do not share the fixed colour contract')
        print(json.dumps(dict(stage='packaged_frame', frame=index)), flush=True)
    for key, filename in (('curvature', 'curvature_transformation.npy'), ('config', 'config.json'),
                          ('profile', 'profile_template.json'), ('registration', 'registration.json')):
        copy(plan[key]['path'], 'geometry/'+filename, expected=plan[key]['sha256'],
             key='profile_template' if key == 'profile' else key)
    copy(DOMAIN, 'geometry/domain.npz', expected=DOMAIN_SHA, key='domain')
    shutil.copytree(numerical_source, destination/'runtime/numerics', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ('legacy_features.py', 'legacy_translation.py', 'legacy_drift.py'):
        copy(Path(legacy_source)/name, 'runtime/legacy/'+name)
    # Snapshot executable sources; do not rely on a mutable installed checkout.
    _copy_runtime_sources(runtime_source, overrides, destination)
    runtime_bindings = {str(p.relative_to(destination)).replace('\\', '/'): sha256(p)
                        for p in (destination/'runtime').rglob('*') if p.is_file()}
    write(destination/'runtime_binding.json', runtime_bindings)
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'Pillow', 'tifffile', 'rawpy')}
    import cv2
    versions['opencv'] = cv2.__version__
    environment = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                   if d.metadata.get('Name')}
    write(destination/'model.json', dict(format='ffbm-full-raw-phase-v1', series='FF_BM', name='FF_BM',
        refitted_heads=complete['refitted_heads'], frozen_heads=[], evaluation_scope=complete['evaluation_scope'],
        scientific_acceptance=False, selected_for_production=False,
        inputs='uncropped Sony-corrected RAW; fixed camera WB and sRGB transfer',
        photometry=photometry, prepared_frames=ready['frames'],
        input_plan_sha256=ready['plan_sha256'],
        inferable_frames=[i for i in ready['frames'] if i == 0 or i-1 in ready['frames']],
        frame_receipts=receipts,
        assets=assets, numerics_binding_sha256=sha256(destination/'runtime/numerics/source_binding.json'),
        runtime_binding_sha256=sha256(destination/'runtime_binding.json'),
        dependencies=versions, installed_python_packages=environment, python=sys.version,
        entrypoint='darsia.fluidflower.phase.bm_raw_model.predict_prepared_frame',
        new_frame_entrypoint='darsia.fluidflower.phase.bm_raw_model.prepare_new_raw_frame',
        new_frame_requirement='original RAW and paired TIFF with hashes, actual legacy drift and explicit preceding prepared frame',
        source_pixels_retained_full_frame=True, feature_metric='original physical content representation, sampled independently per frame',
        source_preparation_model=plan['source_model']))
    launcher = '''from pathlib import Path
import hashlib,json,os,runpy,sys
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OPENCV_FOR_THREADS_NUM'):
 os.environ[name]='1'
os.environ['OPENCV_IPP']='disabled'
os.environ['OPENCV_OPENCL_RUNTIME']='disabled'
root=Path(__file__).resolve().parent
model=json.loads((root/'model.json').read_text())
binding=root/'runtime_binding.json'
if hashlib.sha256(binding.read_bytes()).hexdigest()!=model['runtime_binding_sha256']:
 raise ValueError('BM runtime binding changed')
for name,digest in json.loads(binding.read_text()).items():
 if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
  raise ValueError('BM runtime changed: '+name)
sys.path.insert(0,str(root/'runtime/source'))
sys.argv=[sys.argv[0],'--model',str(root),*sys.argv[1:]]
runpy.run_module('darsia.fluidflower.phase.bm_raw_model',run_name='__main__')
'''
    (destination/'run_inference.py').write_text(launcher, encoding='utf-8')
    return destination
