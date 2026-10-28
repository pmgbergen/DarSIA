"""Materialize bound full-field UM RAW and internal historical metric samples.

Run in the pinned legacy geometry runtime, with the approved native RAW modules
available through --native-runtime. Sources remain read-only. Full sensor PNGs
and exact support are the image products; historical grids are feature-only.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf8')
    temporary.replace(path)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);sys.modules[name]=value;spec.loader.exec_module(value)
    return value


def prepare_plan(args):
    """Resolve exact source names; never substitute a neighbouring acquisition."""
    old=read(args.plan);paths={};directories={}
    for root in old['review_roots']:
        for frame in root['frames']:
            p=Path(frame['photograph'])
            if p.suffix.upper()=='.JPG' and p.name.lower()!='pre_injection.jpg':
                key=(root['run'],p.name)
                if key in paths and paths[key]!=str(p):raise ValueError(f'Ambiguous original {key}')
                paths[key]=str(p);directories.setdefault(root['run'],set()).add(str(p.parent))
    extraction=read(args.model/'extraction.json')
    domains={};groups=defaultdict(lambda: {'regional':[],'optical':[]})
    for frame in extraction['frames']:
        domains[frame['run']]=frame['domain']
        key=(frame['native_manifest'],frame['run'],int(frame['frame_index']))
        groups[key]['regional']=frame['records']
    optical=read(args.optical_cohort)
    for row in optical['regions']:
        run=row['original_inventory']['run']
        p=str(Path(row['review_root'])/'native_frames'/run/'manifest.json')
        groups[(p,run,int(row['frame_index']))]['optical'].append(row)
    manifests={};jobs={};groups_out=[]
    def resolve(run,name):
        key=(run,name)
        if key in paths:return Path(paths[key])
        candidates=[Path(p)/name for p in directories[run] if (Path(p)/name).is_file()]
        if len(candidates)!=1:raise ValueError(f'Missing/ambiguous exact photograph: {key}: {candidates}')
        return candidates[0]
    def bind(manifest_path,run,index):
        if manifest_path not in manifests:
            native=read(manifest_path)
            if native['run']!=run or native['dataset'].lower()!='ffum':raise ValueError('Wrong source series')
            manifests[manifest_path]=(native,sha(manifest_path))
        native,manifest_sha=manifests[manifest_path]
        if index is None:
            name=Path(native['correction_contract']['raw_baseline']).name
            expected=native.get('reference',{}).get('source_sha256')
        else:
            row=next(r for r in native['frame_sources'] if r['frame_index']==index)
            members=row.get('configured_baseline_members')
            if members:
                if len(members)!=1:raise ValueError('A multi-image baseline requires explicit aggregation')
                name=Path(members[0]['path']).name;expected=None
            else:
                name=row['source_name'];expected=row.get('source_sha256')
        photo=resolve(run,name);raw=photo.with_suffix('.NEF')
        photo_sha=sha(photo)
        if expected and expected!=photo_sha:raise ValueError(f'Photograph SHA changed: {photo}')
        identity=hashlib.sha256((manifest_sha+'\n'+str(photo)).encode()).hexdigest()[:20]
        if identity not in jobs:
            period=1 if int(run[3:])<14 else 2
            document=args.documents/f'camera_table2_period{period}.json'
            jobs[identity]=dict(job_id=identity,run=run,index=index,photo=str(photo),photo_sha256=photo_sha,
                raw=str(raw),raw_available=raw.is_file(),raw_sha256=sha(raw) if raw.is_file() else None,
                manifest=manifest_path,manifest_sha256=manifest_sha,domain=domains[run],
                photometry=str(document),photometry_sha256=sha(document))
        return identity
    for (path,run,index),records in groups.items():
        current=bind(path,run,index);baseline=bind(path,run,None)
        previous=bind(path,run,index-1) if index>0 else None
        groups_out.append(dict(manifest=path,run=run,index=index,current=current,baseline=baseline,
                               previous=previous,**records))
    result=dict(format='ffum-full-raw-dataset-plan-v1',model=str(args.model),optical_cohort=str(args.optical_cohort),
        optical_cohort_sha256=sha(args.optical_cohort),profile=str(args.profile),profile_sha256=sha(args.profile),
        jobs=list(jobs.values()),groups=groups_out,missing_raw=[j for j in jobs.values() if not j['raw_available']],
        feature_representation='historical metric grid sampled internally, never stored as new source image',
        full_raw_canvas_origin_jpg_xy=[-19,-11],full_raw_canvas_shape_hw=[5526,8294])
    write(args.output/'dataset_plan.json',result)
    print(json.dumps(dict(stage='planned',jobs=len(jobs),groups=len(groups_out),missing_raw=[j['raw'] for j in result['missing_raw']])),flush=True)


def prepare_run(args):
    import cv2
    import numpy as np
    from PIL import Image
    import darsia
    import darsia.fluidflower
    from threadpoolctl import threadpool_limits
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);threadpool_limits(1)
    # Only this process gains the frozen native package, leaving legacy geometry
    # classes and shared installed packages untouched.
    darsia.fluidflower.__path__.append(str(args.native_runtime/'darsia/fluidflower'))
    here=Path(__file__).resolve().parent
    shared=module('darsia.fluidflower.phase.um_bm_review_regions',here/'um_bm_review_regions.py')
    mapping_module=module('darsia.fluidflower.phase.ffum_review_mapping',here/'ffum_review_mapping.py')
    sampling=module('darsia.fluidflower.phase.um_bm_sampling',here/'um_bm_sampling.py')
    reader=module('darsia.fluidflower.phase.ffum_raw_input',here/'ffum_raw_input.py')
    plan=read(args.output/'dataset_plan.json');started=time.time()
    jobs=[j for j in plan['jobs'] if j['run']==args.run]
    if args.limit:jobs=jobs[:args.limit]
    frame_cache={};rig_cache={}
    for number,job in enumerate(jobs):
        if not job['raw_available']:
            print(json.dumps(dict(stage='missing_raw',job=job['job_id'],raw=job['raw'])),flush=True);continue
        destination=args.output/'metric'/job['job_id'];receipt_path=destination/'receipt.json'
        if receipt_path.exists():
            existing=read(receipt_path)
            if existing['source_job']!=job or existing['profile_sha256']!=plan['profile_sha256']:
                raise ValueError('Existing UM cache belongs to different sources')
            print(json.dumps(dict(stage='cached',job=job['job_id'])),flush=True);continue
        destination.mkdir(parents=True,exist_ok=True)
        native=read(job['manifest'])
        if sha(job['manifest'])!=job['manifest_sha256'] or sha(job['photo'])!=job['photo_sha256']:
            raise ValueError('UM geometry source changed')
        rigpath=shared.shared_drive_path(native['correction_contract']['legacy_rig'])
        rigkey=str(rigpath)
        if rigkey not in rig_cache:
            resize=darsia.ResizeCorrection();resize.load(rigpath/'correction_1.npz')
            drift=darsia.DriftCorrection();drift.load(rigpath/'correction_2.npz')
            curvature=darsia.CurvatureCorrection();curvature.load(rigpath/'correction_3.npz')
            rig_cache[rigkey]=(resize,drift,curvature,[dict(path=str(rigpath/f'correction_{i}.npz'),sha256=sha(rigpath/f'correction_{i}.npz')) for i in (1,2,3)])
        resize,drift,curvature,corrections=rig_cache[rigkey]
        old=darsia.imread(job['photo']).img;resized=resize.correct_array(old)
        cv2.setRNGSeed(0);_,matrix=shared.capture_legacy_drift(drift,resized)
        shape=tuple(map(int,curvature.cache['shape']))
        if shape!=tuple(native['analysis_image_shape_hw']):raise ValueError('UM content shape changed')
        frame_key=job['raw_sha256']+job['photometry_sha256']
        frame_dir=args.output/'full_raw'/job['run']/Path(job['raw']).stem
        if frame_key in frame_cache:
            full=frame_cache[frame_key]
        elif (frame_dir/'receipt.json').exists():
            record=read(frame_dir/'receipt.json')
            if record['raw_sha256']!=job['raw_sha256'] or record['profile_sha256']!=plan['profile_sha256'] or record['photometry_sha256']!=job['photometry_sha256']:
                raise ValueError('Full sensor cache source binding changed')
            if (sha(frame_dir/'full_sensor.png') != record['full_sensor_png_sha256']
                    or sha(frame_dir/'valid_support.npy') != record['valid_support_sha256']):
                raise ValueError('Full sensor image or support checksum changed')
            full=reader.UmRawFrame(np.asarray(Image.open(frame_dir/'full_sensor.png').convert('RGB')),
                np.load(frame_dir/'valid_support.npy'),reader.UmRelativeMapping(read(plan['profile'])),record)
        else:
            full=reader.read_um_full_frame(job['raw'],raw_sha256=job['raw_sha256'],
                profile_path=plan['profile'],profile_sha256=plan['profile_sha256'],
                photometry_path=job['photometry'],photometry_sha256=job['photometry_sha256'])
            frame_dir.mkdir(parents=True,exist_ok=True)
            Image.fromarray(full.rgb).save(frame_dir/'full_sensor.png')
            np.save(frame_dir/'valid_support.npy',full.valid_support)
            write(frame_dir/'receipt.json',dict(full.metadata,
                full_sensor_png_sha256=sha(frame_dir/'full_sensor.png'),
                valid_support_sha256=sha(frame_dir/'valid_support.npy')))
        frame_cache.clear();frame_cache[frame_key]=full
        if (list(full.mapping.origin_jpg_xy)!=plan['full_raw_canvas_origin_jpg_xy']
                or list(full.rgb.shape[:2])!=plan['full_raw_canvas_shape_hw']):
            raise ValueError('Rendered profile/canvas differs from region mapping')
        mapper=mapping_module.UmReviewMapping(curvature.cache['grid'],shape,matrix,
            (old.shape[1],old.shape[0]),(resized.shape[1],resized.shape[0]),full.mapping.origin_jpg_xy)
        cell_check=mapper.validate_all_cells()
        sampled=sampling.sample_full_frame(full.rgb,full.valid_support,mapper.content_to_display,shape)
        # Explicit fixed camera-table unit conversion; no per-frame tone fitting.
        values=sampled.values/np.float32(255)
        domain_ref=job['domain']
        if sha(domain_ref['path'])!=domain_ref['sha256']:raise ValueError('Historical physical domain changed')
        if domain_ref.get('kind') == 'registered-context':
            with np.load(domain_ref['path'], allow_pickle=False) as archive:
                domain=archive['domain'].astype(bool)
                spacing=[abs(float(archive['y_m'][1]-archive['y_m'][0])),abs(float(archive['x_m'][1]-archive['x_m'][0]))]
        else:
            if Path(domain_ref['path']).name != 'facies.npz':
                raise ValueError('Unknown historical UM domain representation')
            # Exact legacy native_domain rule for the source-bound UM rig.
            with np.load(domain_ref['path'], allow_pickle=True) as archive:
                domain=archive['array'] != 1
                spacing=(np.asarray(archive['metadata'].item()['dimensions'])/domain.shape).tolist()
        if domain.shape!=shape:raise ValueError('Historical domain and content shape differ')
        valid=sampled.valid_support & domain
        # Finite source-only values remain available to old feature APIs; all
        # unsupported source pixels remain NaN and all labels use valid domain.
        np.save(destination/'rgb_unit.npy',values)
        np.save(destination/'source_support.npy',sampled.valid_support)
        np.save(destination/'training_domain.npy',valid)
        grid_path=args.output/'geometry'/('curvature_'+corrections[2]['sha256']+'.npz')
        if not grid_path.exists():
            grid_path.parent.mkdir(parents=True,exist_ok=True)
            temporary=grid_path.with_name(grid_path.stem+f'.{os.getpid()}.tmp.npz')
            np.savez(temporary,shape=shape,grid=curvature.cache['grid'])
            temporary.replace(grid_path)
        receipt=dict(source_job=job,profile_sha256=plan['profile_sha256'],full_raw=str(frame_dir),
            content_shape_hw=list(shape),content_spacing_yx_m=spacing,
            full_raw_canvas_origin_jpg_xy=list(full.mapping.origin_jpg_xy),
            drift_matrix=matrix.tolist(),photo_wh=[old.shape[1],old.shape[0]],resized_wh=[resized.shape[1],resized.shape[0]],
            curvature_cache=str(grid_path),correction_sources=corrections,cell_validation=cell_check,
            source_support_pixels=int(sampled.valid_support.sum()),original_domain_pixels=int(domain.sum()),
            training_domain_pixels=int(valid.sum()),source_encoding='fixed camera table uint8 /255',
            purpose='internal feature metric only; source product retains full sensor',
            rgb_unit_sha256=sha(destination/'rgb_unit.npy'),source_support_sha256=sha(destination/'source_support.npy'),
            training_domain_sha256=sha(destination/'training_domain.npy'))
        write(receipt_path,receipt)
        print(json.dumps(dict(stage='raw_prepared',run=args.run,done=number+1,total=len(jobs),job=job['job_id'],raw=job['raw'],seconds=round(time.time()-started,1),training_pixels=int(valid.sum()))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('plan','run'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',type=Path);p.add_argument('--plan',type=Path)
    p.add_argument('--optical-cohort',type=Path);p.add_argument('--profile',type=Path);p.add_argument('--documents',type=Path)
    p.add_argument('--native-runtime',type=Path);p.add_argument('--run');p.add_argument('--limit',type=int)
    args=p.parse_args()
    prepare_plan(args) if args.action=='plan' else prepare_run(args)


if __name__=='__main__':main()
