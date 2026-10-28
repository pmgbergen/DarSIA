"""Source-bound review photographs and existing map-placement position DATA.

Only display RGB is warped. Native labels, production homographies, physical
frames and source arrays stay unchanged. A role is never inferred from a
baseline filename or equal photo shapes.
"""
from copy import deepcopy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re

import numpy as np

FORMAT = 'darsia-painted-review-display-context-v1'
METHOD = 'camera registration (map placement)'
CLEARWATER_LIMIT = ('The explicitly selected clear-water RAW has not been supplied as an original current-route '
    'native display render with its exact current profile/model/grid bindings. No new registration is inferred.')


def _json(value):
    if value is None or type(value) in (str,bool,int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is list:
        return [_json(item) for item in value]
    if type(value) is dict and all(type(key) is str for key in value):
        return {key:_json(item) for key,item in value.items()}
    raise ValueError('Review display metadata must be finite literal JSON')


def _same(left,right):
    return display_context_digest(left)==display_context_digest(right)


def display_context_digest(value):
    """One canonical finite-JSON digest for context, display issues and seals."""
    return hashlib.sha256(json.dumps(_json(value),ensure_ascii=False,allow_nan=False,sort_keys=True,
        separators=(',',':')).encode('utf-8')).hexdigest()


def _bound(path):
    from .painted_adoption_table import _bound as original
    return original(Path(path))


def _checked(value,metadata=False):
    from .painted_adoption_table import _checked as original
    return original(value,metadata=metadata)


def _descriptor(value):
    if not isinstance(value,dict) or not all(key in value for key in ('path','sha256','size_bytes')):
        raise ValueError('An exact source descriptor is required for review images')
    result={key:value[key] for key in ('path','sha256','size_bytes')}
    if (type(result['path']) is not str or not result['path'] or type(result['sha256']) is not str
            or re.fullmatch('[0-9a-f]{64}',result['sha256']) is None
            or type(result['size_bytes']) is not int or result['size_bytes']<0):
        raise ValueError('An exact finite source descriptor is required for review images')
    return result


def _order(values):
    if not isinstance(values,list) or not values or any(not isinstance(run,str) or not run.strip() for run in values):
        raise ValueError('The full explicitly declared series order is required')
    keys=[run.strip().casefold() for run in values]
    if len(keys)!=len(set(keys)):
        raise ValueError('The declared series order repeats a run')
    return keys


def capture_display_context(project,*,declared_order,image_roles=None,limitations=None):
    """Capture original registered order before a scientific worker starts.

    The role bindings are explicit existing receipts, never reconstructed from
    the review cohort. This metadata operation reads no photograph or RAW.
    """
    from .series_geometry import native_series_link,read_series_manifest
    link=native_series_link(project)
    if not link.get('series_root'):
        return None  # Preserve historical projects without native-series context.
    root=Path(link['series_root'])
    before=_bound(root/'series.toml')
    declared=read_series_manifest(root).get('runs') or {}
    if not _same(list(declared_order),declared.get('ids')):
        raise ValueError('The registered series order changed before review context capture')
    _order(list(declared_order))
    if not _same(_bound(root/'series.toml'),before):
        raise ValueError('The registered series changed while capturing review context')
    result=dict(format=FORMAT,series_manifest=before,declared_run_order=list(declared_order),
        declared_runs=_json(declared),image_roles=_json(image_roles or {}),limitations=_json(limitations or {}),anchor_metric_mm=[0.,0.])
    return validate_display_context(result)


def validate_display_context(context,review=None):
    """Recheck the actual original manifest; never shrink order to accepted runs."""
    from .series_geometry import read_series_manifest
    value=_json(context)
    if (not isinstance(value,dict) or set(value)!={'format','series_manifest','declared_run_order','declared_runs','image_roles','limitations','anchor_metric_mm'}
            or value['format']!=FORMAT or not isinstance(value['image_roles'],dict)
            or not isinstance(value['limitations'],dict) or any(type(reason) is not str for reason in value['limitations'].values())
            or not isinstance(value['anchor_metric_mm'],list) or len(value['anchor_metric_mm'])!=2
            or any(type(point) not in (int,float) or point!=0 for point in value['anchor_metric_mm'])):
        raise ValueError('An explicit source-bound review display context is required')
    value['anchor_metric_mm']=[0.,0.]
    order=_order(value['declared_run_order'])
    source=_descriptor(value['series_manifest'])
    if Path(source['path']).name!='series.toml':
        raise ValueError('Display order must bind the original series.toml')
    before=_bound(source['path'])
    manifest=read_series_manifest(Path(source['path']).parent)
    if (not _same((manifest.get('runs') or {}).get('ids'),value['declared_run_order'])
            or not _same(value['declared_run_order'],value['declared_runs'].get('ids'))):
        raise ValueError('The explicitly registered series order or declarations changed')
    if not _same(before,_bound(source['path'])):
        raise ValueError('The current series metadata changed while validating display order')
    if review is not None and any(row['run_id'] not in order for row in review['runs']):
        raise ValueError('A reviewed run is absent from the full declared series')
    for binding in value['image_roles'].values():
        _checked(binding,metadata=True)
    return value


def context_source_sha256(context):
    """Stable actual source closure, separate from capture and UI preferences.

    Admission still rechecks the exact manifest/receipt descriptors and directory
    observation. Equality retains declared full order and actual selected source
    identities, excluding receipt location/time and directory observation time.
    Native output products, brightness, pan/zoom and tolerances are not inputs.
    """
    value=validate_display_context(context)
    roles={}
    for role,binding in value['image_roles'].items():
        document=_checked(binding,metadata=True)
        if role=='first-experiment' and document.get('format')=='darsia-first-experiment-review-roles-v1':
            rows={}
            for run,row in document.get('runs',{}).items():
                if not isinstance(row,dict):
                    raise ValueError('The selected image role contains a malformed run')
                rows[run]={key:_json(row.get(key)) for key in ('format','run_id','role','available','selected_raw',
                    'selected_model','renderer','native_photo','native_photo_pixels','selection','selected_row')}
                observation=row.get('directory_observation') or {}
                rows[run]['raw_name_observation']={key:_json(observation.get(key))
                    for key in ('path','raw_count','first_raw_name','names_sha256')}
            roles[role]=dict(format=document['format'],index=_json(document.get('index')),runs=rows)
        elif role=='clearwater' and document.get('format')=='darsia-clearwater-display-selections-v1':
            roles[role]=dict(format=document['format'],merge=_json(document.get('merge')),
                runs={run:{key:_json(row.get(key)) for key in ('format','run_id','merged_row','reference','processing',
                    'selected_raw','selected_rule','selection_only','historical_render_not_reused')}
                    for run,row in document.get('runs',{}).items()})
        else:
            # A future explicit rendered role has no inferred identity projection.
            # Until an original producer is supported, retain its exact source.
            roles[role]=dict(source=_descriptor(binding))
    return display_context_digest(dict(format=FORMAT,declared_run_order=value['declared_run_order'],
        image_roles=roles,anchor=dict(unit='mm',origin='inner left-wall and bottom intersection',
            x_axis='right',y_axis='up',point=value['anchor_metric_mm'])))


def _review(value):
    from .painted_analysis_exclusions import read_gap_review
    return read_gap_review(value)


def read_review(source):
    """Admit display metadata in the finite reader before any photo is decoded."""
    review=_review(source)
    if review.get('display_context') is not None:
        validate_display_context(review['display_context'],review)
    return review


def _model(row):
    """Read the original RunGeometry fields plus exact retained native renderer."""
    from .map_placement import RunGeometry,_matrix
    document=_checked(row['target_model'],metadata=True)
    run=row['run_id']
    if str(document.get('run_id') or '').casefold()!=run:
        raise ValueError('The review model belongs to another run')
    c=(_matrix(document['H_metric_mm_to_camera_u'],'review metric-to-camera-U') if 'H_metric_mm_to_camera_u' in document else
       np.linalg.inv(_matrix(document['H_camera_u_to_metric_mm'],'review camera-U-to-metric')))
    point_review=((row.get('contract') or {}).get('target_model') or {}).get('point_review')
    if point_review is not None:
        from .rig_point_alignment import effective_matrix,validate_contract_binding
        contract=row['contract']
        validate_contract_binding(contract['geometry_index'],contract['source_model'],contract['target_model'])
        c=effective_matrix(point_review,run,document.get('reference_image') or {})
    native=document['baseline_native_grid']
    certificate=_checked(native['renderer'],metadata=True)
    renderer,inputs,photo=certificate['renderer'],certificate['inputs'],certificate['corrected_baseline']
    if (not _same(_descriptor(document['reference_image']),inputs['raw'])
            or not _same(renderer['raw_acquisition'],inputs['raw'])
            or not _same(native['H_baseline_camera_u_to_rig_corrected_native'],renderer['camera_u_to_native_h'])
            or not _same(photo['archive'],row['native_photo']) or not _same(photo['pixels'],row['native_photo_pixels'])
            or native['shape_hw']!=row['gap']['shape_hw'] or renderer['native_shape_hw']!=native['shape_hw']):
        raise ValueError('The original RAW/renderer/native-photo/grid review bindings changed')
    profile=document['camera_profile']
    if document.get('grid')!='camera_u' or document.get('shape_hw')!=profile['camera_wh'][::-1]:
        raise ValueError('The measured model does not explicitly bind the camera-U pixel grid')
    if (not _same(_descriptor(profile),inputs['profile']) or profile['profile_id']!=renderer['profile_id']
            or profile['camera_wh']!=renderer['camera_u_wh']):
        raise ValueError('The exact calibrated camera-U profile differs from its renderer')
    geometry=RunGeometry(run,c,deepcopy(document['reference_image']),tuple(document.get('shape_hw') or profile['camera_wh'][::-1]),
        row['target_model']['path'],model=document,entry=dict(reference_model=deepcopy(row['target_model'])))
    if point_review is not None:
        geometry.entry['_reference_point_review']=deepcopy(point_review)
    from .map_adoption import native_transport
    n,shape=native_transport(geometry)
    if list(shape)!=native['shape_hw']:
        raise ValueError('The original native transport and renderer grid differ')
    _camera_u_context(geometry,certificate)
    return geometry,n,certificate


def _camera_u_context(geometry,certificate):
    """Compare spatial camera conventions, never per-run RAW/date metadata."""
    profile=geometry.model['camera_profile']
    parameters=_checked(certificate['inputs']['profile'],metadata=True)
    parameters=parameters.get('registration',parameters)
    sensor=parameters['common']['sensor']
    recorded_sensor=certificate['renderer'].get('sensor_metadata')
    if not isinstance(recorded_sensor,dict) or type(recorded_sensor.get('orientation')) is not int:
        raise ValueError('The original renderer lacks its recorded integer RAW orientation')
    convention=certificate['homography_convention']
    if not _same(convention,dict(source='saved corrected-baseline pixel centres (column,row)',
            target='camera-U pixel centres (column,row)',camera_u_axes='x right, y down',raw_orientation=recorded_sensor['orientation'])):
        raise ValueError('The original native-to-camera-U coordinate convention differs')
    grid=certificate['inputs']['uncorrected_grid']
    expected=dict(homography_coordinates=['column','row','one'],pixel_centres='integer indices',
        indexing='ij',array_row_direction='down',array_column_direction='right')
    if (any(not _same(grid.get(key),value) for key,value in expected.items())
            or grid['num_voxels']!=profile['camera_wh'][::-1] or sensor['camera_wh']!=profile['camera_wh']):
        raise ValueError('The exact original camera-U grid/conventions differ')
    return dict(grid='camera_u',profile=deepcopy(profile),camera_wh=deepcopy(profile['camera_wh']),
        shape_hw=deepcopy(grid['num_voxels']),**expected,camera_u_axes=convention['camera_u_axes'],
        raw_orientation=deepcopy(convention['raw_orientation']))


def _profile_equal(first,first_certificate,second,second_certificate):
    return _same(_camera_u_context(first,first_certificate),_camera_u_context(second,second_certificate))


def _first_role(context,review,row,model,certificate):
    """Replay the existing metadata/name-only selected-first RAW observation."""
    binding=context['image_roles'].get('first-experiment')
    if binding is None:
        raise ValueError('No explicit selected-first-injection RAW role receipt is bound')
    roles=_checked(binding,metadata=True)
    if roles.get('format')!='darsia-first-experiment-review-roles-v1':
        raise ValueError('The first-experiment role source has another format')
    role=roles.get('runs',{}).get(row['run_id'])
    if (not isinstance(role,dict) or role.get('format')!='darsia-first-experiment-review-role-v1'
            or role.get('run_id')!=row['run_id'] or role.get('role')!='first-experiment' or role.get('available') is not True
            or not _same(role.get('selected_model'),row['target_model'])
            or not _same(role.get('renderer'),model.model['baseline_native_grid']['renderer'])
            or not _same(role.get('selected_raw'),_descriptor(model.reference_image))
            or not _same(role.get('selected_raw'),certificate['inputs']['raw'])
            or not _same(role.get('native_photo'),row['native_photo'])
            or not _same(role.get('native_photo_pixels'),row['native_photo_pixels'])):
        raise ValueError('The explicit selected-first RAW role does not match this accepted native photograph')
    raw=role['selected_raw'];directory=Path(raw['path']).parent
    protocols=[item for item in certificate['inputs']['protocol_inputs'] if item['role']=='imaging' and item.get('folder')
        and Path(item['folder'])==directory and '_injection' in directory.name.casefold()]
    if len(protocols)!=1 or not _same(protocols[0]['source'],role['selection']):
        raise ValueError('The selected RAW lacks its original unique injection imaging protocol')
    selection=_checked(role['selection'])
    from .painted_adoption_table import _read
    raw_csv=_read(Path(selection['path']))
    if len(raw_csv)!=selection['size_bytes'] or hashlib.sha256(raw_csv).hexdigest()!=selection['sha256']:
        raise ValueError('The original selected-first protocol changed while reading')
    records=list(csv.DictReader(io.StringIO(raw_csv.decode('utf-8-sig'))))
    selected=[item for item in records if Path(item['path'])==Path(raw['path'])]
    if (len(selected)!=1 or selected[0]['source_sha256']!=raw['sha256'] or selected[0]['baseline']!='True'
            or not _same(selected[0],role['selected_row'])
            or Path(sorted(records,key=lambda record:record['datetime'])[0]['path'])!=Path(raw['path'])):
        raise ValueError('The selected RAW is no longer the original protocol baseline and earliest injection photograph')
    before=directory.stat();names=[]
    with os.scandir(directory) as listing:
        for number,item in enumerate(listing,1):
            if number>20000:
                raise ValueError('The original injection directory metadata limit was exceeded')
            if item.name.casefold().endswith('.arw'):
                names.append(item.name)
    after=directory.stat()
    names.sort(key=str.casefold)
    observed=role['directory_observation']
    if (not names or any(re.fullmatch(r'DSC\d{5}\.ARW',name,re.I) is None for name in names)
            or before.st_mtime_ns!=after.st_mtime_ns
            or observed['path']!=str(directory) or observed['raw_count']!=len(names)
            or observed['first_raw_name']!=names[0] or names[0].casefold()!=Path(raw['path']).name.casefold()
            or observed['names_sha256']!=hashlib.sha256(json.dumps(names,separators=(',',':')).encode()).hexdigest()):
        raise ValueError('The bound injection directory no longer has the same first extant numbered RAW')
    _checked(role['selection']);_checked(binding,metadata=True)
    return deepcopy(role)


def _clear_role(context,row):
    """Use old processing receipts only to select the actual RAW, never its H/RGB."""
    binding=context['image_roles'].get('clearwater')
    if binding is None:
        raise ValueError(context['limitations'].get('clearwater',CLEARWATER_LIMIT))
    catalog=_checked(binding,metadata=True)
    if catalog.get('format')=='darsia-ac45-clearwater-source-policy-v1':
        from .clearwater_source_choice import selection_from_policy
        return selection_from_policy(binding,context,row)
    if catalog.get('format')!='darsia-clearwater-display-selections-v1':
        raise ValueError('The explicit clear-water selection source has another format')
    run=row['run_id'];selection=catalog.get('runs',{}).get(run)
    if (not isinstance(selection,dict) or selection.get('format')!='darsia-clearwater-display-selection-v1'
            or selection.get('run_id')!=run or selection.get('selection_only') is not True
            or selection.get('historical_render_not_reused') is not True
            or type(selection.get('selected_rule')) is not str or not selection['selected_rule'].strip()):
        raise ValueError('An explicit original per-run clear-water RAW selection is required')
    merge=_checked(catalog['merge'],metadata=True)
    members=[item for item in merge.get('sources',[]) if item.get('run')==run]
    if (merge.get('complete') is not True or len(members)!=1
            or not _same(members[0].get('merged'),selection['merged_row'])):
        raise ValueError('The clear-water selection is absent from its exact original merge')
    merged=_checked(selection['merged_row'],metadata=True)
    reference=_checked(selection['reference'],metadata=True)
    processing=_checked(selection['processing'],metadata=True)
    raw=_descriptor(selection['selected_raw'])
    if (merged.get('run')!=run or merged.get('complete') is not True
            or any(members[0].get('reference',{}).get(key)!=selection['reference'][key] for key in ('path','sha256'))
            or any(merged.get('reference',{}).get(key)!=selection['reference'][key] for key in ('path','sha256'))
            or reference.get('format')!='ffac-v5-normalized-clearwater-reference-v1'
            or processing.get('format')!='ffac-v5-normalized-clearwater-processing-v1'
            or reference.get('dataset')!='ffac' or processing.get('dataset')!='ffac'
            or reference.get('run')!=run or processing.get('run')!=run
            or reference.get('raw_source_sha256')!=raw['sha256'] or processing.get('raw_source_sha256')!=raw['sha256']
            or Path(processing.get('raw_source_path',''))!=Path(raw['path'])
            or reference.get('correction_receipt_path')!=selection['processing']['path']
            or reference.get('correction_receipt_sha256')!=selection['processing']['sha256']
            or processing.get('selected_rule')!=selection['selected_rule']):
        raise ValueError('The original clear-water RAW selection/reference/processing chain changed')
    from .photograph_io import is_camera_raw
    if not is_camera_raw(Path(raw['path'])) or Path(raw['path']).stat().st_size!=raw['size_bytes']:
        raise ValueError('The explicitly selected clear-water RAW file metadata differs')
    # No historical profile, RGB, valid-mask, drift or registration asset is read.
    for descriptor in (catalog['merge'],selection['merged_row'],selection['reference'],selection['processing'],binding):
        _checked(descriptor,metadata=True)
    return dict(selection,selection_catalog=deepcopy(binding),selected_raw=raw)


def _water_basis(row,geometry,native,selection):
    """Reconstruct only the current-route display plane and existing own geometry."""
    _position_model(row,geometry,_checked(geometry.model['baseline_native_grid']['renderer'],metadata=True))
    own=np.asarray(row['own_frame_native_xy'],np.float64)
    transported=np.asarray(row['transported_frame_native_xy'],np.float64)
    if own.shape!=(4,2) or transported.shape!=(4,2) or not np.isfinite(own).all() or not np.isfinite(transported).all():
        raise ValueError('The display needs the original ordered own/transported frame corners')
    identity=np.eye(3).tolist()
    selection_sources=({key:deepcopy(selection[key]) for key in ('merged_row','reference','processing')}
        if selection.get('source_exception') is None else
        dict(source_policy=deepcopy(selection['source_exception']),authority=deepcopy(selection['authority'])))
    basis=dict(format='darsia-water-surface-basis-v1',run_id=row['run_id'],photo=deepcopy(selection['selected_raw']),
        photo_shape=deepcopy(row['gap']['shape_hw']),photo_to_metric_h=np.linalg.inv(native@geometry.metric_to_grid).tolist(),
        own_geometry_origin_metric_mm=[0.,0.],own_geometry_up_metric_xy=[0.,1.],
        left_bolts_photo_xy=None,bolt_heights_mm=[0,195,390],bottom_edge_photo_xy=own[[3,2]].tolist(),
        own_top_photo_xy=own[:2].tolist(),transported_top_photo_xy=transported[:2].tolist(),port_photo_xy=None,
        provenance=dict(scope='display-only current RAW route and existing own geometry; no new registration/scale',
            selection_sources=selection_sources,
            selected_rule=selection['selected_rule'],current_model=deepcopy(row['target_model']),
            current_profile=deepcopy(geometry.model['camera_profile']),camera_u_to_display_native_h=native.tolist(),
            display_loader='darsia.fluidflower.native.raw_photograph.load_camera_u_rgb8',
            photo_grid='current-native-display-render',photo_to_display_h=identity,display_grid=_grid(row),
            bolt_observations='No exact observed baseline-bolt transfer is bound for this clear-water display plane'))
    if selection.get('source_exception') is not None:
        basis['provenance']['source_specific_instrument_mapping']=deepcopy(selection['instrument'])
    return basis


def _grid(row):
    return dict(format='darsia-painted-review-display-grid-v1',native_photo=deepcopy(row['native_photo']),
        native_photo_pixels=deepcopy(row['native_photo_pixels']),model=deepcopy(row['target_model']),
        shape_hw=deepcopy(row['gap']['shape_hw']),native_to_photo_h=np.eye(3).tolist())


def _state(review,run_id):
    rows={row['run_id']:row for row in review['runs']}
    if run_id not in rows:
        raise ValueError('The selected run is absent from the accepted native review')
    context=validate_display_context(review['display_context'],review)
    return context,rows,rows[run_id]


def _position_model(row,geometry,certificate):
    frame=geometry.model.get('metric_frame') or {}
    if (geometry.model.get('grid')!='camera_u' or frame.get('unit')!='mm'
            or frame.get('origin')!='inner left-wall and bottom intersection'
            or frame.get('x_axis')!='right' or frame.get('y_axis')!='up'):
        raise ValueError('The original metric/camera-U coordinate convention is not explicitly supported')
    return dict(metric_to_camera_u=geometry.metric_to_grid.tolist(),metric_units=frame['unit'],
        source_profile=deepcopy(geometry.model['camera_profile']),camera_u_context=_camera_u_context(geometry,certificate),
        provenance=dict(model=deepcopy(row['target_model']),renderer=deepcopy(geometry.model['baseline_native_grid']['renderer']),
            metric_frame=deepcopy(frame),raw_acquisition=deepcopy(geometry.reference_image)))


def read_position_metadata(review,run_id,*,shift_tolerance_mm=2.,angle_tolerance_degrees=.5):
    """Metadata-only; no native RGB/label array is loaded for these readings."""
    from .painted_review_position import position_readout
    review=_review(review)
    context,rows,_=_state(review,run_id)
    order=_order(context['declared_run_order']);index=order.index(run_id)
    needed={run_id,review['source']['reference_run']}
    if index:
        needed.add(order[index-1])
    models={};errors={}
    for run in needed:
        if run in rows:
            try:
                geometry,_,certificate=_model(rows[run])
                models[run]=_position_model(rows[run],geometry,certificate)
            except (KeyError,TypeError,ValueError,OSError) as error:
                errors[run]=str(error)
    result=position_readout(models,run_id,reference_run=review['source']['reference_run'],
        declared_order=order,anchor_metric_mm=context['anchor_metric_mm'],
        shift_tolerance_mm=shift_tolerance_mm,angle_tolerance_degrees=angle_tolerance_degrees)
    if errors:
        result['source_unavailable']=errors
    return result


def read_gap_preview(review,run_id,component_id=None,*,display_rgb=None):
    """Decorate one original native preview with source-bound display choices."""
    from .painted_analysis_exclusions import read_gap_preview as original
    review=read_review(review)
    packet=(original(review,run_id,component_id) if display_rgb is None else
            original(review,run_id,component_id,display_rgb=display_rgb))
    rows={row['run_id']:row for row in review['runs']};row=rows[run_id]
    packet['display_grid']=_grid(row)
    choices=[dict(image_id='clearwater',label='Klarvann',available=False,reason=CLEARWATER_LIMIT),
        dict(image_id='first-experiment',label='Første forsøksbilde',available=False,reason='No explicit selected-first-injection RAW role receipt is bound'),
        dict(image_id='previous-run',label='Forrige forsøk',available=False,reason='The full declared series order is not bound')]
    try:
        context,rows,row=_state(review,run_id)
        model,_,certificate=_model(row)
        choices[0]['reason']=context['limitations'].get('clearwater',CLEARWATER_LIMIT)
        try:
            _clear_role(context,row)
            choices[0].update(available=True,reason='Original selected clear-water RAW; render on selection through the current profile and fixed own-model native transform, display only')
        except (KeyError,TypeError,ValueError,OSError) as error:
            choices[0]['reason']=str(error)
        try:
            _first_role(context,review,row,model,certificate)
            choices[1].update(available=True,reason='Original selected-first injection RAW and accepted native RGB are exactly bound')
        except (KeyError,TypeError,ValueError,OSError) as error:
            choices[1]['reason']=str(error)
        order=_order(context['declared_run_order']);index=order.index(run_id)
        previous=order[index-1] if index else None
        if previous is None:
            choices[2]['reason']='This is the first run in the declared series order'
        elif previous not in rows:
            choices[2]['reason']=f'The immediate declared previous run {previous.upper()} has no accepted native review product; no run is skipped'
        else:
            other,_,other_certificate=_model(rows[previous])
            if _profile_equal(model,certificate,other,other_certificate):
                choices[2].update(available=True,reason='Immediate declared previous native RGB on the exact same calibrated camera-U profile')
            else:
                choices[2]['reason']='The immediate previous run has a different exact calibrated camera-U profile'
        packet['position_readout']=read_position_metadata(review,run_id)
        packet['position_tolerances']=dict(shift_mm=2.,rotation_degrees=.5)
    except (KeyError,TypeError,ValueError,OSError,ImportError) as error:
        packet['display_context_unavailable']=str(error)
    packet['image_choices']=choices
    return packet


def _display_warp(rgb,transform,shape):
    import cv2
    from darsia._opencv_runtime import scientific_thread,require_ipp_off
    scientific_thread('painted-review-display-warp:before')
    height,width=shape
    image=cv2.warpPerspective(rgb,transform,(width,height),flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    support=cv2.warpPerspective(np.ones(rgb.shape[:2],np.float32),transform,(width,height),
        flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)==1.
    require_ipp_off('painted-review-display-warp:after')
    return image,support


def read_comparison_preview(review,run_id,component_id,image_id,*,photo_reader=None):
    """Load one original RGB; any transport is display only on the current grid."""
    from .painted_analysis_exclusions import read_gap_preview as original
    review=_review(review)
    context,rows,row=_state(review,run_id)
    target,n_target,certificate=_model(row)
    if image_id=='clearwater':
        selection=_clear_role(context,row)
        if photo_reader is None:
            from .native.raw_photograph import load_camera_u_rgb8
            photo_reader=load_camera_u_rgb8
        profile=target.model['camera_profile']
        from darsia._opencv_runtime import scientific_thread,require_ipp_off
        scientific_thread('painted-review-current-raw:before')
        source_options=({} if selection.get('source_exception') is None else
                        dict(source_exception=deepcopy(selection['source_exception'])))
        camera_rgb=photo_reader(Path(selection['selected_raw']['path']),profile_path=Path(profile['path']),
            profile_sha256=profile['sha256'],profile_id=profile['profile_id'],factor=1,
            expected_sha256=selection['selected_raw']['sha256'],**source_options)
        require_ipp_off('painted-review-current-raw:after')
        if (not isinstance(camera_rgb,np.ndarray) or camera_rgb.dtype!=np.uint8
                or camera_rgb.shape!=tuple(target.shape_hw)+(3,)):
            raise ValueError('The original current-profile RAW display renderer returned another camera-U RGB grid')
        image,support=_display_warp(camera_rgb,n_target,row['gap']['shape_hw'])
        basis=_water_basis(row,target,n_target,selection)
        from .painted_water_surface import propose_water_surface
        proposal=propose_water_surface(image,basis=basis)
        validate_display_context(context,review)
        _clear_role(context,row);_model(row)
        grid=_grid(row)
        return dict(image_rgb=image,image_encoding='rgb_uint8',display_grid=grid,native_to_photo_h=np.eye(3).tolist(),
            valid_support=support,image_role='clearwater',source_raw=deepcopy(selection['selected_raw']),
            display_transform=n_target.tolist(),water_surface=dict(basis=basis,proposal=proposal,action=None,
                measurements=None,photo_to_display_h=np.eye(3).tolist(),display_grid=grid))
    current=original(review,run_id,component_id)
    if image_id=='first-experiment':
        _first_role(context,review,row,target,certificate)
        _model(row)
        return dict(image_rgb=np.array(current['image_rgb'],copy=True),image_encoding=current['image_encoding'],
            display_grid=_grid(row),native_to_photo_h=deepcopy(current['native_to_photo_h']),
            valid_support=np.ones(current['image_rgb'].shape[:2],bool),image_role='first-experiment')
    if image_id!='previous-run':
        raise ValueError('The requested review photograph role is not supported')
    order=_order(context['declared_run_order']);index=order.index(run_id)
    if index==0 or order[index-1] not in rows:
        raise ValueError('The immediate declared previous run has no accepted native RGB; no run is skipped')
    previous=rows[order[index-1]]
    source,n_source,source_certificate=_model(previous)
    if not _profile_equal(target,certificate,source,source_certificate):
        raise ValueError('Previous and current native RGB use different exact calibrated camera-U profiles')
    previous_packet=original(review,previous['run_id'],None)
    transform=n_target @ target.metric_to_grid @ np.linalg.inv(source.metric_to_grid) @ np.linalg.inv(n_source)
    image,support=_display_warp(previous_packet['image_rgb'],transform,current['image_rgb'].shape[:2])
    validate_display_context(context,review)
    _model(row);_model(previous)
    return dict(image_rgb=image,image_encoding=previous_packet['image_encoding'],display_grid=_grid(row),
        native_to_photo_h=deepcopy(current['native_to_photo_h']),valid_support=support,
        image_role='previous-run',source_run_id=previous['run_id'],display_transform=transform.tolist())


ATTACHMENTS_FORMAT='darsia-painted-review-display-attachments-v1'


def make_display_attachments(review,*,position_policy=None,water_policy=None,water_actions=None):
    """Compact metadata-only snapshot for an owned reader or Check worker."""
    value=_json(review)
    return dict(format=ATTACHMENTS_FORMAT,review=_descriptor(value['review_binding']),
        display_context_sha256=context_source_sha256(value['display_context']),
        position=_json(position_policy or dict(method=METHOD,shift_tolerance_mm=2.,angle_tolerance_degrees=.5)),
        water=dict(**_json(water_policy or dict(height_tolerance_mm=3.,angle_tolerance_degrees=.5)),
                   actions=_json(water_actions or {})))


def expected_water_surface_basis(review,run_id):
    """Reconstruct from original RAW selection/current model, never supplied DATA."""
    review=_review(review)
    context,_,row=_state(review,run_id)
    selection=_clear_role(context,row)
    geometry,native,_=_model(row)
    return _water_basis(row,geometry,native,selection)


def validate_display_attachments(value,*,review=None):
    """Fresh source validation for Check/seal/Store, without binary photo reads."""
    value=_json(value)
    if (not isinstance(value,dict) or set(value)!={'format','review','display_context_sha256','position','water'}
            or value['format']!=ATTACHMENTS_FORMAT):
        raise ValueError('An exact compact display attachment is required')
    descriptor=_descriptor(value['review'])
    if not _same(descriptor,value['review']):
        raise ValueError('The display review descriptor has unexpected fields')
    retained=read_review(descriptor['path'])
    if not _same(_descriptor(retained['review_binding']),value['review']):
        raise ValueError('The display attachment review binding changed')
    if review is not None and not _same(_descriptor(review['review_binding']),value['review']):
        raise ValueError('The display attachment belongs to another preparation/review')
    context=validate_display_context(retained['display_context'],retained)
    if value['display_context_sha256']!=context_source_sha256(context):
        raise ValueError('The retained display context changed')
    position,water=value['position'],value['water']
    if (not isinstance(position,dict) or set(position)!={'method','shift_tolerance_mm','angle_tolerance_degrees'}
            or position['method']!=METHOD or not isinstance(water,dict)
            or set(water)!={'height_tolerance_mm','angle_tolerance_degrees','actions'} or not isinstance(water['actions'],dict)):
        raise ValueError('Display policies need their exact explicit method/tolerances')
    for item in (position['shift_tolerance_mm'],position['angle_tolerance_degrees'],water['height_tolerance_mm'],water['angle_tolerance_degrees']):
        try:
            finite=type(item) in (int,float) and math.isfinite(item)
        except OverflowError:
            finite=False
        if not finite or item<0:
            raise ValueError('Display warning tolerances must be finite nonnegative numbers')
    for run,action in water['actions'].items():
        if not isinstance(action,dict) or action.get('run_id')!=run:
            raise ValueError('The human water action belongs to another run')
        basis=expected_water_surface_basis(retained,run)
        from .painted_water_surface import water_surface_measurements
        water_surface_measurements(action,expected_basis=basis)
    return value


def display_findings(review,attachments=None):
    """Reconstruct compact preferences and advisory issues for Check and Store.

    Only original metadata is read for camera registration. Any confirmed water
    action must additionally have its expected clearwater basis reconstructed;
    a basis supplied in the attachment is never accepted. The digest covers both
    policies/actions and ordered issues, even when changing a tolerance produces
    the same warning list.
    """
    from .painted_review_position import position_readout
    review=read_review(review)
    if review.get('display_context') is None:
        if attachments is not None:
            raise ValueError('Display preferences require the original bound display context')
        return dict(display_attachments=None,issues=[],display_findings_sha256=display_context_digest(
            dict(display_attachments=None,issues=[])))
    context=validate_display_context(review['display_context'],review)
    attachment=make_display_attachments(review) if attachments is None else _json(attachments)
    attachment=validate_display_attachments(attachment,review=review)
    position,water=attachment['position'],attachment['water']
    rows={row['run_id']:row for row in review['runs']};models={};errors={}
    admitted=[]
    for run,row in rows.items():
        try:
            geometry,_,certificate=_model(row)
            models[run]=_position_model(row,geometry,certificate)
            admitted.append(row)
        except (KeyError,TypeError,ValueError,OSError) as error:
            errors[run]=str(error)
    order=_order(context['declared_run_order']);issues=[]
    for run in order:
        if run not in rows:
            continue
        readout=position_readout(models,run,reference_run=review['source']['reference_run'],declared_order=order,
            anchor_metric_mm=context['anchor_metric_mm'],shift_tolerance_mm=position['shift_tolerance_mm'],
            angle_tolerance_degrees=position['angle_tolerance_degrees'])
        for comparison in ('reference','previous'):
            measurement=readout[comparison]
            if measurement is not None and measurement['warning']:
                issues.append(dict(severity='warning',code='painted_review_camera_registration',
                    message='The existing map-placement camera registration exceeds the editable advisory tolerance; this is a local reading, not a physical movement verdict.',
                    details=dict(run_id=run,comparison=comparison,method=METHOD,measurement=measurement,
                        display_context_sha256=attachment['display_context_sha256'])))
        if run in errors:
            issues.append(dict(severity='info',code='painted_review_position_unavailable',
                message=errors[run],details=dict(run_id=run,display_context_sha256=attachment['display_context_sha256'])))
    bases={run:expected_water_surface_basis(review,run) for run in water['actions']}
    if bases:
        from .painted_water_surface import water_surface_findings
        for row in water_surface_findings(water['actions'],bases,height_tolerance_mm=water['height_tolerance_mm'],
                angle_tolerance_degrees=water['angle_tolerance_degrees']):
            if row['severity']=='warning':
                issues.append(dict(severity='warning',code='painted_review_water_surface',
                    message='The explicitly confirmed water surface differs beyond the editable display tolerance; the existing frame and scale are unchanged.',
                    details=row))
    for row in admitted:
        _model(row)
    validate_display_context(context,review)
    return dict(display_attachments=attachment,issues=issues,display_findings_sha256=display_context_digest(
        dict(display_attachments=attachment,issues=issues)))
