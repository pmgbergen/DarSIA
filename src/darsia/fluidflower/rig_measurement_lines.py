"""Measured distances and their independently tracked image endpoints."""
from copy import deepcopy
import math
import numpy as np
from . import rig_static_references as refs

MAX_LINES = 256
_UNSET = object()
_DIRECTIONS = {'x+': (1., 0.), 'x-': (-1., 0.), 'y+': (0., 1.), 'y-': (0., -1.)}


def _points(value):
    result = refs._array(value, (2,2), 'Measurement endpoints')
    if np.linalg.norm(result[1]-result[0]) < 1e-6:
        raise ValueError('Choose two different endpoints for the measurement line')
    return result


def _length(value):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('Enter a finite physical length in millimetres') from error
    if not math.isfinite(result) or not 0 <= result <= 1e6:
        raise ValueError('Physical length must be between 0 and 1000000 mm')
    return result


def _tracking_identity(line):
    return refs._digest({key:line[key] for key in ('id','source','points_u_xy','endpoint_regions')})


def _placement(value):
    if (not isinstance(value, dict) or not isinstance(value.get('direction'), str)
            or value['direction'] not in _DIRECTIONS):
        raise ValueError('Measurement placement needs a start x/y and direction x+, x-, y+ or y-')
    start = refs._array(value.get('start_xy_mm'), (2,), 'Measurement start x/y')
    if np.any(np.abs(start) > 1e6):
        raise ValueError('Measurement start coordinates must be within 1000000 mm of the rig origin')
    return dict(start_xy_mm=start.tolist(), direction=value['direction'])


def measurement_metric_points(line):
    """Return explicitly placed physical endpoints; a distance alone has no origin."""
    placement = line.get('placement')
    length = _length(line.get('length_mm'))
    if placement is None or length <= 0:
        return None
    placement = _placement(placement)
    start = np.asarray(placement['start_xy_mm'])
    end = start + length*np.asarray(_DIRECTIONS[placement['direction']])
    if np.any(np.abs(end) > 1e6):
        raise ValueError('Measurement endpoints must be within 1000000 mm of the rig origin')
    return np.vstack((start, end)).tolist()


def capture_measurement_line(packet, *, line_id, name, points_u_xy, length_mm=0., plane_id='rig'):
    points = _points(points_u_xy)
    refs._identifier(line_id, 'Measurement line ID')
    refs._identifier(plane_id, 'Measurement plane')
    if not isinstance(name,str) or not name.strip() or len(name)>256:
        raise ValueError('Give the measurement line a name')
    image, support, matrix, photo = refs._packet(packet)
    projected = refs._project(matrix,points)
    height,width = image.shape[:2]
    endpoints, notes = [], []
    for index, (anchor, pixel) in enumerate(zip(points, projected)):
        x,y = np.rint(pixel).astype(int)
        if not (0<=x<width and 0<=y<height and support[y,x]):
            raise ValueError('Place measurement endpoints on available photograph pixels')
        endpoint = None
        note = 'No distinctive static texture near this endpoint; its source position can still be measured manually'
        for radius in (48,32,16,8):
            low = np.maximum(pixel-radius,[0.,0.])
            high = np.minimum(pixel+radius,[width-1.,height-1.])
            corners = refs._project(np.linalg.inv(matrix),[low,[high[0],low[1]],high,[low[0],high[1]]])
            a,b = corners.min(axis=0),corners.max(axis=0)
            try:
                endpoint = refs.capture_region(packet,region_id=f'{line_id}_{index}',
                    name=f'{name.strip()} endpoint {index+1}',rect_u_xywh=[*a,*(b-a)],
                    anchor_u_xy=anchor.tolist(),metric_xy_mm=[0.,0.],plane_id='distance_endpoints')
                # Endpoints have no absolute physical x/y; they must never fit a frame.
                endpoint['purpose']='measurement_endpoint'
                note = ''
                break
            except ValueError as error:
                note = str(error)
        endpoints.append(endpoint)
        notes.append(note)
    result = dict(id=line_id,name=name.strip(),points_u_xy=points.tolist(),length_mm=_length(length_mm),
        plane_id=plane_id,source=photo,source_run_id=packet['run_id'],endpoint_regions=endpoints,
        tracking_notes=notes)
    result['tracking_identity']=_tracking_identity(result)
    return result


def validate_measurement_lines(lines):
    if not isinstance(lines,list) or len(lines)>MAX_LINES:
        raise ValueError(f'At most {MAX_LINES} measurement lines can be saved')
    seen=set()
    for line in lines:
        if not isinstance(line,dict):
            raise ValueError('Each measurement line must be an object')
        key=refs._identifier(line.get('id'),'Measurement line ID')
        if key in seen:
            raise ValueError('Measurement line IDs must be unique')
        seen.add(key)
        refs._identifier(line.get('source_run_id'),'Measurement source run')
        refs._identifier(line.get('plane_id'),'Measurement plane')
        if not isinstance(line.get('name'),str) or not line['name'].strip() or len(line['name'])>256:
            raise ValueError('Each measurement line needs a name')
        _points(line.get('points_u_xy'))
        _length(line.get('length_mm'))
        if line.get('placement') is not None:
            _placement(line['placement'])
            measurement_metric_points(line)
        source=line.get('source',{})
        if source.get('identity') != refs._digest({k:v for k,v in source.items() if k!='identity'}):
            raise ValueError('The measurement source photograph binding changed')
        endpoints=line.get('endpoint_regions')
        if not isinstance(endpoints,list) or len(endpoints)!=2:
            raise ValueError('A measurement line needs two endpoint records')
        for endpoint in endpoints:
            if endpoint is not None:
                refs._validate_region(endpoint)
                if endpoint.get('purpose')!='measurement_endpoint':
                    raise ValueError('A measurement endpoint cannot define absolute physical coordinates')
        if line.get('tracking_identity') != _tracking_identity(line):
            raise ValueError('The measurement line endpoint binding changed')
    return deepcopy(lines)


def update_measurement_line(line, *, name=None, length_mm=None, plane_id=None, placement=_UNSET):
    validate_measurement_lines([line])
    result=deepcopy(line)
    if name is not None:
        if not isinstance(name,str) or not name.strip() or len(name)>256:
            raise ValueError('Give the measurement line a name')
        result['name']=name.strip()
    if length_mm is not None:
        result['length_mm']=_length(length_mm)
    if plane_id is not None:
        result['plane_id']=refs._identifier(plane_id,'Measurement plane')
    if placement is not _UNSET:
        if placement is None:
            result.pop('placement', None)
        else:
            result['placement']=_placement(placement)
    measurement_metric_points(result)
    return result


def _geometry_identity(document, plane_id):
    """Bind a proposal to physical placement without invalidating tracked pixels."""
    regions = [{key: region.get(key) for key in ('id', 'identity', 'metric_xy_mm', 'plane_id')}
               for region in document.get('regions', [])
               if region.get('plane_id', 'rig') == plane_id and region.get('purpose') != 'measurement_endpoint']
    lines = [{key: line.get(key) for key in ('id', 'tracking_identity', 'length_mm', 'placement', 'plane_id')}
             for line in document.get('measurement_lines', [])
             if line.get('plane_id', 'rig') == plane_id and measurement_metric_points(line) is not None]
    return refs._digest(dict(regions=regions, lines=lines, plane_id=plane_id,
        measurement_uncertainty_mm=document.get('measurement_uncertainty_mm', 0.)))


def fit_measurement_reference_model(document, region_observations, measurement_observations, *, plane_id='rig'):
    """Combine real x/y references and explicitly placed measured endpoints.

    Image endpoints are accepted only for their exact tracking definition and a
    common photograph. Distances without placement and dashed guides supply no
    anchors. Geometry remains a proposal and uses the existing fit budgets.
    """
    refs._identifier(plane_id, 'Depth plane ID')
    lines = validate_measurement_lines(document.get('measurement_lines', []))
    regions = deepcopy(document.get('regions', []))
    observations = deepcopy(region_observations)
    used, unplaced, excluded, endpoint_ids = [], [], {}, {}
    photo_identities = set()
    for region in regions:
        observation = observations.get(region.get('id'), {})
        if (region.get('plane_id', 'rig') == plane_id and region.get('purpose') != 'measurement_endpoint'
                and observation.get('status') in refs._ACCEPTED and observation.get('photo_identity')):
            photo_identities.add(observation['photo_identity'])
    occupied = {region.get('id') for region in regions}
    for line in lines:
        if line['plane_id'] != plane_id:
            continue
        key = line['id']
        metric = measurement_metric_points(line)
        if metric is None:
            unplaced.append(key)
            continue
        observation = measurement_observations.get(key, {})
        if observation.get('status') not in refs._ACCEPTED:
            excluded[key] = 'Both endpoints need a recognized or manually corrected observation'
            continue
        if (observation.get('line_id') != key or
                observation.get('tracking_identity') != line['tracking_identity']):
            excluded[key] = 'The observed endpoints belong to another measurement definition'
            continue
        identity = observation.get('photo_identity')
        if not isinstance(identity, str) or not refs._HEX.fullmatch(identity):
            excluded[key] = 'The observed endpoints need an exact photograph binding'
            continue
        try:
            camera = _points(observation.get('points_u_xy'))
        except ValueError as error:
            excluded[key] = str(error)
            continue
        photo_identities.add(identity)
        used.append(key)
        for index, (xy, uv) in enumerate(zip(metric, camera)):
            anchor_id = 'ml_' + refs._digest([key, index])[:32]
            while anchor_id in occupied:
                anchor_id += '_'
            occupied.add(anchor_id)
            endpoint_ids[anchor_id] = (key, index)
            regions.append(dict(id=anchor_id, metric_xy_mm=xy, plane_id=plane_id))
            observations[anchor_id] = dict(status=observation['status'], anchor_u_xy=uv.tolist())
    output = refs.fit_reference_model(regions, observations, plane_id=plane_id,
        measurement_uncertainty_mm=document.get('measurement_uncertainty_mm', 0.))
    output.update(used_line_ids=used, unplaced_line_ids=unplaced, excluded_line_ids=excluded,
        geometry_reference_identity=_geometry_identity(document, plane_id))
    output['used_anchor_ids'] = output['used_region_ids'][:]
    output['used_region_ids'] = [key for key in output['used_region_ids'] if key not in endpoint_ids]
    for residual in output['residuals']:
        if residual['region_id'] in endpoint_ids:
            residual['line_id'], residual['endpoint_index'] = endpoint_ids[residual['region_id']]
    if len(photo_identities) > 1:
        output.update(status='stale', H_metric_mm_to_camera_u=None,
            reason='Reference observations belong to different photographs; find or redraw them in the current run')
    elif output['status'] == 'insufficient' and unplaced:
        output['reason'] += '; give measurement lines a start x/y and direction in the shared rig coordinate system'
    return output


def _observation(line,photo,points,status,score,reason=''):
    value=dict(line_id=line['id'],tracking_identity=line['tracking_identity'],
        photo_identity=photo['identity'],status=status,score=float(score),points_u_xy=None,
        length_u_px=None,length_mm=_length(line['length_mm']),reason=reason)
    value['measurement_pending']=value['length_mm']<=0
    if points is not None:
        array=_points(points)
        value.update(points_u_xy=array.tolist(),length_u_px=float(np.linalg.norm(array[1]-array[0])))
        value['segment_mm_per_u_px']=value['length_mm']/value['length_u_px'] if value['length_mm']>0 else None
    return value


def manual_measurement_observation(line,packet,points_u_xy):
    validate_measurement_lines([line])
    points=_points(points_u_xy)
    image,support,matrix,photo=refs._packet(packet)
    for x,y in np.rint(refs._project(matrix,points)).astype(int):
        if not (0<=y<support.shape[0] and 0<=x<support.shape[1] and support[y,x]):
            raise ValueError('Place both measurement endpoints on available photograph pixels')
    return _observation(line,photo,points,'manual',1.)


def measurement_for_photo(line,packet,result=None):
    validate_measurement_lines([line])
    photo=refs.photo_binding(packet)
    if result is not None and result.get('photo',{}).get('identity')==photo['identity']:
        observation=result.get('measurement_lines',{}).get(line['id'])
        if (observation is not None and observation.get('tracking_identity')==line['tracking_identity']
                and observation.get('photo_identity')==photo['identity']):
            value=deepcopy(observation)
            value['length_mm']=_length(line['length_mm'])
            value['measurement_pending']=value['length_mm']<=0
            span=value.get('length_u_px')
            value['segment_mm_per_u_px']=value['length_mm']/span if span and value['length_mm']>0 else None
            return value
    if line['source_run_id']==packet.get('run_id') and line['source']['identity']==photo['identity']:
        return _observation(line,photo,line['points_u_xy'],'manual',1.)
    return None


def recognize_measurement_lines(document,packet):
    lines=validate_measurement_lines(document.get('measurement_lines',[]))
    if not lines:
        return {}
    photo=refs.photo_binding(packet)
    target_problem=refs._binding_problem(photo)
    if target_problem:
        return {line['id']:_observation(line,photo,None,'stale',0.,target_problem) for line in lines}
    source_problems={}
    temporary=refs.empty_document()
    for line in lines:
        key=line['source']['identity']
        if key not in source_problems:
            source_problems[key]=refs._binding_problem(line['source'])
        if line['source']['identity']!=photo['identity'] and not source_problems[key]:
            temporary['regions'].extend(item for item in line['endpoint_regions'] if item is not None)
    # Distance endpoints carry no absolute x/y and are excluded from planar fitting.
    matched={}
    regions=temporary['regions']
    for start in range(0,len(regions),refs.MAX_REGIONS):
        part=refs.empty_document()
        part['regions']=regions[start:start+refs.MAX_REGIONS]
        matched.update(refs.recognize_regions(part,packet)['observations'])
    output={}
    for line in lines:
        problem=source_problems[line['source']['identity']]
        previous=document.get('runs',{}).get(packet.get('run_id'),{})
        manual=measurement_for_photo(line,packet,previous)
        if problem:
            output[line['id']]=_observation(line,photo,None,'stale',0.,problem)
        elif (manual is not None and manual.get('status')=='manual'
                and previous.get('photo',{}).get('identity')==photo['identity']):
            output[line['id']]=manual
        elif line['source_run_id']==packet.get('run_id') and line['source']['identity']==photo['identity']:
            output[line['id']]=_observation(line,photo,line['points_u_xy'],'manual',1.)
        else:
            endpoints=[matched.get(item['id'],{}) if item else {} for item in line['endpoint_regions']]
            accepted=all(item.get('status') in ('manual','recognized') and item.get('anchor_u_xy') is not None for item in endpoints)
            if accepted and np.linalg.norm(np.asarray(endpoints[1]['anchor_u_xy'])-endpoints[0]['anchor_u_xy'])<1e-6:
                accepted=False
            points=[item['anchor_u_xy'] for item in endpoints] if accepted else None
            status='recognized' if accepted else ('partial' if any(item.get('anchor_u_xy') is not None for item in endpoints) else 'missing')
            score=min((float(item.get('score',0.)) for item in endpoints),default=0.)
            output[line['id']]=_observation(line,photo,points,status,score,
                '' if accepted else 'Both measured endpoints must be found; redraw this run\'s line if necessary')
    return output
