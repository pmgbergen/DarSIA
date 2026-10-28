"""Photo-bound display guides in original camera-U coordinates."""
from copy import deepcopy
import numpy as np

MODES = {'through', 'horizontal', 'vertical'}
MAX_GUIDES = 1024


def _points(value):
    try:
        points = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError('A support line needs two finite image points') from error
    if points.shape != (2, 2) or not np.isfinite(points).all() or np.any(np.abs(points) > 1e9):
        raise ValueError('A support line needs two finite image points')
    if np.linalg.norm(points[1]-points[0]) < 1e-6:
        raise ValueError('Choose two different points for the support line')
    return points


def validate_guides(guides):
    if not isinstance(guides, list) or len(guides) > MAX_GUIDES:
        raise ValueError(f'At most {MAX_GUIDES} support lines can be saved')
    identifiers = set()
    for guide in guides:
        if not isinstance(guide, dict):
            raise ValueError('Each support line must be an object')
        for key in ('id', 'run_id', 'photo_identity'):
            if not isinstance(guide.get(key), str) or not guide[key] or len(guide[key]) > 256:
                raise ValueError('The support line needs its ID, run and photograph identity')
        if guide['id'] in identifiers:
            raise ValueError('Support line IDs must be unique')
        identifiers.add(guide['id'])
        if guide.get('mode') not in MODES or type(guide.get('extend')) is not bool:
            raise ValueError('The support line mode or extent is invalid')
        points = _points(guide.get('points_u_xy'))
        if guide['mode'] == 'horizontal' and points[0, 1] != points[1, 1]:
            raise ValueError('A horizontal support line must share one row')
        if guide['mode'] == 'vertical' and points[0, 0] != points[1, 0]:
            raise ValueError('A vertical support line must share one column')
    return deepcopy(guides)


def _identity(packet, supplied):
    if supplied is not None:
        return supplied
    from .rig_static_references import photo_binding
    return photo_binding(packet)['identity']


def make_guide(packet, *, guide_id, points_u_xy, mode='through', extend=True,
               photo_identity=None):
    if mode not in MODES:
        raise ValueError('Choose a line through the points, horizontal or vertical')
    points = np.asarray(points_u_xy, dtype=float).copy()
    if points.shape != (2, 2):
        raise ValueError('Choose two image points for the support line')
    if mode == 'horizontal':
        points[1, 1] = points[0, 1]
    elif mode == 'vertical':
        points[1, 0] = points[0, 0]
    guide = dict(id=guide_id, run_id=packet.get('run_id'),
        photo_identity=_identity(packet, photo_identity), points_u_xy=points.tolist(),
        mode=mode, extend=extend)
    return validate_guides([guide])[0]


def _matrix(packet):
    value = packet.get('camera_u_to_photo_h')
    if value is None:
        mapping = packet.get('mapping')
        if mapping is None:
            return np.eye(3)
        origin, x, y = np.asarray(mapping.camera_u_to_display_xy([[0,0],[1,0],[0,1]]), float)
        value = [[*(x-origin), 0], [*(y-origin), 0], [*origin, 1]]
        value = np.asarray(value).T
    result = np.asarray(value, dtype=float)
    if result.shape != (3, 3) or not np.isfinite(result).all() or abs(np.linalg.det(result)) < 1e-12:
        raise ValueError('The photograph has no usable camera-U display mapping')
    return result


def _project(matrix, points):
    result = np.column_stack([points, np.ones(len(points))]) @ matrix.T
    if np.any(np.abs(result[:, 2]) < 1e-12):
        raise ValueError('The support line crosses the display horizon')
    return result[:, :2] / result[:, 2:]


def guide_segment(guide, packet, *, photo_identity=None):
    """Return clipped camera-U endpoints; zoom and pan cannot change the line."""
    validate_guides([guide])
    if (guide['run_id'] != packet.get('run_id')
            or guide['photo_identity'] != _identity(packet, photo_identity)):
        return None
    image = packet.get('image_rgb')
    if not isinstance(image, np.ndarray) or image.ndim != 3 or min(image.shape[:2]) < 2:
        return None
    matrix = _matrix(packet)
    points = _project(matrix, _points(guide['points_u_xy']))
    delta = points[1]-points[0]
    lower, upper = (-np.inf, np.inf) if guide['extend'] else (0., 1.)
    for axis, maximum in enumerate((image.shape[1]-1., image.shape[0]-1.)):
        if abs(delta[axis]) < 1e-12:
            if not 0. <= points[0, axis] <= maximum:
                return None
            continue
        a, b = -points[0, axis]/delta[axis], (maximum-points[0, axis])/delta[axis]
        lower, upper = max(lower, min(a,b)), min(upper, max(a,b))
        if lower > upper:
            return None
    if not np.isfinite([lower, upper]).all() or upper-lower < 1e-12:
        return None
    return _project(np.linalg.inv(matrix), points[0]+np.array([lower,upper])[:,None]*delta)
