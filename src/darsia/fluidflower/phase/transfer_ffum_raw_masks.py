"""Pull original UM raster regions onto the full RAW canvas as sparse masks."""
import argparse
from pathlib import Path
import time

import numpy as np

from .ffum_raw_input import UmRelativeMapping
from .ffum_raw_numerics import load_numerics
from .ffum_review_mapping import UmReviewMapping
from .train_ffum_raw_model import bound, emit, read, sha, write


def domain_from_reference(reference):
    path = bound(reference['path'], reference['sha256'])
    if reference.get('kind') == 'registered-context':
        with np.load(path, allow_pickle=False) as archive:
            return archive['domain'].astype(bool)
    if path.name != 'facies.npz':
        raise ValueError('Unknown bound historical UM domain representation')
    # The original UM native_domain rule; no classification is recomputed.
    with np.load(path, allow_pickle=True) as archive:
        return archive['array'] != 1


def sparse_region(payload, native, mapper, raw_mapping, domain, annotation_window, *, chunk_rows=64):
    """Exact nearest-pixel membership, bounded by a conservative mapped window.

    Every source cell edge is sampled at its piecewise-bilinear breakpoints.
    Positive Jacobians were checked on every cell of the bound curvature map.
    The envelope bounds computation only; it never substitutes for the mask.
    """
    bounds, original = annotation_window(payload, canvas_wh=native['canvas_wh'],
        content_box=native['content_box_xywh'])
    x0, y0, x1, y1 = bounds; h, w = domain.shape
    left, right = max(0, x0-.5), min(w-1, x1-.5)
    top, bottom = max(0, y0-.5), min(h-1, y1-.5)
    xx = np.unique(np.r_[left, np.arange(np.ceil(left), np.floor(right)+1), right])
    yy = np.unique(np.r_[top, np.arange(np.ceil(top), np.floor(bottom)+1), bottom])
    edge = np.concatenate((np.column_stack((xx, np.full(len(xx), top))),
        np.column_stack((xx, np.full(len(xx), bottom))),
        np.column_stack((np.full(len(yy), left), yy)),
        np.column_stack((np.full(len(yy), right), yy))))
    transformed = mapper.content_to_display(edge)
    if not np.isfinite(transformed).all():
        raise ValueError('Original UM region window leaves its historical content')
    lower = np.maximum(0, np.floor(transformed.min(0)).astype(int)-2)
    upper = np.minimum(raw_mapping.shape_hw[::-1], np.ceil(transformed.max(0)).astype(int)+3)
    target_shape = tuple(map(int, (upper-lower)[::-1]))
    annotation = np.zeros(target_shape, bool); training = np.zeros(target_shape, bool)
    maximum_roundtrip = 0.
    for start in range(0, target_shape[0], chunk_rows):
        stop = min(start+chunk_rows, target_shape[0])
        yy, xx = np.mgrid[start:stop, :target_shape[1]]
        points = np.column_stack((xx.ravel()+lower[0], yy.ravel()+lower[1]))
        content = mapper.display_to_content(points)
        finite = np.isfinite(content).all(1)
        nearest = np.zeros(content.shape, np.int64)
        nearest[finite] = np.floor(content[finite]+.5).astype(np.int64)
        cx, cy = nearest.T
        valid = finite & (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        for channel in 'RGB':
            sensor = raw_mapping.display_to_sensor(points, channel)
            valid &= ((sensor[:, 0] >= 0) & (sensor[:, 0] <= raw_mapping.sensor_wh[0]-1)
                & (sensor[:, 1] >= 0) & (sensor[:, 1] <= raw_mapping.sensor_wh[1]-1))
        take = np.flatnonzero(valid)
        chosen = original[cy[take]-y0, cx[take]-x0]
        take = take[chosen]
        annotation[start:stop].reshape(-1)[take] = True
        training[start:stop].reshape(-1)[take] = domain[cy[take], cx[take]]
        if len(take):
            checked = take[::max(1, len(take)//128)]
            error = np.linalg.norm(mapper.content_to_display(content[checked])-points[checked], axis=1)
            maximum_roundtrip = max(maximum_roundtrip, float(error.max()))
    if maximum_roundtrip > .01:
        raise ValueError('Transferred UM mask failed pixel-coordinate roundtrip')
    return dict(bbox_xyxy=[*lower.tolist(), *upper.tolist()], shape_hw=list(target_shape),
        original_annotation_pixels=int(original.sum()),
        original_training_pixels=int((original & domain[y0:y1, x0:x1]).sum()),
        transferred_annotation_pixels=int(annotation.sum()), transferred_training_pixels=int(training.sum()),
        roundtrip_max_display_px=maximum_roundtrip), annotation, training


def transfer(geometry, output, profile, snapshot, *, run=None, limit=None):
    geometry, output = Path(geometry), Path(output)
    method = load_numerics(snapshot); relative = UmRelativeMapping(read(profile))
    files = sorted(geometry.glob('*_frame_*.json'))
    if run: files = [p for p in files if read(p)['run'] == run]
    if limit: files = files[:limit]
    for path in files:
        started = time.monotonic(); receipt = read(path)
        target = output/path.stem
        binding = dict(geometry_sha256=sha(path), profile_sha256=sha(profile),
                       numerical_source_sha256=sha(Path(snapshot)/'source_binding.json'))
        if target.with_suffix('.json').exists():
            saved = read(target.with_suffix('.json'))
            if saved['binding'] != binding: raise ValueError('UM region mask binding changed')
            bound(target.with_suffix('.npz'), saved['masks_sha256']); continue
        native = read(bound(receipt['original_manifest'], receipt['original_manifest_sha256']))
        with np.load(receipt['curvature_cache']) as archive:
            mapper = UmReviewMapping(archive['grid'], tuple(map(int, archive['shape'])),
                np.asarray(receipt['drift_matrix']), tuple(receipt['photo_wh']), tuple(receipt['resized_wh']),
                tuple(receipt['display_origin_jpg_xy']))
        mapper.validate_all_cells()
        if tuple(receipt['display_origin_jpg_xy']) != relative.origin_jpg_xy:
            raise ValueError('UM mask and profile origin mismatch')
        domain = domain_from_reference(receipt['physical_context'])
        arrays, records = {}, []
        for index, region in enumerate(receipt['regions']):
            detail, annotation, training = sparse_region(region['original_record']['payload'],
                native, mapper, relative, domain, method.annotation.annotation_window)
            key = f'region_{index:04d}'
            arrays[key+'_annotation'] = np.packbits(annotation, bitorder='little')
            arrays[key+'_training'] = np.packbits(training, bitorder='little')
            records.append(dict(region_id=region['region_id'], original_record=region['original_record'],
                original_extraction=region['original_extraction'], array_prefix=key, **detail,
                training_disposition='eligible according to original record' if receipt['raw'] else 'excluded: missing original RAW; user approved 2518 regional boxes'))
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target.with_suffix('.npz'), **arrays)
        write(target.with_suffix('.json'), dict(binding=binding, run=receipt['run'], frame_index=receipt['frame_index'],
            full_canvas_shape_hw=list(relative.shape_hw), full_canvas_origin_jpg_xy=list(relative.origin_jpg_xy),
            bitorder='little', masks_sha256=sha(target.with_suffix('.npz')), records=records,
            membership='exact original raster, nearest source pixel; separate original-domain training mask',
            source_raw_available=receipt['raw'] is not None, categorical_interpolation='nearest'))
        emit('region_masks', run=receipt['run'], index=receipt['frame_index'], regions=len(records),
             seconds=round(time.monotonic()-started, 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('geometry', 'output', 'profile', 'snapshot'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--run'); parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    transfer(args.geometry, args.output, args.profile, args.snapshot, run=args.run, limit=args.limit)


if __name__ == '__main__': main()
