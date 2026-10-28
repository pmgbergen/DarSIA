"""Prepare immutable FF_BM RAW frames, region masks and internal feature inputs.

Execution is deliberately separate from production selection and shared runners.
An isolated caller supplies the preserved historical drift implementation.
"""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import tifffile
from PIL import Image

from ..native import registration
from .bm_raw_cohort import load_current_bm_cohort
from .bm_raw_photometry import encode_training_rgb
from .bm_raw_regions import BMReviewMapping, CurvatureInverse, content_index_map, sparse_region_mask
from .um_bm_raw_frame import read_full_raw_phase_frame
from .um_bm_review_regions import capture_legacy_drift, shared_data_root as data_root, shared_drive_path
from .um_bm_sampling import sample_full_frame
from ..phase_training.review_regions import annotation_mask


# Locations relative to the configured shared data folder (data_root).
MODELS = Path('FF_BM/DarSIA/models')
MODEL = MODELS/'phase-model-20260925T134533Z-2c8a15b8-ffbm-unified'
LEGACY = MODELS/'phase-model-20260921-ffbm-unified-small-regions'
OPTICAL = MODELS/'phase-model-20260915T140012Z-ffbm-decision'
CONFIG = Path('FF_BM/Analysis/Original/source/fluidflower_benchmark_image_analysis-1.0/image_analysis/phase_segmentation/config.json')
CONFIG_SHA = '7a69e615963d26b512072b434dd25fd72c75750187792617dcd2e4f9fcf66170'
GRID_SHA = '1f431600dc7595756ba82c98f3b39fb28b5b38564a2ed54cf104f04a0184e5c4'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def bound(path, expected):
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError(f'BM source checksum differs: {path}')
    return path


def resolve(path):
    """Map a recorded Z: path onto the configured shared data folder."""
    return shared_drive_path(path)


def bind_file_lens_tables(raw, parameters):
    """Populate a new per-file profile before the common reader verifies it.

    Sony records channel tables per exposure. A previous exposure's tables are
    not a series calibration. Only the new profile copy is modified here.
    """
    lens = parameters['common']['lens']
    with tifffile.TiffFile(raw) as tiff:
        sub = tiff.pages[lens['main_page']].pages[lens['sub_page']]
        channel = tuple(int(v) for v in sub.tags[lens['ca_tag']].value)
    if len(channel) < lens['ca_min_length'] or channel[0] != lens['ca_header']:
        raise ValueError('Unexpected Sony per-file channel-table structure')
    lens['ca_red'] = list(channel[slice(*lens['red_slice'])])
    lens['ca_blue'] = list(channel[slice(*lens['blue_slice'])])
    distortion, red, blue = registration.read_lens_tables(raw, parameters=lens)
    parameters['common']['profiles']['camera-u']['distortion'] = list(distortion)
    return distortion, red, blue


def create_plan(root, *, profile_path, profile_sha256, registration_path, registration_sha256):
    """Freeze source files and both historical/current label policies before work."""
    root = Path(root)
    shared = data_root()
    model, config = shared/MODEL, shared/CONFIG
    root.mkdir(parents=True, exist_ok=False)
    regions, cohort = load_current_bm_cohort(optical_model=shared/OPTICAL, legacy_model=shared/LEGACY,
                                            current_model=model, resolve_path=resolve)
    profile = json.loads(bound(profile_path, profile_sha256).read_text())
    measurement = json.loads(bound(registration_path, registration_sha256).read_text())
    offset = measurement['candidates']['shifted_crop']['offset_xy']
    if offset != [8, 424]:
        raise ValueError('The approved BM TIFF/sensor registration differs')
    if profile['camera_u_display']['camera-u'] != {'white_balance': [2.06640625, 1.0, 1.98828125],
            'exposure': 1.0, 'matrix': None, 'transfer': 'srgb'}:
        raise ValueError('The approved fixed BM photometry differs')
    bound(config, CONFIG_SHA); bound(config.parent/'cache/curvature_transformation.npy', GRID_SHA)
    frames = {r.frame.frame_index: r.frame for r in regions}
    manifest = next(iter(frames.values())).manifest
    wanted = sorted({0, *frames, *(i-1 for i in frames if i > 0)})
    raw_root = shared/'FF_BM/Resfys_copy/Original data/FF_BM_cycle_1/C1_images'
    raw_folders = [raw_root/'211124_c1_injection_20s', raw_root/'211125_003600_c1_mixing_5min']
    inputs = []
    for index in wanted:
        source = manifest['frame_sources'][index]
        photo = bound(resolve(source['source_path']), source['source_sha256'])
        name = photo.stem.split('_')[-1]+'.ARW'
        matches = [folder/name for folder in raw_folders if (folder/name).is_file()]
        if len(matches) != 1:
            raise ValueError(f'No unique original BM RAW for {photo.name}')
        inputs.append(dict(frame_index=index, photo=dict(path=str(photo), sha256=source['source_sha256']),
                           raw=dict(path=str(matches[0]), sha256=sha256(matches[0])),
                           region_ids=[r.record['region_id'] for r in regions if r.frame.frame_index == index]))
    write(root/'cohort.json', cohort)
    write(root/'evidence.json', dict(regions=[r.record for r in regions]))
    plan = dict(series='FF_BM', run='c1', source_model=str(model), source_manifest_sha256=next(iter(frames.values())).manifest_sha256,
                baseline_frame_index=0, frames=inputs, annotated_frames=sorted(frames),
                profile=dict(path=str(profile_path), sha256=profile_sha256),
                registration=dict(path=str(registration_path), sha256=registration_sha256, tiff_to_sensor_xy=offset),
                config=dict(path=str(config), sha256=CONFIG_SHA),
                curvature=dict(path=str(config.parent/'cache/curvature_transformation.npy'), sha256=GRID_SHA),
                input_contract='fixed BM camera white balance and sRGB transfer; float32, no JPEG quantisation',
                numerical_representation='RAW retained full-frame; independently sampled physical content grid for features',
                frozen_optical_heads=[], fit_heads=['co2_presence', 'gas_within_co2'],
                evaluation_scope='training-only; no independent generalisation claim')
    write(root/'plan.json', plan)
    return plan


def _shared_indices(root, mapping, support):
    identity = dict(mapping=asdict(mapping.raw_mapping), drift=mapping.drift_matrix.tolist(),
                    tiff_to_sensor_xy=mapping.tiff_to_sensor_xy, photo_wh=mapping.photo_wh,
                    resized_wh=mapping.resized_wh, curvature_sha256=GRID_SHA,
                    valid_support_sha256=hashlib.sha256(support.tobytes()).hexdigest())
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache = root/'coordinate_cache'; cache.mkdir(exist_ok=True)
    path = cache/f'{key}.npy'; lock = cache/f'{key}.lock'; receipt = cache/f'{key}.json'
    deadline = time.monotonic()+600
    while not receipt.exists():
        try:
            token = lock.open('x'); token.close()
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError('BM coordinate cache owner did not finish')
            time.sleep(1)
            continue
        try:
            mapped = np.lib.format.open_memmap(path, mode='w+', dtype=np.int32, shape=support.shape)
            content_index_map(mapping, support.shape, support, output=mapped)
            mapped.flush(); del mapped
            write(receipt, dict(identity=identity, sha256=sha256(path)))
        finally:
            lock.unlink(missing_ok=True)
    record = json.loads(receipt.read_text())
    bound(path, record['sha256'])
    return np.load(path, mmap_mode='r'), dict(path=str(path), sha256=record['sha256'])


def prepare_frame(root, index, *, drift_factory):
    """Compute one frame with the exact source-bound historical TIFF drift call."""
    started = time.monotonic(); root = Path(root)
    plan = json.loads((root/'plan.json').read_text())
    row = next(r for r in plan['frames'] if r['frame_index'] == index)
    folder = root/f'frame_{index:03d}'; folder.mkdir(exist_ok=False)
    photo = bound(**dict(path=row['photo']['path'], expected=row['photo']['sha256']))
    raw = bound(row['raw']['path'], row['raw']['sha256'])
    config = json.loads(bound(plan['config']['path'], CONFIG_SHA).read_text())
    baseline_row = next(r for r in plan['frames'] if r['frame_index'] == plan['baseline_frame_index'])
    with Image.open(photo) as im:
        photo_rgb = np.asarray(im.convert('RGB')); photo_wh = im.size
    if photo_wh != (7952, 4472):
        raise ValueError('Historical BM no-resize TIFF chain has changed dimensions')
    if index == plan['baseline_frame_index']:
        matrix = np.array([[1., 0., 0.], [0., 1., 0.]])
    else:
        with Image.open(bound(baseline_row['photo']['path'], baseline_row['photo']['sha256'])) as im:
            baseline_rgb = np.asarray(im.convert('RGB'))
        cv2.setRNGSeed(0)
        _, matrix = capture_legacy_drift(drift_factory(baseline_rgb, config=config['drift']), photo_rgb)
        del baseline_rgb
    del photo_rgb
    profile = json.loads(bound(plan['profile']['path'], plan['profile']['sha256']).read_text())
    parameters = profile['registration']
    bind_file_lens_tables(raw, parameters)
    profile['provenance'] = dict(parent=plan['profile'], raw=row['raw'], series='FF_BM', frame_index=index,
                                  change='source file lens tables; same fixed Sony coefficients and colour contract')
    profile_path = folder/'profile.json'; write(profile_path, profile)
    frame = read_full_raw_phase_frame(raw, series='FF_BM', profile_path=profile_path,
                profile_sha256=sha256(profile_path), profile_id='camera-u', expected_sha256=row['raw']['sha256'])
    np.save(folder/'raw_linear_dn.npy', frame.linear_rgb_chw)
    np.save(folder/'raw_valid_support.npy', frame.valid_support)
    inverse = CurvatureInverse(np.load(bound(plan['curvature']['path'], GRID_SHA), allow_pickle=True).item())
    support_check = inverse.validate_rectangular_support()
    mapping = BMReviewMapping(frame.mapping, inverse, matrix, tuple(plan['registration']['tiff_to_sensor_xy']), photo_wh, photo_wh)
    sampled = sample_full_frame(np.moveaxis(frame.linear_rgb_chw, 0, -1), frame.valid_support,
                               mapping.content_to_display, inverse.shape)
    encoded = encode_training_rgb(sampled.values, sampled.valid_support,
                  white_range_dn=frame.certificate['white_range_dn'], **profile['camera_u_display']['camera-u'])
    np.save(folder/'content_rgb.npy', encoded)
    np.save(folder/'content_valid_support.npy', sampled.valid_support)
    del encoded, sampled
    records, index_binding = [], None
    if row['region_ids']:
        indices, index_binding = _shared_indices(root, mapping, frame.valid_support)
        evidence = json.loads((root/'evidence.json').read_text())['regions']
        for n, record in enumerate(r for r in evidence if r['region_id'] in row['region_ids']):
            old_mask = annotation_mask(record['payload'], canvas_wh=inverse.shape[::-1],
                                       content_box=(0, 0, inverse.shape[1], inverse.shape[0]))
            sparse = sparse_region_mask(old_mask, indices)
            if not sparse['pixels']:
                raise ValueError(f'BM annotation lost all RAW support: {record["region_id"]}')
            path = folder/f'region_{n:04d}.npz'; np.savez_compressed(path, **sparse)
            records.append(dict(region_id=record['region_id'], path=str(path), sha256=sha256(path),
                                bbox_xywh=sparse['bbox_xywh'].tolist(), target_pixels=sparse['pixels']))
    arrays = {name: dict(path=str(folder/name), sha256=sha256(folder/name))
              for name in ('raw_linear_dn.npy', 'raw_valid_support.npy', 'content_rgb.npy', 'content_valid_support.npy')}
    receipt = dict(frame_index=index, source=row, full_raw=frame.certificate, raw_mapping=asdict(frame.mapping),
                   drift_matrix=matrix.tolist(), photo_wh=list(photo_wh), resized_wh=list(photo_wh),
                   registration=plan['registration'], curvature=plan['curvature'], invertibility=support_check,
                   photometry=profile['camera_u_display']['camera-u'], arrays=arrays,
                   region_masks=records, coordinate_cache=index_binding, seconds=time.monotonic()-started)
    write(folder/'receipt.json', receipt)
    print(json.dumps(dict(stage='frame_prepared', frame=index, regions=len(records), seconds=receipt['seconds'])), flush=True)
    return receipt
