"""Strict source admission and array loading for explicit phase input bindings.

Source relocations are exact aliases checked against original content digests.
No acquisition is selected, no model is fitted and no new boundary release is
inferred from predicted labels or output differences.
"""
from __future__ import annotations
from collections.abc import Mapping
from pathlib import Path
import re
import json
import hashlib
from collections import OrderedDict
from typing import Callable
import numpy as np
from PIL import Image
from ..native.io import sha256_file, read_bound_json, resolve_asset_path
from ..native.sidecar import assemble_extended_log
from ..phase import inputs as native_inputs, heads
from ..phase.domain import RetainedSourceInferenceCache, _mask_digest
from ..phase.metrics import validate_pixel_area

RAW_ACQUISITION_CONTRACT = 'darsia-raw-phase-acquisition-v1'


def verify_raw_acquisition(record, row, audit, *, parameters):
    """Bind new baseline/history arrays to a RAW-produced acquisition.

    The masked-array and temporal readers below retain their existing numeric
    contracts. This additional source check prevents an old JPEG-positioned
    row being relabelled as the new RAW baseline or initial history.
    """
    from ..image_formats import image_family
    acquisition = row.get('raw_acquisition') or {}
    raw_sha = acquisition.get('sha256')
    expected = record.get('source_sha256', (record.get('source') or {}).get('source_sha256'))
    if (row.get('contract') != RAW_ACQUISITION_CONTRACT or row.get('source_kind') != 'raw'
            or image_family(acquisition.get('path', '')) != 'raw'
            or not isinstance(raw_sha, str) or re.fullmatch('[0-9a-f]{64}', raw_sha) is None
            or raw_sha != row.get(parameters['source_sha_field']) or raw_sha != expected
            or record['full'].get('source_sha256') != raw_sha
            or record['full'].get('binding', {}).get('producer_audit') != row.get('raw_producer_audit')
            or audit.get('contract') != 'darsia-raw-rig-source-v1'
            or audit.get('jpeg_inputs') is not False or audit.get('cpu_only') is not True
            or audit.get('training') is not False
            or audit.get('processor', {}).get('contract') != 'darsia-raw-rig-source-v1'
            or audit.get('sources', {}).get('camera_arw', {}).get('sha256') != raw_sha):
        raise ValueError('New RAW baseline/history differs from its actual RAW producer/source')

def _descriptor(item, kind, aliases):
    if not isinstance(item, dict) or not isinstance(item.get('path'), str):
        raise ValueError('Missing SHA-bound ' + kind)
    digest = item.get('sha256')
    path = resolve_asset_path(item['path'], {} if aliases is None else aliases)
    if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None or (not path.is_file()):
        raise ValueError('Changed SHA-bound ' + kind)
    if 'size_bytes' in item and path.stat().st_size != item['size_bytes']:
        raise ValueError('Changed SHA-bound ' + kind)
    return (path, digest)

def bound_path(item, kind='source artifact', *, aliases=None):
    """Resolve an explicit relocation and verify its original file content."""
    path, digest = _descriptor(item, kind, aliases)
    if sha256_file(path) != digest:
        raise ValueError('Changed SHA-bound ' + kind)
    return path

def load_bound_json(item, kind='source document', *, aliases=None):
    """Load byte-bound JSON through the common native reader, without rehashing."""
    path, digest = _descriptor(item, kind, aliases)
    return read_bound_json(path, digest)

def load_bound_array(item, kind='source array', *, aliases=None, mmap_mode='r'):
    """Load a content-verified array or its explicitly declared NPZ member."""
    path = bound_path(item, kind, aliases=aliases)
    value = np.load(path, allow_pickle=False, mmap_mode=mmap_mode)
    if item.get('array_key'):
        try:
            return value[item['array_key']]
        finally:
            value.close()
    return value


def compose_extended_scientific(descriptor, *, parameters: Mapping, aliases=None):
    """Reconstruct a transient full raster from exact native and disjoint strips.

    The original photograph width, minimum height and native crop are explicit
    DATA. Only bottom padding is permitted. Every strip retains its own content
    digest, float32 encoding and nonoverlapping canvas window.
    """
    if descriptor.get("contract") != parameters["contract"] or descriptor.get("complete") is not True or descriptor.get("storage") != parameters["storage"]:
        raise ValueError("Full-phase scientific descriptor has another correction/storage contract")
    shape = list(descriptor["phase_shape_hw"])
    canvas = parameters["photograph_canvas_wh"]
    if not (len(shape) == 2 and all(type(value) is int for value in shape)
            and shape[0] >= canvas[1] and shape[1] == canvas[0]):
        raise ValueError("Rig phase grid permits explicit bottom padding only")
    centre = load_bound_array(descriptor["native_centre"]["log_linear_rgb"], aliases=aliases)
    if centre.dtype != np.float32 or centre.shape != (3, *parameters["native_shape_hw"]):
        raise ValueError("Sparse RAW ring centre has another source grid/encoding")
    full = np.full((3, *shape), np.nan, np.float32)
    x, y, width, height = parameters["native_box_xywh"]
    full[:, y:y + height, x:x + width] = centre
    covered = np.zeros(shape, bool)
    covered[y:y + height, x:x + width] = True
    for row in descriptor["ring_windows"]:
        a, b, c, d = row["canvas_xyxy"]
        if any(type(value) is not int for value in (a, b, c, d)) or not (0 <= a < c <= shape[1] and 0 <= b < d <= shape[0]) or covered[b:d, a:c].any():
            raise ValueError("Sparse RAW ring overlaps the native centre/another window or escapes its phase grid")
        values = load_bound_array(row["log_linear_rgb"], aliases=aliases)
        if values.dtype != np.float32 or values.shape != (3, d - b, c - a):
            raise ValueError("Sparse RAW ring has another shape/float encoding")
        full[:, b:d, a:c] = values
        covered[b:d, a:c] = True
    return full, np.isfinite(full).all(0)


def load_selected_scientific(receipt, *, parameters: Mapping, aliases=None):
    """Load the unchanged centre and transient rig raster with original checks."""
    native = receipt["native_scientific"]
    log = load_bound_array(native["log_linear_rgb"], aliases=aliases)
    valid = load_bound_array(native["valid_mask"], aliases=aliases)
    full, full_valid = compose_extended_scientific(receipt["full_extended_scientific"], parameters=parameters, aliases=aliases)
    if log.dtype != np.float32 or log.shape != (3, *parameters["native_shape_hw"]):
        raise ValueError("Selected scientific input lost native float32 RAW")
    return log, valid, full, full_valid

def original_native_input(row, *, scientific_native_log=None, parameters, aliases=None):
    """Load the acquisition's original masked RAW/support and preserve its bits."""
    required = tuple(parameters['fields']['native'])
    if any((not row.get(key) for key in required)):
        raise ValueError('Original native source has no exact masked RAW descriptors')
    log_asset = dict(path=row[required[0]], sha256=row[required[1]])
    valid_asset = dict(path=row[required[2]], sha256=row[required[3]])
    log = load_bound_array(log_asset, 'original masked RAW', aliases=aliases)
    valid = load_bound_array(valid_asset, 'original native support', aliases=aliases)
    return native_inputs.original_native_input(log, valid, provenance=dict(source_sha256=row.get(parameters['source_sha_field']), log_linear_rgb=log_asset, valid_mask=valid_asset), scientific_native_log=scientific_native_log, parameters=parameters)

def temporal_policy(reference_row, *, parameters, aliases=None):
    """Verify the reference's declared ring before using its padding convention."""
    keys = tuple(parameters['fields']['temporal'])
    present = [reference_row.get(key) is not None for key in keys]
    if not any(present) or not all(present):
        return native_inputs.temporal_policy(reference_row, parameters=parameters)
    item = dict(path=reference_row[keys[0]], sha256=reference_row[keys[1]])
    policy = native_inputs.temporal_policy(reference_row, verified_ring=item, parameters=parameters)
    path = bound_path(item, 'historical temporal ring', aliases=aliases)
    policy['ring'] = dict(path=str(path), sha256=item['sha256'])
    return policy

def original_temporal_input(native, row, *, parameters, ring_parameters, aliases=None):
    """Load only the acquisition's declared strips with the existing sidecar ABI."""
    if not isinstance(native, native_inputs.MaskedNativeInput):
        raise TypeError('Original temporal input requires a masked native source')
    policy = temporal_policy(row, parameters=parameters, aliases=aliases)
    assembled = None
    if policy['pad_px'] != 0:
        assembled = assemble_extended_log(native.log, Path(policy['ring']['path']), ring_sha256=policy['ring']['sha256'], expected_pad_px=policy['pad_px'], parameters=ring_parameters)
    return native_inputs.original_temporal_input(native, policy=policy, assembled_log=assembled, parameters=parameters)

def original_photo_sand(probability_receipt, *, parameters, aliases=None):
    """Load the original packed photo-domain row; never use phase labels as mask."""
    metadata = probability_receipt.get('metadata', {})
    if metadata.get('masks_order') != list(parameters['mask_names']) or metadata.get('bitorder') != parameters['bitorder']:
        raise ValueError('Original photo domain lacks its frozen packed-mask contract')
    descriptor = probability_receipt['files'][parameters['mask_file']]
    with np.load(bound_path(descriptor, 'original photo masks', aliases=aliases), allow_pickle=False) as archive:
        return native_inputs.original_photo_sand(archive[parameters['mask_array_key']], metadata, descriptor, parameters=parameters)

def validate_model_documents(documents, bindings, *, requirements, heads_parameters):
    """Retain the original frozen role, refit and converged gas-head checks.

    The loader verifies each JSON's actual bytes before this pure admission
    step. Expected identities and document roles are explicit config DATA.
    """
    for name, digest in requirements['expected_sha256'].items():
        if bindings[name]['sha256'] != digest:
            raise ValueError('Frozen model artifact changed: ' + name)
    if requirements.get('profile') is not None:
        profile = documents[requirements['profile']]
        if profile[requirements['profile_model_field']]['sha256'] != requirements['expected_sha256'][requirements['original_regional']]:
            raise ValueError('Original regional profile changed')
    if requirements.get('refit_pair') is not None:
        original, refit = requirements['refit_pair']
        heads.validate_refit(documents[original], documents[refit], parameters=heads_parameters)
    if requirements.get('gas_head') is not None:
        gas = documents[requirements['gas_head']]
        if gas.get('variant') != requirements['gas_variant'] or len(gas['feature_names']) != requirements['gas_feature_count'] or gas['fit'].get('converged') is not True:
            raise ValueError('Frozen gas head is not the declared converged feature model')
    return dict(bindings)

def load_model_documents(bindings, *, requirements, heads_parameters, aliases=None):
    """Verify model JSON bytes, then admit only the explicitly frozen role set."""
    documents = {name: load_bound_json(bindings[name], 'frozen model ' + name, aliases=aliases) for name in requirements['expected_sha256']}
    validate_model_documents(documents, bindings, requirements=requirements, heads_parameters=heads_parameters)
    return documents

def _identity(run, frame_index, source_sha256, *, parameters: Mapping):
    if not isinstance(run, str) or re.fullmatch(parameters['run_pattern'], run) is None or type(frame_index) is not int or (frame_index < 0) or (not isinstance(source_sha256, str)) or (re.fullmatch('[0-9a-f]{64}', source_sha256) is None):
        raise ValueError('Retained inference needs an explicit run/index/source SHA')

def probability_encoding_contract(receipt, native_shape_hw, *, parameters: Mapping):
    """Describe the bound production encoding; do not invent float32 parity.

    Only receipt metadata is inspected here. Probability arrays are not used
    for categorical reuse. The packed eligibility masks are verified separately.
    """
    shape = list(native_shape_hw)
    metadata, binding, files = (receipt.get(key) or {} for key in ('metadata', 'binding', 'files'))
    layout = metadata.get('storage_layout')
    expected = parameters['encoding']['labelled_layout'] if binding.get('labelled') is True else parameters['encoding']['compact_layout']
    if receipt.get('contract') != parameters['contracts']['PROBABILITY_CONTRACT'] or receipt.get('complete') is not True or binding.get('provisional') is not False or (layout != expected) or (metadata.get('native_shape_hw') != shape) or (binding.get('g3_model_sha256') != parameters['expected_models']['g3']) or (binding.get('regional_refit_sha256') != parameters['expected_models']['regional_refit']) or (metadata.get('co2_head_sha256') != parameters['expected_models']['co2']):
        raise ValueError('Retained inference lacks its frozen probability encoding identity')
    working_shape = metadata.get('regional_working_shape_hw')
    if not isinstance(working_shape, list) or len(working_shape) != 2 or any((type(n) is not int or n < 1 for n in working_shape)):
        raise ValueError('Frozen probability working grid is invalid')
    for name in tuple(parameters['encoding']['probability_files']):
        item = files.get(name) or {}
        expected_dtype = 'uint8' if name == 'g3_pixel.npy' and layout != parameters['encoding']['labelled_layout'] else 'float16'
        expected_shape = shape if name == 'g3_pixel.npy' or layout == parameters['encoding']['labelled_layout'] else working_shape
        if item.get('dtype') != expected_dtype or item.get('shape') != expected_shape:
            raise ValueError('Frozen probability storage dtype/grid changed')
    if metadata.get('bitorder') != 'little' or metadata.get('masks_order') != list(tuple(parameters['mask_names'])) or metadata.get('resize_rule') != parameters['encoding']['resize_rule'] or (metadata.get('g3_compact_rule') != parameters['encoding']['compact_rule']):
        raise ValueError('Frozen probability decoding contract changed')
    return dict(contract=parameters['contracts']['ENCODING_CONTRACT'], layout=layout, g3='native float32 -> float16 -> decoded float32' if layout == parameters['encoding']['labelled_layout'] else 'native uint8 round-even(254*p); invalid255; decoded code/254', regional='native bilinear resize before float16 encoding; decoded float32' if layout == parameters['encoding']['labelled_layout'] else 'working-grid float16 -> decoded float32 -> native bilinear resize', native_shape_hw=shape, regional_working_shape_hw=working_shape, float32_fresh_inference_equivalence_claimed=False, probability_artifacts_used_as_training_input=False)

def prepare_retained_source_cache(binding, *, run, frame_index, current_source_sha256, current_log, allowed_domain, producer_audit=None, recompute_old_domain_ring=False, recompute_old_domain_interpolation_footprint=False, parameters: Mapping, aliases: Mapping=None):
    """Seal the original no-release cache populations before fresh inference.

Source/model/encoding proofs, parent-supported float bits and packed masks
are checked before constructing read-only categorical cache, tie and fill
eligibility. None explicitly opts out for a changed/new source. This adapter
does not enable either old-boundary recomputation option."""
    if recompute_old_domain_ring or recompute_old_domain_interpolation_footprint:
        raise ValueError('This input admission path does not enable old-domain boundary release')
    _identity(run, frame_index, current_source_sha256, parameters=parameters)
    if type(recompute_old_domain_ring) is not bool or type(recompute_old_domain_interpolation_footprint) is not bool:
        raise TypeError('Old domain ring recomputation requires an explicit bool policy')
    release_enabled = recompute_old_domain_ring or recompute_old_domain_interpolation_footprint
    current, domain = (np.asarray(current_log), np.asarray(allowed_domain))
    if current.dtype != np.float32 or current.ndim != 3 or current.shape[0] != 3 or (domain.dtype != bool) or (domain.shape != current.shape[1:]):
        raise ValueError('Retained inference requires native float32 and explicit bool domain')
    empty = np.full(domain.shape, 255, np.uint8)
    if binding is None:
        return RetainedSourceInferenceCache(empty, np.zeros(domain.shape, bool), np.zeros(domain.shape, bool), np.zeros(domain.shape, bool), domain.copy(), None, dict(contract=parameters['contracts']['CONTRACT'], run=run, frame_index=frame_index, current_source_sha256=current_source_sha256, prepared_before_fresh_inference=True, cache_enabled=False, cache_reason='explicit no reuse for changed/new source acquisition', cached_parent_modelled_pixels=0, tie_reference_eligible_pixels=0, tie_reference_eligible_mask_sha256=_mask_digest(np.zeros(domain.shape, bool)), prior_categorical_fill_contract=parameters['contracts']['PRIOR_FILL_CONTRACT'], prior_fill_eligible_pixels=0, prior_fill_eligible_mask_sha256=_mask_digest(np.zeros(domain.shape, bool)), old_domain_ring=dict(contract=parameters['contracts']['OLD_DOMAIN_RING_CONTRACT'], enabled=release_enabled, applicable=False, interpolation_footprint_requested=recompute_old_domain_interpolation_footprint, recomputed_pixels=0, recomputed_mask_sha256=_mask_digest(np.zeros(domain.shape, bool)), prepared_before_fresh_inference=True, eligibility_selected_from_output_difference=False), training_input=False, fresh_parity_claimed=False, mass_eligible=False), np.zeros(domain.shape, bool))
    if binding.get('contract') != parameters['contracts']['CONTRACT'] or binding.get('run') != run or binding.get('frame_index') != frame_index:
        raise ValueError('Retained inference binding has a different explicit run/index')
    native = load_bound_json(binding.get('parent_native_manifest'), 'parent native manifest', aliases=aliases)
    frame_receipt = load_bound_json(binding.get('parent_frame_receipt'), 'parent categorical receipt', aliases=aliases)
    exported = load_bound_json(binding.get('parent_probability_receipt'), 'parent probability receipt', aliases=aliases)
    if native.get('run') != run or native.get('analysis_image_shape_hw') != list(domain.shape) or frame_index >= len(native.get('frame_sources') or []):
        raise ValueError('Parent native source does not contain this exact frame/grid')
    frame = native['frame_sources'][frame_index]
    if frame.get('source_sha256') != current_source_sha256:
        raise ValueError('A changed acquisition cannot reuse parent inference')
    receipt_binding = frame_receipt.get('binding') or {}
    if frame_receipt.get('contract') != parameters['contracts']['FRAME_CONTRACT'] or frame_receipt.get('complete') is not True or receipt_binding.get('run') != run or (receipt_binding.get('frame_index') != frame_index) or (receipt_binding.get('native_manifest_sha256') != binding['parent_native_manifest']['sha256']):
        raise ValueError('Parent categorical proof is not bound to this native frame')
    encoding = probability_encoding_contract(exported, domain.shape, parameters=parameters)
    export_binding, metadata = (exported['binding'], exported['metadata'])
    if export_binding.get('run') != run or export_binding.get('frame_index') != frame_index:
        raise ValueError('Parent probability receipt belongs to a different frame')
    original_native_asset = dict(path=metadata.get('native_manifest_path'), sha256=export_binding.get('native_manifest_sha256'))
    original_native = load_bound_json(original_native_asset, 'original probability native manifest', aliases=aliases)
    if original_native.get('run') != run or original_native.get('analysis_image_shape_hw') != list(domain.shape) or frame_index >= len(original_native.get('frame_sources') or []):
        raise ValueError('Probability source uses a different native frame/grid')
    original_frame = original_native['frame_sources'][frame_index]
    scientific_keys = tuple(parameters['scientific_keys'])
    if any((not frame.get(key) or frame[key] != original_frame.get(key) for key in scientific_keys)):
        raise ValueError('Parent source alias changed scientific acquisition/cache bindings')
    if not isinstance(producer_audit, dict) or producer_audit.get('cpu_only') is not True or producer_audit.get('training') is not False or ((producer_audit.get('parent_cache') or {}).get('parent_float_bytes_preserved') is not True) or ((producer_audit.get('processor') or {}).get('parent_native_manifest_sha256') != binding['parent_native_manifest']['sha256']) or (((producer_audit.get('sources') or {}).get('camera_jpeg') or {}).get('sha256') != current_source_sha256) or (((producer_audit.get('sources') or {}).get('camera_arw') or {}).get('sha256') != frame[parameters['source_fields']['raw_sha']]):
        raise ValueError('Producer did not audit preserved parent scientific bits')
    assets = {}
    verified_paths = {}
    for name in tuple(parameters['parent_array_names']):
        item = dict(path=frame[name + '_path'], sha256=frame[name + '_sha256'])
        assets[name] = item
        verified_paths[name] = bound_path(item, 'parent scientific ' + name, aliases=aliases)
    parent = np.load(verified_paths[parameters['parent_array_names'][0]], mmap_mode='r', allow_pickle=False)
    valid = np.load(verified_paths[parameters['parent_array_names'][1]], mmap_mode='r', allow_pickle=False)
    if parent.dtype != np.float32 or parent.shape != current.shape or valid.dtype != bool or (valid.shape != domain.shape) or (not np.isfinite(parent[:, valid]).all()):
        raise ValueError('Parent scientific cache/validity has invalid native dtype/grid')
    if not np.isfinite(current[:, valid]).all() or not np.array_equal(current[:, valid].view(np.uint32), parent[:, valid].view(np.uint32)):
        raise ValueError('Current scientific float bits differ on parent-valid support')
    packed_asset = exported['files'].get(parameters['mask_file'])
    with np.load(bound_path(packed_asset, 'original inference masks', aliases=aliases), allow_pickle=False) as archive:
        packed = archive[parameters['mask_array_key']]
        if packed.dtype != np.uint8 or packed.shape != (len(tuple(parameters['mask_names'])), (domain.size + 7) // 8):
            raise ValueError('Original inference mask packing/grid changed')
        masks = {name: np.unpackbits(row, bitorder=parameters['encoding']['bitorder'], count=domain.size).reshape(domain.shape).astype(bool) for name, row in zip(tuple(parameters['mask_names']), packed)}
    if not np.array_equal(masks['regional_native_domain'], masks['photo_sand_domain'] & masks['feature_support']):
        raise ValueError('Original regional domain differs from its explicit photo/feature support')
    label_asset = frame_receipt.get('labels_map')
    with Image.open(bound_path(label_asset, 'parent categorical L proof', aliases=aliases)) as image:
        labels = np.array(image)
    if labels.dtype != np.uint8 or labels.shape != domain.shape or (not np.isin(labels, (0, 1, 2, 255)).all()):
        raise ValueError('Parent categorical proof does not share the exact native grid')
    cached = domain & valid & masks['regional_native_domain'] & masks['flatfield_valid'] & ~masks['frozen_unresolved'] & np.isin(labels, (0, 1, 2))
    original_cached_pixels = int(cached.sum())
    native_ring = np.zeros(domain.shape, bool)
    interpolation = None
    release = native_ring
    ring = domain & release
    cached &= ~ring
    tie_eligible = domain & valid & masks['photo_sand_domain'] & np.isin(labels, (0, 1, 2)) & ~ring
    prior_fill = tie_eligible & ~cached
    fixed_area = validate_pixel_area(float(metadata['pixel_area_mm2']), domain.shape)
    audit = dict(contract=parameters['contracts']['CONTRACT'], run=run, frame_index=frame_index, current_source_sha256=current_source_sha256, cache_enabled=True, prepared_before_fresh_inference=True, inference_cache_not_training_evidence=True, binding=binding, original_probability_native_manifest=original_native_asset, parent_labels_map=label_asset, parent_scientific_assets=assets, parent_valid_pixels=int(valid.sum()), parent_scientific_bits_verified=True, producer_parent_cache=producer_audit['parent_cache'], probability_encoding=encoding, cache_eligibility='allowed domain AND original photo/feature support AND original flatfield-valid AND parent scientific-valid AND original frozen label!=255 AND parent category0/1/2' + (' AND NOT predeclared recomputed old boundary release' if release_enabled else ''), cached_parent_modelled_pixels=int(cached.sum()), cached_parent_modelled_mask_sha256=_mask_digest(cached), old_domain_ring=dict(contract=parameters['contracts']['OLD_DOMAIN_RING_CONTRACT'], enabled=release_enabled, applicable=True, native_ring_requested=recompute_old_domain_ring, interpolation_footprint_requested=recompute_old_domain_interpolation_footprint, released_mask_definition='native outer ring UNION source-derived exterior-rejected coarse-cell bilinear interpolation footprint' if recompute_old_domain_interpolation_footprint else 'native one-pixel outer ring', native_ring_pixels=int((native_ring & domain).sum()), interpolation_footprint=interpolation, original_domain_mask_sha256=_mask_digest(masks['regional_native_domain']), original_domain='SHA-bound original photo_sand_domain AND feature_support', geometry='one native exterior pixel ring plus exact bilinear influence of original exterior-rejected coarse cells; clipped to allowed domain' if recompute_old_domain_interpolation_footprint else 'one native pixel with a four-connected neighbour in the exterior of the old domain; closed internal holes excluded; clipped to allowed domain', recomputed_pixels=int(ring.sum()), recomputed_mask_sha256=_mask_digest(ring), released_cached_pixels=original_cached_pixels - int(cached.sum()), prepared_before_fresh_inference=True, eligibility_selected_from_output_difference=False, cache_and_prior_fill_excluded=True, classification_policy='fresh frozen-head evaluation with the added domain before four-connected gas minimum; unsupported pixels retain explicit fill flags'), tie_reference_eligible_pixels=int(tie_eligible.sum()), tie_reference_eligible_mask_sha256=_mask_digest(tie_eligible), tie_reference_policy='SHA-bound parent category may select only an equally nearest modelled neighbour; not modelled reuse, not a unique-nearest override', prior_categorical_fill_contract=parameters['contracts']['PRIOR_FILL_CONTRACT'], prior_fill_eligibility='allowed domain AND parent scientific-valid AND original photo-sand AND resolved parent category0/1/2 AND NOT cached parent modelled' + (' AND NOT predeclared recomputed old boundary release' if release_enabled else ''), prior_fill_eligible_pixels=int(prior_fill.sum()), prior_fill_eligible_mask_sha256=_mask_digest(prior_fill), prior_fill_policy='Preserve the prior category only after final nearest fill on non-modelled filled pixels; never demote fresh-modelled pixels or select eligibility from changed pixels', parent_fixed_area_mm2=fixed_area, inherited_parent_rules_retained=True, fresh_parity_claimed=False, mass_eligible=False)
    return RetainedSourceInferenceCache(labels, cached, tie_eligible, prior_fill, domain.copy(), fixed_area, audit, ring)

def _array_sha256(array: np.ndarray, *, buffer_elements) -> str:
    """Hash exact C-order bytes without allocating a full strided-array copy."""
    digest = hashlib.sha256()
    if array.nbytes == 0:
        return digest.hexdigest()
    if array.flags.c_contiguous:
        digest.update(memoryview(array).cast('B'))
    else:
        iterator = np.nditer(array, flags=['external_loop', 'buffered', 'zerosize_ok'], op_flags=['readonly'], order='C', buffersize=buffer_elements)
        for block in iterator:
            digest.update(block.tobytes())
    return digest.hexdigest()

class FixedBaselineDecodeCache:
    """One worker's two-entry byte-bounded cache; hits never share writable RGB."""

    def __init__(self, *, max_bytes: int, parameters: Mapping):
        """Bind a byte-budgeted cache for verified fixed-reference decoding and validation."""
        self.parameters = parameters
        if type(max_bytes) is not int or not 0 <= max_bytes <= self.parameters['maximum_bytes']:
            raise ValueError('Fixed-reference decode cache requires a 0..1 GiB byte budget')
        self.max_bytes = max_bytes
        self._entries = OrderedDict()
        self._bytes = 0
        self._hits = self._misses = self._delegated = self._too_large = 0
        self._validated = OrderedDict()
        self._validation_hits = self._validation_misses = 0
        self._bindings = {}

    def decode(self, role: str, log, valid, *, decoder: Callable | None):
        """Use the original decoder on cold, invalid, current and previous calls.

        The exact input and mask byte hashes are checked on every lookup. A
        changed array cannot retrieve a value decoded from its previous bytes.
        Unsupported representations are delegated, preserving original errors.
        """
        source, support = (np.asarray(log), np.asarray(valid))
        if role not in frozenset(self.parameters['fixed_roles']) or self.max_bytes == 0 or source.dtype != np.float32 or (source.ndim != 3) or (source.shape[0] != 3) or (support.dtype != bool) or (support.shape != source.shape[1:]):
            self._delegated += 1
            return decoder(log, valid)
        log_sha, support_sha = (_array_sha256(source, buffer_elements=self.parameters['array_hash_buffer_elements']), _array_sha256(support, buffer_elements=self.parameters['array_hash_buffer_elements']))
        key = (decoder, source.shape, source.dtype.str, log_sha, support_sha)
        self._bindings[role] = dict(log_array_sha256=log_sha, support_array_sha256=support_sha, shape_chw=list(source.shape), input_dtype=source.dtype.str, support_dtype=support.dtype.str)
        if key in self._entries:
            self._hits += 1
            self._entries.move_to_end(key)
            return self._entries[key].copy()
        self._misses += 1
        result = decoder(log, valid)
        if not isinstance(result, np.ndarray) or result.dtype != np.float32 or result.shape != (*support.shape, 3) or (_array_sha256(source, buffer_elements=self.parameters['array_hash_buffer_elements']) != log_sha) or (_array_sha256(support, buffer_elements=self.parameters['array_hash_buffer_elements']) != support_sha):
            return result
        if result.nbytes > self.max_bytes:
            self._too_large += 1
            return result
        while self._entries and (len(self._entries) >= self.parameters['maximum_entries'] or self._bytes + result.nbytes > self.max_bytes):
            _, removed = self._entries.popitem(last=False)
            self._bytes -= removed.nbytes
        saved = result.copy()
        saved.setflags(write=False)
        self._entries[key] = saved
        self._bytes += saved.nbytes
        return result

    def validate_fixed(self, role: str, log, valid, *, decoder: Callable | None):
        """Skip only an already validated, discarded fixed-flat decode.

        This retains two exact hash keys and no decoded RGB. The accepted
        decoder is still called for every cold, changed or invalid input.
        """
        source, support = (np.asarray(log), np.asarray(valid))
        if role != self.parameters['validation_role'] or source.dtype != np.float32 or source.ndim != 3 or (source.shape[0] != 3) or (support.dtype != bool) or (support.shape != source.shape[1:]):
            decoder(log, valid)
            return None
        log_sha, support_sha = (_array_sha256(source, buffer_elements=self.parameters['array_hash_buffer_elements']), _array_sha256(support, buffer_elements=self.parameters['array_hash_buffer_elements']))
        key = (decoder, source.shape, source.dtype.str, log_sha, support_sha)
        self._bindings[role] = dict(log_array_sha256=log_sha, support_array_sha256=support_sha, shape_chw=list(source.shape), input_dtype=source.dtype.str, support_dtype=support.dtype.str)
        if key in self._validated:
            self._validated.move_to_end(key)
            self._validation_hits += 1
            return None
        self._validation_misses += 1
        decoder(log, valid)
        if _array_sha256(source, buffer_elements=self.parameters['array_hash_buffer_elements']) == log_sha and _array_sha256(support, buffer_elements=self.parameters['array_hash_buffer_elements']) == support_sha:
            while len(self._validated) >= self.parameters['maximum_entries']:
                self._validated.popitem(last=False)
            self._validated[key] = None
        return None

    def clear(self):
        """Discard retained decoded arrays and validation entries and reset retained byte accounting."""
        self._entries.clear()
        self._validated.clear()
        self._bytes = 0

    def receipt(self) -> dict:
        """Report cache byte/count statistics and bindings without replacing source verification."""
        return dict(contract=self.parameters['contract'], fixed_roles=sorted(frozenset(self.parameters['fixed_roles'])), cached_entries=len(self._entries), cached_bytes=self._bytes, max_bytes=self.max_bytes, hits=self._hits, misses=self._misses, delegated_calls=self._delegated, oversized_results=self._too_large, validation_hits=self._validation_hits, validation_misses=self._validation_misses, validation_keys=len(self._validated), validation_rgb_retained_bytes=0, array_bindings=dict(self._bindings), cached_values_readonly=True, hits_return_fresh_mutable_copy=True, source_file_verification_replaced=False, numerical_decoder_changed=False, accepted_running_snapshot_changed=False)
