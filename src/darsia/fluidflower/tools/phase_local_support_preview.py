"""Admit already saved local-support display; never derive categorical phases."""
from __future__ import annotations
from darsia.fluidflower.series_paths import local_cache_root
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from darsia.fluidflower.tools.phase_reference_maps import _parameters, _producer_identity


def resolve(root, run, candidate, index, layer='all', *, reference_mode=False, reference=None):
    from darsia.fluidflower.tools import calibration_pairwise_review as review
    from darsia.fluidflower.tools.phase_reference_maps import build_plan, resolve_reference_overlay
    from darsia.fluidflower.tools.phase_reference_maps import _sha as sha
    from darsia.fluidflower.tools.phase_injection_references import resolve_reference_asset
    from darsia.fluidflower import phase_runtime
    from darsia.fluidflower.phase_saved_review_metadata import saved_physics_profile
    current = _parameters()["current_sources"]["local_preview"]
    if not isinstance(current, dict) or Path(current["path"]).resolve() != Path(__file__).resolve() or sha(Path(__file__)) != current["sha256"] or Path(__file__).stat().st_size != current["size_bytes"]:
        raise ValueError("Saved local-support reader implementation changed")
    plan = build_plan(None, root, run, candidate)
    native = json.loads(Path(plan['native_manifest']['path']).read_text(encoding='utf-8-sig'))
    if reference_mode:
        photo = resolve_reference_asset(reference, root, run) if reference is not None else Path(plan['baseline']['path'])
        original = resolve_reference_overlay(reference, root, run, candidate, 'A', 'all')
        elapsed = float(reference['elapsed_h']) if reference is not None else -1.0
    else:
        photo = review._native_frame_path(root, run, index)
        original = review._native_phase_map_path(root, candidate, 'A', index)
        elapsed = float(native['frame_sources'][index]['elapsed_h'])
    profile = phase_runtime.load_physics_profile(plan['model_root'])
    if profile and profile.get('unified_phase_decision'):
        if reference_mode:
            return resolve_reference_overlay(reference, root, run, candidate, 'A', layer)
        return review.two_phase_overlay_path(original, layer)
    local = saved_physics_profile('local_support')
    settings = local.Settings()
    identity = dict(format=_parameters()['local_preview_format'], phase_sha256=sha(original), photo_sha256=sha(photo), baseline_sha256=plan['baseline']['sha256'], inherited_runtime=plan['physics_runtime'], settings=asdict(settings), implementation_sha256=_producer_identity('local_support'), frame_time_h=elapsed, model_negative_veto=False)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    folder = local_cache_root() / _parameters()['local_cache_suffix'] / key
    target = folder / ('all.png' if layer == 'all' else layer + '.png')
    if target.is_file() and (folder / 'receipt.json').is_file():
        return target
    raise FileNotFoundError(_parameters()['unavailable_local'])
