"""Source-bound RAW photometry with explicit frozen parameter DATA.

The algorithms retain their original operation ordering, Python literal types
and array casts. This module imports no campaign, Studio or legacy controller.
Baseline fitting, source projection and display-response evaluation are distinct
operations; no source image is spatially warped by native correction.
"""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
import hashlib
import json
import os
from pathlib import Path
import cv2
import numpy as np
from ..pipeline.io_ops import retry_io

def _log_rgb(a, shape=None, *, parameters: Mapping):
    """Convert channel-first log RGB to float32 and validate optional geometry."""
    a = np.asarray(a, np.float32)
    if a.ndim != 3 or a.shape[0] != 3 or (shape is not None and a.shape[1:] != shape):
        raise ValueError(f'Expected RGB log array (3,H,W), got {a.shape}')
    return a

def polynomial_basis(h=None, w=None, *, parameters: Mapping):
    """Evaluate the nonconstant spatial polynomial on an explicit grid.

    Coordinates are normalised by the original grid half-width and half-height;
    polynomial degree and optional grid dimensions come from parameters."""
    if h is None:
        h = parameters['polynomial_basis_default_h']
    if w is None:
        w = parameters['polynomial_basis_default_w']
    yy, xx = np.mgrid[:h, :w]
    x, y = ((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    return np.stack([x ** a * y ** b for d in range(1, parameters['polynomial_basis_int_5']) for a in range(d + 1) for b in [d - a]]).astype(np.float64)

def _same_fac(f, *, parameters: Mapping):
    """Select cells whose full edge-padded three-by-three neighbourhood has one class."""
    h, w = f.shape
    p = np.pad(f, 1, mode='edge')
    out = np.ones(f.shape, bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            out &= p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] == f
    return out

def _fit_grid_field(g, fac, keep, *, parameters: Mapping):
    """Fit green-channel polynomial coefficients and per-facies intercepts.

    Only finite supported cells enter the original iteratively reweighted
    least-squares solve. Thresholds, iteration count and Huber scale are DATA."""
    basis = polynomial_basis(parameters=parameters).reshape(parameters['fit_grid_field_int_14'], -1)
    idx = np.flatnonzero((keep & (fac >= 1) & np.isfinite(g)).ravel())
    if len(idx) < parameters['fit_grid_field_int_1000']:
        raise ValueError(f'Too few baseline sand cells: {len(idx)}')
    ff = fac.ravel()[idx]
    classes = [f for f in range(1, parameters['fit_grid_field_int_9']) if np.count_nonzero(ff == f) >= parameters['fit_grid_field_int_30']]
    X = np.column_stack([basis[:, idx].T] + [(ff == f).astype(float) for f in classes])
    y = g.ravel()[idx].astype(float)
    wt = np.ones(len(y))
    for _ in range(parameters['fit_grid_field_int_6']):
        sw = np.sqrt(wt)
        coef = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        wt = np.minimum(1, parameters['fit_grid_field_float_0_05'] / np.maximum(np.abs(y - X @ coef), parameters['fit_grid_field_float_1eminus_09']))
    return (coef[:parameters['fit_grid_field_int_14']], {f: float(coef[parameters['fit_grid_field_int_14'] + i]) for i, f in enumerate(classes)}, {f: int(np.count_nonzero(ff == f)) for f in classes})

def _colour(log_rgb, inv, *, parameters: Mapping):
    """Invert the colour matrix in linear RGB, then return float32 log RGB.

    Invalid input pixels remain NaN; the original exponential, einsum, floor
    and log operation order is preserved."""
    bad = ~np.all(np.isfinite(log_rgb), 0)
    linear = np.exp(np.nan_to_num(log_rgb, nan=-parameters['colour_int_30'])).astype(np.float32)
    out = np.log(np.maximum(np.einsum('ij,jhw->ihw', inv, linear, optimize=True), parameters['colour_float_0_001']))
    out[:, bad] = np.nan
    return out.astype(np.float32)

def _chart_plane(C, CR, usable, M, *, parameters: Mapping):
    """Fit achromatic chart drift against reference samples in the colour matrix.

    Usable patches define the constant and two chart-coordinate coefficients
    in the original robust least-squares order."""
    C, CR = (np.asarray(C), np.asarray(CR))
    u = np.asarray(usable, bool) & np.all(np.isfinite(C), 0) & np.all(np.isfinite(CR), 0)
    if C.shape != (3, parameters['chart_plane_int_24']) or CR.shape != (3, parameters['chart_plane_int_24']) or np.count_nonzero(u) < parameters['chart_plane_int_6']:
        raise ValueError('Invalid chart samples or too few common patches')
    k = np.arange(parameters['chart_plane_int_24'])
    U, V = (k % parameters['chart_plane_int_4'] - parameters['chart_plane_float_1_5'], k // parameters['chart_plane_int_4'] - parameters['chart_plane_float_2_5'])
    X = np.column_stack([np.ones(u.sum()), U[u], V[u]])
    y = (C[:, u] - np.log(np.maximum(M @ np.exp(CR[:, u]), parameters['chart_plane_float_1eminus_09']))).mean(0)
    wt = np.ones(len(y))
    for _ in range(parameters['chart_plane_int_8']):
        sw = np.sqrt(wt)
        b = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]
        wt = np.minimum(1, parameters['chart_plane_float_0_03'] / np.maximum(np.abs(y - X @ b), parameters['chart_plane_float_1eminus_09']))
    return b

@dataclass
class Baseline:
    """Source-bound baseline arrays; their original dtypes are retained."""
    run: str
    M: np.ndarray
    Minv: np.ndarray
    P: np.ndarray
    a: dict[int, float]
    n: dict[int, int]
    fac: np.ndarray
    sand: np.ndarray
    chart: np.ndarray
    cpl: np.ndarray
    Ug: np.ndarray
    Vg: np.ndarray
    taper: np.ndarray
    logB: np.ndarray
    gb0: np.ndarray
    xy: np.ndarray
    CB: np.ndarray
    F: np.ndarray | None
    residual_exclusion_fraction: float

def fit_baseline_grid(*, run, logB, facies, chart_xy, chart_log_rgb, ref_chart_log_rgb, chart_usable, matrix, rig_consensus, flat=None, parameters: Mapping):
    """Fit baseline colour, spatial green field and physical-facies intercepts.

    logB and rig_consensus are registered log RGB. The explicit chart lattice,
    usability flags, colour matrix and optional flat define the frozen fit;
    parameters supply support, smoothing and robust-fit constants."""
    GRID = tuple(parameters['grid_shape_hw'])
    logB = _log_rgb(logB, GRID, parameters=parameters)
    cons = _log_rgb(rig_consensus, GRID, parameters=parameters)
    fac = np.asarray(facies, np.int16)
    xy = np.asarray(chart_xy, np.float64)
    M = np.asarray(matrix, np.float64)
    if fac.shape != GRID or xy.shape != (parameters['fit_baseline_grid_int_24'], 2) or M.shape != (3, 3):
        raise ValueError('Facies, chart or matrix shape mismatch')
    Minv = np.linalg.inv(M)
    B1 = _colour(logB, Minv, parameters=parameters)
    F = None if flat is None else _log_rgb(flat, GRID, parameters=parameters)
    yy, xx = np.mgrid[:GRID[0], :GRID[1]]
    validxy = np.all(np.isfinite(xy), 1)
    if validxy.sum() < parameters['fit_baseline_grid_int_6']:
        raise ValueError('Missing ColorChecker lattice')
    cb = (xx >= xy[validxy, 0].min() - parameters['fit_baseline_grid_int_7']) & (xx <= xy[validxy, 0].max() + parameters['fit_baseline_grid_int_7']) & (yy >= xy[validxy, 1].min() - parameters['fit_baseline_grid_int_7']) & (yy <= xy[validxy, 1].max() + parameters['fit_baseline_grid_int_7'])
    fixchart = (xx >= parameters['fit_baseline_grid_int_569']) & (yy < parameters['fit_baseline_grid_int_131'])
    sand = (fac >= 1) & ~cb & ~fixchart
    keep = sand & _same_fac(fac, parameters=parameters)
    L = B1 - (F if F is not None else 0)
    dbg = L[2] - L[1] - (cons[2] - cons[1])
    fin = keep & np.isfinite(dbg)
    num = cv2.GaussianBlur(np.where(fin, dbg, 0).astype(np.float32), (0, 0), parameters['fit_baseline_grid_gaussian_sigma'])
    den = cv2.GaussianBlur(fin.astype(np.float32), (0, 0), parameters['fit_baseline_grid_gaussian_sigma'])
    dbg = np.where(fin, num / np.maximum(den, parameters['fit_baseline_grid_float_1eminus_06']), np.nan)
    for f in range(1, parameters['fit_baseline_grid_int_9']):
        m = keep & (fac == f) & np.isfinite(dbg)
        if m.sum() > parameters['fit_baseline_grid_int_100']:
            dbg[fac == f] -= np.median(dbg[m])
    fitkeep = keep & np.isfinite(dbg) & (dbg > -parameters['fit_baseline_grid_float_0_08'])
    P, a, n = _fit_grid_field(B1[1], fac, fitkeep, parameters=parameters)
    frac = float(1 - fitkeep.sum() / max(keep.sum(), 1))
    dist = cv2.distanceTransform((~(sand | cb)).astype(np.uint8), cv2.DIST_L2, parameters['fit_baseline_grid_int_5'])
    taper = np.exp(-dist ** 2 / (2 * parameters['fit_baseline_grid_float_8_0'] ** 2)).astype(np.float32)
    cpl = _chart_plane(chart_log_rgb, ref_chart_log_rgb, chart_usable, M, parameters=parameters)
    X = np.column_stack([np.ones(validxy.sum()), xy[validxy]])
    k = np.arange(parameters['fit_baseline_grid_int_24'])
    au = np.linalg.lstsq(X, (k % parameters['fit_baseline_grid_int_4'] - parameters['fit_baseline_grid_float_1_5'])[validxy], rcond=None)[0]
    av = np.linalg.lstsq(X, (k // parameters['fit_baseline_grid_int_4'] - parameters['fit_baseline_grid_float_2_5'])[validxy], rcond=None)[0]
    Ug = (au[0] + au[1] * xx + au[2] * yy).astype(np.float32)
    Vg = (av[0] + av[1] * xx + av[2] * yy).astype(np.float32)
    return Baseline(run, M, Minv, P, a, n, fac, sand, cb, cpl, Ug, Vg, taper, logB, B1[1] - B1[2], xy, np.asarray(chart_log_rgb, np.float32), F, frac)

def baseline_field(fit: Baseline, ref: Baseline, *, parameters: Mapping):
    """Evaluate the baseline-to-reference achromatic correction on the original grid.

    Shared physical-facies intercepts enter supported sand. Outside support,
    the existing taper and chart blend extend the polynomial field."""
    GRID = tuple(parameters['grid_shape_hw'])
    poly = np.einsum('i,ihw->hw', ref.P - fit.P, polynomial_basis(parameters=parameters))
    common = [f for f in fit.a if f in ref.a]
    if not common:
        raise ValueError('No common facies with frozen')
    vals = np.array([ref.a[f] - fit.a[f] for f in common])
    weights = np.array([fit.n[f] for f in common])
    meanoff = float(np.average(vals, weights=weights))
    off = np.full(GRID, np.nan)
    for f, v in zip(common, vals):
        off[(fit.fac == f) & fit.sand] = v
    insand = np.isfinite(off)
    total = np.where(insand, poly + off, np.nan)
    global_level = float(np.nanmedian(total))
    field = np.where(insand, total, fit.taper * (poly + meanoff) + (1 - fit.taper) * global_level)
    dist = cv2.distanceTransform((~fit.chart).astype(np.uint8), cv2.DIST_L2, parameters['baseline_field_int_5'])
    wc = np.exp(-dist ** 2 / (2 * parameters['baseline_field_float_20_0'] ** 2))
    area = ~insand & ~fit.chart & (dist < parameters['baseline_field_int_60'])
    field[area] = wc[area] * -fit.cpl[0] + (1 - wc[area]) * field[area]
    field[fit.chart] = -(fit.cpl[0] + fit.cpl[1] * fit.Ug[fit.chart] + fit.cpl[2] * fit.Vg[fit.chart])
    return (field.astype(np.float32), insand)

def _resize(a, shape, *, parameters: Mapping):
    """Resize a scalar float32 field with the original bilinear interpolation."""
    return cv2.resize(np.asarray(a, np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

def _native_sand_field_tile(fit, ref, facies_tile, lo, width, *, parameters: Mapping):
    """Evaluate the spatial field at native pixel centres before facies offsets.

    Each polynomial term is cast to float32 before addition, preserving the
    original accumulation order."""
    GRID = tuple(parameters['grid_shape_hw'])
    yy = (np.arange(lo, lo + facies_tile.shape[0], dtype=np.float64) - parameters['native_sand_field_tile_float_3_5']) / parameters['native_sand_field_tile_int_8']
    xx = (np.arange(width, dtype=np.float64) - parameters['native_sand_field_tile_float_3_5']) / parameters['native_sand_field_tile_int_8']
    x = ((xx - GRID[1] / 2) / (GRID[1] / 2))[None, :]
    y = ((yy - GRID[0] / 2) / (GRID[0] / 2))[:, None]
    coef = ref.P - fit.P
    out = np.zeros(facies_tile.shape, dtype=np.float32)
    k = 0
    for d in range(1, parameters['native_sand_field_tile_int_5']):
        for a in range(d + 1):
            b = d - a
            out += (coef[k] * x ** a * y ** b).astype(np.float32)
            k += 1
    for f in fit.a:
        if f in ref.a:
            out[facies_tile == f] += ref.a[f] - fit.a[f]
    return out

def apply_registered(log_rgb, fit: Baseline, ref: Baseline, *, facies_native=None, light_field8=None, tile_rows=None, parameters: Mapping):
    """Apply the frozen correction on registered cell or native-pixel geometry.

    Native facies, optional frame-light and tile size are explicit. Colour,
    achromatic field and saturation-dependent flat correction retain their
    original ordering; source coordinates are not warped."""
    if tile_rows is None:
        tile_rows = parameters['apply_registered_default_tile_rows']
    GRID = tuple(parameters['grid_shape_hw'])
    NATIVE = tuple(parameters['native_shape_hw'])
    GB_SAT = parameters['green_blue_saturation']
    source = _log_rgb(log_rgb, parameters=parameters)
    shape = source.shape[1:]
    if shape not in (GRID, NATIVE):
        raise ValueError(f'Not a registered cell/native frame: {shape}')
    if shape == NATIVE and (facies_native is None or np.shape(facies_native) != NATIVE):
        raise ValueError('Native correction requires source-bound native facies')
    field8, _ = baseline_field(fit, ref, parameters=parameters)
    field = field8 if shape == GRID else _resize(field8, shape, parameters=parameters)
    dlight_native = None
    if light_field8 is not None:
        dlight = np.asarray(light_field8, np.float32)
        if dlight.shape != GRID:
            raise ValueError('Light field must match the configured cell grid')
        dlight_native = dlight if shape == GRID else _resize(dlight, shape, parameters=parameters)
        field = field - dlight_native
    fac = fit.fac if shape == GRID else np.asarray(facies_native)
    chart = fit.chart if shape == GRID else _resize(fit.chart.astype(np.float32), shape, parameters=parameters) > parameters['apply_registered_float_0_5']
    sand = (fac >= 1) & ~chart
    taper = fit.taper if shape == GRID else _resize(fit.taper, shape, parameters=parameters)
    tap = np.where(sand, 1.0, taper * ~chart).astype(np.float32)
    F = fit.F if shape == GRID or fit.F is None else np.stack([_resize(f, shape, parameters=parameters) for f in fit.F])
    gb0 = fit.gb0 if shape == GRID else _resize(fit.gb0, shape, parameters=parameters)
    out = np.empty_like(source)
    for lo in range(0, shape[0], tile_rows):
        hi = min(lo + tile_rows, shape[0])
        rgb = _colour(source[:, lo:hi], fit.Minv, parameters=parameters)
        field_tile = field[lo:hi].copy()
        if shape == NATIVE:
            yy, xx = np.mgrid[lo:hi, :shape[1]]
            fixed_chart = (xx >= parameters['apply_registered_int_569'] * parameters['apply_registered_int_8']) & (yy < parameters['apply_registered_int_131'] * parameters['apply_registered_int_8'])
            common = [f for f in fit.a if f in ref.a]
            exact_sand = sand[lo:hi] & ~fixed_chart & np.isin(fac[lo:hi], common)
            exact = _native_sand_field_tile(fit, ref, fac[lo:hi], lo, shape[1], parameters=parameters)
            if dlight_native is not None:
                exact -= dlight_native[lo:hi]
            field_tile[exact_sand] = exact[exact_sand]
        rgb += field_tile[None]
        if F is not None:
            f = F[:, lo:hi]
            gb = rgb[1] - rgb[2]
            alpha = np.clip((GB_SAT - gb) / np.maximum(GB_SAT - gb0[lo:hi], parameters['apply_registered_float_0_2']), 0, 1)
            alpha = np.where(np.isfinite(alpha), alpha, 1)
            fa = f[1:2]
            rgb -= (fa + alpha[None] * (f - fa)) * tap[None, lo:hi]
        out[:, lo:hi] = rgb
    return out

def render_response(corrected_log_linear, response, *, parameters: Mapping):
    """Render corrected log RGB through an explicit matrix and channel response LUTs.

    This is a display conversion; scientific correction arrays remain separate.
    Invalid pixels use the original bright display encoding."""
    src = _log_rgb(corrected_log_linear, parameters=parameters)
    M = np.asarray(response['M'], np.float64)
    luts = response['luts']
    if M.shape != (3, 3) or len(luts) != 3:
        raise ValueError('Invalid frozen response')
    bad = ~np.all(np.isfinite(src), 0)
    lin = np.exp(np.nan_to_num(src, nan=0)).astype(np.float32)
    z = np.maximum(np.einsum('ij,jhw->ihw', M, lin, optimize=True), parameters['render_response_float_0_001'])
    out = np.stack([np.interp(np.log(z[c]), np.asarray(luts[c][0]), np.asarray(luts[c][1])) for c in range(3)])
    out[:, bad] = parameters['render_response_int_255']
    return np.clip(out.transpose(1, 2, 0) / parameters['render_response_int_255'], 0, 1).astype(np.float32)

def load_response(path: Path, *, parameters: Mapping):
    """Load the explicit matrix and channel-LUT response from a JSON document."""
    return json.loads(Path(path).read_text(encoding='utf-8'))['response']

def save_grid_baseline(path: Path, fit: Baseline, provenance: dict, *, parameters: Mapping):
    """Write baseline arrays without pickle and bind their exact NPZ hash in a sidecar.

    Provenance supplies the source, registration, manifest and facies hashes.
    Array member names, dtypes and stored scalar representations are retained."""
    GRID = tuple(parameters['grid_shape_hw'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() != '.npz':
        raise ValueError('Baseline calibration must be a .npz')
    required = ('source_arw_sha256', 'source_review_sha256', 'source_manifest_sha256', 'facies_sha256')
    missing = [k for k in required if not provenance.get(k)]
    if missing:
        raise ValueError(f'Missing baseline source binding: {missing}')
    keys = np.array(sorted(fit.a), dtype=np.int16)
    arrays = {'run': np.asarray(fit.run), 'M': fit.M, 'Minv': fit.Minv, 'P': fit.P, 'facies_keys': keys, 'facies_offsets': np.array([fit.a[int(k)] for k in keys]), 'facies_counts': np.array([fit.n[int(k)] for k in keys]), 'fac': fit.fac, 'sand': fit.sand, 'chart': fit.chart, 'cpl': fit.cpl, 'Ug': fit.Ug, 'Vg': fit.Vg, 'taper': fit.taper, 'logB': fit.logB, 'gb0': fit.gb0, 'xy': fit.xy, 'CB': fit.CB, 'has_F': np.asarray(fit.F is not None), 'F': fit.F if fit.F is not None else np.zeros((0,), np.float32), 'residual_exclusion_fraction': np.asarray(fit.residual_exclusion_fraction)}
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {'contract': parameters['save_grid_baseline_metadata_1cfe92f2'], 'run': fit.run, 'npz_sha256': sha, 'source': provenance, 'grid_shape': list(GRID), 'residual_exclusion_fraction': fit.residual_exclusion_fraction}
    sidecar = path.with_suffix('.json')
    temporary_json = sidecar.with_name(sidecar.stem + '.tmp.json')
    temporary_json.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    os.replace(temporary_json, sidecar)
    return receipt

def load_grid_baseline(path: Path, *, verify=True, parameters: Mapping):
    """Load original-grid baseline arrays and optionally verify the bound sidecar.

    Stored polynomial, intercepts, colour matrix and support fields are returned
    without recasting array dtypes."""
    path = Path(path)
    sidecar = path.with_suffix('.json')
    if verify:
        receipt = json.loads(sidecar.read_text(encoding='utf-8'))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if receipt.get('contract') != parameters['load_grid_baseline_metadata_1cfe92f2']:
            raise ValueError('Unknown frozen calibration contract')
        if sha != receipt.get('npz_sha256'):
            raise ValueError('Calibration SHA does not match sidecar')
    with np.load(path, allow_pickle=False) as z:
        keys = z['facies_keys'].astype(int)
        fit = Baseline(run=str(z['run']), M=z['M'], Minv=z['Minv'], P=z['P'], a={int(k): float(v) for k, v in zip(keys, z['facies_offsets'])}, n={int(k): int(v) for k, v in zip(keys, z['facies_counts'])}, fac=z['fac'], sand=z['sand'], chart=z['chart'], cpl=z['cpl'], Ug=z['Ug'], Vg=z['Vg'], taper=z['taper'], logB=z['logB'], gb0=z['gb0'], xy=z['xy'], CB=z['CB'], F=z['F'] if bool(z['has_F']) else None, residual_exclusion_fraction=float(z['residual_exclusion_fraction']))
    return fit

def sample_chart_log(log8, xy8, *, parameters: Mapping):
    """Median-sample log RGB at the explicit chart centres using nine remap probes."""
    GRID = tuple(parameters['grid_shape_hw'])
    arr = _log_rgb(log8, GRID, parameters=parameters)
    xy = np.asarray(xy8, np.float32)
    if xy.shape != (parameters['sample_chart_log_int_24'], 2):
        raise ValueError('Chart coordinates must be (24,2)')
    out = []
    for c in range(3):
        vals = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                x = (xy[:, 0] + dx).reshape(1, -1)
                y = (xy[:, 1] + dy).reshape(1, -1)
                vals.append(cv2.remap(arr[c], x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)[0])
        out.append(np.nanmedian(np.stack(vals), axis=0))
    return np.asarray(out, np.float32)

def light_basis(x, y, degree, *, parameters: Mapping):
    """Evaluate the achromatic frame-light polynomial at supplied coordinates.

    The fitted degree is explicit. Frozen centre and scale values come from
    parameters rather than being inferred from image dimensions."""
    x, y = ((np.asarray(x, float) - parameters['light_basis_int_85']) / parameters['light_basis_int_85'], (np.asarray(y, float) - parameters['light_basis_int_46']) / parameters['light_basis_int_46'])
    return np.stack([x ** a * y ** b for d in range(degree + 1) for a in range(d + 1) for b in [d - a]], axis=-1)

def estimate_frame_light_grid(fit: Baseline, frame_log8, *, frame_chart_xy8=None, frame_chart_usable=None, minimum_weak_sand_cells=None, parameters: Mapping):
    """Fit frame-light drift relative to its frozen baseline.

    Exact chart samples, weak sand and upper references enter the original
    weighted robust fit. parameters supplies the coarse grid, support limits
    and fit constants; no baseline is silently substituted for a missing frame."""
    if minimum_weak_sand_cells is None:
        minimum_weak_sand_cells = parameters['estimate_frame_light_grid_default_minimum_weak_sand_cells']
    GRID = tuple(parameters['grid_shape_hw'])
    LIGHT_WEIGHTS = np.array(parameters['light_weights'], dtype=np.float64)
    frame = _log_rgb(frame_log8, GRID, parameters=parameters)
    xy = fit.xy if frame_chart_xy8 is None else np.asarray(frame_chart_xy8, np.float32)
    current_chart = sample_chart_log(frame, xy, parameters=parameters)
    valid_chart = np.all(np.isfinite(current_chart), 0) & np.all(np.isfinite(fit.CB), 0)
    if frame_chart_usable is not None:
        valid_chart &= np.asarray(frame_chart_usable, bool)
    if valid_chart.sum() < parameters['estimate_frame_light_grid_int_6']:
        raise ValueError(f'Only {valid_chart.sum()} valid frame ColorChecker patches')
    with np.errstate(all='ignore'):
        coarse = np.nanmean(np.exp(frame).reshape(3, parameters['estimate_frame_light_grid_int_93'], parameters['estimate_frame_light_grid_int_4'], parameters['estimate_frame_light_grid_int_171'], parameters['estimate_frame_light_grid_int_4']), axis=(2, parameters['estimate_frame_light_grid_int_4']))
        coarse_b = np.nanmean(np.exp(fit.logB).reshape(3, parameters['estimate_frame_light_grid_int_93'], parameters['estimate_frame_light_grid_int_4'], parameters['estimate_frame_light_grid_int_171'], parameters['estimate_frame_light_grid_int_4']), axis=(2, parameters['estimate_frame_light_grid_int_4']))
        delta = np.log(coarse) - np.log(coarse_b)
    finite = np.all(np.isfinite(delta), 0)
    chroma_rg = delta[0] - delta[1]
    chroma_bg = delta[2] - delta[1]
    dchart = LIGHT_WEIGHTS @ (current_chart[:, valid_chart] - fit.CB[:, valid_chart])
    chart_centre = float(np.median(dchart))
    weak = finite & (np.abs(chroma_bg) < parameters['estimate_frame_light_grid_float_0_1']) & (np.abs(chroma_rg) < parameters['estimate_frame_light_grid_float_0_15']) & (np.linalg.norm(np.nan_to_num(delta - chart_centre), axis=0) < parameters['estimate_frame_light_grid_float_0_3'])
    fac = fit.fac
    yy8, xx8 = np.mgrid[:GRID[0], :GRID[1]]
    fixed_chart = (xx8 >= parameters['estimate_frame_light_grid_int_569']) & (yy8 < parameters['estimate_frame_light_grid_int_131'])
    top = np.argmax(fac >= 1, axis=0)
    upper8 = (fac == 0) & (yy8 < top[None, :] - parameters['estimate_frame_light_grid_int_6']) & (yy8 > parameters['estimate_frame_light_grid_int_28']) & ~fit.chart & ~fixed_chart & (xx8 > parameters['estimate_frame_light_grid_int_8']) & (xx8 < parameters['estimate_frame_light_grid_int_560'])
    f4 = fac.reshape(parameters['estimate_frame_light_grid_int_93'], parameters['estimate_frame_light_grid_int_4'], parameters['estimate_frame_light_grid_int_171'], parameters['estimate_frame_light_grid_int_4'])
    chart4 = fit.chart.reshape(parameters['estimate_frame_light_grid_int_93'], parameters['estimate_frame_light_grid_int_4'], parameters['estimate_frame_light_grid_int_171'], parameters['estimate_frame_light_grid_int_4']).any((1, 3))
    fixed4 = fixed_chart.reshape(parameters['estimate_frame_light_grid_int_93'], parameters['estimate_frame_light_grid_int_4'], parameters['estimate_frame_light_grid_int_171'], parameters['estimate_frame_light_grid_int_4']).any((1, 3))
    sand = ((f4 >= 1).mean((1, 3)) > parameters['estimate_frame_light_grid_float_0_99']) & ~chart4 & ~fixed4
    upper = upper8.reshape(parameters['estimate_frame_light_grid_int_93'], parameters['estimate_frame_light_grid_int_4'], parameters['estimate_frame_light_grid_int_171'], parameters['estimate_frame_light_grid_int_4']).mean((1, 3)) > parameters['estimate_frame_light_grid_float_0_99']
    y = np.tensordot(LIGHT_WEIGHTS, delta, axes=(0, 0))
    weak_sand = sand & weak
    degree = 2 if weak_sand.sum() >= minimum_weak_sand_cells else 1
    gy, gx = np.mgrid[:parameters['estimate_frame_light_grid_int_93'], :parameters['estimate_frame_light_grid_int_171']]
    X, Y, W = ([], [], [])
    cx, cy = (xy[valid_chart, 0] / parameters['estimate_frame_light_grid_int_4'] - parameters['estimate_frame_light_grid_float_0_375'], xy[valid_chart, 1] / parameters['estimate_frame_light_grid_int_4'] - parameters['estimate_frame_light_grid_float_0_375'])
    X.append(light_basis(cx, cy, degree, parameters=parameters))
    Y.append(dchart)
    W.append(np.full(len(dchart), parameters['estimate_frame_light_grid_float_0_25'] / len(dchart)))
    upper_weak = upper & weak
    if upper_weak.sum() > parameters['estimate_frame_light_grid_int_20']:
        X.append(light_basis(gx[upper_weak], gy[upper_weak], degree, parameters=parameters))
        Y.append(y[upper_weak])
        W.append(np.full(upper_weak.sum(), parameters['estimate_frame_light_grid_float_0_25'] / upper_weak.sum()))
    if weak_sand.sum() > parameters['estimate_frame_light_grid_int_50']:
        X.append(light_basis(gx[weak_sand], gy[weak_sand], degree, parameters=parameters))
        Y.append(y[weak_sand])
        W.append(np.full(weak_sand.sum(), parameters['estimate_frame_light_grid_float_0_5'] / weak_sand.sum()))
    design, target, weights = (np.vstack(X), np.concatenate(Y), np.concatenate(W))
    if len(target) < 3 * design.shape[1]:
        raise ValueError('Insufficient independent light references')
    huber = np.ones(len(target))
    for _ in range(parameters['estimate_frame_light_grid_int_8']):
        sw = np.sqrt(weights * huber)
        coef = np.linalg.lstsq(design * sw[:, None], target * sw, rcond=None)[0]
        huber = np.minimum(1, parameters['estimate_frame_light_grid_float_0_02'] / np.maximum(np.abs(target - design @ coef), parameters['estimate_frame_light_grid_float_1eminus_09']))
    iy, ix = np.mgrid[:GRID[0], :GRID[1]]
    field = light_basis((ix - parameters['estimate_frame_light_grid_float_1_5']) / parameters['estimate_frame_light_grid_int_4'], (iy - parameters['estimate_frame_light_grid_float_1_5']) / parameters['estimate_frame_light_grid_int_4'], degree, parameters=parameters) @ coef
    info = {'degree': degree, 'n_chart': int(valid_chart.sum()), 'n_weak_sand': int(weak_sand.sum()), 'fraction_weak_sand': float(weak_sand.sum() / max(sand.sum(), 1)), 'n_weak_upper': int(upper_weak.sum()), 'chart_log_drift': chart_centre, 'coefficients': coef.tolist()}
    return (field.astype(np.float32), info)

@dataclass(frozen=True)
class ReferenceGrid:
    """Explicit cell geometry and the frozen photometry parameter mapping."""
    shape_hw: tuple[int, int]
    origin_cell_xy: tuple[int, int]
    original_shape_hw: tuple[int, int]
    parameters: Mapping = field(kw_only=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        parameters = self.parameters
        if self.shape_hw[0] <= 0 or self.shape_hw[1] <= 0:
            raise ValueError('Grid shape must be positive')
        if self.original_shape_hw[0] <= 0 or self.original_shape_hw[1] <= 0:
            raise ValueError('Original grid shape must be positive')

    def normalized_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return float64 x/y coordinates normalized about the parent grid centre."""
        parameters = self.parameters
        yy, xx = np.mgrid[:self.shape_hw[0], :self.shape_hw[1]]
        x = (xx + self.origin_cell_xy[0] - self.original_shape_hw[1] / 2) / (self.original_shape_hw[1] / 2)
        y = (yy + self.origin_cell_xy[1] - self.original_shape_hw[0] / 2) / (self.original_shape_hw[0] / 2)
        return (x.astype(np.float64), y.astype(np.float64))

    def basis(self, *, include_constant: bool=False) -> np.ndarray:
        """Build the ordered float64 polynomial basis on the normalized reference grid."""
        parameters = self.parameters
        x, y = self.normalized_coordinates()
        first = 0 if include_constant else 1
        return np.stack([x ** ax * y ** (degree - ax) for degree in range(first, parameters['reference_grid_basis_int_5']) for ax in range(degree + 1)]).astype(np.float64)

def require_green_donor_only_receipt(receipt: dict, *, parameters: Mapping) -> None:
    """Require the source-observed optical-fit eligibility recorded by the caller."""
    if receipt.get('photo_sand_qa_scope') != 'green_donor_mask_only' or receipt.get('green_donor_only_optical_fit_eligible') is not True or receipt.get('photo_sand_qa_pass') is not True:
        raise ValueError('Source-photo green donor mask is not fit-eligible')

def _observed_rgb(array: np.ndarray, grid: ReferenceGrid, *, parameters: Mapping) -> np.ndarray:
    """Require float32 log RGB on the supplied reference-grid geometry."""
    raw = np.asarray(array)
    if raw.dtype != np.float32:
        raise TypeError('frozen requires float32 log-linear RAW; uint8/JPEG is invalid')
    return _log_rgb(raw, grid.shape_hw, parameters=parameters)

def _fit_observed_field(green: np.ndarray, fac: np.ndarray, keep: np.ndarray, grid: ReferenceGrid, *, parameters: Mapping) -> tuple[np.ndarray, dict[int, float], dict[int, int]]:
    """Fit the polynomial and one source-observed optical intercept.

    This optical class is independent of physical facies; its original
    robust solve is intentionally distinct from the multi-facies fit."""
    basis = grid.basis().reshape(parameters['fit_observed_field_int_14'], -1)
    idx = np.flatnonzero((keep & (fac >= 1) & np.isfinite(green)).ravel())
    if len(idx) < parameters['fit_observed_field_int_1000']:
        raise ValueError(f'Too few source-certified sand cells: {len(idx)}')
    ff = fac.ravel()[idx]
    classes = [1]
    X = np.column_stack([basis[:, idx].T] + [(ff == f).astype(float) for f in classes])
    z = green.ravel()[idx].astype(float)
    weight = np.ones(len(z))
    for _ in range(parameters['fit_observed_field_int_6']):
        sw = np.sqrt(weight)
        coef = np.linalg.lstsq(X * sw[:, None], z * sw, rcond=None)[0]
        weight = np.minimum(1, parameters['fit_observed_field_float_0_05'] / np.maximum(np.abs(z - X @ coef), parameters['fit_observed_field_float_1eminus_09']))
    return (coef[:parameters['fit_observed_field_int_14']], {f: float(coef[parameters['fit_observed_field_int_14'] + i]) for i, f in enumerate(classes)}, {f: int(np.count_nonzero(ff == f)) for f in classes})

def fit_observed_baseline(*, run: str, logB: np.ndarray, facies: np.ndarray, chart_xy: np.ndarray, chart_log_rgb: np.ndarray, ref_chart_log_rgb: np.ndarray, chart_usable: np.ndarray, matrix: np.ndarray, rig_consensus: np.ndarray, grid: ReferenceGrid, flat: np.ndarray | None=None, parameters: Mapping) -> Baseline:
    """Fit a baseline using the independently observed zero/one sand support.

    grid defines expanded reference cells and their original coordinates.
    Chart samples, colour matrix, consensus and optional flat are explicit;
    no phase labels or model predictions determine optical support."""
    logB = _observed_rgb(logB, grid, parameters=parameters)
    cons = _observed_rgb(rig_consensus, grid, parameters=parameters)
    fac = np.asarray(facies, np.int16)
    xy = np.asarray(chart_xy, np.float64)
    M = np.asarray(matrix, np.float64)
    if fac.shape != grid.shape_hw or xy.shape != (parameters['fit_observed_baseline_int_24'], 2) or M.shape != (3, 3):
        raise ValueError('Sand mask, chart lattice or matrix shape mismatch')
    if not np.array_equal(np.unique(fac), np.array([0, 1], np.int16)) and (not (np.unique(fac).size == 1 and int(np.unique(fac)[0]) in (0, 1))):
        raise ValueError('Observed-sand optical fit accepts only 0/1 classes')
    Minv = np.linalg.inv(M)
    B1 = _colour(logB, Minv, parameters=parameters)
    F = None if flat is None else _observed_rgb(flat, grid, parameters=parameters)
    yy, xx = np.mgrid[:grid.shape_hw[0], :grid.shape_hw[1]]
    validxy = np.all(np.isfinite(xy), 1)
    if validxy.sum() < parameters['fit_observed_baseline_int_6']:
        raise ValueError('Missing source-bound ColorChecker lattice')
    chart = (xx >= xy[validxy, 0].min() - parameters['fit_observed_baseline_int_7']) & (xx <= xy[validxy, 0].max() + parameters['fit_observed_baseline_int_7']) & (yy >= xy[validxy, 1].min() - parameters['fit_observed_baseline_int_7']) & (yy <= xy[validxy, 1].max() + parameters['fit_observed_baseline_int_7'])
    native_x_cell = xx + grid.origin_cell_xy[0]
    native_y_cell = yy + grid.origin_cell_xy[1]
    fixed_chart = (native_x_cell >= parameters['fit_observed_baseline_int_569']) & (native_y_cell < parameters['fit_observed_baseline_int_131'])
    sand = (fac == 1) & ~chart & ~fixed_chart
    keep = sand & _same_fac(fac, parameters=parameters)
    L = B1 - (F if F is not None else 0)
    dbg = L[2] - L[1] - (cons[2] - cons[1])
    finite = keep & np.isfinite(dbg)
    num = cv2.GaussianBlur(np.where(finite, dbg, 0).astype(np.float32), (0, 0), parameters['fit_observed_baseline_gaussian_sigma'])
    den = cv2.GaussianBlur(finite.astype(np.float32), (0, 0), parameters['fit_observed_baseline_gaussian_sigma'])
    dbg = np.where(finite, num / np.maximum(den, parameters['fit_observed_baseline_float_1eminus_06']), np.nan)
    m = keep & np.isfinite(dbg)
    if m.sum() > parameters['fit_observed_baseline_int_100']:
        dbg[fac == 1] -= np.median(dbg[m])
    fitkeep = keep & np.isfinite(dbg) & (dbg > -parameters['fit_observed_baseline_float_0_08'])
    P, a, n = _fit_observed_field(B1[1], fac, fitkeep, grid, parameters=parameters)
    fraction = float(1 - fitkeep.sum() / max(keep.sum(), 1))
    dist = cv2.distanceTransform((~(sand | chart)).astype(np.uint8), cv2.DIST_L2, parameters['fit_observed_baseline_int_5'])
    taper = np.exp(-dist ** 2 / (2 * parameters['fit_observed_baseline_float_8_0'] ** 2)).astype(np.float32)
    cpl = _chart_plane(chart_log_rgb, ref_chart_log_rgb, chart_usable, M, parameters=parameters)
    X = np.column_stack([np.ones(validxy.sum()), xy[validxy]])
    patch = np.arange(parameters['fit_observed_baseline_int_24'])
    au = np.linalg.lstsq(X, (patch % parameters['fit_observed_baseline_int_4'] - parameters['fit_observed_baseline_float_1_5'])[validxy], rcond=None)[0]
    av = np.linalg.lstsq(X, (patch // parameters['fit_observed_baseline_int_4'] - parameters['fit_observed_baseline_float_2_5'])[validxy], rcond=None)[0]
    Ug = (au[0] + au[1] * xx + au[2] * yy).astype(np.float32)
    Vg = (av[0] + av[1] * xx + av[2] * yy).astype(np.float32)
    return Baseline(run, M, Minv, P, a, n, fac, sand, chart, cpl, Ug, Vg, taper, logB, B1[1] - B1[2], xy, np.asarray(chart_log_rgb, np.float32), F, fraction)

def observed_baseline_field(fit: Baseline, reference: Baseline, grid: ReferenceGrid, *, parameters: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the achromatic field shared by observed baseline and reference.

    Intercepts apply only where both fits support sand. The original polynomial,
    taper and chart blend provide the remaining correction field."""
    if fit.fac.shape != grid.shape_hw or reference.fac.shape != grid.shape_hw:
        raise ValueError('Baseline fit geometry does not match reference grid')
    poly = np.einsum('i,ihw->hw', reference.P - fit.P, grid.basis())
    common = [f for f in fit.a if f in reference.a]
    if not common:
        raise ValueError('No shared source-observed optical class')
    values = np.array([reference.a[f] - fit.a[f] for f in common])
    weights = np.array([fit.n[f] for f in common])
    meanoff = float(np.average(values, weights=weights))
    offset = np.full(grid.shape_hw, np.nan)
    for fac, val in zip(common, values):
        offset[(fit.fac == fac) & fit.sand & reference.sand] = val
    insand = np.isfinite(offset)
    total = np.where(insand, poly + offset, np.nan)
    global_level = float(np.nanmedian(total))
    field = np.where(insand, total, fit.taper * (poly + meanoff) + (1 - fit.taper) * global_level)
    dist = cv2.distanceTransform((~fit.chart).astype(np.uint8), cv2.DIST_L2, parameters['observed_baseline_field_int_5'])
    wc = np.exp(-dist ** 2 / (2 * parameters['observed_baseline_field_float_20_0'] ** 2))
    area = ~insand & ~fit.chart & (dist < parameters['observed_baseline_field_int_60'])
    field[area] = wc[area] * -fit.cpl[0] + (1 - wc[area]) * field[area]
    field[fit.chart] = -(fit.cpl[0] + fit.cpl[1] * fit.Ug[fit.chart] + fit.cpl[2] * fit.Vg[fit.chart])
    return (field.astype(np.float32), insand)

def apply_grid(log_rgb: np.ndarray, fit: Baseline, reference: Baseline, grid: ReferenceGrid, *, light_field: np.ndarray | None=None, source_eligible_mask: np.ndarray | None=None, parameters: Mapping) -> np.ndarray:
    """Correct float32 reference-grid RGB with optional light and eligibility mask.

    Matrix inversion, baseline field and saturation-dependent flat correction
    keep their original order. Eligibility is supplied independently of the fit."""
    GB_SAT = parameters['green_blue_saturation']
    src = _observed_rgb(log_rgb, grid, parameters=parameters)
    field, _ = observed_baseline_field(fit, reference, grid, parameters=parameters)
    if light_field is not None:
        light = np.asarray(light_field, np.float32)
        if light.shape != grid.shape_hw:
            raise ValueError('Frame-light field has wrong reference geometry')
        field = field - light
    rgb = _colour(src, fit.Minv, parameters=parameters)
    rgb += field[None]
    if fit.F is not None:
        sand = fit.sand & reference.sand & ~fit.chart
        alpha = np.clip((GB_SAT - (rgb[1] - rgb[2])) / np.maximum(GB_SAT - fit.gb0, parameters['apply_grid_float_0_2']), 0, 1)
        alpha = np.where(np.isfinite(alpha), alpha, 1)
        tap = np.where(sand, 1.0, fit.taper * ~fit.chart)
        neutral = fit.F[1:2]
        rgb -= (neutral + alpha[None] * (fit.F - neutral)) * tap[None]
    if source_eligible_mask is not None:
        eligible = np.asarray(source_eligible_mask, bool)
        if eligible.shape != grid.shape_hw:
            raise ValueError('Output source-eligibility mask has wrong grid shape')
        rgb[:, ~eligible] = np.nan
    return rgb.astype(np.float32)

def apply_native(log_rgb_run: np.ndarray, fit: Baseline, reference: Baseline, grid: ReferenceGrid, h_ref_to_run: np.ndarray, *, source_eligible_mask: np.ndarray, light_field_ref8: np.ndarray | None=None, destination_origin_xy: tuple[int, int]=None, tile_rows: int=None, parameters: Mapping) -> np.ndarray:
    """Project correction fields onto the original run-native RGB pixels.

    h_ref_to_run maps reference-native coordinates to this source. The optional
    destination_origin_xy locates a camera window; grid supplies cell origin.
    Only fields are resampled. Source RGB is never warped, and invalid or
    independently ineligible pixels remain NaN."""
    if destination_origin_xy is None:
        destination_origin_xy = tuple(parameters['apply_native_default_destination_origin_xy'])
    if tile_rows is None:
        tile_rows = parameters['apply_native_default_tile_rows']
    GB_SAT = parameters['green_blue_saturation']
    source = np.asarray(log_rgb_run)
    if source.dtype != np.float32:
        raise TypeError('frozen requires float32 log-linear RAW; uint8/JPEG is invalid')
    source = _log_rgb(source, parameters=parameters)
    if source.ndim != 3 or source.shape[0] != 3:
        raise ValueError('Expected channel-first native log-linear RAW')
    height, width = source.shape[1:]
    eligible = np.asarray(source_eligible_mask, bool)
    if eligible.shape != (height, width):
        raise ValueError('Native source eligibility mask shape mismatch')
    if fit.fac.shape != grid.shape_hw or reference.fac.shape != grid.shape_hw:
        raise ValueError('Fit and virtual reference use a different grid')
    H = np.asarray(h_ref_to_run, np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < parameters['apply_native_float_1eminus_10']:
        raise ValueError('Expected finite, invertible native homography')
    if tile_rows < 1:
        raise ValueError('tile_rows must be positive')
    light = None
    if light_field_ref8 is not None:
        light = np.asarray(light_field_ref8, np.float32)
        if light.shape != grid.shape_hw:
            raise ValueError('Frame-light field uses a different grid')
    field_ref, _ = observed_baseline_field(fit, reference, grid, parameters=parameters)
    inverse = np.linalg.inv(H)
    output = np.empty_like(source)

    def sample(array: np.ndarray, mx: np.ndarray, my: np.ndarray) -> np.ndarray:
        planes = np.asarray(array, np.float32)
        if planes.ndim == 2:
            return cv2.remap(planes, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return np.stack([sample(plane, mx, my) for plane in planes])
    ox, oy = destination_origin_xy
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        yy, xx = np.mgrid[start:stop, :width]
        xrun = xx.astype(np.float64) + ox
        yrun = yy.astype(np.float64) + oy
        den = inverse[2, 0] * xrun + inverse[2, 1] * yrun + inverse[2, 2]
        if np.any(np.abs(den) < parameters['apply_native_float_1eminus_09']):
            raise ValueError('Native pixel maps to reference infinity')
        xref = (inverse[0, 0] * xrun + inverse[0, 1] * yrun + inverse[0, 2]) / den
        yref = (inverse[1, 0] * xrun + inverse[1, 1] * yrun + inverse[1, 2]) / den
        mx = ((xref - parameters['apply_native_float_3_5']) / parameters['apply_native_int_8'] - grid.origin_cell_xy[0]).astype(np.float32)
        my = ((yref - parameters['apply_native_float_3_5']) / parameters['apply_native_int_8'] - grid.origin_cell_xy[1]).astype(np.float32)
        in_field = (mx >= -parameters['apply_native_float_0_5']) & (mx <= grid.shape_hw[1] - parameters['apply_native_float_0_5']) & (my >= -parameters['apply_native_float_0_5']) & (my <= grid.shape_hw[0] - parameters['apply_native_float_0_5'])
        field = sample(field_ref, mx, my)
        if light is not None:
            field -= sample(light, mx, my)
        corrected = _colour(source[:, start:stop], fit.Minv, parameters=parameters)
        corrected += field[None]
        if fit.F is not None:
            flat = sample(fit.F, mx, my)
            gb0 = sample(fit.gb0, mx, my)
            taper = sample(fit.taper, mx, my)
            chart = sample(fit.chart.astype(np.float32), mx, my)
            alpha = np.clip((GB_SAT - (corrected[1] - corrected[2])) / np.maximum(GB_SAT - gb0, parameters['apply_native_float_0_2']), 0, 1)
            alpha = np.where(np.isfinite(alpha), alpha, 1)
            neutral = flat[1:2]
            tap = taper * (1 - np.clip(chart, 0, 1))
            corrected -= (neutral + alpha[None] * (flat - neutral)) * tap[None]
        valid = eligible[start:stop] & in_field & np.all(np.isfinite(source[:, start:stop]), axis=0) & np.all(np.isfinite(corrected), axis=0)
        corrected[:, ~valid] = np.nan
        output[:, start:stop] = corrected
    return output

def estimate_frame_light_expanded(fit: Baseline, frame_log_ref8: np.ndarray, grid: ReferenceGrid, *, frame_chart_xy_ref8: np.ndarray, frame_chart_usable: np.ndarray, parameters: Mapping) -> tuple[np.ndarray, dict]:
    """Fit frame light on the original centre and evaluate it on the expanded grid.

    The polynomial is fitted once from exact chart and centre signals. Its
    expanded evaluation retains the original centre bytes; added cells do not
    establish independent source eligibility."""
    frame = _observed_rgb(frame_log_ref8, grid, parameters=parameters)
    x0, y0 = (-grid.origin_cell_xy[0], -grid.origin_cell_xy[1])
    h, w = grid.original_shape_hw
    if x0 < 0 or y0 < 0 or x0 + w > grid.shape_hw[1] or (y0 + h > grid.shape_hw[0]):
        raise ValueError('Expanded grid does not contain the frozen frozen centre')
    centre = (slice(y0, y0 + h), slice(x0, x0 + w))
    xy = np.asarray(frame_chart_xy_ref8, np.float32)
    usable = np.asarray(frame_chart_usable, bool)
    if xy.shape != (parameters['estimate_frame_light_expanded_int_24'], 2) or usable.shape != (parameters['estimate_frame_light_expanded_int_24'],):
        raise ValueError('Exact per-frame source-bound chart required')
    local_xy = xy - np.array([x0, y0], np.float32)
    central_fit = replace(fit, fac=fit.fac[centre], sand=fit.sand[centre], chart=fit.chart[centre], Ug=fit.Ug[centre], Vg=fit.Vg[centre], taper=fit.taper[centre], logB=fit.logB[slice(None), *centre], gb0=fit.gb0[centre], xy=fit.xy - np.array([x0, y0], np.float32), F=None if fit.F is None else fit.F[slice(None), *centre])
    original_field, info = estimate_frame_light_grid(central_fit, frame[slice(None), *centre], frame_chart_xy8=local_xy, frame_chart_usable=usable, parameters=parameters)
    degree = int(info['degree'])
    coef = np.asarray(info['coefficients'], np.float64)
    yy, xx = np.mgrid[:grid.shape_hw[0], :grid.shape_hw[1]]
    old_x = xx + grid.origin_cell_xy[0]
    old_y = yy + grid.origin_cell_xy[1]
    full = light_basis((old_x - parameters['estimate_frame_light_expanded_float_1_5']) / parameters['estimate_frame_light_expanded_int_4'], (old_y - parameters['estimate_frame_light_expanded_float_1_5']) / parameters['estimate_frame_light_expanded_int_4'], degree, parameters=parameters) @ coef
    full = full.astype(np.float32)
    if not np.array_equal(full[centre], original_field):
        raise ArithmeticError('Expanded frame light changed frozen central field')
    info.update({'coordinate_frame': parameters['estimate_frame_light_expanded_metadata_0ec005d0'], parameters['estimate_frame_light_expanded_metadata_77e7cf6f']: True, 'added_ring_status': 'diagnostic polynomial forecast; independent QA required', 'grid_origin_cell_xy': list(grid.origin_cell_xy)})
    return (full, info)

def make_local_flat(corrected_new: list[np.ndarray], consensus: np.ndarray, reference: Baseline, common_new_support: np.ndarray, grid: ReferenceGrid, *, parameters: Mapping) -> np.ndarray:
    """Estimate the frozen local flat from corrected donors and independent support.

    Channel residual medians are masked, Gaussian-smoothed and green-centred
    using the explicit original constants."""
    if not corrected_new:
        raise ValueError('At least one corrected new-light donor required')
    _observed_rgb(consensus, grid, parameters=parameters)
    difference = np.nanmedian(np.stack([a - consensus for a in corrected_new]), axis=0)
    valid = reference.sand & np.asarray(common_new_support, bool) & np.all(np.isfinite(difference), axis=0)
    den = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), parameters['make_local_flat_gaussian_sigma'])
    F = np.zeros_like(difference, dtype=np.float32)
    for c in range(3):
        num = cv2.GaussianBlur(np.where(valid, difference[c], 0).astype(np.float32), (0, 0), parameters['make_local_flat_gaussian_sigma'])
        F[c] = num / np.maximum(den, parameters['make_local_flat_float_1eminus_06'])
    F[:, den < parameters['make_local_flat_float_0_05']] = 0
    if valid.sum() > parameters['make_local_flat_int_100']:
        F[:, valid] -= np.median(F[1, valid])
    return F

def save_baseline(path: Path, fit: Baseline, grid: ReferenceGrid, provenance: dict, *, parameters: Mapping) -> dict:
    """Serialize an expanded observed-sand fit with explicit geometry and source hashes.

    The NPZ stores original member names and dtypes; its sidecar binds the
    hash, grid origin and original-grid shape without pickle."""
    path = Path(path)
    if path.suffix.lower() != '.npz' or fit.fac.shape != grid.shape_hw:
        raise ValueError('Expected .npz and exact reference grid')
    required = ('source_arw_sha256', 'source_review_sha256', 'source_manifest_sha256', 'sand_mask_sha256')
    if any((not provenance.get(k) for k in required)):
        raise ValueError('Missing source, geometry or sand-mask SHA binding')
    keys = np.asarray(sorted(fit.a), dtype=np.int16)
    arrays = {'run': np.asarray(fit.run), 'M': fit.M, 'Minv': fit.Minv, 'P': fit.P, 'facies_keys': keys, 'facies_offsets': np.asarray([fit.a[int(k)] for k in keys]), 'facies_counts': np.asarray([fit.n[int(k)] for k in keys]), 'fac': fit.fac, 'sand': fit.sand, 'chart': fit.chart, 'cpl': fit.cpl, 'Ug': fit.Ug, 'Vg': fit.Vg, 'taper': fit.taper, 'logB': fit.logB, 'gb0': fit.gb0, 'xy': fit.xy, 'CB': fit.CB, 'has_F': np.asarray(fit.F is not None), 'F': fit.F if fit.F is not None else np.zeros((0,), np.float32), 'residual_exclusion_fraction': np.asarray(fit.residual_exclusion_fraction)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {'contract': parameters['save_baseline_metadata_fb318c57'], 'run': fit.run, 'npz_sha256': digest, 'grid_shape_hw': list(grid.shape_hw), 'grid_origin_cell_xy': list(grid.origin_cell_xy), 'original_grid_shape_hw': list(grid.original_shape_hw), 'source': provenance, 'residual_exclusion_fraction': fit.residual_exclusion_fraction}
    sidecar = path.with_suffix('.json')
    temp_json = sidecar.with_name(sidecar.stem + '.tmp.json')
    temp_json.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    os.replace(temp_json, sidecar)
    return receipt

@retry_io
def load_baseline(path: Path, *, verify: bool=True, parameters: Mapping) -> tuple[Baseline, ReferenceGrid]:
    """Load a bound observed-sand NPZ and its explicit ReferenceGrid.

    The sidecar contract is required even when verify is false. By default its
    SHA is checked before loading, and the stored mask must match grid geometry."""
    path = Path(path)
    receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if receipt.get('contract') != parameters['load_baseline_metadata_fb318c57']:
        raise ValueError('Unknown observed-sand baseline contract')
    if verify and hashlib.sha256(path.read_bytes()).hexdigest() != receipt.get('npz_sha256'):
        raise ValueError('Baseline NPZ SHA mismatch')
    grid = ReferenceGrid(tuple(receipt['grid_shape_hw']), tuple(receipt['grid_origin_cell_xy']), tuple(receipt['original_grid_shape_hw']), parameters=parameters)
    fit = load_grid_baseline(path, verify=False, parameters=parameters)
    if fit.fac.shape != grid.shape_hw:
        raise ValueError('Baseline mask does not match bound reference grid')
    return (fit, grid)

def validate_homography(h_ref_to_run: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    """Validate and normalise the supplied reference-to-run homography.

    Finite shape, determinant and normalising-element limits are explicit
    frozen parameter DATA; the matrix is divided by its original bottom-right value."""
    h = np.asarray(h_ref_to_run, dtype=np.float64)
    if h.shape != (3, 3) or not np.all(np.isfinite(h)):
        raise ValueError('Expected finite 3x3 cross-run homography')
    if abs(np.linalg.det(h)) < parameters['validate_homography_float_1eminus_10'] or abs(h[2, 2]) < parameters['validate_homography_float_1eminus_10']:
        raise ValueError('Degenerate cross-run homography')
    return h / h[2, 2]

def _map_coordinates(h_ref_to_run: np.ndarray, *, source_shape: tuple[int, int], destination_shape: tuple[int, int], destination_y0: int=None, destination_y1: int | None=None, run_to_reference: bool=False, parameters: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Build inverse-sampling maps at the original native-pixel cell centres.

    The source and destination grids are explicit; only correction fields or
    fitting signals use these maps, not a replacement source-image geometry."""
    if destination_y0 is None:
        destination_y0 = parameters['map_coordinates_default_destination_y0']
    GRID = tuple(parameters['grid_shape_hw'])
    NATIVE = tuple(parameters['native_shape_hw'])
    src = tuple(source_shape)
    dst = tuple(destination_shape)
    if src not in (GRID, NATIVE) or dst not in (GRID, NATIVE):
        raise ValueError(f'Unexpected source/destination geometry {src} -> {dst}')
    y1 = dst[0] if destination_y1 is None else destination_y1
    if not 0 <= destination_y0 < y1 <= dst[0]:
        raise ValueError('Invalid destination tile')
    scale_out = parameters['map_coordinates_int_8'] if dst == GRID else 1
    scale_in = parameters['map_coordinates_int_8'] if src == GRID else 1
    yy, xx = np.mgrid[destination_y0:y1, 0:dst[1]]
    xn = scale_out * xx.astype(np.float64) + (scale_out - 1) / 2
    yn = scale_out * yy.astype(np.float64) + (scale_out - 1) / 2
    h = validate_homography(h_ref_to_run, parameters=parameters)
    t = h if run_to_reference else np.linalg.inv(h)
    den = t[2, 0] * xn + t[2, 1] * yn + t[2, 2]
    if np.any(np.abs(den) < parameters['map_coordinates_float_1eminus_09']):
        raise ValueError('Homography maps a destination pixel to infinity')
    xs = (t[0, 0] * xn + t[0, 1] * yn + t[0, 2]) / den
    ys = (t[1, 0] * xn + t[1, 1] * yn + t[1, 2]) / den
    mx = ((xs - (scale_in - 1) / 2) / scale_in).astype(np.float32)
    my = ((ys - (scale_in - 1) / 2) / scale_in).astype(np.float32)
    return (mx, my)

def remap_to_reference(run_array: np.ndarray, h_ref_to_run: np.ndarray, *, categorical: bool=False, destination_shape: tuple[int, int]=None, parameters: Mapping) -> np.ndarray:
    """Sample a run signal or categorical mask in the reference coordinate grid.

    Continuous values interpolate signal and finite support separately using
    the original thresholds. Categorical values use nearest-neighbour sampling."""
    if destination_shape is None:
        destination_shape = tuple(parameters['remap_to_reference_default_destination_shape'])
    a = np.asarray(run_array)
    if a.ndim not in (2, 3):
        raise ValueError('Expected 2D or channel-first 3D array')
    src_shape = a.shape[-2:]
    mx, my = _map_coordinates(h_ref_to_run, source_shape=src_shape, destination_shape=destination_shape, run_to_reference=True, parameters=parameters)
    mode = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
    planes = a[None] if a.ndim == 2 else a
    out = []
    for plane in planes:
        if categorical:
            z = cv2.remap(np.asarray(plane), mx, my, mode, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        else:
            valid = np.isfinite(plane).astype(np.float32)
            num = cv2.remap(np.where(np.isfinite(plane), plane, 0).astype(np.float32), mx, my, mode, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            den = cv2.remap(valid, mx, my, mode, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            z = np.where(den > parameters['remap_to_reference_float_0_999'], num / np.maximum(den, parameters['remap_to_reference_float_1eminus_06']), np.nan)
        out.append(z)
    return out[0] if a.ndim == 2 else np.stack(out)

def chart_xy_to_reference(chart_xy_run8: np.ndarray, h_ref_to_run: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    """Project explicit chart centres through the inverse native homography.

    Cell scale and centre offset preserve the original reference-coordinate
    arithmetic and float64 transformation."""
    xy = np.asarray(chart_xy_run8, dtype=np.float64)
    if xy.shape != (parameters['chart_xy_to_reference_int_24'], 2):
        raise ValueError('Expected the configured chart lattice')
    h = np.linalg.inv(validate_homography(h_ref_to_run, parameters=parameters))
    p = parameters['chart_xy_to_reference_int_8'] * xy + parameters['chart_xy_to_reference_float_3_5']
    q = cv2.perspectiveTransform(p.astype(np.float64)[None], h)[0]
    return (q - parameters['chart_xy_to_reference_float_3_5']) / parameters['chart_xy_to_reference_int_8']

def _remap_reference_tile(ref_array: np.ndarray, h_ref_to_run: np.ndarray, destination_shape: tuple[int, int], y0: int, y1: int, *, coordinates: tuple[np.ndarray, np.ndarray] | None=None, categorical: bool=False, parameters: Mapping) -> np.ndarray:
    """Sample a reference field into one source tile with optional shared maps."""
    GRID = tuple(parameters['grid_shape_hw'])
    a = np.asarray(ref_array)
    if coordinates is None:
        mx, my = _map_coordinates(h_ref_to_run, source_shape=a.shape[-2:], destination_shape=destination_shape, destination_y0=y0, destination_y1=y1, parameters=parameters)
    else:
        if a.shape[-2:] != GRID:
            raise ValueError('Shared reference coordinates require the configured cell grid')
        mx, my = coordinates
    mode = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
    planes = a[None] if a.ndim == 2 else a
    out = [cv2.remap(np.asarray(p, np.float32), mx, my, mode, borderMode=cv2.BORDER_CONSTANT if categorical else cv2.BORDER_REPLICATE, borderValue=0) for p in planes]
    return out[0] if a.ndim == 2 else np.stack(out)

def _native_exact_sand_field(fit_ref: Baseline, reference: Baseline, facies_tile: np.ndarray, mx_ref8: np.ndarray, my_ref8: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    """Accumulate the reference polynomial and facies offsets at physical points.

    Each polynomial term is cast to float32 before its original in-place sum."""
    GRID = tuple(parameters['grid_shape_hw'])
    x = (mx_ref8 - GRID[1] / 2) / (GRID[1] / 2)
    y = (my_ref8 - GRID[0] / 2) / (GRID[0] / 2)
    coef = reference.P - fit_ref.P
    field = np.zeros(facies_tile.shape, np.float32)
    k = 0
    for d in range(1, parameters['native_exact_sand_field_int_5']):
        for ax in range(d + 1):
            ay = d - ax
            field += (coef[k] * x ** ax * y ** ay).astype(np.float32)
            k += 1
    for fac, level in fit_ref.a.items():
        if fac in reference.a:
            field[facies_tile == fac] += reference.a[fac] - level
    return field

def estimate_frame_light_reference_geometry(fit_ref: Baseline, frame_log_run8: np.ndarray, h_ref_to_run: np.ndarray, *, frame_chart_xy_run8: np.ndarray, frame_chart_usable: np.ndarray, parameters: Mapping) -> tuple[np.ndarray, dict]:
    """Fit frame-light drift after mapping the supplied signal and chart to reference.

    The exact frame chart remains mandatory; there is no inferred chart fallback."""
    GRID = tuple(parameters['grid_shape_hw'])
    source = _log_rgb(frame_log_run8, GRID, parameters=parameters)
    xy = np.asarray(frame_chart_xy_run8, np.float64)
    usable = np.asarray(frame_chart_usable, bool)
    if xy.shape != (parameters['estimate_frame_light_reference_geometry_int_24'], 2) or usable.shape != (parameters['estimate_frame_light_reference_geometry_int_24'],):
        raise ValueError('Exact frame chart centres and usable mask required')
    frame_ref = remap_to_reference(source, h_ref_to_run, parameters=parameters)
    xy_ref = chart_xy_to_reference(xy, h_ref_to_run, parameters=parameters)
    field, info = estimate_frame_light_grid(fit_ref, frame_ref, frame_chart_xy8=xy_ref, frame_chart_usable=usable, parameters=parameters)
    info['coordinate_frame'] = parameters['estimate_frame_light_reference_geometry_metadata_85dfef1a']
    info['frame_chart_source'] = 'exact-run-frame-registered-then-projected'
    return (field, info)

def apply_reference_geometry(log_rgb_run: np.ndarray, fit_ref: Baseline, reference: Baseline, h_ref_to_run: np.ndarray, *, facies_run_native: np.ndarray | None=None, light_field_ref8: np.ndarray | None=None, tile_rows: int=None, parameters: Mapping) -> np.ndarray:
    """Apply a reference-bound colour and optical field without warping source RGB.

    Explicit native support, optional light and homography define each tile.
    The existing support bounds and term-wise float32 native polynomial remain
    unchanged, including the original truncated-edge treatment."""
    if tile_rows is None:
        tile_rows = parameters['apply_reference_geometry_default_tile_rows']
    GRID = tuple(parameters['grid_shape_hw'])
    NATIVE = tuple(parameters['native_shape_hw'])
    GB_SAT = parameters['green_blue_saturation']
    src = _log_rgb(log_rgb_run, parameters=parameters)
    shape = src.shape[1:]
    if shape not in (GRID, NATIVE):
        raise ValueError(f'Unexpected run source shape {shape}')
    if fit_ref.fac.shape != GRID or reference.fac.shape != GRID:
        raise ValueError('Fits must use the reference grid')
    if shape == NATIVE:
        if facies_run_native is None or np.shape(facies_run_native) != NATIVE:
            raise ValueError('Native correction requires a bound optical-support mask')
    h = validate_homography(h_ref_to_run, parameters=parameters)
    field_ref, _ = baseline_field(fit_ref, reference, parameters=parameters)
    light_ref = None
    if light_field_ref8 is not None:
        lf = np.asarray(light_field_ref8, np.float32)
        if lf.shape != GRID:
            raise ValueError('Per-frame light field must be in frozen reference grid')
        light_ref = lf
    out = np.empty_like(src)
    for lo in range(0, shape[0], tile_rows):
        hi = min(lo + tile_rows, shape[0])
        mx, my = _map_coordinates(h, source_shape=GRID, destination_shape=shape, destination_y0=lo, destination_y1=hi, parameters=parameters)
        coords = (mx, my)
        field = _remap_reference_tile(field_ref, h, shape, lo, hi, coordinates=coords, parameters=parameters)
        chart = _remap_reference_tile(fit_ref.chart.astype(np.float32), h, shape, lo, hi, coordinates=coords, categorical=True, parameters=parameters) > parameters['apply_reference_geometry_float_0_5']
        taper = _remap_reference_tile(fit_ref.taper, h, shape, lo, hi, coordinates=coords, parameters=parameters)
        if shape == NATIVE:
            facies = np.asarray(facies_run_native[lo:hi])
        else:
            facies = _remap_reference_tile(fit_ref.fac, h, shape, lo, hi, coordinates=coords, categorical=True, parameters=parameters).astype(np.int16)
        sand = (facies >= 1) & ~chart
        valid_ref = np.isfinite(field) & (mx >= -parameters['apply_reference_geometry_float_0_5']) & (mx <= GRID[1] - parameters['apply_reference_geometry_float_0_5']) & (my >= -parameters['apply_reference_geometry_float_0_5']) & (my <= GRID[0])
        if shape == NATIVE:
            exact = _native_exact_sand_field(fit_ref, reference, facies, mx, my, parameters=parameters)
            common = np.isin(facies, list(set(fit_ref.a) & set(reference.a)))
            mask = sand & common & valid_ref
            field[mask] = exact[mask]
        if light_ref is not None:
            field -= _remap_reference_tile(light_ref, h, shape, lo, hi, coordinates=coords, parameters=parameters)
        rgb = _colour(src[:, lo:hi], fit_ref.Minv, parameters=parameters)
        rgb += field[None]
        if fit_ref.F is not None:
            flat = _remap_reference_tile(fit_ref.F, h, shape, lo, hi, coordinates=coords, parameters=parameters)
            gb0 = _remap_reference_tile(fit_ref.gb0, h, shape, lo, hi, coordinates=coords, parameters=parameters)
            alpha = np.clip((GB_SAT - (rgb[1] - rgb[2])) / np.maximum(GB_SAT - gb0, parameters['apply_reference_geometry_float_0_2']), 0, 1)
            alpha = np.where(np.isfinite(alpha), alpha, 1)
            tap = np.where(sand, 1.0, taper * ~chart)
            neutral = flat[1:2]
            rgb -= (neutral + alpha[None] * (flat - neutral)) * tap[None]
        rgb[:, ~valid_ref] = np.nan
        out[:, lo:hi] = rgb
    return out

def apply_old_field_native(corrected_native_log_rgb: np.ndarray, field_ref8: np.ndarray, grid: ReferenceGrid, h_ref_to_run: np.ndarray, *, destination_origin_xy: tuple[int, int]=None, tile_rows: int=None, parameters: Mapping) -> np.ndarray:
    """Add the frozen scalar old-light residual on the same native coordinates.

    The supplied homography is inverted without normalising it in this operation.
    The mapped offset is added equally to all three channels; original bounds
    and NaN support remain invalid."""
    if destination_origin_xy is None:
        destination_origin_xy = tuple(parameters['apply_old_field_native_default_destination_origin_xy'])
    if tile_rows is None:
        tile_rows = parameters['apply_old_field_native_default_tile_rows']
    rgb = np.asarray(corrected_native_log_rgb)
    field = np.asarray(field_ref8)
    if rgb.dtype != np.float32 or field.dtype != np.float32:
        raise TypeError('Old-light native field requires float32 log-linear RAW')
    if rgb.ndim != 3 or rgb.shape[0] != 3 or field.shape != grid.shape_hw:
        raise ValueError('Native RGB or reference residual field shape mismatch')
    if tile_rows < 1:
        raise ValueError('tile_rows must be positive')
    H = np.asarray(h_ref_to_run, np.float64)
    if H.shape != (3, 3) or not np.all(np.isfinite(H)) or abs(float(np.linalg.det(H))) < parameters['apply_old_field_native_float_1eminus_10']:
        raise ValueError('Expected invertible reference-to-run native homography')
    inverse = np.linalg.inv(H)
    out = rgb.copy()
    height, width = rgb.shape[1:]
    ox, oy = destination_origin_xy
    for start in range(0, height, tile_rows):
        stop = min(start + tile_rows, height)
        yy, xx = np.mgrid[start:stop, :width]
        xrun = xx.astype(np.float64) + ox
        yrun = yy.astype(np.float64) + oy
        den = inverse[2, 0] * xrun + inverse[2, 1] * yrun + inverse[2, 2]
        if np.any(np.abs(den) < parameters['apply_old_field_native_float_1eminus_09']):
            raise ValueError('Native pixel projects to reference infinity')
        xref = (inverse[0, 0] * xrun + inverse[0, 1] * yrun + inverse[0, 2]) / den
        yref = (inverse[1, 0] * xrun + inverse[1, 1] * yrun + inverse[1, 2]) / den
        mx = ((xref - parameters['apply_old_field_native_float_3_5']) / parameters['apply_old_field_native_int_8'] - grid.origin_cell_xy[0]).astype(np.float32)
        my = ((yref - parameters['apply_old_field_native_float_3_5']) / parameters['apply_old_field_native_int_8'] - grid.origin_cell_xy[1]).astype(np.float32)
        in_field = (mx >= -parameters['apply_old_field_native_float_0_5']) & (mx <= grid.shape_hw[1] - parameters['apply_old_field_native_float_0_5']) & (my >= -parameters['apply_old_field_native_float_0_5']) & (my <= grid.shape_hw[0] - parameters['apply_old_field_native_float_0_5'])
        mapped = cv2.remap(field, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        valid = in_field & np.isfinite(mapped) & np.all(np.isfinite(rgb[:, start:stop]), axis=0)
        tile = out[:, start:stop]
        tile += mapped[None]
        tile[:, ~valid] = np.nan
    return out

def native_area_log(linear_native_rgb: np.ndarray, *, parameters: Mapping) -> np.ndarray:
    """Return the log of the exact arithmetic native-area mean.

    Explicit native and area-grid shapes define the cell grouping. The original
    mean uses float64 reduction before log and final float32 conversion."""
    NATIVE_SHAPE_CHW = tuple(parameters['native_shape_chw'])
    AREA_GRID_SHAPE_CHW = tuple(parameters['area_grid_shape_chw'])
    linear = np.asarray(linear_native_rgb)
    if linear.dtype != np.float32 or linear.shape != NATIVE_SHAPE_CHW:
        raise TypeError('Expected full-native CHW float32 camera-linear RAW')
    _, height, width = NATIVE_SHAPE_CHW
    gh, gw = AREA_GRID_SHAPE_CHW[1:]
    if gh * parameters['native_area_log_int_8'] > height or gw * parameters['native_area_log_int_8'] != width:
        raise ValueError('Native/area-grid geometry is inconsistent')
    area = np.mean(linear[:, :gh * parameters['native_area_log_int_8'], :].reshape(3, gh, parameters['native_area_log_int_8'], gw, parameters['native_area_log_int_8']), axis=(2, parameters['native_area_log_int_4']), dtype=np.float64)
    if not np.isfinite(area).all() or np.any(area <= 0):
        raise ValueError('RAW cell-area mean must be finite and positive')
    return np.log(area).astype(np.float32)

def frame_light_from_native_grid(frame_area_log8: np.ndarray, fit, grid: ReferenceGrid, h_reference_to_frame_native: np.ndarray, frame_chart_xy8: np.ndarray, frame_chart_usable: np.ndarray, *, parameters: Mapping) -> tuple[np.ndarray, dict]:
    """Fit frame light from native-area signals and explicit frame chart metadata.

    Only the fitting signal and chart coordinates enter the reference grid.
    The source-native image itself is retained on its original coordinates."""
    AREA_GRID_SHAPE_CHW = tuple(parameters['area_grid_shape_chw'])
    source_grid = np.asarray(frame_area_log8)
    if source_grid.dtype != np.float32 or source_grid.shape != AREA_GRID_SHAPE_CHW:
        raise TypeError('Expected source-bound CHW float32 area-log grid')
    h = validate_homography(h_reference_to_frame_native, parameters=parameters)
    chart_xy = np.asarray(frame_chart_xy8)
    chart_usable = np.asarray(frame_chart_usable)
    if chart_xy.shape != (parameters['frame_light_from_native_grid_int_24'], 2) or chart_usable.shape != (parameters['frame_light_from_native_grid_int_24'],):
        raise ValueError('Expected the configured per-frame chart centres and flags')
    if not np.issubdtype(chart_xy.dtype, np.floating) or chart_usable.dtype != bool:
        raise TypeError('Chart centres must be floats and usability must be boolean')
    centre = remap_to_reference(source_grid, h, parameters=parameters)
    expanded = np.full((3, *grid.shape_hw), np.nan, np.float32)
    x0, y0 = (-grid.origin_cell_xy[0], -grid.origin_cell_xy[1])
    h0, w0 = grid.original_shape_hw
    if centre.shape != (3, h0, w0) or x0 < 0 or y0 < 0 or (y0 + h0 > grid.shape_hw[0]) or (x0 + w0 > grid.shape_hw[1]):
        raise ValueError('Frozen area grid does not fit the expanded reference')
    expanded[:, y0:y0 + h0, x0:x0 + w0] = centre
    xy_ref = chart_xy_to_reference(chart_xy, h, parameters=parameters).astype(np.float32)
    xy_ref -= np.asarray(grid.origin_cell_xy, np.float32)
    return estimate_frame_light_expanded(fit, expanded, grid, frame_chart_xy_ref8=xy_ref, frame_chart_usable=chart_usable, parameters=parameters)

def correct_frame_native(linear_native_rgb: np.ndarray, fit, virtual_reference, grid: ReferenceGrid, h_reference_to_frame_native: np.ndarray, source_eligible_mask: np.ndarray, *, exact_baseline_source: bool, frame_chart_xy8: np.ndarray | None, frame_chart_usable: np.ndarray | None, old_field_ref8: np.ndarray | None, expected_area_log8: np.ndarray | None=None, tile_rows: int=None, parameters: Mapping) -> tuple[np.ndarray, dict]:
    """Correct one native linear-RAW frame using frozen baseline and reference data.

    Exact-baseline input uses its zero light field; other frames require their
    own chart samples. Area-mean verification, original sensor log threshold,
    projected colour/flat correction and optional old-light offset keep their
    original order and dtypes. Unsupported pixels remain NaN."""
    if tile_rows is None:
        tile_rows = parameters['correct_frame_native_default_tile_rows']
    NATIVE_SHAPE_CHW = tuple(parameters['native_shape_chw'])
    AREA_GRID_SHAPE_CHW = tuple(parameters['area_grid_shape_chw'])
    linear = np.asarray(linear_native_rgb)
    if linear.dtype != np.float32 or linear.shape != NATIVE_SHAPE_CHW:
        raise TypeError('Expected full-native CHW float32 camera-linear RAW')
    eligible = np.asarray(source_eligible_mask)
    if eligible.dtype != bool or eligible.shape != NATIVE_SHAPE_CHW[1:]:
        raise TypeError('Expected exact native boolean source-eligibility mask')
    h = validate_homography(h_reference_to_frame_native, parameters=parameters)
    area = native_area_log(linear, parameters=parameters)
    if expected_area_log8 is not None:
        expected = np.asarray(expected_area_log8)
        if expected.dtype != np.float32 or expected.shape != AREA_GRID_SHAPE_CHW:
            raise TypeError('Expected source area receipt must match configured CHW float32 geometry')
        if not np.allclose(area, expected, rtol=0, atol=parameters['correct_frame_native_float_1eminus_06'], equal_nan=True):
            raise ValueError('Full-native RAW does not match the source-bound area grid')
    if exact_baseline_source:
        if frame_chart_xy8 is not None or frame_chart_usable is not None:
            raise ValueError('Exact B source must use its frozen zero light field')
        light = None
        light_info = {'mode': 'exact-source-bound-B', 'zero_field': True}
    else:
        if frame_chart_xy8 is None or frame_chart_usable is None:
            raise ValueError('Non-B frame requires its source-bound ColorChecker')
        light, light_info = frame_light_from_native_grid(area, fit, grid, h, frame_chart_xy8, frame_chart_usable, parameters=parameters)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_native = np.where(linear > parameters['minimum_sensor_signal'], np.log(linear), np.nan).astype(np.float32)
    corrected = apply_native(log_native, fit, virtual_reference, grid, h, source_eligible_mask=eligible, light_field_ref8=light, tile_rows=tile_rows, parameters=parameters)
    if old_field_ref8 is not None:
        old_field = np.asarray(old_field_ref8)
        if old_field.dtype != np.float32 or old_field.shape != grid.shape_hw:
            raise TypeError('Old-light residual must match the float32 reference grid')
        corrected = apply_old_field_native(corrected, old_field, grid, h, tile_rows=tile_rows, parameters=parameters)
    return (corrected, {'contract': parameters['correct_frame_native_metadata_bfb668ea'], 'scientific_or_model_eligibility': False, 'exact_baseline_source': bool(exact_baseline_source), 'per_frame_light': light_info, 'old_light_residual_applied': old_field_ref8 is not None, 'output_encoding': 'chw-float32-log-camera-linear-rgb-with-nan-invalid', 'source_eligible_pixel_count': int(np.count_nonzero(eligible)), 'finite_eligible_pixel_count': int(np.count_nonzero(eligible & np.all(np.isfinite(corrected), axis=0)))})
