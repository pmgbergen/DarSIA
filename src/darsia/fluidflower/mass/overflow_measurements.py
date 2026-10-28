"""Approved overflow feed calculation, independent of any historical gas map.

The original feed and QA functions are retained literally. The caller supplies
checked scale-review documents and current P1 gas volumes; historical model
volumes and rounded historical ratios are never used to calibrate a new run.
"""
from __future__ import annotations

import math
import statistics

import numpy as np


CORRECTIONS = {('ac33', '2023-02-17T17:45:29'): 1584.7, ('ac35', '2023-02-24T14:12:13'): 366.5}
MIN_V = 3.0  # Original minimum model-gas volume in mL for a ratio.
FEED_AST_SHA256 = "39c1eebee25c84a057ca31c3a38316d141ac7d8d970084d39522ce98aa084171"


def feed(rv):
    t, w = rv[:, 0], rv[:, 1]
    t_end = t[-1]
    for i in range(len(t)):
        if t[i] < 4:
            continue
        later = (t > t[i]) & (t <= t[i] + 0.5)
        if later.any() and (w[later].max() - w[i]) < 10:
            t_end = t[i]
            break
    win = (t >= t_end - 4) & (t <= t_end - 0.3)
    q, a = np.polyfit(t[win], w[win], 1)
    vA = w - (a + q * t)
    pre = t < 0
    out = dict(t=t, w=w, vA=vA, t_end=float(t_end), q_late=float(q), late_sd=float(np.std(vA[win])), n=int(len(t)),
               n_pre=int(pre.sum()), t_first=float(t[0]), t_last=float(t[-1]), n_win=int(win.sum()))
    out['vA_pre'] = float(vA[pre].mean()) if pre.any() else None
    if pre.sum() >= 2 and (t[pre].max() - t[pre].min()) >= 0.05:
        out['q_pre'] = float(np.polyfit(t[pre], w[pre], 1)[0])
    if pre.any():
        # Primary C: the original last-pre-injection anchor and late feed fit.
        i0 = int(np.where(pre)[0][-1]); t0, w0 = t[i0], w[i0]
        X = np.c_[t[win] - t0, 0.5 * (t[win] - t0) ** 2]
        (q0, k), *_ = np.linalg.lstsq(X, w[win] - w0, rcond=None)
        out['vC'] = w - (w0 + q0 * (t - t0) + 0.5 * k * (t - t0) ** 2)
        out['q_C_at_t0'], out['drift_C'], out['t0_anchor'] = float(q0), float(k), float(t0)
        out['vC_late_sd'] = float(np.std(out['vC'][win]))
        out['pre_step_mL'] = float(w[pre][-1] - w[pre][0] - q * (t[pre][-1] - t[pre][0])) if pre.sum() >= 2 else None
        both = win | pre
        wts = np.where(pre, 5.0, 1.0)[both]
        c2, c1, c0 = np.polyfit(t[both], w[both], 2, w=wts)
        out['vB'] = w - (c0 + c1 * t + c2 * t * t)
        out['drift_g_per_h2'] = float(2 * c2)
        out['q_B_at0'] = float(c1)
    return out


def qa(r):
    why = []
    if r['t_end'] < 10:
        why.append(f"supply stops at {r['t_end']} h")
    if r['late_sd'] > 1.5:
        why.append(f"late SD {r['late_sd']} g")
    if abs(r['drift']) > 3:
        why.append(f"drift {r['drift']} g/h2")
    if r['vA_pre'] is None or abs(r['vA_pre']) > 50:
        why.append(f"late line misses pre-injection by {r['vA_pre']} mL")
    if r['q_pre'] is not None and abs(r['q_pre']) >= 600:
        why.append(f"pre-injection readings inconsistent (slope {r['q_pre']} g/h)")
    return why


def reviewed_curve(document, *, run):
    """Read the original reviewed rows/corrections and preserve its QA rounding.

    This parses supplied data only. Path, SHA, original review identity and
    recorded authority are the immutable producer/reader's responsibility.
    """
    if type(document) is not dict or type(document.get("records")) is not list or type(document.get("metadata")) is not dict:
        raise ValueError("The original scale-review document is required")
    metadata = document['metadata']
    if metadata.get('weight_data_valid') is False:
        return dict(usable=False, reasons=["The recorded weight data are invalid."], curve=None)
    values = []
    for row in document['records']:
        if type(row) is not dict:
            raise ValueError("Each original reviewed weight row must be a table")
        if row.get('status') not in ('Verified', 'Corrected') or row.get('weight_g') is None:
            continue
        time = row.get('time_from_injection_h')
        weight = CORRECTIONS.get((run, row.get('image_time')), row['weight_g'])
        if (type(time) not in (int, float) or type(weight) not in (int, float)
                or not math.isfinite(time) or not math.isfinite(weight)):
            raise ValueError("Reviewed time and scale values must be finite original numbers")
        values.append((float(time), float(weight)))
    values.sort()
    if len(values) < 30:
        return dict(usable=False, reasons=["Fewer than 30 reviewed scale readings."], curve=None)
    times = [value[0] for value in values]
    if len(set(times)) != len(times):
        raise ValueError("Duplicate reviewed scale times cannot receive repeated measurement weight")
    if not any(time < 0 for time in times):
        return dict(usable=False, reasons=["No pre-injection reading for the recorded feed anchor."], curve=None)
    curve = feed(np.array(values, np.float64))
    if any(not np.isfinite(curve[key]).all() for key in ('t', 'w', 'vA', 'vB', 'vC')):
        raise ValueError("The original feed calculation did not produce finite curves")
    record = dict(t_end=round(curve['t_end'], 2), late_sd=round(curve['late_sd'], 2),
                  drift=round(curve['drift_g_per_h2'], 3),
                  vA_pre=None if curve['vA_pre'] is None else round(curve['vA_pre'], 2),
                  q_pre=round(curve['q_pre'], 2) if 'q_pre' in curve else None)
    reasons = qa(record)
    return dict(usable=not reasons, reasons=reasons, curve=curve, qa=record)


def measured_ratio(curve, frames):
    """Median 1–4 h overflow/current-B1 ratio and its measurement uncertainty.

    The SD is the median ratio uncertainty from the original per-frame
    (feed-model spread/2 + late residual SD). It is not divided by sqrt(n):
    the fitted feed/anchor uncertainty is shared between a run's frames.
    """
    if not isinstance(curve, dict) or not frames:
        raise ValueError("The original usable feed curve and current selected P1 frames are required")
    ratios, errors, selected = [], [], []
    for row in frames:
        if (type(row) is not dict or not {'case_id', 'elapsed_h', 'baseline_gas_volume_m3'} <= row.keys()
                or not isinstance(row['case_id'], str) or not row['case_id']):
            raise ValueError("Explicit current frame identity, time and gas volume required")
        time, model = row['elapsed_h'], row['baseline_gas_volume_m3']
        if (type(time) not in (int, float) or type(model) not in (int, float)
                or not math.isfinite(time) or not math.isfinite(model) or model < 0):
            raise ValueError("Current frame time and physical gas volume must be finite")
        if not 1.0 <= time <= 4.0:
            continue
        if not curve['t'][0] <= time <= curve['t'][-1]:
            raise ValueError("The selected ratio frame lies outside the reviewed measurement interval")
        volume_ml = model * 1e6
        if volume_ml < MIN_V:
            continue
        a = float(np.interp(time, curve['t'], curve['vA']))
        b = float(np.interp(time, curve['t'], curve['vB']))
        c = float(np.interp(time, curve['t'], curve['vC']))
        error = (max(a, b, c)-min(a, b, c))/2 + curve['vC_late_sd']
        ratios.append(c / volume_ml)
        errors.append(error / volume_ml)
        selected.append(row['case_id'])
    if len(selected) != len(set(selected)):
        raise ValueError("A selected current frame cannot enter the overflow ratio twice")
    if not selected:
        return None
    ratio, sigma = float(statistics.median(ratios)), float(statistics.median(errors))
    if not math.isfinite(ratio) or ratio < 0 or not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("The measured run ratio needs finite nonnegative gas and positive uncertainty")
    return dict(ratio=ratio, sigma=sigma, selected_cases=selected)
