"""Internal feature grids for uncropped FF_UM/FF_BM RAW photographs.

Images retain their full RAW canvas. Current and baseline are sampled separately
with their measured frame maps onto the same historical physical coordinates.
Units and photometry are supplied by the caller and are never fitted here.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SampledGrid:
    values: np.ndarray
    valid_support: np.ndarray


def _shape(value):
    if (len(value) != 2 or any(isinstance(v, (bool, np.bool_))
            or not isinstance(v, (int, np.integer)) or v <= 0 for v in value)):
        raise ValueError("Grid dimensions must be two positive integers")
    return tuple(map(int, value))


def sample_full_frame(values, valid_support, content_to_display,
                      content_shape_hw, *, chunk_rows=128, interpolation="linear"):
    """Sample HWC RGB or HW fields at exact mapped pixel centres.

    The float32 result keeps the input units. Unsupported samples are NaN and
    must be excluded using the returned mask. A contributing invalid neighbour
    invalidates a sample; a zero-weight neighbour does not. No crop or mutation
    of the source photograph is performed. Use ``nearest`` for categorical
    labels, which must never be mixed by bilinear interpolation.
    """
    source = np.asarray(values)
    support = np.asarray(valid_support)
    shape = _shape(content_shape_hw)
    if (source.ndim not in (2, 3) or min(source.shape) < 1
            or not np.issubdtype(source.dtype, np.number)
            or np.issubdtype(source.dtype, np.complexfloating)):
        raise ValueError("Real HW scores or HWC channels are required")
    if support.dtype != bool or support.shape != source.shape[:2]:
        raise ValueError("Source support must be a matching boolean HW mask")
    if (not callable(content_to_display) or isinstance(chunk_rows, bool)
            or not isinstance(chunk_rows, (int, np.integer)) or chunk_rows < 1):
        raise ValueError("An explicit frame map and positive chunk size are required")
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("Sampling interpolation must be linear or nearest")
    out = np.full((*shape, *source.shape[2:]), np.nan, dtype=np.float32)
    out_support = np.zeros(shape, bool)
    source_h, source_w = source.shape[:2]
    for start in range(0, shape[0], chunk_rows):
        stop = min(start + chunk_rows, shape[0])
        yy, xx = np.mgrid[start:stop, :shape[1]]
        points = np.asarray(content_to_display(
            np.column_stack((xx.ravel(), yy.ravel()))), dtype=float)
        if points.shape != (xx.size, 2):
            raise ValueError("The frame map must return one xy pair per pixel")
        inside = (np.isfinite(points).all(axis=1) & (points[:, 0] >= 0)
                  & (points[:, 0] <= source_w - 1) & (points[:, 1] >= 0)
                  & (points[:, 1] <= source_h - 1))
        selected = np.flatnonzero(inside)
        q = points[selected]
        if interpolation == "nearest":
            x, y = np.floor(q + .5).astype(np.int64).T
            samples = source[y, x]
            finite = np.isfinite(samples)
            if source.ndim == 3:
                finite = finite.all(axis=1)
            usable = support[y, x] & finite
            out[start:stop].reshape((xx.size, *source.shape[2:]))[selected[usable]] = samples[usable]
            out_support[start:stop].reshape(-1)[selected[usable]] = True
            continue
        x0, y0 = np.floor(q).astype(np.int64).T
        x1, y1 = np.minimum(x0 + 1, source_w - 1), np.minimum(y0 + 1, source_h - 1)
        dx, dy = (q - np.column_stack((x0, y0))).T
        samples = np.zeros((len(q), *source.shape[2:]), np.float32)
        usable = np.ones(len(q), bool)
        for x, y, weight in ((x0, y0, (1-dx)*(1-dy)), (x1, y0, dx*(1-dy)),
                             (x0, y1, (1-dx)*dy), (x1, y1, dx*dy)):
            contributing = weight > 0
            v = source[y[contributing], x[contributing]]
            finite = np.isfinite(v)
            if source.ndim == 3:
                finite = finite.all(axis=1)
            usable[contributing] &= support[y[contributing], x[contributing]] & finite
            factor = weight[contributing].astype(np.float32)
            if source.ndim == 3:
                factor = factor[:, None]
            # Unsupported values never become black evidence or 0*NaN terms.
            samples[contributing] += np.where(np.isfinite(v), v, 0) * factor
        tile = out[start:stop].reshape((xx.size, *source.shape[2:]))
        tile[selected[usable]] = samples[usable]
        out_support[start:stop].reshape(-1)[selected[usable]] = True
    return SampledGrid(out, out_support)


def project_to_full_frame(values, valid_support, display_to_content,
                          content_shape_hw, raw_support, *, chunk_rows=128,
                          interpolation="linear"):
    """Return native or reduced physical-grid fields on the complete RAW canvas.

    ``content_shape_hw`` describes the historical native coordinate map. A
    reduced feature grid uses the same pixel-centre scaling as cv2.resize.
    Its boundary cells extend to the native domain edge; the old calibrated
    domain and both support masks still constrain every output sample.
    Categorical phase labels require ``interpolation="nearest"``; continuous
    probabilities use the default bilinear interpolation.
    """
    content_shape = _shape(content_shape_hw)
    target_support = np.asarray(raw_support)
    if target_support.ndim != 2 or target_support.dtype != bool:
        raise ValueError("Full RAW support must be a boolean HW mask")
    source_shape = np.asarray(values).shape[:2]
    _shape(source_shape)
    scale = np.asarray(source_shape[::-1], float) / content_shape[::-1]

    def target_to_source(points):
        content = np.asarray(display_to_content(points), dtype=float)
        if content.shape != points.shape:
            raise ValueError("The inverse frame map must return one xy pair per pixel")
        inside = (np.isfinite(content).all(axis=1) & (content[:, 0] >= 0)
                  & (content[:, 0] <= content_shape[1]-1) & (content[:, 1] >= 0)
                  & (content[:, 1] <= content_shape[0]-1))
        mapped = np.full(content.shape, np.nan)
        mapped[inside] = np.clip((content[inside]+.5)*scale-.5, 0,
                                  np.asarray(source_shape[::-1])-1)
        return mapped

    result = sample_full_frame(values, valid_support, target_to_source,
                               target_support.shape, chunk_rows=chunk_rows,
                               interpolation=interpolation)
    result.valid_support[:] &= target_support
    result.values[~result.valid_support] = np.nan
    return result
