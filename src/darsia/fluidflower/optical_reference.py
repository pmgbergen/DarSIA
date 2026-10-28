"""Measured optical-reference proposals on one prepared physical image grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def frame_facing_sand_boundary(sand_domain, *, other_material=None):
    """Find sand facing the outside of the map, not water or enclosed holes.

    ``other_material`` marks explicitly mapped non-sand material (for example
    free water or a ColorChecker), which must not be treated as a frame gap.
    Unpainted gaps connected to the edge remain outside the sand map.
    """
    sand = np.asarray(sand_domain)
    if sand.ndim != 2 or sand.dtype != bool:
        raise ValueError("The mapped sand domain must be a boolean image")
    other = np.zeros(sand.shape, bool) if other_material is None else np.asarray(other_material)
    if other.dtype != bool or other.shape != sand.shape:
        raise ValueError("Other mapped material must use the same boolean grid")
    if np.any(sand & other):
        raise ValueError("Sand and other material cannot overlap")
    # Padding includes the part of the measured frame outside a cropped grid.
    space = np.pad(~(sand | other), 1, constant_values=True)
    seeds = np.zeros(space.shape, bool)
    seeds[[0, -1], :] = True
    seeds[:, [0, -1]] = True
    outside = ndimage.binary_propagation(seeds, mask=space)
    adjacent = ndimage.binary_dilation(outside)[1:-1, 1:-1]
    return sand & adjacent


def stable_band_mask(sand_domain, width_mm, pixel_size_mm, *, frame_facing_boundary=None):
    """Measure inward from the external sand boundary, ignoring internal holes.

    The caller may supply the boundary segments adjacent to the frame. This
    avoids treating a water surface or an internal excluded object as a frame
    edge. Neither phase predictions nor intensity values define this mask.
    """
    domain = np.asarray(sand_domain)
    spacing = np.asarray(pixel_size_mm, dtype=float)
    if domain.ndim != 2 or domain.dtype != bool:
        raise ValueError("The stable area needs a boolean sand-domain mask")
    if (spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or isinstance(width_mm, bool) or not np.isfinite(width_mm) or width_mm <= 0):
        raise ValueError("Stable-area width and pixel sizes must be positive millimetres")
    outer = ndimage.binary_fill_holes(domain)
    boundary = outer & ~ndimage.binary_erosion(outer, border_value=0)
    if frame_facing_boundary is not None:
        selected = np.asarray(frame_facing_boundary)
        if selected.dtype != bool or selected.shape != domain.shape:
            raise ValueError("Frame-facing boundary must use the same boolean image grid")
        boundary &= selected
    if not boundary.any():
        return np.zeros(domain.shape, bool)
    # Distance is between sample centres on the measured (possibly anisotropic)
    # grid, rather than an arbitrary number of pixels in the displayed image.
    distance = ndimage.distance_transform_edt(~boundary, sampling=spacing)
    return domain & (distance <= float(width_mm))


def polynomial_terms(x, y, degree):
    """Columns x**ax * y**(d - ax) for d = 1..degree, in the light-field producers' order."""
    return np.stack([np.asarray(x, float)**ax * np.asarray(y, float)**(d - ax)
                     for d in range(1, int(degree) + 1) for ax in range(d + 1)], axis=-1)


def facies_light_field(log_green, facies, *, degree, block=4, minimum_blocks=30, iterations=8):
    """Fit one smooth light field shared by all sand, with one level per facies.

    ``log_green`` is log linear green on a working grid (NaN where unusable) and
    ``facies`` its facies id per cell, negative outside the usable sand. Sand of
    one facies is the same sand everywhere, so after its own level the
    remaining smooth variation is light. Blocks of ``block`` x ``block`` cells
    lying wholly in one facies are reduced to their median, which removes grain
    texture; the fit is least squares with Huber weights. Coordinates are those
    of the light-field producers: x = (column - width / 2) / (width / 2) and
    likewise y. Returns the field's polynomial coefficients without constant
    (in ``polynomial_terms`` order), the level of each facies, and counts.
    """
    log_green = np.asarray(log_green, float)
    facies = np.asarray(facies)
    if log_green.ndim != 2 or facies.shape != log_green.shape or int(degree) < 1 or int(block) < 1:
        raise ValueError("The light field needs one facies per working-grid cell and a positive degree")
    gh, gw = log_green.shape
    h, w = (gh // block) * block, (gw // block) * block
    blocks = lambda value: value[:h, :w].reshape(h // block, block, w // block, block).swapaxes(1, 2).reshape(
        h // block, w // block, block * block)
    ids, values = blocks(facies), blocks(log_green)
    whole = (ids >= 0).all(axis=2) & (ids == ids[..., :1]).all(axis=2) & np.isfinite(values).all(axis=2)
    rows, cols = np.nonzero(whole)
    sample_facies = ids[rows, cols, 0]
    target = np.median(values[rows, cols], axis=1)
    present = [int(value) for value in np.unique(sample_facies)
               if np.count_nonzero(sample_facies == value) >= minimum_blocks]
    keep = np.isin(sample_facies, present)
    terms = polynomial_terms(((cols + .5) * block - .5 - gw / 2) / (gw / 2),
                             ((rows + .5) * block - .5 - gh / 2) / (gh / 2), degree)[keep]
    if not present or np.count_nonzero(keep) < 10 * (len(present) + terms.shape[1]):
        raise ValueError("Too little same-facies sand outside the excluded band to fit a light field")
    levels = np.stack([sample_facies[keep] == value for value in present], axis=-1).astype(float)
    design, target = np.concatenate([levels, terms], axis=1), target[keep]
    weights = np.ones(len(target))
    for _ in range(int(iterations)):
        root = np.sqrt(weights)
        solution = np.linalg.lstsq(design * root[:, None], target * root, rcond=None)[0]
        residual = target - design @ solution
        scale = 1.4826 * float(np.median(np.abs(residual - np.median(residual))))
        limit = 1.345 * max(scale, 1e-9)
        weights = np.where(np.abs(residual) <= limit, 1.0, limit / np.maximum(np.abs(residual), 1e-300))
    coefficients = solution[len(present):]
    remaining = target - levels @ solution[:len(present)]
    # Same-facies spread of log green before and after removing the field.
    spread = {value: [float(np.subtract(*np.percentile(part, [75, 25])))
                      for part in (remaining[mine], (remaining - terms @ coefficients)[mine])]
              for value, mine in ((value, sample_facies[keep] == value) for value in present)}
    return dict(coefficients=coefficients, facies_levels={value: float(level) for value, level in
                zip(present, solution[:len(present)])}, sample_count=int(len(target)),
                block_cells=int(block), degree=int(degree),
                interquartile_log_spread={value: dict(before=before, after=after)
                                          for value, (before, after) in spread.items()})


@dataclass(frozen=True)
class ReferenceSamples:
    run: str
    stable_rgb: np.ndarray
    stable_valid: np.ndarray
    chart_rgb: np.ndarray | None = None


def reference_samples(run, rgb, stable_mask, *, chart_rgb=None, grid_shape=(12, 12)):
    """Reduce only the selected stable area to equally weighted spatial cells.

    RGB is linear and in the same exposure units for every run. Black,
    saturated and nonfinite samples are not usable optical observations.
    Spatial cells retain their indices so missing pixels cannot shift which
    parts of two runs are compared.
    """
    image = np.asarray(rgb)
    mask = np.asarray(stable_mask)
    if (not isinstance(run, str) or not run or image.ndim != 3 or image.shape[-1] != 3
            or mask.dtype != bool or mask.shape != image.shape[:2]
            or len(grid_shape) != 2 or any(type(n) is not int or n < 1 for n in grid_shape)):
        raise ValueError("Reference samples require one named RGB image and its stable mask")
    usable = mask & np.isfinite(image).all(axis=2) & (image > 0).all(axis=2) & (image < 1).all(axis=2)
    cells = np.full((grid_shape[0] * grid_shape[1], 3), np.nan, dtype=np.float64)
    valid = np.zeros(len(cells), bool)
    yy = np.linspace(0, image.shape[0], grid_shape[0] + 1, dtype=int)
    xx = np.linspace(0, image.shape[1], grid_shape[1] + 1, dtype=int)
    for iy in range(grid_shape[0]):
        for ix in range(grid_shape[1]):
            sl = np.s_[yy[iy]:yy[iy + 1], xx[ix]:xx[ix + 1]]
            selected = image[sl][usable[sl]]
            index = iy * grid_shape[1] + ix
            if len(selected):
                cells[index] = np.median(selected, axis=0)
                valid[index] = True
    chart = None if chart_rgb is None else np.asarray(chart_rgb, dtype=np.float64).reshape(-1, 3)
    return ReferenceSamples(run, cells, valid, chart)


def propose_reference(samples, *, use_lighting=True, use_color=True,
                      minimum_stable_cells=3, minimum_chart_patches=6):
    """Choose the medoid: the run with least mean optical correction to all runs.

    The pairwise correction is the mean of its parts: the sand cells (with
    ``use_lighting``) and the ColorChecker (with ``use_color``). Each ranking
    row also gives the mean over the other runs of each part used, as
    ``mean_light_correction`` and ``mean_colour_correction``.

    This is a proposal only. It never changes the saved reference. No phase
    label, chemical recipe, model score or historic campaign ranking is read.
    """
    samples = tuple(samples)
    if (not samples or len({row.run for row in samples}) != len(samples)
            or not (use_lighting or use_color)):
        raise ValueError("Choose distinct optical inputs and at least one correction")
    for row in samples:
        if use_lighting and np.count_nonzero(row.stable_valid) < minimum_stable_cells:
            raise ValueError("Too few stable areas for " + row.run)
        if use_color:
            chart = row.chart_rgb
            if chart is None:
                raise ValueError("Matching ColorChecker observations are required for every run")
            valid = np.isfinite(chart).all(axis=1) & (chart > 0).all(axis=1) & (chart < 1).all(axis=1)
            if np.count_nonzero(valid) < minimum_chart_patches:
                raise ValueError("Too few usable ColorChecker patches for " + row.run)
    costs = np.zeros((len(samples), len(samples)), float)
    light = np.zeros_like(costs) if use_lighting else None
    colour = np.zeros_like(costs) if use_color else None
    for i, left in enumerate(samples):
        for j in range(i + 1, len(samples)):
            right = samples[j]
            parts = []
            if use_lighting:
                if left.stable_rgb.shape != right.stable_rgb.shape:
                    raise ValueError("Stable areas must use the same prepared physical grid")
                common = left.stable_valid & right.stable_valid
                if np.count_nonzero(common) < minimum_stable_cells:
                    raise ValueError("Too few shared stable areas for " + left.run + " / " + right.run)
                delta = np.log(left.stable_rgb[common]) - np.log(right.stable_rgb[common])
                parts.append(float(np.median(np.abs(delta.mean(axis=1)))))
                light[i, j] = light[j, i] = parts[-1]
            if use_color:
                if (left.chart_rgb is None or right.chart_rgb is None
                        or left.chart_rgb.shape != right.chart_rgb.shape):
                    raise ValueError("Matching ColorChecker observations are required for every run")
                a, b = left.chart_rgb, right.chart_rgb
                usable = (np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
                          & (a > 0).all(axis=1) & (b > 0).all(axis=1)
                          & (a < 1).all(axis=1) & (b < 1).all(axis=1))
                if np.count_nonzero(usable) < minimum_chart_patches:
                    raise ValueError("Too few shared ColorChecker patches for " + left.run + " / " + right.run)
                delta = np.log(a[usable]) - np.log(b[usable])
                # Include both exposure and chromatic corrections. A grey chart
                # shift and a colour cast must not both collapse to zero.
                parts.append(float(np.median(np.linalg.norm(delta, axis=1) / np.sqrt(3))))
                colour[i, j] = colour[j, i] = parts[-1]
            costs[i, j] = costs[j, i] = float(np.mean(parts))
    denominator = max(1, len(samples) - 1)
    rows = []
    for i, row in enumerate(samples):
        rows.append(dict(run=row.run, mean_log_correction=float(costs[i].sum() / denominator),
                         maximum_log_correction=float(costs[i].max())))
        if light is not None:
            rows[-1]["mean_light_correction"] = float(light[i].sum() / denominator)
        if colour is not None:
            rows[-1]["mean_colour_correction"] = float(colour[i].sum() / denominator)
    ranking = sorted(rows, key=lambda row: (row["mean_log_correction"], row["maximum_log_correction"], row["run"]))
    return dict(format="darsia-optical-reference-proposal-v1", reference_run=ranking[0]["run"],
                method="minimum-mean-pairwise-optical-correction", ranking=ranking,
                uses_lighting=bool(use_lighting), uses_color=bool(use_color), selected=False)
