"""FF_UM historical review coordinates on an uncropped RAW photograph.

The common geometric anchor is green: the accepted relative JPG profile only
resamples red. Historical rig corrections therefore move annotations, never
the new photograph. The actual frame drift and saved curvature grid are required.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class UmReviewMapping:
    """Bijective portion of content -> drifted/resized JPG -> full RAW display."""

    curvature_grid_rc: np.ndarray
    content_shape_hw: tuple[int, int]
    drift_matrix: np.ndarray
    photo_wh: tuple[int, int]
    resized_wh: tuple[int, int]
    display_origin_jpg_xy: tuple[int, int]

    def __post_init__(self):
        h, w = self.content_shape_hw
        grid = np.asarray(self.curvature_grid_rc)
        if min(h, w) < 2 or grid.shape != (2, h * w) or not np.isfinite(grid).all():
            raise ValueError("A finite saved UM curvature grid is required")
        matrix = np.asarray(self.drift_matrix, dtype=float)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise ValueError("The actual per-frame UM drift matrix is required")
        if any(v <= 0 for v in (*self.photo_wh, *self.resized_wh)):
            raise ValueError("Actual JPG and resized dimensions are required")
        self._grid = grid[::-1].reshape(2, h, w)
        inverse = np.linalg.inv(np.vstack((matrix, [0, 0, 1])))
        scale = np.asarray(self.photo_wh) / self.resized_wh
        self._linear = scale[:, None] * inverse[:2, :2]
        self._offset = (inverse[:2, 2] + .5) * scale - .5
        self._tree = None
        self._cell_certificate = None

    def validate_all_cells(self, *, chunk_rows=64):
        """Check every bilinear cell at all four determinant extrema.

        A bilinear map's Jacobian determinant is affine within a cell, so its
        four corners bound the entire cell. This checks the actual saved grid,
        including folds that a coarse inverse seed grid could miss.
        """
        if self._cell_certificate is not None:
            return dict(self._cell_certificate)
        if chunk_rows < 1:
            raise ValueError("A positive cell-check chunk size is required")
        h, w = self.content_shape_hw
        scale = float(np.linalg.det(self._linear))
        minimum, maximum, bad = np.inf, -np.inf, 0
        for start in range(0, h-1, chunk_rows):
            stop = min(start+chunk_rows, h-1)
            a = self._grid[:, start:stop, :-1].astype(float)
            b = self._grid[:, start:stop, 1:].astype(float)-a
            c = self._grid[:, start+1:stop+1, :-1].astype(float)-a
            d = self._grid[:, start+1:stop+1, 1:].astype(float)-a-b-c
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                gx, gy = b+dy*d, c+dx*d
                determinant = (gx[0]*gy[1]-gx[1]*gy[0])*scale
                minimum = min(minimum, float(np.min(determinant)))
                maximum = max(maximum, float(np.max(determinant)))
                bad += int(np.count_nonzero(~np.isfinite(determinant) | (determinant <= 1e-8)))
        if bad:
            raise ValueError(f"The saved UM coordinate map folds or is singular at {bad} cell corners")
        self._cell_certificate = dict(cells=(h-1)*(w-1), determinant_corners=4*(h-1)*(w-1),
                                      minimum_determinant=minimum, maximum_determinant=maximum,
                                      nonpositive_or_nonfinite_corners=0)
        return dict(self._cell_certificate)

    def content_to_photo(self, xy, *, jacobian=False):
        """Bilinearly evaluate the saved grid, inverse drift and pixel-centre resize."""
        q = np.asarray(xy, dtype=float)
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError("Coordinates must be Nx2 xy pixel centres")
        h, w = self.content_shape_hw
        valid = (np.isfinite(q).all(1) & (q[:, 0] >= 0) & (q[:, 0] <= w-1)
                 & (q[:, 1] >= 0) & (q[:, 1] <= h-1))
        output = np.full(q.shape, np.nan)
        derivative = np.full((len(q), 2, 2), np.nan)
        p = q[valid]
        x0 = np.minimum(np.floor(p[:, 0]).astype(int), w-2)
        y0 = np.minimum(np.floor(p[:, 1]).astype(int), h-2)
        dx, dy = (p - np.column_stack((x0, y0))).T
        a = self._grid[:, y0, x0].T
        b = self._grid[:, y0, x0+1].T
        c = self._grid[:, y0+1, x0].T
        d = self._grid[:, y0+1, x0+1].T
        registered = ((1-dx)*(1-dy))[:, None]*a + (dx*(1-dy))[:, None]*b
        registered += ((1-dx)*dy)[:, None]*c + (dx*dy)[:, None]*d
        output[valid] = registered @ self._linear.T + self._offset
        if jacobian:
            gx = (1-dy)[:, None]*(b-a) + dy[:, None]*(d-c)
            gy = (1-dx)[:, None]*(c-a) + dx[:, None]*(d-b)
            derivative[valid] = np.einsum("ij,njk->nik", self._linear,
                                         np.stack((gx, gy), axis=-1))
            return output, derivative
        return output

    def content_to_display(self, xy, *, jacobian=False):
        value = self.content_to_photo(xy, jacobian=jacobian)
        if jacobian:
            return value[0] - self.display_origin_jpg_xy, value[1]
        return value - self.display_origin_jpg_xy

    def _initialise_inverse(self):
        self.validate_all_cells()
        h, w = self.content_shape_hw
        x = np.unique(np.r_[np.arange(0, w-1, 32), w-1])
        y = np.unique(np.r_[np.arange(0, h-1, 32), h-1])
        xx, yy = np.meshgrid(x, y)
        self._seeds = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        mapped, derivative = self.content_to_display(self._seeds, jacobian=True)
        determinant = np.linalg.det(derivative)
        if np.any(~np.isfinite(determinant)) or np.any(determinant <= 1e-8):
            raise ValueError("The saved UM coordinate map folds or is singular")
        self._tree = cKDTree(mapped)

    def display_to_content(self, xy, *, tolerance=.002, iterations=12, chunk_size=65536):
        """Invert only requested points; reuse a coarse spatial index per frame.

        Points outside the historical content return NaN. A failed inverse is
        never replaced by an assumed affine transform or a nearest label.
        """
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("Coordinates must be Nx2 xy pixel centres")
        if tolerance <= 0 or iterations < 1 or chunk_size < 1:
            raise ValueError("Positive inverse controls are required")
        if self._tree is None:
            self._initialise_inverse()
        result = np.full(xy.shape, np.nan)
        h, w = self.content_shape_hw
        for start in range(0, len(xy), chunk_size):
            target = xy[start:start+chunk_size]
            finite = np.isfinite(target).all(1)
            if not finite.any():
                continue
            target_valid = target[finite]
            _, index = self._tree.query(target_valid, workers=1)
            estimate = self._seeds[index].copy()
            for _ in range(iterations):
                mapped, jac = self.content_to_display(estimate, jacobian=True)
                residual = mapped - target_valid
                determinant = np.linalg.det(jac)
                safe = np.isfinite(determinant) & (determinant > 1e-8)
                step = np.zeros_like(estimate)
                step[safe, 0] = (jac[safe, 1, 1]*residual[safe, 0]
                                  - jac[safe, 0, 1]*residual[safe, 1])/determinant[safe]
                step[safe, 1] = (-jac[safe, 1, 0]*residual[safe, 0]
                                   + jac[safe, 0, 0]*residual[safe, 1])/determinant[safe]
                estimate -= step
                estimate[:, 0] = np.clip(estimate[:, 0], 0, w-1)
                estimate[:, 1] = np.clip(estimate[:, 1], 0, h-1)
                if safe.all() and np.max(np.linalg.norm(residual, axis=1), initial=0) <= tolerance:
                    break
            error = np.linalg.norm(self.content_to_display(estimate)-target_valid, axis=1)
            estimate[~np.isfinite(error) | (error > tolerance)] = np.nan
            result[start:start+len(target)][finite] = estimate
        return result

    def physical_jacobian(self, content_xy, *, content_spacing_yx_m):
        """Return RAW pixels per physical metre, solely from bound legacy axes.

        This diagnostic does not select a feature spacing or resample an image.
        """
        spacing = np.asarray(content_spacing_yx_m, dtype=float)
        if spacing.shape != (2,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("Bound historical physical axes are required")
        _, jac = self.content_to_display(content_xy, jacobian=True)
        return jac / spacing[::-1][None, None, :]


def annotation_content_outline(payload, *, canvas_wh, content_box_xywh, step=4., clip=False):
    """Densify original polygon/box edges in content coordinates; retain its ID separately."""
    if step <= 0:
        raise ValueError("A positive outline sampling distance is required")
    points = payload.get("points") or []
    if not points:
        x, y, w, h = (float(payload[k]) for k in ("x", "y", "width", "height"))
        points = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
    polygon = np.asarray(points, float)*np.asarray(canvas_wh) - content_box_xywh[:2]
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
        raise ValueError("A polygon or rectangle is required")
    if clip:
        # Clip derived drawing coordinates only; the original raster annotation
        # and its record remain the authority for training membership.
        for axis, edge, direction in ((0, 0., 1), (1, 0., 1),
                                     (0, content_box_xywh[2]-1., -1),
                                     (1, content_box_xywh[3]-1., -1)):
            clipped = []
            for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
                inside_a = direction*(a[axis]-edge) >= 0
                inside_b = direction*(b[axis]-edge) >= 0
                if inside_a:
                    clipped.append(a)
                if inside_a != inside_b:
                    clipped.append(a+(b-a)*(edge-a[axis])/(b[axis]-a[axis]))
            polygon = np.asarray(clipped, dtype=float).reshape(-1, 2)
            if not len(polygon):
                return polygon
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        n = max(1, int(np.ceil(np.linalg.norm(b-a)/step)))
        result.append(a+(b-a)*np.arange(n)[:, None]/n)
    return np.concatenate(result)
