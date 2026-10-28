"""FF_BM annotation coordinates on the uncropped Sony RAW display.

The old rig correction is inverted for annotations only. It is never applied to
the RAW image. The TIFF registration is an explicit measured input, separate
from the Sony decoder's optical crop offset.
"""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


class CurvatureInverse:
    """Invert the saved bilinear old-content -> registered-TIFF coordinate map."""

    def __init__(self, cache, *, seed_stride=32, residual_tolerance=1e-4):
        self.shape = tuple(cache["shape"])
        if len(self.shape) != 2 or min(self.shape) < 2:
            raise ValueError("A two-dimensional saved curvature grid is required")
        self.grid = np.asarray(cache["grid"])
        if self.grid.shape != (2, int(np.prod(self.shape))):
            raise ValueError("Curvature grid and content dimensions differ")
        if seed_stride < 1 or residual_tolerance <= 0:
            raise ValueError("Positive inversion stride and residual tolerance required")
        self.tolerance = float(residual_tolerance)
        y = np.unique(np.r_[np.arange(0, self.shape[0], seed_stride), self.shape[0]-1])
        x = np.unique(np.r_[np.arange(0, self.shape[1], seed_stride), self.shape[1]-1])
        yy, xx = np.meshgrid(y, x, indexing="ij")
        self.seeds = np.column_stack((xx.ravel(), yy.ravel())).astype(float)
        values, jacobian = self.forward(self.seeds, derivative=True)
        usable = np.isfinite(values).all(axis=1) & (np.linalg.det(jacobian) > 1e-8)
        if not usable.any():
            raise ValueError("Saved curvature grid has no invertible support")
        self.seeds = self.seeds[usable]
        self.tree = cKDTree(values[usable])

    def forward(self, points, *, derivative=False):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError("Coordinates must be Nx2 xy points")
        h, w = self.shape
        good = (np.isfinite(p).all(axis=1) & (p[:, 0] >= 0) & (p[:, 0] <= w-1)
                & (p[:, 1] >= 0) & (p[:, 1] <= h-1))
        result = np.full(p.shape, np.nan)
        jac = np.full((len(p), 2, 2), np.nan)
        q = p[good]
        x0 = np.minimum(np.floor(q[:, 0]).astype(int), w-2)
        y0 = np.minimum(np.floor(q[:, 1]).astype(int), h-2)
        dx, dy = (q - np.column_stack((x0, y0))).T
        a = self.grid[::-1, y0*w+x0].T
        b = self.grid[::-1, y0*w+x0+1].T
        c = self.grid[::-1, (y0+1)*w+x0].T
        d = self.grid[::-1, (y0+1)*w+x0+1].T
        result[good] = ((1-dx)*(1-dy))[:, None]*a + (dx*(1-dy))[:, None]*b + ((1-dx)*dy)[:, None]*c + (dx*dy)[:, None]*d
        jac[good, :, 0] = (1-dy)[:, None]*(b-a) + dy[:, None]*(d-c)
        jac[good, :, 1] = (1-dx)[:, None]*(c-a) + dx[:, None]*(d-b)
        return (result, jac) if derivative else result

    def __call__(self, registered_xy):
        target = np.asarray(registered_xy, dtype=float)
        if target.ndim != 2 or target.shape[1] != 2:
            raise ValueError("Coordinates must be Nx2 xy points")
        output = np.full(target.shape, np.nan)
        valid = np.isfinite(target).all(axis=1)
        if not valid.any():
            return output
        requested = target[valid]
        _, nearest = self.tree.query(requested, workers=1)
        current = self.seeds[nearest].copy()
        live = np.ones(len(current), bool)
        for _ in range(12):
            value, jac = self.forward(current, derivative=True)
            residual = value-requested
            determinant = jac[:, 0, 0]*jac[:, 1, 1] - jac[:, 0, 1]*jac[:, 1, 0]
            live &= np.isfinite(residual).all(axis=1) & (determinant > 1e-8)
            active = live & (np.max(np.abs(residual), axis=1) > self.tolerance)
            if not active.any():
                break
            j, r, det = jac[active], residual[active], determinant[active]
            step = np.column_stack((j[:, 1, 1]*r[:, 0]-j[:, 0, 1]*r[:, 1],
                                    j[:, 0, 0]*r[:, 1]-j[:, 1, 0]*r[:, 0]))/det[:, None]
            current[active] -= step
        value, jac = self.forward(current, derivative=True)
        determinant = jac[:, 0, 0]*jac[:, 1, 1]-jac[:, 0, 1]*jac[:, 1, 0]
        accepted = (live & np.isfinite(value).all(axis=1)
                    & (determinant > 1e-8)
                    & (np.max(np.abs(value-requested), axis=1) <= self.tolerance))
        current[~accepted] = np.nan
        output[valid] = current
        return output

    def validate_rectangular_support(self, *, chunk_rows=64):
        """Verify positive principal minors everywhere on the bilinear grid.

        On this rectangular domain the P-matrix condition also rules out two
        distinct content points mapping to the same TIFF point. All four cell
        corners are checked, not only the inversion seeds.
        """
        h, w = self.shape
        grid = self.grid[::-1].reshape(2, h, w)
        minima = np.full(3, np.inf)
        for y in range(0, h-1, chunk_rows):
            end = min(h-1, y+chunk_rows)
            a, b = grid[:, y:end, :-1], grid[:, y:end, 1:]
            c, d = grid[:, y+1:end+1, :-1], grid[:, y+1:end+1, 1:]
            for dx in (b-a, d-c):
                for dy in (c-a, d-b):
                    principal = (dx[0], dy[1], dx[0]*dy[1]-dx[1]*dy[0])
                    if any(not np.isfinite(p).all() or np.any(p <= 1e-8) for p in principal):
                        raise ValueError("Saved BM curvature grid has folded, singular or unverified support")
                    minima = np.minimum(minima, [float(p.min()) for p in principal])
        return dict(min_dx_dx=float(minima[0]), min_dy_dy=float(minima[1]),
                    min_determinant=float(minima[2]), cells=(h-1)*(w-1))


@dataclass(frozen=True)
class BMReviewMapping:
    """Complete new-display -> old-review chain, including measured frame drift."""
    raw_mapping: object
    curvature: CurvatureInverse
    drift_matrix: np.ndarray
    tiff_to_sensor_xy: tuple[float, float]
    photo_wh: tuple[int, int]
    resized_wh: tuple[int, int]

    def __post_init__(self):
        matrix = np.asarray(self.drift_matrix, dtype=float)
        offset = np.asarray(self.tiff_to_sensor_xy, dtype=float)
        if (matrix.shape != (2, 3) or not np.isfinite(matrix).all()
                or np.linalg.det(matrix[:, :2]) <= 0
                or offset.shape != (2,) or not np.isfinite(offset).all()):
            raise ValueError("An explicit finite TIFF registration and actual frame drift are required")
        object.__setattr__(self, "drift_matrix", matrix.copy())
        for dimensions in (self.photo_wh, self.resized_wh):
            if len(dimensions) != 2 or any(type(v) is not int or v <= 0 for v in dimensions):
                raise ValueError("Explicit original and historical resized photograph dimensions required")

    def __call__(self, display_xy):
        u = self.raw_mapping.display_to_camera_u_xy(display_xy)
        sensor = self.raw_mapping.camera_u_to_sensor_xy(u)
        tiff = sensor-np.asarray(self.tiff_to_sensor_xy)
        resized = (tiff+.5)*(np.asarray(self.resized_wh)/self.photo_wh)-.5
        registered = resized @ self.drift_matrix[:, :2].T + self.drift_matrix[:, 2]
        return self.curvature(registered)

    def content_to_display(self, content_xy):
        registered = self.curvature.forward(content_xy)
        inverse = np.linalg.inv(np.vstack((self.drift_matrix, [0, 0, 1])))
        resized = registered @ inverse[:2, :2].T + inverse[:2, 2]
        tiff = (resized+.5)*(np.asarray(self.photo_wh)/self.resized_wh)-.5
        sensor = tiff+np.asarray(self.tiff_to_sensor_xy)
        return self.raw_mapping.camera_u_to_display_xy(
            self.raw_mapping.sensor_to_camera_u_xy(sensor))

    def projected_metric(self, content_xy, *, old_spacing_m):
        """Singular physical scales on RAW pixels; never claim a constant metric.

        The physical metric is inherited from the old calibrated rig coordinates.
        It is meaningful only inside that old domain, not the new outer canvas.
        """
        points = np.asarray(content_xy, float)
        spacing = np.asarray(old_spacing_m, float)
        if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Explicit old-content y/x physical spacing required")
        h = .125
        columns = [(self.content_to_display(points+v)-self.content_to_display(points-v))/(2*h)
                   for v in (np.array([h, 0]), np.array([0, h]))]
        jac = np.stack(columns, axis=-1)
        valid = np.isfinite(jac).all(axis=(1, 2))
        scales = np.full((len(points), 2), np.nan)
        for i in np.flatnonzero(valid):
            if np.linalg.det(jac[i]) > 1e-8:
                scales[i] = np.linalg.svd(np.diag(spacing[::-1]) @ np.linalg.inv(jac[i]), compute_uv=False)
        return scales


def content_index_map(mapping, shape_hw, valid_support, *, chunk_rows=64, output=None):
    """Compute one exact nearest-cell pullback per frame for reuse by all regions."""
    h, w = map(int, shape_hw)
    if valid_support.shape != (h, w) or chunk_rows <= 0:
        raise ValueError("Full-frame support and mapping dimensions differ")
    if output is None:
        output = np.empty((h, w), dtype=np.int32)
    if output.shape != (h, w) or output.dtype != np.int32:
        raise ValueError("Content indices need an int32 full-frame output")
    old_h, old_w = mapping.curvature.shape
    if old_h*old_w > np.iinfo(np.int32).max:
        raise ValueError("Historical content exceeds the index storage range")
    for y in range(0, h, chunk_rows):
        end = min(h, y+chunk_rows)
        yy, xx = np.mgrid[y:end, :w]
        xy = mapping(np.column_stack((xx.ravel(), yy.ravel())))
        finite = np.isfinite(xy).all(axis=1)
        cell = np.zeros(xy.shape, dtype=np.int64)
        cell[finite] = np.floor(xy[finite]+.5).astype(np.int64)
        valid = (finite & valid_support[y:end].ravel() & (cell[:, 0] >= 0)
                 & (cell[:, 0] < old_w) & (cell[:, 1] >= 0) & (cell[:, 1] < old_h))
        index = np.full(len(xy), -1, dtype=np.int32)
        index[valid] = cell[valid, 1]*old_w+cell[valid, 0]
        output[y:end] = index.reshape(end-y, w)
    return output


def sparse_region_mask(old_mask, index_map):
    """Store exact membership inside its bounding box, never the box alone."""
    valid = index_map >= 0
    if valid.any() and int(index_map[valid].max()) >= old_mask.size:
        raise ValueError("Content index exceeds the historical annotation canvas")
    mask = np.zeros(index_map.shape, bool)
    mask[valid] = np.asarray(old_mask, bool).ravel()[index_map[valid]]
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return {"bbox_xywh": np.array([0, 0, 0, 0]), "packed_mask": np.empty(0, np.uint8),
                "full_shape_hw": np.array(index_map.shape), "pixels": 0}
    x0, x1, y0, y1 = int(cols[0]), int(cols[-1])+1, int(rows[0]), int(rows[-1])+1
    return {"bbox_xywh": np.array([x0, y0, x1-x0, y1-y0]),
            "packed_mask": np.packbits(mask[y0:y1, x0:x1], axis=None),
            "full_shape_hw": np.array(index_map.shape), "pixels": int(mask.sum())}
