"""Storage-only metric aggregation of already computed native fields.

No image reader, inference, calibration, geometry builder or mass formula lives
here. The producer supplies native pixel masses and both gas-probability fields.
The native mode must bypass this module's reducer in the existing publisher.
"""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal
import math
import numpy as np

RESOLUTIONS_MM = (0.25, 0.5, 1.0, 2.0)
PROBABILITY_FIELDS = ("gas_probability_before_rules", "gas_probability_after_rules")
CELL_COUNT_FIELDS = {
    "static_included_area_native_pixel_count": "count of native pixels in the supplied include mask; same support as existing physical-area sums; zero-area pixels may count",
    "static_full_canvas_probability_pixel_count": "count of all finite full-canvas native pixel centres; independent of N1 domain, physical area and phase labels",
}
MASS_FIELDS = ("total", "aqueous", "gas")
LABEL_LEGEND = {"0": "water", "1": "dissolved", "2": "gas", "255": "excluded"}


def resolution_mm(value):
    """None means leave the original native publication path wholly untouched."""
    if value == "native" and isinstance(value, str):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Storage resolution must be native, 0.25, 0.5, 1 or 2 mm.")
    if not math.isfinite(value) or float(value) not in RESOLUTIONS_MM:
        raise ValueError("Unsupported storage resolution in millimetres.")
    return float(value)


def metric_grid_shape(extent_m, storage_resolution_mm):
    """Require exact decimal JSON endpoints on the origin-zero h lattice.

    Integer ratio arithmetic avoids tolerance, snapping and float-ceil cells.
    This only validates output DATA; native metric coordinates are untouched.
    """
    step_mm = resolution_mm(storage_resolution_mm)
    if step_mm is None:
        raise ValueError("Native storage must bypass the metric grid shape validator.")
    if len(extent_m) != 4 or any(isinstance(v, bool) or not math.isfinite(v) for v in extent_m):
        raise ValueError("Explicit finite metric extent x0,x1,y0,y1 is required.")
    step_n, step_d = (Decimal(str(step_mm)) / Decimal("1000")).as_integer_ratio()
    edges = []
    for value in extent_m:
        value_n, value_d = Decimal(str(value)).as_integer_ratio()
        index, remainder = divmod(value_n * step_d, value_d * step_n)
        if remainder:
            raise ValueError("Storage extent endpoints must lie on the exact origin-zero metric lattice (integer multiples of h).")
        edges.append(index)
    width, height = edges[1] - edges[0], edges[3] - edges[2]
    if width < 1 or height < 1:
        raise ValueError("Metric storage extent must have positive width and height.")
    if width * height > 100_000_000:
        raise ValueError("Declared metric storage grid exceeds the bounded cell count.")
    return height, width


@dataclass(frozen=True)
class MetricStorageGrid:
    native_shape: tuple[int, int]
    shape: tuple[int, int]
    extent_m: tuple[float, float, float, float]
    resolution_mm: float
    precision: str
    cell_index: np.ndarray
    included: np.ndarray
    native_pixel_area_m2: np.ndarray
    cell_area_m2: np.ndarray
    cell_pixel_count: np.ndarray
    probability_cell_index: np.ndarray
    cell_probability_pixel_count: np.ndarray
    probability_coordinate_counts: dict

    def metadata(self):
        return dict(format="darsia-metric-storage-grid-v1", native_shape=list(self.native_shape),
                    shape=list(self.shape), extent_m=list(self.extent_m),
                    resolution_mm=self.resolution_mm, axes="row=y increasing, column=x increasing",
                    assignment="native pixel centre; half-open cells and extent [min,max); exact upper edge STOP; no clamping",
                    cell_count_fields=dict(CELL_COUNT_FIELDS),
                    weighting="physical native pixel area for continuous/static and label-area fields only",
                    probability_weighting="arithmetic native-pixel count; independent of pixel area and N1 domain",
                    probability_cell_membership="literal native x/y pixel centres within the explicit extent",
                    probability_coordinate_counts=dict(self.probability_coordinate_counts),
                    computing_resolution="native", storage_precision=self.precision, reduction_dtype="float64")


def make_grid(x_m, y_m, pixel_area_m2, *, include, extent_m, storage_resolution_mm, precision="float32"):
    """Assign native pixel centres to an explicit fixed metric rig grid.

    Include is the caller's already defined geometric coverage, independent of
    phase labels. Nonzero mass outside it is rejected by aggregate_frame.
    """
    if precision not in {"float32", "float64"}:
        raise ValueError("One declared storage precision, float32 or float64, is required.")
    step_mm = resolution_mm(storage_resolution_mm)
    if step_mm is None:
        return None
    x, y, area = (np.asarray(v) for v in (x_m, y_m, pixel_area_m2))
    included = np.asarray(include)
    if x.ndim != 2 or x.shape != y.shape or x.shape != area.shape or x.shape != included.shape:
        raise ValueError("Metric coordinates, area and include must share one native 2D grid.")
    if included.dtype != np.bool_ or not np.any(included):
        raise ValueError("An explicit nonempty boolean geometric coverage is required.")
    if len(extent_m) != 4 or any(isinstance(v, bool) or not math.isfinite(v) for v in extent_m):
        raise ValueError("Explicit finite metric extent x0,x1,y0,y1 is required.")
    x0, x1, y0, y1 = map(float, extent_m)
    if not x1 > x0 or not y1 > y0:
        raise ValueError("Metric storage extent must have positive width and height.")
    height, width = metric_grid_shape(extent_m, step_mm)
    if not np.isfinite(x[included]).all() or not np.isfinite(y[included]).all():
        raise ValueError("Included pixel centres must have finite existing metric coordinates.")
    if (np.any(x[included] < x0) or np.any(x[included] >= x1)
            or np.any(y[included] < y0) or np.any(y[included] >= y1)):
        raise ValueError("Included native pixel centre lies outside declared half-open metric extent.")
    if not np.isfinite(area[included]).all() or np.any(area[included] < 0):
        raise ValueError("Included physical pixel areas must be finite and nonnegative.")
    # Q34 covers every finite canvas centre, not only the N1/area domain.
    # Validate before cell-area sums, mass reductions or any output side effect.
    coordinate_valid = np.isfinite(x) & np.isfinite(y)
    invalid_count = int(np.count_nonzero(~coordinate_valid))
    if invalid_count:
        raise ValueError("Unassignable native probability coordinates: invalid=" + str(invalid_count))
    probability_inside = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
    outside_count = int(np.count_nonzero(~probability_inside))
    if outside_count:
        raise ValueError("Full-canvas native pixel centre lies outside declared half-open metric extent: count=" + str(outside_count))
    step = step_mm / 1000.0
    ids = np.full(x.shape, -1, np.int64)
    col = np.floor((x[included]-x0)/step).astype(np.int64)
    row = np.floor((y[included]-y0)/step).astype(np.int64)
    if np.any((col < 0) | (col >= width) | (row < 0) | (row >= height)):
        raise ValueError("Included native pixel centre has an unassignable half-open cell index; no clamp is allowed.")
    ids[included] = row * width + col
    flat_ids = ids[included]
    cell_area = np.bincount(flat_ids, weights=area[included].astype(np.float64), minlength=height*width)
    count = np.bincount(flat_ids, minlength=height*width)
    probability_ids = np.full(x.shape, -1, np.int64)
    probability_col = np.floor((x[probability_inside]-x0)/step).astype(np.int64)
    probability_row = np.floor((y[probability_inside]-y0)/step).astype(np.int64)
    if np.any((probability_col < 0) | (probability_col >= width) | (probability_row < 0) | (probability_row >= height)):
        raise ValueError("Full-canvas native pixel centre has an unassignable half-open cell index; no clamp is allowed.")
    probability_ids[probability_inside] = probability_row * width + probability_col
    probability_count = np.bincount(probability_ids[probability_inside], minlength=height*width)
    coordinate_counts = dict(native_pixels=int(x.size), inside_extent=int(probability_inside.sum()),
                             outside_extent=int((~probability_inside).sum()), invalid_coordinates=invalid_count)
    return MetricStorageGrid(x.shape, (height,width), (x0,x1,y0,y1), step_mm, precision,
                             ids, included.copy(), area.astype(np.float64), cell_area, count,
                             probability_ids, probability_count, coordinate_counts)


def _array(grid, value, name):
    result = np.asarray(value)
    if result.shape != grid.native_shape:
        raise ValueError(name + " differs from the native grid.")
    return result


def _sum(grid, value):
    return np.bincount(grid.cell_index[grid.included],
                       weights=np.asarray(value, np.float64)[grid.included],
                       minlength=int(np.prod(grid.shape)))


def _fraction(grid, numerator):
    out = np.full(grid.cell_area_m2.shape, np.nan, np.float64)
    np.divide(numerator, grid.cell_area_m2, out=out, where=grid.cell_area_m2 > 0)
    return out.reshape(grid.shape).astype(np.dtype(grid.precision))


def _mean(grid, value, name):
    value = _array(grid, value, name)
    finite = np.isfinite(value)
    weights = np.where(grid.included & finite, grid.native_pixel_area_m2, 0.)
    denominator = _sum(grid, weights)
    numerator = _sum(grid, np.where(finite, value, 0.) * weights)
    mean = np.full(denominator.shape, np.nan, np.float64)
    np.divide(numerator, denominator, out=mean, where=denominator > 0)
    return mean.reshape(grid.shape).astype(np.dtype(grid.precision)), _fraction(grid, denominator)


def _pixel_mean(grid, value, name):
    """Q28 count mean and valid-pixel fraction, never physical area weights."""
    value = _array(grid, value, name)
    selected = (grid.probability_cell_index >= 0) & np.isfinite(value)
    ids = grid.probability_cell_index[selected]
    valid_count = np.bincount(ids, minlength=int(np.prod(grid.shape)))
    numerator = np.bincount(ids, weights=value[selected].astype(np.float64),
                            minlength=int(np.prod(grid.shape)))
    mean = np.full(valid_count.shape, np.nan, np.float64)
    fraction = np.full(valid_count.shape, np.nan, np.float64)
    np.divide(numerator, valid_count, out=mean, where=valid_count > 0)
    np.divide(valid_count, grid.cell_probability_pixel_count, out=fraction,
              where=grid.cell_probability_pixel_count > 0)
    dtype = np.dtype(grid.precision)
    return mean.reshape(grid.shape).astype(dtype), fraction.reshape(grid.shape).astype(dtype)


def aggregate_static(grid, *, depth_m, porosity, facies):
    """Storage of given static fields; never recompute depth or pore capacity."""
    if grid is None:
        raise ValueError("Native publication must bypass the coarse storage reducer.")
    depth, depth_valid = _mean(grid, depth_m, "depth")
    phi, phi_valid = _mean(grid, porosity, "porosity")
    categories = _array(grid, facies, "facies")
    if not np.issubdtype(categories.dtype, np.integer):
        raise ValueError("Facies must retain existing integer category IDs.")
    majority = np.zeros(int(np.prod(grid.shape)), categories.dtype)
    largest = np.zeros(majority.shape, np.float64)
    # Sorted IDs + strict greater-than makes ties deterministic: lowest ID wins.
    for category in np.unique(categories[grid.included]):
        weights = _sum(grid, np.where(categories == category, grid.native_pixel_area_m2, 0.))
        changed = weights > largest
        majority[changed] = category
        largest[changed] = weights[changed]
    return dict(depth_m_mean=depth, depth_valid_area_fraction=depth_valid,
                porosity_mean=phi, porosity_valid_area_fraction=phi_valid,
                pixel_area_m2_sum=grid.cell_area_m2.reshape(grid.shape).astype(np.dtype(grid.precision)),
                included_area_native_pixel_count=grid.cell_pixel_count.reshape(grid.shape).copy(),
                full_canvas_probability_pixel_count=grid.cell_probability_pixel_count.reshape(grid.shape).copy(),
                facies_majority=majority.reshape(grid.shape),
                facies_majority_area_fraction=_fraction(grid,largest),
                cell_has_area=(grid.cell_area_m2.reshape(grid.shape)>0))


def aggregate_frame(grid, *, concentration_aqueous, saturation_free_gas, phase_labels,
                    masks, probabilities, native_pixel_masses_g, native_mass_totals_g,
                    before_status=None):
    """Aggregate unchanged native products with the explicit Q23 field meanings.

    BEFORE is the producer's decoded pre-categorical pixel score, possibly
    pending. AFTER is exactly the existing final gas indicator, not a fitted
    or reconstructed probability. Missing BEFORE never suppresses AFTER.
    Q28 uses pixel-count means, including excluded/outside-N1-domain labels in
    AFTER's denominator. No historical/255/domain mask becomes label support.
    """
    if grid is None:
        raise ValueError("Native publication must bypass the coarse storage reducer.")
    names=set(probabilities)
    if (PROBABILITY_FIELDS[1] not in names or not names.issubset(PROBABILITY_FIELDS)):
        raise ValueError("Final gas indicator and only Q23 probability fields are required.")
    if PROBABILITY_FIELDS[0] not in names and (
            not isinstance(before_status,dict) or before_status.get("state")!="pending"
            or not isinstance(before_status.get("reason"),str) or not before_status["reason"]):
        raise ValueError("Missing pre-rule score needs an explicit pending reason.")
    labels = _array(grid,phase_labels,"phase labels")
    if labels.dtype != np.uint8 or not np.isin(labels,(0,1,2,255)).all():
        raise ValueError("Phase labels must be unchanged uint8 categories 0,1,2,255.")
    concentration, c_valid = _mean(grid,concentration_aqueous,"concentration")
    saturation, s_valid = _mean(grid,saturation_free_gas,"saturation")
    products = dict(concentration_aqueous=concentration, saturation_free_gas=saturation,
                    concentration_valid_area_fraction=c_valid, saturation_valid_area_fraction=s_valid)
    for category,name in [(0,"water"),(1,"dissolved"),(2,"gas"),(255,"excluded")]:
        products["phase_"+name+"_area_fraction"] = _fraction(
            grid,_sum(grid,np.where(labels==category,grid.native_pixel_area_m2,0.)))
    for name,value in masks.items():
        if not isinstance(name,str) or not name or not name.replace('_','').isalnum():
            raise ValueError("Mask names must be explicit nonempty identifiers.")
        mask=_array(grid,value,name)
        if mask.dtype != np.bool_:
            raise ValueError("Native masks must be boolean; fractions are stored separately.")
        products["mask_"+name+"_area_fraction"] = _fraction(
            grid,_sum(grid,np.where(mask,grid.native_pixel_area_m2,0.)))
    final=_array(grid,probabilities[PROBABILITY_FIELDS[1]],"final gas indicator")
    if (final.dtype not in (np.dtype(bool),np.dtype("uint8"))
            or not np.array_equal(final,labels==2)):
        raise ValueError("AFTER must be the exact final gas indicator (phase_labels == 2).")
    for name in PROBABILITY_FIELDS:
        if name not in probabilities:
            continue
        p=_array(grid,probabilities[name],name)
        if np.isinf(p).any() or np.any(np.isfinite(p) & ((p<0)|(p>1))):
            raise ValueError("Producer gas probability lies outside [0,1] or is infinite.")
        reduced, valid_fraction = _pixel_mean(grid,p,name)
        products[name] = reduced
        if name == PROBABILITY_FIELDS[0]:
            products[name+"_valid_pixel_fraction"] = valid_fraction
        # There is no separately supplied final-label support mask. A=0 belongs
        # only to BEFORE; final label255 remains a non-gas vote, not missing.
    if set(native_pixel_masses_g)!=set(MASS_FIELDS) or set(native_mass_totals_g)!=set(MASS_FIELDS):
        raise ValueError("All three original native pixel masses and scalar totals are required.")
    checks={}
    for name in MASS_FIELDS:
        pixel_mass=_array(grid,native_pixel_masses_g[name],name+" mass")
        expected=native_mass_totals_g[name]
        if (isinstance(expected,bool) or not isinstance(expected,(int,float)) or not math.isfinite(expected)
                or expected<0 or not np.isfinite(pixel_mass).all() or np.any(pixel_mass<0)):
            raise ValueError("Original pixel masses/scalars must be finite and nonnegative.")
        if np.any(pixel_mass[~grid.included]!=0):
            raise ValueError("Nonzero native mass lies outside declared storage coverage.")
        cells=_sum(grid,pixel_mass).reshape(grid.shape).astype(np.dtype(grid.precision))
        total=float(cells.sum(dtype=np.float64))
        error=abs(total-expected)
        allowed=1e-6*abs(expected)
        if error>allowed:
            raise ValueError("Stored cell mass sum does not reproduce native "+name+" total within 1e-6.")
        products["cell_mass_"+name+"_g"]=cells
        checks[name]=dict(native_total_g=expected, stored_cell_sum_g=total,
                          absolute_difference_g=error, relative_limit=1e-6, passed=True)
    return dict(arrays=products,metadata=dict(format="darsia-coarse-stored-frame-v1",
                precision=grid.precision, reduction_dtype="float64", grid=grid.metadata(),
                cell_mass="sum of producer-supplied native pixel masses in grams",
                means_and_fractions="continuous/static means and label/mask fractions retain physical pixel-area weighting",
                probability_semantics="Q23/Q28: BEFORE mean of decoded L/255 over A=255 pixels; AFTER count(labels==2)/all cell pixels",
                probability_weighting="arithmetic pixel count, not pixel area",
                before_valid_fraction="count(A=255)/all native pixel centres assigned to the cell; zero valid mean is NaN",
                after_denominator="all assigned native pixels, including label255 and pixels outside N1 domain",
                after_support_mask="none supplied; no support fraction inferred from domain or label255",
                label_legend=dict(LABEL_LEGEND),
                 before_rules_status=(dict(state="complete") if PROBABILITY_FIELDS[0] in probabilities
                                      else dict(before_status)),
                native_mass_checks=checks, computing_resolution="native"))
