"""Stored products of a lattice run at a coarser storage resolution.

A lattice run (pipeline ``lattice-phase-n1``) computes on its metric lattice of
0.5 or 1 mm. Its bound storage request may store at 1 or 2 mm, an integer
multiple of the computing resolution. Both lattices start at the origin and
the request's extent lies on the stored lattice, so every stored cell is the
union of exactly ``factor x factor`` computing cells.

Each frame keeps its lattice frame files unchanged. Its stage also receives the
stored products, reduced from the frame's lattice products by the native
storage reducers of :mod:`storage_resolution` (one implementation). The
lattice cell centres are the reducer's source points, and their assignment is
checked against the exact block index:

* masses (pore volume times density) and areas: block sums;
* concentration, saturation, depth and porosity: area-weighted means over the
  included cells, with their valid area fractions;
* labels and masks: area fractions; the final gas fraction counts cells;
* facies: area majority, lowest id on ties.

The included cells are the N1 mass domain on the lattice, as native storage
includes the native N1 domain. The stored files are written into the frame's
stage before the frame commit, so the commit binds them and a resumed run finds
them with every committed frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import math
from pathlib import Path

import numpy as np

from .storage_resolution import (PROBABILITY_FIELDS, aggregate_frame, aggregate_static, make_grid,
                                 metric_grid_shape, resolution_mm)

STORED_FORMAT = "darsia-lattice-stored-frame-v1"
ARCHIVE = "lattice_stored_frame.npz"
RECEIPT = "lattice_stored_frame.json"
#: The reducer names two static counts for native pixels; here they count lattice cells.
RENAMED = {"included_area_native_pixel_count": "included_area_lattice_cell_count",
           "full_canvas_probability_pixel_count": "lattice_cell_count"}
#: Lattice runs have no native-centre pre-rule score to store.
BEFORE_RULES = dict(state="pending", reason="lattice_computing_stores_no_pre_rule_score")
#: The native storage limit for a stored mass sum against its source total.
MASS_RELATIVE_LIMIT = 1e-6
REDUCTION = dict(
    method="exact block reduction: each stored cell is the union of block_factor x block_factor "
           "computing-lattice cells; the origin-zero lattices nest exactly",
    reducer="storage_resolution.make_grid, aggregate_frame and aggregate_static on the lattice cell "
            "centres; the assignment equals the exact block index",
    source_points="computing-lattice cells: each 'native pixel' of the reducer metadata is one lattice cell",
    masses="block sums of lattice cell masses (pore volume x density, storage_outputs.pixel_masses)",
    areas="block sums of lattice cell areas over the included cells",
    means="concentration, saturation, depth and porosity: area-weighted means over included finite "
          "cells, with valid area fractions",
    labels="area fractions over the included area; gas_probability_after_rules counts gas cells "
           "among all cells of the stored cell",
    facies="area majority over the included cells, lowest id on ties",
    included="the N1 mass domain on the lattice, as native storage includes the native N1 domain",
    renamed_arrays=dict(RENAMED))


def block_factor(stored_mm, computing_mm):
    """Computing cells along each side of one stored cell; refuse a non-integer ratio."""
    stored, computing = resolution_mm(stored_mm), resolution_mm(computing_mm)
    if stored is None or computing is None:
        raise ValueError("Block reduction needs metric stored and computing resolutions")
    ratio = Decimal(str(stored)) / Decimal(str(computing))
    if ratio < 1:
        raise ValueError("The stored resolution cannot be finer than the computing resolution")
    if ratio != ratio.to_integral_value():
        raise ValueError("The stored resolution must be an integer multiple of the computing resolution")
    return int(ratio)


@dataclass(frozen=True)
class StoredLayer:
    """The stored lattice of one lattice run.

    ``lattice`` is the run's ``ComputingLattice``; ``factor`` computing cells
    span one stored cell along each side; ``shape`` is (rows, columns) of the
    stored grid; ``precision`` is the stored float precision.
    """

    lattice: object
    resolution_mm: float
    factor: int
    shape: tuple
    precision: str

    def metadata(self):
        """The stored grid identity recorded in the manifest, the result and every stored frame."""
        return dict(format=STORED_FORMAT, resolution_mm=self.resolution_mm, block_factor=self.factor,
                    shape=list(self.shape), extent_m=list(self.lattice.extent_m),
                    computing_resolution_mm=self.lattice.resolution_mm, precision=self.precision,
                    axes="storage orientation: row = y increasing, column = x increasing",
                    cells="each stored cell is the union of block_factor x block_factor computing cells")


def stored_layer(request, lattice):
    """The stored layer of an admitted lattice request, or None when stored = computed.

    ``request`` is the storage request with its precision; ``lattice`` the
    run's computing lattice on the request's extent.
    """
    stored = resolution_mm(request["resolution_mm"])
    if stored is None:
        raise ValueError("A lattice run stores on a metric lattice")
    factor = block_factor(stored, lattice.resolution_mm)
    if factor == 1:
        return None
    if request.get("precision") not in {"float32", "float64"}:
        raise ValueError("A stored layer needs the run's float32 or float64 precision")
    shape = tuple(map(int, metric_grid_shape(request["extent_m"], stored)))
    if (tuple(map(float, request["extent_m"])) != tuple(lattice.extent_m)
            or (shape[0] * factor, shape[1] * factor) != tuple(lattice.shape)):
        raise ValueError("The stored and computing lattices do not nest")
    return StoredLayer(lattice, stored, factor, shape, request["precision"])


def block_index(layer):
    """Stored cell id of every computing cell, in computing orientation (row 0 = highest y)."""
    rows, columns = layer.lattice.shape
    stored_row = (rows - 1 - np.arange(rows, dtype=np.int64)) // layer.factor
    stored_column = np.arange(columns, dtype=np.int64) // layer.factor
    return stored_row[:, None] * layer.shape[1] + stored_column[None, :]


def stored_grid(layer, *, include, pixel_area_m2):
    """Assign the computing cells to their stored cells with ``storage_resolution.make_grid``.

    ``include`` (boolean) and ``pixel_area_m2`` lie on the computing lattice in
    computing orientation. The assignment must equal :func:`block_index`.
    """
    x, y = layer.lattice.cell_centres_m()
    include = np.asarray(include)
    grid = make_grid(x, y, np.asarray(pixel_area_m2, np.float64), include=include,
                     extent_m=layer.lattice.extent_m, storage_resolution_mm=layer.resolution_mm,
                     precision=layer.precision)
    expected = block_index(layer)
    if (tuple(grid.shape) != tuple(layer.shape)
            or not np.array_equal(grid.probability_cell_index, expected)
            or not np.array_equal(grid.cell_index, np.where(include, expected, -1))):
        raise ValueError("Stored cells are not the exact blocks of the computing lattice")
    return grid


def reduce_frame(layer, *, labels, concentration, saturation, masks, masses_g, totals_g, include,
                 pixel_area_m2, depth_m, porosity, facies):
    """The stored arrays of one lattice frame and the reducer metadata.

    Every array lies on the computing lattice in computing orientation;
    ``masses_g`` holds the cell masses and ``totals_g`` the frame's scalar
    masses (keys total, aqueous, gas). The reducer refuses mass outside
    ``include`` and stored sums that miss a total by more than 1e-6.
    """
    grid = stored_grid(layer, include=include, pixel_area_m2=pixel_area_m2)
    labels = np.asarray(labels)
    reduced = aggregate_frame(grid, concentration_aqueous=concentration, saturation_free_gas=saturation,
                              phase_labels=labels, masks=masks,
                              probabilities={PROBABILITY_FIELDS[1]: labels == 2},
                              native_pixel_masses_g=masses_g, native_mass_totals_g=totals_g,
                              before_status=BEFORE_RULES)
    static = aggregate_static(grid, depth_m=depth_m, porosity=porosity, facies=facies)
    arrays = dict(reduced["arrays"], **{"static_" + RENAMED.get(key, key): value
                                        for key, value in static.items()})
    # The reducer labels its source as native; here the source is the lattice.
    computing = layer.lattice.resolution_mm
    metadata = dict(reduced["metadata"], computing_resolution=computing,
                    grid=dict(reduced["metadata"]["grid"], computing_resolution=computing))
    return arrays, metadata


def _conservation(masses_g, arrays):
    """Lattice and stored mass sums per component, with the native relative limit."""
    rows = {}
    for component, values in masses_g.items():
        lattice_sum = math.fsum(np.asarray(values, np.float64).ravel())
        stored_sum = math.fsum(np.asarray(arrays["cell_mass_" + component + "_g"], np.float64).ravel())
        difference = abs(stored_sum - lattice_sum)
        if difference > MASS_RELATIVE_LIMIT * abs(lattice_sum):
            raise ValueError("Stored " + component + " mass differs from its lattice cells")
        rows[component] = dict(lattice_cell_sum_g=lattice_sum, stored_cell_sum_g=stored_sum,
                               absolute_difference_g=difference, relative_limit=MASS_RELATIVE_LIMIT,
                               passed=True)
    return rows


def write_stored_frame(stage, layer, *, identity, spec, inferred, phase, capture, evaluated, validation):
    """Write one frame's stored products into its stage, before the frame commit.

    ``identity`` holds the run, job, configuration, code commit and
    calibration of the task. ``inferred`` and ``phase`` are the frame's
    lattice phase result and its saved descriptors; ``capture`` and
    ``evaluated`` are the output capture and the result of
    ``LatticeMass.evaluate``. The archive is written first and the receipt
    last. Returns the entry the frame's stage binding records.
    """
    from PIL import Image
    from .mass.n1_pipeline import DataReader, _array_identity, _write_json, _write_npz
    from .storage_outputs import TOTALS, pixel_masses
    stage = Path(stage)
    if any((stage / name).exists() for name in (ARCHIVE, RECEIPT)):
        raise FileExistsError("Stored lattice products must be new frame files")
    frame, result = capture["frame"], evaluated["result"]
    export = evaluated["export"]["products"]
    labels = np.asarray(inferred["labels"])
    shape = tuple(layer.lattice.shape)
    if (labels.dtype != np.uint8 or labels.shape != shape or not np.array_equal(labels, frame.phase_labels)
            or result["precision"] != layer.precision):
        raise ValueError("Stored products must reduce the unchanged lattice labels at the run precision")
    with Image.open(DataReader(unc_only=False).path(phase["labels"])) as image:
        if not np.array_equal(np.asarray(image), labels):
            raise ValueError("Saved lattice phase labels differ from the stored input")
    masses, density_ids, weight_id = pixel_masses(evaluated["export"], capture["volume_m3"], shape)
    totals = {key: result[name] for key, name in TOTALS.items()}
    lattice_checks = {}
    for component, values in masses.items():
        expected = totals[component]
        actual = float(values.sum(dtype=np.float64))
        if (isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(expected)
                or abs(actual - expected) > MASS_RELATIVE_LIMIT * abs(expected)):
            raise ValueError("Lattice cell masses differ from the frame's " + component + " mass")
        lattice_checks[component] = dict(frame_total_g=expected, lattice_cell_sum_g=actual,
                                          relative_limit=MASS_RELATIVE_LIMIT, passed=True)
    classes = {name: np.asarray(value) for name, value in inferred["classes"].items()}
    masks = dict(classes, main_mass_eligible=result["fields"]["main_mass_eligible_mask"])
    masks.update(frame.excluded_masks or {})
    static = dict(pixel_area_m2=np.broadcast_to(np.asarray(frame.pixel_area_m2), shape),
                  depth_m=np.broadcast_to(np.asarray(frame.depth_m), shape),
                  porosity=np.broadcast_to(np.asarray(frame.porosity), shape), facies=frame.facies)
    arrays, reducer = reduce_frame(
        layer, labels=labels, concentration=export["concentration_aqueous"],
        saturation=export["saturation_free_gas"], masks=masks, masses_g=masses, totals_g=totals,
        include=frame.domain, **static)
    conservation = _conservation(masses, arrays)
    archive = _write_npz(stage / ARCHIVE, arrays)
    products = {name: _array_identity(value) for name, value in export.items()}
    products.update({name + "_mask": _array_identity(value) for name, value in masks.items()
                     if name not in classes})
    geometry = {name: _array_identity(value) for name, value in static.items()}
    geometry["transport_weight_l"] = weight_id
    gas = int(np.count_nonzero(labels == 2))
    document = dict(
        format=STORED_FORMAT, complete=True, case_id=spec["case_id"], source=spec["source"],
        **identity, precision=layer.precision, stored_grid=layer.metadata(),
        lattice=layer.lattice.metadata(), reduction=dict(REDUCTION), reducer=reducer,
        lattice_inputs=dict(products=products, native_geometry=geometry,
                            phase_classes={name: _array_identity(value) for name, value in classes.items()},
                            n1_domain=_array_identity(frame.domain), mass_densities=density_ids),
        before_rules=dict(BEFORE_RULES),
        after_rules=dict(source_array="final lattice phase_labels == 2", source=phase["labels"],
                         values=[0, 1], gas_cells=gas,
                         coarse_reduction="count(final labels==2)/all lattice cells of the stored cell"),
        final_label_counts={str(value): int(np.count_nonzero(labels == value)) for value in (0, 1, 2, 255)},
        frame_mass_totals_g=totals, lattice_cell_mass_checks=lattice_checks,
        mass_conservation=conservation,
        arrays={name: _array_identity(value) for name, value in arrays.items()},
        files=[archive], lattice_frame_files_unchanged=True, validation=dict(validation))
    receipt = _write_json(stage / RECEIPT, document)
    return dict(format=STORED_FORMAT, case_id=spec["case_id"], resolution_mm=layer.resolution_mm,
                block_factor=layer.factor, shape=list(layer.shape), receipt=receipt, files=[archive])


def verify_stored_entry(entry, layer, *, spec, inventory, result, reader, configuration_sha256):
    """Admit a committed frame's stored products before a resume reuses the frame.

    ``inventory`` is the frame's committed stage inventory and ``result`` its
    committed result; ``reader`` verifies file content. Returns the receipt.
    """
    from .storage_outputs import TOTALS
    if (not isinstance(entry, dict) or entry.get("format") != STORED_FORMAT
            or entry.get("case_id") != spec["case_id"] or entry.get("resolution_mm") != layer.resolution_mm
            or entry.get("block_factor") != layer.factor or entry.get("shape") != list(layer.shape)
            or not isinstance(entry.get("files"), list) or len(entry["files"]) != 1):
        raise ValueError("Committed lattice frame lacks its stored products at the requested resolution")
    for bound, name in ((entry["receipt"], RECEIPT), (entry["files"][0], ARCHIVE)):
        if Path(bound["path"]).name != name or inventory.get(name) != bound:
            raise ValueError("Stored lattice products are not bound by the committed stage inventory")
        reader.path(bound)
    document = reader.json(entry["receipt"])
    totals = {key: result[name] for key, name in TOTALS.items()}
    if (document.get("format") != STORED_FORMAT or document.get("complete") is not True
            or document.get("case_id") != spec["case_id"] or document.get("source") != spec["source"]
            or document.get("run") != result["run"] or document.get("precision") != result["precision"]
            or document.get("configuration", {}).get("sha256") != configuration_sha256
            or document.get("stored_grid") != layer.metadata() or document.get("files") != entry["files"]
            or document.get("frame_mass_totals_g") != totals
            or set(document.get("mass_conservation", ())) != set(totals)
            or any(row.get("passed") is not True for row in document["mass_conservation"].values())):
        raise ValueError("Stored lattice receipt differs from its committed frame")
    return document
