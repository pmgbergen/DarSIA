"""Stored products of a lattice run: exact blocks of its computing lattice.

Synthetic lattices check the reduction: shapes and extents, mass conserved
exactly, area-weighted means with valid fractions, label fractions and facies
majorities. The synthetic two-frame run of test_lattice_runner checks the
runner's stored products, their frame commit and a resume. No NAS file is read.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from darsia.fluidflower import lattice_storage as ls
from darsia.fluidflower.computing_lattice import computing_lattice
from darsia.fluidflower.mass.n1_pipeline import _array_identity
from darsia.fluidflower.pipeline import checkpoint, lattice_runner as lr
from darsia.fluidflower.pipeline.job_queue import descriptor
from darsia.fluidflower.storage_outputs import DENSITIES, LA_ENCODING, REQUEST_FORMAT, TOTALS
from test_lattice_runner import FakeInputs, config_of, lattice_world, run  # noqa: F401  (fixture)
from test_phase_rules import parameters as rule_parameters  # noqa: F401  (fixture)
from test_phase_domain import parameters as domain_parameters  # noqa: F401  (fixture)

EXTENT = (0.0, 0.146, 0.002, 0.184)  # on the 2 mm lattice
NESTED = ((0.5, 1.0, 2), (0.5, 2.0, 4), (1.0, 2.0, 2))  # computing, stored, factor


def request(stored, extent=EXTENT, precision="float64"):
    """A storage request with its precision, as lattice_request returns it."""
    return dict(format=REQUEST_FORMAT, gas_score_encoding=LA_ENCODING, resolution_mm=stored,
                extent_m=list(extent), precision=precision)


def layer_for(computing, stored, precision="float64"):
    return ls.stored_layer(request(stored, precision=precision), computing_lattice(EXTENT, computing))


def blocks(layer, values):
    """A computing-orientation lattice array in storage orientation, split into its blocks."""
    k, (rows, columns) = layer.factor, layer.shape
    return np.asarray(values)[::-1].reshape(rows, k, columns, k)


def synthetic_frame(layer, seed=0):
    """Lattice arrays of one synthetic frame.

    Masses and cell areas lie on dyadic grids, so every sum is exact in any
    order: equal facies areas are exact ties, as the reducer's rule needs.
    """
    rng = np.random.default_rng(seed)
    shape = layer.lattice.shape
    include = np.ones(shape, bool)
    include[:, :3] = False  # a strip outside the N1 domain
    area = np.full(shape, 2.0 ** -22)  # about 0.24 mm2
    area[::2] *= 0.75  # cell areas vary by row
    labels = rng.choice(np.array([0, 1, 2], np.uint8), size=shape)
    labels[~include] = 255
    masses = {name: np.where(include, rng.integers(0, 4096, size=shape) / 1024.0, 0.0)
              for name in ("total", "aqueous", "gas")}
    concentration = rng.uniform(0.0, 2.0, shape)
    concentration[rng.random(shape) < 0.2] = np.nan
    facies = rng.choice(np.array([1, 2, 3], np.uint8), size=shape)
    return dict(labels=labels, concentration=concentration, saturation=rng.uniform(0.0, 1.0, shape),
                masks=dict(rig_domain=include.copy(), modelled=rng.random(shape) < 0.5),
                masses_g=masses, totals_g={name: math.fsum(value.ravel()) for name, value in masses.items()},
                include=include, pixel_area_m2=area, depth_m=rng.uniform(0.01, 0.02, shape),
                porosity=rng.uniform(0.3, 0.45, shape), facies=facies)


@pytest.mark.parametrize("computing, stored, factor", [*NESTED, (1.0, 1.0, 1), (0.5, 0.5, 1)])
def test_the_block_factor_is_the_integer_resolution_ratio(computing, stored, factor):
    assert ls.block_factor(stored, computing) == factor


def test_a_stored_resolution_finer_than_computing_is_refused():
    for stored, computing in ((0.5, 1.0), (0.25, 0.5)):
        with pytest.raises(ValueError, match="cannot be finer"):
            ls.block_factor(stored, computing)
    with pytest.raises(ValueError, match="metric stored and computing"):
        ls.block_factor("native", 1.0)


def test_storing_on_the_computing_lattice_has_no_stored_layer():
    assert ls.stored_layer(request(1.0), computing_lattice(EXTENT, 1.0)) is None
    off = (0.0, 0.145, 0.0, 0.183)  # on the 1 mm lattice only
    with pytest.raises(ValueError, match="origin-zero metric lattice"):
        ls.stored_layer(request(2.0, off), computing_lattice(off, 1.0))


@pytest.mark.parametrize("computing, stored, factor", NESTED)
def test_the_stored_lattice_nests_the_computing_lattice_exactly(computing, stored, factor):
    layer = layer_for(computing, stored)
    rows, columns = round((EXTENT[3] - EXTENT[2]) * 1000 / stored), round((EXTENT[1] - EXTENT[0]) * 1000 / stored)
    assert layer.factor == factor and layer.shape == (rows, columns)
    assert layer.lattice.shape == (rows * factor, columns * factor)
    assert layer.metadata()["extent_m"] == list(EXTENT) and layer.metadata()["shape"] == [rows, columns]
    include = np.ones(layer.lattice.shape, bool)
    cell = (computing / 1000.0) ** 2
    grid = ls.stored_grid(layer, include=include, pixel_area_m2=np.full(layer.lattice.shape, cell))
    assert grid.extent_m == EXTENT and grid.shape == (rows, columns) and grid.resolution_mm == stored
    np.testing.assert_allclose(grid.cell_area_m2, (stored / 1000.0) ** 2, rtol=1e-12)
    index = ls.block_index(layer)
    # Storage orientation: the lowest computing row opens stored row 0, the top right closes the grid.
    assert index[-1, 0] == 0 and index[0, -1] == rows * columns - 1
    assert (np.bincount(index.ravel(), minlength=rows * columns) == factor * factor).all()


@pytest.mark.parametrize("precision", ("float64", "float32"))
@pytest.mark.parametrize("computing, stored, factor", NESTED)
def test_mass_is_conserved_exactly_by_block_sums(computing, stored, factor, precision):
    layer = layer_for(computing, stored, precision)
    frame = synthetic_frame(layer)
    arrays, metadata = ls.reduce_frame(layer, **frame)
    for name, values in frame["masses_g"].items():
        cells = arrays["cell_mass_" + name + "_g"]
        assert cells.dtype == np.dtype(precision) and cells.shape == layer.shape
        np.testing.assert_array_equal(cells, blocks(layer, values).sum(axis=(1, 3)))
        assert math.fsum(cells.astype(np.float64).ravel()) == frame["totals_g"][name]
        check = metadata["native_mass_checks"][name]
        assert check["passed"] and check["absolute_difference_g"] == 0.0
    assert metadata["computing_resolution"] == computing and metadata["grid"]["computing_resolution"] == computing


@pytest.mark.parametrize("computing, stored, factor", NESTED)
def test_means_fractions_and_majorities_follow_the_native_reducer(computing, stored, factor):
    layer = layer_for(computing, stored)
    frame = synthetic_frame(layer, seed=1)
    tie = np.zeros((factor, factor), np.uint8)
    tie[:] = 3
    tie[::2, ::2] = tie[1::2, 1::2] = 1  # stored cell (0, 2): equal areas of facies 1 and 3
    flipped = frame["facies"][::-1]
    flipped[0:factor, 2 * factor:3 * factor] = tie
    frame["facies"] = flipped[::-1].copy()
    arrays, _ = ls.reduce_frame(layer, **frame)
    area = blocks(layer, frame["pixel_area_m2"])
    include = blocks(layer, frame["include"])
    included_area = (area * include).sum(axis=(1, 3))
    for name in ("concentration", "depth_m", "porosity"):
        values = blocks(layer, frame[name])
        weight = area * (include & np.isfinite(values))
        expected = np.full(layer.shape, np.nan)
        covered = weight.sum(axis=(1, 3))
        np.divide((np.where(weight > 0, values, 0.0) * weight).sum(axis=(1, 3)), covered, out=expected,
                  where=covered > 0)
        key = {"concentration": "concentration_aqueous", "depth_m": "static_depth_m_mean",
               "porosity": "static_porosity_mean"}[name]
        np.testing.assert_allclose(arrays[key], expected, rtol=1e-12)
        valid = {"concentration": "concentration_valid_area_fraction", "depth_m": "static_depth_valid_area_fraction",
                 "porosity": "static_porosity_valid_area_fraction"}[name]
        with np.errstate(invalid="ignore"):
            np.testing.assert_allclose(arrays[valid], covered / included_area, rtol=1e-12)
    labels = blocks(layer, frame["labels"])
    with np.errstate(invalid="ignore"):
        for value, name in ((0, "water"), (1, "dissolved"), (2, "gas"), (255, "excluded")):
            fraction = (area * include * (labels == value)).sum(axis=(1, 3)) / included_area
            np.testing.assert_allclose(arrays["phase_" + name + "_area_fraction"], fraction, rtol=1e-12)
    np.testing.assert_allclose(arrays["gas_probability_after_rules"], (labels == 2).mean(axis=(1, 3)), rtol=1e-12)
    np.testing.assert_allclose(arrays["static_pixel_area_m2_sum"], included_area, rtol=1e-12)
    assert (arrays["static_included_area_lattice_cell_count"] == include.sum(axis=(1, 3))).all()
    assert (arrays["static_lattice_cell_count"] == factor * factor).all()
    assert not {"static_included_area_native_pixel_count", "static_full_canvas_probability_pixel_count"} & set(arrays)
    facies = blocks(layer, frame["facies"])
    scores = np.stack([(area * include * (facies == value)).sum(axis=(1, 3)) for value in (1, 2, 3)])
    expected = np.where(scores.max(0) > 0, np.argmax(scores, axis=0) + 1, 0)  # argmax keeps the lowest id
    assert (arrays["static_facies_majority"] == expected).all()
    assert arrays["static_facies_majority"][0, 2] == 1  # the tie takes the lowest id
    assert "gas_probability_before_rules" not in arrays


def test_mass_outside_the_included_cells_is_refused():
    layer = layer_for(1.0, 2.0)
    frame = synthetic_frame(layer)
    frame["masses_g"]["gas"][0, 0] = 1.0  # column 0 lies outside the N1 domain
    frame["totals_g"]["gas"] += 1.0
    with pytest.raises(ValueError, match="outside declared storage coverage"):
        ls.reduce_frame(layer, **frame)


@pytest.fixture
def stored_world(lattice_world):
    """The synthetic lattice run, storing at 2 mm over 1 mm computing."""
    lattice_world["config"]["storage"] = dict(lattice_world["config"]["storage"], resolution_mm=2.0,
                                              extent_m=[0.0, 0.146, 0.0, 0.184])
    return lattice_world


def stored_arrays(output, case):
    with np.load(Path(output) / "preparation" / case / ls.ARCHIVE) as archive:
        return {name: archive[name] for name in archive.files}


def test_a_coarser_stored_run_keeps_its_lattice_frames_and_adds_exact_stored_products(stored_world):
    result, output = run(stored_world, "stored")
    expected = dict(resolution_mm=2.0, block_factor=2, shape=[92, 73], computing_resolution_mm=1.0)
    assert {key: result["stored"][key] for key in expected} == expected
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["lattice"]["stored"] == result["stored"] and manifest["lattice"]["lattice"]["shape"] == [184, 146]
    for spec in stored_world["config"]["frames"]:
        final = output / "frames" / spec["case_id"]
        assert {path.name for path in final.iterdir()} == {
            "products.npz", "native_geometry.npz", "native_statistics.json.gz", "result.json", "commit.json"}
        document = json.loads((final / "result.json").read_bytes())
        stage = document["prediction"]["pipeline"]
        entry = stage["stored"]
        assert entry["files"] == [stage["files"][ls.ARCHIVE]] and entry["receipt"] == stage["files"][ls.RECEIPT]
        receipt = json.loads(Path(entry["receipt"]["path"]).read_bytes())
        assert receipt["stored_grid"] == result["stored"] and receipt["before_rules"]["state"] == "pending"
        stored = stored_arrays(output, spec["case_id"])
        with np.load(final / "products.npz") as products, np.load(final / "native_geometry.npz") as geometry:
            assert products["phase_labels"].shape == (184, 146)  # the lattice frame stays on its lattice
            for name, identity in receipt["lattice_inputs"]["products"].items():
                assert _array_identity(products[name]) == identity  # the reduced arrays are the published ones
            assert receipt["lattice_inputs"]["native_geometry"] == document["static"]["arrays"]
            layer = ls.stored_layer(dict(stored_world["config"]["storage"], precision="float64"),
                                    computing_lattice([0.0, 0.146, 0.0, 0.184], 1.0))
            for component, density in DENSITIES.items():
                cells = stored["cell_mass_" + component + "_g"]
                lattice_mass = geometry["transport_weight_l"] * products[density]
                assert cells.shape == (92, 73)
                np.testing.assert_allclose(cells, blocks(layer, lattice_mass).sum(axis=(1, 3)), rtol=1e-12, atol=0)
                assert math.isclose(math.fsum(cells.ravel()), document[TOTALS[component]], rel_tol=1e-6)
                assert receipt["mass_conservation"][component]["passed"]
        assert document["total_mass_g"] > 0
        lr.validate_lattice_frame(spec, document, final, config_of(stored_world))
        missing = deepcopy(document)
        missing["prediction"]["pipeline"].pop("stored")
        with pytest.raises(ValueError, match="lacks its stored products"):
            lr.validate_lattice_frame(spec, missing, final, config_of(stored_world))
        native = dict(config_of(stored_world), storage=dict(stored_world["config"]["storage"], resolution_mm=1.0))
        with pytest.raises(ValueError, match="no coarser stored products"):
            lr.validate_lattice_frame(spec, document, final, native)


def test_an_interrupted_stored_run_resumes_with_the_same_stored_products(stored_world):
    reference, _ = run(stored_world, "uninterrupted")
    FakeInputs.fail_on = "frame_2"
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        run(stored_world, "first")
    FakeInputs.fail_on = None
    first = stored_world["root"] / "first"
    proof = checkpoint.retirement_document(stored_world["task"], first, [dict(pid=1)], dict(kind="test"))
    (first / "retirement.json").write_text(json.dumps(proof), encoding="utf-8")
    task = dict(stored_world["task"], resume_from=dict(attempt_directory=str(first),
                                                       retirement=descriptor(first / "retirement.json")))
    resumed, output = run(stored_world, "second", task)
    assert resumed["reused_case_ids"] == ["frame_1"] and resumed["stored"] == reference["stored"]
    for case, attempt in (("frame_1", first), ("frame_2", output)):
        actual, expected = stored_arrays(attempt, case), stored_arrays(stored_world["root"] / "uninterrupted", case)
        assert actual.keys() == expected.keys()
        for name in expected:
            np.testing.assert_array_equal(actual[name], expected[name])
