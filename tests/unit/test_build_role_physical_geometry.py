"""Numeric mask/field composition; no source or scientific approval."""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess

import numpy as np
import pytest

from darsia.fluidflower.mass import n1_geometry as geometry
from darsia.fluidflower.mass import n1_statistics as statistics


PARENT = "fabf33f814c23980341720a12d96650faa46e75d"


@pytest.fixture
def specimen():
    shape = (3, 6)
    facies = np.tile(np.array([0, 1, 2, 17, 5, 9], np.uint16), (3, 1))
    phi = np.tile(np.array([1., .3, .4, 0., .5, np.nan], np.float32), (3, 1))
    swi = np.tile(np.array([np.nan, .2, .25, np.nan, .3, np.nan], np.float64), (3, 1))
    reporting = np.ones(shape, bool)
    analysis = np.isin(facies, (1, 2, 5))
    material = facies != 9
    material[1, 4] = False  # Finite phi alone must not fill missing source support.
    optical = reporting.copy()
    optical[0, 1] = False
    exclusions = dict(free_water=facies == 0, chart=facies == 17)
    x, y = np.meshgrid(np.linspace(0., .5, 6), np.array([.2, .1, 0.]))
    points = np.array([[0., 0., .010], [.5, 0., .012],
                       [0., .2, .014], [.5, .2, .016]])
    prepared = geometry.PreparedInputs(run="TESTONLY", frame_index=1, elapsed_h=1.,
        current_rgb=np.ones((*shape, 3), np.float32),
        baseline_rgb=np.ones((*shape, 3), np.float32),
        clear_water_rgb=np.ones((*shape, 3), np.float32),
        clear_water_supported_mask=reporting.copy(), phase_labels=np.ones(shape, np.uint8),
        facies=facies, domain=analysis.copy(), supported_mask=optical,
        mass_eligible_mask=np.zeros(shape, bool), bottom_strip_mask=np.zeros(shape, bool),
        below_photo_filled_mask=np.zeros(shape, bool), unknown_physical_mask=~material,
        unquantified_facies_mask=~analysis, depth_m=np.full(shape, .999), porosity=phi,
        pixel_area_m2=np.full(shape, 1e-6), recipe={}, atmosphere={},
        nominal_injected_mass_g=0., dimensions_m=(.2, .5), provenance={"scope": "numeric specimen"})
    kwargs = dict(x_m=x, y_m=y, floor_curve_canvas_y=np.full(6, 2.), depth_points=points,
        facies=facies, porosity=phi, swi=swi, analysis_domain=analysis,
        reporting_domain=reporting, material_supported_mask=material,
        excluded_masks=exclusions, source_binding={"scope": "TESTONLY unsealed arrays"})
    return prepared, kwargs


def snapshot(value):
    if isinstance(value, np.ndarray):
        return value.dtype.str, value.shape, value.tobytes()
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    return deepcopy(value)


def estimate(value, fields, **changes):
    kwargs = dict(porosity_by_facies={1: .3, 2: .4, 5: .5},
        swi_by_facies={1: .2, 2: .25, 5: .3}, source_binding=fields["source_binding"],
        prepared_fields=fields)
    kwargs.update(changes)
    return statistics.build_estimate_geometry(fields["x_m"], fields["y_m"], value.facies,
                                               fields["depth_points"], **kwargs)


def test_material_roles_full_report_unknowns_and_original_depth_are_preserved(specimen):
    prepared, kwargs = specimen
    before = snapshot(dict(prepared=vars(prepared), kwargs=kwargs))
    value, report, fields = geometry.prepare_build_physical(prepared, **kwargs)
    expected_depth = geometry.measured_depth(kwargs["x_m"], kwargs["y_m"], kwargs["depth_points"])
    assert snapshot(value.facies) == snapshot(kwargs["facies"])
    assert snapshot(value.porosity) == snapshot(kwargs["porosity"])
    assert snapshot(value.domain) == snapshot(kwargs["reporting_domain"])
    assert snapshot(value.depth_m) == snapshot(expected_depth["depth_m"])
    assert np.all(value.facies[:, 0] == 0) and np.all(value.porosity[:, 0] == 1)
    assert np.all(value.facies[:, 3] == 17) and np.all(value.porosity[:, 3] == 0)
    known = kwargs["analysis_domain"] & kwargs["material_supported_mask"]
    np.testing.assert_array_equal(value.unknown_physical_mask, value.domain & ~known)
    np.testing.assert_array_equal(value.mass_eligible_mask, known & prepared.supported_mask)
    assert not value.mass_eligible_mask[1, 4] and not value.mass_eligible_mask[0, 1]
    assert report["known_zero_porosity_pixels"] == 3
    assert report["missing_material_support_pixels"] == 4
    assert report["facies_filled"] is False and report["reporting_domain_changed"] is False
    assert snapshot(dict(prepared=vars(prepared), kwargs=kwargs)) == before
    assert not np.any(value.depth_m == .999)  # Saved Build depth is not the measured-depth method.
    assert set(fields["excluded_masks"]) == {"free_water", "chart"}


def test_estimate_reuses_measured_fields_without_interpolation_or_sand_borrow(specimen, monkeypatch):
    value, _, fields = geometry.prepare_build_physical(specimen[0], **specimen[1])
    before = snapshot(fields)
    def forbidden(*args, **kwargs):
        pytest.fail("Prepared Build estimates must not interpolate or borrow again")
    monkeypatch.setattr(statistics, "LinearNDInterpolator", forbidden)
    monkeypatch.setattr(statistics, "_column_neighbours", forbidden)
    result = estimate(value, fields)
    for name, field in (("depth_m", "depth_m"), ("porosity", "porosity"),
                        ("swi", "swi"), ("facies_for_geometry", "facies"), ("valid", "valid")):
        assert snapshot(getattr(result, name)) == snapshot(fields[field])
    assert not result.porosity_from_column.any()
    assert not result.valid[:, (0, 3, 5)].any() and not result.valid[1, 4]
    assert snapshot(fields) == before


def test_floor_keeps_full_report_and_derives_a_disjoint_exclusion(specimen):
    prepared, kwargs = specimen
    kwargs["floor_curve_canvas_y"][2] = 1.
    prepared.phase_labels[2, 2] = 0
    value, report, fields = geometry.prepare_build_physical(prepared, **kwargs)
    below = np.zeros(value.domain.shape, bool)
    below[2, 2] = True
    np.testing.assert_array_equal(fields["excluded_masks"]["below_sand_floor"], below)
    assert value.domain[2, 2] and not value.mass_eligible_mask[2, 2]
    assert value.unknown_physical_mask[2, 2] and not estimate(value, fields).valid[2, 2]
    assert report["below_sand_floor"]["pixels"] == 1
    np.testing.assert_array_equal(value.phase_labels, prepared.phase_labels)


@pytest.mark.parametrize("fault", ["positive_below", "analysis_overlap", "exclusion_overlap",
    "zero_without_exclusion", "analysis_outside", "nonboolean_support", "phi_invalid", "swi_invalid"])
def test_invalid_role_or_floor_inputs_refuse_without_repair(specimen, fault):
    prepared, kwargs = specimen
    if fault == "positive_below":
        kwargs["floor_curve_canvas_y"][2] = 1.
    elif fault == "analysis_overlap":
        kwargs["excluded_masks"]["chart"][0, 1] = True
    elif fault == "exclusion_overlap":
        kwargs["excluded_masks"]["duplicate"] = kwargs["excluded_masks"]["chart"].copy()
    elif fault == "zero_without_exclusion":
        del kwargs["excluded_masks"]["chart"]
    elif fault == "analysis_outside":
        kwargs["reporting_domain"][0, 1] = False
    elif fault == "nonboolean_support":
        kwargs["material_supported_mask"] = kwargs["material_supported_mask"].astype(np.uint8)
    elif fault == "phi_invalid":
        kwargs["porosity"][0, 1] = -1.
    else:
        kwargs["swi"][0, 1] = np.inf
    before = snapshot(dict(prepared=vars(prepared), kwargs=kwargs))
    with pytest.raises(ValueError):
        geometry.prepare_build_physical(prepared, **kwargs)
    assert snapshot(dict(prepared=vars(prepared), kwargs=kwargs)) == before


@pytest.mark.parametrize("fault", ["x", "facies", "depth_points", "source", "expanded_support", "swi_recipe"])
def test_estimate_rejects_changed_join_or_reintroduced_support(specimen, fault):
    value, _, fields = geometry.prepare_build_physical(specimen[0], **specimen[1])
    kwargs = dict(porosity_by_facies={}, swi_by_facies={1: .2, 2: .25, 5: .3},
                  source_binding=deepcopy(fields["source_binding"]), prepared_fields=fields)
    x, y, facies, points = (fields[key].copy() for key in ("x_m", "y_m", "facies", "depth_points"))
    if fault == "x":
        x[0, 0] += 1
    elif fault == "facies":
        facies[0, 0] = 1
    elif fault == "depth_points":
        points[0, 2] += .001
    elif fault == "source":
        kwargs["source_binding"]["scope"] = "different"
    elif fault == "expanded_support":
        fields["valid"][0, 0] = True
    else:
        kwargs["swi_by_facies"][1] = .4
    with pytest.raises(ValueError):
        statistics.build_estimate_geometry(x, y, facies, points, **kwargs)


@pytest.mark.parametrize("chunk", [1, 2, 250000])
def test_default_estimate_matches_frozen_parent_body(specimen, chunk):
    root = Path(__file__).resolve().parents[2]
    path = "src/darsia/fluidflower/mass/n1_statistics.py"
    source = subprocess.check_output(["git", "show", PARENT + ":" + path], cwd=root).decode("utf-8")
    names = {"_coordinates", "build_estimate_geometry"}
    body = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(np=np, LinearNDInterpolator=statistics.LinearNDInterpolator,
        cKDTree=statistics.cKDTree, _column_neighbours=statistics._column_neighbours,
        EstimateGeometry=statistics.EstimateGeometry)
    exec(compile(ast.Module(body=body, type_ignores=[]), path, "exec"), namespace)
    _, data = specimen
    args = (data["x_m"], data["y_m"], data["facies"], data["depth_points"])
    kwargs = dict(porosity_by_facies={1: .3, 2: .4, 5: .5},
                  swi_by_facies={1: .2, 2: .25, 5: .3}, chunk_pixels=chunk)
    expected = namespace["build_estimate_geometry"](*args, **kwargs)
    actual = statistics.build_estimate_geometry(*args, **kwargs)
    assert snapshot(vars(expected)) == snapshot(vars(actual))
