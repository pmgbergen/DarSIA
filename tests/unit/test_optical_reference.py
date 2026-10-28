import numpy as np
import pytest

from darsia.fluidflower.optical_reference import (
    frame_facing_sand_boundary, stable_band_mask, reference_samples, propose_reference,
)


def test_band_starts_at_sand_boundary_with_anisotropic_physical_distances():
    sand = np.zeros((15, 20), bool)
    sand[3:13, 4:17] = True
    sand[7:9, 10:12] = False  # Interior object must not gain a band around it.
    band = stable_band_mask(sand, 1.1, (2., 1.))
    assert band[8, 4] and band[8, 5] and not band[8, 6]
    assert band[3, 8] and not band[4, 8]
    assert not band[8, 9] and not band[8, 12]
    assert not np.any(band & ~sand)


def test_water_surface_and_internal_holes_are_not_frame_facing():
    sand = np.ones((15, 20), bool)
    sand[:4] = False
    sand[8:10, 8:10] = False
    water = np.zeros_like(sand)
    water[:4] = True
    boundary = frame_facing_sand_boundary(sand, other_material=water)
    assert not boundary[4, 5:15].any()
    assert boundary[5, 0] and boundary[-1, 5]
    assert not boundary[7:11, 7:11].any()
    band = stable_band_mask(sand, 1., (1., 1.), frame_facing_boundary=boundary)
    assert not band[4, 8] and band[-2, 5]


def samples(run, exposure, cast=(1., 1., 1.), interior=0.4):
    rgb = np.full((24, 24, 3), interior)
    mask = np.zeros((24, 24), bool)
    mask[:, :4] = True
    rgb[mask] = .3 * exposure * np.asarray(cast)
    chart = np.full((24, 3), .3) * exposure * np.asarray(cast)
    return reference_samples(run, rgb, mask, chart_rgb=chart, grid_shape=(6, 6))


def test_medoid_is_central_optical_reference_and_interior_co2_has_no_effect():
    cohort = [samples("dark", .6), samples("middle", 1.), samples("bright", 1.6)]
    result = propose_reference(cohort)
    assert result["reference_run"] == "middle"
    changed = [samples("dark", .6, interior=.02),
               samples("middle", 1., interior=.9), samples("bright", 1.6, interior=.07)]
    assert propose_reference(changed) == result
    assert result["selected"] is False


def test_independent_color_proposal_sees_cast_even_with_unchanged_lighting():
    cohort = [samples("green", 1., (1., 1.6, 1.)),
              samples("neutral", 1.), samples("red", 1., (1.6, 1., 1.))]
    assert propose_reference(cohort, use_lighting=False)["reference_run"] == "neutral"
    assert all(row["mean_log_correction"] > 0 for row in propose_reference(cohort)["ranking"])


def test_missing_chart_or_stable_support_never_produces_an_automatic_choice():
    a = samples("a", 1.)
    b = reference_samples("b", np.full((24, 24, 3), .4), np.zeros((24, 24), bool),
                          grid_shape=(6, 6))
    with pytest.raises(ValueError, match="stable"):
        propose_reference([a, b], use_color=False)
    with pytest.raises(ValueError, match="ColorChecker"):
        propose_reference([a, b], use_lighting=False)


def _part_means(cohort):
    """Each run's mean sand and ColorChecker distance to the other runs, computed here directly."""
    count = len(cohort)
    light, colour = np.zeros((count, count)), np.zeros((count, count))
    for i, left in enumerate(cohort):
        for j, right in enumerate(cohort):
            if i != j:
                common = left.stable_valid & right.stable_valid
                delta = np.log(left.stable_rgb[common]) - np.log(right.stable_rgb[common])
                light[i, j] = np.median(np.abs(delta.mean(axis=1)))
                delta = np.log(left.chart_rgb) - np.log(right.chart_rgb)
                colour[i, j] = np.median(np.linalg.norm(delta, axis=1) / np.sqrt(3))
    return ({row.run: light[i].sum() / (count - 1) for i, row in enumerate(cohort)},
            {row.run: colour[i].sum() / (count - 1) for i, row in enumerate(cohort)},
            {row.run: ((light[i] + colour[i]) / 2) for i, row in enumerate(cohort)})


def test_each_row_gives_its_mean_sand_and_colourchecker_distance_and_keeps_the_old_keys():
    cohort = [samples("dark", .6, (1., 1.2, 1.)), samples("middle", 1.),
              samples("bright", 1.6, (1.3, 1., 1.)), samples("cool", .9, (1., 1., 1.4))]
    light, colour, costs = _part_means(cohort)
    result = propose_reference(cohort)
    assert set(result) == {"format", "reference_run", "method", "ranking", "uses_lighting", "uses_color", "selected"}
    assert result["reference_run"] == result["ranking"][0]["run"]
    keys = [(row["mean_log_correction"], row["maximum_log_correction"], row["run"]) for row in result["ranking"]]
    assert keys == sorted(keys)
    for row in result["ranking"]:
        assert list(row) == ["run", "mean_log_correction", "maximum_log_correction",
                             "mean_light_correction", "mean_colour_correction"]
        assert row["mean_light_correction"] == pytest.approx(light[row["run"]], rel=1e-12)
        assert row["mean_colour_correction"] == pytest.approx(colour[row["run"]], rel=1e-12)
        # The old keys: the mean and the largest of the pairwise means of the parts.
        assert row["mean_log_correction"] == pytest.approx(costs[row["run"]].sum() / 3, rel=1e-12)
        assert row["maximum_log_correction"] == pytest.approx(costs[row["run"]].max(), rel=1e-12)
    only_light = propose_reference(cohort, use_color=False)["ranking"]
    only_colour = propose_reference(cohort, use_lighting=False)["ranking"]
    assert all("mean_colour_correction" not in row and row["mean_light_correction"] == row["mean_log_correction"]
               for row in only_light)
    assert all("mean_light_correction" not in row and row["mean_colour_correction"] == row["mean_log_correction"]
               for row in only_colour)
