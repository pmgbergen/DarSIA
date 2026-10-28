"""Fresh-plan storage choices retain the original scientific input identities."""
from copy import deepcopy

import pytest

from darsia.fluidflower import production_plan_options as options
from darsia.fluidflower import storage_profile
from darsia.fluidflower.storage_outputs import LA_ENCODING, REQUEST_FORMAT


def source():
    def bound(role):
        return dict(path=role + ".json", sha256="a" * 64, size_bytes=71)

    return dict(
        completion_bound=bound("completion"),
        configuration_bound=bound("phase-one-configuration"),
        expected_phase1_commit="c" * 40,
        fitted_calibration=bound("calibration"),
        fitted_anchor=dict(elapsed_h=2.25, affected_pore_volume_m3=0.0001),
        pixel_domain_data=dict(
            pixel_domain=dict(n16=dict(authority=bound("n16-authority"))),
            frames={"case-one": dict(pixel_domain_geometry=bound("geometry"),
                                    frozen_response_basis=bound("basis"))}),
        storage_data=dict(format=REQUEST_FORMAT, resolution_mm=1.0,
                          gas_score_encoding=LA_ENCODING, extent_m=[0., 1., 0., 1.]),
        storage_profile=dict(profile="full", store_colour_coordinate=True),
    )


def test_saved_choices_are_an_independent_exact_copy():
    original = source()
    snapshot = deepcopy(original)
    result = options.phase2_source_options(original)
    assert result == original == snapshot
    result["pixel_domain_data"]["frames"]["case-one"]["frozen_response_basis"]["path"] = "changed"
    result["fitted_anchor"]["elapsed_h"] = 99
    result["storage_data"]["extent_m"][1] = 9
    assert original == snapshot


@pytest.mark.parametrize("step", [0.25, 0.5, 1.0, 2.0])
def test_supported_sizes_change_only_the_stored_resolution(step):
    original = source()
    snapshot = deepcopy(original)
    result = options.phase2_source_options(original, stored_resolution=step)
    expected = deepcopy(original)
    expected["storage_data"]["resolution_mm"] = step
    assert result == expected
    assert original == snapshot
    assert "precision" not in result["storage_data"]
    assert "computing_resolution_mm" not in result["storage_data"]


@pytest.mark.parametrize("present", [False, True])
def test_native_keeps_absent_or_none_request(present):
    original = source()
    original.pop("storage_data")
    if present:
        original["storage_data"] = None
    assert options.phase2_source_options(original, stored_resolution="native") == original
    assert options.phase2_source_options(original) == original


def test_native_preserves_saved_extent_and_encoding():
    original = source()
    expected = deepcopy(original)
    expected["storage_data"]["resolution_mm"] = "native"
    assert options.phase2_source_options(original, stored_resolution="native") == expected


@pytest.mark.parametrize("storage_request", [None, {}, dict(format=REQUEST_FORMAT,
    resolution_mm="native", gas_score_encoding=LA_ENCODING)])
def test_coarse_size_never_invents_a_request_or_extent(storage_request):
    original = source()
    if storage_request is None:
        original.pop("storage_data")
    else:
        original["storage_data"] = storage_request
    snapshot = deepcopy(original)
    with pytest.raises(ValueError):
        options.phase2_source_options(original, stored_resolution=0.5)
    assert original == snapshot


@pytest.mark.parametrize("extent", [None, [], [0., 1., 0.], [0., 0., 0., 1.],
    [0., float("nan"), 0., 1.], [0., 1.0001, 0., 1.], [False, 1., 0., 1.]])
def test_malformed_or_off_lattice_extent_is_not_repaired(extent):
    original = source()
    original["storage_data"]["extent_m"] = extent
    with pytest.raises(ValueError):
        options.phase2_source_options(original, stored_resolution=0.5)


def test_new_resolution_must_also_fit_the_unchanged_extent():
    original = source()
    original["storage_data"].update(resolution_mm=0.25, extent_m=[0., 0.00125, 0., 0.00125])
    assert options.phase2_source_options(original) == original
    with pytest.raises(ValueError, match="exact origin-zero metric lattice"):
        options.phase2_source_options(original, stored_resolution=0.5)


@pytest.mark.parametrize("step", [None, True, False, 0., 0.75, "0.5", "coarse", float("inf")])
def test_unsupported_resolution_is_refused(step):
    with pytest.raises(ValueError):
        options.phase2_source_options(source(), stored_resolution=step)


@pytest.mark.parametrize("field,value", [("format", "unknown"), ("gas_score_encoding", "invented"),
    ("resolution_mm", 0.75), ("computing_resolution_mm", 0.5)])
def test_override_cannot_hide_invalid_original_storage(field, value):
    original = source()
    original["storage_data"][field] = value
    with pytest.raises(ValueError):
        options.phase2_source_options(original, stored_resolution="native")


@pytest.mark.parametrize("chosen", ["full", "lean"])
@pytest.mark.parametrize("coordinate", [None, False, True])
def test_explicit_profile_preserves_coordinate_and_all_scientific_sources(chosen, coordinate):
    original = source()
    original["storage_profile"] = dict(profile="full")
    if coordinate is not None:
        original["storage_profile"]["store_colour_coordinate"] = coordinate
    snapshot = deepcopy(original)
    result = options.phase2_source_options(original, storage_profile=chosen, stored_resolution=0.5)
    expected = deepcopy(original)
    expected["storage_profile"].update(profile=chosen, colour_intermediates="keep")
    expected["storage_data"]["resolution_mm"] = 0.5
    assert result == expected and original == snapshot
    assert storage_profile.require_run_choice(chosen, result, pipeline="n1")["profile"] == chosen
    other = "lean" if chosen == "full" else "full"
    with pytest.raises(ValueError, match="but its job chose"):
        storage_profile.require_run_choice(other, result, pipeline="n1")


def test_missing_profile_is_preserved_until_explicitly_chosen():
    original = source()
    original.pop("storage_profile")
    assert options.phase2_source_options(original) == original
    result = options.phase2_source_options(original, storage_profile="lean")
    assert result["storage_profile"] == dict(profile="lean", colour_intermediates="keep")
    assert "storage_profile" not in original


@pytest.mark.parametrize("setting", ["lean", {}, dict(profile="unknown"),
    dict(profile="full", colour_intermediates="drop"), dict(profile="full", unknown=True),
    dict(profile="lean", store_colour_coordinate=1)])
@pytest.mark.parametrize("chosen", [None, "lean"])
def test_profile_override_does_not_repair_invalid_existing_settings(setting, chosen):
    original = source()
    original["storage_profile"] = setting
    with pytest.raises(ValueError):
        options.phase2_source_options(original, storage_profile=chosen)


@pytest.mark.parametrize("chosen", ["full", "lean"])
def test_profile_override_refuses_saved_drop(chosen):
    original = source()
    original["storage_profile"] = dict(profile="lean", colour_intermediates="drop")
    assert options.phase2_source_options(original) == original
    with pytest.raises(ValueError, match="KEEP"):
        options.phase2_source_options(original, storage_profile=chosen)


@pytest.mark.parametrize("chosen", [True, {}, "unknown", "LEAN"])
def test_new_profile_uses_original_choice_validation(chosen):
    with pytest.raises(ValueError, match="storage_profile"):
        options.phase2_source_options(source(), storage_profile=chosen)
