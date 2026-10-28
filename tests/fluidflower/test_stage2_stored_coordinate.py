"""Real stored-artifact to stage-2 composition; all scientific values are fixtures."""

import hashlib
import json
import math
from dataclasses import replace

import numpy as np
import pytest

from darsia.fluidflower.stage2_postprocessing import (
    COLOUR_COORDINATE_ARCHIVE,
    COLOUR_COORDINATE_ARRAY,
    COLOUR_COORDINATE_FORMAT,
    BudgetTerms,
    Stage2Config,
)
from darsia.fluidflower.stage2_stored_coordinate import (
    StoredCoordinateFrame,
    process_coordinate_cycle,
    process_coordinate_frame,
    read_coordinate_inputs,
)
from darsia.fluidflower.titration import M_CO2_G_PER_MOL


@pytest.fixture
def config():
    # Test choices only. The workflow has no defaults for any of these fields.
    return Stage2Config(
        naoh_g_per_l=0.05,
        btb_g_per_l=0.452,
        btb_form="sodium_salt",
        pka_indicator=7.10,
        temperature_c=25.0,
        blue_fraction_floor=1e-9,
        clip_negative_dic=True,
        f_min=0.05,
        p_co2_atm=1.0,
        quantified_labels=(2,),
        injection_gas_density_kg_per_m3=1.872,
        free_gas_density_kg_per_m3=1.842,
        water_density_kg_per_m3=997.0,
        gas_density_reference="test injection15C/rig20C at1atm",
    )


def _stored(tmp_path, name, coordinate):
    """Create an actual SHA-bound stage-1 coordinate artifact, without a photograph."""
    colour = np.asarray(coordinate, dtype=np.float32)
    directory = tmp_path / name
    directory.mkdir()
    archive = directory / COLOUR_COORDINATE_ARCHIVE
    np.savez(archive, **{COLOUR_COORDINATE_ARRAY: colour})
    data = archive.read_bytes()
    record = dict(
        format=COLOUR_COORDINATE_FORMAT,
        path=COLOUR_COORDINATE_ARCHIVE,
        array=COLOUR_COORDINATE_ARRAY,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        identity=dict(dtype=colour.dtype.str, shape=list(colour.shape)),
    )
    (directory / "result.json").write_text(
        json.dumps(dict(case_id=name, colour_coordinate=record)), encoding="utf-8"
    )
    return StoredCoordinateFrame(
        frame_id=name,
        frame_directory=directory,
        labels=np.full(colour.shape, 2, dtype=np.uint8),
        pore_volume_m3=np.full(colour.shape, 1e-6),
        valid_colour_mask=None,
        gas_mask=None,
        outside_mask=None,
        pixel_area_m2=np.full(colour.shape, 1e-6),
    )


def _calibration(colour):
    """Deliberately simple test calibration, never a proposed series calibration."""
    return 1.0 - colour.astype(np.float64)


def test_actual_coordinate_drives_titration_masks_and_regions(tmp_path, config):
    stored = _stored(tmp_path, "t1", [[0.25, 0.99, np.nan], [0.5, 0.8, 0.4]])
    stored = replace(
        stored,
        labels=np.array([[2, 2, 2], [1, 2, 2]], dtype=np.uint8),
        gas_mask=np.array([[False, False, False], [False, True, False]]),
        outside_mask=np.array([[False, False, False], [False, False, True]]),
    )
    region = np.array([[True, False, False], [False, False, False]])
    result = process_coordinate_frame(
        config, stored, adapter=_calibration, regions={"measured": region}
    )
    assert result.summary.total.quantified_pixels == 2
    assert result.summary.total.saturated_pixels == 1
    assert result.not_quantified.tolist() == [[False, False, True], [True, True, True]]
    assert np.isnan(result.dic_mol_per_l[0, 1:]).all()
    expected_dic = config.titration_model().dic_from_blue_fraction(0.75)
    expected_g = expected_dic * 1000.0 * 1e-6 * M_CO2_G_PER_MOL
    assert result.summary.by_region["measured"].mass_lower_g == pytest.approx(expected_g)
    assert result.summary.by_region["measured"].mass_upper_g == pytest.approx(expected_g)
    assert result.mass_upper_g[0, 1] > result.mass_lower_g[0, 1]


def test_stored_cycle_uses_actual_baseline_and_preserves_unknown_budget_terms(
    tmp_path, config
):
    baseline = _stored(tmp_path, "baseline", [[0.2, np.nan]])
    current = _stored(tmp_path, "t1", [[0.5, np.nan]])
    terms = BudgetTerms(
        injection_intervals=((10.0, 2.0), (5.0, 1.0)),
        displaced_water_g=None,
        left_overflow_g=None,
        quality_flags=("test missing scale",),
    )
    result = process_coordinate_cycle(
        config,
        cycle_run="ordinary_cycle",
        baseline=baseline,
        frames=iter(((current, terms),)),
        adapter=_calibration,
    )
    row = result.rows[0]
    model = config.titration_model()
    background = float(model.dic_from_blue_fraction(1.0 - np.float32(0.2)))
    current_dic = float(model.dic_from_blue_fraction(0.5))
    grams_per_molar = 1000.0 * 1e-6 * M_CO2_G_PER_MOL
    assert row["background_dissolved_g"] == pytest.approx(background * grams_per_molar)
    assert row["dissolved_new_lower_g"] == pytest.approx(
        (current_dic - background) * grams_per_molar
    )
    assert row["injected_g"] == pytest.approx(25.0 * 1e-3 * 1.872)
    assert not row["free_gas_known"] and not row["overflow_known"]
    assert math.isnan(row["free_gas_g"]) and math.isnan(row["left_overflow_g"])
    assert row["quality_flags"] == "test missing scale"
    assert row["gas_density_reference"] == config.gas_density_reference
    assert result.baseline.frame_id == "baseline"
    assert result.frames[0].frame_id == "t1"


def test_cycle_reads_baseline_then_each_frame_when_the_generator_yields_it(
    tmp_path, config
):
    baseline = _stored(tmp_path, "baseline", [[0.1]])
    events = []

    def calibration(colour):
        events.append("calibrate")
        return _calibration(colour)

    def frames():
        assert events == ["calibrate"]
        first = _stored(tmp_path, "t1", [[0.2]])
        terms = BudgetTerms(
            injection_intervals=((1.0, 1.0),),
            displaced_water_g=0.0,
            left_overflow_g=0.0,
        )
        yield first, terms
        assert events == ["calibrate", "calibrate"]
        second = _stored(tmp_path, "t2", [[0.3]])
        yield second, terms
        assert events == ["calibrate", "calibrate", "calibrate"]

    result = process_coordinate_cycle(
        config,
        cycle_run="streamed_cycle",
        baseline=baseline,
        frames=frames(),
        adapter=calibration,
    )
    assert [row["frame_id"] for row in result.rows] == ["t1", "t2"]


def test_tampered_coordinate_is_refused_before_the_calibration(tmp_path):
    stored = _stored(tmp_path, "tampered", [[0.2]])
    archive = stored.frame_directory / COLOUR_COORDINATE_ARCHIVE
    archive.write_bytes(archive.read_bytes() + b"changed")

    def must_not_run(colour):
        pytest.fail("changed source reached calibration")

    with pytest.raises(ValueError, match="SHA-256"):
        read_coordinate_inputs(stored, adapter=must_not_run)


@pytest.mark.parametrize(
    "calibration,match",
    [
        (lambda colour: np.full((2, 1), 0.5), "coordinate grid"),
        (lambda colour: np.nan_to_num(colour, nan=0.5), "invalid coordinate support"),
        (lambda colour: np.full(colour.shape, 2.0), "outside"),
    ],
)
def test_calibration_cannot_change_grid_range_or_restore_missing_colour(
    tmp_path, calibration, match
):
    stored = _stored(tmp_path, "sample", [[0.2, np.nan]])
    with pytest.raises(ValueError, match=match):
        read_coordinate_inputs(stored, adapter=calibration)


def test_invalid_support_is_captured_before_a_calibration_mutates_its_input(tmp_path):
    stored = _stored(tmp_path, "sample", [[0.2, np.nan]])

    def repair_in_place(colour):
        colour[np.isnan(colour)] = 0.5
        return colour

    with pytest.raises(ValueError, match="invalid coordinate support"):
        read_coordinate_inputs(stored, adapter=repair_in_place)


def test_original_frame_grid_and_cycle_contracts_still_apply(tmp_path, config):
    baseline = _stored(tmp_path, "baseline", [[0.1]])
    current = _stored(tmp_path, "t1", [[0.2, 0.3]])
    with pytest.raises(ValueError, match="label grid"):
        read_coordinate_inputs(
            replace(current, labels=np.full((1, 1), 2, np.uint8)),
            adapter=_calibration,
        )
    terms = BudgetTerms(
        injection_intervals=((1.0, 1.0),),
        displaced_water_g=None,
        left_overflow_g=None,
    )
    with pytest.raises(ValueError, match="baseline grid"):
        process_coordinate_cycle(
            config,
            cycle_run="wrong_grid",
            baseline=baseline,
            frames=((current, terms),),
            adapter=_calibration,
        )
    with pytest.raises(ValueError, match="at least one frame"):
        process_coordinate_cycle(
            config,
            cycle_run="empty",
            baseline=baseline,
            frames=(),
            adapter=_calibration,
        )


def test_a_cycle_requires_explicit_budget_terms_for_each_stored_frame(tmp_path, config):
    baseline = _stored(tmp_path, "baseline", [[0.1]])
    current = _stored(tmp_path, "t1", [[0.2]])
    with pytest.raises(ValueError, match="BudgetTerms"):
        process_coordinate_cycle(
            config,
            cycle_run="missing_terms",
            baseline=baseline,
            frames=((current, None),),
            adapter=_calibration,
        )
    with pytest.raises(TypeError, match="adapter"):
        process_coordinate_frame(config, current)
    with pytest.raises(ValueError, match="adapter"):
        read_coordinate_inputs(current, adapter=None)
