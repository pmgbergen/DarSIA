"""Independent image choices retain old records and invalidate changed bases."""

import pytest

from darsia.fluidflower import image_normalisation as method


@pytest.mark.parametrize("lighting", [method.OFF, method.MATCH_REFERENCE])
@pytest.mark.parametrize("color", [method.COLOR_OFF, method.COLOR_DARSIA, method.COLOR_UNIFORM24])
def test_independent_corrections_round_trip(lighting, color):
    reference = "run-b" if lighting != method.OFF or color != method.COLOR_OFF else None
    record = method.choice_record("another-series", ["run-a", "run-b"], lighting,
                                  reference, color_method=color)
    loaded = method.saved_choice({"image_normalisation": record},
                                 "another-series", ["run-a", "run-b"])
    assert loaded == record
    assert method.selected_corrections(loaded) == {
        "lighting": lighting != method.OFF, "color_method": color}


def test_legacy_records_keep_their_actual_native_method():
    for lighting, color in [(method.OFF, method.COLOR_OFF),
                            (method.MATCH_REFERENCE, method.COLOR_SAVED_NATIVE)]:
        record = method.choice_record("other", ["r"], lighting,
                                      "r" if lighting != method.OFF else None)
        loaded = method.saved_choice({"image_normalisation": record}, "other", ["r"])
        assert loaded == record and "color_method" not in loaded
        assert method.selected_corrections(loaded)["color_method"] == color


def test_color_only_requires_reference_and_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="reference"):
        method.choice_record("other", ["r"], method.OFF,
                             color_method=method.COLOR_UNIFORM24)
    with pytest.raises(ValueError, match="ColorChecker"):
        method.choice_record("other", ["r"], method.OFF, color_method="invented")


def test_saving_color_only_preserves_geometry_and_historical_model(tmp_path):
    import tomllib
    original = (b'series_id = "other"\n[runs]\nids = ["r"]\n'
                b'[geometry]\nreference = "unchanged"\n'
                b'[phase_normalisation]\nlegacy_record = "unchanged"\n')
    (tmp_path / "series.toml").write_bytes(original)
    saved = method.save_choice(tmp_path, "other", ["r"], method.OFF, "r",
                               expected_bytes=original, color_method=method.COLOR_UNIFORM24)
    value = tomllib.loads(saved.decode())
    assert value["geometry"] == {"reference": "unchanged"}
    assert value["phase_normalisation"] == {"legacy_record": "unchanged"}
    assert method.selected_corrections(value["image_normalisation"]) == {
        "lighting": False, "color_method": method.COLOR_UNIFORM24}
