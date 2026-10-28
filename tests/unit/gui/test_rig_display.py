"""Synthetic identity metadata only; no geometry or numerical calculations."""

from types import SimpleNamespace

import pytest

from darsia.fluidflower.rig_display import read_rig_display


def write_series(root, *, key="sample", profile=None, rig_class="medium"):
    root.mkdir(exist_ok=True)
    text = f'series_id = "{key}"\nrig_class = "{rig_class}"\n'
    if profile is not None:
        text += f'rig_profile = "{profile}"\n'
    (root / "series.toml").write_text(text, encoding="utf-8")


def catalogue(kind):
    assert kind == "rig"
    return {"FF AC": SimpleNamespace(width=0.90, height=0.49)}


def test_explicit_start_profile_uses_existing_named_profile_without_writes(tmp_path):
    write_series(tmp_path, profile="FF AC")
    before = (tmp_path / "series.toml").read_bytes()
    display = read_rig_display(tmp_path, "sample", catalogue_loader=catalogue)
    assert display.line == "Rig: Albus (FF AC, 0.90 × 0.49 m) — chosen at Start"
    assert display.profile == "FF AC" and display.physical_rig == "Albus"
    assert (tmp_path / "series.toml").read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["series.toml"]


def test_bilbo_preserves_selected_run_dimensions_instead_of_inventing_defaults(
    tmp_path,
):
    write_series(tmp_path, profile="FF Bilbo")
    display = read_rig_display(
        tmp_path,
        "sample",
        loaded_config={"rig": {"width": 0.92, "height": 0.55}},
        catalogue_loader=lambda kind: pytest.fail("Use provided existing dimensions"),
    )
    assert "Bilbo (FF Bilbo, 0.92 × 0.55 m)" in display.line
    assert "selected run" in display.note


@pytest.mark.parametrize("registered_id", ["FF_AC", "ffac"])
def test_known_legacy_ff_ac_identity_does_not_claim_historical_start_choice(
    tmp_path, registered_id
):
    write_series(tmp_path, key="FF_AC")
    display = read_rig_display(tmp_path, registered_id, catalogue_loader=catalogue)
    assert display.physical_rig == "Albus"
    assert "chosen at Start" not in display.line
    assert "explicitly identified by the project owner" in display.note


def test_same_dimensions_never_infer_physical_identity(tmp_path):
    write_series(tmp_path)
    display = read_rig_display(
        tmp_path,
        "sample",
        loaded_config={"rig": {"width": 0.9, "height": 0.49}},
        catalogue_loader=lambda kind: pytest.fail("No named profile was selected"),
    )
    assert not display.profile and not display.physical_rig
    assert "physical rig not recorded" in display.line
    assert "0.90 × 0.49 m" in display.line


def test_explicit_common_path_reads_only_that_metadata(tmp_path):
    write_series(tmp_path)
    common = tmp_path / "shared.toml"
    common.write_text("[rig]\nwidth=1.25\nheight=0.85\n", encoding="utf-8")
    with (tmp_path / "series.toml").open("a", encoding="utf-8") as stream:
        stream.write('[paths]\ncommon_config="shared.toml"\n')
    display = read_rig_display(tmp_path, "sample")
    assert display.width == 1.25 and display.height == 0.85
    assert "common configuration" in display.note


def test_explicit_missing_profile_does_not_guess_another_profile(tmp_path):
    write_series(tmp_path, profile="Unlisted physical rig")
    display = read_rig_display(tmp_path, "sample", catalogue_loader=catalogue)
    assert display.profile == "Unlisted physical rig"
    assert display.physical_rig == ""
    assert "dimensions not recorded" in display.line


@pytest.mark.parametrize("width", [0, -1, True, float("nan"), "0.9"])
def test_invalid_actual_dimensions_remain_unreported_not_replaced_by_catalogue(
    tmp_path, width
):
    write_series(tmp_path, profile="FF AC")
    display = read_rig_display(
        tmp_path,
        "sample",
        loaded_config={"rig": {"width": width, "height": 0.49}},
        catalogue_loader=lambda kind: pytest.fail("Do not mask invalid config values"),
    )
    assert display.width is None
    assert "dimensions not recorded" in display.line


def test_missing_and_mismatched_series_are_explicit(tmp_path):
    assert read_rig_display(None).line == "Rig: no series selected"
    assert read_rig_display(tmp_path).line == "Rig: identity not recorded"
    write_series(tmp_path, key="other")
    assert "does not match" in read_rig_display(tmp_path, "sample").line
