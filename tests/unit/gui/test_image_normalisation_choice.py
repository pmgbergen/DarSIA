"""Named choices do not alter saved correction inputs or open numerical gates."""

from copy import deepcopy
import datetime as dt
import getpass
import tomllib

import pytest

from darsia.fluidflower import image_normalisation as method
from darsia.fluidflower.image_normalisation_step import ImageNormalisationChoice


def manifest(tmp_path, series="SERIES_X", runs=("x1", "x2")):
    raw = ('series_id = "' + series + '"\n[runs]\nids = ' + repr(list(runs)).replace("'", '"')
           + '\n[phase_normalisation]\nformat = "existing-saved-method"\nunknown = 43\n'
             '[method.phase_model]\nmodel_version = "saved-classifier"\n').encode()
    (tmp_path / "series.toml").write_bytes(raw)
    return raw


def test_named_choice_retains_saved_method_and_is_not_numerical_admission(tmp_path):
    raw = manifest(tmp_path)
    saved = method.save_choice(tmp_path, "SERIES_X", ("x1", "x2"), method.MATCH_REFERENCE,
                               "x2", expected_bytes=raw)
    before = tomllib.loads(raw.decode())
    after = tomllib.loads(saved.decode())
    assert after["phase_normalisation"] == before["phase_normalisation"]
    assert after["method"] == before["method"]
    assert after["image_normalisation"]["reference_run"] == "x2"
    assert after["image_normalisation"]["numerical_status"] == "not_run"
    assert "ranking" not in after["image_normalisation"]
    decision = after["image_normalisation"]["decision"]
    assert decision["actor"] == getpass.getuser()
    assert dt.datetime.fromisoformat(decision["approved_utc"]).utcoffset() == dt.timedelta(0)
    assert decision["decision_ref"] == "image-normalisation-method-choice"
    assert decision["reason"] == "Selected reference lighting from run X2 for this series."


def test_pure_choice_read_retains_decision_and_old_absence_without_new_events(monkeypatch):
    def no_event():
        raise AssertionError("Reading or building a choice must not create a human event")
    monkeypatch.setattr(method.getpass, "getuser", no_event)
    legacy = method.choice_record("SERIES_X", ("x1",), method.OFF)
    assert "decision" not in legacy
    assert method.saved_choice(dict(image_normalisation=legacy), "SERIES_X", ("x1",)) == legacy
    decision = dict(actor="reviewer", approved_utc="2026-10-23T00:00:00+00:00",
                    reason="Selected Off for image normalisation in this series.",
                    decision_ref=method.CHOICE_DECISION_REF)
    saved = method.choice_record("SERIES_X", ("x1",), method.OFF, decision=decision)
    assert method.saved_choice(dict(image_normalisation=saved), "SERIES_X", ("x1",))["decision"] == decision
    assert saved["decision"] is not decision


@pytest.mark.parametrize("field,value", [("actor", ""), ("approved_utc", "2026-10-23T00:00:00"),
                                         ("decision_ref", "handoff-N497"), ("reason", "")])
def test_saved_human_choice_requires_ordinary_complete_decision_fields(field, value):
    decision = dict(actor="reviewer", approved_utc="2026-10-23T00:00:00+00:00",
                    reason="Selected Off for image normalisation in this series.",
                    decision_ref=method.CHOICE_DECISION_REF)
    decision[field] = value
    with pytest.raises(ValueError):
        method.choice_record("SERIES_X", ("x1",), method.OFF, decision=decision)


@pytest.mark.parametrize("choice,reference", [(method.OFF, None), (method.MATCH_REFERENCE, "x1")])
def test_every_series_can_choose_new_corrections_without_rewriting_a_saved_model(choice, reference):
    record = method.choice_record("SERIES_Y", ("x1", "x2"), choice, reference)
    assert record["method"] == choice
    assert record["numerical_status"] == "not_run"


def test_off_has_no_reference_and_stale_save_preserves_external_edit(tmp_path):
    raw = manifest(tmp_path)
    assert "reference_run" not in method.choice_record("SERIES_X", ("x1", "x2"), method.OFF)
    changed = raw + b'\n[external]\nvalue = 1\n'
    (tmp_path / "series.toml").write_bytes(changed)
    with pytest.raises(ValueError, match="series changed"):
        method.save_choice(tmp_path, "SERIES_X", ("x1", "x2"), method.OFF, expected_bytes=raw)
    assert (tmp_path / "series.toml").read_bytes() == changed


def test_saved_choice_with_another_run_list_is_rejected():
    document = dict(image_normalisation=method.choice_record("SERIES_X", ("x1", "x2"), method.OFF))
    original = deepcopy(document)
    with pytest.raises(ValueError, match="differs from this series"):
        method.saved_choice(document, "SERIES_X", ("x1",))
    assert document == original


def test_saved_choice_accepts_the_same_canonical_series_without_rewriting_identity(tmp_path):
    from darsia.fluidflower.mass.n1_pipeline import asset
    from darsia.fluidflower.pipeline.setup_image_preparation import independent_choice
    record = method.choice_record("seriesx", ("x1",), method.MATCH_REFERENCE, "x1",
                                  color_method=method.COLOR_UNIFORM24)
    document = dict(series_id="SERIES_X", runs=dict(ids=["x1"]),
                    method=dict(active="phase_model"), image_normalisation=record)
    assert method.saved_choice(document, "SERIES_X", ("x1",)) == record
    import toml
    path = tmp_path / "series.toml"
    path.write_text(toml.dumps(document), encoding="utf-8")
    original = path.read_bytes()
    assert independent_choice(asset(path))[1] == record
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="differs from this series"):
        method.saved_choice(document, "SERIES_Y", ("x1",))


def test_gui_explicit_choice_persists_without_claiming_fit(qtbot, tmp_path):
    raw = manifest(tmp_path)
    panel = ImageNormalisationChoice()
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "SERIES_X")
    qtbot.waitUntil(panel.can_close)
    assert panel.method.currentIndex() == -1 and not panel.save_button.isEnabled()
    assert "light-only" in panel.status.text()
    assert (tmp_path / "series.toml").read_bytes() == raw
    panel.method.setCurrentIndex(panel.method.findData(method.MATCH_REFERENCE))
    assert not panel.save_button.isEnabled()
    panel.reference.setCurrentIndex(panel.reference.findData("x2"))
    panel.stable_band.setValue(5.)
    panel.save_button.click()
    qtbot.waitUntil(panel.can_close)
    assert "Image settings saved" in panel.status.text()
    assert method.saved_choice(tomllib.loads((tmp_path / "series.toml").read_text()), "SERIES_X", ("x1", "x2"))["reference_run"] == "x2"
    assert all("v5" not in panel.method.itemText(i).casefold() for i in range(panel.method.count()))
    # Sand-type equalisation is now a selectable method.
    assert panel.method.model().item(panel.method.findData(method.EQUALISE_SAND)).isEnabled()


def test_save_feedback_names_choices_and_dirty_state_without_processing(qtbot, tmp_path):
    from threading import Event
    raw = manifest(tmp_path)
    entered, release = Event(), Event()
    def write(*args, **kwargs):
        entered.set()
        release.wait(5)
        return method.save_choice(*args, **kwargs)
    panel = ImageNormalisationChoice(writer=write)
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "SERIES_X")
    qtbot.waitUntil(panel.can_close)
    panel.method.setCurrentIndex(panel.method.findData(method.MATCH_REFERENCE))
    panel.color_method.setCurrentIndex(panel.color_method.findData(method.COLOR_UNIFORM24))
    panel.reference.setCurrentIndex(panel.reference.findData("x2"))
    panel.stable_band.setValue(4.5)
    assert "Unsaved" in panel.status.text()
    assert panel.save_button.text() == "Save image settings"
    panel.save_button.click()
    try:
        qtbot.waitUntil(entered.is_set)
        assert panel.save_button.text() == "Saving…" and not panel.save_button.isEnabled()
        assert "Saving" in panel.status.text()
    finally:
        release.set()
        qtbot.waitUntil(panel.can_close)
    assert panel.save_button.text() == "✓ Image settings saved"
    assert "Uniform24" in panel.status.text() and "X2" in panel.status.text()
    assert "4.5 mm" in panel.status.text() and "No images were processed" in panel.status.text()
    saved = (tmp_path / "series.toml").read_bytes()
    assert saved != raw
    panel.stable_band.setValue(5.)
    assert "Unsaved" in panel.status.text() and panel.save_button.text() == "Save image settings"
    panel.stable_band.setValue(4.5)
    assert panel.save_button.text() == "✓ Image settings saved"
    assert (tmp_path / "series.toml").read_bytes() == saved


def test_saved_model_does_not_lock_the_new_series_choice_or_write_on_navigation(qtbot, tmp_path):
    raw = manifest(tmp_path, "SERIES_Y", ("x1", "x2"))
    panel = ImageNormalisationChoice()
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "SERIES_Y")
    qtbot.waitUntil(panel.can_close)
    assert panel.method.currentIndex() == -1
    assert panel.method.isEnabled()
    assert not panel.save_button.isEnabled()
    assert (tmp_path / "series.toml").read_bytes() == raw


@pytest.mark.parametrize("paths,family", [(["one.ARW"], "raw"), (["one.jpg"], "processed"),
                                         (["one.ARW", "two.jpg"], "mixed"), ([], "unknown")])
def test_source_format_comes_from_saved_setup_selection_without_file_access(paths, family):
    document = dict(image_selection=dict(runs=dict(x1=dict(selected=[dict(source_path=p) for p in paths]))))
    assert method.selected_source_family(document) == family


def test_saved_native_choice_remains_distinct_and_navigation_does_not_save(qtbot, tmp_path):
    raw = manifest(tmp_path)
    raw = method.save_choice(tmp_path, "SERIES_X", ("x1", "x2"), method.MATCH_REFERENCE,
                             "x2", expected_bytes=raw)
    panel = ImageNormalisationChoice()
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "SERIES_X")
    qtbot.waitUntil(panel.can_close)
    assert panel.color_method.currentData() == method.COLOR_SAVED_NATIVE
    assert "existing" in panel.color_method.currentText()
    assert panel.method.currentData() == method.MATCH_REFERENCE
    assert panel.reference.currentData() == "x2"
    assert (tmp_path / "series.toml").read_bytes() == raw
    panel.method.setCurrentIndex(panel.method.findData(method.OFF))
    # Turning light off does not silently switch the separate colour choice.
    assert panel.color_method.currentData() == method.COLOR_SAVED_NATIVE
    assert panel.reference.isEnabled()
    assert not panel.save_button.isEnabled()
    assert "independent color correction" in panel.status.text()
    assert (tmp_path / "series.toml").read_bytes() == raw


def test_light_only_and_both_off_are_independent_explicit_choices(qtbot, tmp_path):
    raw = manifest(tmp_path)
    raw = method.save_choice(tmp_path, "SERIES_X", ("x1", "x2"), method.MATCH_REFERENCE,
                             "x2", expected_bytes=raw)
    panel = ImageNormalisationChoice()
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "SERIES_X")
    qtbot.waitUntil(panel.can_close)
    panel.color_method.setCurrentIndex(panel.color_method.findData(method.COLOR_OFF))
    assert panel.method.currentData() == method.MATCH_REFERENCE
    panel.stable_band.setValue(5.)
    panel.save_button.click()
    qtbot.waitUntil(panel.can_close)
    saved = method.saved_choice(tomllib.loads((tmp_path / "series.toml").read_text()),
                               "SERIES_X", ("x1", "x2"))
    assert method.selected_corrections(saved) == dict(lighting=True, color_method=method.COLOR_OFF)
    assert saved["reference_run"] == "x2"
    panel.method.setCurrentIndex(panel.method.findData(method.OFF))
    assert not panel.reference.isEnabled()
    panel.save_button.click()
    qtbot.waitUntil(panel.can_close)
    saved = method.saved_choice(tomllib.loads((tmp_path / "series.toml").read_text()),
                               "SERIES_X", ("x1", "x2"))
    assert method.selected_corrections(saved) == dict(lighting=False, color_method=method.COLOR_OFF)
    assert "reference_run" not in saved


def test_stable_band_is_explicit_preserved_and_does_not_edit_on_load(qtbot, tmp_path):
    raw = manifest(tmp_path)
    raw = method.save_choice(tmp_path, "SERIES_X", ("x1", "x2"), method.MATCH_REFERENCE,
                             "x2", expected_bytes=raw)
    panel = ImageNormalisationChoice()
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "SERIES_X")
    qtbot.waitUntil(panel.can_close)
    assert panel.stable_band.value() == 0.0
    assert (tmp_path / "series.toml").read_bytes() == raw
    panel.stable_band.setValue(6.25)
    assert not panel.save_button.isEnabled()  # Legacy combined fit must not silently ignore the new band.
    panel.color_method.setCurrentIndex(panel.color_method.findData(method.COLOR_OFF))
    assert (tmp_path / "series.toml").read_bytes() == raw
    panel.save_button.click()
    qtbot.waitUntil(panel.can_close)
    saved_raw = (tmp_path / "series.toml").read_bytes()
    saved = method.saved_choice(tomllib.loads(saved_raw.decode()), "SERIES_X", ("x1", "x2"))
    assert saved["stable_area"] == dict(kind="sand-boundary-inward-band", width_mm=6.25)
    panel.set_context(tmp_path, "SERIES_X")
    qtbot.waitUntil(panel.can_close)
    assert panel.stable_band.value() == 6.25
    assert (tmp_path / "series.toml").read_bytes() == saved_raw
    panel.method.setCurrentIndex(panel.method.findData(method.OFF))
    assert not panel.stable_band.isEnabled()
    assert panel.stable_band.value() == 6.25


@pytest.mark.parametrize("lighting,color", [
    (method.OFF, method.COLOR_DARSIA), (method.MATCH_REFERENCE, method.COLOR_DARSIA),
    (method.OFF, method.COLOR_UNIFORM24), (method.MATCH_REFERENCE, method.COLOR_UNIFORM24),
    (method.OFF, method.COLOR_OFF), (method.MATCH_REFERENCE, method.COLOR_OFF),
])
def test_independent_producer_choices_are_enabled_and_saved(qtbot, tmp_path, lighting, color):
    manifest(tmp_path)
    panel = ImageNormalisationChoice()
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "SERIES_X")
    qtbot.waitUntil(panel.can_close)
    index = panel.color_method.findData(color)
    assert panel.color_method.model().item(index).isEnabled()
    panel.method.setCurrentIndex(panel.method.findData(lighting))
    panel.color_method.setCurrentIndex(index)
    panel.stable_band.setValue(5)
    needs_reference = lighting == method.MATCH_REFERENCE or color != method.COLOR_OFF
    assert panel.reference.isEnabled() is needs_reference
    if needs_reference:
        panel.reference.setCurrentIndex(panel.reference.findData("x2"))
    assert panel.save_button.isEnabled()
    panel.save_button.click()
    qtbot.waitUntil(panel.can_close)
    saved = method.saved_choice(tomllib.loads((tmp_path / "series.toml").read_text()),
                               "SERIES_X", ("x1", "x2"))
    assert method.selected_corrections(saved) == dict(lighting=lighting == method.MATCH_REFERENCE,
                                                     color_method=color)
    assert saved.get("reference_run") == ("x2" if needs_reference else None)

def test_gui_equalise_sand_saves_its_band_left_out_including_none(qtbot, tmp_path):
    raw = manifest(tmp_path)
    panel = ImageNormalisationChoice()
    qtbot.addWidget(panel)
    panel.set_context(tmp_path, "SERIES_X")
    qtbot.waitUntil(panel.can_close)
    panel.method.setCurrentIndex(panel.method.findData(method.EQUALISE_SAND))
    assert panel.method.model().item(panel.method.currentIndex()).isEnabled()
    assert "left out" in panel.band_label.text() and panel.stable_band.isEnabled()
    panel.color_method.setCurrentIndex(panel.color_method.findData(method.COLOR_UNIFORM24))
    panel.reference.setCurrentIndex(panel.reference.findData("x2"))
    assert panel.save_button.isEnabled()  # 0 mm leaves nothing out
    assert panel.stable_band.specialValueText().startswith("None")
    panel.stable_band.setValue(40.)
    panel.save_button.click()
    qtbot.waitUntil(panel.can_close)
    saved = method.saved_choice(tomllib.loads((tmp_path / "series.toml").read_text()), "SERIES_X", ("x1", "x2"))
    assert saved["exclusion_band"]["width_mm"] == 40.0 and "stable_area" not in saved
    assert "band left out: 40 mm" in panel.status.text()
    assert (tmp_path / "series.toml").read_bytes() != raw
