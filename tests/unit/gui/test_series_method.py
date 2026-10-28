"""Candidate helper/panel tests; import becomes package-qualified on adoption."""

import copy
import tomllib

import pytest

from darsia.fluidflower import series_method as methods


@pytest.fixture
def manifest(tmp_path):
    path = tmp_path / "series.toml"
    path.write_text(
        'schema_version = 1\nseries_id = "demo"\n'
        '[paths]\nraw = "../raw photographs"\n'
        '[runs]\nids = ["ac52", "ac19"]\n'
        "[unrelated]\nflag = true\nlimit = 0.125\n"
        "created = 2026-10-15T12:30:00+02:00\n"
        'nested = { label = "保持", values = [1, 2, 3] }\n',
        encoding="utf-8",
    )
    return path


def test_unset_and_unknown_choice_are_not_chosen_without_writes(manifest):
    before = manifest.read_bytes()
    assert methods.read_series_method(manifest.parent).active == ""
    assert manifest.read_bytes() == before
    manifest.write_bytes(before + b'\n[method]\nactive = "unexpected"\n')
    before = manifest.read_bytes()
    assert methods.read_series_method(manifest.parent).active == ""
    assert manifest.read_bytes() == before


def test_model_settings_describes_only_supported_combinations_without_changing_choices():
    original = methods.MethodSelection(active="colour_path", colour_calibration="original/fit.json")
    phase = methods.MethodSelection(active="phase_model", model_version="models/v10",
                                    mass_calibration="calibration/fit.json")
    assert methods.model_settings_description(original)["quantification"] == "Colour signal + SimpleFlash"
    assert methods.model_settings_description(phase)["quantification"] == "Titration"
    assert "Expert Knowledge" in methods.model_settings_description(phase)["phase_guidance"]
    assert "Not used" in methods.model_settings_description(original)["phase_guidance"]
    assert methods.MethodSelection().active == ""
    assert "Choose" in methods.model_settings_description(methods.MethodSelection())["quantification"]
    assert methods.METHODS == ("colour_path", "phase_model")


def test_common_model_panel_preserves_existing_api_refs_and_bytes(manifest, qtbot):
    from PySide6.QtWidgets import QCheckBox
    saved = methods.MethodSelection(active="phase_model", model_version="模型/<v10>",
        mass_calibration="calibration/fit.json", colour_calibration="legacy/colour.json")
    before = manifest.read_bytes()
    panel = methods.ModelSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_selection(saved)
    assert isinstance(panel, methods.SeriesMethodPanel)
    assert panel.title.text() == "Model settings"
    assert panel.selection() == saved
    assert panel.quantification_model.text() == "Titration"
    assert "Expert Knowledge" in panel.phase_guidance.text()
    assert "cannot currently be selected independently" in panel.combination_note.text()
    assert not panel.findChildren(QCheckBox)
    panel.colour_path_button.click()
    assert panel.selection().active == "colour_path"
    assert panel.quantification_model.text() == "Colour signal + SimpleFlash"
    assert "Not used" in panel.phase_guidance.text()
    assert not panel.model_version.isEnabled() and panel.colour_calibration.isEnabled()
    panel.phase_model_button.click()
    assert panel.selection() == saved
    assert manifest.read_bytes() == before


def test_save_preserves_unrelated_and_unknown_values_and_reload(manifest):
    manifest.write_bytes(
        manifest.read_bytes()
        + (
            '\n[method]\nactive = "colour_path"\nnote = "retain"\n'
            '[method.colour_path]\ncalibration = "old/colour"\nfuture = [4, 5]\n'
            '[method.phase_model]\nmodel_version = "old-model"\n'
            'mass_calibration = "old-fit"\nreviewed = true\n'
        ).encode()
    )
    original = tomllib.loads(manifest.read_text(encoding="utf-8"))
    choice = methods.MethodSelection(
        active="phase_model",
        model_version="模型/版本 α",
        mass_calibration="../calibration/β.json",
        colour_calibration="do not replace",
    )
    saved = methods.save_series_method(manifest.parent, choice)
    after = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert {key: value for key, value in after.items() if key != "method"} == {
        key: value for key, value in original.items() if key != "method"
    }
    assert after["method"]["note"] == "retain"
    assert after["method"]["phase_model"]["reviewed"] is True
    assert after["method"]["colour_path"] == original["method"]["colour_path"]
    assert saved == methods.read_series_method(manifest.parent)
    assert saved.model_version == choice.model_version
    assert saved.mass_calibration == choice.mass_calibration
    assert saved.colour_calibration == "old/colour"
    methods.save_series_method(
        manifest.parent,
        methods.MethodSelection(
            active="colour_path", colour_calibration="relative/new colour"
        ),
    )
    switched = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert switched["method"]["phase_model"] == after["method"]["phase_model"]
    assert switched["method"]["colour_path"]["future"] == [4, 5]


def test_invalid_manifest_and_invalid_choice_do_not_change_bytes(manifest):
    before = manifest.read_bytes()
    with pytest.raises(ValueError, match="Choose"):
        methods.save_series_method(manifest.parent, methods.MethodSelection())
    assert manifest.read_bytes() == before


def test_prepared_model_retains_only_explicit_calibration_ancestry(manifest):
    from dataclasses import replace
    previous = methods.MethodSelection(active="phase_model", model_version="models/old",
                                       mass_calibration="fits/old.json")
    methods.save_series_method(manifest.parent, previous)
    original = tomllib.loads(manifest.read_text(encoding="utf-8"))
    chosen = replace(previous, model_version="models/new", mass_calibration="")
    methods.save_series_method(manifest.parent, chosen, prepared_model=True,
                               expected_bytes=manifest.read_bytes())
    saved = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert {k: v for k, v in saved.items() if k != "method"} == {
        k: v for k, v in original.items() if k != "method"}
    assert methods.read_series_method(manifest.parent) == chosen
    refs = saved["method"]["phase_model"]
    assert refs["previous_mass_calibrations"] == ["fits/old.json"]
    assert methods.calibration_ancestry_reference(refs) == "fits/old.json"
    assert methods.calibration_ancestry_reference(dict(refs, mass_calibration="fits/new.json")) == "fits/new.json"


def test_prepared_model_stale_context_and_invalid_choice_preserve_original(manifest):
    choice = methods.MethodSelection(active="phase_model", model_version="models/new")
    original = manifest.read_bytes()
    with pytest.raises(methods.ManifestChangedError, match="context changed"):
        methods.save_series_method(manifest.parent, choice, prepared_model=True,
                                   context_current=lambda: False)
    assert manifest.read_bytes() == original
    assert not list(manifest.parent.glob(".series.toml.*.tmp"))
    with pytest.raises(ValueError, match="no active mass calibration"):
        methods.save_series_method(manifest.parent,
            methods.MethodSelection(active="phase_model", model_version="new", mass_calibration="old"),
            prepared_model=True)
    assert manifest.read_bytes() == original


def test_prepared_model_save_keeps_an_explicit_calibration_lock(tmp_path):
    from tests.unit.test_series_calibration_approval import bound_products
    from darsia.fluidflower.series_calibration_approval import approve_series_method
    case = bound_products(tmp_path)
    approve_series_method(case.root, case.selection)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match="locked"):
        methods.save_series_method(case.root,
            methods.MethodSelection(active="phase_model", model_version="models/new"),
            prepared_model=True)
    assert {path: path.read_bytes() for path in before} == before


def test_invalid_reference_table_stays_not_chosen_without_losing_data(manifest):
    manifest.write_bytes(
        manifest.read_bytes()
        + b'\n[method]\nactive = "phase_model"\nphase_model = 17\n'
    )
    before = manifest.read_bytes()
    loaded = methods.read_series_method(manifest.parent)
    assert loaded.active == "" and loaded.error
    with pytest.raises(ValueError, match="must be a TOML table"):
        methods.save_series_method(
            manifest.parent,
            methods.MethodSelection(
                active="phase_model", model_version="model", mass_calibration="fit"
            ),
        )
    assert manifest.read_bytes() == before
    manifest.write_bytes(b"invalid = [\n")
    before = manifest.read_bytes()
    loaded = methods.read_series_method(manifest.parent)
    assert loaded.active == "" and loaded.error
    with pytest.raises(ValueError):
        methods.save_series_method(
            manifest.parent,
            methods.MethodSelection(
                active="phase_model", model_version="model", mass_calibration="fit"
            ),
        )
    assert manifest.read_bytes() == before


def test_old_load_and_edit_during_save_are_rejected(manifest, monkeypatch):
    original = manifest.read_bytes()
    manifest.write_bytes(original + b'\nnew_field = "newer"\n')
    newer = manifest.read_bytes()
    choice = methods.MethodSelection(active="colour_path", colour_calibration="fit")
    with pytest.raises(methods.ManifestChangedError, match="loaded"):
        methods.save_series_method(manifest.parent, choice, expected_bytes=original)
    assert manifest.read_bytes() == newer
    dumps = methods.toml.dumps
    concurrent = newer + b'concurrent = "preserve this edit"\n'

    def editing_dumps(value):
        manifest.write_bytes(concurrent)
        return dumps(value)

    monkeypatch.setattr(methods.toml, "dumps", editing_dumps)
    with pytest.raises(methods.ManifestChangedError, match="while saving"):
        methods.save_series_method(manifest.parent, choice)
    assert manifest.read_bytes() == concurrent
    assert not list(manifest.parent.glob(".series.toml.*.tmp"))


def test_replace_failure_and_semantic_roundtrip_failure_preserve_original(
    manifest, monkeypatch
):
    original = manifest.read_bytes()
    choice = methods.MethodSelection(active="colour_path", colour_calibration="fit")

    def denied(_source, _destination):
        raise PermissionError("simulated local manifest replace denial")

    monkeypatch.setattr(methods.os, "replace", denied)
    with pytest.raises(PermissionError, match="denial"):
        methods.save_series_method(manifest.parent, choice)
    assert manifest.read_bytes() == original
    assert not list(manifest.parent.glob(".series.toml.*.tmp"))
    monkeypatch.undo()
    dumps = methods.toml.dumps

    def broken_dumps(value):
        altered = copy.deepcopy(value)
        altered["unrelated"]["limit"] = 999.0
        return dumps(altered)

    monkeypatch.setattr(methods.toml, "dumps", broken_dumps)
    with pytest.raises(ValueError, match="other TOML values"):
        methods.save_series_method(manifest.parent, choice)
    assert manifest.read_bytes() == original


def test_bom_crlf_and_nonfinite_unknown_values_survive(manifest):
    original = manifest.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    manifest.write_bytes(
        b"\xef\xbb\xbf"
        + original
        + b"\r\n[limits]\r\nmissing = nan\r\nmaximum = inf\r\n"
    )
    methods.save_series_method(
        manifest.parent,
        methods.MethodSelection(
            active="phase_model", model_version="model", mass_calibration="fit"
        ),
    )
    after = manifest.read_bytes()
    assert after.startswith(b"\xef\xbb\xbf")
    assert b"\n" not in after.replace(b"\r\n", b"")
    assert methods.read_series_method(manifest.parent).active == "phase_model"


def test_panel_saved_refs_win_and_only_explicit_use_emits(qtbot, manifest):
    panel = methods.SeriesMethodPanel()
    qtbot.addWidget(panel)
    emitted = []
    panel.use_requested.connect(emitted.append)
    original = manifest.read_bytes()
    saved = methods.MethodSelection(
        active="phase_model",
        model_version="saved-model",
        mass_calibration="saved-fit",
        colour_calibration="saved-colour",
    )
    panel.set_selection(
        saved,
        phase_model_reference="selected-other",
        mass_calibration_reference="other-fit",
    )
    assert panel.selection() == saved
    assert not emitted
    panel.model_version.setText("edited-model")
    panel.colour_path_button.click()
    assert not emitted and manifest.read_bytes() == original
    assert panel.colour_calibration.isEnabled()
    assert not panel.model_version.isEnabled()
    panel.use_button.click()
    assert len(emitted) == 1
    assert emitted[0].active == "colour_path"
    assert emitted[0].colour_calibration == "saved-colour"
    assert emitted[0].model_version == "edited-model"
    assert manifest.read_bytes() == original  # Panel never owns persistence.


def test_panel_never_fabricates_missing_refs_or_chooses_implicitly(qtbot):
    panel = methods.SeriesMethodPanel()
    qtbot.addWidget(panel)
    panel.set_selection(methods.MethodSelection())
    assert not panel.use_button.isEnabled()
    assert panel.selection() == methods.MethodSelection()
    panel.set_selection(
        methods.MethodSelection(),
        phase_model_reference="actual/model",
        mass_calibration_reference="../actual-fit",
    )
    assert panel.selection().model_version == "actual/model"
    assert panel.selection().mass_calibration == "../actual-fit"
    assert panel.selection().colour_calibration == ""
    assert panel.selection().active == ""
    panel.phase_model_button.click()
    assert panel.use_button.isEnabled()
    panel.set_selection(methods.MethodSelection(error="broken TOML"))
    assert not panel.use_button.isEnabled()
    assert panel.status.text() == "broken TOML"
    panel.set_selection(methods.MethodSelection(active="phase_model"), available=False)
    assert not panel.use_button.isEnabled()


@pytest.mark.parametrize(
    "choice",
    [
        methods.MethodSelection(active="phase_model", model_version="model"),
        methods.MethodSelection(active="phase_model", mass_calibration="fit"),
        methods.MethodSelection(active="colour_path", colour_calibration="  \t "),
    ],
)
def test_missing_chosen_reference_rejects_save_without_any_write(manifest, choice):
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match="before saving this method"):
        methods.save_series_method(manifest.parent, choice)
    assert manifest.read_bytes() == original
    assert not list(manifest.parent.glob(".series.toml.*.tmp"))


def test_saved_active_missing_refs_remains_visible_with_warning_and_no_fallback(
    qtbot, manifest
):
    manifest.write_bytes(
        manifest.read_bytes()
        + b'\n[method]\nactive = "phase_model"\n[method.colour_path]\ncalibration = "available-other-method"\n'
    )
    original = manifest.read_bytes()
    loaded = methods.read_series_method(manifest.parent)
    assert loaded.active == "phase_model"
    assert methods.missing_method_references(loaded) == (
        "model version",
        "mass calibration",
    )
    panel = methods.SeriesMethodPanel()
    qtbot.addWidget(panel)
    panel.set_selection(loaded)
    assert panel.phase_model_button.isChecked()
    assert not panel.colour_path_button.isChecked()
    assert "missing" in panel.status.text()
    assert not panel.use_button.isEnabled()
    panel.model_version.setText("  opaque/模型 ")
    assert not panel.use_button.isEnabled()
    panel.mass_calibration.setText("../opaque/fit.json")
    assert panel.use_button.isEnabled()
    assert panel.selection().model_version == "  opaque/模型 "
    assert manifest.read_bytes() == original


@pytest.mark.parametrize("active", methods.METHODS)
def test_calibration_method_can_be_saved_before_any_model_or_calibration(manifest, active):
    original = tomllib.loads(manifest.read_text(encoding="utf-8"))
    saved = methods.save_series_calibration_method(manifest.parent, active,
                                                  expected_bytes=manifest.read_bytes())
    updated = tomllib.loads(manifest.read_text(encoding="utf-8"))
    assert updated == dict(original, method=dict(active=active))
    assert saved == methods.MethodSelection(active=active)
    assert sorted(path.name for path in manifest.parent.iterdir()) == ["series.toml"]


def test_calibration_method_save_keeps_references_and_unknown_data_with_bom_crlf(manifest):
    document = manifest.read_text(encoding="utf-8") + (
        '\n[method]\nactive="phase_model"\nfuture="retain"\n'
        '[method.phase_model]\nmodel_version="nonexistent/model"\nmass_calibration="nonexistent/mass"\n'
        '[method.colour_path]\ncalibration="nonexistent/colour"\n')
    manifest.write_bytes(b"\xef\xbb\xbf" + document.replace("\n", "\r\n").encode())
    original = manifest.read_bytes()
    before = tomllib.loads(original.decode("utf-8-sig"))
    saved = methods.save_series_calibration_method(manifest.parent, "colour_path", expected_bytes=original)
    updated = manifest.read_bytes()
    after = tomllib.loads(updated.decode("utf-8-sig"))
    before["method"]["active"] = "colour_path"
    assert after == before
    assert saved == methods.MethodSelection(active="colour_path", model_version="nonexistent/model",
        mass_calibration="nonexistent/mass", colour_calibration="nonexistent/colour")
    assert updated.startswith(b"\xef\xbb\xbf")
    assert b"\n" not in updated.replace(b"\r\n", b"")


def test_calibration_method_save_rejects_stale_manifest_and_context(manifest):
    original = manifest.read_bytes()
    manifest.write_bytes(original + b"\n# another explicit edit\n")
    changed = manifest.read_bytes()
    with pytest.raises(methods.ManifestChangedError, match="since it was loaded"):
        methods.save_series_calibration_method(manifest.parent, "phase_model", expected_bytes=original)
    assert manifest.read_bytes() == changed
    with pytest.raises(methods.ManifestChangedError, match="context changed"):
        methods.save_series_calibration_method(manifest.parent, "phase_model", expected_bytes=changed,
                                               context_current=lambda: False)
    assert manifest.read_bytes() == changed
    assert not list(manifest.parent.glob(".series.toml.*.tmp"))


def test_calibration_method_page_has_only_two_choices_and_needs_no_references(qtbot, manifest):
    from PySide6.QtWidgets import QLineEdit, QRadioButton

    panel = methods.SeriesMethodPanel(method_choice_only=True, require_bound_approval=True)
    qtbot.addWidget(panel)
    panel.set_selection(methods.MethodSelection())
    panel.set_approval_context(manifest.parent, manifest_bytes=manifest.read_bytes())
    panel.show()
    emitted = []
    panel.use_requested.connect(emitted.append)
    assert panel.title.text() == "Calibration Method"
    assert len(panel.findChildren(QRadioButton)) == 2
    assert not any(field.isVisible() for field in panel.findChildren(QLineEdit))
    assert not panel.use_button.isEnabled()
    panel.phase_model_button.click()
    assert panel.use_button.isEnabled()
    panel.use_button.click()
    assert emitted == [methods.MethodSelection(active="phase_model")]
    assert not panel.calls.timer.isActive() and not panel.calls.preview.busy
    panel.set_selection(emitted[0])
    assert panel.use_button.text() == "✓ Calibration method saved"
    assert not panel.use_button.isEnabled()
    assert "missing" not in panel.status.text().lower()
    panel.colour_path_button.click()
    assert panel.use_button.isEnabled()
    assert panel.selection().active == "colour_path"


def test_calibration_method_page_keeps_opaque_references_without_checking_them(qtbot, manifest, monkeypatch):
    from darsia.fluidflower import series_calibration_approval

    monkeypatch.setattr(series_calibration_approval, "selection_binding",
                        lambda *_args: pytest.fail("Method choice must not inspect model or calibration artifacts"))
    saved = methods.MethodSelection(active="phase_model", model_version="not/available",
                                    mass_calibration="not/available/mass", colour_calibration="old/colour")
    panel = methods.SeriesMethodPanel(method_choice_only=True, require_bound_approval=True)
    qtbot.addWidget(panel)
    panel.set_selection(saved)
    panel.set_approval_context(manifest.parent, manifest_bytes=manifest.read_bytes())
    qtbot.wait(150)  # Exceeds the original automatic approval-reader delay.
    assert panel.selection() == saved
    assert not panel.calls.timer.isActive() and not panel.calls.preview.busy
    assert not panel.calls.ready()
    panel.colour_path_button.click()
    assert panel.use_button.isEnabled()
    assert panel.selection().model_version == saved.model_version
    assert panel.selection().mass_calibration == saved.mass_calibration


def test_calibration_method_page_keeps_existing_explicit_lock(qtbot, manifest):
    panel = methods.SeriesMethodPanel(method_choice_only=True)
    qtbot.addWidget(panel)
    panel.set_selection(methods.MethodSelection(active="phase_model"))
    panel.set_approval_context(manifest.parent, record={"locked": True}, manifest_bytes=manifest.read_bytes())
    assert not panel.colour_path_button.isEnabled() and not panel.phase_model_button.isEnabled()
    assert not panel.use_button.isEnabled()
    assert not panel.unlock_button.isHidden()
