"""RAW configuration settings in genuine private Store/export/Apply transactions.

The camera document proves its explicit run assignment only; these cases do
not decode an acquisition, fit a model or execute the scientific rig setup.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tomllib

import pytest
import toml

from darsia.fluidflower import corrected_setup as setup
from darsia.fluidflower import painted_setup_apply as rebuilt
from .test_geometry_imported_rasters import snapshot
from .test_painted_setup_apply import named_setup, _painting, _files  # noqa: F401
from .test_painted_adoption_table import imported, synthetic_payload_admission  # noqa: F401


@pytest.fixture
def raw_named_setup(named_setup, monkeypatch):
    """Keep the historical fixture injector confined to its original inputs."""
    monkeypatch.setattr(toml, "dumps", toml.encoder.dumps)
    return named_setup


def _settings(tmp_path, *, native_padding_bytes=0):
    """Bind a real private JSON document with one explicit profile assignment."""
    profile = tmp_path / "raw-camera.json"
    document = dict(registration=dict(common=dict(profiles={"selected": {}})),
                    source=dict(profiles=dict(one="selected")))
    if native_padding_bytes:
        document["native_metadata_padding"] = "x" * native_padding_bytes
    profile.write_text(json.dumps(document) + "\n", encoding="utf-8")
    return dict(run_id="one", raw_camera=dict(configuration=str(profile),
        sha256=hashlib.sha256(profile.read_bytes()).hexdigest(), profile_id="selected", run_id="one"),
        rig_path=str(tmp_path / "fresh-rig"))


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_raw_settings_share_original_apply_cas_and_exact_completion(raw_named_setup, tmp_path):
    """Change only selected-run settings and reuse the exact retained recipe."""
    data = raw_named_setup
    export = _painting(data, tmp_path)
    settings = _settings(tmp_path)
    before = {path: path.read_bytes() for path in _files(data)}
    scientific_before = snapshot(data.data.original)
    attempt = setup.apply_corrected_selected_run(data.project, export,
        rebuild_facies_by_name=True, raw_setup_settings=settings)
    record = json.loads(attempt.read_bytes())
    complete = json.loads((export.destination / "apply-complete.json").read_bytes())
    claim = json.loads((export.destination / "apply-claim.json").read_bytes())
    for value in (claim, record, complete):
        assert value["facies_rebuild"]["raw_setup_settings"] == settings
    context = deepcopy(complete["facies_rebuild"])
    context.pop("raw_setup_settings")
    for kind in ("runs", "merged"):
        path = data.series / "config" / kind / "one.toml"
        ordinary, _ = rebuilt.rebuild_bytes(before[path], context, revision_id=export.revision_id,
            replacements=export.receipt["binding"]["replacements"])
        expected = tomllib.loads(ordinary.decode("utf-8-sig"))
        expected["data"]["raw_camera"] = deepcopy(settings["raw_camera"])
        expected["data"]["use_cache"] = False
        expected["rig"] = dict(path=settings["rig_path"])
        actual = tomllib.loads(path.read_text(encoding="utf-8"))
        assert setup._typed_equal(actual, expected)
        assert rebuilt.raw_settings_bytes(path.read_bytes(), settings) == path.read_bytes()
    common = data.series / "config/common.toml"
    expected_common, _ = rebuilt.rebuild_bytes(before[common], context,
        revision_id=export.revision_id, is_common=True)
    assert common.read_bytes() == expected_common
    assert "data" not in tomllib.loads(common.read_text())
    assert "rig" not in tomllib.loads(common.read_text())
    for kind in ("runs", "merged"):
        path = data.series / "config" / kind / "two.toml"
        assert path.read_bytes() == before[path]
    assert setup.require_corrected_export_for_build([common, data.series / "config/merged/one.toml"]) == export.receipt
    assert snapshot(data.data.original) == scientific_before
    all_after = snapshot(data.project.root)
    assert setup.apply_corrected_selected_run(data.project, export,
        rebuild_facies_by_name=True, raw_setup_settings=settings) == attempt
    assert snapshot(data.project.root) == all_after
    changed = deepcopy(settings)
    changed["rig_path"] = str(tmp_path / "another-fresh-rig")
    with pytest.raises(ValueError, match="completed application.*RAW Setup settings"):
        setup.apply_corrected_selected_run(data.project, export,
            rebuild_facies_by_name=True, raw_setup_settings=changed)
    assert snapshot(data.project.root) == all_after


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
@pytest.mark.parametrize("fault", ["no_rebuild", "wrong_run", "profile_assignment", "occupied_output"])
def test_raw_settings_refuse_invalid_input_before_original_claim(raw_named_setup, tmp_path, fault):
    """Keep every original TOML and exclusive claim untouched on refusal."""
    data = raw_named_setup
    export = _painting(data, tmp_path)
    settings = _settings(tmp_path)
    rebuild = True
    if fault == "no_rebuild":
        rebuild = False
    elif fault == "wrong_run":
        settings["raw_camera"]["run_id"] = "two"
    elif fault == "profile_assignment":
        settings["raw_camera"]["profile_id"] = "unassigned"
    else:
        output = Path(settings["rig_path"])
        output.mkdir()
        (output / "preserved-partial.json").write_text("{}\n", encoding="utf-8")
    before = {path: path.read_bytes() for path in _files(data)}
    with pytest.raises(ValueError, match="by-name|another selected run|run assignment|not empty"):
        setup.apply_corrected_selected_run(data.project, export,
            rebuild_facies_by_name=rebuild, raw_setup_settings=settings)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not (export.destination / "apply-claim.json").exists()


def test_raw_settings_respect_original_bound_fresh_output(tmp_path):
    """Neither ordinary dual nor private output authority can be redirected."""
    settings = _settings(tmp_path)
    for dual in (dict(format=setup.CONFIGURATION_BINDING_V2_FORMAT,
                      output_pointer_change=dict(after=str(tmp_path / "original-fresh-rig"))),
                 dict(format=setup.PRIVATE_MANUAL_CONFIGURATION_FORMAT,
                      output_pointer_changes={"rig.path": dict(after=str(tmp_path / "original-fresh-rig"))})):
        receipt = dict(run_id="one", binding=dict(dual_configuration_binding=dual))
        with pytest.raises(ValueError, match="original bound fresh target"):
            rebuilt.require_raw_output(settings, receipt)


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_rehashed_raw_recipe_still_replays_the_retained_postimage(raw_named_setup, tmp_path):
    """Matching receipt hashes cannot substitute a different RAW recipe."""
    data = raw_named_setup
    export = _painting(data, tmp_path)
    settings = _settings(tmp_path)
    attempt = setup.apply_corrected_selected_run(data.project, export,
        rebuild_facies_by_name=True, raw_setup_settings=settings)
    complete_path = export.destination / "apply-complete.json"
    complete = json.loads(complete_path.read_bytes())
    context = complete["facies_rebuild"]
    context["raw_setup_settings"]["rig_path"] = str(tmp_path / "foreign-output")
    for path in (export.destination / "apply-claim.json", attempt):
        value = json.loads(path.read_bytes())
        value["facies_rebuild"] = deepcopy(context)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    complete["attempt_receipt"] = setup._descriptor(attempt)
    complete_path.write_text(json.dumps(complete) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact by-name transformation"):
        setup._apply_completion(export.destination, export.receipt)


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_raw_camera_bytes_join_original_apply_cas(raw_named_setup, tmp_path, monkeypatch):
    """A camera edit between preparation and CAS cannot become a new input."""
    data = raw_named_setup
    export = _painting(data, tmp_path)
    settings = _settings(tmp_path)
    before = {path: path.read_bytes() for path in _files(data)}
    original = rebuilt.rebuild_bytes
    changed = False

    def edit_after_rebuild(*args, **kwargs):
        nonlocal changed
        result = original(*args, **kwargs)
        if kwargs.get("replacements") is not None and not changed:
            Path(settings["raw_camera"]["configuration"]).write_text("{}\n", encoding="utf-8")
            changed = True
        return result

    monkeypatch.setattr(rebuilt, "rebuild_bytes", edit_after_rebuild)
    with pytest.raises(RuntimeError, match="changed"):
        setup.apply_corrected_selected_run(data.project, export,
            rebuild_facies_by_name=True, raw_setup_settings=settings)
    assert changed
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert not (export.destination / "apply-claim.json").exists()


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
@pytest.mark.parametrize("change_during_cas", [False, True])
def test_large_native_camera_uses_original_bound_reader_and_apply_cas(
        raw_named_setup, tmp_path, monkeypatch, change_during_cas):
    """Accept larger native metadata while retaining exact camera-byte CAS.

    Padding enters the initial document before its SHA binding is captured.
    The original profile assignment and actual export/Apply transaction are
    exercised; no sensor, shape basis or scientific output is qualified.
    """
    data = raw_named_setup
    export = _painting(data, tmp_path)
    settings = _settings(tmp_path, native_padding_bytes=1024**2 + 512)
    camera_path = Path(settings["raw_camera"]["configuration"])
    camera_before = camera_path.read_bytes()
    assert len(camera_before) > setup.LIMIT
    assert hashlib.sha256(camera_before).hexdigest() == settings["raw_camera"]["sha256"]
    before = {path: path.read_bytes() for path in _files(data)}
    original_rebuild = rebuilt.rebuild_bytes
    changed = False

    def rebuild_with_camera_edit(*args, **kwargs):
        """Expose a genuine source edit after preparation and before the CAS."""
        nonlocal changed
        result = original_rebuild(*args, **kwargs)
        if kwargs.get("replacements") is not None and not changed:
            camera_path.write_bytes(camera_before + b" ")
            changed = True
        return result

    if change_during_cas:
        monkeypatch.setattr(rebuilt, "rebuild_bytes", rebuild_with_camera_edit)
        with pytest.raises(RuntimeError, match="changed"):
            setup.apply_corrected_selected_run(data.project, export,
                rebuild_facies_by_name=True, raw_setup_settings=settings)
        assert changed
        assert all(path.read_bytes() == raw for path, raw in before.items())
        assert not (export.destination / "apply-claim.json").exists()
    else:
        attempt = setup.apply_corrected_selected_run(data.project, export,
            rebuild_facies_by_name=True, raw_setup_settings=settings)
        assert json.loads(attempt.read_bytes())["state"] == "complete"
        for kind in ("runs", "merged"):
            path = data.series / "config" / kind / "one.toml"
            assert tomllib.loads(path.read_text())["data"]["raw_camera"] == settings["raw_camera"]
        assert setup.require_corrected_export_for_build(
            data.series / "config/merged/one.toml") == export.receipt
        assert camera_path.read_bytes() == camera_before
