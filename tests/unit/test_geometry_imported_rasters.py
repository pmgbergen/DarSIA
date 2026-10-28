"""Lossless authority, explicit transitions, and unchanged ordinary Store path.

All inputs are tiny synthetic local arrays. These tests do not qualify the real
adopted-v1 Build or a registered geometry correction.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import toml

from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower.geometry_workspace_frame import FRAME_BINDING_FORMAT, adopted_content_digest


class Project:
    def __init__(self, root):
        self.root = root
        self.manifest = {"rig": {"type": "medium"}}
        self.metadata = []
        self.audit = []

    def update_metadata(self, *args, **kwargs):
        self.metadata.append((args, kwargs))

    def _audit(self, *args):
        self.audit.append(args)


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


@pytest.fixture
def imported(tmp_path):
    original = tmp_path / "original"
    original.mkdir()
    arrays = {"labels": np.array([[1, 1, 1], [2, 2, 2]], dtype=np.uint8),
              "facies": np.array([[1, 3, 1], [6, 2, 4]], dtype=np.uint8),
              "baseline": np.zeros((2, 3, 3), dtype=np.float32),
              "depth": np.full((2, 3), .01, dtype=np.float64)}
    for name, array in arrays.items():
        np.savez(original / (name + ".npz"), array=array)
    csv = original / "full.csv"
    csv.write_bytes(b"id,name,porosity,permeability,swi CO2,custom\r\n"
                    b"1,Sand1,0.4,10,0.12,keep\r\n2,Sand2,0.3,20,0.13,keep\r\n"
                    b"3,Boundary3,0.2,30,0.14,keep\r\n4,Boundary4,0.1,40,0.15,keep\r\n"
                    b"6,Boundary6,0.5,60,0.16,keep\r\n")
    diagnosis = original / "diagnosis.json"
    diagnosis.write_text('{"synthetic_different_pixels":3}\n', encoding="utf-8")
    project = Project(tmp_path / "workspace")
    config = geometry.default_geometry_config(project)
    config.update(baseline_path=str(original / "baseline.npz"), labels_path=str(original / "labels.npz"),
                  depth_path=str(original / "depth.npz"), facies_source_path=str(original / "facies.npz"),
                  facies_properties_source_path=str(csv), permeability_unit="source_unit",
                  regions=[dict(label_id=1, facies_id=1, name="Sand1", analysis_role="porous_medium",
                                enabled=True, porosity=.4, permeability=10.),
                           dict(label_id=2, facies_id=2, name="Sand2", analysis_role="porous_medium",
                                enabled=True, porosity=.3, permeability=20.)])
    config = sources.bind_imported_rasters(config, sources={
        "labels": sources.descriptor(config["labels_path"]),
        "facies": sources.descriptor(config["facies_source_path"]),
        "facies_properties": sources.descriptor(csv)}, diagnosis=sources.descriptor(diagnosis))
    return SimpleNamespace(project=project, config=config, arrays=arrays, original=original)


def save(data, config=None):
    return geometry.GeometryRevisionStore(data.project).save_revision(
        config or data.config, actor="synthetic owner", reason="explicit synthetic adoption",
        preserve_failed_artifacts=True)


def test_genuine_store_copies_both_independent_arrays_and_entire_csv(imported):
    before = snapshot(imported.original)
    revision = save(imported)
    assert len(revision["revision_id"]) == 32
    artifacts = revision["artifacts"]
    for key in sources.KEYS:
        row = artifacts["source_mode_files"][key]
        assert row == sources.descriptor(row["path"])
        assert Path(row["path"]).read_bytes() == Path(imported.config[sources.PATH_KEYS[key]]).read_bytes()
    assert np.array_equal(geometry._load_array(Path(artifacts["labels_path"])), imported.arrays["labels"])
    assert np.array_equal(geometry._load_array(Path(artifacts["facies_path"])), imported.arrays["facies"])
    assert revision["validation"]["summary"]["full_properties_ids"] == [1, 2, 3, 4, 6]
    assert revision["validation"]["summary"]["estimated_pore_volume_m3"] is None
    assert revision["validation"]["summary"]["active_area_fraction"] is None
    detail = artifacts["source_mode_difference"]
    assert detail["different_pixels"] == 3 and detail["pixel_count"] == 6
    assert "independent of labels" in detail["notice"]
    assert snapshot(imported.original) == before
    assert geometry.GeometryRevisionStore(imported.project).current()["revision_id"] == revision["revision_id"]


def test_preview_and_current_config_keep_the_same_authoritative_facies(imported):
    revision = save(imported)
    config = geometry.GeometryRevisionStore(imported.project).current_config()
    assert config["source_mode"] == sources.IMPORTED
    preview = geometry.load_geometry_preview(config, max_width=3, max_height=2)
    assert np.array_equal(preview["labels"], imported.arrays["labels"])
    assert np.array_equal(preview["facies"], imported.arrays["facies"])
    assert config["depth_path"] == revision["artifacts"]["depth_path"]


@pytest.mark.parametrize("key", ["labels", "facies", "facies_properties"])
def test_changed_source_is_refused_even_with_cached_validation(imported, key):
    revision = save(imported)
    target = Path(imported.config[sources.PATH_KEYS[key]])
    target.write_bytes(target.read_bytes() + b"external edit")
    store = geometry.GeometryRevisionStore(imported.project)
    before = snapshot(imported.project.root)
    with pytest.raises(ValueError, match="source changed"):
        store.save_revision(imported.config, actor="owner", reason="no hidden repair", validation=revision["validation"])
    assert snapshot(imported.project.root) == before


@pytest.mark.parametrize("mutation", ["region", "mode_removed", "facies_path", "diagnosis"])
def test_save_refuses_silent_recanonicalization_or_relabelling(imported, mutation):
    save(imported)
    config = deepcopy(imported.config)
    if mutation == "region":
        config["regions"][0]["facies_id"] = 3
    elif mutation == "mode_removed":
        config.pop("source_mode")
        config.pop("imported_rasters")
    elif mutation == "facies_path":
        config["facies_source_path"] = config["labels_path"]
    else:
        config["imported_rasters"]["diagnosis"]["sha256"] = "f" * 64
    before = snapshot(imported.project.root)
    with pytest.raises(ValueError, match="independently authoritative"):
        save(imported, config)
    assert snapshot(imported.project.root) == before


def test_absent_mode_keeps_original_raw_and_painted_derivation(imported):
    config = deepcopy(imported.config)
    config.pop("source_mode")
    config.pop("imported_rasters")
    expected, properties = geometry._canonical_facies(config)
    assert not np.array_equal(expected, imported.arrays["facies"])
    revision = save(imported, config)
    assert np.array_equal(geometry._load_array(Path(revision["artifacts"]["facies_path"])), expected)
    assert "source_mode_files" not in revision["artifacts"]
    assert [row["id"] for row in properties] == [1, 2]
    assert "source_mode" not in revision["configuration"]


def test_first_ordinary_save_never_requires_project_manifest_or_current_default(imported):
    config = deepcopy(imported.config)
    config.pop("source_mode")
    config.pop("imported_rasters")
    # This was a valid ordinary Store client before the imported source mode.
    # No project.manifest exists, and no existing revision needs to be read.
    project = SimpleNamespace(root=imported.original.parent / "ordinary_workspace",
                              update_metadata=lambda *args, **kwargs: None,
                              _audit=lambda *args: None)
    revision = geometry.GeometryRevisionStore(project).save_revision(
        config, actor="ordinary owner", reason="original first-save behavior")
    assert "source_mode" not in revision["configuration"]
    assert geometry.GeometryRevisionStore(project).current_path.is_file()


def test_existing_imported_history_cannot_be_bypassed_when_current_is_absent(imported):
    save(imported)
    store = geometry.GeometryRevisionStore(imported.project)
    store.current_path.rename(store.current_path.with_name("current-preserved.json"))
    config = deepcopy(imported.config)
    config.pop("source_mode")
    config.pop("imported_rasters")
    before = snapshot(imported.project.root)
    with pytest.raises(ValueError, match="independently authoritative"):
        save(imported, config)
    assert snapshot(imported.project.root) == before


@pytest.mark.parametrize("key", ["labels", "facies"])
@pytest.mark.parametrize("suffix", [".png", ".tiff"])
def test_imported_authority_refuses_image_bytes_renamed_to_npz(imported, key, suffix):
    from PIL import Image
    image = imported.original / (key + suffix)
    Image.fromarray(imported.arrays[key]).save(image)
    config = deepcopy(imported.config)
    config[sources.PATH_KEYS[key]] = str(image)
    bindings = deepcopy(config["imported_rasters"]["sources"])
    bindings[key] = sources.descriptor(image)
    config = sources.bind_imported_rasters(config, sources=bindings,
                                           diagnosis=config["imported_rasters"]["diagnosis"])
    before = snapshot(imported.original)
    with pytest.raises(ValueError, match="original NPZ archives"):
        save(imported, config)
    assert not imported.project.root.exists()
    assert snapshot(imported.original) == before


def test_csv_must_cover_every_independent_facies_id(imported):
    csv = Path(imported.config["facies_properties_source_path"])
    csv.write_bytes(b"id,name,porosity,permeability\n1,A,.4,10\n2,B,.3,20\n")
    config = sources.bind_imported_rasters(imported.config, sources={
        "labels": sources.descriptor(imported.config["labels_path"]),
        "facies": sources.descriptor(imported.config["facies_source_path"]),
        "facies_properties": sources.descriptor(csv)},
        diagnosis=imported.config["imported_rasters"]["diagnosis"])
    with pytest.raises(ValueError, match="omits raster facies IDs"):
        save(imported, config)
    assert not imported.project.root.exists()


def test_explicit_edit_retains_parent_full_csv_and_actual_difference(imported):
    original_revision = save(imported)
    before_original = snapshot(imported.original)
    parent_artifacts = snapshot(Path(original_revision["artifacts"]["facies_path"]).parent)
    draft = sources.prepare_canonical_edit(imported.project, actor="explicit user", reason="edit layer labels")
    assert draft["source_mode"] == sources.CANONICAL_EDIT
    assert draft["canonical_import_parent"]["revision_id"] == original_revision["revision_id"]
    assert draft["provenance"]["registered_geometry_authorized"] is False
    edited = imported.project.root / "edited-labels.npz"
    labels = imported.arrays["labels"].copy()
    labels[0, 0] = 2
    np.savez(edited, array=labels)
    draft["labels_path"] = str(edited)
    expected, _ = geometry._canonical_facies(draft)
    revision = save(imported, draft)
    assert revision["revision_id"] != original_revision["revision_id"]
    assert np.array_equal(geometry._load_array(Path(revision["artifacts"]["facies_path"])), expected)
    assert Path(revision["artifacts"]["facies_properties_path"]).read_bytes() == Path(imported.config["facies_properties_source_path"]).read_bytes()
    detail = revision["artifacts"]["source_mode_difference"]
    assert detail["labels_changed_from_parent"] == 1
    assert detail["different_pixels"] == int(np.count_nonzero(expected != imported.arrays["facies"]))
    assert "not an adopted-v1 identity gate" in detail["known_change"]
    assert snapshot(imported.original) == before_original
    assert snapshot(Path(original_revision["artifacts"]["facies_path"]).parent) == parent_artifacts


def test_canonical_edit_cannot_change_parent_csv_or_physical_properties(imported):
    save(imported)
    draft = sources.prepare_canonical_edit(imported.project, actor="user", reason="explicit draft")
    draft["regions"][0]["porosity"] = .9
    with pytest.raises(ValueError, match="physical CSV properties"):
        save(imported, draft)


def test_original_nan_water_csv_values_remain_none_and_are_not_false_edits(imported):
    csv = Path(imported.config["facies_properties_source_path"])
    csv.write_bytes(csv.read_bytes() + b"0,water,nan,nan,,unchanged\r\n")
    original_properties = geometry._load_properties(csv)
    assert original_properties[0]["porosity"] is None
    assert original_properties[0]["permeability"] is None
    labels = imported.arrays["labels"].copy()
    facies = imported.arrays["facies"].copy()
    labels[0, 0] = 3
    facies[0, 0] = 0
    np.savez(Path(imported.config["labels_path"]), array=labels)
    np.savez(Path(imported.config["facies_source_path"]), array=facies)
    config = deepcopy(imported.config)
    config["regions"].append(dict(label_id=3, facies_id=0, name="water", analysis_role="free_water",
                                  enabled=True, porosity=None, permeability=None))
    config = sources.bind_imported_rasters(config, sources={
        name: sources.descriptor(config[sources.PATH_KEYS[name]]) for name in sources.KEYS},
        diagnosis=config["imported_rasters"]["diagnosis"])
    save(imported, config)
    draft = sources.prepare_canonical_edit(imported.project, actor="user", reason="explicit water-preserving draft")
    revision = save(imported, draft)
    assert Path(revision["artifacts"]["facies_properties_path"]).read_bytes() == csv.read_bytes()
    assert revision["configuration"]["regions"][-1]["porosity"] is None
    assert revision["configuration"]["regions"][-1]["permeability"] is None


def test_canonical_export_keeps_genuine_store_authority_and_full_csv(imported, monkeypatch):
    from darsia.fluidflower import corrected_setup as export
    series = imported.original.parent / "series"
    (series / "config/merged").mkdir(parents=True)
    (series / "config/runs").mkdir()
    (series / "series.toml").write_text(toml.dumps({"series_id": "synthetic", "runs": {"ids": ["run1"]}}))
    original_config = dict(depth={"depth_map": imported.config["depth_path"]},
                           facies={"props": imported.config["facies_properties_source_path"]})
    for kind in ("runs", "merged"):
        (series / "config" / kind / "run1.toml").write_text(toml.dumps(original_config))
    config = deepcopy(imported.config)
    fields = {}
    for name, key in (("shape_corrected_baseline", "baseline_path"), ("labels", "labels_path"),
                      ("facies", "facies_source_path"), ("depth", "depth_path")):
        fields[name] = {**sources.descriptor(config[key]), "shape": list(geometry._load_array(Path(config[key])).shape)}
    correction = imported.original / "correction.json"
    correction.write_text("{}")
    config.update(frame="rig-corrected", frame_binding=dict(format=FRAME_BINDING_FORMAT,
        source_reference_run="historical", display_reference_run="run1",
        reference_shape_corrections=[sources.descriptor(correction)], runs={"run1": dict(fields=fields,
            configuration=sources.descriptor(series / "config/merged/run1.toml"),
            configured_depth_map=sources.descriptor(config["depth_path"]),
            facies_properties=sources.descriptor(config["facies_properties_source_path"]))}))
    # Bind the changed frame metadata before the explicit Store operation.
    config = sources.bind_imported_rasters(config, sources=config["imported_rasters"]["sources"],
                                           diagnosis=config["imported_rasters"]["diagnosis"])
    config["provenance"] = {"adopted_content_sha256": adopted_content_digest(config)}
    revision = save(imported, config)
    monkeypatch.setattr(export, "native_series_link", lambda _project: {"series_root": str(series)})
    import darsia
    monkeypatch.setattr(darsia, "imread", lambda _path: object())
    monkeypatch.setattr(export, "_require_exact_grid", lambda *_args: None)
    result = export.export_corrected_setup(imported.project, reference=SimpleNamespace(run_id="run1"), actor="owner")
    assert result.revision_id == revision["revision_id"]
    assert (result.destination / "native_facies.npz").read_bytes() == Path(config["facies_source_path"]).read_bytes()
    assert (result.destination / "native_labels.npz").read_bytes() == Path(config["labels_path"]).read_bytes()
    assert (result.destination / "facies_properties.csv").read_bytes() == Path(config["facies_properties_source_path"]).read_bytes()
    assert result.receipt["actual_Build_run"] is False
    assert result.receipt["lossless_rebuilt_gate"] == "NOT RUN"


def test_advanced_imported_view_is_readonly_with_explicit_draft_action(imported):
    from PySide6.QtWidgets import QApplication, QTabWidget, QWidget, QLineEdit, QVBoxLayout, QLabel, QPushButton, QComboBox, QTableWidget
    from darsia.fluidflower.pages.geometry_facies.page import GeometryFaciesPage
    app = QApplication.instance() or QApplication([])
    tabs = QTabWidget()
    panel = QWidget()
    layout = QVBoxLayout(panel)
    field = QLineEdit("authoritative source")
    layout.addWidget(field)
    table = QTableWidget(1, 1)
    table.setCellWidget(0, 0, QComboBox())
    layout.addWidget(table)
    view_selector = QComboBox()
    layout.addWidget(view_selector)
    tabs.addTab(panel, "Fields")
    data = SimpleNamespace(config=deepcopy(imported.config), tabs=tabs, canonical_edit_button=QPushButton(),
                           save_button=QPushButton(), geometry_quality_basis=QLabel("ordinary basis"),
                           background=view_selector, import_rig=QPushButton(), place_injector=QPushButton())
    data.place_injector.setEnabled(False)
    GeometryFaciesPage._sync_imported_source_mode(data)
    assert not field.isEnabled() and not data.save_button.isEnabled()
    assert not data.canonical_edit_button.isHidden()
    assert view_selector.isEnabled()
    assert not data.import_rig.isEnabled() and not data.place_injector.isEnabled()
    assert "FULL original properties CSV" in data.geometry_quality_basis.text()
    assert GeometryFaciesPage._collect_config(data) == imported.config
    # The real page repopulates its transient table-cell editors before restoring
    # ordinary editing. No stale Qt wrapper may remain in the saved state list.
    table.setCellWidget(0, 0, QComboBox())
    data.config = {"source_mode": sources.CANONICAL_EDIT}
    GeometryFaciesPage._sync_imported_source_mode(data)
    assert field.isEnabled() and data.canonical_edit_button.isHidden()
    assert table.isEnabled() and table.cellWidget(0, 0).isEnabled()
    assert data.geometry_quality_basis.text() == "ordinary basis"
    assert data.import_rig.isEnabled() and not data.place_injector.isEnabled()
    assert app is not None


# N54: opaque registered-source fields survive real Qt table round trips.
# These tiny synthetic rows do not qualify or copy the actual FF_AC geometry.
def registered_table_page_TESTONLY():
    from PySide6.QtWidgets import QApplication, QTableWidget, QComboBox, QCheckBox, QDoubleSpinBox
    from darsia.fluidflower.pages.geometry_facies.page import GeometryFaciesPage
    app=QApplication.instance() or QApplication([])
    policy=QComboBox();policy.addItem("Reviewed", "allow_reviewed_run_overrides")
    detect=QCheckBox();detect.setChecked(True)
    displacement=QDoubleSpinBox();displacement.setValue(.5)
    affected=QDoubleSpinBox();affected.setValue(10.)
    page=SimpleNamespace(run_geometry_table=QTableWidget(0,9),run_geometry_policy=policy,
        run_geometry_change_detection=detect,run_geometry_displacement_threshold=displacement,
        run_geometry_affected_threshold=affected,_mark_dirty=lambda *args:None)
    return app,page,GeometryFaciesPage


def test_N54_all43_registered_row_provenance_survives_GUI_collect_without_source_rebinding():
    app,page,cls=registered_table_page_TESTONLY()
    original=[]
    for i in range(43):
        row=dict(enabled=True,run_id="ac"+str(17+i),acquisition_state="unknown",geometry_choice="run_override",
            source_image_path="TESTONLY/photo"+str(i)+".npz",labels_path="TESTONLY/labels"+str(i)+".npz",
            facies_path="",property_change_proposals=[],registration=dict(parent_revision_id="a"*32,
                labels=dict(path="TESTONLY/labels"+str(i)+".npz",sha256="b"*64,size_bytes=10),
                status="registered",authority={"TESTONLY":"unchanged"}),
            additional_provenance={"TESTONLY":[i,{"immutable":"kept"}]})
        original.append(deepcopy(row));cls._append_run_geometry_row(page,row)
    actual=cls._collect_run_geometry(page)["runs"]
    assert actual==original
    actual[0]["registration"]["authority"]["TESTONLY"]="changed only returned copy"
    assert cls._collect_run_geometry(page)["runs"]==original


def test_N54_renamed_GUI_row_cannot_inherit_another_runs_registration():
    app,page,cls=registered_table_page_TESTONLY()
    cls._append_run_geometry_row(page,dict(run_id="ac18",registration={"TESTONLY":"ac18-only proof"}))
    page.run_geometry_table.item(0,1).setText("ac47")
    actual=cls._collect_run_geometry(page)["runs"][0]
    assert actual["run_id"]=="ac47" and "registration" not in actual


@pytest.mark.parametrize("optional", [{}, {"facies_path": ""},
    {"property_change_proposals": [{"TESTONLY": "preserve pending review"}]}])
def test_N54_legacy_registered_contract_roundtrips_without_optional_field_insertion(optional):
    from darsia.fluidflower.geometry_facies import normalize_run_geometry_config
    app, page, cls = registered_table_page_TESTONLY()
    row = dict(run_id="ac17", enabled=True, acquisition_state="unknown",
        geometry_choice="run_override", labels_path="TESTONLY/labels.npz",
        source_image_path="TESTONLY/photo.npz", registration={"TESTONLY": "original"},
        **optional)
    contract = normalize_run_geometry_config({"runs": [row],
        "provenance": {"TESTONLY": "original contract"}})
    page.config = {"run_geometry": deepcopy(contract)}
    cls._append_run_geometry_row(page, row)
    assert cls._collect_run_geometry(page) == contract
    assert page.config["run_geometry"] == contract
    assert app is not None


def test_N54_user_can_add_then_clear_a_previously_absent_facies_path():
    app, page, cls = registered_table_page_TESTONLY()
    row = dict(run_id="ac17", registration={"TESTONLY": "original"})
    cls._append_run_geometry_row(page, row)
    assert "facies_path" not in cls._collect_run_geometry(page)["runs"][0]
    page.run_geometry_table.item(0, 6).setText("TESTONLY/user-selected.npz")
    changed = cls._collect_run_geometry(page)["runs"][0]
    assert changed["facies_path"] == "TESTONLY/user-selected.npz"
    assert changed["registration"] == row["registration"]
    cls._append_run_geometry_row(page, changed)
    page.run_geometry_table.item(1, 6).setText("")
    cleared = cls._collect_run_geometry(page)["runs"][1]
    assert cleared["facies_path"] == ""
    assert cleared["registration"] == row["registration"]
    assert app is not None


def test_N54_changed_label_path_does_not_relabel_original_registration_descriptor():
    app,page,cls=registered_table_page_TESTONLY()
    original=dict(run_id="ac18",geometry_choice="run_override",labels_path="TESTONLY/original.npz",
        registration=dict(labels={"path":"TESTONLY/original.npz","sha256":"a"*64,"size_bytes":10}))
    cls._append_run_geometry_row(page,original)
    page.run_geometry_table.item(0,5).setText("TESTONLY/different.npz")
    actual=cls._collect_run_geometry(page)["runs"][0]
    assert actual["labels_path"]=="TESTONLY/different.npz"
    assert actual["registration"]==original["registration"]
    # Existing _verify(record.labels, row.labels_path) refuses this mismatch.


@pytest.mark.parametrize("diagnostic,expected_status", [
    (None, "Not checked"),
    ({"run_id": "ac18", "status": "not_comparable", "reason": "TESTONLY missing comparable map"}, "Not comparable"),
    ({"run_id": "ac18", "status": "comparable", "p95_displacement_m": .001,
      "affected_boundary_fraction": .05, "substantial_change": False}, "Within threshold"),
    ({"run_id": "ac18", "status": "comparable", "p95_displacement_m": .003,
      "affected_boundary_fraction": .25, "substantial_change": True}, "Review"),
])
def test_N54_worker_validation_survives_derived_diagnostics_but_real_run_edit_invalidates(
    diagnostic, expected_status,
):
    from PySide6.QtWidgets import QLabel, QPushButton, QTableWidget

    app, page, cls = registered_table_page_TESTONLY()
    cls._append_run_geometry_row(page, dict(
        run_id="ac18", geometry_choice="run_override", labels_path="TESTONLY/original.npz",
    ))
    config = geometry.default_geometry_config()
    page.config = deepcopy(config)
    page.validation = None
    page.preview_data = None
    page._applying = False
    page.save_button = QPushButton()
    page.geometry_summary = QLabel()
    page.geometry_quality_basis = QLabel()
    page.validation_details = QLabel()
    page.issues = QTableWidget(0, 3)
    page._collect_config = lambda: deepcopy(page.config)
    page._populate_config = lambda _config: None
    page._finish_worker = lambda: None
    page._show_run_geometry_diagnostics = lambda summary: cls._show_run_geometry_diagnostics(page, summary)
    page._show_validation = lambda report: cls._show_validation(page, report)
    page.run_geometry_table.itemChanged.connect(lambda *args: cls._mark_dirty(page, *args))
    report = dict(config_sha256=geometry.geometry_config_digest(config), issues=[], summary=dict(
        label_count=2, facies_count=1, error_count=0, warning_count=0,
        run_geometry_change_diagnostics=[] if diagnostic is None else [diagnostic],
    ))

    # The real finished callback and actual Qt itemChanged signal route reproduce
    # the old loss of validation; no worker, scientific input or preview is run.
    cls._worker_finished(page, dict(config=config, validation=report, preview=None))
    assert page.validation == report
    assert page.save_button.isEnabled()
    assert expected_status in page.run_geometry_table.item(0, 7).text()
    assert "0 errors" in page.geometry_summary.text()
    assert not page.run_geometry_table.signalsBlocked()

    # Only derived writes are blocked. A genuine edit still uses the original
    # handler, clears the report and closes the Save gate.
    page.run_geometry_table.item(0, 5).setText("TESTONLY/changed-by-user.npz")
    assert page.validation is None
    assert not page.save_button.isEnabled()
    assert page.geometry_summary.text() == "Local changes need validation."
    assert app is not None
