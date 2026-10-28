"""A GUI colour-path rebuild reuses genuinely completed image corrections."""
from copy import deepcopy

import pytest

from darsia.fluidflower.mass.n1_pipeline import DataReader, _write_json
from darsia.fluidflower.pipeline import setup_image_preparation as subject


def completed_basis(tmp_path, monkeypatch):
    from darsia.fluidflower.native.full_photometry import ORDER
    from darsia.fluidflower.native.raw_rig_source import execution_identity
    choice = dict(method="off", color_method="off")
    authority = _write_json(tmp_path/"authority.json", {})
    request = _write_json(tmp_path/"request.json", dict(baseline=dict(path="original.ARW")))
    operations = {run: _write_json(tmp_path/(run+"-build.json"), dict(raw_build_request=request))
                  for run in ("one", "two")}
    runs = {run: dict(source_binding={"immutable":run}, baseline={"pixels":run}) for run in operations}
    photo = _write_json(tmp_path/"photometry.json", dict(runs=runs))
    paths = _write_json(tmp_path/"old-colour-paths.json", dict(format="darsia-setup-colour-paths-v1",complete=True))
    old = dict(format=subject.FORMAT,complete=True,choice=choice,operations=operations,
        choice_authority=authority,raw_execution_identity=execution_identity(),correction_order=ORDER,
        runs=runs,photometry_record=photo,colour_paths=paths,colorpaths_performed=True)
    pin = _write_json(tmp_path/"completed-basis.json",old)
    config = dict(job_root=str(tmp_path/"job"),runs=list(operations),output_root=str(tmp_path/"new-output"),
        choice_authority=authority,adopted_revision={},native_parameters={},
        colour_path_scope="per-run",reuse_image_basis=pin)
    monkeypatch.setattr(subject,"independent_choice",lambda *a,**k: ({},choice))
    monkeypatch.setattr(subject,"completed_build_operations",lambda *a,**k: operations)
    return config,old,paths


def test_rebuild_reuses_same_corrections_and_passes_original_sample_receipt(tmp_path,monkeypatch):
    config,old,paths=completed_basis(tmp_path,monkeypatch)
    def no_decode(*a,**k):
        pytest.fail("Completed corrections must not decode RAW or fit again")
    calls=[]
    def refit(operations,prepared,authority,folder,**options):
        assert operations == old["operations"]
        assert prepared["runs"] == old["runs"]
        assert prepared["record"] == old["photometry_record"]
        assert authority == old["choice_authority"]
        assert options["reuse_receipt"] == paths
        calls.append(options["workers"])
        folder.mkdir(parents=True)
        return _write_json(folder/"receipt.json",dict(format="darsia-setup-colour-paths-v2",scope="per-run"))
    progress=[]
    pin=subject.run_image_preparation(config,workers=6,producer=no_decode,colour_path_producer=refit,
                                      progress=progress.append)
    result=DataReader().json(pin)
    assert calls == [6]
    assert result["runs"] == old["runs"] and result["photometry_record"] == old["photometry_record"]
    assert result["reused_image_basis"] == config["reuse_image_basis"]
    assert result["colour_path_scope"] == "per-run"
    assert any(row["step"]=="reusing_completed_image_corrections" for row in progress)


@pytest.mark.parametrize("change",["choice","operations","choice_authority","raw_execution_identity",
                                     "correction_order","colour_paths","colorpaths_performed","photometry"])
def test_rebuild_rejects_changed_completed_source_before_any_producer(tmp_path,monkeypatch,change):
    config,old,_=completed_basis(tmp_path,monkeypatch)
    if change == "photometry":
        old["photometry_record"]=_write_json(tmp_path/"changed-photo.json",dict(runs={}))
    elif change in {"colour_paths","colorpaths_performed"}:
        old[change]=None
    elif change in {"choice","operations","choice_authority","raw_execution_identity"}:
        old[change]={"changed":True}
    else:
        old[change]="different"
    config["reuse_image_basis"]=_write_json(tmp_path/"changed-basis.json",old)
    def forbidden(*a,**k):
        pytest.fail("A changed completed basis cannot launch numerical work")
    with pytest.raises(ValueError,match="cannot be reused|photometry record differs"):
        subject.run_image_preparation(config,producer=forbidden,colour_path_producer=forbidden)


def test_per_run_protocol_has_explicit_reuse_source_not_an_untyped_path(tmp_path):
    from darsia.fluidflower.pipeline.series_setup_protocol import validate_setup_operation,FORMAT
    root=str(tmp_path.resolve())
    pin=_write_json(tmp_path/"pin.json",{})
    project=_write_json(tmp_path/"series.json",{})
    config=dict(format=FORMAT,run=subject.IMAGE_RUN,operation=subject.OPERATION,project_root=root,
        project_manifest=project,adopted_revision=pin,series_manifest=pin,native_parameters=pin,
        metadata_output=dict(maximum_output_bytes=10000,maximum_file_bytes=1000),setup_plan=pin,
        target_machine="machine",output_scope=dict(metadata_roots=[root],scratch_roots=[]),
        runs=["one"],job_root=root,choice_authority=pin,output_root=root,
        colour_path_scope="per-run",reuse_image_basis=pin)
    validate_setup_operation(config)
    changed=deepcopy(config);changed["reuse_image_basis"]=root
    with pytest.raises(ValueError):
        validate_setup_operation(changed)
    changed=deepcopy(config);changed.pop("colour_path_scope")
    with pytest.raises(ValueError,match="explicit per-run"):
        validate_setup_operation(changed)


def retired_partial_basis(tmp_path, monkeypatch):
    from darsia.fluidflower.pipeline.setup_colour_paths import PARTIAL_REUSE_FORMAT
    config, old, _paths = completed_basis(tmp_path, monkeypatch)
    leaf = {key: _write_json(tmp_path/(key+".json"), dict(original_leaf=key))
            for key in ("claim", "owner", "retirement", "stop", "failure")}
    partial = {key: deepcopy(old[key]) for key in (
        "choice", "operations", "choice_authority", "raw_execution_identity", "correction_order", "photometry_record")}
    partial.update(format=PARTIAL_REUSE_FORMAT, complete=False, source_root=str(tmp_path/"retired-attempt"),
        file_versions={}, image_basis=dict(runs=old["runs"]), failed_attempt=leaf["failure"],
        retirement={key: leaf[key] for key in ("claim", "owner", "retirement", "stop")},
        optical_sources={run:{} for run in old["runs"]}, frame_selection=dict(runs={run:[] for run in old["runs"]}))
    config["reuse_image_basis"] = _write_json(tmp_path/"retired-partial.json", partial)
    return config, partial


def test_failed_retired_basis_reuses_fits_and_delegates_actual_partial_validation(tmp_path, monkeypatch):
    config, partial = retired_partial_basis(tmp_path, monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("The genuine finished photometry must not decode or fit again")
    calls = []
    def colour_paths(operations, prepared, authority, folder, **options):
        assert prepared["runs"] == partial["image_basis"]["runs"]
        assert prepared["record"] == partial["photometry_record"]
        assert options["reuse_partial"] == config["reuse_image_basis"]
        assert "reuse_receipt" not in options
        calls.append(options["workers"])
        folder.mkdir(parents=True)
        return _write_json(folder/"receipt.json", dict(complete=True, actual_new_paths=True))
    # These leaves and numerical sample/version payloads belong to the optical
    # producer's validation. The orchestrator reads only its typed manifest and
    # photometry record, avoiding a duplicate scan of the retired attempt.
    original_json = DataReader.json
    audit_paths = {pin["path"] for pin in partial["retirement"].values()} | {partial["failed_attempt"]["path"]}
    def metadata_only(self, pin):
        if pin.get("path") in audit_paths:
            pytest.fail("Do not duplicate the optical producer's retirement audit")
        return original_json(self, pin)
    monkeypatch.setattr(DataReader, "json", metadata_only)
    progress = []
    result = DataReader().json(subject.run_image_preparation(config, workers=6, producer=forbidden,
        colour_path_producer=colour_paths, progress=progress.append))
    assert calls == [6] and result["complete"] is True
    assert result["runs"] == partial["image_basis"]["runs"]
    assert result["reused_image_basis"] == config["reuse_image_basis"]
    assert any(row["step"] == "reusing_retired_image_corrections" for row in progress)
    assert not any(row["step"] == "reusing_completed_image_corrections" for row in progress)
    retained = DataReader().json(config["reuse_image_basis"])
    assert retained == partial and retained["complete"] is False


@pytest.mark.parametrize("change", ["complete", "format", "choice", "operations", "choice_authority",
    "raw_execution_identity", "correction_order", "source_root", "file_versions", "image_basis",
    "optical_sources", "frame_selection", "failed_attempt", "retirement", "stop", "photometry"])
def test_failed_partial_refuses_incompatible_or_untyped_reuse_before_numerics(tmp_path, monkeypatch, change):
    config, partial = retired_partial_basis(tmp_path, monkeypatch)
    if change == "photometry":
        partial["photometry_record"] = _write_json(tmp_path/"different-photo.json", dict(runs={}))
    elif change == "stop":
        del partial["retirement"]["stop"]
    elif change == "complete":
        partial[change] = True
    else:
        partial[change] = None
    config["reuse_image_basis"] = _write_json(tmp_path/"changed-partial.json", partial)
    def forbidden(*args, **kwargs):
        pytest.fail("An invalid retained basis cannot launch any numerical work")
    with pytest.raises(ValueError, match="cannot be reused|typed retired partial|photometry record differs"):
        subject.run_image_preparation(config, producer=forbidden, colour_path_producer=forbidden)
