"""N1 selection integration checks through the original portable binding.

Plan admission is injected to isolate the Jobs selection boundary. Bound P1
configuration/geometry use the original readers and validator, and saved image
choices use the original B4 resolver. Portable-creation fixtures add opaque
authored camera bytes for the original catalogue's stat checks. Completion
products and scientific computations are absent; the final producer is inert.
"""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from darsia.fluidflower import job_filesystem, jobs_metadata as model
from darsia.fluidflower import series_image_selection as b4
from darsia.fluidflower.pipeline import alias_job, job_inputs, staged_data
from .gui.test_jobs_metadata import bytes_of, put, selection_fixture
from .test_staged_data import phase1_data  # noqa: F401


@pytest.fixture
def production(tmp_path, phase1_data, monkeypatch):
    root, _, original_frames, resolved, _ = selection_fixture(tmp_path)
    manifest = root / "series.toml"
    manifest.write_bytes(manifest.read_bytes() + b'[roots]\nseries="."\n')
    source = phase1_data
    template = deepcopy(source["frames"][0])
    source.update(run="ac52", frames=[], ordered_case_ids=[])
    for i, (original, saved) in enumerate(zip(original_frames, resolved["selected"])):
        frame = deepcopy(template)
        digest = "cd"[i] * 64
        frame.update(case_id=original["case_id"], frame_index=saved["frame_index"],
                     elapsed_h=saved["elapsed_h"])
        frame["source"].update(
            run="ac52", work_index=frame["frame_index"], elapsed_h=frame["elapsed_h"],
            source_sha256=digest, previous_source_sha256="bc"[i] * 64,
            source_binding=dict(source_sha256=digest))
        frame["native_task"]["geometry"] = put(root / f"geometry-{i}.json", dict(
            camera_jpeg_path=saved["source_path"], camera_jpeg_sha256=digest))
        source["frames"].append(frame)
        source["ordered_case_ids"].append(frame["case_id"])
    source_bound = put(root / "phase1.json", source)
    config = dict(
        format="darsia-n1-run-v1", run="ac52", precision="float32",
        phase1=dict(configuration=source_bound),
        frames=[dict(case_id=f["case_id"], frame_index=f["frame_index"],
                     elapsed_h=f["elapsed_h"], source_binding=deepcopy(f["source"]["source_binding"]))
                for f in source["frames"]])
    declaration = dict(run="ac52", configuration=put(root / "phase2.json", config))
    value = dict(pipeline="n1", job=dict(precision="float32"), runs=[declaration])
    plan_path = root / "execution.toml"
    plan_path.write_text(
        'format="darsia-series-v1"\nadopted_for_execution=true\npipeline="n1"\n'
        'job_id="tiny"\n[job]\nprecision="float32"\n', encoding="utf-8")
    plan = model.descriptor(plan_path)
    metadata = b4.RunImageMetadata(
        "ac52", tuple(dict(source_path=s["source_path"], elapsed_h=s["elapsed_h"],
                           frame_index=s["frame_index"], image_id=f["case_id"])
                      for s, f in zip(resolved["selected"], source["frames"])),
        0.5, 1.0, tuple(resolved["source_metadata"]), (), str(root))
    b4.save_run_image_selection(root, metadata, [f["source_path"] for f in metadata.frames],
                                expected_bytes=manifest.read_bytes())

    def admit(path, *, expected_sha256):
        assert path == plan_path and expected_sha256 == plan["sha256"]
        bound = declaration["configuration"]
        return deepcopy(value), [model.JobSnapshotReader().read(Path(bound["path"]), bound)], plan

    def provider(runs):
        assert runs == ["ac52"]
        return [b4.resolve_saved_selection(root, "ac52", metadata)]

    def rewrite():
        declaration["configuration"] = put(root / "phase2.json", config)

    def forbidden(*args, **kwargs):
        pytest.fail("Selection admission must not verify completion or start science")

    monkeypatch.setattr(staged_data, "verify_phase1_completion", forbidden)
    return dict(root=root, plan=plan, value=value, config=config, source=source,
                declaration=declaration, metadata=metadata, admit=admit,
                provider=provider, rewrite=rewrite)


@pytest.mark.parametrize("aliases", [False, True])
def test_n1_original_sources_keep_phase_two_descriptor_and_read_only_metadata(production, monkeypatch, aliases):
    p = production
    if aliases:
        geometry = p["source"]["frames"][0]["native_task"]["geometry"]
        logical_geometry = dict(geometry, path="C:/unavailable-source/geometry.json")
        p["source"]["frames"][0]["native_task"]["geometry"] = logical_geometry
        source = put(p["root"] / "phase1.json", p["source"])
        logical_source = dict(source, path="C:/unavailable-source/phase1.json")
        p["config"]["phase1"]["configuration"] = logical_source
        p["value"]["data_aliases"] = [dict(logical=logical_source, physical=source)]
        p["config"]["data_aliases"] = [dict(logical=logical_geometry, physical=geometry)]
        p["rewrite"]()
    original_read = job_inputs.InputCatalog._read_metadata
    opened = []

    def read(path):
        opened.append(Path(path).name)
        return original_read(path)

    monkeypatch.setattr(job_inputs.InputCatalog, "_read_metadata", staticmethod(read))
    before = bytes_of(p["root"])
    binding = model.validate_current_selection(
        p["plan"], series_root=p["root"], series_id="unit",
        selection_provider=p["provider"], admitter=p["admit"])
    run = binding["runs"][0]
    assert run["configuration"] == p["declaration"]["configuration"]
    assert run["configuration"] != p["config"]["phase1"]["configuration"]
    assert [c["case_id"] for c in run["cases"]] == ["ac52_f1", "ac52_f9"]
    assert [c["elapsed_h"] for c in run["cases"]] == [0.25, 2.0]
    assert [c["b4_input_index"] for c in run["cases"]] == [0, 1]
    assert [c["original_camera_source"] for c in run["cases"]] == [
        dict(path=f["source_path"], sha256="cd"[i] * 64)
        for i, f in enumerate(p["metadata"].frames)]
    assert opened == ["phase1.json", "geometry-0.json", "geometry-1.json"]
    assert binding["execution_entries_unchanged"] and not binding["raw_hashed_by_this_UI"]
    assert bytes_of(p["root"]) == before
    assert not any(Path(f["source_path"]).exists() for f in p["metadata"].frames)


def _refuses_before_producer(production, monkeypatch, *, match, provider=None):
    p = production
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        job_filesystem.SharedFolderAdmission(True, "smb", "unit transport", "synthetic admission"))
    calls = []
    target = p["root"].parent / "never-created-job"
    before = bytes_of(p["root"])
    with pytest.raises(ValueError, match=match):
        model.create_job_from_plan(
            p["plan"], target, display_name="tiny", series_label="unit",
            series_root=p["root"], series_id="unit", admitter=p["admit"],
            selection_provider=provider or p["provider"],
            preparer=lambda *args, **kwargs: calls.append((args, kwargs)))
    assert not calls and not target.exists()
    assert bytes_of(p["root"]) == before


@pytest.mark.parametrize("change", ["source", "time", "order"])
def test_saved_selection_mismatch_stops_original_producer(production, monkeypatch, change):
    rows = production["provider"](["ac52"])
    selected = rows[0]["selected"]
    if change == "source":
        selected[0]["source_path"] = str(production["root"] / "another-camera.jpg")
    elif change == "time":
        selected[0]["elapsed_h"] = 0.5
    else:
        selected.reverse()
    _refuses_before_producer(production, monkeypatch, match="source path/order/acquisition",
                            provider=lambda runs: deepcopy(rows))


def test_original_blacklist_resolver_refusal_stops_producer(production, monkeypatch):
    production["metadata"].frames[0]["blacklisted"] = True
    _refuses_before_producer(production, monkeypatch, match="not an eligible protocol source")


@pytest.mark.parametrize("change", ["alias_conflict", "plan_precision", "source_precision"])
def test_alias_and_precision_disagreement_stops_original_producer(production, monkeypatch, change):
    p = production
    source = p["config"]["phase1"]["configuration"]
    if change == "alias_conflict":
        logical = dict(source, path="C:/unavailable-source/phase1.json")
        p["value"]["data_aliases"] = [dict(logical=logical, physical=source)]
        p["config"]["data_aliases"] = [dict(
            logical=logical, physical=dict(source, path=str(p["root"] / "different.json")))]
        p["config"]["phase1"]["configuration"] = logical
    elif change == "plan_precision":
        p["config"]["precision"] = "float64"
    else:
        p["source"]["precision"] = "float64"
        p["config"]["phase1"]["configuration"] = put(Path(source["path"]), p["source"])
    p["rewrite"]()
    _refuses_before_producer(production, monkeypatch, match="aliases" if change == "alias_conflict" else "precision")


def _write_aliased_plan(p, aliases):
    """Keep the injected plan admission and actual bound TOML aliases identical."""
    p["value"]["data_aliases"] = deepcopy(aliases)
    lines = ['format="darsia-series-v1"', 'adopted_for_execution=true',
             'pipeline="n1"', 'job_id="tiny"', '[job]', 'precision="float32"']
    for alias in aliases:
        lines.append('[[data_aliases]]')
        for side in ("logical", "physical"):
            lines.append('[data_aliases.' + side + ']')
            lines.extend(key + "=" + json.dumps(alias[side][key])
                         for key in ("path", "sha256", "size_bytes"))
    path = p["root"] / "execution.toml"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    p["plan"] = model.descriptor(path)


@pytest.fixture
def portable_production(production):
    p = production
    previous = p["source"]["phase"]["initial_previous"]["source"]["source_sha256"]
    for index, (frame, saved) in enumerate(zip(p["source"]["frames"], p["metadata"].frames)):
        path = Path(saved["source_path"])
        path.write_bytes(b"authored opaque acquisition bytes " + str(index).encode("ascii"))
        camera = model.descriptor(path)
        frame["source"].update(source_sha256=camera["sha256"],
            previous_source_sha256=previous, source_binding=dict(source_sha256=camera["sha256"]))
        previous = camera["sha256"]
        frame["native_task"]["geometry"] = put(p["root"] / f"geometry-{index}.json",
            dict(camera_jpeg_path=str(path), camera_jpeg_sha256=camera["sha256"]))
        p["config"]["frames"][index]["source_binding"] = deepcopy(frame["source"]["source_binding"])
    p["config"]["phase1"]["configuration"] = put(p["root"] / "phase1.json", p["source"])
    p["rewrite"]()
    physical = deepcopy(p["declaration"]["configuration"])
    logical = dict(physical, path="C:/unavailable-source/phase2.json")
    p["declaration"]["configuration"] = logical
    p["configuration_alias"] = dict(logical=logical, physical=physical)
    _write_aliased_plan(p, [p["configuration_alias"]])

    def admit(path, *, expected_sha256):
        assert path == Path(p["plan"]["path"]) and expected_sha256 == p["plan"]["sha256"]
        _, _, roots = alias_job._manifest(model.descriptor(p["root"] / "series.toml"))
        catalog = job_inputs.InputCatalog(roots, p["value"]["data_aliases"])
        configuration = catalog.doc(p["declaration"]["configuration"], "configuration", "fixture.phase2")
        return deepcopy(p["value"]), [configuration], p["plan"]

    p["admit"] = admit
    return p


def test_aliased_n1_configuration_reaches_portable_binding_before_inert_producer(portable_production, monkeypatch):
    p = portable_production
    monkeypatch.setattr(job_filesystem, "inspect_shared_folder", lambda _path:
        job_filesystem.SharedFolderAdmission(True, "smb", "unit transport", "synthetic admission"))
    original_portable = alias_job.portable_selection_binding
    conversions, calls = [], []

    def capture(binding, **kwargs):
        converted = original_portable(binding, **kwargs)
        conversions.append((deepcopy(binding), converted))
        return converted

    monkeypatch.setattr(alias_job, "portable_selection_binding", capture)
    target = p["root"].parent / "inert-created-job"
    before = bytes_of(p["root"])

    def preparer(path, destination, *, expected_sha256, series_manifest, storage_profile):
        assert len(conversions) == 1 and not destination.exists()
        assert expected_sha256 == p["plan"]["sha256"] and storage_profile == "lean"
        assert series_manifest == model.descriptor(p["root"] / "series.toml")
        calls.append((path, destination))
        put(destination / "job.json", dict(producer="inert metadata fixture", expected_runs=["ac52"]))
        return dict(complete=True, numerical_calls_started=False,
            series=conversions[0][1]["execution_plan"])

    result = model.create_job_from_plan(p["plan"], target, display_name="Production",
        series_label="unit", series_root=p["root"], series_id="unit",
        selection_provider=p["provider"], admitter=p["admit"], preparer=preparer,
        storage_profile="lean", machine="Main")
    assert calls == [(Path(p["plan"]["path"]), target)]
    original_binding, portable = conversions[0]
    assert original_binding["runs"][0]["configuration"] == p["declaration"]["configuration"]
    physical = p["configuration_alias"]["physical"]
    assert portable["runs"][0]["configuration"] == dict(root_alias="series",
        relative_path="phase2.json", sha256=physical["sha256"], bytes=physical["size_bytes"],
        role="configuration")
    assert [row["case_id"] for row in portable["runs"][0]["cases"]] == ["ac52_f1", "ac52_f9"]
    assert model.read_json(target / "ui" / "image_selection_binding.json") == portable
    assert result["worker_started"] is False and bytes_of(p["root"]) == before
    assert not Path(p["declaration"]["configuration"]["path"]).exists()


@pytest.mark.parametrize("problem,match", [
    ("alias_sha", "alias changes source SHA"),
    ("alias_size", "alias changes source size"),
    ("physical_sha", "metadata SHA/size changed"),
    ("physical_size", "input size changed"),
    ("outside_root", "No declared root"),
    ("conflict", "Conflicting source input aliases"),
])
def test_aliased_phase_two_identity_and_roots_refuse_before_producer(portable_production, monkeypatch, problem, match):
    p = portable_production
    aliases = [deepcopy(p["configuration_alias"])]
    physical_path = Path(aliases[0]["physical"]["path"])
    if problem == "alias_sha":
        aliases[0]["physical"]["sha256"] = "f" * 64
    elif problem == "alias_size":
        aliases[0]["physical"]["size_bytes"] += 1
    elif problem == "physical_sha":
        raw = physical_path.read_bytes()
        physical_path.write_bytes(raw.replace(b"float32", b"float64"))
    elif problem == "physical_size":
        physical_path.write_bytes(physical_path.read_bytes() + b" ")
    else:
        alternative = (p["root"].parent if problem == "outside_root" else p["root"]) / "other-phase2.json"
        alternative.write_bytes(physical_path.read_bytes())
        moved = model.descriptor(alternative)
        if problem == "outside_root":
            aliases[0]["physical"] = moved
        else:
            aliases.append(dict(logical=deepcopy(aliases[0]["logical"]), physical=moved))
    _write_aliased_plan(p, aliases)
    _refuses_before_producer(p, monkeypatch, match=match)


def test_portable_configuration_cannot_take_an_alias_absent_from_bound_plan(portable_production, monkeypatch):
    p = portable_production
    path = Path(p["plan"]["path"])
    # Deliberately keep the injected admission's alias while omitting it from
    # the actual bound plan: only that plan may supply the portable locator.
    path.write_text(path.read_text(encoding="utf-8").split('[[data_aliases]]', 1)[0], encoding="utf-8")
    p["plan"] = model.descriptor(path)
    _refuses_before_producer(p, monkeypatch, match="No declared root")
