"""Temporary approved-version -> Store -> export proof; no experiment data."""
from copy import deepcopy
from dataclasses import replace
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
import darsia

from darsia.fluidflower import corrected_setup as export
from darsia.fluidflower import geometry_facies as geometry
from darsia.fluidflower import geometry_raster_sources as sources
from darsia.fluidflower import indicator_response
from darsia.fluidflower import layer_painting as lp
from darsia.fluidflower import layer_painting_manual as manual
from darsia.fluidflower import layer_painting_polygons_only as polygons
from darsia.fluidflower import layer_painting_version_archive as versions
from darsia.fluidflower import map_adoption as adoption
from darsia.fluidflower import painted_adoption_table as contract
from darsia.fluidflower import painted_adoption_geometry as geometry_proof
from darsia.fluidflower import round5_geometry
from darsia.fluidflower.published_dataset import simulator_geometry
from darsia.fluidflower.series_geometry import RigFrame

from .test_geometry_imported_rasters import Project
from .test_geometry_registered_export_apply import registered_setup  # noqa: F401
from .test_map_adoption_store import MAP, IDENTITY, SHIFTED


_REAL_PAYLOAD_LINK = geometry_proof._require_corrected_payload_link


@pytest.fixture(autouse=True)
def synthetic_payload_admission(monkeypatch):
    """Write explicit RGB parent grids; never bypass a product admission check.

    The shared registered fixture predates the RAW basis and used generic RGB
    arrays without an optical type. Scalar grids, dimensions, pixels and the
    original per-run configurations keep their existing values. The real owned
    producer below supplies the NEW certificate; the old NPZ guard stays closed.
    """
    from . import test_geometry_registered_export_apply as registered
    original_archive = registered._archive

    def archive(path, values):
        if values.ndim == 3:
            darsia.OpticalImage(values, color_space="RGB", space_dim=2,
                indexing="ij", dimensions=[.49, .9], origin=[0., .49],
                series=False).save(path)
        else:
            original_archive(path, values)

    monkeypatch.setattr(registered, "_archive", archive)
    return _REAL_PAYLOAD_LINK


@pytest.fixture
def imported(tmp_path, request):
    """The original registered fixture, with an explicit full extra-column CSV."""
    original = tmp_path / "original"
    original.mkdir()
    arrays = {"labels": np.array([[1, 1, 1], [2, 2, 2]], np.uint8),
              "facies": np.array([[1, 3, 1], [6, 2, 4]], np.uint8),
              "baseline": np.zeros((2, 3, 3), np.float32),
              "depth": np.full((2, 3), .01, np.float64)}
    for name, value in arrays.items():
        np.savez(original / (name + ".npz"), array=value)
    csv_path = original / "full.csv"
    raw = (b'id,name,porosity,permeability,swi CO2,krel_gas,pc [mbar],custom\r\n'
           b'1,Sand1,0.4,10,0.12,0.8,3,keep\r\n2,Sand2,0.3,20,0.13,0.7,4,coarse\r\n'
           b'3,Boundary3,0.2,30,0.14,0.6,5,unused\r\n4,Boundary4,0.1,40,0.15,0.5,6,unused\r\n'
           b'6,Boundary6,0.5,60,0.16,0.4,7,unused\r\n')
    mode = getattr(request, "param", "ordinary")
    if mode in {"agree", "conflict"}:
        raw += b'7,Sand1,0.6,99,' + (b'0.91' if mode == "conflict" else b'0.12') + b',0.8,3,keep\r\n'
    if mode == "outside_source":
        raw += b'9,Outside the rig frame,0,,0.77,0.44,9,explicit-frame\r\n'
    csv_path.write_bytes(raw)
    diagnosis = original / "diagnosis.json"
    diagnosis.write_bytes(b'{"synthetic":true}\n')
    project = Project(tmp_path / "workspace")
    config = geometry.default_geometry_config(project)
    config.update(baseline_path=str(original / "baseline.npz"), labels_path=str(original / "labels.npz"),
        depth_path=str(original / "depth.npz"), facies_source_path=str(original / "facies.npz"),
        facies_properties_source_path=str(csv_path), permeability_unit="source_unit",
        regions=[dict(label_id=1, facies_id=1, name="Sand1", analysis_role="porous_medium",
                      enabled=True, porosity=.4, permeability=10.),
                 dict(label_id=2, facies_id=2, name="Sand2", analysis_role="porous_medium",
                      enabled=True, porosity=.3, permeability=20.)])
    config = sources.bind_imported_rasters(config,
        sources={key: sources.descriptor(config[sources.PATH_KEYS[key]]) for key in sources.KEYS},
        diagnosis=sources.descriptor(diagnosis))
    return SimpleNamespace(project=project, config=config, arrays=arrays, original=original)


def _basis_inputs(data, folder, *, bind_profile=True):
    """Private fake acquisitions and actual series/profile/source descriptors."""
    folder.mkdir()
    raw_root = folder / "raw"
    raw_root.mkdir()
    photos = {}
    for run in ("one", "two"):
        photo = raw_root / (run + ".ARW")
        photo.write_bytes(("unit-only fake RAW acquisition " + run).encode())
        photos[run] = photo
    policy = dict(minimum_matches=4, minimum_inlier_fraction=.8,
        heldout_zones=[["sand", [0, 0, 2, 1], 1]], heldout_minimum_matches=2,
        heldout_min_correlation=.7)
    profile = round5_geometry.write_new_json(folder / "camera.json", dict(
        registration=dict(common=dict(profiles={"camera": {}},
            sensor=dict(camera_wh=[3, 2]), review=policy, matching={})),
        source=dict(profiles=dict(one="camera", two="camera"))))
    manifest = data.series / "series.toml"
    # Only this private fixture's series metadata is extended. Original run
    # TOMLs, Store revisions and the adopted parent NPZ bytes are never rewritten.
    import toml
    document = toml.loads(manifest.read_text())
    document.setdefault("schema_version", 1)
    if bind_profile:
        document["native"] = dict(configuration=profile["path"], sha256=profile["sha256"])
    else:
        document.pop("native", None)
    manifest.write_text(toml.dumps(document), encoding="utf-8")
    data.project.manifest.setdefault("tools", {}).setdefault("native_series", {})["series_root"] = str(data.series)
    protocols = []
    for role in ("imaging", "injection", "pressure_temperature"):
        path = folder / (role + ".csv")
        path.write_bytes(b"time,value\n0,1\n")
        protocols.append(dict(role=role, folder=None, sheet=None, source=sources.descriptor(path)))
    return SimpleNamespace(folder=folder, raw_root=raw_root, photos=photos,
        profile=profile, protocols=protocols, manifest=manifest)


def _owned_basis(data, inputs, run, to_u, *, native_rgb_scale=None):
    """Use the real owned writer; only decoder/registration/runtime are synthetic.

    This 2x3 producer test proves saved bytes, source/request/release joins and
    full grid compatibility. It makes no native-decoder or optical qualification
    claim. No scientific reader, ownership check or authority gate is mocked.
    """
    from darsia._opencv_runtime import scientific_thread, completed_frame
    from darsia.fluidflower import rig_raw_build as build
    from darsia.fluidflower.native import raw_positioning, registration
    folder = inputs.folder / run
    folder.mkdir()
    configuration = Path(data.config["frame_binding"]["runs"][run]["configuration"]["path"])
    camera = dict(configuration=inputs.profile["path"], sha256=inputs.profile["sha256"],
                  profile_id="camera", run_id=run)
    output_root = folder / "products"
    output_root.mkdir()
    destination = output_root / "new_basis"
    request = dict(format=build.BASIS_REQUEST_FORMAT, run_id=run,
        configuration_files=list(build.capture_configuration_sources(configuration)),
        series_context=dict(root=str(data.series), manifest=sources.descriptor(inputs.manifest)),
        baseline=sources.descriptor(inputs.photos[run]), profile=inputs.profile,
        operation_overrides={"data.raw_camera": camera}, protocol_inputs=inputs.protocols,
        output=dict(destination=str(destination), intended_root=str(output_root),
            forbidden_raw_roots=[str(inputs.raw_root)], maximum_output_bytes=10_000_000,
            maximum_file_bytes=2_000_000), shape_basis_certificate=None)
    request_binding = round5_geometry.write_new_json(folder / "request.json", request)
    # Unit DATA supplier only: bind real named producer/writer/reader sources,
    # not the full installed package. Whole-package and native qualification
    # remain separate integration evidence; every admitted source file is real.
    package = Path(build.__file__).resolve().parent
    implementation = [sources.descriptor(path) for path in sorted([
        package / "rig_raw_build.py", package / "rig_raw_write.py",
        package / "rig_raw_input.py", package / "raw_shape_basis_reader.py",
        package / "native/raw_positioning.py", package / "native/registration.py",
        package.parent / "corrections/typecorrection.py",
    ])]
    closure = dict(base_commit="1" * 40, tree="2" * 40, files=deepcopy(implementation))
    # Owner-release mode is current authority. This is explicit synthetic test
    # evidence, not a statement that this fixture source was released for a rig.
    release = round5_geometry.write_new_json(folder / "release.json", dict(
        source_closure=closure, operation_request=request_binding,
        released_utc=datetime.now(timezone.utc).isoformat()))
    authority = dict(source_closure=closure, operation_request=request_binding, root_release=release)
    runtime_root = folder / "runtime"
    runtime_root.mkdir()
    executable = runtime_root / "python-wrapper.exe"
    executable.write_bytes(b"synthetic executable identity; never executed")
    libraries = {}
    for name in ("cv2", "rawpy", "numpy"):
        path = runtime_root / (name + ".py")
        path.write_bytes(b"# synthetic wrapper identity\n")
        libraries[name] = dict(version="synthetic-unit-wrapper", module=sources.descriptor(path))
    runtime = dict(python="synthetic unit runtime", executable=sources.descriptor(executable),
        identity_scope="Python module wrappers and executable file identities",
        native_binary_equivalence_claimed=False, production_qualification_claimed=False,
        libraries=libraries)
    fields = data.config["frame_binding"]["runs"][run]["fields"]
    old = darsia.imread(Path(fields["shape_corrected_baseline"]["path"]))
    metadata = old.metadata()
    metadata.pop("type", None)
    linear = np.arange(18, dtype=np.float32).reshape(3, 2, 3) * np.float32(10)
    if native_rgb_scale is not None:
        linear = linear * np.float32(native_rgb_scale)
    image = darsia.OpticalImage(np.moveaxis(linear, 0, -1) / np.float32(1000), **metadata).img_as(float)
    correction = darsia.TypeCorrection(np.float32)
    sensor = SimpleNamespace(metadata=dict(white_level=1100, black_level=100, orientation=1))
    quality = dict(fit_n=8, fit_inlier_fraction=1., sampled_jacobian_min=1.,
        heldout=dict(sand=dict(n=3, p50_px=.1, p95_px=.2, max_px=.3, median_match_correlation=.9)))
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(build, "_source_snapshot", lambda: deepcopy(implementation))
        patch.setattr(build, "_runtime_identity", lambda: deepcopy(runtime))
        patch.setattr(raw_positioning, "read_sensor", lambda *args, **kwargs: sensor)
        patch.setattr(raw_positioning, "camera_u", lambda *args, **kwargs: linear.copy())
        patch.setattr(registration, "review_to_camera", lambda *args, **kwargs: (np.asarray(to_u).copy(), deepcopy(quality)))
        context = build.begin_raw_rig_build(configuration, inputs.photos[run], camera, destination,
            configuration_sources=build.capture_configuration_sources(configuration),
            request_binding=request_binding, artifact_kind="raw_shape_basis")
        rig = SimpleNamespace(_raw_build_id=context.operation_id, raw_camera=camera,
            raw_baseline_sha256=context.raw_source["sha256"], baseline_path=inputs.photos[run],
            shape_corrected_baseline=image, shape_corrections=[correction],
            _raw_setup_input_metadata=image.metadata(),
            _raw_basis_opencv=scientific_thread("unit-shape-basis:setup"),
            _raw_reading_opencv=completed_frame(scientific_thread("unit-shape-basis:reading-before"),
                                               "unit-shape-basis:reading-after"))
        context.begin_setup(rig, destination / "figures")
        context.output.begin_save(rig, destination)
        context.output.image(image, destination / "shape_corrected_baseline.npz")
        context.output.correction(correction, destination / "shape_correction_0_typecorrection.npz")
        with context.output.open(destination / "meta_data.json") as stream:
            stream.write(json.dumps(dict(baseline_path=str(inputs.photos[run]), raw_camera=camera,
                raw_baseline_sha256=context.raw_source["sha256"], raw_build_id=context.operation_id)).encode())
        context.output.finish_save()
        certificate = context.publish(rig)
    return certificate, authority


def _models(data, folder, inputs, *, unmapped=False, mutate=None, native_rgb_scale=None):
    folder.mkdir()
    edges = round5_geometry.write_new_json(folder / "edges.json", {"synthetic": True})
    parent = sources.descriptor(Path(data.revision["artifacts"]["facies_properties_path"]).parent / "revision.json")
    runs = {}
    for run, matrix in (("one", IDENTITY), ("two", SHIFTED)):
        photograph = sources.descriptor(inputs.photos[run])
        frame = dict(H_camera_to_mm=np.linalg.inv(np.asarray(matrix)).tolist(),
                     sand_area_factor14=1.0, explicit_certificate={"synthetic": True})
        saved = round5_geometry.write_new_json(folder / (run + "-source.json"),
            dict(run=run, conditional_frame=frame))
        to_native = np.asarray(SHIFTED if unmapped and run == "one" else IDENTITY)
        renderer, authority = _owned_basis(data, inputs, run, np.linalg.inv(to_native),
                                            native_rgb_scale=native_rgb_scale)
        value = round5_geometry.reference_model(run, frame, reference_image=photograph,
            camera_profile=dict(inputs.profile, profile_id="camera", camera_wh=[3, 2]),
            source_model=saved, edge_review=edges,
            baseline_configuration=data.config["frame_binding"]["runs"][run]["configuration"],
            baseline_native_renderer=renderer, raw_shape_basis_authority=authority,
            raw_shape_basis_workspace=parent)
        if mutate is not None:
            mutate(run, value, folder)
        bound = round5_geometry.write_new_json(folder / (run + ".json"), value)
        runs[run] = dict(reference_model=bound)
    index = folder / "index.json"
    round5_geometry.write_new_json(index, dict(format=round5_geometry.FORMAT, runs=runs))
    return index


def prepared_painting(data, tmp_path, *, same_facies=False, water=False, outside=False, unmapped=False,
                      method="manual", mutate_model=None, preparation_frame=None,
                      corrected_source=False, raw_source=False, fixture=False, during_prepare=None, continuation=False,
                      outside_targets=False, own_frame_change=None, native_rgb_scale=None):
    inputs = _basis_inputs(data, tmp_path / "owned_basis", bind_profile=not raw_source)
    image = inputs.photos["one"]
    if corrected_source:
        image = Path(data.config["canonical_import_parent"]["imported_configuration"]["baseline_path"])
    session = lp.PaintingSession("one", image_path=str(image), image_stamp=lp.image_stamp(image),
                                 image_size=(3, 2), permeability_unit="mD")
    session.facies = {"Sand2": lp.FaciesEntry("free_water" if water else "porous_medium",
                                            None if water else .22, None if water else 28.),
                      "Sand1": lp.FaciesEntry("porous_medium", .44, 12.)}
    session.layers = [lp.PaintedLayer(5, "Upper", "#aaaaaa", "Sand1"),
                      lp.PaintedLayer(6, "Lower", "#bbbbbb", "Sand1" if same_facies else "Sand2")]
    if same_facies:
        session.facies.pop("Sand2")
    frame = RigFrame(corners={"top_left": (0., 1.) if outside else (0., 0.),
        "top_right": (0., 2.), "bottom_right": (1., 2.),
        "bottom_left": (1., 1.) if outside else (1., 0.)},
        width_m=.9, height_m=.49, run_id="one", image=str(image), image_size=(3, 2))
    layers = np.array([[0, 5, 6], [0, 5, 6]], np.uint8) if outside else MAP.copy()
    if fixture:
        assert method == "manual"
        session.facies["Boundary3"] = lp.FaciesEntry("colorchecker", None, None)
        session.layers.extend([lp.PaintedLayer(7, "Hidden sand", "#cccccc", "Sand1"),
                               lp.PaintedLayer(8, "Fixture", "#dddddd", "Boundary3")])
        points = [(0., 0.), (0., 1.), (1., 1.), (1., 0.)]
        session.polygons = [lp.Polygon(7, points.copy()), lp.Polygon(8, points.copy())]
    inside = lp.frame_mask(frame, layers.shape, (1., 1.))
    if method == "manual":
        saved_map = manual.ManualMap(layers.copy(), inside, np.zeros(layers.shape, bool), (3, 2))
        version = manual.save_manual_version(data.project, session, saved_map, name="Synthetic painting")
        approval = manual.approve_manual_version(version)
    elif method == "classifier":
        proposal = lp.LayerProposal(layers.copy(), np.full(layers.shape, 255, np.uint8),
                                    (3, 2), lp.proposal_digest(session, frame))
        layers = manual.painted_layers(session, proposal, frame=frame)
        version = lp.save_version(data.project, session, proposal, [], name="Synthetic proposal")
        approval = lp.approve_version(version)
    else:
        assert method == "polygons"
        session.polygons = [lp.Polygon(5, [(0., 0.), (0., 1.), (1., 1.), (1., 0.)]),
                            lp.Polygon(6, [(0., 1.), (0., 2.), (1., 2.), (1., 1.)])]
        proposal = polygons.make_proposal((3, 2), session, frame=frame)
        layers = polygons.painted_layers(session, proposal, frame=frame)
        version = versions.save_map_version(data.project, session, proposal, [],
            name="Synthetic polygons", method=polygons.METHOD, proposal_writer=polygons.save_proposal)
        approval = polygons.approve_version(version)
    labels_path = data.config["canonical_import_parent"]["imported_configuration"]["labels_path"]
    grids = {}
    for run, row in data.config["frame_binding"]["runs"].items():
        labels_path = Path(row["fields"]["labels"]["path"])
        grids[run] = dict(metadata=geometry._darsia_metadata(labels_path), shape_hw=[2, 3],
                          source=sources.descriptor(labels_path))
    index = _models(data, tmp_path / "models", inputs, unmapped=unmapped, mutate=mutate_model,
                    native_rgb_scale=native_rgb_scale)
    current_frame = deepcopy(frame)
    if preparation_frame is not None:
        preparation_frame(current_frame)
    # Genuine temporary own-frame/project authority for the new placement policy.
    own_frames = {}
    for run in ("one", "two"):
        own = deepcopy(frame)
        own.run_id, own.image = run, str(inputs.photos[run])
        own_frames[run] = own.to_dict()
    if own_frame_change is not None:
        own_frame_change(own_frames)
    data.project.manifest["tools"]["native_series"]["rig_frames"] = own_frames
    (data.project.root / "series.json").write_text(json.dumps(data.project.manifest), encoding="utf-8")
    fixture_options = {}
    if continuation:
        from darsia.fluidflower import photo_edge_continuation as policy
        binding = contract.capture_version(version, approval, current_frame, layers, project=data.project)
        fixture_options.update(continuation=policy.build_contracts(data.project, index, "one",
            ["one"] if outside and not outside_targets else ["one", "two"],
            [row.layer_id for row in session.layers], target_grid="native", source_frame=current_frame),
            approved_source=binding)
    if fixture:
        from darsia.fluidflower.layer_painting_fixtures import version_fixture_underlay
        underlay = version_fixture_underlay(version, frame=current_frame)
        assert underlay is not None and not np.array_equal(underlay.analysis_layers, layers)
        fixture_options.update(fixture_underlay=underlay, fixture_frame=current_frame)
    result = adoption.prepare_adoption(tmp_path / "preparation", layers,
        reference_run="one", geometry_index=index,
        runs=["one"] if outside and not outside_targets else ["one", "two"],
        roles={layer.layer_id: session.facies[layer.facies].role for layer in session.layers},
        names={layer.layer_id: layer.name for layer in session.layers},
        facies={layer.layer_id: layer.facies for layer in session.layers},
        version={key: version[key] for key in ("name", "kind", "sha256", "created_utc", "folder")}
                | ({"proposal_method": version["proposal_method"]} if "proposal_method" in version else {})
                | {"approval": approval}, current_labels=lambda run: deepcopy(grids[run]),
        painting_version=version, painting_frame=current_frame, painting_project=data.project, progress=during_prepare, **fixture_options)
    return result, version, session, frame, layers


def _saved(data, prepared):
    config, summary = adoption.adoption_candidate(data.revision, prepared["folder"])
    revision = geometry.GeometryRevisionStore(data.project).save_revision(config,
        actor="synthetic", reason="explicit synthetic painting table", preserve_failed_artifacts=True)
    return config, summary, revision


def test_committed_selected_run_reuses_typed_table_without_reading_another_run_raster(
        registered_setup, tmp_path, monkeypatch):
    prepared, *_ = prepared_painting(registered_setup, tmp_path, continuation=True)
    _config, _summary, revision = _saved(registered_setup, prepared)
    foreign = revision["artifacts"]["run_geometry"]["two"]
    foreign_paths = {Path(value).resolve() for key, value in foreign.items() if key.endswith("_path")}
    other = next(row for row in revision["configuration"]["run_geometry"]["runs"] if row["run_id"] == "two")
    foreign_paths.add(Path(other["registration"]["labels"]["path"]).resolve())
    foreign_paths.add(Path(other["source_image_path"]).resolve())
    original_load = np.load
    opened = []
    def load(path, *args, **kwargs):
        name = Path(path) if isinstance(path, (str, Path)) else Path(path.name)
        if name.resolve() in foreign_paths:
            pytest.fail("Apply for one run reopened another run's raster")
        opened.append(name)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", load)
    monkeypatch.setattr(contract, "_derive_uncached",
        lambda *args: pytest.fail("Apply re-derived the already materialized all-run table"))
    from darsia.fluidflower import photo_edge_continuation
    monkeypatch.setattr(photo_edge_continuation, "verify_native_proof",
        lambda *args: pytest.fail("Apply repeated the already completed all-run proof"))
    result = sources.registered_run_sources(registered_setup.project.root, revision, "one")
    assert result["committed_painted_table"]["format"] == "darsia-committed-painted-store-run-v1"
    assert result["files"]["facies_properties"]["sha256"] == revision["configuration"][contract.KEY]["table"]["sha256"]
    assert opened
    sources.verify_registered_sources(result)
    table_path = Path(revision["artifacts"]["facies_properties_path"])
    table_path.write_bytes(table_path.read_bytes().replace(b"Sand1", b"Wrong"))
    with pytest.raises(ValueError, match="changed|differs"):
        sources.verify_registered_sources(result)


def test_uncommitted_painting_candidate_still_requires_complete_derivation(
        registered_setup, tmp_path, monkeypatch):
    prepared, *_ = prepared_painting(registered_setup, tmp_path)
    candidate, _summary = adoption.adoption_candidate(registered_setup.revision, prepared["folder"])
    contract._DERIVED.clear()
    monkeypatch.setattr(contract, "_derive_uncached", lambda *args: (_ for _ in ()).throw(
        ValueError("complete candidate derivation is still required")))
    with pytest.raises(ValueError, match="complete candidate derivation"):
        sources.canonical_parent_payload(candidate, run_id="one")


def test_setup_reuses_complete_selected_raster_proof_without_decoding_again(
        registered_setup, tmp_path, monkeypatch):
    """Real Apply reader: repeated consumers keep metadata, never NPZ arrays."""
    from darsia.fluidflower import setup_source_reads as reads
    from darsia.fluidflower import painted_store_replay
    prepared, *_ = prepared_painting(registered_setup, tmp_path, continuation=True)
    _config, _summary, revision = _saved(registered_setup, prepared)
    original = np.load
    decoded = []

    def counted(path, *args, **kwargs):
        decoded.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", counted)
    with reads.reuse_setup_source_reads():
        result = sources.registered_run_sources(registered_setup.project.root, revision, "one")
        assert decoded, "The first admission must run the original numerical proof"
        decoded.clear()
        for _ in range(5):
            sources.verify_registered_sources(result)
        assert not decoded, "Unchanged Apply consumers must not repeatedly decompress the same rasters"
        retained = reads._PAINTED_STORE.get()
        assert len(retained["entries"]) == 1
        with contract._FILES.fresh():
            painted_store_replay.verify(result)
        assert decoded, "An explicitly fresh boundary still runs the original proof"
        decoded.clear()
        painted_store_replay.verify(result)
        assert decoded, "A forced-fresh proof is not retained as an ordinary success"
        decoded.clear()
        painted_store_replay.verify(result)
        assert not decoded
        path = Path(result["files"]["facies"]["path"])
        path.write_bytes(path.read_bytes() + b"changed")
        with pytest.raises(ValueError, match="changed|differs"):
            painted_store_replay.verify(result)
        assert not retained["entries"]
    assert not retained["entries"]


def test_selected_v2_fixture_underlay_uses_its_published_frame_policy_without_all_run_proof(
        registered_setup, tmp_path, monkeypatch):
    from darsia.fluidflower import photo_edge_continuation, layer_painting_fixtures
    prepared, *_ = prepared_painting(registered_setup, tmp_path, fixture=True, continuation=True)
    _config, _summary, revision = _saved(registered_setup, prepared)
    monkeypatch.setattr(photo_edge_continuation, "verify_native_proof",
        lambda *args: pytest.fail("Selected underlay replay must not repeat all-run admission"))
    foreign = next(row for row in revision["configuration"]["run_geometry"]["runs"] if row["run_id"] == "two")
    forbidden = {Path(foreign["labels_path"]).resolve(), Path(foreign["source_image_path"]).resolve()}
    original = np.load
    def load(path, *args, **kwargs):
        name = Path(path) if isinstance(path, (str, Path)) else Path(path.name)
        if name.resolve() in forbidden:
            pytest.fail("Selected underlay replay opened a foreign run raster")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", load)
    labels = sources._integer_map(revision["artifacts"]["run_geometry"]["one"]["labels_path"])
    result = layer_painting_fixtures.adopted_underlay(revision["configuration"], "one", labels)
    assert result is not None


@pytest.mark.parametrize("method", ["manual", "classifier", "polygons"])
def test_each_actual_saved_map_format_binds_its_approval_and_facies_without_fallback(registered_setup, tmp_path, method):
    prepared, version, _, _, _ = prepared_painting(registered_setup, tmp_path, method=method)
    candidate, summary = adoption.adoption_candidate(registered_setup.revision, prepared["folder"])
    assert prepared["painting_facies"]["record"]["sha256"] == version["sha256"]
    proof = prepared["painting_geometry"]
    assert set(proof["raw_shape_basis"]["authorities"]) == {"one", "two"}
    for row in proof["models"].values():
        model = json.loads(Path(row["path"]).read_bytes())
        grid = model["baseline_native_grid"]
        assert grid["parent_workspace_revision"] == prepared["painting_facies"]["frame_basis"]["workspace_revision"]
        assert "full_review" not in grid["raw_shape_basis_authority"]
        certificate = json.loads(Path(grid["renderer"]["path"]).read_bytes())
        assert certificate["artifact_kind"] == "raw_shape_basis"
        assert certificate["renderer"]["saved_shape_chain_replay_bit_identical"] is True
    payload = sources.canonical_parent_payload(candidate)
    assert payload["painted_table_bytes"]
    assert summary["facies_table"]["sha256"] == candidate[contract.KEY]["table"]["sha256"]


@pytest.mark.parametrize("imported", ["agree"], indirect=True)
def test_approved_session_table_is_automatic_full_and_roundtrips_store_and_exports(registered_setup, tmp_path, monkeypatch):
    data = registered_setup
    before = Path(data.config["facies_properties_source_path"]).read_bytes()
    prepared, version, session, _, _ = prepared_painting(data, tmp_path)
    assert "painting_facies" in prepared and not prepared["blockers"]
    assert prepared["painting_facies"]["facies_snapshot"]["permeability_unit"] == "mD"
    config, summary, revision = _saved(data, prepared)
    assert [(row["label_id"], row["facies_id"], row["porosity"]) for row in config["regions"]] == [(5, 1, .44), (6, 2, .22)]
    raw = Path(revision["artifacts"]["facies_properties_path"]).read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert [row["name"] for row in rows] == ["Sand2", "Sand1"]
    assert rows[1]["swi CO2"] == "0.12" and rows[1]["krel_gas"] == "0.8"
    assert rows[1]["custom"] == "keep" and rows[1]["permeability"] == "12.0"
    assert len(rows) == 2 and "swi" not in rows[0]
    assert [row["id"] for row in rows] == ["2", "1"]
    assert config[contract.KEY]["table"]["source_max_id"] == 7
    assert config[contract.KEY]["table"]["retired_source_ids"] == [3, 4, 6, 7]
    assert Path(data.config["facies_properties_source_path"]).read_bytes() == before
    assert summary["facies_table"]["sha256"] == config[contract.KEY]["table"]["sha256"]
    assert revision == json.loads((data.project.root / "overrides/geometry_facies/current.json").read_bytes())
    outputs = {}
    for run in ("one", "two"):
        resolved = sources.registered_run_sources(data.project.root, revision, run)
        assert Path(resolved["files"]["facies_properties"]["path"]).read_bytes() == raw
        output = export.export_corrected_setup(data.project,
            reference=SimpleNamespace(run_id=run), actor="synthetic")
        outputs[run] = output
        assert (output.destination / "facies_properties.csv").read_bytes() == raw
        assert output.receipt["binding"]["properties"] == resolved["files"]["facies_properties"]
        reused = export.export_corrected_setup(data.project,
            reference=SimpleNamespace(run_id=run), actor="synthetic")
        assert reused.reused and reused.receipt == output.receipt
    from darsia.fluidflower import setup_rig as dispatch
    build_calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: build_calls.append(args))
    for run, output in outputs.items():
        export.apply_corrected_selected_run(data.project, output)
        path = data.series / "config/merged" / (run + ".toml")
        assert export.require_corrected_export_for_build(path) == output.receipt
        dispatch.setup_rig(dispatch.Rig, path, False)
        assert build_calls[-1][1] is path and build_calls[-1][2] is False
        assert build_calls[-1][0].setup is dispatch.Rig.setup
        reused = export.export_corrected_setup(data.project,
            reference=SimpleNamespace(run_id=run), actor="synthetic")
        assert reused.reused and (reused.destination / "facies_properties.csv").read_bytes() == raw
    assert len(build_calls) == 2
    facies, roles, table, identity = simulator_geometry.run_geometry(revision, "one")
    source = next(row for row in table if row["name"] == "Sand1")
    cells = dict(zip(source["source_columns"], source["source_cells"]))
    assert cells["swi CO2"] == "0.12" and source["swi"] is None
    assert cells["pc [mbar]"] == "3" and cells["custom"] == "keep"
    _, document, _, _ = simulator_geometry.export_geometry(facies, roles, table,
        width_m=.9, height_m=.49,
        depth_points=[[-1., -1., .01], [2., -1., .01], [.5, 2., .01]],
        resolution_mm=2., inner_frame="canvas", sand_surface="highest_sand",
        fixture_polygons_m={}, fixture_roles=(), source=identity)
    published = json.loads(document)
    copied = next(row for row in published["facies_table"] if row["name"] == "Sand1")
    assert copied["source_columns"] == source["source_columns"]
    assert copied["source_cells"] == source["source_cells"]


def test_two_painted_layers_of_one_facies_produce_one_csv_row(registered_setup, tmp_path):
    prepared, _, _, _, _ = prepared_painting(registered_setup, tmp_path, same_facies=True)
    config, _, revision = _saved(registered_setup, prepared)
    assert [row["facies_id"] for row in config["regions"]] == [1, 1]
    rows = list(csv.DictReader(io.StringIO(Path(revision["artifacts"]["facies_properties_path"]).read_text())))
    assert len(rows) == 1 and rows[0]["name"] == "Sand1"


@pytest.mark.parametrize("imported", ["conflict"], indirect=True)
def test_conflicting_duplicate_parent_cells_fail_before_save_with_every_value(registered_setup, tmp_path):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    previous = geometry.GeometryRevisionStore(data.project).current()["revision_id"]
    with pytest.raises(ValueError) as caught:
        adoption.adoption_candidate(data.revision, prepared["folder"])
    assert all(text in str(caught.value) for text in ("Sand1", "swi CO2", "id 1='0.12'", "id 7='0.91'"))
    assert geometry.GeometryRevisionStore(data.project).current()["revision_id"] == previous


@pytest.mark.parametrize("change", ["facies", "map_sha", "table", "manual_ids", "remove_contract", "parent_csv", "source_map"])
def test_source_approval_table_and_physics_tampering_is_refused(registered_setup, tmp_path, change):
    data = registered_setup
    prepared, version, _, _, _ = prepared_painting(data, tmp_path)
    candidate, _ = adoption.adoption_candidate(data.revision, prepared["folder"])
    if change == "manual_ids":
        with pytest.raises(ValueError, match="automatic"):
            adoption.adoption_candidate(data.revision, prepared["folder"], {5: 2, 6: 1})
        return
    if change in {"facies", "map_sha"}:
        path = Path(version["folder"]) / ("session.json" if change == "facies" else "version.json")
        value = json.loads(path.read_bytes())
        if change == "facies":
            value["facies"]["Sand1"]["porosity"] = .99
        else:
            value["sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    elif change == "table":
        candidate[contract.KEY]["table"]["sha256"] = "0" * 64
    elif change == "remove_contract":
        candidate.pop(contract.KEY)
    elif change == "parent_csv":
        path = Path(candidate[contract.KEY]["source"]["csv"]["path"])
        path.write_bytes(path.read_bytes() + b" ")
    else:
        path = Path(prepared["folder"]) / "source_map.npz"
        np.savez_compressed(path, array=np.full((2, 3), 5, np.uint8))
    with pytest.raises(ValueError):
        sources.canonical_parent_payload(candidate)


def test_exact_automatic_facies_id_echo_is_idempotent_without_writes(registered_setup, tmp_path):
    """An unchanged legacy API echo is allowed; an override is a different action."""
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    current = geometry.GeometryRevisionStore(data.project).current_path
    index = Path(prepared["folder"]) / "index.json"
    protected = {current: current.read_bytes(), index: index.read_bytes()}
    expected, summary = adoption.adoption_candidate(data.revision, prepared["folder"])
    echoed, echoed_summary = adoption.adoption_candidate(data.revision, prepared["folder"], {5: 1, 6: 2})
    assert echoed == expected and echoed_summary == summary
    assert all(path.read_bytes() == before for path, before in protected.items())


def test_free_water_keeps_physical_porosity_one_and_is_out_of_co2_eligibility(registered_setup, tmp_path):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path, water=True)
    config, _, revision = _saved(data, prepared)
    water = next(row for row in config["regions"] if row["label_id"] == 6)
    assert water["analysis_role"] == "free_water" and water["porosity"] == 1.0 and water["enabled"] is True
    labels = geometry._load_array(Path(config["labels_path"]))
    _, eligible, _ = indicator_response._geometry_masks(config)
    assert not np.any(eligible[labels == 6]) and np.all(eligible[labels == 5])
    output = export.export_corrected_setup(data.project,
        reference=SimpleNamespace(run_id="one"), actor="synthetic")
    table = list(csv.DictReader(io.StringIO((output.destination / "facies_properties.csv").read_text())))
    saved_water = next(row for row in table if row["name"] == "Sand2")
    assert saved_water["analysis_role"] == "free_water" and saved_water["porosity"] == "1.0"
    assert revision["configuration"]["regions"] == config["regions"]


def test_proven_outside_frame_gets_only_established_nonphysical_row_and_empty_extra_cells(registered_setup, tmp_path):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path, outside=True)
    assert prepared["outside_frame_proof"]["format"] == contract.OUTSIDE_FORMAT
    config, _, revision = _saved(data, prepared)
    outside = next(row for row in config["regions"] if row["label_id"] == 0)
    assert (outside["analysis_role"], outside["porosity"], outside["permeability"], outside["enabled"]) == ("frame", 0., None, False)
    _, eligible, _ = indicator_response._geometry_masks(config)
    assert not np.any(eligible[:, 0])
    output = export.export_corrected_setup(data.project,
        reference=SimpleNamespace(run_id="one"), actor="synthetic")
    raw = (output.destination / "facies_properties.csv").read_bytes()
    assert raw == Path(revision["artifacts"]["facies_properties_path"]).read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    frame = next(row for row in rows if row["name"] == lp.OUTSIDE_FACIES)
    assert frame["swi CO2"] == frame["custom"] == frame["krel_gas"] == ""
    proof = config[contract.KEY]["table"]["provenance"][-1]
    assert dict(proof["fields"])["swi CO2"] == "not-applicable:proven-nonphysical-frame"


def test_missing_or_interior_zero_frame_proof_never_becomes_nonphysical(registered_setup, tmp_path):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path, outside=True)
    index = Path(prepared["folder"]) / "index.json"
    value = json.loads(index.read_bytes())
    value.pop("outside_frame_proof")
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="inside-mask proof"):
        adoption.adoption_candidate(data.revision, prepared["folder"])


def test_rebound_wrong_inside_mask_is_rejected_by_exact_frame_replay(registered_setup, tmp_path):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path, outside=True)
    index = Path(prepared["folder"]) / "index.json"
    value = json.loads(index.read_bytes())
    mask = value["outside_frame_proof"]["runs"]["one"]
    path = Path(prepared["folder"]) / mask["path"]
    with path.open("wb") as stream:
        np.savez_compressed(stream, inside=np.ones((2, 3), bool))
    bound = sources.descriptor(path)
    value["outside_frame_proof"]["runs"]["one"] = dict(bound, path=mask["path"])
    index.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="exact mask"):
        adoption.adoption_candidate(data.revision, prepared["folder"])


def test_unmapped_native_zero_is_not_outside_even_when_source_zero_is_outside(registered_setup, tmp_path):
    previous = geometry.GeometryRevisionStore(registered_setup.project).current()["revision_id"]
    with pytest.raises(ValueError, match="interior or unmapped"):
        prepared_painting(registered_setup, tmp_path, outside=True, unmapped=True)
    assert geometry.GeometryRevisionStore(registered_setup.project).current()["revision_id"] == previous


@pytest.mark.parametrize("imported", ["outside_source"], indirect=True)
def test_matching_parent_outside_name_keeps_blank_extra_cells_and_new_identity(registered_setup, tmp_path):
    prepared, _, _, _, _ = prepared_painting(registered_setup, tmp_path, outside=True)
    candidate, summary = adoption.adoption_candidate(registered_setup.revision, prepared["folder"])
    row = next(row for row in summary["facies_table"]["rows"] if row["name"] == lp.OUTSIDE_FACIES)
    assert row["swi CO2"] == row["custom"] == ""
    assert row["id"] == "10"
    provenance = candidate[contract.KEY]["table"]["provenance"][-1]
    assert dict(provenance["fields"])["custom"] == "not-applicable:proven-nonphysical-frame"
    assert provenance["parent_ids"] == [9]


def test_ordinary_canonical_table_bytes_and_physical_guards_remain_unchanged(registered_setup):
    data = registered_setup
    payload = sources.canonical_parent_payload(data.config)
    assert "painted_table_bytes" not in payload
    assert Path(data.revision["artifacts"]["facies_properties_path"]).read_bytes() == Path(data.config["facies_properties_source_path"]).read_bytes()
    changed = deepcopy(data.config)
    changed["regions"][0]["porosity"] += .01
    with pytest.raises(ValueError, match="original physical CSV properties"):
        sources.canonical_parent_payload(changed)
    output = export.export_corrected_setup(data.project,
        reference=SimpleNamespace(run_id="one"), actor="synthetic")
    expected = Path(data.config["facies_properties_source_path"]).read_bytes()
    assert (output.destination / "facies_properties.csv").read_bytes() == expected
    assert output.receipt["binding"]["properties"] == data.rows["one"]["facies_properties"]
    assert export.export_corrected_setup(data.project,
        reference=SimpleNamespace(run_id="one"), actor="synthetic").reused
    export.apply_corrected_selected_run(data.project, output)
    assert export.require_corrected_export_for_build(data.series / "config/merged/one.toml") == output.receipt


@pytest.mark.parametrize("stage", ["reuse", "apply", "build"])
def test_original_csv_cannot_replace_the_derived_table_in_export_apply_or_build(
        registered_setup, tmp_path, stage):
    data = registered_setup
    prepared, _, _, _, _ = prepared_painting(data, tmp_path)
    _saved(data, prepared)
    output = export.export_corrected_setup(data.project,
        reference=SimpleNamespace(run_id="one"), actor="synthetic")
    if stage == "build":
        export.apply_corrected_selected_run(data.project, output)
    old = data.rows["one"]["facies_properties"]
    path = output.destination / "facies_properties.csv"
    path.write_bytes(Path(old["path"]).read_bytes())
    receipt = deepcopy(output.receipt)
    receipt["binding"]["properties"] = deepcopy(old)
    receipt["files"]["facies_properties.csv"] = sources.descriptor(path)
    (output.destination / "export.json").write_text(json.dumps(receipt))
    output = replace(output, receipt=receipt)
    with pytest.raises(ValueError):
        if stage == "reuse":
            export.export_corrected_setup(data.project,
                reference=SimpleNamespace(run_id="one"), actor="synthetic")
        elif stage == "apply":
            export.apply_corrected_selected_run(data.project, output)
        else:
            export.require_corrected_export_for_build(data.series / "config/merged/one.toml")


def test_manual_map_approval_does_not_become_historical_metric_frame_authority(registered_setup, tmp_path):
    def changed(frame):
        frame.width_m += .123
        frame.corners["top_left"] = (.01, .01)
    prepared, version, _, original_frame, _ = prepared_painting(
        registered_setup, tmp_path, preparation_frame=changed)
    binding = prepared["painting_facies"]
    assert binding["record"]["sha256"] == version["sha256"]
    assert binding["frame"]["width_m"] != original_frame.width_m
    assert binding["frame"]["corners"]["top_left"] == [.01, .01]
    assert binding["frame_basis"]["historical_frame_bound"] is False
    assert binding["frame_basis"]["basis"] == "explicit-current-frame-at-preparation"
    candidate, summary = adoption.adoption_candidate(registered_setup.revision, prepared["folder"])
    assert candidate["width_m"] == registered_setup.config["width_m"]
    assert "no exact historical metric frame" in summary["frame_basis"]
    path = Path(prepared["folder"]) / "index.json"
    value = json.loads(path.read_bytes())
    value["painting_facies"]["frame_basis"]["historical_frame_bound"] = True
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="basis must remain separate"):
        adoption.adoption_candidate(registered_setup.revision, prepared["folder"])


def test_approved_reference_quadrilateral_survives_painted_preparation(registered_setup, tmp_path):
    from darsia.fluidflower.rig_reference_quadrilateral import reference_frame_geometry

    reference = {}

    def reviewed(frame):
        geometry = reference_frame_geometry(frame.corners,
            measurements_m={"width": frame.width_m, "height": frame.height_m})
        frame.reference_geometry = geometry["reference_geometry"]
        frame.reference_geometry.update(point_descriptions={
            "top_left": "top_of_free_water", "top_right": "top_of_free_water",
            "bottom_right": "rig_frame", "bottom_left": "rig_frame"},
            point_review_identity="a" * 64)
        frame.source = "reviewed reference points"
        reference.update(frame.to_dict())

    prepared, version, _, _, layers = prepared_painting(
        registered_setup, tmp_path, preparation_frame=reviewed)
    binding = prepared["painting_facies"]
    assert binding["frame"] == reference
    assert binding["approval"]["sha256"] == version["sha256"]
    _, replayed, frame = contract.verify_version(binding)
    np.testing.assert_array_equal(replayed, layers)
    assert frame.to_dict() == reference
    frame.reference_geometry["point_descriptions"]["top_left"] = "changed locally"
    assert binding["frame"] == reference
    adoption.adoption_candidate(registered_setup.revision, prepared["folder"])


def test_polygons_frame_basis_is_true_only_for_the_exact_saved_frame(registered_setup, tmp_path):
    prepared, _, _, frame, _ = prepared_painting(registered_setup, tmp_path, method="polygons")
    assert prepared["painting_facies"]["frame"] == frame.to_dict()
    assert prepared["painting_facies"]["frame_basis"]["historical_frame_bound"] is True
    folder = tmp_path / "changed"
    folder.mkdir()
    with pytest.raises(ValueError):
        prepared_painting(registered_setup, folder, method="polygons",
                         preparation_frame=lambda value: setattr(value, "width_m", .91))


@pytest.mark.parametrize("change", ["photo", "profile", "metric_matrix", "native_matrix", "bridge"])
def test_same_sized_foreign_photo_profile_and_forged_model_compositions_refuse_before_write(
        registered_setup, tmp_path, change):
    def mutate(run, value, folder):
        if run != "one":
            return
        if change == "photo":
            photo = folder / "foreign.png"
            Image.fromarray(np.full((2, 3, 3), 244, np.uint8)).save(photo)
            value["reference_image"] = sources.descriptor(photo)
        elif change == "profile":
            value["camera_profile"]["profile_id"] = "same-sized-foreign-profile"
        elif change == "metric_matrix":
            value["H_camera_u_to_metric_mm"][0][2] += .125
        elif change == "native_matrix":
            value["baseline_native_grid"]["H_baseline_camera_u_to_rig_corrected_native"][0][2] += .125
        else:
            value["baseline_to_model_registration"] = round5_geometry.write_new_json(folder / "bad-bridge.json",
                dict(accepted=True, exact_same_RAW_sha256="f" * 64,
                     H_baseline_u_to_model_u=IDENTITY, maximum_displacement_px=0., displacement_limit_px=.5))
    with pytest.raises(ValueError):
        prepared_painting(registered_setup, tmp_path, mutate_model=mutate)
    assert not (tmp_path / "preparation").exists()


@pytest.mark.parametrize("grid", [None, [], [1], "not-a-grid", 1])
def test_checksum_bound_non_object_native_grid_refuses_before_preparation(
        registered_setup, tmp_path, grid):
    def malformed(run, model, folder):
        if run == "one":
            model["baseline_native_grid"] = grid

    # _models publishes this actual JSON and a fresh SHA/size descriptor. The
    # refusal must concern its schema, not a stale or mismatching checksum.
    with pytest.raises(
            ValueError, match="The rigs need their RAW basis"):
        prepared_painting(registered_setup, tmp_path, mutate_model=malformed)
    model = sources._verify(sources.descriptor(tmp_path / "models" / "one.json"))
    assert json.loads(Path(model["path"]).read_bytes())["baseline_native_grid"] == grid
    assert not (tmp_path / "preparation").exists()


def test_raw_without_actual_configured_profile_is_refused(registered_setup, tmp_path):
    with pytest.raises(ValueError, match="RAW needs a bound camera-U profile"):
        prepared_painting(registered_setup, tmp_path, raw_source=True)
    assert not (tmp_path / "preparation").exists()


def test_same_sized_raw_profile_cannot_replace_the_actual_run_assignment(tmp_path):
    parameters = round5_geometry.write_new_json(tmp_path / "parameters.json", dict(
        registration=dict(common=dict(sensor=dict(camera_wh=[3, 2]),
                                      profiles={"actual": {}, "foreign": {}})),
        source=dict(profiles=dict(one="actual"))))
    model = dict(shape_hw=[2, 3], reference_image=dict(path=str(tmp_path / "photo.ARW")),
                 camera_profile={**parameters, "profile_id": "actual", "camera_wh": [3, 2]})
    context = dict(kind="camera_u", parameters=parameters, profile_id="actual", camera_wh=[3, 2])
    guards = geometry_proof._profile(model, "one", context, contract._checked, source_run="one")
    assert guards == [parameters, parameters]
    context["profile_id"] = "foreign"
    with pytest.raises(ValueError, match="approved session's configured profile"):
        geometry_proof._profile(model, "one", context, contract._checked, source_run="one")
    context["profile_id"] = "actual"
    model["camera_profile"]["profile_id"] = "foreign"
    with pytest.raises(ValueError):
        geometry_proof._profile(model, "one", context, contract._checked, source_run="one")


def test_same_sized_raw_parameters_cannot_borrow_another_bound_file(tmp_path):
    value = dict(registration=dict(common=dict(sensor=dict(camera_wh=[3, 2]),
                                              profiles={"actual": {}})),
                 source=dict(profiles=dict(one="actual")))
    parameters = round5_geometry.write_new_json(tmp_path / "parameters.json", value)
    value["unrelated_identity"] = True
    foreign = round5_geometry.write_new_json(tmp_path / "foreign.json", value)
    model = dict(shape_hw=[2, 3], reference_image=dict(path=str(tmp_path / "photo.ARW")),
                 camera_profile={**foreign, "profile_id": "actual", "camera_wh": [3, 2]})
    context = dict(kind="camera_u", parameters=parameters, profile_id="actual", camera_wh=[3, 2])
    with pytest.raises(ValueError, match="parameter identities differ"):
        geometry_proof._profile(model, "one", context, contract._checked, source_run="one")


def test_real_old_corrected_payload_link_remains_closed_despite_matching_renderer_and_shape(
        registered_setup, tmp_path, synthetic_payload_admission):
    """Old NPZ payloads do not acquire authority from accurate grid descriptors.

    New adoptions now refuse a legacy model at their earlier basis preflight.
    This direct old-reader boundary retains its independent stronger assertion:
    even byte-valid pixels and a complete matching spatial grid cannot open it.
    """
    from darsia.fluidflower import rig_raw_build as build
    row = registered_setup.config["frame_binding"]["runs"]["one"]
    bound = row["fields"]["shape_corrected_baseline"]
    image = darsia.imread(Path(bound["path"]))
    renderer = dict(source_npz=sources.descriptor(bound["path"]),
                    source_pixels=build._array_identity(image.img), source_grid=build._grid(image),
                    native_shape_hw=list(image.img.shape[:2]), native_to_camera_u_h=deepcopy(IDENTITY))
    model = dict(baseline_native_grid=dict(shape_hw=[2, 3]), reference_image=sources.descriptor(bound["path"]))
    assert renderer["native_shape_hw"] == model["baseline_native_grid"]["shape_hw"]
    assert synthetic_payload_admission is geometry_proof._require_corrected_payload_link
    with pytest.raises(ValueError, match="no admitted renderer-to-saved corrected-baseline payload link"):
        geometry_proof._require_corrected_payload_link(registered_setup.config, "one", model, renderer)
    assert not (tmp_path / "preparation").exists()


def test_old_corrected_baseline_cannot_be_used_as_camera_u_source(registered_setup, tmp_path):
    with pytest.raises(ValueError, match="old corrected NPZ is not admitted"):
        prepared_painting(registered_setup, tmp_path, corrected_source=True)
    assert not (tmp_path / "preparation").exists()


def test_fixture_analysis_replays_carrier_and_occluded_drawn_ids_through_store(registered_setup, tmp_path):
    """Owned temporary RAW basis; all painting and underlay proof is real."""
    data = registered_setup
    prepared, version, session, frame, carrier = prepared_painting(data, tmp_path, fixture=True)
    binding = prepared[contract.PREPARATION_KEY]
    _, approved, _ = contract.verify_version(binding)
    assert np.array_equal(approved, carrier)
    with np.load(Path(prepared["folder"]) / "source_map.npz", allow_pickle=False) as saved:
        analysis = saved["array"]
    assert not np.array_equal(analysis, carrier)
    assert set(np.unique(analysis)) == {6, 8}
    proof = prepared["painting_analysis_derivation"]
    assert proof["format"] == contract.FIXTURE_DERIVATION_FORMAT
    assert proof["version"] == binding["version"] and proof["approval"] == binding["approval"]
    config, _, revision = _saved(data, prepared)
    rows = {row["label_id"]: row for row in config["regions"]}
    assert set(rows) == {6, 7, 8}  # 7 is known only from the actual drawn underlay.
    assert rows[7]["analysis_role"] == "porous_medium" and rows[7]["porosity"] == .44
    assert rows[8]["analysis_role"] == "colorchecker" and rows[8]["porosity"] == 0.
    assert rows[8]["permeability"] is None and rows[8]["enabled"] is False
    payload = sources.canonical_parent_payload(revision["configuration"])
    assert payload["painted_table_bytes"] == Path(revision["artifacts"]["facies_properties_path"]).read_bytes()
    with pytest.raises(ValueError, match="exact approved saved painting"):
        contract.capture_version(version, binding["approval"], frame, analysis, project=data.project)


@pytest.mark.parametrize("damage", ["provenance", "source_map", "native_underlay", "drawn_ids", "frame"])
def test_fixture_derivation_drift_refuses_before_store(registered_setup, tmp_path, damage):
    prepared, _, _, _, carrier = prepared_painting(registered_setup, tmp_path, fixture=True)
    folder = Path(prepared["folder"])
    current_path = registered_setup.project.root / "overrides/geometry_facies/current.json"
    current_before = current_path.read_bytes()
    index = json.loads((folder / "index.json").read_bytes())
    if damage == "provenance":
        index["painting_analysis_derivation"]["carrier_sha256"] = "0" * 64
    elif damage == "source_map":
        path = folder / "source_map.npz"
        np.savez_compressed(path, array=carrier)
        index["source_map"] = dict(sources.descriptor(path), path="source_map.npz")
    elif damage == "native_underlay":
        row = index["runs"][0]
        row["fixture_underlay"] = index["source_fixture_underlay"]
    elif damage == "drawn_ids":
        index["runs"][0]["fixture_underlay_layer_ids"] = []
    else:
        index["fixture_frame"]["width_m"] *= 2
    (folder / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError):
        adoption.adoption_candidate(registered_setup.revision, folder)
    assert current_path.read_bytes() == current_before


@pytest.mark.parametrize("damage", ["model", "configuration_copy", "original_configuration", "source_underlay", "native_labels", "native_underlay", "geometry_index"])
def test_final_publication_refuses_same_sized_late_drift_without_native_zero_labels(registered_setup, tmp_path, damage):
    """Late source/output drift must leave only retained partial files, never index.json."""
    observed = {}
    def during_prepare(message):
        if "TWO" not in message.upper():
            return
        folders = list((tmp_path / "preparation").iterdir())
        assert len(folders) == 1
        folder = folders[0]
        with np.load(folder / "one/labels.npz", allow_pickle=False) as saved:
            assert not np.any(saved["array"] == 0)
        original_configuration = json.loads((tmp_path / "models/one.json").read_bytes())["baseline_configuration"]["path"]
        targets = {"model": tmp_path / "models/one.json",
                   "original_configuration": Path(original_configuration),
                   "configuration_copy": folder / "configuration_sources/one.toml",
                   "source_underlay": folder / "source_fixture_underlay.npz",
                   "native_labels": folder / "one/labels.npz",
                   "native_underlay": folder / "one/fixture_underlay.npz",
                   "geometry_index": tmp_path / "models/index.json"}
        target = targets[damage]
        raw = target.read_bytes()
        # One byte, same size; JSON trailing whitespace stays syntactically valid.
        replacement = (raw[:-1] + (b" " if raw[-1:] != b" " else b"\n")) if target.suffix == ".json" else bytes([raw[0] ^ 1]) + raw[1:]
        assert len(replacement) == len(raw) and replacement != raw
        target.write_bytes(replacement)
        observed.update(folder=folder, target=target, bytes=replacement)
    with pytest.raises(ValueError):
        prepared_painting(registered_setup, tmp_path, fixture=True, during_prepare=during_prepare)
    assert observed
    assert not (observed["folder"] / "index.json").exists()
    assert observed["target"].read_bytes() == observed["bytes"]
    assert (observed["folder"] / "source_map.npz").exists()


@pytest.mark.parametrize("part", ["folder", "name", "approval", "full_record", "kind"])
def test_same_sha_distinct_saved_version_cannot_replace_literal_preparation_identity(registered_setup, tmp_path, part):
    prepared, version, session, _, _ = prepared_painting(registered_setup, tmp_path, fixture=True)
    _, saved = manual.load_manual_version(version)
    other = manual.save_manual_version(registered_setup.project, session, saved, name="Another saved map")
    other_approval = manual.approve_manual_version(other)
    assert other["sha256"] == version["sha256"] and other["folder"] != version["folder"]
    folder = Path(prepared["folder"])
    first = Path(version["folder"])
    second = Path(other["folder"])
    protected = [*first.iterdir(), *second.iterdir(), first.parent / "approvals.jsonl"]
    before = {str(path): path.read_bytes() for path in protected if path.is_file()}
    index = json.loads((folder / "index.json").read_bytes())
    if part == "full_record":
        index["map_version"] = dict(other, approval=other_approval)
    elif part == "approval":
        index["map_version"]["approval"] = other_approval
    elif part == "kind":
        index["map_version"]["kind"] = "proposal"
    else:
        index["map_version"][part] = other[part]
    (folder / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="literal saved version"):
        adoption.adoption_candidate(registered_setup.revision, folder)
    assert all(Path(path).read_bytes() == raw for path, raw in before.items())


@pytest.mark.parametrize("fixture", [False, True])
def test_photo_continuation_preserves_current_owned_basis_and_analysis_derivation(
        registered_setup, tmp_path, fixture):
    """Use current real admission; only the established tiny decoder is synthetic."""
    from darsia.fluidflower import photo_edge_continuation as policy
    prepared, version, session, frame, original = prepared_painting(
        registered_setup, tmp_path, fixture=fixture, continuation=True)
    assert prepared["format"] == adoption.PREPARATION_FORMAT_V2
    assert "raw_shape_basis" in prepared["painting_geometry"]
    assert prepared["approved_source"] == prepared["painting_facies"]
    assert prepared["map_version"]["sha256"] == version["sha256"]
    folder = Path(prepared["folder"])
    assert policy.read_contracts(folder / "continuation.json")["runs"].keys() == {"one", "two"}
    if fixture:
        assert prepared["painting_analysis_derivation"]["carrier_sha256"] != prepared["painting_analysis_derivation"]["analysis_sha256"]
        assert prepared["source_fixture_underlay"]
    assert adoption.read_preparation(folder)["approved_source"] == prepared["approved_source"]
    candidate, summary, revision = _saved(registered_setup, prepared)
    for run in ("one", "two"):
        bound = sources.registered_run_sources(registered_setup.project.root, revision, run)
        assert bound["registration"]["kind"] == sources.PAINTED_V2
        with np.load(bound["files"]["labels"]["path"], allow_pickle=True) as product:
            assert {"inside", "continued", "effective_coverage", "gaps"} <= set(product.files)
            assert not product["gaps"].any()
            if run == "two":
                assert product["continued"][:, 0].all()
                assert not product["support"][:, 0].any()
                assert product["effective_coverage"][:, 0].all()
        _, _, _, exported = simulator_geometry.run_geometry(revision, run)
        assert exported["photo_edge_continuation"]["adopted_by_this_export"] is False
        if fixture:
            assert exported.get("_drawn_fixture_underlay") is not None
    assert summary["runs"] == ["one", "two"]


def test_photo_continuation_reuses_only_an_unchanged_complete_native_proof(
        registered_setup, tmp_path, monkeypatch):
    from darsia.fluidflower import photo_edge_continuation as policy
    prepared, *_ = prepared_painting(registered_setup, tmp_path, continuation=True)
    folder = Path(prepared["folder"])
    policy._NATIVE_PROOFS.clear()
    original, calls = geometry_proof.verify_proof, []
    def observed(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(geometry_proof, "verify_proof", observed)
    first = policy.verify_native_proof(prepared, folder)
    assert policy.verify_native_proof(prepared, folder) == first
    assert len(calls) == 1, "Each per-run check reuses the same fully admitted all-run proof"
    copied = folder / prepared["painting_geometry"]["baseline_configurations"]["two"]["copy"]["path"]
    copied.write_bytes(copied.read_bytes() + b"\nchanged retained configuration\n")
    with pytest.raises(ValueError):
        policy.verify_native_proof(prepared, folder)
    assert len(calls) == 2, "Any consumed-file drift forces the original admission again"


def test_photo_continuation_memo_rehashes_original_raw_when_file_stamps_collide(
        registered_setup, tmp_path, monkeypatch):
    from darsia.fluidflower import photo_edge_continuation as policy
    prepared, *_ = prepared_painting(registered_setup, tmp_path, continuation=True)
    folder = Path(prepared["folder"])
    policy._NATIVE_PROOFS.clear()
    policy.verify_native_proof(prepared, folder)
    cached = next(iter(policy._NATIVE_PROOFS.values()))
    assert cached["checks"], "The memo retains the original finite metadata readers"
    stamps = {str(Path(row["path"]).resolve()): row["stamp"]
              for row in cached["dependencies"]["files"].values()}
    original_stamp = contract._FILES.stamp
    monkeypatch.setattr(contract._FILES, "stamp", lambda path:
        stamps.get(str(Path(path).resolve()), original_stamp(path)))
    raw = Path(prepared["approved_source"]["frame_basis"]["photograph"]["path"])
    before = raw.read_bytes()
    raw.write_bytes(bytes([before[0] ^ 1]) + before[1:])
    with pytest.raises(ValueError, match="Shape-basis source changed"):
        policy.verify_native_proof(prepared, folder)


def _outside_target_frame(frames):
    """A distinct frozen target frame, partly beyond this tiny photograph."""
    frames["two"]["corners"] = dict(top_left=[1., 2.], top_right=[1., 3.],
        bottom_right=[2., 3.], bottom_left=[2., 2.])


def _v2_outside(data, tmp_path):
    return prepared_painting(data, tmp_path, outside=True, continuation=True,
        outside_targets=True, own_frame_change=_outside_target_frame)


def test_v2_outside_proof_replays_own_frame_and_preserves_store_export_labels(registered_setup, tmp_path):
    """Zero edge donors outside the own frame remain exact zeros, never sand."""
    from darsia.fluidflower import photo_edge_continuation as policy
    data = registered_setup
    prepared, version, _, frame, original = _v2_outside(data, tmp_path)
    folder = Path(prepared["folder"])
    assert prepared["outside_frame_proof"]["format"] == contract.OUTSIDE_FORMAT_V2
    source_files = {p: p.read_bytes() for p in Path(version["folder"]).rglob("*") if p.is_file()}
    expected = np.asarray([[0, 0, 5], [0, 0, 5]], np.int32)
    own_inside = np.asarray([[False, False, False], [False, False, True]])
    with np.load(folder / "two/labels.npz", allow_pickle=True) as product:
        assert np.array_equal(product["array"], expected)
        assert np.array_equal(product["inside"], own_inside)
        assert product["filled"][:, 0].all() and not product["array"][:, 0].any()
        assert not product["gaps"].any()
        assert np.array_equal(product["effective_coverage"], own_inside)
    row = prepared["runs"][1]
    legacy = contract._native_inside(row, lp.frame_mask(frame, original.shape, (1., 1.)))
    assert not np.array_equal(legacy["labels"] != 0, own_inside)
    proof = prepared["outside_frame_proof"]["runs"]["two"]
    with np.load(folder / proof["path"]) as saved:
        assert np.array_equal(saved["inside"], own_inside)
    assert adoption.read_preparation(folder)["format"] == adoption.PREPARATION_FORMAT_V2
    _, _, revision = _saved(data, prepared)
    resolved = sources.registered_run_sources(data.project.root, revision, "two")
    with np.load(resolved["files"]["labels"]["path"], allow_pickle=True) as saved:
        assert np.array_equal(saved["array"], expected)
        assert np.array_equal(saved["inside"], own_inside)
    export.export_corrected_setup(data.project,
        reference=SimpleNamespace(run_id="one"), actor="synthetic display-reference proof")
    exported = export.export_corrected_setup(data.project,
        reference=SimpleNamespace(run_id="two"), actor="synthetic own-frame proof")
    with np.load(exported.destination / "native_labels.npz", allow_pickle=True) as saved:
        assert np.array_equal(saved["array"], expected)
        assert np.array_equal(saved["inside"], own_inside)
    run_contract = policy.load_contract(folder / "continuation.json", "two")
    policy.verify_product(exported.destination / "native_labels.npz", original, run_contract,
                          counts=row["continuation"])
    outside = next(v for v in revision["configuration"]["regions"] if v["label_id"] == 0)
    assert (outside["analysis_role"], outside["porosity"], outside["enabled"]) == ("frame", 0., False)
    assert all(p.read_bytes() == content for p, content in source_files.items())


def test_v2_outside_proof_refuses_zero_inside_the_target_own_frame(registered_setup, tmp_path):
    """A real own-frame gap is not cleared by the technical proof branch."""
    current = geometry.GeometryRevisionStore(registered_setup.project).current_path
    before = current.read_bytes()
    # The target retains columns1..2 while its transported source has zeros
    # in columns0..1. Column1 is therefore an interior gap, not outside.
    with pytest.raises(ValueError, match="zero labels are INSIDE its frozen own frame"):
        prepared_painting(registered_setup, tmp_path, outside=True, continuation=True,
                         outside_targets=True)
    assert current.read_bytes() == before


@pytest.mark.parametrize("field", ["array", "filled", "support", "inside", "continued", "effective_coverage", "gaps"])
def test_v2_outside_proof_replays_every_mask_even_after_descriptor_rebinding(registered_setup, tmp_path, field):
    prepared, _, _, frame, original = _v2_outside(registered_setup, tmp_path)
    folder = Path(prepared["folder"])
    row = prepared["runs"][1]
    path = folder / row["labels"]["path"]
    with np.load(path, allow_pickle=True) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    if field == "array":
        values[field][1, 2] = 0
    else:
        values[field][0, 0] = not values[field][0, 0]
    np.savez_compressed(path, **values)
    row["labels"].update({key: value for key, value in sources.descriptor(path).items() if key != "path"})
    with pytest.raises(ValueError, match="V2 (labels differ|continuation mask differs)"):
        contract.verify_outside_proof(prepared, folder, original, frame)


def test_v2_outside_proof_refuses_rebound_wrong_own_inside_mask(registered_setup, tmp_path):
    prepared, _, _, frame, original = _v2_outside(registered_setup, tmp_path)
    folder = Path(prepared["folder"])
    mask = prepared["outside_frame_proof"]["runs"]["two"]
    path = folder / mask["path"]
    np.savez_compressed(path, inside=np.ones((2, 3), bool))
    mask.update({key: value for key, value in sources.descriptor(path).items() if key != "path"})
    with pytest.raises(ValueError, match="exact mask"):
        contract.verify_outside_proof(prepared, folder, original, frame)


@pytest.mark.parametrize("dependency", ["continuation", "geometry_index", "target_model"])
def test_v2_outside_proof_derivation_cache_tracks_policy_and_model_dependencies(registered_setup, tmp_path, dependency):
    prepared, _, _, _, _ = _v2_outside(registered_setup, tmp_path)
    folder = Path(prepared["folder"])
    contract._DERIVED.clear()
    contract.candidate_table(registered_setup.revision, prepared, folder)
    if dependency == "continuation":
        path = folder / "continuation.json"
    elif dependency == "geometry_index":
        path = Path(prepared["geometry_index"]["path"])
    else:
        path = Path(prepared["runs"][1]["target_model"]["path"])
    cached = next(iter(contract._DERIVED.values()))
    captured = {Path(row["path"]).resolve() for row in cached["dependencies"]["files"].values()}
    assert path.resolve() in captured
    path.write_bytes(path.read_bytes() + b"\n ")
    with pytest.raises(ValueError):
        contract.candidate_table(registered_setup.revision, prepared, folder)


def test_v2_legacy_outside_proof_retains_valid_historical_source_frame(registered_setup, tmp_path):
    prepared, _, _, frame, original = prepared_painting(
        registered_setup, tmp_path, outside=True, continuation=True)
    folder = Path(prepared["folder"])
    proof = prepared["outside_frame_proof"]
    proof["format"] = contract.OUTSIDE_FORMAT
    proof.pop("policy")
    proof.pop("continuation")
    # The identity reference's two frame masks coincide, so this also models
    # a valid historical V2 preparation with its original V1-format proof.
    contract.verify_outside_proof(prepared, folder, original, frame)


def test_v2_legacy_proof_format_cannot_hide_a_replayed_own_frame_gap(registered_setup, tmp_path):
    """An otherwise valid old proof cannot overrule a larger own-frame mask."""
    from darsia.fluidflower import photo_edge_continuation as policy
    data = registered_setup
    prepared, _, session, frame, original = prepared_painting(data, tmp_path, outside=True)
    folder = Path(prepared["folder"])
    # Model a historical V2 closure retaining the earlier V1 proof. Its own
    # target frame comes from genuinely captured temporary project authority.
    own = data.project.manifest["tools"]["native_series"]["rig_frames"]["one"]
    own["corners"] = dict(top_left=[0., 0.], top_right=[0., 2.],
                          bottom_right=[1., 2.], bottom_left=[1., 0.])
    (data.project.root / "series.json").write_text(json.dumps(data.project.manifest), encoding="utf-8")
    contracts = policy.build_contracts(data.project, prepared["geometry_index"]["path"],
        "one", ["one"], [layer.layer_id for layer in session.layers], target_grid="native")
    sidecar = folder / "continuation.json"
    policy.save_contracts(sidecar, contracts)
    row = prepared["runs"][0]
    path = folder / row["labels"]["path"]
    with np.load(path, allow_pickle=True) as archive:
        values = {key: archive[key].copy() for key in archive.files}
    replay = policy.apply_policy(original, contracts["one"])
    assert np.array_equal(values["array"], replay["labels"])
    assert np.array_equal(values["filled"], replay["filled"])
    assert np.array_equal(values["support"], replay["support"])
    values.update({key: replay[key] for key in ("inside", "continued", "effective_coverage", "gaps")})
    np.savez_compressed(path, **values)
    row["labels"].update({key: value for key, value in sources.descriptor(path).items() if key != "path"})
    row["continuation"] = dict(replay["counts"], contract_run="one", authority="saved rig frame")
    prepared.update(format=adoption.PREPARATION_FORMAT_V2,
        approved_source=deepcopy(prepared["painting_facies"]),
        continuation=dict(policy=policy.POLICY, file=sidecar.name, adopted=False,
            **{key: value for key, value in sources.descriptor(sidecar).items() if key != "path"}))
    assert prepared["outside_frame_proof"]["format"] == contract.OUTSIDE_FORMAT
    assert replay["counts"]["inside_gap_pixels"] == 2
    with pytest.raises(ValueError, match="zero labels are INSIDE its frozen own frame"):
        contract.verify_outside_proof(prepared, folder, original, frame)


@pytest.mark.parametrize("changed", ["target_model", "source_run", "frame_authority", "policy"])
def test_v2_outside_proof_refuses_other_contract_identity(registered_setup, tmp_path, changed):
    prepared, _, _, frame, original = _v2_outside(registered_setup, tmp_path)
    if changed == "target_model":
        prepared["runs"][1]["target_model"] = prepared["runs"][0]["target_model"]
    elif changed == "source_run":
        prepared["reference_run"] = "two"
    elif changed == "frame_authority":
        prepared["runs"][1]["continuation"]["authority"] = "series configuration"
    else:
        prepared["continuation"]["policy"] = "unapproved-policy"
    with pytest.raises(ValueError, match="V2 outside proof"):
        contract.verify_outside_proof(prepared, prepared["folder"], original, frame)


def test_v1_outside_proof_never_selects_v2_from_sidecar_presence(registered_setup, tmp_path):
    prepared, _, _, frame, original = prepared_painting(registered_setup, tmp_path, outside=True)
    folder = Path(prepared["folder"])
    path = folder / "continuation.json"
    path.write_bytes(b"not a valid or authoritative continuation\n")
    contract.verify_outside_proof(prepared, folder, original, frame)
    proof = prepared["outside_frame_proof"]
    proof.update(format=contract.OUTSIDE_FORMAT_V2, policy="nearest-photo-edge-to-target-frame-v1",
                 continuation=dict(sources.descriptor(path), path=path.name))
    with pytest.raises(ValueError, match="inside-mask proof"):
        contract.verify_outside_proof(prepared, folder, original, frame)


def test_original_painted_apply_and_build_follow_relocated_parent_csv_without_rebinding(
        registered_setup, tmp_path, monkeypatch):
    from darsia.fluidflower import storage_paths
    from darsia.fluidflower import setup_rig as dispatch
    data = registered_setup
    prepared, _version, _session, _, _ = prepared_painting(data, tmp_path)
    config, _summary, revision = _saved(data, prepared)
    parent_binding = deepcopy(config[contract.KEY]["source"]["csv"])
    declared = Path(parent_binding["path"])
    archived = tmp_path / "relocated-parent.csv"
    declared.rename(archived)
    source = str(declared).replace("\\", "/").casefold()
    target = str(archived).replace("\\", "/")
    monkeypatch.setattr(storage_paths, "_alias_rows", lambda **_: [(source, target)])
    current_path = data.project.root / "overrides/geometry_facies/current.json"
    before_revision = current_path.read_bytes()
    calls = []
    monkeypatch.setattr(dispatch, "_original_setup_rig", lambda *args: calls.append(args))
    for run in ("one", "two"):
        output = export.export_corrected_setup(data.project, reference=SimpleNamespace(run_id=run))
        applied = export.apply_corrected_selected_run(data.project, output, rebuild_facies_by_name=True)
        context = json.loads(applied.read_bytes())["facies_rebuild"]
        assert context["parent_csv"] == parent_binding
        merged = data.series / "config/merged" / (run + ".toml")
        assert export.require_corrected_export_for_build(merged) == output.receipt
        dispatch.setup_rig(dispatch.Rig, merged, False)
        assert calls[-1][1] == merged
        assert export.export_corrected_setup(data.project, reference=SimpleNamespace(run_id=run)).reused
        assert export.apply_corrected_selected_run(data.project, output, rebuild_facies_by_name=True) == applied
    assert len(calls) == 2
    assert current_path.read_bytes() == before_revision
    assert not declared.exists()
