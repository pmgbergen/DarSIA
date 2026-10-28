"""Optional source-record plumbing, not native mask or physical qualification."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import toml

from darsia.fluidflower import rig_raw_build as raw
from darsia.fluidflower.pipeline import series_setup_protocol as protocol
from darsia.fluidflower.pipeline import series_setup_inputs as inputs
from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from darsia.presets.workflows.config.rig import RigConfig
from darsia.fluidflower.store_map import StoreMapError
from .test_series_setup_inputs import controls, _doc, _file


def _declaration():
    """Return transport-only DATA which does not claim mask admission."""
    return dict(format="source-record-fixture", run_id="sample",
                provenance=dict(physical_qualification=False, literal_value=1))


def _rig_toml(path, exclusions=None, native_product=None):
    """Write explicit tiny configuration bytes with an optional literal record."""
    rig = dict(width=.9, height=.49, dim=2)
    if exclusions is not None:
        rig["analysis_exclusions"] = (json.dumps(exclusions, ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), allow_nan=False) if isinstance(exclusions, dict) else exclusions)
    if native_product is not None:
        rig["native_product"] = (json.dumps(native_product, ensure_ascii=True, sort_keys=True,
            separators=(",", ":"), allow_nan=False) if isinstance(native_product, dict) else native_product)
    path.write_text(toml.dumps(dict(rig=rig)), encoding="utf-8")
    return path


def _native_contract(controls, *, with_gap):
    """Bind selected metadata using the original sidecar writer, with no decoded science."""
    from darsia.fluidflower import photo_edge_continuation as continuation_policy
    from darsia.fluidflower.series_geometry import RigFrame

    folder = controls.root / "native-declarations"
    identity = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    shape = [3, 4]
    source = _file(folder / "source.npz", b"opaque source labels; no array decoding")
    labels = _file(folder / "labels.npz", b"opaque selected labels; no array decoding")
    exported = _file(folder / "exported.npz", Path(labels["path"]).read_bytes())
    photo = _file(folder / "photograph.arw", b"opaque photograph; no sensor decoding")
    baseline = _file(folder / "baseline.npz", b"opaque corrected baseline; no array decoding")
    correction = _file(folder / "shape.npz", b"opaque saved shape asset; no array decoding")
    renderer_request = _doc(folder / "renderer-request.json", dict(metadata_transport_fixture=True))
    marker = _doc(folder / "owner.json", dict(metadata_transport_fixture=True))
    renderer = _doc(folder / "renderer.json", dict(format=raw.FORMAT,
        artifact_kind="raw_shape_basis", complete_rig_model=False, run_id="sample",
        production_qualification_claimed=False, corrected_baseline=dict(archive=baseline),
        renderer=dict(source_npz=baseline, raw_acquisition=controls.baseline,
            shape_corrections=[correction], saved_shape_chain_replay=dict(correction_assets=[correction])),
        inputs=dict(raw=controls.baseline, profile=controls.request["profile"], operation_request=renderer_request),
        output_ownership=dict(request=renderer_request, owner=marker)))
    grid = dict(shape_hw=shape, renderer=renderer,
                H_baseline_camera_u_to_rig_corrected_native=identity)
    model_document = dict(format="darsia-measured-run-geometry-v1", run_id="sample",
        physical_qualification=False, reference_image=photo, shape_hw=shape,
        H_metric_mm_to_camera_u=identity, baseline_native_grid=grid)
    model = _doc(folder / "model.json", model_document)
    geometry_index = _doc(folder / "geometry-index.json", dict(
        format="darsia-measured-run-geometry-v1", runs=dict(sample=dict(reference_model=model))))
    frame = RigFrame(corners=dict(top_left=(0., 0.), top_right=(0., 3.),
        bottom_right=(2., 3.), bottom_left=(2., 0.)), width_m=.003, height_m=.002,
        run_id="sample", source="synthetic frame metadata", image=photo["path"], image_size=(4, 3))
    project = _doc(folder / "frame-authority.json", dict(tools=dict(native_series=dict(
        series_root=controls.request["series_context"]["root"], rig_frames=dict(sample=frame.to_dict())))))
    frozen_model = dict(descriptor=model, document=model_document, run_id="sample",
                        shape_hw=shape, photograph=photo, metric_to_grid=identity)
    selected_contract = dict(format=continuation_policy.FORMAT, policy=continuation_policy.POLICY,
        target_grid="native", source_run="sample", target_run="sample",
        source_model=deepcopy(frozen_model), target_model=deepcopy(frozen_model),
        geometry_index=geometry_index, own_frame=frame.to_dict(),
        frame_sha256=continuation_policy._json_sha(frame.to_dict()),
        frame_authority=dict(kind="saved rig frame", project=continuation_policy._freeze(project["path"])),
        known_ids=[1], source_fill="no-inherited-photo-fill", adopted=False)
    continuation_path = folder / "continuation.json"
    continuation_policy.save_contracts(continuation_path, {"sample": selected_contract})
    continuation = raw._descriptor(continuation_path)
    version = _doc(folder / "version.json", dict(metadata_transport_fixture=True))
    painting = dict(version=version, files=dict(labels=source), frame_basis=dict(
        workspace_revision=controls.revision, photograph=photo,
        camera_profile=dict(parameters=controls.request["profile"])))
    local = lambda bound: dict(bound, path=Path(bound["path"]).name)
    preparation_document = dict(format="darsia-fluidflower-painted-map-adoption-preparation-v2",
        source_map=local(source), painting_facies=painting, geometry_index=geometry_index,
        runs=[dict(run_id="sample", labels=local(labels), reference_model=model, target_model=model)],
        continuation=dict(file=continuation_path.name, sha256=continuation["sha256"],
                          size_bytes=continuation["size_bytes"]))
    expected = [controls.revision, model, labels, exported, source, continuation, photo, baseline,
                correction, renderer_request, marker, renderer, geometry_index, version,
                controls.baseline, controls.request["profile"]]
    gap = None
    if with_gap:
        mask = _file(folder / "mask.npz", b"opaque mask; no array decoding")
        mask_row = dict(artifact=mask, field="approved_analysis_gap",
                        identity=dict(dtype="|b1", shape=shape, sha256="0" * 64))
        review = _doc(folder / "review.json", dict(metadata_transport_fixture=True,
            source=dict(painting_facies=painting, geometry_index=geometry_index), source_map=source,
            runs=[dict(run_id="sample", product=labels, base_product=labels,
                reference_model=model, target_model=model, native_photo=photo,
                target_correction_bindings=[], contract=selected_contract)]))
        approval = _doc(folder / "approval.json", dict(metadata_transport_fixture=True, review=review))
        preparation_document["analysis_exclusions"] = dict(approval=approval, runs=dict(sample=mask_row))
        expected.extend([approval, mask, review])
    preparation = _doc(folder / "preparation.json", preparation_document)
    expected.append(preparation)
    if with_gap:
        gap = dict(format="darsia-approved-analysis-gap-native-v1", meaning="outside-analysis-unquantified",
            run_id="sample", preparation=preparation, approval=approval, mask=mask,
            field=mask_row["field"], identity=mask_row["identity"],
            source_labels=labels, exported_labels=exported, source_model=model, native_grid=grid,
            source_to_export_native_h=identity, provenance=dict(physical_qualification=False))
    native = dict(format="darsia-painted-native-product-v1", run_id="sample",
        adopted_revision=controls.revision, preparation=preparation, source_labels=labels,
        exported_labels=exported, source_model=model, native_grid=grid,
        source_to_export_native_h=identity, analysis_exclusions=gap, target_corrections=[])
    return native, expected


def _bind_native_controls(controls, native):
    """Join synthetic Apply controls to the actual request and optional source records."""
    plan, request = deepcopy(controls.plan), deepcopy(controls.request)
    records = dict(native_product=deepcopy(native))
    if native["analysis_exclusions"] is not None:
        records["analysis_exclusions"] = deepcopy(native["analysis_exclusions"])
    request.update(deepcopy(records))
    selected = plan["build_plans"][0]
    selected["request_template"].update(deepcopy(records))
    selected["raw_setup_settings"] = dict(run_id="sample", rig_path=request["output"]["destination"],
        raw_camera=dict(configuration=request["profile"]["path"], sha256=request["profile"]["sha256"],
                        profile_id="fixture", run_id="sample"), **deepcopy(records))
    plan_bound = _doc(controls.plan_path, plan)
    exported = json.loads(Path(controls.export["path"]).read_text(encoding="utf-8"))
    exported["binding"] = deepcopy(records)
    export_bound = _doc(Path(controls.export["path"]), exported)
    applied = json.loads(Path(controls.applied["path"]).read_text(encoding="utf-8"))
    applied["export"] = export_bound
    applied_bound = _doc(Path(controls.applied["path"]), applied)
    coordinator = deepcopy(controls.coordinator)
    coordinator["setup_plan"] = plan_bound
    coordinator["runs"][0].update(export=export_bound, apply_completion=applied_bound)
    config = dict(deepcopy(controls.build), setup_plan=plan_bound, apply_completion=applied_bound,
                  coordinator_completion=_doc(controls.coordinator_path, coordinator),
                  raw_build_request=_doc(controls.request_path, request))
    return config, request


def test_absent_rig_record_keeps_original_serialized_fields(tmp_path):
    """Legacy configuration neither gains a null field nor retains a previous record."""
    path = _rig_toml(tmp_path / "legacy.toml")
    config = RigConfig(analysis_exclusions=_declaration(), native_product=_declaration()).load(path, results=tmp_path)
    assert config.analysis_exclusions is config.native_product is None
    assert config.to_dict() == dict(width=.9, height=.49, dim=2)


@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_ordered_rig_record_roundtrip_preserves_literal_data_and_owns_its_copy(tmp_path, field):
    """Ordered TOML loading and serialization preserve a source declaration without verifying it."""
    expected = _declaration()
    common = _rig_toml(tmp_path / "common.toml", **{("exclusions" if field == "analysis_exclusions" else field): expected})
    selected = tmp_path / "selected.toml"
    expected["provenance"]["literal_value"] = 2
    selected.write_text(toml.dumps(dict(rig={field: json.dumps(expected, allow_nan=False)})), encoding="utf-8")
    config = RigConfig().load([common, selected], results=tmp_path)
    assert getattr(config, field) == expected
    serialized = config.to_dict()
    decoded = json.loads(serialized[field])
    decoded["provenance"]["literal_value"] = 9
    assert getattr(config, field) == expected
    path = tmp_path / "roundtrip.toml"
    path.write_text(toml.dumps(dict(rig=config.to_dict())), encoding="utf-8")
    assert getattr(RigConfig().load(path, results=tmp_path), field) == expected


@pytest.mark.parametrize("invalid", ["not-a-record", [1, 2], 0])
@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_rig_record_must_decode_from_a_json_object(tmp_path, invalid, field):
    """A new optional slot cannot silently turn a path or scalar into source proof."""
    path = _rig_toml(tmp_path / "invalid.toml", **{("exclusions" if field == "analysis_exclusions" else field): invalid})
    with pytest.raises(ValueError, match="original source record"):
        RigConfig().load(path, results=tmp_path)


def test_actual_no_gap_native_contract_retains_null_and_empty_arrays_through_toml(controls):
    """The genuine no-gap contract shape retains its mandatory null through the real TOML codec."""
    native, _ = _native_contract(controls, with_gap=False)
    assert native["analysis_exclusions"] is None and native["target_corrections"] == []
    config = RigConfig(width=.9, height=.49, dim=2, native_product=native)
    encoded = config.to_dict()
    assert isinstance(encoded["native_product"], str)
    assert encoded["native_product"] == json.dumps(native, ensure_ascii=True, sort_keys=True,
                                                separators=(",", ":"), allow_nan=False)
    path = controls.root / "no-gap.toml"
    path.write_text(toml.dumps(dict(rig=encoded)), encoding="utf-8")
    loaded = RigConfig().load(path, results=controls.root)
    assert loaded.native_product == native
    assert loaded.native_product["analysis_exclusions"] is None
    assert loaded.native_product["target_corrections"] == []
    assert loaded.to_dict() == encoded


@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_optional_json_transport_retains_deep_null_empty_and_typed_values(tmp_path, field):
    """Nested nulls, empty containers and literal bool/int/float values survive without defaults."""
    expected = dict(_declaration(), snapshot={"none": None, "empty_list": [], "empty_object": {},
        "mixed": [None, [], {}, True, 1, 1.0, "æ"]})
    config = RigConfig(width=.9, height=.49, dim=2, **{field: expected})
    encoded = config.to_dict()
    assert encoded[field].isascii() and "\\u00e6" in encoded[field]
    path = tmp_path / "nested.toml"
    path.write_text(toml.dumps(dict(rig=encoded)), encoding="utf-8")
    assert path.read_bytes().isascii()
    loaded = RigConfig().load(path, results=tmp_path)
    actual = getattr(loaded, field)
    assert json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(expected, sort_keys=True, allow_nan=False)
    assert [type(value) for value in actual["snapshot"]["mixed"]] == [type(value) for value in expected["snapshot"]["mixed"]]
    assert actual["snapshot"]["none"] is None
    assert config.to_dict() == loaded.to_dict()


@pytest.mark.parametrize("encoded", ["null", "[]", "1", '{"x":1,"x":2}',
    '{"outer":{"x":1,"x":2}}', '{"value":NaN}', '{"value":1e999}'])
@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_optional_json_transport_refuses_nonobjects_duplicates_and_nonfinite(tmp_path, field, encoded):
    """Strict object decoding rejects ambiguous or nonfinite records without repairing their content."""
    path = tmp_path / "invalid-record.toml"
    path.write_text(toml.dumps(dict(rig=dict(width=.9, height=.49, dim=2, **{field: encoded}))), encoding="utf-8")
    with pytest.raises(ValueError, match="original source record JSON"):
        RigConfig().load(path, results=tmp_path)


@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_optional_json_transport_does_not_require_an_extra_lexical_identity(tmp_path, field):
    """An already bound pretty JSON string decodes to the same literal source object."""
    expected = dict(z=None, a=[{}, []], nested=dict(flag=False, value=1.0))
    pretty = json.dumps(expected, ensure_ascii=False, indent=2, allow_nan=False)
    path = tmp_path / "pretty.toml"
    path.write_text(toml.dumps(dict(rig=dict(width=.9, height=.49, dim=2, **{field: pretty}))), encoding="utf-8")
    config = RigConfig().load(path, results=tmp_path)
    assert getattr(config, field) == expected
    assert config.to_dict()[field] == json.dumps(expected, ensure_ascii=True, sort_keys=True,
                                               separators=(",", ":"), allow_nan=False)


def test_absent_full_request_retains_exact_original_document_and_input_pins(controls):
    """The real request reader retains the original ten-field contract without the option."""
    expected = deepcopy(controls.request)
    actual, pins = raw.read_build_request(controls.build["raw_build_request"], artifact_kind="raw_rig_baseline")
    assert actual == expected and "analysis_exclusions" not in actual
    assert pins == tuple([controls.build["raw_build_request"], expected["baseline"], expected["profile"],
                          expected["series_context"]["manifest"], *expected["configuration_files"],
                          *(row["source"] for row in expected["protocol_inputs"]), expected["shape_basis_certificate"]])


@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_shape_basis_request_cannot_gain_native_analysis_fields(controls, field):
    """Shape-only requests retain their exact original fields and cannot admit this option."""
    request = deepcopy(controls.request)
    request.update(format=raw.BASIS_REQUEST_FORMAT, shape_basis_certificate=None,
                   operation_overrides={"data.raw_camera": {}})
    request[field] = _declaration()
    bound = _doc(controls.root / "invalid-shape-request.json", request)
    with pytest.raises(ValueError, match="fields or artifact role"):
        raw.read_build_request(bound, artifact_kind="raw_shape_basis")


@pytest.mark.parametrize("invalid", [None, "not-a-record", []])
@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_optional_request_record_cannot_be_null_or_a_locator(controls, invalid, field):
    """An authored optional field requires the original record rather than a replacement shorthand."""
    request = dict(deepcopy(controls.request), **{field: invalid})
    bound = _doc(controls.root / "invalid-optional-request.json", request)
    with pytest.raises(ValueError, match="original source record"):
        raw.read_build_request(bound, artifact_kind="raw_rig_baseline")


@pytest.mark.parametrize("change", ["missing_nested", "typed_difference"])
def test_full_request_cannot_change_the_native_products_gap_record(controls, change):
    """Two supplied records must describe the same literal optional gap before dependency reads."""
    gap = _declaration()
    native = dict(_declaration(), analysis_exclusions=deepcopy(gap))
    if change == "missing_nested":
        native.pop("analysis_exclusions")
    else:
        native["analysis_exclusions"]["provenance"]["literal_value"] = True
    request = dict(deepcopy(controls.request), analysis_exclusions=gap, native_product=native)
    bound = _doc(controls.root / "mismatched-native-request.json", request)
    with pytest.raises(ValueError, match="native product and analysis exclusions differ"):
        raw.read_build_request(bound, artifact_kind="raw_rig_baseline")


@pytest.mark.parametrize("with_gap", [False, True])
def test_full_request_pins_real_native_dependencies_without_decoding(controls, monkeypatch, with_gap):
    """A full request retains every actual source byte pin, including a no-gap native product."""
    import numpy as np

    def no_arrays(*args, **kwargs):
        """Metadata dependency binding must not decode the opaque scientific leaves."""
        raise AssertionError("No scientific array may be decoded by request binding")

    native, expected = _native_contract(controls, with_gap=with_gap)
    request = dict(deepcopy(controls.request), native_product=native)
    if with_gap:
        request["analysis_exclusions"] = deepcopy(native["analysis_exclusions"])
    bound = _doc(controls.request_path, request)
    monkeypatch.setattr(np, "load", no_arrays)
    actual, pins = raw.read_build_request(bound, artifact_kind="raw_rig_baseline")
    assert actual == request
    assert all(row in pins for row in expected)
    dependencies = raw._native_request_dependencies(request)
    assert len({(row["path"], row["sha256"]) for row in dependencies}) == len(dependencies)
    assert not Path(request["output"]["destination"]).exists()


def test_full_request_detects_changed_optional_source_bytes(controls):
    """An unchanged request cannot accept a changed same-size native source leaf."""
    native, _ = _native_contract(controls, with_gap=True)
    bound = _doc(controls.request_path, dict(deepcopy(controls.request), native_product=native))
    path = Path(native["analysis_exclusions"]["mask"]["path"])
    original = path.read_bytes()
    path.write_bytes(b"X" + original[1:])
    with pytest.raises(ValueError):
        raw.read_build_request(bound, artifact_kind="raw_rig_baseline")
    assert path.stat().st_size == len(original)


@pytest.mark.parametrize("with_gap", [False, True])
def test_real_catalogue_retains_native_dependencies_and_original_export_join(controls, monkeypatch, with_gap):
    """Real catalogue/root checks bind the declared native files without granting scientific use."""
    import numpy as np

    def no_arrays(*args, **kwargs):
        """A control-only catalogue cannot consume native array bodies."""
        raise AssertionError("No array may be decoded by the control catalogue")

    native, expected = _native_contract(controls, with_gap=with_gap)
    config, request = _bind_native_controls(controls, native)
    catalog = InputCatalog({"private": str(controls.root)})
    monkeypatch.setattr(np, "load", no_arrays)
    result = inputs.catalogue_setup_inputs(catalog, config)
    assert result["raw_request"] == request
    native_rows = [row for row in catalog.entries.values()
                   if any(use.startswith("setup.raw.native.") for use in row["uses"])]
    for source in expected:
        assert any(row["reference"]["sha256"] == source["sha256"]
                   and row["reference"]["bytes"] == source["size_bytes"] for row in native_rows)
    assert not Path(request["output"]["destination"]).exists()


def test_native_dependency_root_admission_precedes_hash_reader(controls, monkeypatch):
    """A discovered descriptor outside the catalogue roots refuses before its bytes are opened."""
    native, _ = _native_contract(controls, with_gap=False)
    outside = controls.root.parent / (controls.root.name + "-outside.json")
    native["source_model"] = _doc(outside, dict(format="outside-transport-fixture"))
    config, _ = _bind_native_controls(controls, native)
    original_open = Path.open

    def admitted_open(path, *args, **kwargs):
        """Detect any ordering regression which opens the unadmitted nested source."""
        if path == outside:
            raise AssertionError("The unadmitted source must not be opened")
        return original_open(path, *args, **kwargs)

    def no_raw_verifier(*args, **kwargs):
        """The original request verifier follows, rather than precedes, root admission."""
        raise AssertionError("The original request verifier must remain unreachable")

    monkeypatch.setattr(Path, "open", admitted_open)
    monkeypatch.setattr(raw, "read_build_request", no_raw_verifier)
    with pytest.raises(StoreMapError, match="No declared root"):
        inputs.catalogue_setup_inputs(InputCatalog({"private": str(controls.root)}), config)


def test_native_dependency_cannot_borrow_a_byte_identical_alias(controls):
    """The optional source graph preserves the original physical-locator rule."""
    native, _ = _native_contract(controls, with_gap=False)
    config, _ = _bind_native_controls(controls, native)
    source = native["source_model"]
    copied = _file(controls.root / "aliased-model.json", Path(source["path"]).read_bytes())
    catalog = InputCatalog({"private": str(controls.root)}, aliases=[dict(logical=source, physical=copied)])
    with pytest.raises(ValueError, match="original physical locators"):
        inputs.catalogue_setup_inputs(catalog, config)


@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_actual_export_cannot_change_a_planned_native_record(controls, field):
    """An independently rebound Apply receipt cannot substitute its planned literal source record."""
    native, _ = _native_contract(controls, with_gap=True)
    config, _ = _bind_native_controls(controls, native)
    exported = json.loads(Path(controls.export["path"]).read_text(encoding="utf-8"))
    exported["binding"][field]["run_id"] = "different"
    export_bound = _doc(Path(controls.export["path"]), exported)
    applied = json.loads(Path(controls.applied["path"]).read_text(encoding="utf-8"))
    applied["export"] = export_bound
    config["apply_completion"] = _doc(Path(controls.applied["path"]), applied)
    coordinator = json.loads(controls.coordinator_path.read_text(encoding="utf-8"))
    coordinator["runs"][0].update(export=export_bound, apply_completion=config["apply_completion"])
    config["coordinator_completion"] = _doc(controls.coordinator_path, coordinator)
    with pytest.raises(ValueError, match="planned " + field):
        inputs.catalogue_setup_inputs(InputCatalog({"private": str(controls.root)}), config)


@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_raw_build_cannot_lose_ordered_toml_record_before_output_reservation(controls, monkeypatch, field):
    """A request without the configured option refuses before profile interpretation or writers."""
    def forbidden(*args, **kwargs):
        """Mark the first operation that must remain unreachable after the identity mismatch."""
        raise AssertionError("No profile or output admission may follow a lost source record")

    monkeypatch.setattr(raw, "profile_parameters", forbidden)
    monkeypatch.setattr(raw, "OwnedRawOutput", forbidden)
    request = controls.request
    camera = dict(configuration=request["profile"]["path"], sha256=request["profile"]["sha256"],
                  profile_id="fixture", run_id="sample")
    with pytest.raises(ValueError, match="ordered configuration files"):
        raw.begin_raw_rig_build(Path(controls.configuration["path"]), Path(request["baseline"]["path"]),
            camera, Path(request["output"]["destination"]), configuration_sources=(controls.configuration,),
            request_binding=controls.build["raw_build_request"], **{field: _declaration()})
    assert not Path(request["output"]["destination"]).exists()


def _optional_plan(controls, field="analysis_exclusions"):
    """Declare matching plan/settings DATA without asserting scientific mask verification."""
    plan = deepcopy(controls.plan)
    selected = plan["build_plans"][0]
    request = selected["request_template"]
    record = _declaration()
    request[field] = deepcopy(record)
    selected["raw_setup_settings"] = dict(run_id="sample", rig_path=request["output"]["destination"],
        raw_camera=dict(configuration=request["profile"]["path"], sha256=request["profile"]["sha256"],
                        profile_id="fixture", run_id="sample"), **{field: deepcopy(record)})
    return plan


@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_setup_metadata_retains_exact_matching_optional_record(controls, field):
    """The existing pure plan validator joins identical authored request/settings records."""
    plan = _optional_plan(controls, field)
    before = deepcopy(plan)
    assert protocol.validate_setup_plan(plan) is plan
    assert plan == before


@pytest.mark.parametrize("change", ["settings_absent", "request_absent", "typed_difference"])
@pytest.mark.parametrize("field", ["analysis_exclusions", "native_product"])
def test_setup_request_cannot_disagree_with_its_original_optional_settings(controls, change, field):
    """Missing and bool-versus-number differences refuse before any operation is staged."""
    plan = _optional_plan(controls, field)
    selected = plan["build_plans"][0]
    if change == "settings_absent":
        selected["raw_setup_settings"].pop(field)
    elif change == "request_absent":
        selected["request_template"].pop(field)
    else:
        selected["raw_setup_settings"][field]["provenance"]["literal_value"] = True
    with pytest.raises(ValueError, match=field):
        protocol.validate_setup_plan(plan)


@pytest.mark.parametrize("change", ["missing_nested", "typed_difference"])
def test_setup_metadata_cannot_bind_different_gap_records_to_one_native_product(controls, change):
    """Even matching settings/request copies cannot contradict the native product's nested gap."""
    plan = _optional_plan(controls)
    selected = plan["build_plans"][0]
    native = _declaration()
    if change == "typed_difference":
        native["analysis_exclusions"] = _declaration()
        native["analysis_exclusions"]["provenance"]["literal_value"] = True
    for value in (selected["request_template"], selected["raw_setup_settings"]):
        value["native_product"] = deepcopy(native)
    with pytest.raises(ValueError, match="native product and analysis exclusions differ"):
        protocol.validate_setup_plan(plan)


@pytest.mark.parametrize("fields", [(), ("analysis_exclusions",), ("native_product",), ("analysis_exclusions", "native_product")])
def test_original_setup_forwards_only_an_explicit_rig_record(controls, monkeypatch, fields):
    """The original setup forwards exact DATA without claiming a mask or scientific build."""
    setup = importlib.import_module("darsia.presets.workflows.setup.setup_rig")
    records = {field: _declaration() for field in fields}
    if set(fields) == {"analysis_exclusions", "native_product"}:
        records["native_product"]["analysis_exclusions"] = deepcopy(records["analysis_exclusions"])
    data = SimpleNamespace(data=[], baseline=Path(controls.baseline["path"]), use_cache=False,
        raw_camera=dict(configuration=controls.request["profile"]["path"],
                        sha256=controls.request["profile"]["sha256"], profile_id="fixture", run_id="sample"))
    config = SimpleNamespace(data=data, depth=SimpleNamespace(depth_map="depth"),
        labeling=SimpleNamespace(labels="labels"), facies=SimpleNamespace(path="facies", props="props"),
        protocols=SimpleNamespace(imaging="imaging", injection="injection", pressure_temperature="state", blacklist=None),
        rig=SimpleNamespace(path=controls.root / "not-created-rig", **records),
        corrections=None, image_porosity=None, check=lambda *args: None)
    monkeypatch.setattr(setup, "FluidFlowerConfig", lambda *a, **k: config)
    monkeypatch.setattr(setup.darsia, "ProtocolledExperiment", lambda **k: "explicit fixture experiment")
    from darsia.fluidflower import rig_raw_basis
    from darsia.presets.workflows.config import utils
    monkeypatch.setattr(utils, "_get_section_from_toml", lambda *a: dict(raw_camera=data.raw_camera, use_cache=False))
    monkeypatch.setattr(raw, "read_build_request", lambda *a, **k: ({}, ()))
    monkeypatch.setattr(rig_raw_basis, "require_protocol_sources", lambda *a: None)
    calls = []
    operation = SimpleNamespace(raw_source=controls.baseline, operation_id="fixture", publish=lambda rig: None)

    def begin(*args, **kwargs):
        """Capture only argument forwarding at the already separately tested producer boundary."""
        calls.append(("begin", kwargs))
        return operation

    class RecordingRig:
        """Record setup parameters without constructing any scientific buffers."""
        def setup(self, **kwargs):
            """Retain caller arguments for exact optional-record comparison."""
            calls.append(("setup", kwargs))

        def save(self, folder):
            """Record the original save call without opening output files."""
            calls.append(("save", folder))

    monkeypatch.setattr(raw, "begin_raw_rig_build", begin)
    setup.setup_rig(RecordingRig, Path(controls.configuration["path"]),
                    raw_build_request=controls.build["raw_build_request"])
    for name, arguments in calls[:2]:
        assert name in {"begin", "setup"}
        for field in ("analysis_exclusions", "native_product"):
            assert (field in arguments) is (field in fields)
            if field in fields:
                assert arguments[field] == records[field]
    assert not config.rig.path.exists()
