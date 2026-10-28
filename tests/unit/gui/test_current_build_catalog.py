"""UNRUN metadata/stat tests using unsealed synthetic files, not accepted Builds.

Payloads are deliberately invalid image/archive bytes. No science, original
publication verifier, startup, capacity or independent eligibility is mocked.
The collector may describe these DATA; the worker still must authenticate them.
"""
from copy import deepcopy
import ast
import inspect

import pytest

from darsia_cache.current_build_binding import FORMAT
from darsia.fluidflower.pipeline.job_inputs import InputCatalog
from .test_alias_job_inputs import MemoryFiles


def closure_TESTONLY(*, local_metric=False):
    files = MemoryFiles("C:/TESTONLY")
    doc, payload = files.doc, files.file
    native, phase = doc("native.json", {}), doc("phase.json", {})
    baseline, raw = payload("baseline.ARW"), payload("ac52_f1.ARW")
    series, toml = payload("series.toml"), payload("configuration.toml")
    adoption = {"TESTONLY": "no authenticated adoption"}
    if local_metric:
        table = payload("facies.csv")
        artifact = payload("chart/ac52/chart_labels.npz")
        artifact["path"] = "ac52/chart_labels.npz"
        preparation = doc("chart/preparation.json", dict(chart_transfers={"ac52": dict(artifact=artifact)}))
        adoption.update(configuration=dict(painted_adoption_table=dict(preparation=preparation)),
                        artifacts=dict(source_mode_files=dict(facies_properties=table)))
    adopted = doc("adopted.json", adoption)
    models = {name: dict(data=payload(name + ".npz"), sidecar=doc(name + ".json", {}))
              for name in ("fit", "reference")}
    models["old_field"] = None
    raw_document = dict(baseline_raw=baseline, models=models, shape_assets=[payload("shape.npz")])
    old_raw, new_raw = doc("old_raw.json", raw_document), doc("new_raw.json", raw_document)
    request = dict(format="darsia-raw-rig-build-request-v1", run_id="ac52", baseline=baseline,
        profile=native, series_context=dict(root=files.root, manifest=series),
        shape_basis_certificate=doc("shape_certificate.json", {}), configuration_files=[toml],
        protocol_inputs=[dict(role="imaging", source=payload("protocol.csv"))],
        output=dict(destination=files.root + "/owned", intended_root=files.root), operation_overrides={})
    plan = doc("setup_plan.json", dict(format="darsia-series-setup-plan-v1", adopted_revision=adopted,
        series_manifest=series, build_plans=[dict(run_id="ac52", native_parameters=native,
            configuration_paths=[toml["path"]], request_template={key: value for key, value in request.items()
                                                                if key != "configuration_files"})]))
    request_pin = doc("raw_request.json", request)
    ownership = {name: doc("owned/" + name + ".json", {"TESTONLY": "unsealed owner DATA"})
                 for name in ("owner", "setup_started", "save_started", "save_completed")}
    outputs = list(ownership.values()) + [payload("owned/" + name + ".npz")
        for name in ("shape_corrected_baseline", "labels", "facies", "depth", "porosity")]
    certificate = doc("owned/certificate.json", dict(inputs={"operation_request": request_pin},
        outputs=outputs, output_ownership=ownership))
    result = dict(format="darsia-series-setup-result-v1", pipeline="series-setup", operation="build",
        run="ac52", complete=True, worker_references_closed=True, setup_plan=plan,
        raw_build_request=request_pin, configuration_files=[toml], baseline_renderer=certificate, produced_files=outputs)
    operation = doc("owned/operation_result.json", result)
    bridge = doc("bridge.json", dict(format="darsia-build-native-bridge-v1", operation_result=operation,
        native_configuration=native, photometry_source=old_raw, raw_source_binding=new_raw,
        producer={"module": payload("bridge_producer.py")}))
    native_frame = doc("native_frame.json", dict(run="ac52", acquisition=raw))
    bridge_frame = doc("frame.json", dict(format="darsia-build-native-frame-v1", bridge=bridge,
        raw_source_binding=new_raw, native_frame=native_frame, products=payload("native.npz")))
    inputs = {name: doc(name + ".json", {}) for name in ("old_geometry", "old_raw_audit",
        "old_native_configuration", "old_acquisition_collection", "old_acquisition_configuration",
        "hardware_pose", "top_photo_evidence")}
    inputs.update(native_configuration=native, phase_configuration=phase, new_bridge_frame=bridge_frame,
        bridge=bridge, native_frame=native_frame, old_raw_binding=old_raw, raw_source_binding=new_raw,
        top_curve=payload("top_curve.npy"))
    geometry_producer = payload("geometry_producer.py")
    geometry = doc("geometry.json", dict(coordinate_transport=dict(inputs=inputs, producer=geometry_producer),
        frame_photo_top_keep=payload("photo_top.npz")))
    context = {key: value for key, value in inputs.items() if key != "top_curve"}
    context.update(geometry=geometry, geometry_producer=geometry_producer)
    area = dict(role="measured_parent", historical=payload("historical.area.npz"),
        measured=payload("measured.area.npz"), factor=1.25,
        revision=dict(mass_eligible=False, bottom_mass_allowed=False, vertical_scale_independently_verified=False))
    area_inputs = dict(context, **{name: doc(name + ".json", {}) for name in (
        "measurement", "physical_scale_plan", "physical_scale_receipt", "measured_frame_receipt",
        "historical_frame_receipt", "parent_seal", "parent_registration")})
    area_provenance = doc("area_data.json", dict(area=area, inputs=area_inputs, producer=payload("area_producer.py")))
    components = payload("static_components.npz")
    static_inputs = dict(context, **{name: doc(name + ".json", {}) for name in (
        "old_configuration", "old_phase_configuration", "old_native_frame")},
        old_components=payload("old_components.npz"), producer=payload("static_producer.py"))
    static_provenance = doc("static_components.json", dict(format="darsia-static-component-coordinate-data-v1",
        run="ac52", case_id="ac52_f1", acquisition=raw, inputs=static_inputs, components=components,
        physical_qualification=False))
    root = files.root + "/publication"
    job = doc("publication/job.json", dict(pipeline="series-setup"))
    task = doc("publication/results/ac52/task.json", dict(pipeline="series-setup", run="ac52", operation="build"))
    copied = doc("publication/publishing/ac52/products/metadata/operation_result.json", result)
    completed_value = dict(result, attempt_id="a" * 32, files={"operation_result.json": operation})
    completed = doc("publication/results/ac52/completed.json", completed_value)
    completion_copy = doc("publication/publishing/ac52/products/completed.json", completed_value)
    owner = doc("publication/publishing/ac52/owner.json", {"TESTONLY": "no actual owner or claim"})

    def reference(bound, relative):
        return dict(relative_path=relative, sha256=bound["sha256"], bytes=bound["size_bytes"])

    products = [dict(product_id=name, sha256=bound["sha256"], bytes=bound["size_bytes"],
                     metadata_copy=reference(bound, relative))
        for name, bound, relative in (("completed.json", completion_copy, "completed.json"),
            ("operation_result.json", copied, "metadata/operation_result.json"))]
    index = doc("publication/publishing/ac52/products/products_index.json",
        dict(format="darsia-products-index-v1", run="ac52", attempt_id="a" * 32, products=products))
    done = doc("publication/done/ac52.json", dict(format="darsia-alias-done-v1", run="ac52", attempt_id="a" * 32,
        completed=reference(completed, "results/ac52/completed.json"),
        products_index=reference(index, "publishing/ac52/products/products_index.json")))
    readset = sorted([dict(relative_path=bound["path"][len(root) + 1:], descriptor=bound)
        for bound in (job, task, done, copied, completed, completion_copy, owner, index)], key=lambda row: row["relative_path"])
    binding = dict(format=FORMAT, run="ac52", publication=dict(root=root, job=job, task=task,
        done=done, operation_result=copied, readset=readset), operation_result=operation, bridge=bridge,
        frames=[dict(case_id="ac52_f1", raw_acquisition=raw, bridge_frame=bridge_frame, geometry=geometry,
            area_provenance=area_provenance, static_provenance=static_provenance)])
    if local_metric:
        binding["metric_model"] = doc("metric.json", {
            key: payload("metric/" + key + ".npz") for key in (
                "source_model", "current_edges", "baseline_configuration", "model_photograph",
                "baseline_to_model_registration")})
    config = dict(format="darsia-native-phase-n1-run-v1", run="ac52", native={"configuration": native},
        phase={"configuration": phase}, mass={"template": {}}, current_build_binding=binding,
        frames=[dict(case_id="ac52_f1", source=dict(source_sha256=raw["sha256"], source_binding={"raw_acquisition": raw}),
            geometry=geometry, native_task=dict(kind="raw_rig", geometry=native_frame, processor={"raw_binding": new_raw}),
            phase_roles=dict(area=area, static=dict(components=components, source_ids=[1], unchanged_source=True)))])
    return files, config


def test_named_original_closure_is_catalogued_without_opening_payloads_or_claiming_acceptance():
    files, config = closure_TESTONLY()
    before = deepcopy(config)
    catalog = files.catalog()
    assert catalog.current_build(config) == config["current_build_binding"]
    assert config == before
    uses = {use for row in catalog.entries.values() for use in row["uses"]}
    assert {"current_build.operation_result", "current_build.baseline_renderer", "current_build.ownership.owner",
        "current_build.frame.ac52_f1.geometry_inputs.top_curve", "current_build.frame.ac52_f1.geometry_producer",
        "current_build.frame.ac52_f1.area_inputs.parent_seal",
        "current_build.frame.ac52_f1.static_inputs.old_components"} <= uses
    assert all(use.startswith("current_build.") for use in uses)
    assert all(path.endswith(".json") for path in files.opened)
    assert any(path.endswith("porosity.npz") for path in files.statted)
    assert not any(path.endswith(".npz") for path in files.opened)
    assert config["frames"][0]["phase_roles"]["area"]["revision"]["mass_eligible"] is False


def test_same_named_build_closure_replays_for_p1_without_a_mass_template():
    files, config = closure_TESTONLY()
    config.pop("mass")
    config.update(format="darsia-native-phase-statistics-run-v1", statistics={})
    first, second = files.catalog(), files.catalog()
    assert first.current_build(config) == second.current_build(deepcopy(config))
    assert first.entries == second.entries
    assert all(path.endswith(".json") for path in files.opened)
    assert "mass" not in config and "independent_eligibility" not in config["current_build_binding"]


@pytest.mark.parametrize("fault", [None, "missing_role", "changed_sha", "changed_role"])
def test_p1_original_replay_demands_the_exact_prepared_named_build_roles(fault):
    """UNRUN _Replay metadata unit; the binding is not a claimed worker service."""
    from types import SimpleNamespace
    from pathlib import Path
    import hashlib
    from darsia.fluidflower.pipeline.run_io import _Replay
    from darsia.fluidflower.pipeline.job_inputs import _key
    from darsia.fluidflower.pipeline.store_runtime import InputHashMismatch
    files, config = closure_TESTONLY()
    config.pop("mass")
    config.update(format="darsia-native-phase-statistics-run-v1", statistics={})
    catalog = files.catalog()
    catalog.current_build(config)
    uses = {use: deepcopy(row["reference"]) for row in catalog.entries.values() for use in row["uses"]}
    selected = "current_build.frame.ac52_f1.area_inputs.parent_seal"
    if fault == "missing_role":
        del uses[selected]
    elif fault == "changed_sha":
        uses[selected]["sha256"] = "f" * 64
    elif fault == "changed_role":
        uses[selected]["role"] = "image"
    def for_use(use):
        if use not in uses:
            raise InputHashMismatch("TESTONLY missing prepared consumer")
        ref = uses[use]
        return SimpleNamespace(reference=ref, path=Path(files.root) / ref["relative_path"])
    locations = {}
    def bind(path, row):
        locations[_key(path)] = row
    def digest(path):
        return hashlib.sha256(files.files[str(path).replace("\\", "/")]).hexdigest()
    binding = SimpleNamespace(verified=SimpleNamespace(for_use=for_use), _metadata_bytes=files.read,
        _bind_data_path=bind, native_input_digest=digest)
    replay = _Replay(binding, [])
    if fault is None:
        replay.current_build(config)
        assert replay.entries == catalog.entries and set(replay.used) == set(uses)
    else:
        with pytest.raises(InputHashMismatch):
            replay.current_build(config)
    assert all(path.endswith(".json") for path in files.opened)


@pytest.mark.parametrize("fault", ["missing_file", "changed_metadata", "missing_index", "unrelated_source",
                                   "relocated_publication", "relocated_owned_output", "geometry_graph"])
def test_missing_or_rebound_evidence_never_produces_a_complete_current_build_catalogue(fault):
    files, config = closure_TESTONLY()
    binding = config["current_build_binding"]
    aliases = []
    if fault == "missing_file":
        del files.files[binding["bridge"]["path"]]
    elif fault == "changed_metadata":
        files.files[binding["bridge"]["path"]] = b"{}"
    elif fault == "missing_index":
        binding["publication"]["readset"] = [row for row in binding["publication"]["readset"]
            if not row["relative_path"].endswith("products_index.json")]
    elif fault == "unrelated_source":
        config["frames"][0]["source"]["source_sha256"] = "f" * 64
    elif fault in {"relocated_publication", "relocated_owned_output"}:
        original = binding["publication"]["job"] if fault == "relocated_publication" else binding["operation_result"]
        copy = files.file("relocated.json", files.files[original["path"]])
        aliases = [dict(logical=original, physical=copy)]
    else:
        import json
        bound = binding["frames"][0]["geometry"]
        value = json.loads(files.files[bound["path"]])
        value["coordinate_transport"]["inputs"]["invented_truth"] = files.doc("truth.json", {"accepted": True})
        replacement = files.doc("different_geometry.json", value)
        binding["frames"][0]["geometry"] = replacement
        config["frames"][0]["geometry"] = replacement
    with pytest.raises((ValueError, KeyError)):
        files.catalog(aliases=aliases).current_build(config)


def test_catalog_dispatch_requires_explicit_binding_for_both_original_n1_and_p1_consumers():
    tree = ast.parse(inspect.getsource(InputCatalog))
    collect = next(node for node in tree.body[0].body if isinstance(node, ast.FunctionDef) and node.name == "collect")
    calls = [node for node in ast.walk(collect) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "current_build"]
    assert len(calls) == 2
    for format_ in ("darsia-native-phase-n1-run-v1", "darsia-native-phase-statistics-run-v1"):
        outer = next(node for node in ast.walk(collect) if isinstance(node, ast.If)
            and format_ in ast.unparse(node.test))
        guard = next(node for node in outer.body if isinstance(node, ast.If)
                     and ast.unparse(node.test) == "'current_build_binding' in config")
        assert any(call in list(ast.walk(guard)) for call in calls)


def test_optional_analysis_exclusion_requires_its_actual_preparation_and_native_closure():
    # Rebind each changed ancestor using actual TESTONLY bytes. The refusal is
    # the missing original closure, not an accidentally stale checksum.
    import json
    files, config = closure_TESTONLY()
    binding = config["current_build_binding"]

    def document(bound):
        return json.loads(files.files[bound["path"]])

    result = document(binding["operation_result"])
    plan = document(result["setup_plan"])
    plan["adopted_revision"] = files.doc("adopted-with-exclusion.json", {"configuration": {"analysis_exclusions": None}})
    result["setup_plan"] = files.doc("new-setup-plan.json", plan)
    operation = files.doc("new-operation-result.json", result)
    publication = binding["publication"]
    old_shared = publication["operation_result"]
    shared = files.doc(old_shared["path"][len(files.root) + 1:], result)
    old_done = publication["done"]
    done = document(old_done)
    readset = {row["relative_path"]: deepcopy(row) for row in publication["readset"]}
    index_row = readset[done["products_index"]["relative_path"]]
    index = document(index_row["descriptor"])
    completed = dict(result, attempt_id=done["attempt_id"], files={"operation_result.json": operation})

    def replace(relative, value):
        pin = files.doc("publication/" + relative, value)
        readset[relative]["descriptor"] = pin
        return dict(relative_path=relative, sha256=pin["sha256"], bytes=pin["size_bytes"])

    done["completed"] = replace(done["completed"]["relative_path"], completed)
    parent = done["products_index"]["relative_path"].rsplit("/", 1)[0]
    for row in index["products"]:
        value = completed if row["product_id"] == "completed.json" else result
        ref = replace(parent + "/" + row["metadata_copy"]["relative_path"], value)
        row.update(sha256=ref["sha256"], bytes=ref["bytes"])
        row["metadata_copy"].update(sha256=ref["sha256"], bytes=ref["bytes"])
    done["products_index"] = replace(done["products_index"]["relative_path"], index)
    relative_done = old_done["path"][len(publication["root"]) + 1:]
    replace(relative_done, done)
    publication.update(operation_result=shared, done=readset[relative_done]["descriptor"],
        readset=[readset[key] for key in sorted(readset)])
    binding["operation_result"] = operation
    with pytest.raises(ValueError, match="original preparation and native request closure"):
        files.catalog().current_build(config)
