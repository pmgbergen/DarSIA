"""UNRUN source-increment tests: unsealed DATA only, never Build acceptance.

No fixture is a released plan, authenticated publication or eligible physical
context. These tests exercise literal associations and explicit refusals.
"""
from copy import deepcopy
import ast
import inspect

import pytest

from darsia_cache.current_build_binding import (
    FORMAT, checked_current_build_binding, current_build_binding_inputs,
)
from darsia_cache.first_qualification import checked_current_build_plan_metadata
from darsia_cache.policy import CachePolicyError


def descriptor_TESTONLY(name, *, digest="a" * 64, size=7):
    return dict(path="C:/TESTONLY/" + name, sha256=digest, size_bytes=size)


def binding_TESTONLY():
    root = "C:/TESTONLY/publication"
    paths = {"job": "job.json", "task": "results/ac52/task.json", "done": "done/ac52.json",
             "operation_result": "publishing/ac52/metadata/operation_result.json"}
    publication = dict(root=root, **{key: descriptor_TESTONLY("publication/" + path)
                                   for key, path in paths.items()})
    publication["readset"] = sorted(
        [dict(relative_path=path, descriptor=deepcopy(publication[key])) for key, path in paths.items()],
        key=lambda row: row["relative_path"])
    raw = descriptor_TESTONLY("ac52_f1.ARW")
    geometry = descriptor_TESTONLY("geometry.json")
    frame = dict(case_id="ac52_f1", raw_acquisition=raw, bridge_frame=descriptor_TESTONLY("frame.json"),
        geometry=geometry, area_provenance=descriptor_TESTONLY("area_data.json"),
        static_provenance=descriptor_TESTONLY("static_components.json"))
    binding = dict(format=FORMAT, run="ac52", publication=publication,
        operation_result=descriptor_TESTONLY("owned/operation_result.json"),
        bridge=descriptor_TESTONLY("bridge.json"), frames=[frame])
    config = dict(format="darsia-native-phase-n1-run-v1", run="ac52", mass={"template": {}},
        frames=[dict(case_id="ac52_f1", source=dict(source_sha256=raw["sha256"],
            source_binding={"raw_acquisition": deepcopy(raw)}), geometry=deepcopy(geometry),
            native_task={"kind": "raw_rig"})], current_build_binding=binding)
    return binding, config


def catalog_TESTONLY(binding):
    return {"inputs": [dict(reference=dict(role=role, sha256=bound["sha256"], bytes=bound["size_bytes"]),
                            uses=[use]) for use, bound, role in current_build_binding_inputs(binding)]}


def test_unsealed_binding_is_copied_data_and_never_an_acceptance():
    binding, config = binding_TESTONLY()
    before = deepcopy(binding)
    checked = checked_current_build_binding(binding, config=config)
    assert checked == before and checked is not binding
    checked["frames"][0]["case_id"] = "changed"
    assert binding == before
    assert not {"accepted", "complete", "passed", "configuration", "task", "plan"} & binding.keys()
    assert checked_current_build_binding(dict(binding, independent_eligibility=None), config=config)


def test_same_unsealed_binding_joins_p1_without_mass_calibration_or_stage_rewrite():
    binding, config = binding_TESTONLY()
    config.pop("mass")
    config.update(format="darsia-native-phase-statistics-run-v1", statistics={})
    before = deepcopy(config)
    assert checked_current_build_binding(binding, config=config) == binding
    assert config == before and "mass" not in config
    assert "independent_eligibility" not in binding


@pytest.mark.parametrize("fault", ["missing_statistics", "pixel_domain", "other_stage", "rebound_geometry"])
def test_p1_binding_refuses_missing_consumer_or_changed_frame(fault):
    binding, config = binding_TESTONLY()
    config.pop("mass")
    config.update(format="darsia-native-phase-statistics-run-v1", statistics={})
    if fault == "missing_statistics":
        del config["statistics"]
    elif fault == "pixel_domain":
        config["statistics"]["pixel_domain"] = None
    elif fault == "other_stage":
        config["format"] = "darsia-n1-phase2-run-v1"
    else:
        config["frames"][0]["geometry"]["sha256"] = "b" * 64
    with pytest.raises(ValueError):
        checked_current_build_binding(binding, config=config)


@pytest.mark.parametrize("where", ["binding", "publication", "readset", "descriptor", "frame"])
def test_every_binding_layer_is_closed(where):
    binding, config = binding_TESTONLY()
    target = {"binding": binding, "publication": binding["publication"],
        "readset": binding["publication"]["readset"][0], "descriptor": binding["bridge"],
        "frame": binding["frames"][0]}[where]
    target["invented_acceptance"] = True
    with pytest.raises(ValueError, match="closed DATA"):
        checked_current_build_binding(binding, config=config)


@pytest.mark.parametrize("claim", [False, True, {}, {"accepted": True}, "a" * 64])
def test_no_supplied_independent_eligibility_claim_is_accepted(claim):
    binding, config = binding_TESTONLY()
    binding["independent_eligibility"] = claim
    with pytest.raises(ValueError, match="eligibility is unimplemented"):
        checked_current_build_binding(binding, config=config)


@pytest.mark.parametrize("field,value", [("sha256", 123), ("sha256", "A" * 64),
    ("size_bytes", True), ("size_bytes", -1), ("size_bytes", 7.0), ("size_bytes", float("nan")),
    ("size_bytes", float("inf")), ("path", "../outside.json"), ("path", "C:/TESTONLY/../outside.json")])
def test_descriptor_types_and_nonfinite_values_are_refused(field, value):
    binding, _config = binding_TESTONLY()
    binding["bridge"][field] = value
    with pytest.raises(ValueError):
        checked_current_build_binding(binding)


@pytest.mark.parametrize("fault", ["duplicate", "case_duplicate", "reordered", "escape", "missing", "different_copy"])
def test_publication_readset_is_exact_bounded_and_rooted(fault):
    binding, _config = binding_TESTONLY()
    publication = binding["publication"]
    rows = publication["readset"]
    if fault == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif fault == "case_duplicate":
        copy = deepcopy(rows[0]); copy["relative_path"] = copy["relative_path"].upper()
        copy["descriptor"]["path"] = copy["descriptor"]["path"].upper(); rows.append(copy)
    elif fault == "reordered":
        rows.reverse()
    elif fault == "escape":
        rows[0]["relative_path"] = "../outside.json"
    elif fault == "missing":
        rows.pop(0)
    else:
        publication["operation_result"]["sha256"] = "b" * 64
    with pytest.raises(ValueError):
        checked_current_build_binding(binding)


@pytest.mark.parametrize("fault", ["source", "geometry", "case", "kind", "duplicate_frame", "pixel_domain"])
def test_binding_must_join_the_actual_supported_consumer(fault):
    binding, config = binding_TESTONLY()
    frame = config["frames"][0]
    if fault == "source":
        frame["source"]["source_binding"]["raw_acquisition"]["path"] = "C:/TESTONLY/unrelated.ARW"
    elif fault == "geometry":
        frame["geometry"]["sha256"] = "b" * 64
    elif fault == "case":
        frame["case_id"] = "ac52_f2"
    elif fault == "kind":
        frame["native_task"]["kind"] = "rig"
    elif fault == "duplicate_frame":
        binding["frames"].append(deepcopy(binding["frames"][0]))
    else:
        config["mass"]["template"]["pixel_domain"] = None
    with pytest.raises(ValueError):
        checked_current_build_binding(binding, config=config)


def test_current_plan_metadata_join_adds_no_authority_or_mutation():
    binding, config = binding_TESTONLY()
    catalog = catalog_TESTONLY(binding)
    assert checked_current_build_plan_metadata(config, binding["operation_result"], catalog) == binding
    assert "independent_eligibility" not in binding


@pytest.mark.parametrize("fault", ["missing_binding", "unrelated_build", "missing_role", "duplicate_role", "role", "sha", "bytes"])
def test_current_plan_requires_exact_embedded_build_and_named_catalog_roles(fault):
    binding, config = binding_TESTONLY()
    catalog = catalog_TESTONLY(binding)
    build = deepcopy(binding["operation_result"])
    if fault == "missing_binding":
        config.pop("current_build_binding")
    elif fault == "unrelated_build":
        build["path"] = "C:/TESTONLY/other-operation.json"
    elif fault == "missing_role":
        catalog["inputs"].pop(0)
    elif fault == "duplicate_role":
        catalog["inputs"].append(deepcopy(catalog["inputs"][0]))
    else:
        ref = catalog["inputs"][0]["reference"]
        ref[{"role": "role", "sha": "sha256", "bytes": "bytes"}[fault]] = {
            "role": "image", "sha": "b" * 64, "bytes": True}[fault]
    with pytest.raises(CachePolicyError, match="Current Build metadata binding refused"):
        checked_current_build_plan_metadata(config, build, catalog)


def test_schema_stays_stdlib_only_and_current_plan_retains_unconditional_stop():
    from pathlib import Path
    import darsia_cache.current_build_binding as schema
    from darsia_cache import first_qualification
    tree = ast.parse(Path(schema.__file__).read_text(encoding="utf-8"))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert all(not (getattr(node, "module", "") or "").startswith("darsia.") for node in imports)
    assert all(not name.name.startswith("darsia") for node in imports if isinstance(node, ast.Import) for name in node.names)
    function = ast.parse(inspect.getsource(first_qualification.checked_current_reference_plan)).body[0]
    assert isinstance(function.body[-1], ast.Raise)
    assert "execution binding STOP" in ast.unparse(function.body[-1])
    assert first_qualification.CURRENT_V9_REFERENCE_RELEASE is None
