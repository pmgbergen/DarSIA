"""Positive by-name configuration transactions for an adopted painting.

The original corrected export and Apply own file selection and CAS writes.
This helper only binds their immutable adopted revision and parent CSV, then
rederives each exact new TOML. It never edits a Store, physical recipe or fitted
model, and it grants no RAW output or training-context authority.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

FORMAT = "darsia-painted-setup-configuration-rebuild-v1"


def native_rig_literals(value, *, run_id=None):
    """Keep bound native contracts as dictionaries until final TOML encoding."""
    allowed = {"native_product": "darsia-painted-native-product-v1",
               "analysis_exclusions": "darsia-approved-analysis-gap-native-v1"}
    if not isinstance(value, dict) or not value or not set(value) <= set(allowed):
        raise ValueError("Native rig settings require only explicit native product/exclusion dictionaries")
    result = {}
    for key, contract in value.items():
        if (not isinstance(contract, dict) or contract.get("format") != allowed[key]
                or not isinstance(contract.get("run_id"), str)
                or (run_id is not None and contract["run_id"] != run_id)):
            raise ValueError("Native rig settings belong to another format or selected run")
        # In particular, retain nested nulls. TOML dictionary serialization
        # drops them, so dictionaries must not be handed to the TOML encoder.
        result[key] = json.loads(json.dumps(contract, ensure_ascii=False, allow_nan=False))
    if (len({item["run_id"] for item in result.values()}) != 1
            or (set(result) == set(allowed)
                and result["native_product"].get("analysis_exclusions") != result["analysis_exclusions"])):
        raise ValueError("The native product and analysis exclusion settings disagree")
    return result


def native_rig_toml(value):
    """Serialize the producer's exact dictionaries losslessly under rig keys."""
    return {key: json.dumps(contract, ensure_ascii=True, allow_nan=False,
                            sort_keys=True, separators=(",", ":"))
            for key, contract in native_rig_literals(value).items()}


def clears_analysis_exclusions(value):
    """Only an explicit new native contract with literal null retires old C metadata."""
    if value is None:
        return False
    native = native_rig_literals(value).get("native_product")
    return (native is not None and "analysis_exclusions" in native
            and native["analysis_exclusions"] is None)


def _bound_bytes(descriptor):
    """Keep the bounded bytes actually decoded equal to their frozen identity."""
    from . import corrected_setup as setup
    raw = setup._small(descriptor["path"])
    if (len(raw) != descriptor["size_bytes"]
            or hashlib.sha256(raw).hexdigest() != descriptor["sha256"]):
        raise ValueError("The painted configuration source differs from its frozen bytes")
    setup._verified(descriptor)
    return raw


def raw_settings_for_run(value, *, run_id):
    """Validate explicit per-run RAW settings without choosing a camera profile."""
    from .rig_raw_input import camera_binding
    required = {"run_id", "raw_camera", "rig_path"}
    optional = {"native_product", "analysis_exclusions"}
    if (not isinstance(value, dict) or not required <= set(value) or set(value) - required - optional
            or value["run_id"] != run_id):
        raise ValueError("RAW Setup settings require the exact selected run")
    camera = camera_binding(value["raw_camera"])
    if camera is None or camera["run_id"] != run_id:
        raise ValueError("RAW Setup camera belongs to another selected run")
    target = value["rig_path"]
    if not isinstance(target, str) or not target or not Path(target).is_absolute():
        raise ValueError("RAW Setup requires an explicit absolute rig output")
    result = dict(run_id=run_id, raw_camera=camera, rig_path=target)
    declared = {key: value[key] for key in optional if key in value}
    if declared:
        result.update(native_rig_literals(declared, run_id=run_id))
    return result


def require_raw_output(settings, receipt, *, fresh=False):
    """Retain the original declared output boundary and its fresh-build guard."""
    from . import corrected_setup as setup
    settings = raw_settings_for_run(settings, run_id=receipt["run_id"])
    dual = receipt["binding"].get("dual_configuration_binding") or {}
    pointer = dual.get("output_pointer_change")
    if dual.get("format") == setup.PRIVATE_MANUAL_CONFIGURATION_FORMAT:
        pointer = dual["output_pointer_changes"]["rig.path"]
    if pointer is not None and settings["rig_path"] != pointer["after"]:
        raise ValueError("RAW Setup rig output differs from its original bound fresh target")
    if fresh:
        target = Path(settings["rig_path"])
        if not setup._same(target.resolve(), target):
            raise ValueError("The fresh RAW rig output must not resolve through another location")
        if target.exists() and (not target.is_dir() or next(target.iterdir(), None) is not None):
            raise ValueError("The RAW rig output is not empty; existing artifacts were preserved")


def raw_settings_bytes(before, settings):
    """Write explicit RAW settings and optional lossless native contract strings."""
    import toml
    import tomllib
    from . import corrected_setup as setup
    settings = raw_settings_for_run(settings, run_id=settings["run_id"])
    document = tomllib.loads(before.decode("utf-8-sig"))
    expected = deepcopy(document)
    for section in ("data", "rig"):
        if section in expected and not isinstance(expected[section], dict):
            raise ValueError("RAW Setup settings require ordinary data and rig tables")
        expected.setdefault(section, {})
    expected["data"]["raw_camera"] = deepcopy(settings["raw_camera"])
    expected["data"]["use_cache"] = False
    expected["rig"]["path"] = settings["rig_path"]
    declared = {key: settings[key] for key in ("native_product", "analysis_exclusions") if key in settings}
    if declared:
        if clears_analysis_exclusions(declared):
            expected["rig"].pop("analysis_exclusions", None)
        expected["rig"].update(native_rig_toml(declared))
    if setup._typed_equal(document, expected):
        return before
    after = toml.dumps(expected).encode("utf-8")
    if not setup._typed_equal(tomllib.loads(after.decode("utf-8")), expected):
        raise ValueError("RAW Setup serialization changed another TOML value or type")
    return after


def context_for_current(current, *, raw_setup_settings=None):
    """Freeze the actual immutable Store record and its original CSV identity."""
    from . import corrected_setup as setup
    table = current["configuration"]["painted_adoption_table"]
    path = Path(current["artifacts"]["facies_properties_path"]).parent / "revision.json"
    bound = setup._descriptor(path)
    if json.loads(_bound_bytes(bound)) != current:
        raise ValueError("The selected painting revision differs from its immutable Store record")
    parent = deepcopy(table["source"]["csv"])
    setup._verified(parent)
    context = dict(format=FORMAT, revision=bound, parent_csv=parent)
    if raw_setup_settings is not None:
        context["raw_setup_settings"] = raw_settings_for_run(
            raw_setup_settings, run_id=raw_setup_settings.get("run_id") if isinstance(raw_setup_settings, dict) else None)
    return context


def read_context(context, *, revision_id):
    """Read only the bound revision and CSV; never follow a latest revision."""
    from . import corrected_setup as setup
    keys = {"format", "revision", "parent_csv"}
    if (not isinstance(context, dict) or set(context) not in (keys, keys | {"raw_setup_settings"})
            or context["format"] != FORMAT):
        raise ValueError("The painted configuration transaction lacks its exact source binding")
    current = json.loads(_bound_bytes(context["revision"]))
    if current.get("revision_id") != revision_id:
        raise ValueError("The configuration transaction belongs to another adopted revision")
    config = current["configuration"]
    if context["parent_csv"] != config["painted_adoption_table"]["source"]["csv"]:
        raise ValueError("The configuration transaction changed its original parent CSV")
    if "raw_setup_settings" in context:
        settings = context["raw_setup_settings"]
        canonical = raw_settings_for_run(settings, run_id=settings.get("run_id") if isinstance(settings, dict) else None)
        if not setup._typed_equal(settings, canonical):
            raise ValueError("The RAW Setup transaction changed its canonical settings")
    raw = _bound_bytes(context["parent_csv"])
    return config, raw


def rebuild_bytes(before, context, *, revision_id, replacements=None, is_common=False, native_rig=None):
    """Rebuild exact named inputs, optionally applying the original four pointers."""
    from .painted_facies_configuration import rebuild_painted_facies_toml
    from . import corrected_setup as setup
    config, parent = read_context(context, revision_id=revision_id)
    rebuilt = rebuild_painted_facies_toml(before, config, parent_csv_bytes=parent)
    after = rebuilt.toml_bytes
    if replacements is not None:
        after = setup.corrected_config_bytes(after, replacements, native_rig=native_rig)
    if not is_common and "raw_setup_settings" in context:
        after = raw_settings_bytes(after, context["raw_setup_settings"])
    return after, deepcopy(rebuilt.evidence)


def require_rebuilt_inputs(config, paths, context, *, revision_id, receipt=None):
    """Apply the existing stale-id seat belt to actual deterministic postimages."""
    import tomllib
    from . import corrected_setup as setup
    from .painted_facies_inputs import require_current_facies_inputs
    documents = {}
    for path in paths:
        path = Path(path)
        if path.is_file() and str(path) not in documents:
            after, _ = rebuild_bytes(setup._small(path), context, revision_id=revision_id,
                                     is_common=receipt is not None and is_common_target(path, receipt))
            documents[str(path)] = tomllib.loads(after.decode("utf-8-sig"))
    require_current_facies_inputs(config, documents)


def completion_context(value, claim, record, *, receipt):
    """Require all three immutable application records to name the same recipe."""
    context = value.get("facies_rebuild")
    if context is None:
        if "facies_rebuild" in claim or "facies_rebuild" in record:
            raise ValueError("The completed application dropped its painted configuration recipe")
        return None
    if claim.get("facies_rebuild") != context or record.get("facies_rebuild") != context:
        raise ValueError("The application records disagree on their painted configuration recipe")
    config, _ = read_context(context, revision_id=receipt["revision_id"])
    if "raw_setup_settings" in context:
        require_raw_output(context["raw_setup_settings"], receipt)
    if context["revision"]["sha256"] != receipt["current_sha256"]:
        raise ValueError("The painted configuration recipe differs from the exact exported revision")
    properties = receipt["binding"]["properties"]
    from . import corrected_setup as setup
    current = json.loads(_bound_bytes(context["revision"]))
    actual = setup._descriptor(current["artifacts"]["facies_properties_path"])
    if any(actual[key] != properties[key] for key in ("sha256", "size_bytes")):
        raise ValueError("The painted configuration recipe differs from the exported adopted table")
    return context


def is_common_target(path, receipt):
    """Only the original bundle's literal common target has no pointer changes."""
    from . import corrected_setup as setup
    common = receipt["binding"].get("common_configuration")
    return common is not None and setup._same(path, common["path"])


def verify_applied_postimage(before, after, context, *, receipt, target):
    """Replay the pure named transformation against retained before bytes."""
    common = is_common_target(target, receipt)
    replacements = None if common else receipt["binding"]["replacements"]
    expected, _ = rebuild_bytes(before, context, revision_id=receipt["revision_id"],
                                replacements=replacements, is_common=common,
                                native_rig=None if common else setup_native_rig(receipt["binding"]))
    if after != expected:
        raise ValueError("The applied configuration differs from its exact by-name transformation")


def setup_native_rig(binding):
    """Project only explicit contracts already bound to the corrected export."""
    value = {key: binding[key] for key in ("native_product", "analysis_exclusions") if key in binding}
    return native_rig_literals(value) if value else None


def original_common_binding(path, receipt, destination):
    """Keep the original shared input binding after its first proved application.

    This does not admit an arbitrary edited common file. The original Apply
    completion checks its exact before/after bytes and whole named recipe.
    """
    from . import corrected_setup as setup
    bound = (receipt.get("binding") or {}).get("common_configuration")
    if bound is None or not setup._same(path, bound["path"]):
        raise ValueError("The common configuration does not match the original export")
    complete_path = Path(destination) / "apply-complete.json"
    if not complete_path.is_file():
        setup._verified(bound)
        return deepcopy(bound)
    header = json.loads(setup._small(complete_path))
    if "facies_rebuild" not in header:
        setup._verified(bound)
        return deepcopy(bound)
    completed = setup._apply_completion(Path(destination), receipt)
    matches = [row for row in completed["configurations"] if setup._same(row["after"]["path"], path)]
    if not matches:
        # A later selected run binds the shared file's already-applied bytes;
        # its own transaction has no reason to replace that same common file.
        setup._verified(bound)
        return deepcopy(bound)
    if len(matches) != 1 or any(matches[0]["before"][key] != bound[key] for key in ("sha256", "size_bytes")):
        raise ValueError("The applied common configuration lacks its original bound before bytes")
    return deepcopy(bound)
