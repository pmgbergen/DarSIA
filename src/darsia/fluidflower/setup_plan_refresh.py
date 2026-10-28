"""A saved lighting choice may refresh a Setup plan before any Apply work."""
from copy import deepcopy
import hashlib
from pathlib import Path
import tomllib

from .pipeline.series_setup_inputs import _same

FORMAT = "darsia-setup-normalisation-refresh-v1"
IMAGE_FORMAT = "darsia-setup-image-preparation-refresh-v1"
LIMIT = 1024 * 1024


def _raw(path):
    with Path(path).open("rb") as stream:
        value = stream.read(LIMIT + 1)
    if len(value) > LIMIT:
        raise ValueError("Series settings exceed the 1 MiB metadata limit")
    return value


def _matches(raw, bound):
    return len(raw) == bound["size_bytes"] and hashlib.sha256(raw).hexdigest() == bound["sha256"]


def _normalisation_only(previous, current, folder):
    """Both series documents, when only the lighting choice (and added aliases) changed.

    Storage alias rows added since the plan only relocate reads to exact copies
    (``folder`` is the manifest folder their tables are read against); every
    earlier row and every other setting must stay the same.
    """
    from .image_normalisation import saved_choice
    from .series_method import _same_values
    from .series_paths import series_key
    from .storage_paths import only_added_storage_aliases
    before, after = (tomllib.loads(raw.decode("utf-8-sig")) for raw in (previous, current))
    old, new = deepcopy(before), deepcopy(after)
    old_choice, new_choice = old.pop("image_normalisation", None), new.pop("image_normalisation", None)
    same = _same_values(old, new) or only_added_storage_aliases(old, new, folder)
    if not same or old_choice == new_choice:
        raise ValueError("Setup inputs changed beyond the lighting choice; prepare a new current Setup plan")
    if (not isinstance(new_choice,dict)
            or series_key(new_choice.get("series_id")) != series_key(after["series_id"])
            or saved_choice(after, new_choice["series_id"], after["runs"]["ids"]) is None):
        raise ValueError("Save an explicit image normalisation choice before updating this Setup job")
    return before, after


def inspect_refresh(root, plan):
    """Prove a current GUI choice without modifying the job or source series."""
    bound = plan["series_manifest"]
    current = _raw(bound["path"])
    if _matches(current, bound):
        return None
    snapshot = Path(root) / "setup-controls" / "series-manifest.toml"
    if snapshot.exists():
        previous = _raw(snapshot)
        if not _matches(previous, bound):
            raise ValueError("The retained Setup series snapshot differs from its original identity")
    else:
        # Older jobs have no snapshot. The GUI's deterministic serializer can
        # recover an absent old choice only when the ORIGINAL byte SHA matches.
        # No plausible-but-different reconstruction is admitted.
        import toml
        document = tomllib.loads(current.decode("utf-8-sig"))
        document.pop("image_normalisation", None)
        encoded = toml.dumps(document)
        candidates = [bom + encoded.replace("\n", newline).encode()
                      for newline in ("\n", "\r\n") for bom in (b"", b"\xef\xbb\xbf")]
        matches = [value for value in candidates if _matches(value, bound)]
        if len(matches) != 1:
            raise ValueError("The original Setup series snapshot is unavailable; prepare a new current Setup plan")
        previous = matches[0]
    before, after = _normalisation_only(previous, current, Path(bound["path"]).parent)
    return dict(format=FORMAT, previous_bytes=previous, current_bytes=current,
        previous_document=before, current_document=after,
        current_series=dict(path=bound["path"], sha256=hashlib.sha256(current).hexdigest(), size_bytes=len(current)))


def original_manifest_roots(plan, refresh):
    """Replay only the exact captured old series metadata, never current pixels."""
    from .store_map import manifest_roots
    previous = refresh["previous_bytes"]
    if not _matches(previous, plan["series_manifest"]):
        raise ValueError("The immutable old series replay lost its original SHA identity")
    document = tomllib.loads(previous.decode("utf-8-sig"))
    if not _same(document, refresh["previous_document"]):
        raise ValueError("The original Setup series replay document changed")
    return manifest_roots(Path(plan["series_manifest"]["path"]).parent, document)


def same_plan_authority(previous, current):
    """Keep geometry, camera, protocols, images, runs and output authority exact."""
    before, after = deepcopy(previous), deepcopy(current)
    old_series, new_series = before.pop("series_manifest"), after.pop("series_manifest")
    if old_series["path"] != new_series["path"]:
        return False
    for document, series in ((before, old_series), (after, new_series)):
        for build in document["build_plans"]:
            context = build["request_template"]["series_context"]
            if not _same(context.get("manifest"), series):
                return False
            context.pop("manifest")
    return _same(before, after)


def validate_transition(previous_job, replacement_job, receipt, *, history_root):
    """Validate one immutable publication generation, not the latest job file."""
    ignored = {"commit", "environment_lock"}
    if receipt.get("image_preparation_refresh") is not None:
        return _validate_image_transition(previous_job, replacement_job, receipt, history_root)
    refresh = receipt.get("plan_refresh")
    if refresh is None:
        if not _same({k:v for k,v in previous_job.items() if k not in ignored},
                     {k:v for k,v in replacement_job.items() if k not in ignored}):
            raise ValueError("The Setup producer changed the original scientific bindings")
        return
    if refresh.get("format") != FORMAT:
        raise ValueError("Unknown Setup plan refresh proof")
    history = Path(history_root).resolve()
    def read(pin):
        path = Path(pin["path"])
        if not path.resolve().is_relative_to(history):
            raise ValueError("Setup refresh evidence must remain in this immutable history generation")
        raw = _raw(path) if path.suffix == ".toml" else path.read_bytes()
        if not _matches(raw, pin):
            raise ValueError("Setup refresh history bytes changed")
        return raw
    import json
    old_plan, new_plan = (json.loads(read(refresh[key]).decode("utf-8-sig"))
                          for key in ("previous_plan", "current_plan"))
    old_series, new_series = (read(refresh[key]) for key in ("previous_series", "current_series"))
    _normalisation_only(old_series, new_series, Path(new_plan["series_manifest"]["path"]).parent)
    if (not _matches(old_series, old_plan["series_manifest"])
            or not _matches(new_series, new_plan["series_manifest"])
            or not same_plan_authority(old_plan, new_plan)):
        raise ValueError("The refreshed Setup plan changed its original geometry or input authority")
    ignored |= {"setup_plan", "series", "series_manifest"}
    if not _same({k:v for k,v in previous_job.items() if k not in ignored},
                 {k:v for k,v in replacement_job.items() if k not in ignored}):
        raise ValueError("The refreshed Setup job changed its original execution scope")
    for old, new in ((previous_job, replacement_job),):
        for field, previous_pin, current_pin in (
                ("setup_plan", refresh["previous_plan"], refresh["current_plan"]),
                ("series", old_plan["series_manifest"], new_plan["series_manifest"]),
                ("series_manifest", old_plan["series_manifest"], new_plan["series_manifest"])):
            old_ref, new_ref = old[field], new[field]
            if (old_ref["sha256"] != previous_pin["sha256"] or old_ref["bytes"] != previous_pin["size_bytes"]
                    or new_ref["sha256"] != current_pin["sha256"] or new_ref["bytes"] != current_pin["size_bytes"]
                    or not _same({k:v for k,v in old_ref.items() if k not in {"sha256","bytes"}},
                                 {k:v for k,v in new_ref.items() if k not in {"sha256","bytes"}})):
                raise ValueError("Setup job references differ from the refreshed original controls")


def _validate_image_transition(previous, current, receipt, history_root):
    """Add new optical work only after preserving the exact completed geometry."""
    import json
    from .pipeline.series_setup_protocol import validate_setup_operation
    refresh = receipt["image_preparation_refresh"]
    if refresh.get("format") != IMAGE_FORMAT or receipt.get("plan_refresh") is not None:
        raise ValueError("Unknown image preparation upgrade proof")
    history = Path(history_root).resolve()
    def read(pin):
        path = Path(pin["path"])
        if not path.resolve().is_relative_to(history):
            raise ValueError("Image upgrade evidence must remain in its immutable history")
        raw = _raw(path) if path.suffix == ".toml" else path.read_bytes()
        if not _matches(raw, pin):
            raise ValueError("Image preparation upgrade evidence changed")
        return raw
    before, after = (read(refresh[key]) for key in ("previous_series", "current_series"))
    plan = json.loads(read(refresh["plan"]).decode("utf-8-sig"))
    if before != after:
        _normalisation_only(before, after, Path(plan["series_manifest"]["path"]).parent)
    if (not _matches(before, plan["series_manifest"])
            or previous["setup_plan"] != current["setup_plan"]
            or previous["setup_plan"]["sha256"] != refresh["plan"]["sha256"]):
        raise ValueError("Image preparation must retain the original geometry plan")
    config = json.loads(read(refresh["configuration"]).decode("utf-8-sig"))
    validate_setup_operation(config)
    from .image_normalisation import saved_choice, selected_corrections
    selected = tomllib.loads(after.decode("utf-8-sig"))
    choice = saved_choice(selected, selected["series_id"], plan["runs"])
    if (selected.get("method", {}).get("active") != "phase_model"
            or choice is None or "color_method" not in choice
            or selected_corrections(choice)["color_method"] == "saved_native"):
        raise ValueError("Image upgrade requires an explicit current independent image choice")
    from .pipeline.setup_image_preparation import image_output_scope
    if (any(config.get(key) != plan[key] for key in ("project_root", "project_manifest",
            "adopted_revision", "native_parameters", "target_machine"))
            or config.get("output_scope") != image_output_scope(plan)
            or Path(config["job_root"]).resolve() != Path(receipt["original_plan"]["path"]).parent.parent.resolve()):
        raise ValueError("Image upgrade changed original geometry, native parameters or output authority")
    if (config.get("operation") != "image-preparation"
            or config.get("run") != "setup-image-preparation"
            or config.get("setup_plan") != receipt["original_plan"]
            or config.get("runs") != plan["runs"]
            or not _matches(after, config.get("choice_authority", {}))
            or config["choice_authority"]["path"] != plan["series_manifest"]["path"]):
        raise ValueError("Image preparation differs from its actual new choice authority")
    ignored = {"commit", "environment_lock", "expected_runs", "setup_operations", "image_preparation"}
    if not _same({k:v for k,v in previous.items() if k not in ignored},
                 {k:v for k,v in current.items() if k not in ignored}):
        raise ValueError("Image upgrade changed existing Setup scientific bindings")
    binding = current.get("image_preparation", {})
    config_pin = binding.get("configuration", {})
    if (config_pin.get("sha256") != refresh["configuration"]["sha256"]
            or config_pin.get("size_bytes") != refresh["configuration"]["size_bytes"]
            or binding.get("choice_authority") != config["choice_authority"]):
        raise ValueError("Image upgrade job differs from its new typed operation")
    expected = ["setup-apply", *plan["runs"]]
    operations = [dict(run="setup-apply", operation="apply"),
                  *[dict(run=run, operation="build") for run in plan["runs"]]]
    previous_image = previous.get("image_preparation")
    if previous_image is not None:
        old_config_raw = read(refresh["previous_configuration"])
        old_pin = previous_image["configuration"]
        if not _matches(old_config_raw, old_pin):
            raise ValueError("Image retry lost the original operation configuration")
        reuse = config.get("reuse_image_basis")
        if reuse is not None:
            if refresh.get("reused_image_basis") != reuse:
                raise ValueError("Image reuse differs from its original completed basis")
            if refresh.get("retired_partial_reuse") is not None:
                from .setup_image_partial_reuse import FORMAT as PARTIAL_FORMAT
                partial = json.loads(read(refresh["retired_partial_reuse"]).decode("utf-8-sig"))
                failure = json.loads(read(partial["failed_attempt"]).decode("utf-8-sig"))
                retirement = json.loads(read(partial["retirement"]["retirement"]).decode("utf-8-sig"))
                if (refresh["retired_partial_reuse"] != reuse or partial.get("format") != PARTIAL_FORMAT
                        or partial.get("complete") is not False or not failure.get("failed_utc")
                        or failure.get("run") != "setup-image-preparation"
                        or failure.get("task", {}).get("job_id") != previous["job_id"]
                        or failure.get("task", {}).get("configuration", {}).get("sha256") != old_pin["sha256"]
                        or retirement.get("attempt_id") != failure.get("attempt_id")
                        or any(retirement.get(key) is not True for key in
                               ("complete", "all_absent", "worker_os_references_closed"))):
                    raise ValueError("Image sample reuse lacks its actual failed and retired attempt")
            else:
                old_done = json.loads(read(refresh["previous_image_completion"]).decode("utf-8-sig"))
                old_result = json.loads(read(refresh["previous_image_result"]).decode("utf-8-sig"))
                if (old_result.get("operation") != "image-preparation"
                        or old_result.get("complete") is not True
                        or old_result.get("image_preparation") != reuse
                        or old_result.get("task", {}).get("job_id") != previous["job_id"]
                        or old_done.get("attempt_id") != old_result.get("attempt_id")):
                    raise ValueError("Image reuse lacks its actual prior Setup completion")
        expected = [*expected, "setup-image-preparation"]
        operations = [*operations, dict(run="setup-image-preparation", operation="image-preparation")]
    current_expected = ["setup-apply", *plan["runs"], "setup-image-preparation"]
    current_operations = [dict(run="setup-apply", operation="apply"),
        *[dict(run=run, operation="build") for run in plan["runs"]],
        dict(run="setup-image-preparation", operation="image-preparation")]
    if (previous["expected_runs"] != expected or previous.get("setup_operations") != operations
            or current["expected_runs"] != current_expected
            or current.get("setup_operations") != current_operations):
        raise ValueError("Image upgrade may only append one typed preparation operation")
    if (receipt.get("retained_setup_apply", {}).get("run") != "setup-apply"
            or {row["run"] for row in receipt.get("retained_completed_builds", [])} != set(plan["runs"])
            or len(receipt.get("retained_completed_builds", [])) != len(plan["runs"])
            or receipt.get("retained_apply") is None):
        raise ValueError("Image upgrade requires every original completed Apply and Build")
