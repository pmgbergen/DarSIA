"""Bind saved optical work from a genuinely failed, retired Setup attempt."""
from copy import deepcopy
from pathlib import Path

from .pipeline.job_queue import descriptor, read_json


FORMAT = "darsia-retired-setup-colour-path-partials-v1"


def review_partial_reuse(root, review, config):
    """Inspect named metadata only; original numerical samples stay untouched."""
    from .native.full_photometry import ORDER
    from .native.raw_rig_source import execution_identity
    from .pipeline.setup_image_preparation import independent_choice, completed_build_operations
    from .pipeline.setup_colour_paths import preflight_setup_colour_paths
    from .mass.n1_pipeline import DataReader

    root = Path(root)
    partials = review.get("image_partials", [])
    if len(partials) != 1 or not (root / "failed/setup-image-preparation.json").is_file():
        return None
    source = Path(partials[0]["source"])
    record_path = source / "photometry/independent_photometry.json"
    prior_reuse = None
    reader = DataReader()
    if not record_path.is_file():
        prior_reuse = config.get("reuse_image_basis")
        if prior_reuse is None:
            return None  # Interrupted before all independent fits were saved.
        previous = reader.json(prior_reuse)
        if (previous.get("format") != FORMAT or previous.get("complete") is not False
                or previous.get("choice_authority") != config["choice_authority"]):
            return None
        record_path = Path(previous["photometry_record"]["path"])
        record_pin = descriptor(record_path)
        if record_pin != previous["photometry_record"]:
            raise ValueError("Previously retained photometry changed before this retry")
    if config["choice_authority"] != descriptor(config["choice_authority"]["path"]):
        return None
    record_pin = descriptor(record_path)
    record = reader.json(record_pin)
    choice_pin = descriptor(record_path.parent / "image_normalisation_choice.json")
    original_choice = reader.json(choice_pin)
    _series, choice = independent_choice(config["choice_authority"])
    if (original_choice.get("choice") != choice
            or original_choice.get("source_series") != config["choice_authority"]):
        return None  # A new user choice needs its own correction fits.
    operations = completed_build_operations(root, config["runs"])
    if set(record.get("runs", {})) != set(operations):
        return None
    # Local result metadata is relocated by this transaction. Such a first
    # generation needs new fits; older retained Builds keep stable paths.
    if any(Path(pin["path"]).resolve().is_relative_to((root / "results").resolve())
           for pin in operations.values()):
        return None
    identity = execution_identity()
    for run, row in record["runs"].items():
        binding = reader.json(row["source_binding"])
        model = reader.json(row["fit"])
        if (binding.get("execution_identity") != identity or binding.get("correction_order") != ORDER
                or binding.get("normalisation", {}).get("choice") != choice):
            return None
        if model.get("source", {}).get("operation_result") != operations[run]:
            return None
    optical = preflight_setup_colour_paths(config["choice_authority"], config["runs"])
    document = dict(format=FORMAT, complete=False, source_root=str(source),
        file_versions=deepcopy(partials[0]["snapshot"]["files"]),
        operations=operations, image_basis=dict(runs=deepcopy(record["runs"])),
        photometry_record=record_pin, choice=choice,
        choice_authority=deepcopy(config["choice_authority"]),
        raw_execution_identity=identity, correction_order=ORDER, **optical)
    if prior_reuse is not None:
        if any(previous.get(key) != document[key] for key in
               ("operations", "choice", "raw_execution_identity", "correction_order")):
            return None
        document["prior_reuse"] = deepcopy(prior_reuse)
    return document


def bind_archived_failure(document, review, archived, archived_attempts):
    """Use actual archived evidence; never promote failed work to completion."""
    from .pipeline.job_queue import write_immutable
    document = deepcopy(document)
    proofs = review["retirement"]
    proof = proofs[0] if isinstance(proofs, list) else proofs
    attempt = review["attempt_id"]
    old_attempt, saved_attempt = next((old, saved) for old, saved in archived_attempts
                                     if old.name == attempt)
    root = Path(review["claim"]).parents[2]
    def relocated(pin):
        original = Path(pin["path"])
        if original.is_relative_to(old_attempt):
            path = saved_attempt / original.relative_to(old_attempt)
        else:
            path = Path(archived) / original.relative_to(root)
        actual = descriptor(path)
        if any(actual[key] != pin[key] for key in ("sha256", "size_bytes")):
            raise ValueError("Retired image evidence changed during its metadata archive")
        return actual
    document["retirement"] = {key: relocated(proof[key]) for key in ("claim", "owner", "retirement")}
    document["retirement"]["stop"] = proof["stop"]
    document["failed_attempt"] = descriptor(Path(archived) / "failed/setup-image-preparation.json")
    failure = read_json(document["failed_attempt"]["path"])
    if failure.get("attempt_id") != attempt or not failure.get("failed_utc"):
        raise ValueError("Partial reuse lost its original failed attempt")
    path = Path(archived).parent / "sources/retired-colour-path-partials.json"
    write_immutable(path, document)
    return descriptor(path)
