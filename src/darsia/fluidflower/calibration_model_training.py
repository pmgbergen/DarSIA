"""Launch a prepared RAW refit through the retained native Studio lifecycle."""
from pathlib import Path


def require_prepared_normalisation(request, series_root):
    """Admit the new producer's actual saved choice, not historical V5 replay.

    This only applies to fresh RAW requests whose prepared source bindings
    prove full-photograph correction before the crop. Historical routes keep
    their original numerical-readiness guard.
    """
    import tomllib
    from .image_normalisation import saved_choice
    from .phase_saved_assets import VerifiedAssetReader
    from .phase_training.raw_review_training import FAMILY
    from .native.full_photometry import ORDER
    if request.get("settings", {}).get("training_family") != FAMILY or series_root is None:
        raise ValueError("Saved normalisation is required before launch")
    document = tomllib.loads((Path(series_root) / "series.toml").read_text(encoding="utf-8-sig"))
    choice = saved_choice(document, document["series_id"], document["runs"]["ids"])
    if choice is None or not choice.get("decision"):
        raise ValueError("Save the image-normalisation choice before refitting")
    basis = request["settings"].get("raw_image_basis") or {}
    if not basis.get("runs"):
        raise ValueError("The fresh request has no prepared image basis")
    reader = VerifiedAssetReader()
    for row in basis["runs"]:
        binding = reader.json(row["raw_binding"])
        normalisation = binding.get("normalisation") or {}
        if binding.get("correction_order") != ORDER or normalisation.get("choice") != choice:
            raise ValueError("The prepared training photographs use another saved image-normalisation choice")
        record = reader.json(normalisation["record"])
        if record.get("format") != "darsia-prepared-image-normalisation-v1" or record.get("choice") != choice:
            raise ValueError("The prepared image-normalisation record differs from its source binding")
    return choice


def launch_prepared_training(steps, result, *, name=None):
    """Use original review admission, naming, QProcess and completed-model signals.

    A verified coordinate transfer is derivative evidence. Its original source
    bundles admit the saved review session; its projected bundles are the exact
    frozen inputs consumed by the numerical request.
    """
    from .phase_training.raw_review_training import FAMILY
    from .phase_workflow import build_phase_training_command
    request = result["request"]
    if request.get("settings", {}).get("training_family") != FAMILY:
        raise ValueError("Calibration refitting needs a prepared RAW training request")
    studio = steps.studio
    if studio.training_process is not None:
        raise ValueError("A phase-model training process is already running")
    bundles = result["bundles"]
    source_bundles = result.get("source_bundles") or bundles
    text = lambda pairs: "\n".join(str(a) + " | " + str(b) for a, b in pairs)
    studio.bundles.setPlainText(text(source_bundles))
    steps.model_name.setText(name or request["model_id"])
    def launch():
        studio.bundles.setPlainText(text(bundles))
        studio._launch_training_job(request, build_phase_training_command(request),
            Path(request["request_path"]).parent,
            "Fitting the selected phase-model architecture on the current corrected and cropped images...")
    steps.phase_workflow.run_training(launch, saved_labels_only=True)
    if studio.training_process is None:
        raise ValueError("The prepared phase-model process did not start")
    return studio.training_process
