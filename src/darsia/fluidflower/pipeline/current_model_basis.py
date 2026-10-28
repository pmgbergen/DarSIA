"""The selected trained classifier and its producer-recorded image basis.

This is provenance validation, not a fabricated model approval. The fresh
trainer's actual heads, policy, photometry and source grid must agree before
the current Build runner can consume that model.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

FORMAT = "darsia-current-phase-model-v1"
BASIS_FORMAT = "darsia-trained-image-basis-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()


def _model_root(root, reference):
    from ..storage_paths import resolve_storage_path
    from ..phase_model_library import FORMAT as LIBRARY_FORMAT, resolve_reference
    path = Path(reference)
    if not path.is_absolute():
        path = Path(root) / path
    path = resolve_storage_path(path, series_roots=(Path(root),))
    if path.is_dir():
        return path
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if value.get("format") == LIBRARY_FORMAT:
        return resolve_reference(path, series_root=root)
    if value.get("model_root"):
        return resolve_storage_path(value["model_root"], series_roots=(Path(root),))
    if path.name in {"summary.json", "training_request.json"}:
        return path.parent
    raise ValueError("The selected model reference does not identify trained model files")

UNBOUND_MODEL_REASON = (
    "The selected phase model is not yet bound to this calibration's prepared image basis. "
    "Finish crop and image adjustments, then train the selected model on the prepared photographs in Phase model."
)


def _choice_basis(choice):
    """The pixel-producing choices, excluding the save's actor/time and labels."""
    if choice is None:
        return None
    from ..image_normalisation import selected_corrections
    corrections = selected_corrections(choice)
    enabled = corrections["lighting"] or corrections["color_method"] != "off"
    basis = dict(format=choice.get("format"), series_id=choice.get("series_id"),
        runs=choice.get("runs"), corrections=corrections,
        reference_run=choice.get("reference_run") if enabled else None,
        stable_area=choice.get("stable_area") if corrections["lighting"] else None)
    if corrections.get("light_field"):
        # Only sand-type equalisation adds these, so older bases keep their value.
        basis.update(exclusion_band=choice.get("exclusion_band"), light_field=choice.get("light_field"))
    return basis


def _spatial_basis(certificate):
    """Retain only the renderer's actual grid; no protocol/date/region fields."""
    renderer = certificate["renderer"]
    grid = renderer["source_grid"]
    metadata = grid["metadata"]
    rasters = {Path(value["path"]).name: value["sha256"]
               for value in certificate.get("outputs", [])
               if Path(value["path"]).name in {"labels.npz", "facies.npz"}}
    return dict(shape=renderer["native_shape_hw"], profile=renderer["profile_id"],
        camera_u_wh=renderer["camera_u_wh"],
        source_rasters=rasters,
        shape_corrections=[dict(sha256=value["sha256"], size_bytes=value.get("size_bytes"))
                           for value in renderer["shape_corrections"]],
        num_voxels=grid["num_voxels"], voxel_size=grid["voxel_size"],
        physical={key: metadata[key] for key in ("space_dim", "indexing", "dimensions", "origin")})


def _photometry_basis(binding, document):
    """Actual independent fits and array identities, without reading arrays."""
    from ..native.full_photometry import INDEPENDENT
    if binding.get("photometric_method") != INDEPENDENT:
        raise ValueError(UNBOUND_MODEL_REASON)
    result = dict(method=INDEPENDENT, order=binding["correction_order"], models={})
    for role in ("fit", "reference"):
        model = binding["models"][role]
        sidecar = document(model["sidecar"])
        if (sidecar.get("format") != INDEPENDENT
                or sidecar["data"]["sha256"] != model["data"]["sha256"]):
            raise ValueError("The selected photometric model and sidecar have different data")
        result["models"][role] = {key: sidecar[key] for key in (
            "native_shape_hw", "native_to_camera_u_h", "baseline_chart_roi_rc",
            "baseline_color_fit", "baseline_light_fit", "correction_order", "color_interface")}
        result["models"][role]["data_sha256"] = model["data"]["sha256"]
        # These are the producer's actual sampling rasters, not its full
        # geometry revision or material properties (porosity/permeability).
        support = sidecar.get("stable_support") or {}
        result["models"][role]["source_rasters"] = {
            key: support[key]["sha256"] for key in ("source_labels", "source_facies") if key in support}
    return result


def _require_prepared_basis(root, basis, selection, reader, build_snapshot):
    """Compare saved choices and selected Build metadata, never decoded images."""
    import tomllib
    import numpy as np
    from ..image_normalisation import saved_choice
    from ..mass.n1_pipeline import asset
    from .current_phase_recipe import require_runtime_configuration

    documents = {}
    def document(pin):
        key = pin["path"], pin["sha256"]
        if key not in documents:
            documents[key] = reader.json(pin)
        return documents[key]

    authority = (selection or {}).get("input_authority", {}).get("current_series")
    authority = authority or asset(Path(root) / "series.toml")
    manifest = tomllib.loads(reader.path(authority).read_text(encoding="utf-8-sig"))
    choice = (saved_choice(manifest, manifest["series_id"], manifest["runs"]["ids"])
              if manifest.get("image_normalisation") is not None else None)
    native = manifest.get("native")
    current_native = None
    if native is not None:
        path = Path(native["configuration"])
        current_native = dict(path=str(path if path.is_absolute() else Path(root) / path), sha256=native["sha256"])
        reader.path(current_native)
        if current_native["sha256"] != basis["native_configuration"]["sha256"]:
            try:
                require_runtime_configuration(current_native, basis["native_configuration"], reader=reader)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(UNBOUND_MODEL_REASON) from error

    selected = {row["run"] for row in selection["runs"]} if selection is not None else None
    operations = None
    prepared_images = None
    if build_snapshot is not None:
        operations = {row["run"]: row["result"]["files"]["operation_result.json"]
                      for row in build_snapshot.get("runs", [])
                      if row.get("operation") == "build" and row.get("state") == "Complete"}
        from .setup_image_preparation import OPERATION, FORMAT
        completed = [row for row in build_snapshot.get("runs", [])
                     if row.get("operation") == OPERATION and row.get("state") == "Complete"]
        if completed:
            if len(completed) != 1:
                raise ValueError(UNBOUND_MODEL_REASON)
            result = completed[0]["result"]
            pin = result.get("image_preparation")
            if pin is None:
                pin = document(result["files"]["operation_result.json"])["image_preparation"]
            prepared_images = document(pin)
            if (prepared_images.get("format") != FORMAT or prepared_images.get("complete") is not True
                    or prepared_images.get("operations") != operations
                    or _choice_basis(prepared_images.get("choice")) != _choice_basis(choice)):
                raise ValueError(UNBOUND_MODEL_REASON)
        elif choice is not None and "color_method" in choice and _choice_basis(choice)["corrections"]["color_method"] != "saved_native":
            raise ValueError(UNBOUND_MODEL_REASON)
    for row in basis["runs"]:
        if selected is not None and row["run"] not in selected:
            continue
        # Historical shape-only declarations cannot stand in for prepared RAW
        # source metadata. Their old models and original annotations stay intact.
        if not all(isinstance(row.get(key), dict) for key in ("raw_binding", "source_manifest", "native_configuration")):
            raise ValueError(UNBOUND_MODEL_REASON)
        bound = document(row["raw_binding"])
        if (row["native_configuration"].get("sha256") != basis["native_configuration"]["sha256"]
                or bound.get("run") != row["run"]
                or _choice_basis(bound.get("normalisation", {}).get("choice")) != _choice_basis(choice)):
            raise ValueError(UNBOUND_MODEL_REASON)
        if operations is None:
            continue
        if row["run"] not in operations:
            raise ValueError(UNBOUND_MODEL_REASON)
        source = document(row["source_manifest"])
        preparation = source.get("current_build_preparation")
        if not isinstance(preparation, dict):
            raise ValueError(UNBOUND_MODEL_REASON)
        retained = document(preparation)
        if not isinstance(retained.get("operation_result"), dict):
            raise ValueError(UNBOUND_MODEL_REASON)
        previous = document(retained["operation_result"])
        current = document(operations[row["run"]])
        try:
            old_certificate = document(previous["baseline_renderer"])
            certificate = document(current["baseline_renderer"])
            renderer = certificate["renderer"]
            before_h = np.asarray(old_certificate["renderer"]["native_to_camera_u_h"], float)
            after_h = np.asarray(renderer["native_to_camera_u_h"], float)
            coherent = (source["run"] == retained["run"] == current["run"] == row["run"]
                and retained["raw_binding"]["sha256"] == row["raw_binding"]["sha256"]
                and source["raw_binding"]["sha256"] == row["raw_binding"]["sha256"]
                and source["native_configuration"]["sha256"] == basis["native_configuration"]["sha256"]
                and current["operation"] == "build" and current["complete"] is True
                and bound["baseline_raw"]["sha256"] == certificate["inputs"]["raw"]["sha256"]
                and bound["baseline_raw"]["sha256"] == old_certificate["inputs"]["raw"]["sha256"]
                and (current_native is None or certificate["inputs"]["profile"]["sha256"] == current_native["sha256"])
                and row["native_shape_hw"] == bound["native_shape_hw"] == renderer["native_shape_hw"]
                and _spatial_basis(old_certificate) == _spatial_basis(certificate)
                and before_h.shape == after_h.shape == (3, 3)
                and np.isfinite(before_h).all() and np.isfinite(after_h).all()
                and np.allclose(before_h, after_h, rtol=0, atol=1e-10))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(UNBOUND_MODEL_REASON) from error
        if not coherent:
            raise ValueError(UNBOUND_MODEL_REASON)
        if prepared_images is not None:
            try:
                current_binding = document(prepared_images["runs"][row["run"]]["source_binding"])
                same_photometry = (_photometry_basis(bound, document)
                                   == _photometry_basis(current_binding, document))
            except (KeyError, TypeError) as error:
                raise ValueError(UNBOUND_MODEL_REASON) from error
            if not same_photometry:
                raise ValueError(UNBOUND_MODEL_REASON)


def require_current_model_basis(root, selection=None, *, reader=None, model_reference=None,
                                build_snapshot=None):
    """Bind actual saved fresh heads and the image basis their trainer consumed."""
    from ..series_method import read_series_method
    from ..mass.n1_pipeline import asset
    from .current_inputs import current_input_reader
    from ..phase_training.model import DualPhaseSignalModel
    from ..phase_training.raw_frame_context import TEMPORAL_CONTRACT

    method = read_series_method(root)
    reference = method.model_version if model_reference is None else model_reference
    if method.active != "phase_model" or not reference:
        raise ValueError("Choose and approve a phase model for the final prepared images first.")
    reader = current_input_reader(reader)
    try:
        folder = _model_root(root, reference)
    except FileNotFoundError as error:
        raise ValueError(UNBOUND_MODEL_REASON) from error
    basis_path = folder / "image_basis.json"
    if not basis_path.is_file():
        raise ValueError(UNBOUND_MODEL_REASON)
    basis_pin = asset(basis_path)
    basis = reader.json(basis_pin)
    if (basis.get("format") != BASIS_FORMAT or not isinstance(basis.get("runs"), list)
            or not basis["runs"] or not isinstance(basis.get("native_configuration"), dict)):
        raise ValueError(UNBOUND_MODEL_REASON)
    if basis.get("input_photometry_sha256") != _digest(basis.get("input_photometry")):
        raise ValueError("The trained model has no coherent producer-recorded image basis")
    if (basis.get("encoding") != "float32-camera-linear-unit-rgb"
            or basis.get("temporal_contract") != TEMPORAL_CONTRACT):
        raise ValueError("The selected model uses another scientific encoding or acquisition-history contract")
    reader.path(basis["native_configuration"])
    summary_pin = asset(folder / "summary.json")
    if reader.json(summary_pin).get("image_basis") != basis_pin:
        raise ValueError("The training summary does not bind this actual image basis")
    policy_pin = asset(folder / "phase_decision_policy.json")
    policy = reader.json(policy_pin)["models"]["generic"]
    heads = {}
    for role in ("co2_presence", "gas_within_co2"):
        pin = asset(folder / "models" / "generic" / (role + "_signal.json"))
        document = reader.json(pin)
        model = DualPhaseSignalModel.from_dict(document)
        provenance = model.training_provenance
        if (provenance.get("image_basis") != basis_pin
                or provenance.get("input_photometry") != basis["input_photometry"]
                or policy[role].get("model_sha256") != document["model_sha256"]):
            raise ValueError("The selected classifier or decision policy belongs to another image basis")
        threshold = policy[role].get("presence_threshold")
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not 0 <= threshold <= 1:
            raise ValueError("The selected classifier has no finite probability decision threshold")
        heads[role] = pin
    value = dict(format=FORMAT, family="fresh-native-logistic", image_basis=basis_pin,
        native_configuration=deepcopy(basis["native_configuration"]), heads=heads,
        decision_policy=policy_pin, training_summary=summary_pin, model_root=str(folder))
    if selection is not None:
        required = {row["run"] for row in selection["runs"]}
        covered = [row["run"] for row in basis["runs"]]
        if len(set(covered)) != len(covered) or not required <= set(covered):
            raise ValueError("The trained image basis does not cover every selected calibration run")
    _require_prepared_basis(root, basis, selection, reader, build_snapshot)
    return value
