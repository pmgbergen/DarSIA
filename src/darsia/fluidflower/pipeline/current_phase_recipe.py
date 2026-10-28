"""Project the registered frozen phase method onto a current Build run.

Only original model/parameter and source-support roles are retained here.
Current per-acquisition geometry and RAW transforms are produced separately;
historical phase labels and historical physical fields are never inputs.
"""
from copy import deepcopy
from pathlib import Path

from ..mass.n1_pipeline import DataReader, asset
from ..native.source import _path


def _runtime_parameters(binding, reader):
    """Apply declared storage relocations to alias targets, keeping source bytes."""
    native = deepcopy(reader.json(binding))
    aliases = native["source"].get("asset_aliases", {})
    updated = {}
    for original, target in aliases.items():
        actual = str(reader.resolve_path(target))
        updated[original] = actual
        updated[target] = actual
    native["source"]["asset_aliases"] = updated
    return native


def require_runtime_configuration(original, current, *, reader):
    """Allow an explicit current executor binding without changing parameter DATA."""
    if original == current:
        return
    from .current_inputs import current_input_reader
    from ..native.raw_rig_source import execution_identity
    reader = current_input_reader(reader)
    old, new = _runtime_parameters(original, reader), _runtime_parameters(current, reader)
    if new["source"].get("raw_execution_identity") != execution_identity():
        raise ValueError("The new RAW parameter source does not bind the current executor")
    old["source"].pop("raw_execution_identity", None)
    new["source"].pop("raw_execution_identity", None)
    # Storage relocations (asset aliases) are not parameter data: every asset
    # stays SHA-bound. A relocation both declare must still name one target.
    old_aliases = old["source"].pop("asset_aliases", {})
    new_aliases = new["source"].pop("asset_aliases", {})
    if any(old_aliases[key] != new_aliases[key] for key in set(old_aliases) & set(new_aliases)):
        raise ValueError("Current RAW parameters relocate a saved Build asset elsewhere")
    import json
    if json.dumps(old, sort_keys=True, allow_nan=False) != json.dumps(new, sort_keys=True, allow_nan=False):
        raise ValueError("Current RAW parameters change the saved Build's numerical or geometry data")


def retained_phase_recipe(native_binding, plan_binding, run, *, reader=None, selected_model=None, source_only=False):
    """Read original declared source roles for one run, without decoding pixels."""
    from .current_inputs import current_input_reader
    reader = current_input_reader(reader)
    native = _runtime_parameters(native_binding, reader)

    def relocated(bound):
        result = deepcopy(bound)
        result["path"] = str(reader.resolve_path(_path(bound["path"], parameters=native)))
        result["path"] = str(reader.path(result))
        return result

    def document(bound):
        return reader.json(relocated(bound))

    def descriptors(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys():
                bound = deepcopy(value)
                bound["path"] = str(reader.resolve_path(_path(value["path"], parameters=native)))
                # Payloads remain unopened until their original consumer runs.
                return bound
            return {key: descriptors(item) for key, item in value.items()}
        if isinstance(value, list):
            return [descriptors(item) for item in value]
        return deepcopy(value)

    plan = document(plan_binding)
    selected = plan["runs"][run]
    source_manifest = document(selected["source_manifest"])
    if source_manifest["run"] != run or not source_manifest["frame_sources"]:
        raise ValueError("The original method has no selected source declaration for this run")
    # This is the original producer's fixed output layout. Its own source/run
    # identity is checked; only invariant recipe roles are extracted below.
    original = source_manifest["frame_sources"][0]
    receipt_binding = asset(Path(relocated(plan_binding)["path"]).parent /
        "selected_phase_frames" / run / (original["source_sha256"] + ".json"))
    receipt = reader.json(receipt_binding)
    if (receipt["run"] != run or receipt["source_sha256"] != original["source_sha256"]
            or receipt["elapsed_h"] != original["elapsed_h"]):
        raise ValueError("The retained phase recipe belongs to another original acquisition")
    inherited = receipt["qa"]["native_inference"]["historical_stage_conventions"]
    native_manifest = document(selected["parent_native_manifest"])
    marker = native_manifest["v5_track1_photo_geometry"]
    first = marker["frames"][0]
    photo = document(dict(path=first["receipt_path"], sha256=first["receipt_sha256"]))
    if photo["phase_labels_used_for_mask"] is not False:
        raise ValueError("The original photo support was derived from phase labels")
    photo_mask = relocated(dict(path=photo["photo_sand_reference_mask_path"],
        sha256=photo["photo_sand_reference_mask_sha256"]))
    photo_mask.setdefault("size_bytes", Path(photo_mask["path"]).stat().st_size)
    barrier = document(plan["frozen_barrier_reference"]["geometry_manifest"])
    static = dict(components=relocated(plan["frozen_barrier_reference"]["components"]),
                  source_ids=deepcopy(barrier["source_component_ids"]), unchanged_source=False)
    static["components"].setdefault("size_bytes", Path(static["components"]["path"]).stat().st_size)
    legacy = selected_model is None and not source_only
    models = {}
    if legacy:
        for role, requirement in (("native", "native_centre"), ("edge", "rig_edge")):
            original_model = native["model_bundles"][role]
            models[role] = dict(documents=descriptors(original_model["documents"]),
                requirements=deepcopy(native["parameters"]["model_requirements"][requirement]),
                adapter=original_model["original_model"]["adapter"])
        for head in ("co2", "g3"):
            if models["native"]["documents"][head]["sha256"] != plan["heads"][head]["sha256"]:
                raise ValueError("The registered model differs from the retained phase method: " + head)
        placement = native["model_bundles"]["placement"]
        models["placement"] = {key: descriptors(placement["documents"][key])
                               for key in ("pixel_model", "regional_model")}
        if placement["projected_models_active"]:
            models["placement"]["projected_models"] = dict(
                pixel=descriptors(placement["projected_pixel_model"]),
                regional=descriptors(placement["projected_regional_model"]))
    envelope = None
    if legacy:
        envelope_plan = document(native["chain_assets"]["parent_chain"]["final2_plan"])
        envelope_binding = asset(reader.resolve_path(_path(envelope_plan["output_root"], parameters=native)) /
                                 "envelopes" / run / "receipt.json")
        envelope = reader.json(envelope_binding)
        if (envelope["run"] != run or envelope["contract"] != envelope_plan["contract"]
                or envelope["binding"]["floor_curve"]["sha256"] != native["chain_assets"]["floor_curve"]["sha256"]):
            raise ValueError("The original gas envelopes differ from this run/floor method")
    phase = dict(parameters=deepcopy(native["parameters"]), model_bundles=models,
        geometry_parameters=deepcopy(native["geometry_parameters"]),
        barrier_parameters=deepcopy(native["barrier_parameters"]),
        geometry_fields=dict(reference_to_frame=native["source"]["symbols"]["reference_to_frame_h_key"],
            reference_to_metric=native["geometry_parameters"]["fields"]["reference_metric_field"]),
        chain_assets=dict(floor=descriptors(native["chain_assets"]["floor_curve"]),
            zones=descriptors(native["chain_assets"]["zones"])),
        phase_shape_hw=deepcopy(plan["phase_grid"]["shape_hw"]))
    if envelope is not None:
        phase["chain_assets"].update(envelopes=envelope_binding, envelope_contract=envelope["contract"])
    if source_only:
        phase["model_bundles"] = dict(family="source-only-no-inference")
    if selected_model is not None:
        # The current classifier is its own model family. None of the inherited
        # native/edge/placement heads is passed to an inference consumer.
        phase["model_bundles"] = dict(family="fresh-native-logistic", selected=deepcopy(selected_model))
    grid = plan["phase_grid"]
    geometry = phase["parameters"]["geometry"]
    if (grid["native_origin_xy"] != geometry["rig_origin_xy"]
            or grid["source_native_shape_hw"] != geometry["native_shape_hw"]
            or grid["source_content_box_xywh"] != geometry["source_box_xywh"]
            or envelope is not None and any(envelope[key]["shape_hw"] != phase["phase_shape_hw"] for key in ("envelope1", "envelope2"))):
        raise ValueError("The original source, envelope and phase canvases differ")
    placement_metadata = {key: deepcopy(receipt["qa"]["p1"][key]) for key in (
        "contract", "spec_sha256", "run", "light_epoch_id", "light_epoch_table", "cpu_only",
        "fitting_performed", "training_labels_read", "validation_annotations_read", "floor_rule_applied",
        "frozen_co2_head_sha256", "co2_inference_performed")}
    if placement_metadata["light_epoch_id"] != plan["light_epochs"]["run_to_epoch"][run]:
        raise ValueError("The retained method has inconsistent light epochs")
    roles = dict(static=static, retained_source_cache=None, encoding_contract=None,
        light_epoch_id=placement_metadata["light_epoch_id"], placement_metadata=placement_metadata,
        inherited_spacing_m=deepcopy(inherited["inherited_map"]["spacing_m"]),
        probability_spacing_m=deepcopy(inherited["g3_probability"]["spacing_m"]),
        photo_sand_reference=photo_mask,
        photo_proof=dict(sha256=photo_mask["sha256"], phase_labels_used=False,
                         original_photo_receipt=relocated(dict(path=first["receipt_path"], sha256=first["receipt_sha256"]))))
    temporal = receipt["qa"]["native_inference"]["input_provenance"]["baseline_temporal"]["policy"]
    context_keys = ("photo_certified_sand_mask_path", "photo_certified_sand_mask_sha256",
        "photo_sand_qa_pass", "photo_sand_qa_scope", "photo_sand_reference_mask_path",
        "photo_sand_reference_mask_sha256", "photo_sand_reference_receipt_sha256", "enclosed_hole_policy")
    training_context = {key: deepcopy(photo[key]) for key in context_keys if key in photo}
    for key in training_context:
        if key.endswith("_path"):
            training_context[key] = str(reader.resolve_path(_path(training_context[key], parameters=native)))
    return dict(phase=phase, phase_roles=roles, temporal_pad_px=temporal["pad_px"],
        native_configuration=deepcopy(native_binding), original_plan=deepcopy(plan_binding),
        original_recipe_receipt=receipt_binding,
        original_photo=photo, training_context=training_context,
        original_native_manifest=relocated(selected["parent_native_manifest"]))


def build_photometry_binding(context, recipe, output, *, reader=None, image_choice_authority=None,
                             prepared_photometry=None):
    """Bind frozen photometry to the actual Build's saved RAW coordinate basis.

    This assembles metadata. The unchanged RAW producer verifies and consumes
    the bound model arrays and the acquisition after the user starts the job.
    """
    import numpy as np
    from ..mass.n1_pipeline import _write_json
    from ..native import source, raw_rig_source
    from .current_inputs import current_input_reader
    reader = current_input_reader(reader)
    choice = None
    if image_choice_authority is not None:
        import tomllib
        from ..image_normalisation import saved_choice
        series = tomllib.loads(reader.path(image_choice_authority).read_text(encoding="utf-8-sig"))
        choice = saved_choice(series, series["series_id"], series["runs"]["ids"])
        if choice is None:
            raise ValueError("Save the series image-normalisation choice before preparing training images")
    if prepared_photometry is not None:
        if "source_binding" in prepared_photometry:
            prepared_photometry = prepared_photometry["source_binding"]
        value = reader.json(prepared_photometry)
        from ..native.full_photometry import INDEPENDENT
        if (value.get("photometric_method") != INDEPENDENT or value.get("run") != context.run
                or value["baseline_raw"]["sha256"] != context.provenance["raw_acquisition"]["sha256"]
                or value["native_shape_hw"] != list(context.grid["num_voxels"])
                or value["normalisation"]["choice"] != choice):
            raise ValueError("The prepared independent photometry differs from this Build or image choice")
        return deepcopy(prepared_photometry)
    if choice is not None:
        from ..image_normalisation import selected_corrections
        selected_corrections_value = selected_corrections(choice)
        if (selected_corrections_value["color_method"] not in {"off", "saved_native"}
                or selected_corrections_value["lighting"] and selected_corrections_value["color_method"] == "off"):
            raise ValueError("Prepare independent optical inputs for the selected colour/lighting steps")
    selected = recipe["phase"]["model_bundles"].get("selected")
    if selected is not None:
        return _selected_photometry_binding(context, selected, output, reader,
            choice=choice, choice_authority=image_choice_authority)
    native = _runtime_parameters(recipe["native_configuration"], reader)
    original = reader.json(recipe["original_native_manifest"])
    photo = recipe["original_photo"]
    raw = context.provenance["raw_acquisition"]
    if photo["v5_B_arw_sha256"] != raw["sha256"]:
        raise ValueError("The current Build baseline differs from the frozen photometric baseline")
    processor = original["v5_photometry"]["processor"]
    calibration_path = reader.resolve_path(_path(processor["calibration_manifest_path"], parameters=native))
    calibration_bound = dict(path=str(calibration_path), sha256=processor["calibration_manifest_sha256"])
    calibration = reader.json(calibration_bound)
    verified = source.verify_calibration_baselines(calibration_bound["path"],
        calibration_bound["sha256"], context.run, parameters=native)
    models = {}
    for role, run in (("fit", context.run), ("reference", "virtual_reference")):
        models[role] = {"data" if row["kind"] == "baseline" else "sidecar":
            {key: row[key] for key in ("path", "sha256")}
            for row in verified["assets"] if row["run"] == run}
    if choice is not None and choice["method"] != "off":
        reference_run = choice["reference_run"]
        reference_assets = source.verify_calibration_baselines(calibration_bound["path"],
            calibration_bound["sha256"], reference_run, parameters=native)
        models["reference"] = {"data" if row["kind"] == "baseline" else "sidecar":
            {key: row[key] for key in ("path", "sha256")}
            for row in reference_assets["assets"] if row["run"] == reference_run}
    old = [row for row in calibration.get("old_B_local_residual_fields", ()) if row["run"] == context.run]
    if len(old) > 1:
        raise ValueError("The frozen run has conflicting old-light fields")
    models["old_field"] = None
    if old:
        path = reader.resolve_path(_path(old[0]["field_path"], parameters=native))
        if not path.is_absolute():
            path = calibration_path.parents[3] / path
        models["old_field"] = dict(path=str(path), sha256=old[0]["field_sha256"])
    certificate = reader.json(context.provenance["baseline_renderer"])
    original_renderer = reader.json(dict(
        path=str(reader.resolve_path(_path(photo["v5_B_registration_certificate_path"], parameters=native))),
        sha256=photo["v5_B_registration_certificate_sha256"]))
    if original_renderer["sources"]["arw_sha256"] != raw["sha256"]:
        raise ValueError("The frozen reference renderer belongs to another RAW baseline")
    matrix = lambda value: source.source_homography(value, parameters=native)
    original_h = matrix(original_renderer["review_to_camera_h"])
    key = native["source"]["symbols"]["reference_to_frame_h_key"]
    old_reference = np.linalg.inv(matrix(photo["H_v5_B_to_frame_native"])) @ matrix(photo[key])
    reference = matrix(np.linalg.inv(matrix(certificate["renderer"]["native_to_camera_u_h"]))
                       @ original_h @ old_reference)
    metadata = deepcopy(certificate["inputs"]["uncorrected_grid"]["metadata"])
    for key in ("type", "color_space"):
        metadata.pop(key, None)
    value = dict(contract=raw_rig_source.CONTRACT, run=context.run,
        profile_id=context.provenance["camera_binding"]["profile_id"],
        native_shape_hw=deepcopy(context.grid["num_voxels"]), baseline_raw=deepcopy(raw),
        baseline_reference_to_native_h=reference.tolist(),
        shape_assets=deepcopy(certificate["renderer"]["shape_corrections"]),
        image_metadata=metadata, models=models, execution_identity=raw_rig_source.execution_identity())
    if choice is not None:
        from ..native.full_photometry import ORDER
        choice_pin = _write_json(Path(output).with_name("image_normalisation_choice.json"), dict(
            format="darsia-prepared-image-normalisation-v1", choice=deepcopy(choice),
            source_series=deepcopy(image_choice_authority)))
        value.update(correction_order=ORDER,
            normalisation=dict(choice=deepcopy(choice), record=choice_pin))
    raw_rig_source.validate_binding(value, native)
    return _write_json(Path(output), value)


def _selected_photometry_binding(context, selected, output, reader, *, choice=None, choice_authority=None):
    """Consume the trainer's actual per-run RAW processor on the same crop/grid."""
    import numpy as np
    from ..mass.n1_pipeline import _write_json
    from ..native.raw_rig_source import validate_binding
    basis = reader.json(selected["image_basis"])
    rows = [row for row in basis["runs"] if row["run"] == context.run]
    if len(rows) != 1:
        raise ValueError("The trained image basis has no exact source-grid binding for this run")
    row = rows[0]
    binding = reader.json(row["raw_binding"])
    if choice is not None:
        from ..image_normalisation import selected_corrections
        actual_choice = binding.get("normalisation", {}).get("choice", {})
        keys = ("format", "series_id", "method", "reference_run", "runs")
        if (any(actual_choice.get(key) != choice.get(key) for key in keys)
                or selected_corrections(actual_choice) != selected_corrections(choice)
                or any(actual_choice.get(key) != choice.get(key)
                       for key in ("stable_area", "exclusion_band", "light_field"))):
            raise ValueError("The selected model was trained with another saved image-normalisation choice")
    manifest = reader.json(row["source_manifest"])
    audit = reader.json(manifest["reference"]["audit"])
    renderer = reader.json(context.provenance["baseline_renderer"])["renderer"]
    if (binding["run"] != context.run or manifest["run"] != context.run
            or binding["baseline_raw"]["sha256"] != context.provenance["raw_acquisition"]["sha256"]
            or list(row["native_shape_hw"]) != list(context.grid["num_voxels"])
            or binding["native_shape_hw"] != list(context.grid["num_voxels"])
            or not np.allclose(np.asarray(audit["renderer"]["native_to_camera_u_h"], float),
                               np.asarray(renderer["native_to_camera_u_h"], float), rtol=0, atol=1e-10)):
        raise ValueError("The selected model was trained on another crop, camera basis or RAW baseline")
    actual = reader.json(context.provenance["native_configuration"])
    validate_binding(binding, actual)
    for group in ("fit", "reference"):
        for pin in binding["models"][group].values():
            reader.path(pin)
    if binding["models"]["old_field"] is not None:
        reader.path(binding["models"]["old_field"])
    return _write_json(Path(output), binding)
