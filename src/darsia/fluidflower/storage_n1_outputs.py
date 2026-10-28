"""Optional N1 mass-only storage from evaluated fields and saved phase-one evidence.

No native processor, classifier, inference or calibration is called here. The
old N1 result is serialized first. These new flat files join its original frame
inventory before commit. This is a separate positive producer contract, never a
native-phase series receipt. The caller still owns admission and publication.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path

import numpy as np

from . import storage_outputs as output
from .storage_resolution import LABEL_LEGEND, PROBABILITY_FIELDS
from .pipeline import staged_data as staged
from .pipeline.series import bind_mass_frame

FRAME_FORMAT = "darsia-n1-storage-frame-v1"
PRODUCER_KIND = "n1-mass-only"


def _copy(bound, destination, final, reader, *, kind):
    """Copy exact already-saved bytes into this new frame, not a new producer.

    The copy goes through the N111-retried publisher; each try copies anew.
    """
    from .pipeline.io_ops import publish_file
    path = reader.path(bound)
    lengths = []

    def copy(target):
        lengths.clear()
        with Path(path).open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                lengths.append(len(chunk))
    digest = publish_file(destination, copy)
    if digest != bound["sha256"] or sum(lengths) != bound["size_bytes"]:
        raise ValueError("Saved phase-one bytes changed during exact storage copy")
    return dict(path=str(Path(final) / Path(destination).name),
                sha256=digest, size_bytes=sum(lengths), kind=kind)


class Phase2Storage:
    """An explicit phase-one metadata closure, consumed one frame at a time."""

    def __init__(self, config, reader, request):
        from .mass.n1_pipeline import DataReader
        if type(reader) is not DataReader:
            raise TypeError("The original content-verifying N1 DataReader is required")
        if request["resolution_mm"] == "native":
            raise ValueError("Default/native N1 has no optional coarse output hook")
        if config.get("format") != "darsia-n1-run-v1" or config.get("precision") != request["precision"]:
            raise ValueError("Optional N1 storage requires a genuine precision-bound phase-two configuration")
        self.config, self.reader, self.request = config, reader, request
        self.binding = deepcopy(config.get("phase1"))
        binding = self.binding
        staged._keys(binding, {"configuration", "completion", "code_commit", "frames"}, {"origin"}, where="phase-two evidence")
        if "origin" in binding:
            from .pipeline.reproduction_lineage import validate_phase1_origin
            validate_phase1_origin(binding["origin"])
            if binding["origin"]["run"] != config["run"]:
                raise ValueError("Phase-one reproduction origin belongs to another run")
        if not isinstance(binding["frames"], dict) or not binding["frames"]:
            raise ValueError("Explicit sealed phase-two case evidence mapping required")
        self.phase1 = staged.validate_phase1_run(reader.json(binding["configuration"]))
        for key, value in self.phase1["statistics"].items():
            actual = config.get(key)
            if key == "provenance" and isinstance(actual, dict) and "common_frozen_refit" in actual:
                from .pipeline.common_calibration import JOIN_FORMAT
                if (not isinstance(actual["common_frozen_refit"], dict)
                        or actual["common_frozen_refit"].get("format") != JOIN_FORMAT):
                    raise ValueError("Only the recognized verified common C provenance extension may be added")
                actual = {field: content for field, content in actual.items() if field != "common_frozen_refit"}
            if key != "format" and not staged._same_json(actual, value):
                raise ValueError("N1 storage scientific DATA differs from its phase-one statistics template: " + key)
        self.completion = reader.json(binding["completion"])
        core = self.completion
        order = [spec["case_id"] for spec in config["frames"]]
        if (core.get("format") != staged.COMPLETION_FORMAT or core.get("complete") is not True
                or core.get("history_complete") is not True or core.get("run") != config["run"]
                or core.get("precision") != request["precision"]
                or core.get("configuration") != binding["configuration"]
                or core.get("code_commit") != binding["code_commit"]
                or core.get("ordered_case_ids") != order
                or self.phase1["ordered_case_ids"] != order
                or self.phase1["run"] != config["run"]
                or self.phase1["precision"] != request["precision"]
                or not isinstance(core.get("frames"), list) or len(core["frames"]) != len(order)
                or set(binding["frames"]) != set(order)):
            raise ValueError("N1 storage requires the exact complete original phase-one run/order")
        if (not isinstance(binding["code_commit"], str) or len(binding["code_commit"]) != 40
                or any(c not in "0123456789abcdef" for c in binding["code_commit"])):
            raise ValueError("Explicit actual phase-one Git commit required")
        self.specs = dict(zip(order, self.phase1["frames"]))
        for case, bound in zip(order, core["frames"]):
            staged._keys(binding["frames"][case], {"commit", "statistics", "before_rules"}, {"photograph"},
                         where="phase-two case evidence")
            staged._bound(bound, where="phase-one actual core commit")
            if not staged._same_json(bound, binding["frames"][case]["commit"]):
                raise ValueError("Phase-two case commit differs from the phase-one core")

    def frame(self, spec):
        """Verify the selected saved commit and normal N1 binding, no replay."""
        case, reader = spec["case_id"], self.reader
        original = self.specs[case]
        refs = self.binding["frames"][case]
        record = staged._frame_record(reader.json(refs["commit"]), original,
            self.binding["configuration"], self.request["precision"], refs["commit"])
        expected = bind_mass_frame(original, record["native"], record["phase"],
                                  baseline=self.phase1["phase"]["reference"]["full"])
        # Only the explicitly selected N16 route has these two source-bound
        # additions; compare the unchanged phase-one mass input independently.
        compared = spec
        if "n16" in self.config.get("pixel_domain", {}):
            if not {"pixel_domain_geometry", "frozen_response_basis"} <= set(spec):
                raise ValueError("N16 requires its actual geometry and frozen original response")
            compared = {key: value for key, value in spec.items()
                        if key not in {"pixel_domain_geometry", "frozen_response_basis"}}
        if not staged._same_json(expected, compared):
            raise ValueError("N1 frame differs from its exact sealed phase-one mass input")
        required = {"commit", "statistics", "before_rules"}
        staged._keys(refs, required, {"photograph"}, where="phase-two case evidence")
        for key in ("statistics", "before_rules", "photograph"):
            if not staged._same_json(refs.get(key), record.get(key)):
                raise ValueError("Phase-two saved evidence differs from its committed phase-one producer")
        before = reader.json(record["before_rules"]["receipt"])
        if (before.get("format") != staged.BEFORE_FORMAT or before.get("complete") is not True
                or not staged._same_json(before.get("source"), record["source"])
                or before.get("png") != record["before_rules"].get("png")
                or before.get("before_rules", {}).get("state") != record["before_rules"]["state"]):
            raise ValueError("Saved phase-one BEFORE receipt/source differs")
        staged._before_metadata(before["before_rules"], record["native"]["full"]["phase_shape_hw"])
        return record, before


def write_n1_frame(temporary, final, *, manifest, manifest_asset, spec, config,
                   evaluated, capture, context, extensive_inputs=None):
    """Use the four unchanged reducers, then add only new immutable flat files."""
    from PIL import Image
    if extensive_inputs is not None:
        from .storage_extensive_io import validate_inputs
        validate_inputs(extensive_inputs)
    temporary, final = Path(temporary), Path(final)
    if (final.name != spec["case_id"] or final.parent.name != "frames"
            or temporary.parent != final.parent or not temporary.name.startswith("." + spec["case_id"] + ".")
            or not temporary.name.endswith(".partial") or not temporary.is_dir()
            or not staged._same_json(config, context.config)):
        raise ValueError("N1 storage must join the caller's existing owned frame partial/configuration")
    record, before_doc = context.frame(spec)
    reader, request = context.reader, context.request
    result, frame = evaluated["result"], capture["frame"]
    if frame is not evaluated["response"]["frame"]:
        raise ValueError("N1 storage capture is not the original evaluated frame")
    labels = np.asarray(frame.phase_labels)
    if (labels.dtype != np.uint8 or labels.ndim != 2 or not np.isin(labels, (0, 1, 2, 255)).all()
            or request["precision"] != result["precision"]):
        raise ValueError("N1 storage requires unchanged final labels/precision")
    with Image.open(reader.path(record["phase"]["labels"])) as image:
        if not np.array_equal(np.asarray(image), labels):
            raise ValueError("N1 storage final labels differ from the phase-one commit")
    shape = labels.shape
    grid = output.make_grid(capture["x_m"], capture["y_m"],
        np.broadcast_to(np.asarray(frame.pixel_area_m2), shape), include=frame.domain,
        extent_m=request["extent_m"], storage_resolution_mm=request["resolution_mm"],
        precision=request["precision"])
    files = []
    copied_core = _copy(context.binding["completion"], temporary / "storage_phase1_completed.json",
                        final, reader, kind="operational_receipt")
    copied_commit = _copy(context.binding["frames"][spec["case_id"]]["commit"],
        temporary / "storage_phase1_commit.json", final, reader, kind="operational_receipt")
    copied_before = _copy(record["before_rules"]["receipt"], temporary / "storage_phase1_before.json",
                          final, reader, kind="operational_receipt")
    files.extend((copied_core, copied_commit, copied_before))
    before = deepcopy(before_doc["before_rules"])
    probabilities = {PROBABILITY_FIELDS[1]: labels == 2}
    if before["state"] == "complete":
        bound = _copy(record["before_rules"]["png"],
            temporary / "storage_gas_probability_before_rules.png", final, reader, kind="map")
        files.append(bound)
        probabilities[PROBABILITY_FIELDS[0]] = output.decode_before(
            temporary / "storage_gas_probability_before_rules.png", shape, request["precision"])
    masks = reader.archive(spec["classes"], ("rig_domain", "modelled", "filled_observed", "filled_unobserved"))
    if any(value.dtype != bool or value.shape != shape for value in masks.values()):
        raise ValueError("Saved phase-one class masks differ from the native grid")
    masks["main_mass_eligible"] = result["fields"]["main_mass_eligible_mask"]
    masks.update(frame.excluded_masks or {})
    genuine = extensive_inputs is not None and extensive_inputs.get("qualification") == "genuine-n15-n16-source-producer"
    if genuine:
        from .storage_extensive_io import _producer_binding
        binding = _producer_binding(extensive_inputs["producer_binding"])
        masses = evaluated["n16_producer"]["measured_masses"]
        if any(output._array_identity(masses[name]) != binding["measured_mass_identities"][name]
               for name in output.TOTALS):
            raise ValueError("Actual N15 measured component arrays changed before storage")
        density_ids = {name: deepcopy(binding["provider_map_identities"][name]) for name in output.DENSITIES.values()}
        weight_id = output._array_identity(np.asarray(capture["volume_m3"], np.float64) * 1000.)
        if weight_id != binding["native_geometry"]["arrays"]["transport_weight_l"]:
            raise ValueError("Actual N15 transport weight changed before storage")
    else:
        masses, density_ids, weight_id = output.pixel_masses(evaluated["export"], capture["volume_m3"], shape)
    totals, checks = {}, {}
    for component, name in output.TOTALS.items():
        expected = output.quantified_totals(result)[component]
        if isinstance(expected, bool) or not isinstance(expected, (int, float)) or not math.isfinite(expected) or expected < 0:
            raise ValueError("Original N1 scalar mass must be finite/nonnegative")
        actual = float(masses[component].sum(dtype=np.float64))
        if abs(actual - expected) > 1e-6 * abs(expected):
            raise ValueError("Native pixel mass differs from original N1 " + component)
        totals[component] = expected
        checks[component] = dict(native_total_g=expected, pixel_sum_g=actual, relative_limit=1e-6, passed=True)
    reduced = output.aggregate_frame(grid,
        concentration_aqueous=evaluated["export"]["products"]["concentration_aqueous"],
        saturation_free_gas=evaluated["export"]["products"]["saturation_free_gas"],
        phase_labels=labels, masks=masks, probabilities=probabilities,
        native_pixel_masses_g=masses, native_mass_totals_g=totals, before_status=before)
    static = output.aggregate_static(grid, depth_m=np.broadcast_to(np.asarray(frame.depth_m), shape),
                                     porosity=np.broadcast_to(np.asarray(frame.porosity), shape), facies=frame.facies)
    arrays = dict(reduced["arrays"], **{"static_" + key: value for key, value in static.items()})
    archive = temporary / "storage_stored_frame.npz"
    output._write_npz(archive, arrays)
    files.append(dict(output._asset(archive), path=str(final / archive.name), kind="map"))
    photo = dict(state="pending", reason="explicit_saved_pre_light_photograph_not_bound")
    if record.get("photograph", {}).get("state") == "pending":
        pending = reader.json(record["photograph"]["receipt"])
        if (pending.get("format") != "darsia-phase-original-photograph-pending-v1"
                or pending.get("state") != "pending"
                or pending.get("reason") != photo["reason"]
                or not staged._same_json(pending.get("source"), record["source"])):
            raise ValueError("Phase-one original-photo pending receipt differs")
    if record.get("photograph", {}).get("state") == "complete":
        original_photo = reader.json(record["photograph"]["receipt"])
        request_photo = context.phase1.get("photographs", {}).get(spec["case_id"])
        if request_photo is None:
            raise ValueError("Saved original photograph lacks its explicit phase-one producer request")
        # Reuse the qualified pixel/ROI/pre-light validator on the already-saved
        # PNG. The evidence remains phase-one evidence, never regenerated here.
        from .storage_saved_export import _photograph
        _photograph(record["photograph"]["receipt"],
            dict(run=config["run"], case_id=spec["case_id"], precision=request["precision"],
                 source=record["source"], grid=dict(native_shape=list(shape))),
            resolve_asset=reader.path, geometry_producer_sha256=request_photo["geometry_producer_sha256"])
        old_receipt = _copy(record["photograph"]["receipt"], temporary / "storage_phase1_photo.json",
                            final, reader, kind="operational_receipt")
        png = _copy(record["photograph"]["png"], temporary / "storage_original_photo.png", final, reader, kind="map")
        new_photo = deepcopy(original_photo)
        new_photo["photograph"] = {k: png[k] for k in ("path", "sha256", "size_bytes")}
        new_photo["phase1_original_receipt"] = deepcopy(record["photograph"]["receipt"])
        output._write_json(temporary / "storage_original_photo.json", new_photo)
        receipt = dict(output._asset(temporary / "storage_original_photo.json"),
                       path=str(final / "storage_original_photo.json"), kind="operational_receipt")
        files.extend((old_receipt, png, receipt))
        photo = dict(state="complete", receipt=receipt, png=png, original_receipt_copy=old_receipt,
                     geometry_producer_sha256=request_photo["geometry_producer_sha256"])
    document = dict(format=FRAME_FORMAT, producer_kind=PRODUCER_KIND, complete=True,
        job_id=manifest["job_id"], run=config["run"], case_id=spec["case_id"],
        precision=request["precision"], configuration=deepcopy(manifest["configuration"]),
        code_commit=manifest["code"]["commit"], calibration_sha256=config["calibration"]["sha256"],
        source=deepcopy(record["source"]), source_binding=deepcopy(spec["source_binding"]),
        request=deepcopy(request), manifest_sha256=manifest_asset["sha256"],
        result=dict(output._asset(temporary / "result.json"), path=str(final / "result.json")),
        phase1=dict(binding=deepcopy(context.binding), core_copy=copied_core, commit_copy=copied_commit,
                    before_copy=copied_before), before_rules=before, photograph=photo,
        after_rules=dict(source_array="final phase_labels == 2", source=deepcopy(record["phase"]["labels"]),
            values=[0, 1], gas_pixels=int((labels == 2).sum()), label_legend=dict(LABEL_LEGEND),
            coarse_reduction="count(final labels==2)/all native cell pixels, irrespective of N1 domain or area",
            support_mask=None, support_fraction="not inferred; no explicit label support mask supplied"),
        final_label_counts={str(v): int(np.count_nonzero(labels == v)) for v in (0, 1, 2, 255)},
        native_mass_density_identities=density_ids, native_transport_weight_l_identity=weight_id,
        native_mass_totals_g=totals, native_pixel_mass_checks=checks,
        coarse=dict(reduced["metadata"], arrays={k: output._array_identity(v) for k, v in arrays.items()},
                    per_frame_geometry=True, shared_static_geometry_assumed=False),
        files=files, existing_native_products_unchanged=True)
    output.bind_analysis_scope(document, result)
    if genuine:
        document["n16_extensive_producer"] = deepcopy(binding)
    if extensive_inputs is not None:
        from .storage_extensive_io import prepare_payload, write_payload, declared_files
        product = prepare_payload(extensive_inputs, grid, labels=labels,
            saturation=evaluated["export"]["products"]["saturation_free_gas"],
            masses=masses, totals=totals)
        document["extensive"] = write_payload(temporary, final, product, document)
        files.extend(declared_files(document["extensive"]).values())
    output._write_json(temporary / "storage_completed.json", document)
    return {Path(bound["path"]).name: bound["kind"] for bound in files} | {"storage_completed.json": "operational_receipt"}
