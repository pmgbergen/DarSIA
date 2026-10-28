"""Optional N14 IO over Q69 synthetic or actual N15/N16 source arrays.

The genuine extension binds original producer, configuration, precision and
chronology before the caller's existing commit. No production adoption is
granted here; the original classic/native and Q69 defaults stay unchanged.
"""
from copy import deepcopy
from pathlib import Path, PureWindowsPath
import re

import numpy as np

from . import storage_extensive as extensive
from . import storage_outputs as output

# Existing Q69 synthetic contract stays byte/schema compatible. The additional
# qualification below binds the actual N15 producer; neither is production GO.
GENUINE_INPUT = "darsia-n15-n16-extensive-input-v1"
GENUINE_QUALIFICATION = "genuine-n15-n16-source-producer"
INPUT_FORMAT = "darsia-extensive-io-input-v1"
FRAME_FORMAT = "darsia-extensive-io-frame-v1"
QUALIFICATION = "explicit-synthetic-inputs"
ARCHIVE = "storage_extensive_frame.npz"
RECEIPT = "storage_extensive_frame.json"
PHYSICAL_KEYS = {
    "depth_m", "porosity", "pixel_pore_volume_m3", "facies", "physical_mask",
    "physical_properties_valid_mask", "mass_support_mask", "hidden_mask",
    "coverage_masks", "intensive_validity_masks", "saturation_valid_mask",
    "native_extensive_totals", "channels", "provenance",
}


def _genuine(value):
    return isinstance(value, dict) and value.get("qualification") == GENUINE_QUALIFICATION


def validate_inputs(value):
    if _genuine(value):
        if (set(value) != {"format", "qualification", "physical_producer_admitted", "physical_inputs", "producer_binding"}
                or value["format"] != GENUINE_INPUT
                or value["physical_producer_admitted"] is not False
                or not isinstance(value["physical_inputs"], dict)
                or set(value["physical_inputs"]) != PHYSICAL_KEYS):
            raise ValueError("Closed genuine N15/N16 extensive input required")
        _producer_binding(value["producer_binding"])
        return value["physical_inputs"]
    if (not isinstance(value, dict)
            or set(value) != {"format", "qualification", "physical_producer_admitted", "physical_inputs"}
            or value["format"] != INPUT_FORMAT or value["qualification"] != QUALIFICATION
            or value["physical_producer_admitted"] is not False
            or not isinstance(value["physical_inputs"], dict)
            or set(value["physical_inputs"]) != PHYSICAL_KEYS):
        raise ValueError("Explicit closed synthetic-only extensive inputs required")
    return value["physical_inputs"]


def prepare_payload(value, grid, *, labels, saturation, masses, totals):
    physical = validate_inputs(value)
    if not _genuine(value) and any(isinstance(row, dict) and row.get("status") == extensive.KNOWN_ZERO
                                 for row in physical["channels"].values()):
        raise ValueError("Q69 synthetic IO does not admit the genuine N16 known-zero extension")
    if grid is None:
        raise ValueError("Extensive I/O requires an explicitly requested coarse grid")
    product = extensive.aggregate_extensive(grid, **physical, phase_labels=labels,
        saturation_free_gas=saturation, native_pixel_masses_g=masses,
        native_mass_totals_g=totals)
    if _genuine(value):
        expected = value["producer_binding"]["measured_mass_identities"]
        if any(not extensive._exact(extensive._identity(masses[name]), expected[name]) for name in output.TOTALS):
            raise ValueError("Writer measured masses differ from the actual N15 producer arrays")
        product["producer_binding"] = deepcopy(value["producer_binding"])
        unknown = physical["channels"]["U"]
        n16 = value["producer_binding"]["n16"]
        if unknown is None:
            scope = physical["provenance"].get("analysis_scope")
            if (not isinstance(scope, dict) or scope.get("whole_rig_mass_complete") is not False
                    or scope.get("unknown_mass_g") is not None or np.any(physical["hidden_mask"])
                    or n16["U_layer_present"] is not False or n16["U_total_g"] != 0
                    or n16.get("co2_adjacent") is not True):
                raise ValueError("Unavailable scoped U requires the actual empty quantified hidden support")
        elif n16["U_layer_present"] is False:
            extensive._known_zero_channel("U", unknown)
            if not extensive._exact(unknown["producer_binding"]["n16"], n16):
                raise ValueError("Known-zero U does not bind the actual producer receipt")
        elif unknown.get("native_total_g", 0) <= 0 or unknown.get("status") == extensive.KNOWN_ZERO:
            raise ValueError("Nonzero U layer requires the actual positive U contribution")
    return product


def frame_identity(document):
    keys = {"format", "job_id", "run", "case_id", "configuration", "code_commit",
            "calibration_sha256", "precision", "source", "request"}
    if document["format"] == "darsia-n1-storage-frame-v1":
        keys |= {"source_binding", "manifest_sha256"}
    elif document["format"] != output.FRAME_FORMAT:
        raise ValueError("Only an existing native or genuine N1 storage identity is admitted")
    return deepcopy({key: document[key] for key in keys})


def declared_files(section):
    """Admit exactly two optional files; this never reads their payloads."""
    if section is None:
        return {}
    if (not isinstance(section, dict)
            or set(section) != {"format", "qualification", "physical_producer_admitted", "receipt", "archive"}
            or section["format"] != FRAME_FORMAT or section["qualification"] not in {QUALIFICATION, GENUINE_QUALIFICATION}
            or section["physical_producer_admitted"] is not False):
        raise ValueError("Extensive I/O section is not the closed synthetic-only format")
    result = {}
    for key, name, kind in (("receipt", RECEIPT, "operational_receipt"), ("archive", ARCHIVE, "map")):
        row = section[key]
        if (not isinstance(row, dict) or set(row) != {"path", "sha256", "size_bytes", "kind"}
                or not isinstance(row["path"], str) or PureWindowsPath(row["path"]).name != name
                or type(row["size_bytes"]) is not int or row["size_bytes"] < 1
                or not isinstance(row["sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                or row["kind"] != kind):
            raise ValueError("Extensive I/O descriptor/name/kind differs")
        result[name] = row
    if PureWindowsPath(section["receipt"]["path"]).parent != PureWindowsPath(section["archive"]["path"]).parent:
        raise ValueError("Extensive I/O files belong to different frames")
    return result


def write_payload(temporary, final, product, document):
    """Add only new immutable files to the original caller-owned frame partial."""
    temporary, final = Path(temporary), Path(final)
    if any((temporary / name).exists() or (final / name).exists() for name in (ARCHIVE, RECEIPT)):
        raise FileExistsError("Extensive outputs must be new immutable frame files")
    archive = temporary / ARCHIVE
    output._write_npz(archive, product["arrays"])
    metadata = product["metadata"]
    # Reuse the strict pure reader; no second reduction or producer is run.
    extensive.read_extensive_archive(archive.read_bytes(), metadata,
        expected_input_identities=metadata["input_identities"],
        native_mass_totals_g=metadata["native_mass_totals_g"],
        native_extensive_totals=metadata["native_extensive_totals"],
        expected_channels=metadata["channels"], expected_provenance=metadata["provenance"],
        expected_grid_metadata=metadata["grid"])
    bound_archive = dict(output._asset(archive), path=str(final / ARCHIVE), kind="map")
    qualification = GENUINE_QUALIFICATION if "producer_binding" in product else QUALIFICATION
    if qualification == GENUINE_QUALIFICATION:
        verify_producer_document(product["producer_binding"], document)
    envelope = dict(format=FRAME_FORMAT, qualification=qualification,
        physical_producer_admitted=False, frame_identity=frame_identity(document),
        archive=bound_archive, api_metadata=metadata)
    if qualification == GENUINE_QUALIFICATION:
        envelope["producer_binding"] = deepcopy(product["producer_binding"])
    receipt = temporary / RECEIPT
    output._write_json(receipt, envelope)
    section = dict(format=FRAME_FORMAT, qualification=qualification,
        physical_producer_admitted=False, archive=bound_archive,
        receipt=dict(output._asset(receipt), path=str(final / RECEIPT), kind="operational_receipt"))
    declared_files(section)
    return section


def _read_payload(stored, resolve_asset):
    from . import storage_saved_products as saved
    document = stored["storage_document"]
    section = document.get("extensive")
    files = declared_files(section)
    if not files:
        return dict(state="not_requested", physical_binding_admitted=False,
                    producer_identity_admitted=False)
    loaded = []
    envelope = saved._json(saved._read_bytes(files[RECEIPT], resolve_asset, loaded))
    genuine = section["qualification"] == GENUINE_QUALIFICATION
    expected_keys = {"format", "qualification", "physical_producer_admitted", "frame_identity", "archive", "api_metadata"}
    if genuine:
        expected_keys.add("producer_binding")
    if (not isinstance(envelope, dict)
            or set(envelope) != expected_keys
            or envelope["format"] != FRAME_FORMAT or envelope["qualification"] != section["qualification"]
            or envelope["physical_producer_admitted"] is not False
            or not extensive._exact(envelope["frame_identity"], frame_identity(document))
            or not extensive._exact(envelope["archive"], files[ARCHIVE])):
        raise ValueError("Extensive I/O original frame/source/descriptor identity differs")
    if genuine:
        binding = envelope["producer_binding"]
        verify_producer_document(binding, document)
        if not extensive._exact(binding, stored["native_scalar_reference"].get("n16_extensive_producer")):
            raise ValueError("Extensive producer differs from the original committed N15 result")
    metadata = envelope["api_metadata"]
    if (not isinstance(metadata, dict) or not {"input_identities", "native_mass_totals_g",
            "native_extensive_totals", "channels", "provenance", "grid"} <= set(metadata)):
        raise ValueError("Extensive I/O API metadata is incomplete")
    totals = output.quantified_totals(stored["native_scalar_reference"])
    result = extensive.read_extensive_archive(
        saved._read_bytes(files[ARCHIVE], resolve_asset, loaded), metadata,
        expected_input_identities=metadata["input_identities"], native_mass_totals_g=totals,
        native_extensive_totals=metadata["native_extensive_totals"],
        expected_channels=metadata["channels"], expected_provenance=metadata["provenance"],
        expected_grid_metadata=stored["grid"])
    for component in output.TOTALS:
        key = "cell_mass_" + component + "_g"
        actual, original = result["arrays"][key], stored["arrays"][key]
        if actual.dtype != original.dtype or not np.array_equal(actual, original):
            raise ValueError("Extensive " + component + " cell masses differ from committed coarse arrays")
    if genuine:
        if (not extensive._exact(metadata["input_identities"], binding["extensive_input_identities"])
                or not extensive._exact(metadata["native_extensive_totals"], binding["native_extensive_totals"])
                or not extensive._exact(metadata["channels"], binding["channel_metadata"])
                or not extensive._exact(metadata["provenance"], binding["extensive_provenance"])):
            raise ValueError("Extensive original producer input array identities differ")
        value = dict(result, state="genuine_n15_n16_io", loaded_assets=loaded,
            original_frame_identity=deepcopy(envelope["frame_identity"]),
            producer_binding=deepcopy(binding), physical_binding_verified=True,
            producer_identity_verified=True, production_permission_granted=False,
            U_layer_present=binding["n16"]["U_layer_present"])
    else:
        value = dict(result, state="synthetic_io_only", loaded_assets=loaded,
                     original_frame_identity=deepcopy(envelope["frame_identity"]))
    if stored.get("analysis_scope") is not None:
        value["approved_analysis_gap_area_fraction"] = stored["arrays"]["mask_approved_analysis_gap_area_fraction"]
    return output.read_mass_scope(value, stored["native_scalar_reference"], totals)


def read_committed_payload(record, *, producer_kind, resolve_asset, manifest=None, manifest_asset=None):
    """First enforce the existing positive producer/commit/source reader."""
    if producer_kind == "native-phase-n1":
        from .storage_saved_products import read_stored_frame
        stored = read_stored_frame(record, resolve_asset=resolve_asset)
    elif producer_kind == "n1-mass-only":
        from .storage_n1_saved_products import read_n1_stored_frame
        stored = read_n1_stored_frame(record, manifest=manifest, manifest_asset=manifest_asset,
                                      resolve_asset=resolve_asset)
    else:
        raise ValueError("An explicit original native or genuine N1 producer kind is required")
    return _read_payload(stored, resolve_asset)


def read_indexed_extensive_run(index, *, producer_kind, current_machine_id, collections=(), source_paths=None,
                               qualification=QUALIFICATION):
    """One committed frame at a time; existing resolver/collect rules unchanged."""
    from . import storage_saved_products as saved
    from .storage_indexed_products import indexed_asset_resolver
    from .pipeline import products_publication as publication
    resolve = indexed_asset_resolver(index, current_machine_id=current_machine_id,
                                    collections=collections, source_paths=source_paths)
    completion = saved._json(publication.read_published_metadata(index))
    if producer_kind == "native-phase-n1":
        run = saved.read_stored_run(completion, resolve_asset=resolve, stream=True)
    elif producer_kind == "n1-mass-only":
        from .storage_n1_saved_products import read_n1_stored_run
        run = read_n1_stored_run(completion, resolve_asset=resolve, stream=True)
    else:
        raise ValueError("An explicit original native or genuine N1 producer kind is required")
    if qualification not in {QUALIFICATION, GENUINE_QUALIFICATION}:
        raise ValueError("Explicit supported extensive IO qualification required")
    def frames():
        for frame in run["frames"]:
            actual = _read_payload(frame, resolve)
            expected_state = "genuine_n15_n16_io" if qualification == GENUINE_QUALIFICATION else "synthetic_io_only"
            if actual["state"] != expected_state:
                raise ValueError("Indexed extensive IO qualification differs from the committed producer")
            yield actual
    state = "genuine_n15_n16_io" if qualification == GENUINE_QUALIFICATION else "synthetic_io_only"
    return dict(state=state, producer_kind=producer_kind,
                ordered_case_ids=list(run["ordered_case_ids"]), frames=frames(),
                physical_binding_admitted=False, producer_identity_admitted=False)


PRODUCER_KEYS = {"contract", "configuration", "manifest_sha256", "code_commit",
    "calibration_sha256", "precision", "source_binding", "phase", "classes", "area",
    "case_id", "frame_index", "elapsed_h", "chronological_position", "source_geometry",
    "frozen_response_basis", "native_products", "native_geometry", "provider_map_identities",
    "measured_mass_identities", "extensive_input_identities", "native_mass_totals_g",
    "native_extensive_totals", "capacity_input_identities", "channel_metadata", "extensive_provenance", "n16", "precision_policy"}


def _producer_binding(value):
    if (not isinstance(value, dict) or set(value) != PRODUCER_KEYS
            or value["contract"] != "darsia-n15-n16-actual-producer-v1"
            or value["precision"] not in {"float32", "float64"}
            or re.fullmatch(r"[0-9a-f]{40}", value["code_commit"]) is None
            or type(value["chronological_position"]) is not int
            or value["chronological_position"] < 0
            or set(value["measured_mass_identities"]) != set(output.TOTALS)
            or value["n16"].get("capacity_selector") != "literal-R3-hidden255-solubility"):
        raise ValueError("Actual N15/N16 producer binding is incomplete")
    return value


def verify_producer_document(binding, document):
    _producer_binding(binding)
    if (document.get("format") != "darsia-n1-storage-frame-v1"
            or not extensive._exact(binding, document.get("n16_extensive_producer"))
            or any(not extensive._exact(binding[key], document[key]) for key in
                ("configuration", "manifest_sha256", "code_commit", "calibration_sha256",
                  "precision", "source_binding", "case_id"))
            or not extensive._exact(binding["native_mass_totals_g"],
                document.get("quantified_native_mass_totals_g", document["native_mass_totals_g"]))
            or not extensive._exact(binding["extensive_provenance"].get("analysis_scope"),
                                   document.get("analysis_scope"))):
        raise ValueError("Genuine extensive producer does not match the original N1 storage document")


def bind_n16_producer(evaluated, record, *, manifest, manifest_asset, spec,
                      configuration, static, products, chronological_position):
    """Bind actual source arrays before result.json and the original commit exist.

    Original provider maps are never cast. New native contribution exports alone
    use math.dtype; actual float64 contributions go directly to the reducers.
    Production remains float32 when the existing manifest/request says float32.
    """
    p = evaluated["n16_producer"]
    geometry, masks = p["geometry"], p["masks"]
    measured = evaluated["result"]["fields"]["main_mass_eligible_mask"]
    positive = np.isin(evaluated["response"]["frame"].phase_labels, (1, 2))
    mass_support = measured & positive
    intensives = dict(depth_m_mean=masks["valid"], porosity_mean=masks["valid"],
        concentration_aqueous=masks["measured"] & np.isfinite(evaluated["export"]["products"]["concentration_aqueous"]),
        saturation_free_gas=masks["saturation_valid"])
    totals = {name: record[key] for name, key in output.TOTALS.items()}
    provenance = dict(producer="actual optional N15/N16 pipeline", chronological_position=chronological_position,
        source_binding=deepcopy(spec["source_binding"]), source_geometry=deepcopy(spec["pixel_domain_geometry"]),
        precision_policy=deepcopy(p["precision"]), scientific_eligibility=False,
        production_permission_granted=False)
    if evaluated["result"].get("analysis_scope") is not None:
        provenance["analysis_scope"] = deepcopy(evaluated["result"]["analysis_scope"])
    channel_binding = dict(code_commit=manifest["code"]["commit"],
        configuration=deepcopy(configuration), source_binding=deepcopy(spec["source_binding"]),
        n16=deepcopy(evaluated["pixel_domain"]["n16"]))
    def channel(values, support, total):
        return dict(pixel_values_g=values, support_mask=support, native_total_g=total,
                    producer_binding=deepcopy(channel_binding))
    channels = dict(M_maalt=channel(p["measured_masses"]["total"], mass_support, record["M_measured_g"]),
        M_kant=channel(p["edge"], masks["estimated"], p["M_edge_g"]),
        U=(dict(status=extensive.KNOWN_ZERO, native_total_g=0.,
                producer_binding=deepcopy(channel_binding)) if p["unknown"]["U_layer"] is None else
           channel(p["unknown"]["U_per_pixel_g"], p["hidden"], p["unknown"]["U_g"])))
    if (evaluated["result"].get("analysis_scope") is not None and not np.any(p["hidden"])
            and p["unknown"]["U_layer"] is None and p["observation"]["co2_adjacent"]):
        # The zero sum on an empty quantified subset is not a no-adjacency
        # decision for the unknown remainder. Use the API's unavailable channel.
        channels["U"] = None
    physical = dict(depth_m=geometry["depth_m"], porosity=geometry["porosity"],
        pixel_pore_volume_m3=geometry["pixel_pore_volume_m3"], facies=geometry["facies"],
        physical_mask=masks["physical"], physical_properties_valid_mask=masks["valid"],
        mass_support_mask=mass_support, hidden_mask=p["hidden"],
        coverage_masks=dict(measured=masks["measured"], estimated=masks["estimated"], hidden=p["hidden"]),
        intensive_validity_masks=intensives, saturation_valid_mask=masks["saturation_valid"],
        native_extensive_totals=p["totals"], channels=channels, provenance=provenance)
    # Match the qualified pure API's literal identity names without running a
    # reduction here. The reader compares these with its committed API metadata.
    identities = {key: extensive._identity(value) for key, value in (
        ("pixel_area_m2", geometry["physical_area_m2"]), ("depth_m", physical["depth_m"]),
        ("porosity", physical["porosity"]), ("pixel_pore_volume_m3", physical["pixel_pore_volume_m3"]),
        ("physical_mask", masks["physical"]), ("physical_properties_valid_mask", masks["valid"]),
        ("mass_support_mask", mass_support), ("hidden_mask", p["hidden"]),
        ("saturation_valid_mask", masks["saturation_valid"]), ("facies", physical["facies"]),
        ("phase_labels", evaluated["response"]["frame"].phase_labels),
        ("saturation_free_gas", evaluated["export"]["products"]["saturation_free_gas"]))}
    identities.update({"cell_mass_" + name + "_g": extensive._identity(value)
        for name, value in p["measured_masses"].items()})
    identities.update({"intensive_" + name + "_validity_mask": extensive._identity(value)
        for name, value in intensives.items()})
    identities.update({"coverage_" + name + "_mask": extensive._identity(value)
        for name, value in physical["coverage_masks"].items()})
    for name, value in channels.items():
        if value is None or value.get("status") == extensive.KNOWN_ZERO:
            continue
        identities["cell_" + name + "_g"] = extensive._identity(value["pixel_values_g"])
        identities["channel_" + name + "_support_mask"] = extensive._identity(value["support_mask"])
    binding = dict(contract="darsia-n15-n16-actual-producer-v1", configuration=deepcopy(configuration),
        manifest_sha256=manifest_asset["sha256"], code_commit=manifest["code"]["commit"],
        calibration_sha256=record["bindings"]["calibration_sha256"], precision=record["precision"],
        source_binding=deepcopy(spec["source_binding"]), phase=deepcopy(spec["labels"]),
        classes=deepcopy(spec["classes"]), area=deepcopy(spec["area"]), case_id=spec["case_id"],
        frame_index=spec["frame_index"], elapsed_h=spec["elapsed_h"],
        chronological_position=chronological_position, source_geometry=deepcopy(spec["pixel_domain_geometry"]),
        frozen_response_basis=deepcopy(spec["frozen_response_basis"]), native_products=deepcopy(products),
        native_geometry=deepcopy(static),
        provider_map_identities={name: extensive._identity(evaluated["export"]["products"][name])
            for name in ("concentration_aqueous", "saturation_free_gas", "mass_total", "mass_aqueous", "mass_free_gas")},
        measured_mass_identities={name: extensive._identity(value) for name, value in p["measured_masses"].items()},
        extensive_input_identities=identities, native_mass_totals_g=totals,
        native_extensive_totals=deepcopy(p["totals"]), n16=deepcopy(evaluated["pixel_domain"]["n16"]),
        precision_policy=deepcopy(p["precision"]), extensive_provenance=deepcopy(provenance),
        capacity_input_identities={name: extensive._identity(value) for name, value in p["capacity_inputs"].items()},
        channel_metadata={name: (dict(status=extensive.PENDING) if row is None else
            deepcopy(row) if row.get("status") == extensive.KNOWN_ZERO else
            dict(status="supplied, output-only sum", native_total_g=row["native_total_g"],
                 producer_binding=deepcopy(row["producer_binding"]))) for name, row in channels.items()})
    _producer_binding(binding)
    record["n16_extensive_producer"] = deepcopy(binding)
    evaluated["extensive_inputs"] = dict(format=GENUINE_INPUT, qualification=GENUINE_QUALIFICATION,
        physical_producer_admitted=False, physical_inputs=physical, producer_binding=binding)
