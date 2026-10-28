"""Optional native Rig support with explicit source and persistence bindings.

The shared native-product reader owns adoption and scientific mask replay. This
adapter additionally joins that result to the current RAW reading setup. It
neither measures physical properties nor promotes source approval into QA.
"""
from __future__ import annotations

from darsia.fluidflower import preset_wasserstein_inputs as _preset_io

from dataclasses import dataclass, replace
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


_IMAGES = ("baseline", "shape_corrected_baseline", "depth", "labels", "facies",
           "porosity", "permeability")
_MASKS = ("approved_gap", "porosity_unknown", "permeability_unknown",
          "porosity_nonfinite", "permeability_nonfinite")
_FORMAT = "darsia-rig-native-analysis-state-v1"


def _json(value):
    """Freeze finite JSON while retaining scalar types and original metadata."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _plain(value):
    """Return a detached finite JSON snapshot."""
    return json.loads(_json(value))


def _mask(value, shape):
    """Own an exact native boolean mask through immutable byte storage."""
    array = np.asarray(value)
    if array.dtype != bool or array.shape != tuple(shape):
        raise ValueError("Native analysis masks must retain their exact boolean grid")
    return np.frombuffer(array.tobytes(order="C"), dtype=bool).reshape(shape)


@dataclass(frozen=True)
class _AnalysisState:
    """Detached source proof and independently immutable causes of exclusion."""
    proof_json: str
    provenance_json: str
    approved_gap: np.ndarray
    porosity_unknown: np.ndarray
    permeability_unknown: np.ndarray
    porosity_nonfinite: np.ndarray
    permeability_nonfinite: np.ndarray
    properties_json: str | None = None
    persistence_json: str | None = None


@dataclass
class _PreparedAnalysisFields:
    """Own admitted native arrays only until their matching setup calls consume them."""
    state: _AnalysisState
    sources: dict
    images: dict


def _state(rig):
    """Distinguish undeclared legacy behavior from a malformed active state."""
    value = getattr(rig, "_analysis_domain", None)
    if value is not None and not isinstance(value, _AnalysisState):
        raise ValueError("The declared native analysis state changed")
    return value


def _checked_json(binding, *, source=True):
    """Read one existing descriptor with the original RAW byte guards."""
    from .rig_raw_build import _verify, _verify_inputs
    verify = _verify_inputs if source else _verify
    verify([binding])
    value = json.loads(_preset_io.read_bytes(Path(binding["path"])))
    verify([binding])
    return value


def _binding(value):
    """Retain the declared original file identity without recapturing its bytes."""
    return {key: value[key] for key in ("path", "sha256", "size_bytes")}


def _same_file(left, right, *, same_path=True):
    """Compare original file bytes, allowing only an explicitly permitted copy."""
    return (all(left[key] == right[key] for key in ("sha256", "size_bytes"))
            and (not same_path or Path(left["path"]).resolve() == Path(right["path"]).resolve()))


def _same_source(left, right):
    """Keep source bytes exact while accepting only declared storage moves."""
    from .storage_paths import same_storage_path
    return (_same_file(left, right, same_path=False)
            and same_storage_path(left["path"], right["path"]))


def _same_camera(left, right):
    from .storage_paths import same_storage_path
    return (isinstance(left, dict) and isinstance(right, dict)
            and left.keys() == right.keys()
            and _json({key: value for key, value in left.items() if key != "configuration"})
                == _json({key: value for key, value in right.items() if key != "configuration"})
            and same_storage_path(left["configuration"], right["configuration"]))


def _source_array(binding):
    """Load a declared label product using its original array reader and guards."""
    from .geometry_facies import _load_array
    from .rig_raw_build import _verify
    _verify([binding])
    value = np.asarray(_load_array(Path(binding["path"])))
    _verify([binding])
    return value


def _read_native(contract, native_product, *, labels, run):
    """Call the shared original adoption/mask replay without reproducing it."""
    from .painted_analysis_exclusions import read_native_analysis_exclusions, read_native_product
    if native_product is not None:
        result = read_native_product(native_product, actual_labels=labels, run_id=run)
        if _json(result["contract"].get("analysis_exclusions")) != _json(contract):
            raise ValueError("Native product and optional analysis exclusions differ")
        return result
    return read_native_analysis_exclusions(contract, actual_labels=labels, run_id=run)


def _export_receipt(paths):
    """Reuse the original corrected-export admission rather than infer its receipt."""
    from .corrected_setup import require_corrected_export_for_build
    return require_corrected_export_for_build(paths)


def _native_identity(rig, proof):
    """Join actual reading pixels and full grid to both bound renderer records."""
    from .rig_raw_build import FORMAT, _array_identity, _grid
    image = rig.shape_corrected_baseline
    actual_pixels, actual_grid = _array_identity(image.img), _grid(image)
    grid = proof["native_grid"]
    if (list(image.img.shape[:2]) != grid["shape_hw"]
            or _json(list(image.dimensions)) != _json(grid["legacy_dimensions_m"])):
        raise ValueError("Actual RAW baseline differs from the approved native grid")
    to_u = grid["H_rig_corrected_native_to_baseline_camera_u"]
    from_u = grid["H_baseline_camera_u_to_rig_corrected_native"]
    certificates = {_json(binding): binding for binding in (proof["basis"], grid["renderer"])}
    for key, binding in certificates.items():
        certificate = _checked_json(binding)
        if (key == _json(proof["basis"])
                and (certificate.get("artifact_kind") != "raw_shape_basis"
                     or certificate.get("complete_rig_model") is not False)):
            raise ValueError("Native inputs require the original shape-only RAW basis role")
        renderer, inputs = certificate["renderer"], certificate["inputs"]
        checks = {
            "certificate format": certificate.get("format") == FORMAT,
            "run": certificate.get("run_id") == proof["camera"]["run_id"],
            "RAW source": _same_source(inputs["raw"], proof["raw"]),
            "profile source": _same_source(inputs["profile"], proof["profile"]),
            "camera binding": _same_camera(inputs["camera_binding"], proof["camera"]),
            "renderer profile": renderer["profile_id"] == proof["camera"]["profile_id"],
            "renderer RAW source": _same_source(renderer["raw_acquisition"], proof["raw"]),
            "native shape": renderer["native_shape_hw"] == grid["shape_hw"],
            "baseline pixels": _json(certificate["corrected_baseline"]["pixels"]) == _json(actual_pixels),
            "baseline grid": _json(certificate["corrected_baseline"]["grid"]) == _json(actual_grid),
            "renderer pixels": _json(renderer["source_pixels"]) == _json(actual_pixels),
            "renderer grid": _json(renderer["source_grid"]) == _json(actual_grid),
            "native H": _json(certificate["H_rig_corrected_native_to_baseline_camera_u"]) == _json(to_u),
            "renderer H": _json(renderer["native_to_camera_u_h"]) == _json(to_u),
            "inverse H": _json(renderer["camera_u_to_native_h"]) == _json(from_u),
        }
        failed = [name for name, equal in checks.items() if not equal]
        if failed:
            raise ValueError("Actual RAW source, baseline pixels, full grid or native H differs: " + ", ".join(failed))
    from .storage_paths import same_storage_path
    if (not _same_camera(getattr(rig, "raw_camera", None), proof["camera"])
            or getattr(rig, "raw_baseline_sha256", None) != proof["raw"]["sha256"]
            or not same_storage_path(rig.baseline_path, proof["raw"]["path"])):
        raise ValueError("The Rig no longer names its actual RAW reading source")


def _check_sources(rig, proof):
    """Recheck original source pins and actual images without repeating science."""
    from .rig_raw_build import _array_identity, _verify_inputs
    _verify_inputs(proof["dependencies"])
    request = _checked_json(proof["request"])
    if (_json(request) != _json(proof["request_document"])
            or _json(request.get("analysis_exclusions")) != _json(proof["contract"])
            or _json(request.get("native_product")) != _json(proof["native_product"])
            or not _same_source(request["baseline"], proof["raw"])
            or not _same_source(request["profile"], proof["profile"])
            or not _same_source(request["shape_basis_certificate"], proof["basis"])
            or request["run_id"] != proof["camera"]["run_id"]):
        raise ValueError("Native analysis source proof differs from the original RAW request")
    receipt = _checked_json(proof["export"])
    if (_json(receipt["binding"].get("analysis_exclusions")) != _json(proof["contract"])
            or _json(receipt["binding"].get("native_product")) != _json(proof["native_product"])):
        raise ValueError("Saved native analysis state differs from its original export")
    if not _same_file(receipt["files"]["facies_properties.csv"], proof["property_source"]):
        raise ValueError("Saved native properties differ from the original export table")
    for role, name in (("labels", "native_labels.npz"), ("facies", "native_facies.npz")):
        if not _same_file(receipt["files"][name], proof["sources"][role]):
            raise ValueError("Saved native inputs differ from the original export files")
        if _json(_array_identity(getattr(rig, role).img)) != _json(proof["source_arrays"][role]):
            raise ValueError("Rig labels or facies differ from their admitted native product")
    _native_identity(rig, proof)


def _replay(rig, proof):
    """Replay the shared scientific source only at admission or restoration."""
    _check_sources(rig, proof)
    result = _read_native(proof["contract"], proof["native_product"],
                          labels=rig.labels.img, run=proof["camera"]["run_id"])
    if (_json(result["native_grid"]) != _json(proof["native_grid"])
            or _json(result["provenance"]) != _json(proof["source_provenance"])):
        raise ValueError("The original native product grid or provenance changed")
    return result


def _provenance(proof):
    """Keep source metadata and negative qualification scope separate from approval."""
    selected = proof["native_product"] if proof["native_product"] is not None else proof["contract"]
    return dict(native_product=selected, source_provenance=proof["source_provenance"],
        native_raw_build=proof, physical_qualification_claimed=False,
        reporting_scope="requested native canvas; approved gaps and unassigned properties counted separately")


def prepare_analysis_domain(rig, contract, *, labels_path, facies_path, native_product=None):
    """Admit native labels only after their original export and actual RAW join.

    An undeclared legacy path returns immediately. A fully painted native product
    can be admitted without fabricating an empty gap approval. The later original
    full-build registration/basis consistency checks remain required.
    """
    if contract is None and native_product is None:
        return None
    from .rig_raw_build import RawRigBuild, _array_identity, _descriptor, read_build_request, _verify_inputs
    operation = getattr(rig, "_raw_build", None)
    if not isinstance(operation, RawRigBuild) or operation.artifact_kind != "raw_rig_baseline":
        raise ValueError("Native analysis inputs require the original full RAW build")
    contract, native_product = _plain(contract), _plain(native_product)
    request, _ = read_build_request(operation.request_binding, artifact_kind=operation.artifact_kind)
    if (_json(request) != _json(operation.request)
            or _json(request.get("analysis_exclusions")) != _json(contract)
            or _json(request.get("native_product")) != _json(native_product)
            or not _same_source(request["baseline"], operation.raw_source)
            or not _same_source(request["profile"], operation.profile)
            or request["run_id"] != operation.camera["run_id"]
            or _json(request["configuration_files"]) != _json(list(operation.configurations))):
        raise ValueError("Native analysis inputs differ from the actual RAW request")
    configurations = list(operation.configurations)
    receipt = _export_receipt([Path(row["path"]) for row in configurations])
    if (_json(receipt["binding"].get("analysis_exclusions")) != _json(contract)
            or _json(receipt["binding"].get("native_product")) != _json(native_product)):
        raise ValueError("Native analysis inputs differ from the actual corrected export")
    sources = dict(labels=_descriptor(labels_path), facies=_descriptor(facies_path))
    for role, name in (("labels", "native_labels.npz"), ("facies", "native_facies.npz")):
        if not _same_file(sources[role], receipt["files"][name]):
            raise ValueError("Native labels or facies are not their actual corrected export")
    labels, facies = (_source_array(sources[role]) for role in ("labels", "facies"))
    selected = native_product if native_product is not None else contract
    if not _same_file(sources["labels"], selected["exported_labels"], same_path=False):
        raise ValueError("Native labels differ from the declared Store byte copy")
    result = _read_native(contract, native_product, labels=labels, run=operation.camera["run_id"])
    shape = tuple(result["native_grid"]["shape_hw"])
    if labels.shape != shape or facies.shape != shape:
        raise ValueError("Native labels and facies must retain the exact approved canvas")
    gap = _mask(np.zeros(shape, bool) if result["mask"] is None else result["mask"], shape)
    export = _descriptor(Path(labels_path).parent / "export.json")
    if _json(_checked_json(export)) != _json(receipt):
        raise ValueError("Corrected export changed while native inputs were admitted")
    property_source = _binding(receipt["files"]["facies_properties.csv"])
    dependencies = [operation.request_binding, request["baseline"], request["profile"],
                    export, property_source, *sources.values(),
                    request["shape_basis_certificate"], result["native_grid"]["renderer"], *result["dependencies"]]
    unique = {_json(_binding(row)): _binding(row) for row in dependencies}
    proof = dict(contract=contract, native_product=native_product, export=export, sources=sources,
        source_arrays=dict(labels=_array_identity(labels), facies=_array_identity(facies)),
        basis=_binding(request["shape_basis_certificate"]), native_grid=result["native_grid"],
        source_provenance=result["provenance"], property_source=property_source,
        raw=_binding(operation.raw_source), profile=_binding(operation.profile), camera=operation.camera,
        operation_id=operation.operation_id, request=operation.request_binding, request_document=request,
        admission_configuration_files=configurations,
        configuration_scope="verified during original live Build admission; mutable TOML/series not required on restore",
        dependencies=list(unique.values()))
    _native_identity(rig, proof)
    _verify_inputs(proof["dependencies"])
    state = _AnalysisState(_json(proof), _json(_provenance(proof)), gap,
                           *(_mask(np.zeros(shape, bool), shape) for _ in range(4)))
    rig._analysis_domain = state
    import darsia
    metadata = rig.shape_corrected_baseline.metadata()
    metadata.pop("type", None)
    metadata["series"] = False
    rig._prepared_analysis_fields = _PreparedAnalysisFields(state,
        {role: Path(bound["path"]).resolve() for role, bound in sources.items()},
        {role: darsia.ScalarImage(array, **deepcopy(metadata))
         for role, array in (("labels", labels), ("facies", facies))})
    return state


def setup_prepared_analysis_field(rig, field, path, *, log=None, show_plot=False):
    """Consume an already admitted native map on its exact original setup call."""
    prepared = getattr(rig, "_prepared_analysis_fields", None)
    if not isinstance(prepared, _PreparedAnalysisFields):
        return False
    if prepared.state is not _state(rig):
        del rig._prepared_analysis_fields
        return False
    if field not in prepared.images or Path(path).resolve() != prepared.sources[field]:
        return False
    setattr(rig, field, prepared.images.pop(field))
    if not prepared.images:
        del rig._prepared_analysis_fields
    if log or show_plot:
        import matplotlib.pyplot as plt
        if log:
            plt.figure()
            plt.imshow(getattr(rig, field).img)
            plt.colorbar()
            plt.title(field.capitalize())
            plt.savefig(Path(log) / (field + ".png"), dpi=500)
            plt.close()
        if show_plot:
            plt.figure()
            plt.imshow(rig.baseline.img)
            plt.imshow(rig.facies.img, alpha=0.5)
            plt.title("Facies overlay on baseline")
            plt.show()
    from darsia.presets.workflows.rig import logger
    logger.info("%s setup completed.", field.capitalize())
    return True


def _unassigned(facies, labels):
    """Use literal parsed property IDs, never a label value or property threshold."""
    return np.zeros(facies.shape, bool) if labels is None else ~np.isin(facies, tuple(labels))


def apply_property_support(rig, facies_props):
    """Expose NaN only for actually unassigned properties, preserving all known bits."""
    state = _state(rig)
    if state is None:
        return
    from .rig_raw_build import _array_identity, _json_value
    proof = json.loads(state.proof_json)
    _check_sources(rig, proof)
    unknowns, nonfinite, label_sets, properties = {}, {}, {}, {}
    for name in ("porosity", "permeability"):
        labels = getattr(facies_props, name + "_labels")
        label_sets[name] = None if labels is None else _json_value(sorted(labels))
        unknown = _mask(_unassigned(rig.facies.img, labels), state.approved_gap.shape)
        image = getattr(rig, name)
        if image.img.shape != state.approved_gap.shape:
            raise ValueError("Native physical properties changed their canvas")
        nonfinite[name] = _mask(~np.isfinite(image.img) & ~unknown, state.approved_gap.shape)
        if unknown.any():
            if not np.issubdtype(image.img.dtype, np.floating):
                raise TypeError("Unassigned physical properties need their original floating storage")
            image = image.copy()
            image.img[unknown] = np.nan
            setattr(rig, name, image)
        unknowns[name], properties[name] = unknown, _array_identity(image.img)
    rig._analysis_domain = replace(state, porosity_unknown=unknowns["porosity"],
        permeability_unknown=unknowns["permeability"],
        porosity_nonfinite=nonfinite["porosity"], permeability_nonfinite=nonfinite["permeability"],
        properties_json=_json(dict(identities=properties, assigned_labels=label_sets,
            counts={name + suffix: int(values[name].sum())
                    for values, suffix in ((unknowns, "_unassigned_pixels"), (nonfinite, "_nonfinite_pixels"))
                    for name in ("porosity", "permeability")})))


def _check_properties(rig, state):
    """Require unchanged property bits and the original parsed assignment masks."""
    from .rig_raw_build import _array_identity
    if state.properties_json is None:
        raise ValueError("Native physical assignment support was not recorded")
    row = json.loads(state.properties_json)
    for name, mask in (("porosity", state.porosity_unknown), ("permeability", state.permeability_unknown)):
        values = getattr(rig, name).img
        nonfinite = getattr(state, name + "_nonfinite")
        if (_json(_array_identity(getattr(rig, name).img)) != _json(row["identities"][name])
                or not np.array_equal(_unassigned(rig.facies.img, row["assigned_labels"][name]), mask)
                or not np.array_equal(~np.isfinite(values) & ~mask, nonfinite)
                or row["counts"][name + "_unassigned_pixels"] != int(mask.sum())
                or row["counts"][name + "_nonfinite_pixels"] != int(nonfinite.sum())):
            raise ValueError("Native physical properties or assignment support changed")


def _check_original_properties(rig, state):
    """Restore assignment and values from the actual original CSV parser."""
    from darsia.presets.workflows.facies_props import FaciesProps
    from .rig_raw_build import _array_identity
    proof = json.loads(state.proof_json)
    original = FaciesProps.load(rig.facies, Path(proof["property_source"]["path"]))
    recorded = json.loads(state.properties_json)
    for name in ("porosity", "permeability"):
        labels = getattr(original, name + "_labels")
        unknown = _unassigned(rig.facies.img, labels)
        image = getattr(original, name)
        image.img[unknown] = np.nan
        if (_json(_array_identity(image.img)) != _json(recorded["identities"][name])
                or not np.array_equal(unknown, getattr(state, name + "_unknown"))):
            raise ValueError("Saved physical properties differ from their original table")


def _accounting_provenance(state):
    """Add measured accounting causes without modifying original source/QA metadata."""
    result = json.loads(state.provenance_json)
    result["physical_property_support"] = json.loads(state.properties_json)["counts"]
    if state.persistence_json is not None:
        result["saved_output_verification"] = json.loads(state.persistence_json)
    return result


def wrap_analysis_geometry(rig, geometry):
    """Retain the original geometry for a complete canvas and guard an unknown one."""
    state = _state(rig)
    if state is None:
        return geometry
    _check_properties(rig, state)
    unknown = (state.porosity_unknown | state.permeability_unknown
               | state.porosity_nonfinite | state.permeability_nonfinite)
    if not np.any(state.approved_gap | unknown):
        return geometry
    from .native_analysis_domain import AnalysisDomainGeometry
    return AnalysisDomainGeometry(geometry, state.approved_gap, _accounting_provenance(state),
                                  unknown_properties_mask=unknown)


def _image_record(image):
    """Bind stored image bits and original spatial/temporal metadata."""
    from .rig_raw_build import _array_identity, _json_value
    return dict(array=_array_identity(image.img), metadata=_json_value(image.metadata()),
                voxel_size=_json_value(image.voxel_size), num_voxels=_json_value(image.num_voxels))


def _check_archives(folder, records):
    """Read original saved images one at a time and verify actual persisted content."""
    import darsia
    for name in _IMAGES:
        image = darsia.imread(Path(folder) / (name + ".npz"))
        if _json(_image_record(image)) != _json(records[name]):
            raise ValueError("Saved native analysis image differs: " + name)
        del image


def save_analysis_state(rig, folder, owned, metadata):
    """Save explicit masks through the original bounded writer and bind metadata."""
    state = _state(rig)
    if state is None:
        return
    if owned is None:
        raise ValueError("Native analysis persistence requires its original owned RAW save")
    from .rig_raw_build import _array_identity, _descriptor
    _check_properties(rig, state)
    _check_sources(rig, json.loads(state.proof_json))
    masks = {key: getattr(state, key) for key in _MASKS}
    target = Path(folder) / "analysis_domain_masks.npz"
    with owned.open(target) as stream:
        np.savez(stream, **masks)
    metadata["analysis_domain"] = dict(format=_FORMAT, proof=json.loads(state.proof_json),
        provenance=json.loads(state.provenance_json), properties=json.loads(state.properties_json),
        masks=_descriptor(target), mask_arrays={key: _array_identity(value) for key, value in masks.items()},
        images={name: _image_record(getattr(rig, name)) for name in _IMAGES})


def _saved_state(rig, folder, metadata, *, replay):
    """Reconstruct only verified saved masks and replay their original source proof."""
    from .rig_raw_build import _array_identity, _verify
    row = metadata["analysis_domain"]
    if (not isinstance(row, dict) or set(row) != {"format", "proof", "provenance", "properties", "masks", "mask_arrays", "images"}
            or row["format"] != _FORMAT or set(row["mask_arrays"]) != set(_MASKS)
            or set(row["images"]) != set(_IMAGES)
            or Path(row["masks"]["path"]).resolve() != (Path(folder) / "analysis_domain_masks.npz").resolve()):
        raise ValueError("Saved native analysis metadata changed")
    _verify([row["masks"]])
    with np.load(_preset_io.source(row["masks"]["path"]), allow_pickle=False) as archive:
        if set(archive.files) != set(_MASKS):
            raise ValueError("Saved native analysis mask members changed")
        masks = {key: _mask(archive[key], row["proof"]["native_grid"]["shape_hw"]) for key in _MASKS}
    _verify([row["masks"]])
    if any(_json(_array_identity(value)) != _json(row["mask_arrays"][key]) for key, value in masks.items()):
        raise ValueError("Saved native analysis mask bits changed")
    proof = row["proof"]
    if (metadata.get("raw_build_id") != proof["operation_id"]
            or _json(metadata.get("raw_camera")) != _json(proof["camera"])
            or metadata.get("raw_baseline_sha256") != proof["raw"]["sha256"]):
        raise ValueError("Saved native analysis RAW operation metadata differs")
    if _json(row["provenance"]) != _json(_provenance(proof)):
        raise ValueError("Saved native analysis provenance changed")
    if replay:
        result = _replay(rig, proof)
        gap = np.zeros(masks["approved_gap"].shape, bool) if result["mask"] is None else result["mask"]
        if not np.array_equal(gap, masks["approved_gap"]):
            raise ValueError("Saved approved gap differs from original scientific replay")
    else:
        _check_sources(rig, proof)
    state = _AnalysisState(_json(proof), _json(row["provenance"]), masks["approved_gap"],
        masks["porosity_unknown"], masks["permeability_unknown"], masks["porosity_nonfinite"],
        masks["permeability_nonfinite"], _json(row["properties"]))
    _check_properties(rig, state)
    if replay:
        _check_original_properties(rig, state)
    for name in _IMAGES:
        if _json(_image_record(getattr(rig, name))) != _json(row["images"][name]):
            raise ValueError("Loaded native analysis image differs: " + name)
    _check_archives(folder, row["images"])
    return state


def verify_saved_analysis_state(rig, folder, metadata):
    """Refuse swallowed physical-image write failures before owned save completion."""
    state = _state(rig)
    if state is None:
        if "analysis_domain" in metadata:
            raise ValueError("Saved native analysis metadata has no active source state")
        return
    actual = json.loads(_preset_io.read_bytes(Path(folder) / "meta_data.json"))
    if _json(actual) != _json(metadata):
        raise ValueError("Saved Rig metadata differs from the declared native analysis state")
    restored = _saved_state(rig, folder, metadata, replay=False)
    if (restored.proof_json != state.proof_json or restored.provenance_json != state.provenance_json
            or restored.properties_json != state.properties_json):
        raise ValueError("Saved native analysis state changed during persistence")


def restore_analysis_state(rig, folder, metadata):
    """Restore declared masks before geometry; absence keeps the exact legacy path."""
    if "analysis_domain" not in metadata:
        return
    state = _saved_state(rig, folder, metadata, replay=True)
    report = _saved_inventory(folder, metadata)
    rig._analysis_domain = replace(state, persistence_json=_json(report))
    return report


def _saved_inventory(folder, metadata):
    """Use existing completed output coverage, reporting honest missing coverage."""
    from .rig_raw_build import CERTIFICATE_NAME, FORMAT, _descriptor
    folder = Path(folder)
    required = {"meta_data.json", "analysis_domain_masks.npz", *(name + ".npz" for name in _IMAGES)}
    certificate_path = folder / CERTIFICATE_NAME
    if not _preset_io.exists(certificate_path, 'exists'):
        return dict(scope="immutable source and contract replay; completed output inventory unavailable",
                    checked_outputs=[], uncovered_outputs=sorted(required))
    certificate = _checked_json(_descriptor(certificate_path), source=False)
    proof = metadata["analysis_domain"]["proof"]
    if (certificate.get("format") != FORMAT or certificate.get("artifact_kind") != "raw_rig_baseline"
            or certificate["output_ownership"]["operation_id"] != proof["operation_id"]
            or _json(certificate["inputs"]["operation_request"]) != _json(proof["request"])):
        raise ValueError("Saved RAW output inventory belongs to another operation")
    outputs = {Path(row["path"]).resolve(): row for row in certificate["outputs"]}
    checked = []
    for name in sorted(required):
        path = folder / name
        binding = outputs.get(path.resolve())
        if binding is not None:
            if not _same_file(_descriptor(path), binding):
                raise ValueError("Saved native analysis output changed: " + name)
            checked.append(name)
    missing = sorted(required - set(checked))
    return dict(scope="original completed output inventory" if not missing else
                "partial original output inventory plus immutable source and contract replay",
                checked_outputs=checked, uncovered_outputs=missing)
