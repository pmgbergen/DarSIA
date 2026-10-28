"""Read only the frozen geometric rig needed by the photometry comparison.

The comparison applies the saved shape chain to a byte-verified raw photograph.
Depth, porosity, facies, and color corrections are not needed for that operation.
The exception is an active FF_AC sand registration, which uses the facies map;
that case deliberately retains the full historical rig loader.
"""

from __future__ import annotations

from pathlib import Path
import tomllib
from zipfile import ZipFile

import numpy as np


_SHAPE_CORRECTIONS = frozenset({
    "TypeCorrection", "Resize", "ResizeCorrection", "DriftCorrection",
    "CurvatureCorrection",
})
_COLOR_CORRECTIONS = frozenset({
    "ColorCorrection", "RelativeColorCorrection", "IlluminationCorrection",
    "PatchwiseIlluminationCorrection",
})


def serialized_rig_fingerprint(folder: Path) -> tuple[tuple[str, int, int], ...]:
    """Cheap invalidation token for an already SHA-verified serialized rig."""
    folder = Path(folder)
    rows = []
    for path in sorted(folder.iterdir(), key=lambda item: item.name):
        if path.is_file():
            stat = path.stat()
            rows.append((path.name, stat.st_size, stat.st_mtime_ns))
    return tuple(rows)


def _array_shape(path: Path) -> tuple[int, ...]:
    """Read the NPY header inside a saved DarSIA image without loading pixels."""
    with ZipFile(path) as archive, archive.open("array.npy") as member:
        version = np.lib.format.read_magic(member)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(member)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(member)
        else:
            raise ValueError(f"Unsupported serialized rig image version {version}: {path}")
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"Serialized rig baseline is not RGB: {path}")
    return tuple(int(value) for value in shape)


def _baseline_path(folder: Path) -> Path:
    corrected = folder / "shape_corrected_baseline.npz"
    return corrected if corrected.exists() else folder / "baseline.npz"


def _correction_name(path: Path) -> str:
    # Opening the ZIP and reading class_name does not decompress the often very
    # large drift, illumination, or curvature arrays.
    with np.load(path, allow_pickle=False) as archive:
        value = archive["class_name"].item()
    if not isinstance(value, str) or value not in _SHAPE_CORRECTIONS | _COLOR_CORRECTIONS:
        raise ValueError(f"Unsupported saved rig correction in {path}: {value!r}")
    return value


def load_shape_only_rig(
    folder: Path,
    *,
    dataset: str,
    expected_shape: tuple[int, int] | list[int],
    reference_date,
):
    """Load the exact saved shape chain, excluding unrelated physical fields.

    The caller must separately verify the frozen rig identity and hashes. A
    legacy ``Resize`` correction is loaded through DarSIA's established alias.
    """
    import darsia

    folder = Path(folder)
    if tuple(_array_shape(_baseline_path(folder))[:2]) != tuple(expected_shape):
        raise ValueError("Serialized geometry rig differs from the frozen review grid")
    if dataset == "ffac":
        from darsia.presets.workflows.rig import Rig

        rig = Rig()
    elif dataset == "ffum":
        import ffum

        rig = ffum.MuseumRig()
    else:
        raise ValueError("Geometry-only preview supports registered FF_AC and FF_UM results")

    split_shape = sorted(folder.glob("shape_correction_*.npz"))
    split_color = list(folder.glob("color_correction_*.npz"))
    paths = (split_shape if split_shape or split_color else
             sorted(folder.glob("correction_*.npz")))
    rig.shape_corrections = []
    rig.color_corrections = []
    for path in paths:
        if _correction_name(path) not in _SHAPE_CORRECTIONS:
            continue
        correction = darsia.read_correction(path)
        if not rig._is_shape_correction(correction):
            raise ValueError(f"Saved shape correction is not geometric: {path}")
        rig.shape_corrections.append(correction)
    if not rig.shape_corrections:
        raise ValueError(f"Serialized rig has no shape corrections: {folder}")
    rig.reference_date = reference_date
    rig._photometry_rig_folder = folder
    rig._photometry_rig_fingerprint = serialized_rig_fingerprint(folder)
    return rig


def _frozen_ffac_rig_path(contract: dict) -> Path:
    from darsia.fluidflower.review_storage import resolve_review_path

    paths = contract.get("config_paths") or []
    if not paths:
        raise ValueError("Frozen FF_AC geometry configuration is not recorded")
    results, rig = None, None
    for item in paths:
        path = resolve_review_path(Path(item))
        with path.open("rb") as stream:
            config = tomllib.load(stream)
        results = (config.get("data") or {}).get("results", results)
        rig = (config.get("rig") or {}).get("path", rig)
    if rig:
        return resolve_review_path(Path(rig))
    if results:
        return resolve_review_path(Path(results)) / "setup" / "rig"
    raise ValueError("Frozen FF_AC geometry rig is not identified")


def load_registered_shape_rig(anchor, native, dataset, run, anchor_index):
    """Verify the historical source/rig contract and load its geometric chain.

    This has the same identity checks as the original full-rig comparison path.
    An active FF_AC registration retains the full loader because registration
    derives its static domain from saved facies.
    """
    import darsia

    from darsia.fluidflower.review_storage import resolve_review_path
    from darsia.fluidflower.series_paths import config_files
    from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig

    contract = native.get("correction_contract") or {}
    config = FluidFlowerConfig(config_files(dataset, run), require_data=True, require_results=False)
    experiment = darsia.ProtocolledExperiment.init_from_config(config)
    # A dialog can retain this rig while the user moves between images. Check
    # the complete saved timeline up front rather than validating only the
    # image that happened to be selected when the rig entered the cache.
    for frame in anchor["frames"]:
        source_row = native["frame_sources"][frame["frame_index"]]
        source_path = resolve_review_path(Path(source_row["source_path"]))
        observed_h = float(experiment.time_since_start(experiment.get_datetime(source_path)))
        if abs(observed_h - float(source_row["elapsed_h"])) > 1e-6:
            raise ValueError("Current experiment protocol differs from the frozen image timeline")
    selected_artifacts = []
    if dataset == "ffac":
        from darsia.presets.workflows.rig import Rig

        frozen = _frozen_ffac_rig_path(contract)
        selected = resolve_review_path(Path(config.rig.path))
        if not frozen.samefile(selected):
            raise ValueError("Current and frozen FF_AC configurations select different geometry rigs")
        registration = contract.get("rig_registration") or {}
        if registration.get("active"):
            if not registration.get("profile"):
                raise ValueError("Frozen spatial registration cannot be reconstructed")
            rig = Rig.load(selected)
            from darsia.fluidflower.tools.phase_rig_registration import install_registration

            install_registration(rig, registration["profile"], run=run)
            if tuple(rig.shape_corrected_baseline.shape[:2]) != tuple(native["analysis_image_shape_hw"]):
                raise ValueError("Serialized geometry rig differs from the frozen review grid")
            rig.load_experiment(experiment)
        else:
            rig = load_shape_only_rig(
                selected, dataset=dataset,
                expected_shape=native["analysis_image_shape_hw"],
                reference_date=experiment.experiment_start,
            )
    elif dataset == "ffum":
        if (contract.get("rig_registration") or {}).get("active"):
            raise ValueError("Frozen spatial registration cannot be reconstructed")
        declared = contract.get("rig_path") or contract.get("legacy_rig")
        if not declared:
            raise ValueError("Frozen FF_UM geometry rig is not recorded")
        selected = resolve_review_path(Path(declared))
        if contract.get("rig_path") and contract.get("legacy_rig") and not selected.samefile(
                resolve_review_path(Path(contract["legacy_rig"]))):
            raise ValueError("Frozen FF_UM geometry rig paths disagree")
        optical = anchor.get("method", {}).get("optical_provenance") or {}
        selected_artifacts = ((optical.get("optical_binding") or {}).get("recorded_rig") or {}).get(
            "selected_optical_artifacts") or []
        from darsia.fluidflower.saved_mass_rendering import sha

        for item in selected_artifacts:
            if sha(resolve_review_path(Path(item["path"]))) != item["sha256"]:
                raise ValueError("Frozen FF_UM geometry rig changed: " + str(item["path"]))
        rig = load_shape_only_rig(
            selected, dataset=dataset,
            expected_shape=native["analysis_image_shape_hw"],
            reference_date=experiment.experiment_start,
        )
        for item in selected_artifacts:
            if sha(resolve_review_path(Path(item["path"]))) != item["sha256"]:
                raise ValueError("Frozen FF_UM geometry rig changed while loading")
    else:
        raise ValueError("Geometry-only preview supports registered FF_AC and FF_UM results")

    if not hasattr(rig, "_photometry_rig_folder"):
        rig._photometry_rig_folder = selected
        rig._photometry_rig_fingerprint = serialized_rig_fingerprint(selected)
    return rig
