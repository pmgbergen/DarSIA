import logging
from pathlib import Path

import numpy as np

import darsia
from darsia.presets.workflows.analysis.analysis_context import select_image_paths
from darsia.presets.workflows.basis import label_ids_from_image, select_labels_for_basis
from darsia.presets.workflows.calibration.metadata import (
    read_calibration_metadata,
    validate_basis_metadata,
)
from darsia.presets.workflows.config.fluidflower_config import FluidFlowerConfig
from darsia.presets.workflows.heterogeneous_color_to_mass_analysis import (
    HeterogeneousColorToMassAnalysis,
)
from darsia.presets.workflows.utils.images import load_images_with_cache

logger = logging.getLogger(__name__)


def _load_baseline_color_spectrum_for_color_to_mass(
    *,
    ignore_mode: str,
    baseline_color_spectrum_folder: Path,
    required_labels: set[int],
) -> darsia.LabelColorSpectrumMap | None:
    """Load baseline colour spectrum for color-to-mass calibration if configured."""

    if ignore_mode == "none":
        return None

    if ignore_mode not in ("baseline", "expanded"):
        raise ValueError(
            f"Unsupported ignore_baseline_spectrum mode '{ignore_mode}' in "
            "color-to-mass calibration. Valid modes are: 'none', 'baseline', "
            "'expanded'."
        )

    spectrum_files = list(baseline_color_spectrum_folder.glob("color_spectrum_*.json"))
    if len(spectrum_files) == 0:
        raise FileNotFoundError(
            "Baseline colour spectrum files were not found, but "
            f"ignore_baseline_spectrum='{ignore_mode}' requires them. Expected files "
            f"matching 'color_spectrum_*.json' in {baseline_color_spectrum_folder}. "
            "Run color-path calibration first or set ignore_baseline_spectrum='none'."
        )

    baseline_color_spectrum = darsia.LabelColorSpectrumMap.load(
        baseline_color_spectrum_folder
    )

    missing_labels = sorted(
        required_labels.difference(set(baseline_color_spectrum.keys()))
    )
    if len(missing_labels) > 0:
        raise FileNotFoundError(
            "Baseline colour spectrum is incomplete for color-to-mass calibration. "
            f"Missing labels: {missing_labels}. Folder: "
            f"{baseline_color_spectrum_folder}."
        )

    return baseline_color_spectrum


def calibration_color_to_mass_analysis(
    cls,
    path: Path,
    ref_path: Path | None = None,
    reset: bool = False,
    show: bool = False,
    rois: dict[str, darsia.CoordinateArray] | None = None,
    default: bool = False,
):
    """Calibration of color to mass analysis.

    This function calibrates the color to mass analysis based on the provided configuration.

    Args:
        cls: The class of the rig to be calibrated.
        path: The path to the configuration file.
        ref_path: The path to the reference configuration file (if any).
        reset: Whether to reset existing calibration data.
        show: Whether to perform a final test run to demonstrate the calibration results.
        rois: Regions of interest for calibration (if any).
        default: Whether to perform default calibration without interactive steps.

    """
    # ! ---- LOAD RUN AND RIG ----

    config = FluidFlowerConfig(path, require_data=True, require_results=False)
    config.check("color_paths", "rig", "data", "protocol", "color_to_mass")

    # Mypy type checking
    assert config.data is not None
    assert config.protocol is not None
    assert config.color_paths is not None
    assert config.rig is not None
    assert config.color_to_mass is not None

    # ! ---- LOAD EXPERIMENT ----
    experiment = darsia.ProtocolledExperiment.init_from_config(config)

    # ! ---- LOAD RIG ----
    fluidflower = cls.load(config.rig.path)
    fluidflower.load_experiment(experiment)

    # ! ---- LOAD COLOR PATHS ----

    selected_basis, selected_labels = select_labels_for_basis(
        fluidflower, config.color_to_mass.basis
    )
    current_label_ids = label_ids_from_image(selected_labels)

    color_paths_calibration_file = config.color_paths.calibration_file
    color_paths_metadata = read_calibration_metadata(
        color_paths_calibration_file / "metadata.json"
    )
    validate_basis_metadata(
        metadata=color_paths_metadata,
        expected_basis=selected_basis,
        expected_label_ids=current_label_ids,
        artifact="color_paths",
    )

    color_paths = darsia.LabelColorPathMap.load(color_paths_calibration_file)

    # Pick a reference color path - merely for visualization
    reference_label = config.color_paths.reference_label
    reference_color_path = color_paths[reference_label]
    custom_cmap = reference_color_path.get_color_map()
    if show and False:
        reference_color_path.show_path()

    baseline_color_spectrum = _load_baseline_color_spectrum_for_color_to_mass(
        ignore_mode=config.color_paths.ignore_baseline_spectrum,
        baseline_color_spectrum_folder=config.color_paths.baseline_color_spectrum_folder,
        required_labels=set(color_paths.keys()),
    )

    # ! ---- LOAD IMAGES ----
    calibration_image_paths = select_image_paths(
        config, experiment, all=False, sub_config=config.color_to_mass
    )

    # Cache calibration images for performance
    calibration_images: list[darsia.Image] = load_images_with_cache(
        rig=fluidflower,
        paths=calibration_image_paths,
        use_cache=config.data.use_cache,
        cache_dir=config.data.cache,
    )

    # ! ---- ALLOCATE EMPTY INTERPOLATIONS ----

    color_path_interpolation = {
        label: darsia.ColorPathInterpolation(
            color_path=color_path,
            color_mode=darsia.ColorMode.RELATIVE,
            values=color_path.relative_distances,
            ignore_spectrum=(
                baseline_color_spectrum[label]
                if baseline_color_spectrum is not None
                else None
            ),
        )
        for label, color_path in color_paths.items()
    }

    # ! ---- DEACTIVATE INSENSITIVE COLOR PATHS ----

    # TODO move this to another calibration function.

    # Util 1.
    threshold = config.color_to_mass.threshold

    # Determine distance from color path to baseline spectrum (consider the furthest
    # away color to measure sensitivity)
    if baseline_color_spectrum is None:
        distances = {label: 1.0 for label in color_paths}
        reference_distance = 1.0
    else:
        distances = {
            label: max(
                [
                    float(baseline_color_spectrum[label].distance(c))
                    for c in color_path.colors
                ]
            )
            for label, color_path in color_paths.items()
        }
        reference_distance = max(distances.values())

    # Decide which labels to ignore based on the two metrics
    ignore_labels = []
    for label in np.unique(selected_labels.img):
        relative_distance = distances[label] / reference_distance
        if relative_distance < threshold:
            ignore_labels.append(label)
    logger.info(f"\033[93mIgnoring labels based on distance: {ignore_labels}\033[0m")

    # Illustrate the ignored labels for the calibration images through grayscaling
    if show and len(ignore_labels) > 0:
        for img in calibration_images[-1:]:
            _img = img.copy()
            for mask, label in darsia.Masks(selected_labels, return_label=True):
                if label not in ignore_labels:
                    continue
                _img.img[mask.img] = np.mean(_img.img[mask.img], axis=1, keepdims=True)
            _img.show(cmap=custom_cmap, title="Ignored labels")

    # Util 2. Adapt the interpolation values based on the reference color path

    # Rescale color paths based on reference interpolation.
    # Determine distance from color path to reference color path.
    reference_interpolation = color_path_interpolation[reference_label]
    interpolation_values = {
        label: max(
            [
                max(0.0, float(reference_interpolation(c)))
                for c in color_path.relative_colors
            ]
        )
        for label, color_path in color_paths.items()
    }
    for label in color_path_interpolation:
        if interpolation_values[label] > 0:
            color_path_interpolation[label].values *= interpolation_values[label]

    # Overwrite the color paths with updated interpolation values
    for label in np.unique(selected_labels.img):
        if label in config.color_paths.ignore_labels or label in ignore_labels:
            color_path_interpolation[label] = color_path_interpolation[reference_label]

    # ! ---- COLOR PATH INTERPRETATION ---- ! #

    color_path_interpretation = {
        label: darsia.ColorPathInterpolation(
            color_path=color_path,
            color_mode=darsia.ColorMode.RELATIVE,
            values=color_path.equidistant_distances,
            ignore_spectrum=(
                baseline_color_spectrum[label]
                if baseline_color_spectrum is not None
                else None
            ),
        )
        for label, color_path in color_paths.items()
    }

    # ! ---- SIGNAL FUNCTIONS ---- ! #

    # ! ---- POROSITY-INFORMED AVERAGING ---- ! #
    # TODO: holes in the segmentation?
    image_porosity = fluidflower.image_porosity
    restoration = darsia.VolumeAveraging(
        rev=darsia.REV(size=0.005, img=fluidflower.baseline),
        mask=image_porosity,
        # labels=fluidflower.labels,
    )

    # ! ---- FROM COLOR PATH TO MASS ----

    experiment_start = experiment.experiment_start
    state = experiment.pressure_temperature_protocol.get_state(experiment_start)
    gradient = experiment.pressure_temperature_protocol.get_gradient(experiment_start)
    co2_mass_analysis = darsia.CO2MassAnalysis(
        baseline=fluidflower.baseline,
        atmospheric_pressure=state.pressure,
        atmospheric_temperature=state.temperature,
        atmospheric_pressure_gradient=gradient.pressure,
        atmospheric_temperature_gradient=gradient.temperature,
    )

    # Fetch signal functions from reference calibration.
    if not reset and ref_path is not None:
        # Start from reference calibration if available
        ref_config = FluidFlowerConfig(
            ref_path, require_data=False, require_results=False
        )
        assert (
            ref_config.color_to_mass.calibration_folder.exists()
        ), "Reference calibration folder does not exist."
        color_analysis = HeterogeneousColorToMassAnalysis.load(
            folder=ref_config.color_to_mass.calibration_folder,
            baseline=fluidflower.baseline,
            labels=selected_labels,
            co2_mass_analysis=co2_mass_analysis,
            geometry=fluidflower.geometry,
            restoration=restoration,
            basis=selected_basis,
        )
        color_analysis.color_path_interpretation = color_path_interpretation

    elif not reset and config.color_to_mass.calibration_folder.exists():
        # Start from existing calibration if available
        color_analysis = HeterogeneousColorToMassAnalysis.load(
            folder=config.color_to_mass.calibration_folder,
            baseline=fluidflower.baseline,
            labels=selected_labels,
            co2_mass_analysis=co2_mass_analysis,
            geometry=fluidflower.geometry,
            restoration=restoration,
            basis=selected_basis,
        )
    else:
        # Start from scratch
        signal_functions = {}
        for label in color_path_interpolation:
            if label in config.color_paths.ignore_labels or label in ignore_labels:
                signal_functions[label] = darsia.PWTransformation(
                    color_paths[reference_label].equidistant_distances,
                    np.zeros(len(color_paths[reference_label].equidistant_distances)),
                )
                continue
            try:
                signal_functions[label] = darsia.PWTransformation(
                    color_paths[label].equidistant_distances,
                    color_path_interpolation[label].values,
                )
            except Exception as e:
                signal_functions[label] = darsia.PWTransformation(
                    color_paths[reference_label].equidistant_distances,
                    np.zeros(len(color_paths[reference_label].equidistant_distances)),
                )
                ignore_labels.append(label)
                logger.warning(f"Error processing label {label}: {e}")
        flash = darsia.SimpleFlash(
            min_value_aq=0,
            max_value_aq=0.75,
            min_value_g=0.75,
            max_value_g=1.0,
            restoration=None,
        )
        color_analysis = HeterogeneousColorToMassAnalysis(
            baseline=fluidflower.baseline,
            labels=selected_labels,
            color_mode=darsia.ColorMode.RELATIVE,
            color_path_interpretation=color_path_interpretation,
            signal_functions=signal_functions,
            flash=flash,
            co2_mass_analysis=co2_mass_analysis,
            geometry=fluidflower.geometry,
            restoration=restoration,
            ignore_labels=config.color_paths.ignore_labels + ignore_labels,
            basis=selected_basis,
        )

    # ! ---- INTERACTIVE CALIBRATION ---- ! #

    rois = rois or {}
    rois.update(
        {
            "full": darsia.CoordinateArray(
                [fluidflower.baseline.origin, fluidflower.baseline.opposite_corner]
            )
        }
    )

    # Perform local calibration
    if default:
        logger.info(
            "\033[93mSkipping interactive calibration. "
            "Using default signal functions.\033[0m"
        )
    else:
        color_analysis.manual_calibration(
            images=calibration_images,
            experiment=experiment,
            rois=rois,
            cmap=custom_cmap,
        )

    for label in np.unique(selected_labels.img):
        if label in config.color_paths.ignore_labels or label in ignore_labels:
            color_paths[label] = color_paths[reference_label]

    # Store calibration
    color_analysis.save(config.color_to_mass.calibration_folder)

    # Test run
    if show:
        for img in calibration_images:
            mass = color_analysis(img).mass
            mass.show(title=f"Mass image {img.time}", cmap=custom_cmap, delay=False)
