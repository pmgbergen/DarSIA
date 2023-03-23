"""
Module containing presets for concentration analyses as used for
analyzing the benchmark experiments.

"""

import numpy as np

import darsia


def benchmark_binary_cleaning_preset(
    base: darsia.GeneralImage, options: dict
) -> darsia.CombinedModel:
    """
    Cleaning methods also used in the benchmark_concentration_analysis_preset.

    Args:
        base (darsia.GeneralImage): baseline image
        options (dict): options same as in benchmark_concentration_analysis_preset.

    """
    original_size = base.img.shape[:2]
    binary_cleaning = darsia.CombinedModel(
        [
            # Binary inpainting
            darsia.BinaryRemoveSmallObjects(key="prior ", **options),
            darsia.BinaryFillHoles(key="prior ", **options),
            # Resize and Smoothing
            darsia.Resize(dtype=np.float32, key="prior ", **options),
            darsia.TVD(key="prior ", **options),
            darsia.Resize(dsize=tuple(reversed(original_size))),
            # Conversion to boolean
            darsia.StaticThresholdModel(0.5),
        ]
    )

    return binary_cleaning


def benchmark_concentration_analysis_preset(
    base: darsia.GeneralImage, labels: np.ndarray, options: dict
) -> darsia.PriorPosteriorConcentrationAnalysis:
    """
    The strategy for identifying any phase is constructed as a pipeline
    of the following steps:

    1. Use monochromatic signal reduction
    2. Restoration (upscaling) of signal
    3. Prior strategy providing a first detection.
        a. Thresholding.
        b. binary inpainting
        c. resizing and smoothing
        d. conversion to boolean data
    4. Posterior strategy reviewing the first three steps.

    Args:
        base (darsia.GeneralImage): baseline image
        labels (np.ndarray): labeling of domain in facies
        options (dict): dictionary holding all tuning parameters

    Returns:
        darsia.ConcentrationAnalysis: concentration analysis for detecting CO2.

    """

    ########################################################################
    # Define signal reduction
    signal_reduction = darsia.MonochromaticReduction(**options)

    ########################################################################
    # Treat all facies the same
    balancing = None

    ########################################################################
    # Define restoration object - coarsen, tvd, resize
    original_size = base.img.shape[:2]
    restoration = darsia.CombinedModel(
        [
            darsia.Resize(key="restoration ", **options),
            darsia.TVD(key="restoration ", **options),
            darsia.Resize(dsize=tuple(reversed(original_size))),
        ]
    )

    ########################################################################
    # Combine the three models as prior:
    # 1. Thresholding
    # 2. Binary cleaning

    # Prior model
    prior_model = darsia.CombinedModel(
        [
            # Thresholding
            darsia.ThresholdModel(labels, key="prior ", **options),
            # Binary cleaning
            benchmark_binary_cleaning_preset(base, options),
        ]
    )

    ########################################################################
    # Define a posterior model
    posterior_model = darsia.BinaryDataSelector(key="posterior ", **options)

    ########################################################################
    # Combine all to define a concentration analysis object for CO2.
    concentration_analysis = darsia.PriorPosteriorConcentrationAnalysis(
        base,
        signal_reduction,
        balancing,
        restoration,
        prior_model,
        posterior_model,
        labels,
        **options,
    )

    return concentration_analysis
