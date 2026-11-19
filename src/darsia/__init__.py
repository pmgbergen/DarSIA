"""DarSIA provides tools for quantitative image processing with support for various and
common data types in computational and laboratory porous media research.

"""

# The modules of darsia are imported on first use of one of their names (PEP 562):
# `import darsia` is cheap, `darsia.Image` imports `darsia.image.image`. A new public
# name has to be added to the tables below to be available as `darsia.<name>`.

import importlib

_ATTRIBUTES: dict[str, tuple[str, ...]] = {
    # module: public names defined (or re-exported) there
    "darsia.image.coordinatesystem": (
        "CoordinateSystem",
        "annotations",
        "check_equal_coordinatesystems",
    ),
    "darsia.image.image": (
        "ExtensiveImage",
        "Image",
        "OpticalImage",
        "ScalarImage",
    ),
    "darsia.image.indexing": (
        "cartesianToMatrixIndexing",
        "interpret_indexing",
        "matrixToCartesianIndexing",
        "to_cartesian_indexing",
        "to_matrix_indexing",
    ),
    "darsia.image.patches": ("Patches",),
    "darsia.image.imread": (
        "imread",
        "imread_from_bytes",
        "imread_from_dicom",
        "imread_from_npz",
        "imread_from_numpy",
        "imread_from_optical",
        "imread_from_vtu",
    ),
    "darsia.image.arithmetics": (
        "stack",
        "superpose",
        "weight",
    ),
    "darsia.measure.integration": (
        "ExtrudedGeometry",
        "ExtrudedPorousGeometry",
        "Geometry",
        "PorousGeometry",
        "WeightedGeometry",
    ),
    "darsia.utils.convergence_status": ("ConvergenceStatus",),
    "darsia.measure.emd": ("EMD",),
    "darsia.measure.beckmann_linalg": (
        "BeckmannAMGSolver",
        "BeckmannCGSolver",
        "BeckmannDirectSolver",
        "BeckmannKSPFieldSplitSolver",
        "BeckmannKSPSolver",
        "BeckmannLinearSolver",
        "BeckmannLinearSolverFactory",
        "BeckmannLinearSolverType",
    ),
    "darsia.measure.beckmann_problem": (
        "BeckmannProblem",
        "L1Mode",
        "MobilityMode",
    ),
    "darsia.measure.beckmann_convergence_criteria": ("BeckmannConvergenceCriteria",),
    "darsia.measure.beckmann_convergence_history": ("BeckmannConvergenceHistory",),
    "darsia.measure.beckmann_newton_solver": ("BeckmannNewtonSolver",),
    "darsia.measure.beckmann_bregman_solver": ("BeckmannBregmanSolver",),
    "darsia.measure.beckmann_gprox_solver": ("BeckmannGproxPGHDSolver",),
    "darsia.measure.wasserstein": (
        "wasserstein_distance",
        "wasserstein_distance_to_vtk",
    ),
    "darsia.measure.norms": (
        "distance",
        "h1_distance",
        "h1_semi_norm",
        "l1_distance",
        "l2_distance",
    ),
    "darsia.utils.timings": (
        "Timings",
        "timed",
        "timing_decorator",
        "timings",
    ),
    "darsia.single_image_analysis.contour_smoother": (
        "ContourSmoother",
        "ContourSmootherSequence",
        "GaussianSmoother",
        "MovingAverageSmoother",
        "PolyDPSmoother",
        "SavitzkyGolaySmoother",
    ),
    "darsia.utils.arithmetics": ("array_product",),
    "darsia.utils.point": (
        "BasePoint",
        "Coordinate",
        "CoordinateArray",
        "Voxel",
        "VoxelArray",
        "VoxelCenter",
        "VoxelCenterArray",
        "make_coordinate",
        "make_voxel",
        "make_voxel_center",
        "to",
        "to_coordinate",
        "to_voxel",
        "to_voxel_center",
    ),
    "darsia.utils.sort": ("sort_quad",),
    "darsia.utils.box": (
        "bounding_box",
        "bounding_box_inverse",
        "perimeter",
        "random_patches",
    ),
    "darsia.utils.interpolation": (
        "illumination_interpolation",
        "interpolate_measurements_2d",
        "interpolate_to_image",
        "interpolate_to_image_from_csv",
        "polynomial_interpolation",
    ),
    "darsia.utils.segmentation": (
        "group_labels",
        "label_image",
        "make_consecutive",
        "reassign_labels",
        "segment",
    ),
    "darsia.utils.coloranalysis": ("hsv_spectrum",),
    "darsia.utils.array_slice": (
        "array_slice",
        "array_slice_argument",
    ),
    "darsia.utils.derivatives": (
        "backward_diff",
        "forward_diff",
        "laplace",
    ),
    "darsia.utils.linear_solvers.solver": ("Solver",),
    "darsia.utils.linear_solvers.jacobi": ("Jacobi",),
    "darsia.utils.linear_solvers.cg": ("CG",),
    "darsia.utils.linear_solvers.mg": ("MG",),
    "darsia.utils.andersonacceleration": ("AndersonAcceleration",),
    "darsia.utils.dtype": ("convert_dtype",),
    "darsia.utils.formats": ("Format",),
    "darsia.utils.grid": (
        "Grid",
        "generate_grid",
    ),
    "darsia.utils.fv": (
        "FVDivergence",
        "FVFullFaceReconstruction",
        "FVMass",
        "FVTangentialFaceReconstruction",
        "cell_to_face_average",
        "face_to_cell",
    ),
    "darsia.utils.kernels": (
        "BaseKernel",
        "GaussianKernel",
        "LinearKernel",
        "kernel_from_dict",
    ),
    "darsia.utils.extractcharacteristicdata": ("extract_characteristic_data",),
    "darsia.utils.masks": (
        "Masks",
        "MasksIterator",
    ),
    "darsia.utils.detection": (
        "detect_closest_point",
        "detect_color",
        "detect_value",
        "monochromatic_concentration_analysis",
        "orthogonal_colors",
    ),
    "darsia.utils.standard_images": (
        "StandardDtype",
        "TYPE_CHECKING",
        "full_like",
        "ones_like",
        "roi_to_mask",
        "zeros_like",
    ),
    "darsia.utils.approximations": (
        "ApproximationSpace",
        "LinearApproximation",
        "PolynomialApproximationSpace",
        "RadialPolynomialApproximationSpace",
    ),
    "darsia.utils.slices": (
        "add_slice_pairs",
        "add_slices",
        "subtract_slice_pairs",
        "subtract_slices",
    ),
    "darsia.image.subregions": (
        "InterpolationOption",
        "extract_quadrilateral_ROI",
        "restrict_image_to_landmarks",
    ),
    "darsia.image.roi": ("ROI",),
    "darsia.corrections.basecorrection": ("BaseCorrection",),
    "darsia.corrections.typecorrection": ("TypeCorrection",),
    "darsia.corrections.shape.transformation": (
        "BaseTransformation",
        "TransformationCorrection",
    ),
    "darsia.corrections.shape.curvature": (
        "CurvatureCorrection",
        "load_curvature_correction_config_from_dict",
        "load_curvature_correction_config_from_toml",
    ),
    "darsia.corrections.shape.affine": (
        "AffineCorrection",
        "AffineTransformation",
    ),
    "darsia.corrections.shape.translation": (
        "TranslationCorrection",
        "TranslationEstimator",
    ),
    "darsia.corrections.shape.piecewiseperspective": ("PiecewisePerspectiveTransform",),
    "darsia.corrections.shape.generalizedperspective": (
        "GeneralizedPerspectiveCorrection",
        "GeneralizedPerspectiveTransformation",
    ),
    "darsia.corrections.shape.rotation": ("RotationCorrection",),
    "darsia.corrections.shape.drift": ("DriftCorrection",),
    "darsia.corrections.shape.deformation": ("DeformationCorrection",),
    "darsia.corrections.shape.resize": ("ResizeCorrection",),
    "darsia.corrections.color.colorbalance": (
        "AdaptiveBalance",
        "AffineBalance",
        "BaseBalance",
        "ColorBalance",
        "WhiteBalance",
        "affine_balance",
        "color_balance",
        "white_balance",
    ),
    "darsia.corrections.color.colorcheckerfinder": (
        "ColorCheckerPosition",
        "find_colorchecker",
    ),
    "darsia.corrections.color.illuminationcorrection": ("IlluminationCorrection",),
    "darsia.corrections.color.patchwiseilluminationcorrection": (
        "PatchwiseIlluminationCorrection",
    ),
    "darsia.corrections.color.dynamicilluminationcorrection": (
        "DynamicIlluminationCorrection",
    ),
    "darsia.corrections.color.colorcorrection": (
        "ColorChecker",
        "ColorCheckerAfter2014",
        "ColorCorrection",
        "CustomColorChecker",
        "colour",
    ),
    "darsia.corrections.color.relativecolorcorrection": ("RelativeColorCorrection",),
    "darsia.corrections.color.experimentalcolorcorrection": (
        "ClassicColorChecker",
        "EOTF",
        "ExperimentalColorCorrection",
    ),
    "darsia.image.coordinatetransformation": (
        "CoordinateTransformation",
        "lir",
    ),
    "darsia.signals.reduction.signalreduction": ("SignalReduction",),
    "darsia.signals.reduction.monochromatic": ("MonochromaticReduction",),
    "darsia.signals.reduction.dimensionreduction": (
        "AxisReduction",
        "extrude_along_axis",
        "reduce_axis",
    ),
    "darsia.signals.color.color_mode": ("ColorMode",),
    "darsia.signals.color.color_range": ("ColorRange",),
    "darsia.signals.color.discrete_color_range": (
        "DiscreteColorRange",
        "NUMBA_AVAILABLE",
        "color_to_index_numba",
        "flatten_index_numba",
    ),
    "darsia.signals.color.color_spectrum": ("ColorSpectrum",),
    "darsia.signals.color.color_path": (
        "ColorPath",
        "define_color_path",
    ),
    "darsia.signals.color.label_color_spectrum_map": ("LabelColorSpectrumMap",),
    "darsia.signals.color.label_color_map": ("LabelColorMap",),
    "darsia.signals.color.label_color_path_map": ("LabelColorPathMap",),
    "darsia.signals.color.color_path_regression": (
        "ColorPathInputs",
        "LabelColorPathMapRegression",
    ),
    "darsia.signals.models.basemodel": (
        "HeterogeneousModel",
        "Model",
    ),
    "darsia.signals.models.combinedmodel": ("CombinedModel",),
    "darsia.signals.models.linearmodel": (
        "HeterogeneousLinearModel",
        "LinearModel",
        "ScalingModel",
    ),
    "darsia.signals.models.clipmodel": ("ClipModel",),
    "darsia.signals.models.thresholdmodel": ("ThresholdModel",),
    "darsia.signals.models.staticthresholdmodel": ("StaticThresholdModel",),
    "darsia.signals.models.dynamicthresholdmodel": (
        "DynamicThresholdModel",
        "GlobalMinTwoPeakHistogrammAnalysis",
        "HistogrammBasedThresholding",
        "OtsuTwoPeakHistogrammAnalysis",
        "StandardOtsu",
        "TwoPeakHistogrammAnalysis",
    ),
    "darsia.signals.models.binarydataselector": (
        "BaseCriterion",
        "BinaryDataSelector",
        "CombinedCriterion",
        "GradientModulusCriterion",
        "RelativeValueCriterion",
        "TransformedValueCriterion",
        "ValueCriterion",
    ),
    "darsia.signals.models.kernelinterpolation": (
        "AdvancedKernelInterpolation",
        "KernelInterpolation",
    ),
    "darsia.signals.models.pwtransformation": (
        "PWTransformation",
        "PchipTransformation",
        "monotonicity_message",
        "monotonicity_violation",
    ),
    "darsia.signals.models.evaluation": (
        "EvaluationMode",
        "KernelEvaluation",
        "KernelSignalFunction",
        "LinearEvaluation",
        "SplineEvaluation",
        "curve_inverse",
        "select_affine_supports",
        "signal_curve",
    ),
    "darsia.signals.models.color_path_interpolation": (
        "ColorPathFunction",
        "ColorPathInterpolation",
        "IGNORE_MODES",
    ),
    "darsia.restoration.tvd": (
        "TVD",
        "tvd",
    ),
    "darsia.restoration.median": ("Median",),
    "darsia.restoration.resize": (
        "Resize",
        "SINGLE_CALL_CHANNELS",
        "equalize_voxel_size",
        "resize",
        "uniform_refinement",
    ),
    "darsia.restoration.binaryinpaint": (
        "BinaryFillHoles",
        "BinaryLocalConvexCover",
        "BinaryRemoveSmallObjects",
    ),
    "darsia.restoration.h1_regularization": ("H1_regularization",),
    "darsia.restoration.split_bregman_tvd": ("split_bregman_tvd",),
    "darsia.restoration.averaging": (
        "REV",
        "VolumeAveraging",
        "porosity_based_averaging",
        "volume_average",
    ),
    "darsia.multi_image_analysis.translationanalysis": ("TranslationAnalysis",),
    "darsia.multi_image_analysis.concentrationanalysis": (
        "ConcentrationAnalysis",
        "PriorPosteriorConcentrationAnalysis",
    ),
    "darsia.multi_image_analysis.model_calibration": (
        "AbsoluteVolumeModelObjectiveMixin",
        "AbstractModelObjective",
        "InjectionRateModelObjectiveMixin",
    ),
    "darsia.multi_image_analysis.balancing_calibration": (
        "AbstractBalancingCalibration",
        "ContinuityBasedBalancingCalibrationMixin",
    ),
    "darsia.multi_image_analysis.imageregistration": (
        "DiffeomorphicImageRegistration",
        "ImageRegistration",
        "MultiscaleDiffeomorphicImageRegistration",
    ),
    "darsia.multi_image_analysis.segmentationcomparison": ("SegmentationComparison",),
    "darsia.single_image_analysis.contouranalysis": (
        "ContourAnalysis",
        "contour_length",
    ),
    "darsia.manager.analysisbase": ("AnalysisBase",),
    "darsia.manager.concentrationanalysisbase": ("ConcentrationAnalysisBase",),
    "darsia.manager.traceranalysis": ("TracerAnalysis",),
    "darsia.manager.co2analysis": ("CO2Analysis",),
    "darsia.assistants.base_assistant": ("BaseAssistant",),
    "darsia.assistants.point_selection_assistant": ("PointSelectionAssistant",),
    "darsia.assistants.box_selection_assistant": ("BoxSelectionAssistant",),
    "darsia.assistants.rectangle_selection_assistant": ("RectangleSelectionAssistant",),
    "darsia.assistants.rotation_correction_assistant": ("RotationCorrectionAssistant",),
    "darsia.assistants.subregion_assistant": ("SubregionAssistant",),
    "darsia.assistants.crop_assistant": ("CropAssistant",),
    "darsia.assistants.labels_assistant": (
        "LabelsAssistant",
        "LabelsAssistantMenu",
        "LabelsMaskSelectionAssistant",
        "LabelsMergeAssistant",
        "LabelsPickAssistant",
        "LabelsSegmentAssistant",
        "MonochromaticAssistant",
    ),
    "darsia.pvt": ("FLUIDS",),
    "darsia.multiphase.flash": (
        "AdvancedFlash",
        "Flash",
        "IdentityFlash",
        "SimpleFlash",
        "load_flash",
        "new_flash",
    ),
    "darsia.multiphase.mass_analysis": (
        "AdvancedCO2MassAnalysis",
        "CO2MassAnalysis",
        "EPSILON",
        "FluidMassAnalysis",
        "GRAVITY",
        "HydrostaticState",
        "MassAnalysisResults",
        "PA2BAR",
        "ThresholdAnalysisResults",
        "TracerMassAnalysis",
    ),
    "darsia.multiphase.fluidflower_co2_meta": ("FluidFlowerCO2Meta",),
    "darsia.multiphase.multiphase_time_series_data": (
        "MultiphaseTimeSeriesData",
        "TimeSeriesData",
    ),
    "darsia.multiphase.multiphase_time_series_analysis": (
        "MultiphaseTimeSeriesAnalysis",
    ),
    "darsia.experiment.experiment": (
        "Experiment",
        "ProtocolledExperiment",
    ),
    "darsia.experiment.protocols": (
        "ImagingInterval",
        "ImagingProtocol",
        "ImagingProtocolOld",
        "InjectionProtocol",
        "PressureTemperatureProtocol",
        "ThermodynamicState",
        "logger",
    ),
    "darsia.corrections.readcorrection": (
        "AnyCorrection",
        "read_correction",
    ),
    "darsia.presets.workflows.simple_run_analysis": (
        "SimpleMassAnalysisResults",
        "SimpleMultiphaseTimeSeriesData",
        "SimpleRunAnalysis",
    ),
    "_bisect": ("bisect_left",),
    "_warnings": ("warn",),
    "abc": (
        "ABC",
        "abstractmethod",
    ),
    "collections": ("namedtuple",),
    "colour_checker_detection.detection.segmentation": (
        "detect_colour_checkers_segmentation",
    ),
    "darsia.image.conversion_kernels": (
        "to_float64",
        "to_ubyte",
    ),
    "darsia.presets.workflows.config.fluidflower_config": ("FluidFlowerConfig",),
    "darsia.presets.workflows.config.time_data": ("TimeWindow",),
    "darsia.pvt.fluid": (
        "Fluid",
        "get_fluid",
    ),
    "darsia.pvt.system": ("FluidSystem",),
    "darsia.pvt.table": ("PropertyTable",),
    "darsia.signals.color.utils": ("get_mean_color",),
    "darsia.utils.augmented_plotting": ("plot_contour_on_image",),
    "darsia.utils.features": ("FeatureDetection",),
    "dataclasses": (
        "dataclass",
        "field",
    ),
    "datetime": (
        "datetime",
        "timedelta",
    ),
    "enum": (
        "Enum",
        "StrEnum",
    ),
    "itertools": ("combinations",),
    "math": ("ceil",),
    "matplotlib.colors": ("LinearSegmentedColormap",),
    "matplotlib.widgets": ("RectangleSelector",),
    "mpl_toolkits.mplot3d.art3d": ("Poly3DCollection",),
    "mpl_toolkits.mplot3d.axes3d": ("Axes3D",),
    "numba.core.decorators": (
        "jit",
        "njit",
    ),
    "operator": ("itemgetter",),
    "pathlib._local": ("Path",),
    "plotly.subplots": ("make_subplots",),
    "scipy.interpolate._ndgriddata": ("NearestNDInterpolator",),
    "scipy.interpolate._rbfinterp": ("RBFInterpolator",),
    "scipy.ndimage._interpolation": ("map_coordinates",),
    "scipy.optimize._lsq.least_squares": ("least_squares",),
    "scipy.signal._peak_finding": ("find_peaks",),
    "scipy.signal._savitzky_golay": ("savgol_filter",),
    "scipy.sparse.linalg._interface": ("LinearOperator",),
    "scipy.spatial.transform._rotation": ("Rotation",),
    "scipy.stats._stats_py": ("hmean",),
    "shapely.geometry.point": ("Point",),
    "shapely.geometry.polygon": ("Polygon",),
    "sklearn.decomposition._pca": ("PCA",),
    "sklearn.linear_model._base": ("LinearRegression",),
    "sklearn.linear_model._ransac": ("RANSACRegressor",),
    "sklearn.manifold._locally_linear": ("LocallyLinearEmbedding",),
    "subprocess": ("check_output",),
    "typing": (
        "Any",
        "Callable",
        "ClassVar",
        "List",
        "Literal",
        "Optional",
        "Sequence",
        "Tuple",
        "Union",
        "cast",
        "overload",
        "override",
    ),
}
"""Public names of darsia, by the module providing them."""

_MODULES: dict[str, str] = {
    "PIL_Image": "PIL.Image",
    "abc": "abc",
    "augmented_plotting": "darsia.utils.augmented_plotting",
    "color_spectrum_kernels": "darsia.signals.color.color_spectrum_kernels",
    "colour_checker_detection": "colour_checker_detection",
    "copy": "copy",
    "cv2": "cv2",
    "da": "darsia",
    "darsia": "darsia",
    "go": "plotly.graph_objects",
    "interpolate": "scipy.interpolate",
    "io": "io",
    "itertools": "itertools",
    "json": "json",
    "linalg": "darsia.utils.linalg",
    "logging": "logging",
    "math": "math",
    "mpatches": "matplotlib.patches",
    "ndi": "scipy.ndimage",
    "np": "numpy",
    "numba": "numba",
    "optimize": "scipy.optimize",
    "patches": "matplotlib.patches",
    "pd": "pandas",
    "platform": "platform",
    "plotting": "darsia.utils.plotting",
    "plt": "matplotlib.pyplot",
    "px": "plotly.express",
    "pyamg": "pyamg",
    "pydicom": "pydicom",
    "quadrature": "darsia.utils.quadrature",
    "scipy": "scipy",
    "skimage": "skimage",
    "sp": "scipy",
    "sps": "scipy.sparse",
    "time": "time",
    "tomllib": "tomllib",
    "tracemalloc": "tracemalloc",
    "warnings": "warnings",
}
"""Modules available as attributes of darsia (e.g. `darsia.linalg`)."""

_ALIASES: dict[str, tuple[str, str]] = {
    "Contour": ("numpy", "ndarray"),
    "tm": ("time", "time"),
}
"""Names bound to an attribute of another name or module."""

_LAZY: dict[str, tuple[str, str | None]] = {
    **{name: (module, name) for module, names in _ATTRIBUTES.items() for name in names},
    **{name: (module, None) for name, module in _MODULES.items()},
    **_ALIASES,
}
"""Module and attribute (None: the module itself) of each public name."""

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    """Import the module providing `name` on first use (afterwards a plain lookup).

    Subpackages (e.g. `darsia.presets`) are available as attributes as well.
    """
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name not in _LAZY:
        try:
            return importlib.import_module(f"{__name__}.{name}")
        except ModuleNotFoundError as error:
            if error.name != f"{__name__}.{name}":
                raise
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
    module, attribute = _LAZY[name]
    value = importlib.import_module(module)
    if attribute is not None:
        value = getattr(value, attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
