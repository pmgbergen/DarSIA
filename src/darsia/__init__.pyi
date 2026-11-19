"""Names of `darsia` for type checkers and IDEs; at runtime they are imported on
first use (see `__init__.py`). Keep in sync with the tables in `__init__.py`."""

import abc as abc
import copy as copy
import io as io
import itertools as itertools
import json as json
import logging as logging
import math as math
import platform as platform
import time as time
import tomllib as tomllib
import tracemalloc as tracemalloc
import warnings as warnings
from _bisect import bisect_left as bisect_left
from _warnings import warn as warn
from abc import ABC as ABC
from abc import abstractmethod as abstractmethod
from collections import namedtuple as namedtuple
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from datetime import datetime as datetime
from datetime import timedelta as timedelta
from enum import Enum as Enum
from enum import StrEnum as StrEnum
from itertools import combinations as combinations
from math import ceil as ceil
from operator import itemgetter as itemgetter
from pathlib._local import Path as Path
from subprocess import check_output as check_output
from time import time as tm
from typing import Any as Any
from typing import Callable as Callable
from typing import ClassVar as ClassVar
from typing import List as List
from typing import Literal as Literal
from typing import Optional as Optional
from typing import Sequence as Sequence
from typing import Tuple as Tuple
from typing import Union as Union
from typing import cast as cast
from typing import overload as overload
from typing import override as override

import colour_checker_detection as colour_checker_detection
import cv2 as cv2
import numba as numba
import numpy as np
import pandas as pd
import pyamg as pyamg
import pydicom as pydicom
import scipy as scipy
import scipy as sp
import skimage as skimage
from colour_checker_detection.detection.segmentation import (
    detect_colour_checkers_segmentation as detect_colour_checkers_segmentation,
)
from matplotlib import patches as mpatches
from matplotlib import patches as patches
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap as LinearSegmentedColormap
from matplotlib.widgets import RectangleSelector as RectangleSelector
from mpl_toolkits.mplot3d.art3d import Poly3DCollection as Poly3DCollection
from mpl_toolkits.mplot3d.axes3d import Axes3D as Axes3D
from numba.core.decorators import jit as jit
from numba.core.decorators import njit as njit
from numpy import ndarray as Contour
from PIL import Image as PIL_Image
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots as make_subplots
from scipy import interpolate as interpolate
from scipy import ndimage as ndi
from scipy import optimize as optimize
from scipy import sparse as sps
from scipy.interpolate._ndgriddata import NearestNDInterpolator as NearestNDInterpolator
from scipy.interpolate._rbfinterp import RBFInterpolator as RBFInterpolator
from scipy.ndimage._interpolation import map_coordinates as map_coordinates
from scipy.optimize._lsq.least_squares import least_squares as least_squares
from scipy.signal._peak_finding import find_peaks as find_peaks
from scipy.signal._savitzky_golay import savgol_filter as savgol_filter
from scipy.sparse.linalg._interface import LinearOperator as LinearOperator
from scipy.spatial.transform._rotation import Rotation as Rotation
from scipy.stats._stats_py import hmean as hmean
from shapely.geometry.point import Point as Point
from shapely.geometry.polygon import Polygon as Polygon
from sklearn.decomposition._pca import PCA as PCA
from sklearn.linear_model._base import LinearRegression as LinearRegression
from sklearn.linear_model._ransac import RANSACRegressor as RANSACRegressor
from sklearn.manifold._locally_linear import (
    LocallyLinearEmbedding as LocallyLinearEmbedding,
)

import darsia as da
import darsia as darsia

# Subpackages
from darsia import assistants as assistants
from darsia import corrections as corrections
from darsia import experiment as experiment
from darsia import image as image
from darsia import manager as manager
from darsia import measure as measure
from darsia import multi_image_analysis as multi_image_analysis
from darsia import multiphase as multiphase
from darsia import presets as presets
from darsia import pvt as pvt
from darsia import restoration as restoration
from darsia import signals as signals
from darsia import single_image_analysis as single_image_analysis
from darsia import utils as utils
from darsia.assistants.base_assistant import BaseAssistant as BaseAssistant
from darsia.assistants.box_selection_assistant import (
    BoxSelectionAssistant as BoxSelectionAssistant,
)
from darsia.assistants.crop_assistant import CropAssistant as CropAssistant
from darsia.assistants.labels_assistant import LabelsAssistant as LabelsAssistant
from darsia.assistants.labels_assistant import (
    LabelsAssistantMenu as LabelsAssistantMenu,
)
from darsia.assistants.labels_assistant import (
    LabelsMaskSelectionAssistant as LabelsMaskSelectionAssistant,
)
from darsia.assistants.labels_assistant import (
    LabelsMergeAssistant as LabelsMergeAssistant,
)
from darsia.assistants.labels_assistant import (
    LabelsPickAssistant as LabelsPickAssistant,
)
from darsia.assistants.labels_assistant import (
    LabelsSegmentAssistant as LabelsSegmentAssistant,
)
from darsia.assistants.labels_assistant import (
    MonochromaticAssistant as MonochromaticAssistant,
)
from darsia.assistants.point_selection_assistant import (
    PointSelectionAssistant as PointSelectionAssistant,
)
from darsia.assistants.rectangle_selection_assistant import (
    RectangleSelectionAssistant as RectangleSelectionAssistant,
)
from darsia.assistants.rotation_correction_assistant import (
    RotationCorrectionAssistant as RotationCorrectionAssistant,
)
from darsia.assistants.subregion_assistant import (
    SubregionAssistant as SubregionAssistant,
)
from darsia.corrections.basecorrection import BaseCorrection as BaseCorrection
from darsia.corrections.color.colorbalance import AdaptiveBalance as AdaptiveBalance
from darsia.corrections.color.colorbalance import AffineBalance as AffineBalance
from darsia.corrections.color.colorbalance import BaseBalance as BaseBalance
from darsia.corrections.color.colorbalance import ColorBalance as ColorBalance
from darsia.corrections.color.colorbalance import WhiteBalance as WhiteBalance
from darsia.corrections.color.colorbalance import affine_balance as affine_balance
from darsia.corrections.color.colorbalance import color_balance as color_balance
from darsia.corrections.color.colorbalance import white_balance as white_balance
from darsia.corrections.color.colorcheckerfinder import (
    ColorCheckerPosition as ColorCheckerPosition,
)
from darsia.corrections.color.colorcheckerfinder import (
    find_colorchecker as find_colorchecker,
)
from darsia.corrections.color.colorcorrection import ColorChecker as ColorChecker
from darsia.corrections.color.colorcorrection import (
    ColorCheckerAfter2014 as ColorCheckerAfter2014,
)
from darsia.corrections.color.colorcorrection import ColorCorrection as ColorCorrection
from darsia.corrections.color.colorcorrection import (
    CustomColorChecker as CustomColorChecker,
)
from darsia.corrections.color.colorcorrection import colour as colour
from darsia.corrections.color.dynamicilluminationcorrection import (
    DynamicIlluminationCorrection as DynamicIlluminationCorrection,
)
from darsia.corrections.color.experimentalcolorcorrection import EOTF as EOTF
from darsia.corrections.color.experimentalcolorcorrection import (
    ClassicColorChecker as ClassicColorChecker,
)
from darsia.corrections.color.experimentalcolorcorrection import (
    ExperimentalColorCorrection as ExperimentalColorCorrection,
)
from darsia.corrections.color.illuminationcorrection import (
    IlluminationCorrection as IlluminationCorrection,
)
from darsia.corrections.color.patchwiseilluminationcorrection import (
    PatchwiseIlluminationCorrection as PatchwiseIlluminationCorrection,
)
from darsia.corrections.color.relativecolorcorrection import (
    RelativeColorCorrection as RelativeColorCorrection,
)
from darsia.corrections.readcorrection import AnyCorrection as AnyCorrection
from darsia.corrections.readcorrection import read_correction as read_correction
from darsia.corrections.shape.affine import AffineCorrection as AffineCorrection
from darsia.corrections.shape.affine import AffineTransformation as AffineTransformation
from darsia.corrections.shape.curvature import (
    CurvatureCorrection as CurvatureCorrection,
)
from darsia.corrections.shape.curvature import (
    load_curvature_correction_config_from_dict as load_curvature_correction_config_from_dict,
)
from darsia.corrections.shape.curvature import (
    load_curvature_correction_config_from_toml as load_curvature_correction_config_from_toml,
)
from darsia.corrections.shape.deformation import (
    DeformationCorrection as DeformationCorrection,
)
from darsia.corrections.shape.drift import DriftCorrection as DriftCorrection
from darsia.corrections.shape.generalizedperspective import (
    GeneralizedPerspectiveCorrection as GeneralizedPerspectiveCorrection,
)
from darsia.corrections.shape.generalizedperspective import (
    GeneralizedPerspectiveTransformation as GeneralizedPerspectiveTransformation,
)
from darsia.corrections.shape.piecewiseperspective import (
    PiecewisePerspectiveTransform as PiecewisePerspectiveTransform,
)
from darsia.corrections.shape.resize import ResizeCorrection as ResizeCorrection
from darsia.corrections.shape.rotation import RotationCorrection as RotationCorrection
from darsia.corrections.shape.transformation import (
    BaseTransformation as BaseTransformation,
)
from darsia.corrections.shape.transformation import (
    TransformationCorrection as TransformationCorrection,
)
from darsia.corrections.shape.translation import (
    TranslationCorrection as TranslationCorrection,
)
from darsia.corrections.shape.translation import (
    TranslationEstimator as TranslationEstimator,
)
from darsia.corrections.typecorrection import TypeCorrection as TypeCorrection
from darsia.experiment.experiment import Experiment as Experiment
from darsia.experiment.experiment import ProtocolledExperiment as ProtocolledExperiment
from darsia.experiment.protocols import ImagingInterval as ImagingInterval
from darsia.experiment.protocols import ImagingProtocol as ImagingProtocol
from darsia.experiment.protocols import ImagingProtocolOld as ImagingProtocolOld
from darsia.experiment.protocols import InjectionProtocol as InjectionProtocol
from darsia.experiment.protocols import (
    PressureTemperatureProtocol as PressureTemperatureProtocol,
)
from darsia.experiment.protocols import ThermodynamicState as ThermodynamicState
from darsia.experiment.protocols import logger as logger
from darsia.image.arithmetics import stack as stack
from darsia.image.arithmetics import superpose as superpose
from darsia.image.arithmetics import weight as weight
from darsia.image.conversion_kernels import to_float64 as to_float64
from darsia.image.conversion_kernels import to_ubyte as to_ubyte
from darsia.image.coordinatesystem import CoordinateSystem as CoordinateSystem
from darsia.image.coordinatesystem import annotations as annotations
from darsia.image.coordinatesystem import (
    check_equal_coordinatesystems as check_equal_coordinatesystems,
)
from darsia.image.coordinatetransformation import (
    CoordinateTransformation as CoordinateTransformation,
)
from darsia.image.coordinatetransformation import lir as lir
from darsia.image.image import ExtensiveImage as ExtensiveImage
from darsia.image.image import Image as Image
from darsia.image.image import OpticalImage as OpticalImage
from darsia.image.image import ScalarImage as ScalarImage
from darsia.image.imread import imread as imread
from darsia.image.imread import imread_from_bytes as imread_from_bytes
from darsia.image.imread import imread_from_dicom as imread_from_dicom
from darsia.image.imread import imread_from_npz as imread_from_npz
from darsia.image.imread import imread_from_numpy as imread_from_numpy
from darsia.image.imread import imread_from_optical as imread_from_optical
from darsia.image.imread import imread_from_vtu as imread_from_vtu
from darsia.image.indexing import cartesianToMatrixIndexing as cartesianToMatrixIndexing
from darsia.image.indexing import interpret_indexing as interpret_indexing
from darsia.image.indexing import matrixToCartesianIndexing as matrixToCartesianIndexing
from darsia.image.indexing import to_cartesian_indexing as to_cartesian_indexing
from darsia.image.indexing import to_matrix_indexing as to_matrix_indexing
from darsia.image.patches import Patches as Patches
from darsia.image.roi import ROI as ROI
from darsia.image.subregions import InterpolationOption as InterpolationOption
from darsia.image.subregions import (
    extract_quadrilateral_ROI as extract_quadrilateral_ROI,
)
from darsia.image.subregions import (
    restrict_image_to_landmarks as restrict_image_to_landmarks,
)
from darsia.manager.analysisbase import AnalysisBase as AnalysisBase
from darsia.manager.co2analysis import CO2Analysis as CO2Analysis
from darsia.manager.concentrationanalysisbase import (
    ConcentrationAnalysisBase as ConcentrationAnalysisBase,
)
from darsia.manager.traceranalysis import TracerAnalysis as TracerAnalysis
from darsia.measure.beckmann_bregman_solver import (
    BeckmannBregmanSolver as BeckmannBregmanSolver,
)
from darsia.measure.beckmann_convergence_criteria import (
    BeckmannConvergenceCriteria as BeckmannConvergenceCriteria,
)
from darsia.measure.beckmann_convergence_history import (
    BeckmannConvergenceHistory as BeckmannConvergenceHistory,
)
from darsia.measure.beckmann_gprox_solver import (
    BeckmannGproxPGHDSolver as BeckmannGproxPGHDSolver,
)
from darsia.measure.beckmann_linalg import BeckmannAMGSolver as BeckmannAMGSolver
from darsia.measure.beckmann_linalg import BeckmannCGSolver as BeckmannCGSolver
from darsia.measure.beckmann_linalg import BeckmannDirectSolver as BeckmannDirectSolver
from darsia.measure.beckmann_linalg import (
    BeckmannKSPFieldSplitSolver as BeckmannKSPFieldSplitSolver,
)
from darsia.measure.beckmann_linalg import BeckmannKSPSolver as BeckmannKSPSolver
from darsia.measure.beckmann_linalg import BeckmannLinearSolver as BeckmannLinearSolver
from darsia.measure.beckmann_linalg import (
    BeckmannLinearSolverFactory as BeckmannLinearSolverFactory,
)
from darsia.measure.beckmann_linalg import (
    BeckmannLinearSolverType as BeckmannLinearSolverType,
)
from darsia.measure.beckmann_newton_solver import (
    BeckmannNewtonSolver as BeckmannNewtonSolver,
)
from darsia.measure.beckmann_problem import BeckmannProblem as BeckmannProblem
from darsia.measure.beckmann_problem import L1Mode as L1Mode
from darsia.measure.beckmann_problem import MobilityMode as MobilityMode
from darsia.measure.emd import EMD as EMD
from darsia.measure.integration import ExtrudedGeometry as ExtrudedGeometry
from darsia.measure.integration import ExtrudedPorousGeometry as ExtrudedPorousGeometry
from darsia.measure.integration import Geometry as Geometry
from darsia.measure.integration import PorousGeometry as PorousGeometry
from darsia.measure.integration import WeightedGeometry as WeightedGeometry
from darsia.measure.norms import distance as distance
from darsia.measure.norms import h1_distance as h1_distance
from darsia.measure.norms import h1_semi_norm as h1_semi_norm
from darsia.measure.norms import l1_distance as l1_distance
from darsia.measure.norms import l2_distance as l2_distance
from darsia.measure.wasserstein import wasserstein_distance as wasserstein_distance
from darsia.measure.wasserstein import (
    wasserstein_distance_to_vtk as wasserstein_distance_to_vtk,
)
from darsia.multi_image_analysis.balancing_calibration import (
    AbstractBalancingCalibration as AbstractBalancingCalibration,
)
from darsia.multi_image_analysis.balancing_calibration import (
    ContinuityBasedBalancingCalibrationMixin as ContinuityBasedBalancingCalibrationMixin,
)
from darsia.multi_image_analysis.concentrationanalysis import (
    ConcentrationAnalysis as ConcentrationAnalysis,
)
from darsia.multi_image_analysis.concentrationanalysis import (
    PriorPosteriorConcentrationAnalysis as PriorPosteriorConcentrationAnalysis,
)
from darsia.multi_image_analysis.imageregistration import (
    DiffeomorphicImageRegistration as DiffeomorphicImageRegistration,
)
from darsia.multi_image_analysis.imageregistration import (
    ImageRegistration as ImageRegistration,
)
from darsia.multi_image_analysis.imageregistration import (
    MultiscaleDiffeomorphicImageRegistration as MultiscaleDiffeomorphicImageRegistration,
)
from darsia.multi_image_analysis.model_calibration import (
    AbsoluteVolumeModelObjectiveMixin as AbsoluteVolumeModelObjectiveMixin,
)
from darsia.multi_image_analysis.model_calibration import (
    AbstractModelObjective as AbstractModelObjective,
)
from darsia.multi_image_analysis.model_calibration import (
    InjectionRateModelObjectiveMixin as InjectionRateModelObjectiveMixin,
)
from darsia.multi_image_analysis.segmentationcomparison import (
    SegmentationComparison as SegmentationComparison,
)
from darsia.multi_image_analysis.translationanalysis import (
    TranslationAnalysis as TranslationAnalysis,
)
from darsia.multiphase.flash import AdvancedFlash as AdvancedFlash
from darsia.multiphase.flash import Flash as Flash
from darsia.multiphase.flash import IdentityFlash as IdentityFlash
from darsia.multiphase.flash import SimpleFlash as SimpleFlash
from darsia.multiphase.flash import load_flash as load_flash
from darsia.multiphase.flash import new_flash as new_flash
from darsia.multiphase.fluidflower_co2_meta import (
    FluidFlowerCO2Meta as FluidFlowerCO2Meta,
)
from darsia.multiphase.mass_analysis import EPSILON as EPSILON
from darsia.multiphase.mass_analysis import GRAVITY as GRAVITY
from darsia.multiphase.mass_analysis import PA2BAR as PA2BAR
from darsia.multiphase.mass_analysis import (
    AdvancedCO2MassAnalysis as AdvancedCO2MassAnalysis,
)
from darsia.multiphase.mass_analysis import CO2MassAnalysis as CO2MassAnalysis
from darsia.multiphase.mass_analysis import FluidMassAnalysis as FluidMassAnalysis
from darsia.multiphase.mass_analysis import HydrostaticState as HydrostaticState
from darsia.multiphase.mass_analysis import MassAnalysisResults as MassAnalysisResults
from darsia.multiphase.mass_analysis import (
    ThresholdAnalysisResults as ThresholdAnalysisResults,
)
from darsia.multiphase.mass_analysis import TracerMassAnalysis as TracerMassAnalysis
from darsia.multiphase.multiphase_time_series_analysis import (
    MultiphaseTimeSeriesAnalysis as MultiphaseTimeSeriesAnalysis,
)
from darsia.multiphase.multiphase_time_series_data import (
    MultiphaseTimeSeriesData as MultiphaseTimeSeriesData,
)
from darsia.multiphase.multiphase_time_series_data import (
    TimeSeriesData as TimeSeriesData,
)
from darsia.presets.workflows.config.fluidflower_config import (
    FluidFlowerConfig as FluidFlowerConfig,
)
from darsia.presets.workflows.config.time_data import TimeWindow as TimeWindow
from darsia.presets.workflows.simple_run_analysis import (
    SimpleMassAnalysisResults as SimpleMassAnalysisResults,
)
from darsia.presets.workflows.simple_run_analysis import (
    SimpleMultiphaseTimeSeriesData as SimpleMultiphaseTimeSeriesData,
)
from darsia.presets.workflows.simple_run_analysis import (
    SimpleRunAnalysis as SimpleRunAnalysis,
)
from darsia.pvt import FLUIDS as FLUIDS
from darsia.pvt.fluid import Fluid as Fluid
from darsia.pvt.fluid import get_fluid as get_fluid
from darsia.pvt.system import FluidSystem as FluidSystem
from darsia.pvt.table import PropertyTable as PropertyTable
from darsia.restoration.averaging import REV as REV
from darsia.restoration.averaging import VolumeAveraging as VolumeAveraging
from darsia.restoration.averaging import (
    porosity_based_averaging as porosity_based_averaging,
)
from darsia.restoration.averaging import volume_average as volume_average
from darsia.restoration.binaryinpaint import BinaryFillHoles as BinaryFillHoles
from darsia.restoration.binaryinpaint import (
    BinaryLocalConvexCover as BinaryLocalConvexCover,
)
from darsia.restoration.binaryinpaint import (
    BinaryRemoveSmallObjects as BinaryRemoveSmallObjects,
)
from darsia.restoration.h1_regularization import H1_regularization as H1_regularization
from darsia.restoration.median import Median as Median
from darsia.restoration.resize import SINGLE_CALL_CHANNELS as SINGLE_CALL_CHANNELS
from darsia.restoration.resize import Resize as Resize
from darsia.restoration.resize import equalize_voxel_size as equalize_voxel_size
from darsia.restoration.resize import resize as resize
from darsia.restoration.resize import uniform_refinement as uniform_refinement
from darsia.restoration.split_bregman_tvd import split_bregman_tvd as split_bregman_tvd
from darsia.restoration.tvd import TVD as TVD
from darsia.restoration.tvd import tvd as tvd
from darsia.signals.color import color_spectrum_kernels as color_spectrum_kernels
from darsia.signals.color.color_mode import ColorMode as ColorMode
from darsia.signals.color.color_path import ColorPath as ColorPath
from darsia.signals.color.color_path import define_color_path as define_color_path
from darsia.signals.color.color_path_regression import (
    ColorPathInputs as ColorPathInputs,
)
from darsia.signals.color.color_path_regression import (
    LabelColorPathMapRegression as LabelColorPathMapRegression,
)
from darsia.signals.color.color_range import ColorRange as ColorRange
from darsia.signals.color.color_spectrum import ColorSpectrum as ColorSpectrum
from darsia.signals.color.discrete_color_range import NUMBA_AVAILABLE as NUMBA_AVAILABLE
from darsia.signals.color.discrete_color_range import (
    DiscreteColorRange as DiscreteColorRange,
)
from darsia.signals.color.discrete_color_range import (
    color_to_index_numba as color_to_index_numba,
)
from darsia.signals.color.discrete_color_range import (
    flatten_index_numba as flatten_index_numba,
)
from darsia.signals.color.label_color_map import LabelColorMap as LabelColorMap
from darsia.signals.color.label_color_path_map import (
    LabelColorPathMap as LabelColorPathMap,
)
from darsia.signals.color.label_color_spectrum_map import (
    LabelColorSpectrumMap as LabelColorSpectrumMap,
)
from darsia.signals.color.utils import get_mean_color as get_mean_color
from darsia.signals.models.basemodel import HeterogeneousModel as HeterogeneousModel
from darsia.signals.models.basemodel import Model as Model
from darsia.signals.models.binarydataselector import BaseCriterion as BaseCriterion
from darsia.signals.models.binarydataselector import (
    BinaryDataSelector as BinaryDataSelector,
)
from darsia.signals.models.binarydataselector import (
    CombinedCriterion as CombinedCriterion,
)
from darsia.signals.models.binarydataselector import (
    GradientModulusCriterion as GradientModulusCriterion,
)
from darsia.signals.models.binarydataselector import (
    RelativeValueCriterion as RelativeValueCriterion,
)
from darsia.signals.models.binarydataselector import (
    TransformedValueCriterion as TransformedValueCriterion,
)
from darsia.signals.models.binarydataselector import ValueCriterion as ValueCriterion
from darsia.signals.models.clipmodel import ClipModel as ClipModel
from darsia.signals.models.color_path_interpolation import IGNORE_MODES as IGNORE_MODES
from darsia.signals.models.color_path_interpolation import (
    ColorPathFunction as ColorPathFunction,
)
from darsia.signals.models.color_path_interpolation import (
    ColorPathInterpolation as ColorPathInterpolation,
)
from darsia.signals.models.combinedmodel import CombinedModel as CombinedModel
from darsia.signals.models.dynamicthresholdmodel import (
    DynamicThresholdModel as DynamicThresholdModel,
)
from darsia.signals.models.dynamicthresholdmodel import (
    GlobalMinTwoPeakHistogrammAnalysis as GlobalMinTwoPeakHistogrammAnalysis,
)
from darsia.signals.models.dynamicthresholdmodel import (
    HistogrammBasedThresholding as HistogrammBasedThresholding,
)
from darsia.signals.models.dynamicthresholdmodel import (
    OtsuTwoPeakHistogrammAnalysis as OtsuTwoPeakHistogrammAnalysis,
)
from darsia.signals.models.dynamicthresholdmodel import StandardOtsu as StandardOtsu
from darsia.signals.models.dynamicthresholdmodel import (
    TwoPeakHistogrammAnalysis as TwoPeakHistogrammAnalysis,
)
from darsia.signals.models.evaluation import EvaluationMode as EvaluationMode
from darsia.signals.models.evaluation import KernelEvaluation as KernelEvaluation
from darsia.signals.models.evaluation import (
    KernelSignalFunction as KernelSignalFunction,
)
from darsia.signals.models.evaluation import LinearEvaluation as LinearEvaluation
from darsia.signals.models.evaluation import SplineEvaluation as SplineEvaluation
from darsia.signals.models.evaluation import curve_inverse as curve_inverse
from darsia.signals.models.evaluation import (
    select_affine_supports as select_affine_supports,
)
from darsia.signals.models.evaluation import signal_curve as signal_curve
from darsia.signals.models.kernelinterpolation import (
    AdvancedKernelInterpolation as AdvancedKernelInterpolation,
)
from darsia.signals.models.kernelinterpolation import (
    KernelInterpolation as KernelInterpolation,
)
from darsia.signals.models.linearmodel import (
    HeterogeneousLinearModel as HeterogeneousLinearModel,
)
from darsia.signals.models.linearmodel import LinearModel as LinearModel
from darsia.signals.models.linearmodel import ScalingModel as ScalingModel
from darsia.signals.models.pwtransformation import (
    PchipTransformation as PchipTransformation,
)
from darsia.signals.models.pwtransformation import PWTransformation as PWTransformation
from darsia.signals.models.pwtransformation import (
    monotonicity_message as monotonicity_message,
)
from darsia.signals.models.pwtransformation import (
    monotonicity_violation as monotonicity_violation,
)
from darsia.signals.models.staticthresholdmodel import (
    StaticThresholdModel as StaticThresholdModel,
)
from darsia.signals.models.thresholdmodel import ThresholdModel as ThresholdModel
from darsia.signals.reduction.dimensionreduction import AxisReduction as AxisReduction
from darsia.signals.reduction.dimensionreduction import (
    extrude_along_axis as extrude_along_axis,
)
from darsia.signals.reduction.dimensionreduction import reduce_axis as reduce_axis
from darsia.signals.reduction.monochromatic import (
    MonochromaticReduction as MonochromaticReduction,
)
from darsia.signals.reduction.signalreduction import SignalReduction as SignalReduction
from darsia.single_image_analysis.contour_smoother import (
    ContourSmoother as ContourSmoother,
)
from darsia.single_image_analysis.contour_smoother import (
    ContourSmootherSequence as ContourSmootherSequence,
)
from darsia.single_image_analysis.contour_smoother import (
    GaussianSmoother as GaussianSmoother,
)
from darsia.single_image_analysis.contour_smoother import (
    MovingAverageSmoother as MovingAverageSmoother,
)
from darsia.single_image_analysis.contour_smoother import (
    PolyDPSmoother as PolyDPSmoother,
)
from darsia.single_image_analysis.contour_smoother import (
    SavitzkyGolaySmoother as SavitzkyGolaySmoother,
)
from darsia.single_image_analysis.contouranalysis import (
    ContourAnalysis as ContourAnalysis,
)
from darsia.single_image_analysis.contouranalysis import (
    contour_length as contour_length,
)
from darsia.utils import augmented_plotting as augmented_plotting
from darsia.utils import linalg as linalg
from darsia.utils import plotting as plotting
from darsia.utils import quadrature as quadrature
from darsia.utils.andersonacceleration import (
    AndersonAcceleration as AndersonAcceleration,
)
from darsia.utils.approximations import ApproximationSpace as ApproximationSpace
from darsia.utils.approximations import LinearApproximation as LinearApproximation
from darsia.utils.approximations import (
    PolynomialApproximationSpace as PolynomialApproximationSpace,
)
from darsia.utils.approximations import (
    RadialPolynomialApproximationSpace as RadialPolynomialApproximationSpace,
)
from darsia.utils.arithmetics import array_product as array_product
from darsia.utils.array_slice import array_slice as array_slice
from darsia.utils.array_slice import array_slice_argument as array_slice_argument
from darsia.utils.augmented_plotting import (
    plot_contour_on_image as plot_contour_on_image,
)
from darsia.utils.box import bounding_box as bounding_box
from darsia.utils.box import bounding_box_inverse as bounding_box_inverse
from darsia.utils.box import perimeter as perimeter
from darsia.utils.box import random_patches as random_patches
from darsia.utils.coloranalysis import hsv_spectrum as hsv_spectrum
from darsia.utils.convergence_status import ConvergenceStatus as ConvergenceStatus
from darsia.utils.derivatives import backward_diff as backward_diff
from darsia.utils.derivatives import forward_diff as forward_diff
from darsia.utils.derivatives import laplace as laplace
from darsia.utils.detection import detect_closest_point as detect_closest_point
from darsia.utils.detection import detect_color as detect_color
from darsia.utils.detection import detect_value as detect_value
from darsia.utils.detection import (
    monochromatic_concentration_analysis as monochromatic_concentration_analysis,
)
from darsia.utils.detection import orthogonal_colors as orthogonal_colors
from darsia.utils.dtype import convert_dtype as convert_dtype
from darsia.utils.extractcharacteristicdata import (
    extract_characteristic_data as extract_characteristic_data,
)
from darsia.utils.features import FeatureDetection as FeatureDetection
from darsia.utils.formats import Format as Format
from darsia.utils.fv import FVDivergence as FVDivergence
from darsia.utils.fv import FVFullFaceReconstruction as FVFullFaceReconstruction
from darsia.utils.fv import FVMass as FVMass
from darsia.utils.fv import (
    FVTangentialFaceReconstruction as FVTangentialFaceReconstruction,
)
from darsia.utils.fv import cell_to_face_average as cell_to_face_average
from darsia.utils.fv import face_to_cell as face_to_cell
from darsia.utils.grid import Grid as Grid
from darsia.utils.grid import generate_grid as generate_grid
from darsia.utils.interpolation import (
    illumination_interpolation as illumination_interpolation,
)
from darsia.utils.interpolation import (
    interpolate_measurements_2d as interpolate_measurements_2d,
)
from darsia.utils.interpolation import interpolate_to_image as interpolate_to_image
from darsia.utils.interpolation import (
    interpolate_to_image_from_csv as interpolate_to_image_from_csv,
)
from darsia.utils.interpolation import (
    polynomial_interpolation as polynomial_interpolation,
)
from darsia.utils.kernels import BaseKernel as BaseKernel
from darsia.utils.kernels import GaussianKernel as GaussianKernel
from darsia.utils.kernels import LinearKernel as LinearKernel
from darsia.utils.kernels import kernel_from_dict as kernel_from_dict
from darsia.utils.linear_solvers.cg import CG as CG
from darsia.utils.linear_solvers.jacobi import Jacobi as Jacobi
from darsia.utils.linear_solvers.mg import MG as MG
from darsia.utils.linear_solvers.solver import Solver as Solver
from darsia.utils.masks import Masks as Masks
from darsia.utils.masks import MasksIterator as MasksIterator
from darsia.utils.point import BasePoint as BasePoint
from darsia.utils.point import Coordinate as Coordinate
from darsia.utils.point import CoordinateArray as CoordinateArray
from darsia.utils.point import Voxel as Voxel
from darsia.utils.point import VoxelArray as VoxelArray
from darsia.utils.point import VoxelCenter as VoxelCenter
from darsia.utils.point import VoxelCenterArray as VoxelCenterArray
from darsia.utils.point import make_coordinate as make_coordinate
from darsia.utils.point import make_voxel as make_voxel
from darsia.utils.point import make_voxel_center as make_voxel_center
from darsia.utils.point import to as to
from darsia.utils.point import to_coordinate as to_coordinate
from darsia.utils.point import to_voxel as to_voxel
from darsia.utils.point import to_voxel_center as to_voxel_center
from darsia.utils.segmentation import group_labels as group_labels
from darsia.utils.segmentation import label_image as label_image
from darsia.utils.segmentation import make_consecutive as make_consecutive
from darsia.utils.segmentation import reassign_labels as reassign_labels
from darsia.utils.segmentation import segment as segment
from darsia.utils.slices import add_slice_pairs as add_slice_pairs
from darsia.utils.slices import add_slices as add_slices
from darsia.utils.slices import subtract_slice_pairs as subtract_slice_pairs
from darsia.utils.slices import subtract_slices as subtract_slices
from darsia.utils.sort import sort_quad as sort_quad
from darsia.utils.standard_images import TYPE_CHECKING as TYPE_CHECKING
from darsia.utils.standard_images import StandardDtype as StandardDtype
from darsia.utils.standard_images import full_like as full_like
from darsia.utils.standard_images import ones_like as ones_like
from darsia.utils.standard_images import roi_to_mask as roi_to_mask
from darsia.utils.standard_images import zeros_like as zeros_like
from darsia.utils.timings import Timings as Timings
from darsia.utils.timings import timed as timed
from darsia.utils.timings import timing_decorator as timing_decorator
from darsia.utils.timings import timings as timings
