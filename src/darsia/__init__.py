"""Root directory for DarSIA.

isort:skip_file

"""

# Basic imports
from darsia.image.coordinatesystem import *
from darsia.image.image import *
from darsia.image.indexing import *
from darsia.image.patches import *
from darsia.image.imread import *
from darsia.image.arithmetics import *

# Numerical integration
from darsia.measure.integration import *

# EMD and Wasserstein distances
from darsia.utils.convergence_status import *  # Needed by measure
from darsia.measure.emd import *
from darsia.measure.beckmann_problem import *
from darsia.measure.beckmann_newton_solver import *
from darsia.measure.beckmann_bregman_solver import *
from darsia.measure.beckmann_gprox_solver import *
from darsia.measure.wasserstein import *

# Utilities
from darsia.utils.point import *
from darsia.utils.sort import *
from darsia.utils.box import *
from darsia.utils.interpolation import *
from darsia.utils.segmentation import *
from darsia.utils.coloranalysis import *
from darsia.utils.array_slice import *
from darsia.utils.derivatives import *
from darsia.utils.linear_solvers.solver import *
from darsia.utils.linear_solvers.jacobi import *
from darsia.utils.linear_solvers.cg import *
from darsia.utils.linear_solvers.mg import *
from darsia.utils.andersonacceleration import *
from darsia.utils.dtype import *
from darsia.utils.formats import *
from darsia.utils.grid import *
from darsia.utils.fv import *
from darsia.utils.kernels import *
from darsia.utils.extractcharacteristicdata import *
from darsia.utils.masks import *
from darsia.utils import linalg
from darsia.utils import quadrature
from darsia.utils import augmented_plotting
from darsia.utils import plotting
from darsia.utils.detection import *
from darsia.utils.standard_images import *
from darsia.utils.approximations import *
from darsia.utils.slices import *

# Image subregions (required specific placing)
from darsia.image.subregions import *

# Corrections and transformations
from darsia.corrections.basecorrection import *
from darsia.corrections.typecorrection import *
from darsia.corrections.shape.transformation import *
from darsia.corrections.shape.curvature import *
from darsia.corrections.shape.affine import *
from darsia.corrections.shape.translation import *
from darsia.corrections.shape.piecewiseperspective import *
from darsia.corrections.shape.generalizedperspective import *
from darsia.corrections.shape.rotation import *
from darsia.corrections.shape.drift import *
from darsia.corrections.shape.deformation import *
from darsia.corrections.color.colorbalance import *
from darsia.corrections.color.colorcheckerfinder import *
from darsia.corrections.color.illuminationcorrection import *
from darsia.corrections.color.dynamicilluminationcorrection import *
from darsia.corrections.color.colorcorrection import *
from darsia.corrections.color.relativecolorcorrection import *
from darsia.corrections.color.experimentalcolorcorrection import *
from darsia.corrections.readcorrection import *
from darsia.image.coordinatetransformation import *  # Requires affine correction

# Signals, reduction, and models
from darsia.signals.models.basemodel import *
from darsia.signals.models.combinedmodel import *
from darsia.signals.models.linearmodel import *
from darsia.signals.models.clipmodel import *
from darsia.signals.models.thresholdmodel import *
from darsia.signals.models.staticthresholdmodel import *
from darsia.signals.models.dynamicthresholdmodel import *
from darsia.signals.models.binarydataselector import *
from darsia.signals.models.kernelinterpolation import *
from darsia.signals.models.pwtransformation import *
from darsia.signals.reduction.signalreduction import *
from darsia.signals.reduction.monochromatic import *
from darsia.signals.reduction.dimensionreduction import *

# Restoration
from darsia.restoration.tvd import *
from darsia.restoration.median import *
from darsia.restoration.resize import *
from darsia.restoration.binaryinpaint import *
from darsia.restoration.h1_regularization import *
from darsia.restoration.split_bregman_tvd import *
from darsia.restoration.averaging import *

# Analysis
from darsia.multi_image_analysis.translationanalysis import *
from darsia.multi_image_analysis.concentrationanalysis import *
from darsia.multi_image_analysis.model_calibration import *
from darsia.multi_image_analysis.balancing_calibration import *
from darsia.multi_image_analysis.imageregistration import *
from darsia.multi_image_analysis.segmentationcomparison import *
from darsia.single_image_analysis.contouranalysis import *

# Managers
from darsia.manager.analysisbase import *
from darsia.manager.concentrationanalysisbase import *
from darsia.manager.traceranalysis import *
from darsia.manager.co2analysis import *

# Assistants
from darsia.assistants.base_assistant import *
from darsia.assistants.point_selection_assistant import *
from darsia.assistants.box_selection_assistant import *
from darsia.assistants.rectangle_selection_assistant import *
from darsia.assistants.rotation_correction_assistant import *
from darsia.assistants.subregion_assistant import *
from darsia.assistants.crop_assistant import *
from darsia.assistants.labels_assistant import *

# Multiphase flow
from darsia.multiphase.flash import *
from darsia.multiphase.mass_analysis import *
from darsia.multiphase.multiphase_time_series_data import *
