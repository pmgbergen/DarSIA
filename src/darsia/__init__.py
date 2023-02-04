"""Root directory for DarSIA.

Includes:

    conversion, image class, subregions, coordinate system, and curvature correction

isort:skip_file

"""
from darsia.image.coordinatesystem import *
from darsia.image.image import *
from darsia.image.patches import *
from darsia.image.subregions import *
from darsia.mathematics.derivatives import *
from darsia.mathematics.norms import *
from darsia.mathematics.stoppingcriterion import *
from darsia.mathematics.solvers import *
from darsia.mathematics.regularization import *
from darsia.utils.conversions import *
from darsia.utils.resolution import *
from darsia.utils.box import *
from darsia.utils.segmentation import *
from darsia.utils.coloranalysis import *
from darsia.corrections.shape.curvature import *
from darsia.corrections.shape.translation import *
from darsia.corrections.shape.piecewiseperspective import *
from darsia.corrections.shape.curvature import *
from darsia.corrections.shape.drift import *
from darsia.corrections.color.colorcorrection import *
from darsia.corrections.color.experimentalcolorcorrection import *
from darsia.analysis.translationanalysis import *
from darsia.analysis.concentrationanalysis import *
from darsia.analysis.compactionanalysis import *
from darsia.analysis.segmentationcomparison import *
from darsia.analysis.contouranalysis import *
from darsia.manager.analysisbase import *
from darsia.manager.concentrationanalysisbase import *
from darsia.manager.traceranalysis import *
from darsia.manager.co2analysis import *
from darsia.transformations.signals.basemodel import *
from darsia.transformations.signals.linearmodel import *
from darsia.transformations.signals.thresholdmodel import *
from darsia.transformations.signals.staticthresholdmodel import *
from darsia.transformations.signals.dynamicthresholdmodel import *
from darsia.transformations.signals.binarydataselector import *
from darsia.transformations.colors.monochromatic import *
from darsia.regularization.tvd import *
from darsia.regularization.binaryinpaint import *
