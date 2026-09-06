###########
Corrections
###########

.. currentmodule:: darsia

Corrections are callables that take an :class:`Image` and return a corrected
:class:`Image`. *Colour* corrections calibrate recorded colours against a colour
checker or reference illumination; *shape* corrections change the geometry of the
data container (curvature, perspective, drift, deformation). All of them derive
from :class:`BaseCorrection` and can be passed to :func:`imread` to run during
reading.

Base
====

.. autosummary::
   :toctree: generated/

   BaseCorrection
   TypeCorrection
   read_correction

Colour correction
=================

With a known or approximately known colour-checker location, recorded colours are
matched to reference values.

.. autosummary::
   :toctree: generated/

   ColorCorrection
   ColorChecker
   ColorCheckerAfter2014
   CustomColorChecker
   ClassicColorChecker
   find_colorchecker
   RelativeColorCorrection

Illumination correction
=======================

.. autosummary::
   :toctree: generated/

   IlluminationCorrection
   PatchwiseIlluminationCorrection
   DynamicIlluminationCorrection

Colour balance
==============

.. autosummary::
   :toctree: generated/

   ColorBalance
   WhiteBalance
   AffineBalance
   AdaptiveBalance
   color_balance
   white_balance
   affine_balance

Shape correction
================

.. autosummary::
   :toctree: generated/

   CurvatureCorrection
   AffineCorrection
   GeneralizedPerspectiveCorrection
   DriftCorrection
   DeformationCorrection
   TranslationCorrection
   RotationCorrection
   ResizeCorrection
   TransformationCorrection

Stand-alone transformations
===========================

.. autosummary::
   :toctree: generated/

   AffineTransformation
   GeneralizedPerspectiveTransformation
   PiecewisePerspectiveTransform
   TranslationEstimator

Advanced
========

.. autosummary::
   :toctree: generated/

   BaseBalance
   BaseTransformation
   EOTF
   ExperimentalColorCorrection
   load_curvature_correction_config_from_dict
   load_curvature_correction_config_from_toml
