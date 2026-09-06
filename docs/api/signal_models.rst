#############
Signal models
#############

.. currentmodule:: darsia

A concentration analysis proceeds in two stages: a *signal reduction* collapses a
(possibly multichannel) difference image to a scalar signal, and a *model* maps
that signal to a physical quantity such as a concentration or a saturation. Models
share the :class:`Model` interface and can be composed and calibrated.

Signal reduction
================

.. autosummary::
   :toctree: generated/

   SignalReduction
   MonochromaticReduction
   AxisReduction
   reduce_axis
   extrude_along_axis

Model base
==========

.. autosummary::
   :toctree: generated/

   Model
   HeterogeneousModel
   CombinedModel

Linear and scaling models
=========================

.. autosummary::
   :toctree: generated/

   LinearModel
   HeterogeneousLinearModel
   ScalingModel

Thresholding models
===================

.. autosummary::
   :toctree: generated/

   StaticThresholdModel
   DynamicThresholdModel
   ThresholdModel
   ClipModel
   PWTransformation

Histogram analysis
==================

.. autosummary::
   :toctree: generated/

   StandardOtsu
   OtsuTwoPeakHistogrammAnalysis
   TwoPeakHistogrammAnalysis
   GlobalMinTwoPeakHistogrammAnalysis
   HistogrammBasedThresholding

Kernel interpolation
====================

.. autosummary::
   :toctree: generated/

   KernelInterpolation
   AdvancedKernelInterpolation

Colour-path models
==================

.. autosummary::
   :toctree: generated/

   ColorPathInterpolation
   ColorPathFunction

Binary data selection
=====================

.. autosummary::
   :toctree: generated/

   BinaryDataSelector
   ValueCriterion
   RelativeValueCriterion
   TransformedValueCriterion
   GradientModulusCriterion
   CombinedCriterion
   BaseCriterion
