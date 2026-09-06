########################
Analysis & registration
########################

.. currentmodule:: darsia

High-level tools that compare images over time: concentration analysis from a
baseline and a sequence of test images, registration of images that have moved or
deformed, and calibration of the underlying models. The *managers* wire a full
pipeline together for common experiment types.

Concentration analysis
======================

.. autosummary::
   :toctree: generated/

   ConcentrationAnalysis
   PriorPosteriorConcentrationAnalysis

Image registration
==================

.. autosummary::
   :toctree: generated/

   ImageRegistration
   DiffeomorphicImageRegistration
   MultiscaleDiffeomorphicImageRegistration
   TranslationAnalysis
   SegmentationComparison

Calibration
===========

.. autosummary::
   :toctree: generated/

   AbstractModelObjective
   AbsoluteVolumeModelObjectiveMixin
   InjectionRateModelObjectiveMixin
   AbstractBalancingCalibration
   ContinuityBasedBalancingCalibrationMixin

Managers
========

.. autosummary::
   :toctree: generated/

   AnalysisBase
   ConcentrationAnalysisBase
   TracerAnalysis
   CO2Analysis
