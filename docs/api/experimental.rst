############
Experimental
############

.. currentmodule:: darsia

.. warning::

   The objects on this page have an **unstable API** and may change or be
   removed without notice. They are exposed for early adopters and for internal
   use by the workflow presets.

Experiment and protocols
========================

Data models for an imaging experiment and its imaging / injection /
pressure-temperature protocols.

.. autosummary::
   :toctree: generated/

   Experiment
   ProtocolledExperiment
   ImagingProtocol
   ImagingInterval
   InjectionProtocol
   PressureTemperatureProtocol
   ThermodynamicState

Multiphase analysis
===================

Flash calculations and CO2 mass analysis for multiphase FluidFlower experiments.

.. autosummary::
   :toctree: generated/

   Flash
   SimpleFlash
   AdvancedFlash
   CO2MassAnalysis
   AdvancedCO2MassAnalysis
   MassAnalysisResults
   ThresholdAnalysisResults
   FluidFlowerCO2Meta
   MultiphaseTimeSeriesAnalysis
   MultiphaseTimeSeriesData
   TimeSeriesData
