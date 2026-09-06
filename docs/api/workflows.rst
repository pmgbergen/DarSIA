###############
Workflow system
###############

.. currentmodule:: darsia

``darsia.presets.workflows`` is a configuration-driven pipeline: a single TOML
file describes the rig, the imaging protocol, the corrections, the calibration
and the analysis, and the workflow runs them end to end. It powers the DarSIA
GUI and the ``darsia`` command-line tools.

.. seealso::

   The narrative :doc:`../user_guide/index` (workflow chapter) and the reference
   under ``src/darsia/presets/workflows/doc/`` in the repository walk through
   writing a configuration.

Top-level exports
=================

.. autosummary::
   :toctree: generated/

   SimpleRunAnalysis
   SimpleMassAnalysisResults
   SimpleMultiphaseTimeSeriesData
   FluidFlowerConfig
   TimeWindow

.. currentmodule:: darsia.presets.workflows

Key building blocks
===================

.. autosummary::
   :toctree: generated/

   rig.Rig
   analysis.analysis_context.prepare_analysis_context
