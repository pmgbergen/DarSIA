##########
Assistants
##########

.. currentmodule:: darsia

Assistants open an interactive Matplotlib window so a user can pick points,
boxes, rotations or labels and feed the result straight into a correction or
analysis. They all derive from :class:`BaseAssistant`.

Geometry selection
==================

.. autosummary::
   :toctree: generated/

   PointSelectionAssistant
   BoxSelectionAssistant
   RectangleSelectionAssistant
   SubregionAssistant
   CropAssistant
   RotationCorrectionAssistant

Label editing
=============

.. autosummary::
   :toctree: generated/

   LabelsAssistant
   LabelsPickAssistant
   LabelsMaskSelectionAssistant
   LabelsMergeAssistant
   LabelsSegmentAssistant
   MonochromaticAssistant

Advanced
========

.. autosummary::
   :toctree: generated/

   BaseAssistant
   LabelsAssistantMenu
