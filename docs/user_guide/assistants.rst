######################
Interactive assistants
######################

.. currentmodule:: darsia

Assistants open a Matplotlib window so you can pick points, boxes, rotations or
labels by clicking, and hand the result straight to a correction or an analysis.
They are the interactive counterpart to hand-writing pixel coordinates into a
config. All derive from :class:`BaseAssistant`.

Geometry
========

============================================  ==================================================
:class:`PointSelectionAssistant`              click a set of points
:class:`BoxSelectionAssistant`,               drag one or more rectangles; used to collect
:class:`RectangleSelectionAssistant`          colour-calibration sample windows
:class:`SubregionAssistant`,                  pick a region of interest, get a ready-to-paste
:class:`CropAssistant`                        ``[roi]`` block or a crop config
:class:`RotationCorrectionAssistant`          pick two points defining "horizontal"
============================================  ==================================================

Labels
======

:class:`LabelsAssistant` is a small menu-driven editor for a segmentation label
field: :class:`LabelsPickAssistant`, :class:`LabelsMergeAssistant`,
:class:`LabelsSegmentAssistant`, :class:`LabelsMaskSelectionAssistant`,
:class:`MonochromaticAssistant`.

.. code-block:: python

   import darsia

   samples = darsia.BoxSelectionAssistant(image)()   # returns a list of slices

These require an interactive Matplotlib backend, so they are used from scripts
and the GUI, not during an unattended docs or CI build.

See also
========

* :doc:`gui/index` -- the GUI wraps the same picks as forms
* :doc:`../api/assistants`
