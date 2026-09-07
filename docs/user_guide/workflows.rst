###################
The workflow system
###################

.. currentmodule:: darsia

Everything in the preceding chapters can be scripted by hand. For a full
experiment -- dozens of images, several corrections, a calibration, a batch of
analyses -- DarSIA provides ``darsia.presets.workflows``: a
**configuration-driven pipeline** where a single TOML file describes the rig, the
imaging protocol, the corrections, the calibration and the analyses, and the
pipeline runs them end to end.

The five phases
===============

#. **Setup** -- one-time per experiment: rig geometry, depth map, facies
   segmentation, protocol files, geometric correction.
#. **Calibration** -- colour embedding and colour-to-mass calibration.
#. **Analysis** -- the batch processing: cropping, mass / saturation /
   concentration maps, segmentation contours, finger detection, thresholding.
#. **Helper** -- interactive inspection tools (ROI picker, result viewer, colour
   histograms) for choosing good settings.
#. **Comparison** -- across runs: event timing, Wasserstein distances.

Registries
==========

The config format is built on *define-once, reference-by-key* registries: data
selectors, ROIs, colour embeddings and export formats live in top-level sections
and every phase refers to them by name. Getting this pattern is the key to
reading and writing configs.

Where to go
===========

* :doc:`gui/index` -- **the practical tutorial**: install, launch the GUI, and
  walk a real ``sample_config.toml`` through all five phases with screenshots and
  a field-by-field configuration reference.
* The command-line workflow docs in ``src/darsia/presets/workflows/doc/`` cover
  the same ground from the ``user_interface_*`` CLI angle (``quickstart.md``,
  ``config-reference.md``, ``workflow-setup.md`` and siblings).
* :doc:`../api/workflows` -- the Python entry points.

Run a phase from the command line with, e.g.::

   uv run python -m darsia.presets.workflows.user_interface_analysis \
     --mass --segmentation --config sample_config.toml
