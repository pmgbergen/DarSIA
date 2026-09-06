##########
User guide
##########

.. note::

   The narrative user guide is under construction. The pages below will be
   filled in over the coming iterations; in the meantime the :doc:`../api/index`
   documents every public object.

The user guide explains the concepts behind DarSIA and how its pieces fit
together:

* **The image data model** -- :class:`~darsia.Image`, :class:`~darsia.ScalarImage`
  and :class:`~darsia.OpticalImage`, physical metadata, coordinate systems and
  patches.
* **Reading images** -- optical photographs, DICOM stacks and simulation data
  through :func:`~darsia.imread`.
* **Corrections** -- colour, curvature, affine / perspective, drift and
  deformation corrections.
* **Restoration** -- total-variation and H1 denoising.
* **Segmentation and registration** -- geometry segmentation and image
  registration for compaction / settling.
* **Concentration analysis** -- tracer and CO2 concentration maps, signal models
  and model calibration.
* **Distances** -- Earth Mover's and Wasserstein distances.
* **Interactive assistants** -- Matplotlib-based helpers for picking ROIs and
  correction parameters.
* **The workflow system** -- the configuration-driven ``presets.workflows``
  pipeline.
* **The graphical interface** -- a from-scratch guide to the DarSIA GUI.
