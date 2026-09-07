##############
Reading images
##############

.. currentmodule:: darsia

:func:`imread` is the single entry point for getting data into DarSIA. It looks
at the file extension, dispatches to the right reader, and returns an
:class:`Image` (or a list of images for a folder or a list of paths).

.. code-block:: python

   import darsia

   # one optical photograph, 2.8 m x 1.5 m
   photo = darsia.imread("baseline.jpg", width=2.8, height=1.5)

   # a whole folder of photographs as one space-time image
   series = darsia.imread_from_optical(["f/img_0.jpg", "f/img_1.jpg"], dimensions=[1.5, 2.8])

   # a 3D DICOM stack
   volume = darsia.imread("scan/", dim=3)

Supported formats
=================

=====================  ===========================================================
optical (jpg/png/tif)  :func:`imread_from_optical` -- returns :class:`OpticalImage`
DICOM (.dcm)           :func:`imread_from_dicom` -- 2D/3D/4D medical stacks
simulation (.vtu)      :func:`imread_from_vtu` -- meshio-backed, mixed-dimensional
NumPy (.npy/.npz)      :func:`imread_from_numpy`, :func:`imread_from_npz`
in-memory bytes        :func:`imread_from_bytes`
=====================  ===========================================================

Applying corrections while reading
==================================

Pass correction objects via ``transformations=`` and they run as the image is
read, so downstream code only ever sees corrected data:

.. code-block:: python

   curvature = darsia.CurvatureCorrection(config=cfg["curvature"])
   photo = darsia.imread("baseline.jpg", transformations=[curvature],
                         width=2.8, height=1.5)

See :doc:`shape_correction`, :doc:`color_correction` and
:doc:`illumination_correction` for the correction catalogue.

See also
========

* :doc:`../auto_examples/io/plot_reading_images`
* :doc:`../auto_examples/io/plot_numpy_images`
* :doc:`../api/io`
