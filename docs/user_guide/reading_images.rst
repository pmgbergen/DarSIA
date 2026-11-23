##############
Reading images
##############

.. currentmodule:: darsia

:func:`imread` is the single entry point for getting data into DarSIA. It reads
files, folders and lists of files, and the bytes of an image file (e.g. received
over a network). For paths it dispatches on the file extension to the right reader.
It returns an :class:`Image`; a folder or a list of paths gives one space-time
image (a series), not a list of images.

.. code-block:: python

   import darsia

   # one optical photograph, 2.8 m x 1.5 m
   photo = darsia.imread("baseline.jpg", width=2.8, height=1.5)

   # the same photograph, given as the bytes of the file
   photo = darsia.imread(data, width=2.8, height=1.5)

   # photographs as one space-time image
   series = darsia.imread(["f/img_0.jpg", "f/img_1.jpg"], dimensions=[1.5, 2.8])

   # a 3D DICOM stack
   volume = darsia.imread("scan/", dim=3)

Supported formats
=================

=====================  ===========================================================
optical (jpg/png/tif)  :func:`imread_from_optical` -- returns :class:`OpticalImage`
DICOM (.dcm)           :func:`imread_from_dicom` -- 2D/3D/4D medical stacks
simulation (.vtu)      :func:`imread_from_vtu` -- meshio-backed, mixed-dimensional
NumPy (.npy/.npz)      :func:`imread_from_numpy`, :func:`imread_from_npz`
in-memory bytes        :func:`imread_from_bytes` -- read like a single optical file
scalar grids (.csv)    one row per cell: cell-center coordinates and value
=====================  ===========================================================

All of them are reached through :func:`imread`.

Scalar grids in CSV format
--------------------------

There is no standard for images in CSV format. DarSIA follows the reporting format
of the 11th SPE Comparative Solution Project (SPE11): one row per cell, with the
coordinates of the cell center and the value as columns (x, y, value), and an
optional header. :meth:`ScalarImage.to_csv` writes this format; :func:`imread`
reads it back (2D), as :class:`ScalarImage`, or as :class:`ExtensiveImage` with
``extensive=True``.

Dates
-----

A given ``date`` is used as is (also ``None``). Without one, the date of an optical
image (file or bytes) is taken from its EXIF data (:func:`exif_date`); without EXIF
date, the image has no date.

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
