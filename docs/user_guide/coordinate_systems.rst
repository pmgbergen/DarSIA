##################
Coordinate systems
##################

.. currentmodule:: darsia

A :class:`CoordinateSystem` is attached to every :class:`Image`. It converts
between **voxel** indices (row, column, ...) and **Cartesian** coordinates
(x, y in metres, with the origin and axis orientation fixed by the image
metadata). This is what lets you say "give me the region between (0.1, 0.1) and
(0.4, 0.3) metres" instead of counting pixels.

.. code-block:: python

   cs = image.coordinatesystem
   voxel = cs.voxel([0.25, 0.1])        # Cartesian -> pixel index
   xy = cs.coordinate([100, 250])       # pixel index -> Cartesian
   # cs.voxels / cs.coordinates do the same for arrays of points

Points and arrays of points
===========================

DarSIA distinguishes coordinate kinds explicitly so an accidental row/column vs
x/y swap is caught early:

* :class:`Coordinate` / :class:`CoordinateArray` -- Cartesian, metres
* :class:`Voxel` / :class:`VoxelArray` -- integer pixel indices
* :class:`VoxelCenter` / :class:`VoxelCenterArray` -- pixel centres

Constructors :func:`make_coordinate`, :func:`make_voxel` and the ``to_*``
converters move between them.

Regions and patches
===================

* :meth:`Image.subregion` -- cut out a rectangular ROI, given in voxels or as a
  :class:`Coordinate` pair.
* :class:`ROI` -- a named rectangle in physical coordinates, reused across a
  workflow.
* :class:`Patches` -- tile an image into a regular grid of sub-images, e.g. for
  patch-wise illumination correction or local translation estimation.
* :class:`CoordinateTransformation` -- align two images that share a scene but
  have different relative coordinate systems (this changes *metadata*, not
  pixels; contrast with the geometric corrections in :doc:`shape_correction`).

See also
========

* :doc:`image_data_model`
* :doc:`../api/images`
