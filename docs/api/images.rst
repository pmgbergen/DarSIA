####################
Images & coordinates
####################

.. currentmodule:: darsia

DarSIA represents an image as an ``(n + 1)``-dimensional array carrying a
physical interpretation: ``n`` spatial dimensions plus an optional time axis,
with a scalar or vectorial range. All images share the :class:`Image` interface;
:class:`ScalarImage` and :class:`OpticalImage` are convenience specialisations.

A :class:`CoordinateSystem` translates between voxel indices and Cartesian
coordinates, so subregions, patches and regions of interest can be addressed in
metres rather than pixels.

Image containers
================

.. autosummary::
   :toctree: generated/

   Image
   ScalarImage
   OpticalImage

Coordinate systems
==================

.. autosummary::
   :toctree: generated/

   CoordinateSystem
   CoordinateTransformation
   check_equal_coordinatesystems

Regions, patches and subregions
===============================

.. autosummary::
   :toctree: generated/

   Patches
   ROI
   extract_quadrilateral_ROI

Image arithmetic
================

.. autosummary::
   :toctree: generated/

   weight
   superpose
   stack

Advanced
========

.. autosummary::
   :toctree: generated/

   ExtensiveImage
   interpret_indexing
   to_matrix_indexing
   to_cartesian_indexing
   matrixToCartesianIndexing
   cartesianToMatrixIndexing
