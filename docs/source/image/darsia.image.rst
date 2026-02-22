Physical Images and I/O (:mod:`darsia.image`)
=============================================

.. currentmodule:: darsia.image

.. automodule:: darsia.image
   :members:
   :undoc-members:
   :show-inheritance:

Image data types
----------------
DarSIA supports general ``(n+1)``-dimensional images, where ``n`` denotes the spatial dimension and ``+1`` indicates the possibility for space-time realizations of time series. Furthermore, the range of images can be both scalar and vectorial. While all images are of type ``Image``, convenient special cases are provided for scalar images, and optical images, i.e., photographs.

.. currentmodule:: darsia.image.image

.. autosummary::

   Image
   ScalarImage
   OpticalImage

Physical notion
---------------
A central feature of images in DarSIA is their physical interpretation. In particular, coordinate systems providing translations between voxel and Cartesian coordinates allows for dimensionally meaningful extraction of subregions etc.

.. currentmodule:: darsia.image.coordinatesystem

.. autosummary::

   CoordinateSystem

A unified interface for reading images
--------------------------------------
Provided an image stored to file, a unified interface is provided being able to read various data formats, as ``jpg``, ``png``, ``dcm``, ``vtu``, etc.

.. currentmodule:: darsia.image.imread

.. autosummary::

   imread

It is essentially a wrapper for dedicated image reading routines for specific data types - it reacts to the data ending.

.. autosummary::

   imread_from_numpy
   imread_from_optical
   imread_from_dicom
   imread_from_vtu

.. note:: Transformations can be integrated in the pre-processing and reading of images, see :mod:`darsia.corrections``.

Arithmetics
-----------
Provided a set of images, various compositions of these can be constructed.

.. currentmodule:: darsia.image.arithmetics

.. autosummary::

   weight
   superpose
   stack

Coordinate transformation
-------------------------
Contrary to correction methods, which alter the data, coordinate transformation aim at altering the metadata. Provided two physical images with two differing relative coordinate systems, these can be aligned through transformation.

.. currentmodule:: darsia.image.coordinatetransformation

.. autosummary::

   CoordinateTransformation

Utilities
---------
Various utilities are used across DarSIA, e.g., for converting matrix to Cartesian axis, extracting patches and subregions.

.. currentmodule:: darsia.image

.. autosummary::
  
   indexing.to_matrix_indexing
   indexing.to_cartesian_indexing
   indexing.interpret_indexing
   indexing.matrixToCartesianIndexing
   indexing.cartesianToMatrixIndexing
   patches.Patches
   subregions.extract_quadrilateral_ROI
