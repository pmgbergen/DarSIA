###########
Restoration
###########

.. currentmodule:: darsia

Restoration algorithms alter the smoothness or shape of image data without
changing its physical interpretation: denoising, resampling, spatial averaging
and binary clean-up.

Denoising and regularization
============================

.. autosummary::
   :toctree: generated/

   TVD
   tvd
   split_bregman_tvd
   H1_regularization
   Median

Resizing
========

.. autosummary::
   :toctree: generated/

   Resize
   resize
   equalize_voxel_size
   uniform_refinement

Spatial averaging
=================

.. autosummary::
   :toctree: generated/

   VolumeAveraging
   REV
   volume_average
   porosity_based_averaging

Binary clean-up
===============

.. autosummary::
   :toctree: generated/

   BinaryFillHoles
   BinaryRemoveSmallObjects
   BinaryLocalConvexCover
