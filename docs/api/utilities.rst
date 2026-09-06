#########
Utilities
#########

.. currentmodule:: darsia

Lower-level building blocks used across DarSIA. Most users will not need these
directly, but they are public and stable.

Grids and finite volumes
========================

.. autosummary::
   :toctree: generated/

   Grid
   generate_grid
   FVDivergence
   FVMass
   FVFullFaceReconstruction
   FVTangentialFaceReconstruction
   cell_to_face_average
   face_to_cell

Points, voxels and coordinates
==============================

.. autosummary::
   :toctree: generated/

   Coordinate
   CoordinateArray
   Voxel
   VoxelArray
   VoxelCenter
   VoxelCenterArray
   make_coordinate
   make_voxel
   make_voxel_center
   to_coordinate
   to_voxel
   to_voxel_center

Derivatives
===========

.. autosummary::
   :toctree: generated/

   forward_diff
   backward_diff
   laplace

Interpolation
=============

.. autosummary::
   :toctree: generated/

   polynomial_interpolation
   interpolate_to_image
   interpolate_to_image_from_csv
   interpolate_measurements_2d
   illumination_interpolation

Kernels
=======

.. autosummary::
   :toctree: generated/

   BaseKernel
   GaussianKernel
   LinearKernel

Linear solvers
==============

.. autosummary::
   :toctree: generated/

   Solver
   CG
   Jacobi
   MG

Approximation and acceleration
==============================

.. autosummary::
   :toctree: generated/

   AndersonAcceleration
   ApproximationSpace
   LinearApproximation
   PolynomialApproximationSpace
   RadialPolynomialApproximationSpace

Detection
=========

.. autosummary::
   :toctree: generated/

   detect_color
   detect_value
   detect_closest_point
   orthogonal_colors
   monochromatic_concentration_analysis

Data types and miscellany
=========================

.. autosummary::
   :toctree: generated/

   convert_dtype
   Format
   FeatureDetection
   ConvergenceStatus
   extract_characteristic_data
   hsv_spectrum
   bounding_box
   bounding_box_inverse
   perimeter
   random_patches
   sort_quad
   full_like
   ones_like
   zeros_like
