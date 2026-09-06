#########
Distances
#########

.. currentmodule:: darsia

Image-to-image distances that respect mass transport. :func:`wasserstein_distance`
solves a Beckmann (dynamic optimal-transport) problem; :class:`EMD` wraps a
discrete Earth Mover's Distance. Physical integration of a signal to a mass
requires a :class:`Geometry`.

.. autosummary::
   :toctree: generated/

   wasserstein_distance
   wasserstein_distance_to_vtk
   EMD

Geometries
==========

.. autosummary::
   :toctree: generated/

   Geometry
   PorousGeometry
   WeightedGeometry
   ExtrudedGeometry
   ExtrudedPorousGeometry

Beckmann problem and solvers
============================

.. autosummary::
   :toctree: generated/

   BeckmannProblem
   BeckmannNewtonSolver
   BeckmannBregmanSolver
   BeckmannGproxPGHDSolver
   L1Mode
   MobilityMode

Advanced
========

.. autosummary::
   :toctree: generated/

   BeckmannLinearSolver
   BeckmannLinearSolverFactory
   BeckmannLinearSolverType
   BeckmannCGSolver
   BeckmannAMGSolver
   BeckmannDirectSolver
   BeckmannKSPSolver
   BeckmannKSPFieldSplitSolver
   BeckmannConvergenceCriteria
   BeckmannConvergenceHistory
