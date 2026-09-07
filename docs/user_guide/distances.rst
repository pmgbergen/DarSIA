#########
Distances
#########

.. currentmodule:: darsia

To compare two mass distributions -- two CO2 plumes, an experiment against a
simulation -- a pixel-wise difference is misleading: shifting a plume by one
pixel should be a small change, not a large one. Transport-based distances
measure *how far mass has to move*.

.. code-block:: python

   import darsia

   d = darsia.wasserstein_distance(mass_a, mass_b, method="cv2.emd")

   # DarSIA's own Beckmann solver also returns the optimal transport fields
   d, info = darsia.wasserstein_distance(
       mass_a, mass_b, method="newton",
       options={"num_iter": 100, "return_info": True},
   )

============================================  ==================================================
:func:`wasserstein_distance`                  unified entry point; ``method`` is ``"cv2.emd"``,
                                              ``"newton"``, ``"bregman"`` or ``"gprox"``
:class:`EMD`                                  discrete Earth Mover's Distance (OpenCV)
:class:`BeckmannProblem`                      the dynamic optimal-transport formulation
:class:`BeckmannNewtonSolver`,                iterative solvers for the Beckmann problem
:class:`BeckmannBregmanSolver`,
:class:`BeckmannGproxPGHDSolver`
============================================  ==================================================

The distributions must have equal (integrated) mass. Turning a raw signal into a
mass needs a :class:`Geometry` (depth and porosity) -- see
:class:`PorousGeometry` and :func:`darsia.wasserstein_distance_to_vtk` for
exporting the solution.

See also
========

* :doc:`../auto_examples/distances/plot_distances`
* :doc:`../auto_examples/distances/plot_wasserstein_split_square`
* :doc:`../api/distances`
