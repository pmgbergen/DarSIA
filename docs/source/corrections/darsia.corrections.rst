Shape and color corrections (:mod:`darsia.corrections`)
=======================================================

.. currentmodule:: darsia.corrections

.. automodule:: darsia.corrections
   :members:
   :undoc-members:
   :show-inheritance:

The base correction
-------------------

All corrections follow the design of a base correction object.

.. autosummary::
   
   basecorrection
  
Color corrections
-----------------
There is two types of corrections: color and shape altering corrections.

.. toctree::
   :maxdepth: 1

   darsia.corrections.color.colorcorrection
   darsia.corrections.color.experimentalcorrection

Shape corrections
-----------------

.. toctree::
   :maxdepth: 1

   darsia.corrections.shape.curvature
   darsia.corrections.shape.deformation
