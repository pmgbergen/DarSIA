###############
Getting started
###############

DarSIA turns images of a lab or field experiment into quantitative physical
data. This page gets you from a fresh clone to a first result; follow the links
at the end for the concepts and the full workflow.

.. toctree::
   :hidden:

   installation

Install
=======

Python 3.12+, from a clone (see :doc:`installation` for uv, pip, ``petsc4py``
and the GUI):

.. code-block:: bash

   git clone https://github.com/pmgbergen/darsia.git
   cd darsia
   uv sync --extra dev

Your first image
================

.. code-block:: python

   import darsia

   # read a photograph and attach its real-world size (metres)
   image = darsia.imread("baseline.jpg", width=2.8, height=1.5)
   image.show("baseline")

   # a copy with a 10 cm metric grid
   image.add_grid(dx=0.1, dy=0.1).show("with grid")

   # cut a region of interest by physical coordinates ...
   roi = image.subregion(darsia.make_coordinate([[1.5, 0.0], [2.8, 0.7]]))
   roi.show("roi (metres)")

   # ... or by pixel index
   image.subregion((slice(200, 900), slice(1500, 2800))).show("roi (pixels)")

   print(image.metadata())

Every :class:`~darsia.Image` carries its physical extent, acquisition time and
coordinate system, so the rest of DarSIA can work in metres and seconds rather
than pixels and frame numbers.

A first analysis
================

The typical task is turning a *sequence* of images into a concentration or mass
field, by comparing each frame to a baseline:

.. code-block:: python

   import darsia

   analysis = darsia.ConcentrationAnalysis(
       base=baseline,
       signal_reduction=darsia.MonochromaticReduction(color="red"),
       restoration=darsia.TVD(),
       model=darsia.LinearModel(scaling=4.0),
   )
   concentration = analysis(test_image)

See :doc:`../user_guide/concentration_analysis` for what each stage does and
:doc:`../auto_examples/analysis/plot_co2_analysis` for a runnable version.

Where to next
=============

* :doc:`../user_guide/index` -- the concepts, one subsystem at a time.
* :doc:`../auto_examples/index` -- runnable, image-rich examples.
* :doc:`../user_guide/gui/index` -- the point-and-click workflow, no scripting.
* :doc:`../api/index` -- every public class and function.
