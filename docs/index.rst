:html_theme.sidebar_secondary.remove:

######
DarSIA
######

**DarSIA** -- *Darcy Scale Image Analysis* -- is an open-source Python toolbox for
quantitative image analysis of dynamics in porous media. It provides physically
meaningful image containers, correction and restoration algorithms, segmentation
and registration tools, and ready-made workflows for extracting concentrations,
phase distributions and displacements from laboratory and simulation images.

Supported data includes optical images (PNG, JPG, TIF, ...), DICOM stacks, and
simulation output (VTU and anything :mod:`meshio` can read).

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Getting started
      :link: getting_started/index
      :link-type: doc

      Install DarSIA and run your first analysis in five minutes.

   .. grid-item-card:: :octicon:`book` User guide
      :link: user_guide/index
      :link-type: doc

      Narrative documentation of the image model, corrections, analysis
      workflows and the graphical interface.

   .. grid-item-card:: :octicon:`image` Examples
      :link: auto_examples/index
      :link-type: doc

      Runnable, image-rich examples grouped by topic.

   .. grid-item-card:: :octicon:`code` API reference
      :link: api/index
      :link-type: doc

      Every public class and function, grouped by topic.

   .. grid-item-card:: :octicon:`tools` Contributor guide
      :link: development/index
      :link-type: doc

      Set up a development environment, run the tests, and build the docs.

   .. grid-item-card:: :octicon:`mortar-board` About
      :link: about/index
      :link-type: doc

      How to cite DarSIA, related publications, and the team behind it.

.. toctree::
   :hidden:
   :maxdepth: 1

   getting_started/index
   user_guide/index
   auto_examples/index
   api/index
   development/index
   release_notes/index
   about/index
