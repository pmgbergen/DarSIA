.. _gui_guide:

#######################
The graphical interface
#######################

DarSIA ships a Qt desktop application that wraps the configuration-driven
workflow pipeline: you fill in a TOML config and click buttons instead of
writing a driver script. Launch it with::

   uv run python -m darsia.gui

The chapters below are a from-scratch tutorial built around
:download:`sample_config.toml <sample_config.toml>`, a complete FluidFlower
configuration with placeholder paths.

.. toctree::
   :maxdepth: 1

   overview
   01-getting-started
   02-quickstart-walkthrough
   03-core-concepts
   04-configuration-reference
   05-advanced-analysis
   06-troubleshooting-faq
