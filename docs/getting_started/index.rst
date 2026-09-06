###############
Getting started
###############

.. note::

   This section is being rebuilt. For now, see the :doc:`README
   <../about/index>` on GitHub for installation instructions.

Installation
============

DarSIA requires **Python 3.12 or newer**. It is not yet on PyPI, so install it
from a clone of the repository.

Using `uv <https://github.com/astral-sh/uv>`_ (recommended)::

   git clone https://github.com/pmgbergen/darsia.git
   cd darsia
   uv sync --extra dev

Using ``pip``::

   git clone https://github.com/pmgbergen/darsia.git
   cd darsia
   pip install -e ".[dev]"

To build the documentation as well, add the ``docs`` extra
(``uv sync --extra docs`` or ``pip install -e ".[dev,docs]"``).

Quickstart
==========

.. code-block:: python

   import darsia

   # Read an image and attach physical dimensions (in metres).
   image = darsia.imread("baseline.jpg", width=0.92, height=0.55)
   image.show()

   # Work in physical coordinates.
   roi = image.subregion(coordinates=[[0.1, 0.1], [0.4, 0.3]])
   roi.show()

A five-minute tour and a "load your first image" walkthrough will follow here.
