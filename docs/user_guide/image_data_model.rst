####################
The image data model
####################

.. currentmodule:: darsia

Everything in DarSIA revolves around :class:`Image`: an array of pixel values
that also *knows what it represents physically* -- its extent in metres, its
acquisition time, whether it is a scalar field or an RGB photograph, and how its
voxel indices map to Cartesian coordinates.

Image types
===========

============================  ====================================================
:class:`Image`                the general ``(n+1)``-dimensional container: ``n``
                              spatial axes plus an optional time axis, scalar or
                              vectorial range
:class:`ScalarImage`          convenience subclass for single-channel data
                              (concentrations, saturations, depth maps)
:class:`OpticalImage`         convenience subclass for photographs, with
                              colour-space awareness (``to_trichromatic``,
                              ``to_monochromatic``)
============================  ====================================================

Constructing an image
=====================

Most images enter DarSIA through :func:`imread` (see :doc:`reading_images`), but
you can also wrap a NumPy array directly:

.. code-block:: python

   import numpy as np
   import darsia

   arr = np.random.rand(200, 300)
   image = darsia.ScalarImage(arr, width=0.3, height=0.2, dim=2)

   image.show()          # quick matplotlib view
   sub = image.subregion(darsia.make_coordinate([[0.05, 0.05], [0.2, 0.15]]))
   print(image.metadata())

Key ideas
=========

* **Physical size, not pixels.** ``width``/``height`` (or ``dimensions``) give the
  domain its real extent; regions, patches and ROIs are then addressed in metres.
* **Time series.** An image can carry a time axis; ``image.time_slice(i)`` pulls
  out one frame, ``image.time_num`` counts them.
* **Metadata travels with the data.** Corrections that only change geometry update
  the metadata; corrections that change pixels do not (see
  :doc:`../api/corrections`).
* **Arithmetic.** :func:`weight`, :func:`superpose` and :func:`stack` combine
  images while keeping their physical meaning consistent.

See also
========

* :doc:`coordinate_systems` -- voxel/Cartesian conversion and patches
* :doc:`../auto_examples/io/plot_readme_example` -- a worked first example
* :doc:`../api/images` -- full API
