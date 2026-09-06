##############
Reading images
##############

.. currentmodule:: darsia

:func:`imread` is the unified entry point. It dispatches on the file extension to
a format-specific reader and returns an :class:`Image` (or a list of images when
given a folder or a sequence of paths). Physical dimensions, acquisition time and
correction objects can be supplied as keyword arguments and are applied while
reading.

.. autosummary::
   :toctree: generated/

   imread

Format-specific readers
=======================

.. autosummary::
   :toctree: generated/

   imread_from_optical
   imread_from_dicom
   imread_from_vtu
   imread_from_numpy
   imread_from_npz
   imread_from_bytes
