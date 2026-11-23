##############
Reading images
##############

.. currentmodule:: darsia

:func:`imread` is the unified entry point. It reads paths (a file, a folder, a list
of files) and the bytes of an image file; for paths it dispatches on the file
extension to a format-specific reader (also scalar grids in CSV format). It returns
an :class:`Image`; a folder or a sequence of paths gives one space-time image. Physical dimensions, acquisition time and
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
   exif_date
