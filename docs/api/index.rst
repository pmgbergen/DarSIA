#############
API reference
#############

Every public object listed here is importable directly from the top-level
namespace, for example::

   import darsia

   image = darsia.imread("photo.jpg", width=1.0, height=0.6)
   corrected = darsia.ColorCorrection()(image)
   d = darsia.wasserstein_distance(mass_a, mass_b)

The reference is organised by topic. Within each page, the most commonly used
objects are listed first; deeper machinery is grouped under an *Advanced* heading.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Images & I/O
      :link: images
      :link-type: doc

      Physical image containers, coordinate systems, and readers for optical,
      DICOM and simulation data.

   .. grid-item-card:: Corrections
      :link: corrections
      :link-type: doc

      Colour, illumination, curvature, affine and drift corrections.

   .. grid-item-card:: Restoration
      :link: restoration
      :link-type: doc

      Total-variation and H1 denoising, resizing, averaging, binary inpainting.

   .. grid-item-card:: Segmentation
      :link: segmentation
      :link-type: doc

      Watershed-based geometry segmentation and label management.

   .. grid-item-card:: Signal models
      :link: signal_models
      :link-type: doc

      Signal reduction and the model hierarchy mapping signals to concentrations.

   .. grid-item-card:: Colour analysis
      :link: color
      :link-type: doc

      Colour paths, ranges, spectra and label-to-colour maps.

   .. grid-item-card:: Analysis & registration
      :link: analysis
      :link-type: doc

      Concentration analysis, image registration, calibration and managers.

   .. grid-item-card:: Distances
      :link: distances
      :link-type: doc

      Earth Mover's and Wasserstein distances and the Beckmann solvers.

   .. grid-item-card:: Single-image analysis
      :link: single_image_analysis
      :link-type: doc

      Contour extraction and smoothing.

   .. grid-item-card:: Assistants
      :link: assistants
      :link-type: doc

      Matplotlib-interactive helpers for picking ROIs and parameters.

   .. grid-item-card:: Utilities
      :link: utilities
      :link-type: doc

      Grids, finite volumes, derivatives, interpolation, kernels, solvers.

   .. grid-item-card:: Workflow system
      :link: workflows
      :link-type: doc

      The configuration-driven ``presets.workflows`` pipeline.

   .. grid-item-card:: Experimental
      :link: experimental
      :link-type: doc

      ``experiment`` and ``multiphase`` -- unstable, evolving APIs.

.. toctree::
   :hidden:
   :maxdepth: 1

   images
   io
   corrections
   restoration
   segmentation
   signal_models
   color
   analysis
   distances
   single_image_analysis
   assistants
   utilities
   workflows
   experimental
