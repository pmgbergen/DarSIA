# Configuration Reference

This is the user-facing map of workflow TOML sections currently loaded by `FluidFlowerConfig` and workflow interfaces.

## Core sections
- `[data]`: image folder, baseline image name, results folder, optional cache toggles.
- `[rig]`: geometry metadata (`dim`, `width`, `height`, optional `resolution`, optional `path`).
- `[corrections]`: type/resize/drift/curvature/color/relative_color/illumination settings.
- `[restoration]`: restoration method and method-specific options.
- `[labeling]`: colored label image and generated labels artifact.
- `[facies]`: facies properties and mapping to labels.
- `[depth]`: depth measurements and depth-map path.
- `[protocols]`: imaging, injection, blacklist, pressure_temperature inputs.
- `[color_paths]`: color-path calibration data selection and options.
- `[color_to_mass]`: color-to-mass calibration options.
- `[analysis]`: analysis data and feature-specific subsections.
- `[download]`: download utility config (optional).

## Shared data registry (recommended)
Define reusable selections in top-level `[data]` subsections:
- `[data.interval.<key>]`
- `[data.time.<key>]`
- `[data.path.<key>]`

Then reference these keys from workflow sections (for example `color_paths.data = ["calib_a"]`).

## ROI registry
Define reusable ROI entries under top-level `[roi.<key>]` and reference keys from:
- `analysis.mass.roi = ["roi_key"]`
- `analysis.volume.roi = ["roi_key"]`
- `analysis.fingers.roi = ["roi_key"]`
- `color_paths.rois = ["roi_key"]`

## Analysis subsections
- `[analysis.data]`: selected analysis image set
- `[analysis.segmentation]`: contour config(s)
- `[analysis.mass]`: mass analysis and optional ROIs
- `[analysis.volume]`: volume analysis and optional ROIs
- `[analysis.fingers]`: finger detection mode/threshold and optional ROIs

## Notes on legacy vs current naming
- Current rig section is `[rig]`.
- Curvature settings are part of `[corrections.curvature.*]`.
- Prefer data/ROI registries for reusable definitions.

For practical examples, see:
- [Image selection](./image-selection.md)
- [ROI reference](./roi-reference.md)
- [Corrections](./corrections.md)
