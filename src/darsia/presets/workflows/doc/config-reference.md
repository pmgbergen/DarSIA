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
- `[video]`: media utility config for protocol-time ordered MP4/GIF generation.

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

### Segmentation contour options
For each segmentation entry, the following contour styling keys are supported:
- `mode`
- `thresholds`
- `color`
- `alpha`
- `linewidth`

Optional contour-value labels:
- `show_values` (bool, default `false`)
- `value_color` (`[r,g,b]`, default = contour `color`)
- `value_size` (float font scale, default `0.5`)
- `value_alpha` (float in `[0, 1]`, default `1.0`)
- `value_density` (float, default `0.35`)
- `value_min_distance_px` (float, default `40.0`)
- `value_max_per_contour` (int, default `3`)
- `value_format` (string format, default `"{:.2f}"`)

The value-label keys can be provided directly in the segmentation section or nested in
`[analysis.segmentation.values]` (or `[analysis.segmentation.<name>.values]` for
multiple segmentations).

## Notes on legacy vs current naming
- Current rig section is `[rig]`.
- Curvature settings are part of `[corrections.curvature.*]`.
- Prefer data/ROI registries for reusable definitions.

For practical examples, see:
- [Image selection](./image-selection.md)
- [ROI reference](./roi-reference.md)
- [Corrections](./corrections.md)

## Video utility section

Use `[video]` to assemble protocol-time ordered media from stored analysis images.

- `analysis`: source selector (`segmentation`, `fingers`, `cropping`, `mass`, `volume`)
- `[video.output]`: `formats` (`mp4`, `gif`), `fps`, optional `resolution`, `filename`, `codec`, `quality`
- `[video.overlay]`: elapsed-time/note toggles and text-box styling (`font_scale`, colors, position, alpha, padding, etc.)
- `[video.source]` (optional): override source folder and scanning behavior (`folder`, `extensions`, `pattern`, `recursive`)

When no source folder is given, defaults are resolved under `[data.results]`:
- segmentation → `segmentation`
- cropping → `cropped_images`
- fingers → `fingers`

Generated media files are saved to `<results>/videos/`.
