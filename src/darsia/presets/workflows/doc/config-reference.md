# Configuration Reference

This is the user-facing map of workflow TOML sections currently loaded by `FluidFlowerConfig` and workflow interfaces.

## Core sections
- `[data]`: image folder(s), baseline image name, results folder, optional cache toggles.
- `[rig]`: geometry metadata (`dim`, `width`, `height`, optional `resolution`, optional `path`).
- `[corrections]`: type/resize/drift/curvature/color/relative_color/illumination settings.
- `[restoration]`: restoration method and method-specific options.
- `[labeling]`: colored label image and generated labels artifact.
- `[facies]`: facies properties and mapping to labels.
- `[depth]`: depth measurements and depth-map path.
- `[protocols]`: imaging, injection, blacklist, pressure_temperature inputs and imaging setup mode.
- `[color_paths]`: color-path calibration data selection and options.
- `[color_to_mass]`: color-to-mass calibration options.
- `[analysis]`: analysis data and feature-specific subsections.
- `[download]`: download utility config (optional).
- `[utils]`: optional utility defaults (calibration bundle import/export paths).

### Protocol setup mode
Optional key under `[protocols]`:

```toml
[protocols]
imaging_mode = "exif" # or "ctime"
```

Default is `"exif"` to match the setup protocol extractor behavior.

For multi-folder input via `[data].folders`, configure one imaging protocol per folder:

```toml
[data]
folders = ["/data/run_a", "/data/run_b"]

[protocols.imaging]
"/data/run_a" = "/protocols/imaging_a.csv"
"/data/run_b" = ["/protocols/imaging_b.xlsx", "Sheet1"]
```

## Utils section
Optional keys for utility workflows:

```toml
[utils.calibration]
export_bundle = "/absolute/path/to/calibration_bundle.zip"
import_bundle = "/absolute/path/to/calibration_bundle.zip"
```

Legacy flat keys are also supported:

```toml
[utils]
export_calibration_bundle = "/absolute/path/to/calibration_bundle.zip"
import_calibration_bundle = "/absolute/path/to/calibration_bundle.zip"
```
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
- `color_to_mass.rois = ["roi_key"]`

## Analysis subsections
- `[analysis.data]`: selected analysis image set
- `[analysis.cropping]`: cropping image selection and output formats (`formats = ["npz", "jpg"]`)
- `[analysis.segmentation]`: contour config(s)
- `[analysis.mass]`: mass analysis and optional ROIs
- `[analysis.volume]`: volume analysis and optional ROIs
- `[analysis.fingers]`: finger detection mode/threshold and optional ROIs
- `[analysis.thresholding]`: threshold selected analysis modes and export mask previews

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

### Thresholding options
Use `[analysis.thresholding]` (only under `[analysis]`) to threshold selected scalar
analysis outputs. Supported `modes`:
- `concentration_aq`
- `saturation_g`
- `mass_total`
- `mass_g`
- `mass_aq`

Supported keys:
- `formats` (list of output formats: `["jpg", "npz"]`)
- `folder` (output folder, defaults to `<results>/thresholding`)
- `[analysis.thresholding.layers.<name>]` (one mask layer per entry):
  - `mode` (`concentration_aq`, `saturation_g`, `mass_total`, `mass_g`, `mass_aq`)
  - `threshold_min` (float)
  - `threshold_max` (float)
  - `label` (string)
  - `fill` (`[r,g,b]`)
  - `stroke` (`[r,g,b]`)
  - `fill_alpha` (float in `[0, 1]`)
  - `stroke_width` (int `>= 0`)
- `[analysis.thresholding.legend]` (compact text-box style keys aligned with
  `[video.overlay]`):
  - `show`
  - `font_scale`
  - `thickness`
  - `line_spacing`
  - `position`
  - `text_color`
  - `box_enabled`
  - `box_color`
  - `box_alpha`
  - `box_padding`

Notes:
- JPG and NPZ outputs are stored in separate subfolders: `<folder>/jpg/` and `<folder>/npz/`.
- JPG outputs are source-image overlays using each layer’s `fill` and `stroke` styling.
- Legacy `modes` + `thresholds` is still accepted and mapped to default layers.

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

- `[video.output]`: `formats` (`mp4`, `gif`), `fps`, optional `resolution`, `filename`, `codec`, `quality`
- `[video.overlay]`: elapsed-time/note toggles and text-box styling (`font_scale`, colors, position, alpha, padding, etc.)
- `[video.source]` (required): source folder and scanning behavior (`folder`, `extensions`, `pattern`, `recursive`)

Generated media files are saved to `<results>/videos/`.
