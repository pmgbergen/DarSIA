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
- `[format]`: named export-format presets for analysis image outputs.
- `[helper]`: optional helper workflows (currently ROI helper).
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
- `analysis.expert_knowledge.saturation_g = ["roi_key"]`
- `analysis.expert_knowledge.concentration_aq = ["roi_key"]`
- `color_paths.rois = ["roi_key"]`
- `color_to_mass.rois = ["roi_key"]`

## Format registry
Define reusable export presets under top-level `[format.<type>.<identifier>]`.
Supported `<type>` values:
- `jpg`
- `png`
- `npz`
- `npy`
- `csv`

Example:
```toml
[format.jpg.4k]
resolution = [2160, 4096]
cmap = "matplotlib.viridis"

[format.npy.my_npy]
dtype = "np.float32"
```

Use these identifiers from `[analysis].formats`:
```toml
[analysis]
data = ["analysis_set"]
formats = ["my_npy", "4k"]
```

Outputs are written to `<type>_<identifier>` subfolders (for example `jpg_4k`).

## Analysis subsections
- `[analysis.data]`: selected analysis image set
- `[analysis]`: optional `formats` (list of format identifiers from `[format.*.*]`)
- `[analysis.cropping]`: cropping image selection and output formats (`formats = ["npz", "jpg"]`)
- `[analysis.segmentation]`: contour config(s)
- `[analysis.mass]`: mass analysis and optional ROIs
  - Optional `export` controls which scalar products are written to disk.
    - Default (`None`): `["mass"]`
    - Supported values:
      - `mass`
      - `rescaled_mass`
      - `extensive_mass`
      - `extensive_rescaled_mass`
      - `saturation_g`
      - `rescaled_saturation_g`
      - `concentration_aq`
      - `rescaled_concentration_aq`
    - Values are parsed case-insensitively.
  - Outputs are written per selected product with split format folders:
    - `<results>/mass/{npz,jpg}/`
    - `<results>/rescaled_mass/{npz,jpg}/`
    - `<results>/extensive_mass/{npz,jpg}/`
    - `<results>/extensive_rescaled_mass/{npz,jpg}/`
    - `<results>/saturation_g/{npz,jpg}/`
    - `<results>/rescaled_saturation_g/{npz,jpg}/`
    - `<results>/concentration_aq/{npz,jpg}/`
    - `<results>/rescaled_concentration_aq/{npz,jpg}/`
- `[analysis.volume]`: volume analysis and optional ROIs
- `[analysis.fingers]`: finger detection mode/threshold and optional ROIs
- `[analysis.thresholding]`: threshold selected analysis modes and export mask previews
- `[analysis.expert_knowledge]`: optional ROI constraints for saturation and concentration

## Helper subsections
- `[helper.roi]`: interactive ROI assistant on selected data.
  - `mode` (default `none`): one of
    - `none`
    - `concentration_aq`
    - `saturation_g`
    - `mass` (alias for `mass_total`)
    - `mass_total`
    - `mass_g`
    - `mass_aq`
    - `rescaled_mass`
  - `rescaled_saturation_g`
  - `rescaled_concentration_aq`
  - `data`: selector key or keys resolved from top-level `[data.*]` registry.
- `[helper.roi_viewer]`: interactive ROI viewer for ROI registry entries.
  - `data`: selector key or keys resolved from top-level `[data.*]` registry.
  - ROI selector in the viewer supports `all`, `none`, and each ROI registry key.
  - Images are downscaled and preloaded in-memory for fast image/ROI switching.
- `[helper.results]`: interactive result reader for scalar analysis artifacts.
  - `data`: selector key(s) resolved from top-level `[data.*]` registry.
  - `mode`: result mode folder name under `<results>/` (for example `rescaled_mass`).
  - `format`: `npz`/`csv` or format registry key resolving to `npz`/`csv`.
  - `cmap` (optional): `matplotlib.<name>` or `color_path.from_<basis>.<label>`.
  - `roi` (optional): ROI key/list from top-level `[roi.*]`; omitted means full image.

Shorthand is supported for ROI Viewer:
```toml
[helper]
data = ["analysis_imgs"]
```

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

Supported segmentation `mode` values:
- `saturation_g`
- `concentration_aq`
- `mass` (backward-compatible alias for total mass)
- `rescaled_mass`
- `rescaled_saturation_g`
- `rescaled_concentration_aq`

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
- `colorchannel.<name>` (named entry from top-level `[colorchannel.<name>]`)
- `colorrange.<name>` (binary mask from `[colorrange.<name>]`)
- `rescaled_mass`
- `rescaled_saturation_g`
- `rescaled_concentration_aq`

Supported keys:
- `formats` (list of output formats: `["jpg", "npz"]`)
- `folder` (output folder, defaults to `<results>/thresholding`)
- `[analysis.thresholding.layers.<name>]` (one mask layer per entry):
  - `mode` (legacy mass modes, rescaled modes, `colorchannel.<name>`, `colorrange.<name>`)
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

## Named colorchannel section
Define reusable color channels (used by `colorchannel.<name>` modes):

```toml
[colorchannel.my_channel]
color_space = "RGB" # one of RGB, BGR, HSV, HLS, LAB
channel = "r"       # channel name depends on color_space; case-insensitive
```

## Named color-range section
Define reusable binary color ranges (used by `colorrange.<name>` modes):

```toml
[colorrange.custom_range]
color_space = "HSV" # one of RGB, BGR, HSV, HLS, LAB
range = [[0.2, 0.4], [0.5, "none"], [0.8, "none"]]
```

- `range` must contain exactly 3 channel bounds.
- Each channel bound is `[min, max]`.
- Use `"none"` for open bounds.
- Hue wrap-around is supported in HSV/HLS by using `min > max`.
### Expert-knowledge options
Use `[analysis.expert_knowledge]` to constrain where selected scalar products may be
non-zero.

Supported keys:
- `saturation_g` (list of ROI registry keys)
- `concentration_aq` (list of ROI registry keys)

Example:
```toml
[analysis.expert_knowledge]
saturation_g = ["roi_key_1", "roi_key_2"]
concentration_aq = ["roi_key_3"]
```

Notes:
- Both lists default to empty.
- Empty lists are a strict no-op (analysis behavior is unchanged).
- If provided, values outside the union of the configured ROIs are set to zero.

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
