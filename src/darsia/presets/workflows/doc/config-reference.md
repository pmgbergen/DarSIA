# Configuration Reference

This is the user-facing map of workflow TOML sections currently loaded by `FluidFlowerConfig` and workflow interfaces.

## Core sections
- `[data]`: image folder(s), baseline image name, results folder, optional cache toggles.
- `[rig]`: geometry metadata (`dim`, `width`, `height`, optional `path`).
- `[corrections]`: type/resize/drift/curvature/color/relative_color/illumination settings.
- `[restoration]`: restoration steps (resize, volume averaging, TVD) in free order.
- `[labeling]`: colored label image and generated labels artifact.
- `[facies]`: facies properties and mapping to labels.
- `[depth]`: depth measurements, depth-map path, and optional interpolation-resolution for depth-grid construction.
- `[protocols]`: imaging, injection, blacklist, pressure_temperature inputs and imaging setup mode.
- `[color]`: centralized color embedding registry (`[color.path.*]`, `[color.range.*]`, `[color.channel.*]`).
- `[calibration]`: calibration entrypoints (`[calibration.color]`, `[calibration.mass]`).
- `[analysis]`: analysis data and feature-specific subsections.
- `[expert_knowledge]`: optional positive/negative ROI constraints for
  saturation/concentration, shared by both analysis and calibration.
- `[format]`: named export-format presets for analysis image outputs.
- `[helper]`: optional helper workflows (ROI, ROI viewer, results, color).
- `[download]`: download utility config (optional).
- `[utils]`: optional utility defaults (calibration bundle import/export paths).

### Protocol setup mode
Optional key under `[protocols]`:

```toml
[protocols]
imaging_mode = "exif" # or "ctime"
```

Default is `"exif"` to match the setup protocol extractor behavior.

`[protocols].imaging` is always a per-folder table — one row per entry in `[data].folders`,
even with a single folder. Folder keys must exactly match `[data].folders`:

```toml
[data]
folders = ["/data/run_a"]

[protocols.imaging]
"/data/run_a" = "/protocols/imaging_a.csv"
```

For multiple folders, add one row per folder:

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

## Restoration section
The restoration acts on the signal, before the flash, and on the images of color
modes. It is a list of steps, applied in the given order; a method may occur more
than once.
```toml
[restoration]
active = true                  # optional; false skips the restoration
ignore = ["boolean_porosity"]  # optional; for all averaging and denoising steps

[[restoration.steps]]
method = "resize"
options = { mode = "scale", scale = 0.2 }

[[restoration.steps]]
method = "tvd"
options = { weight = 0.1, method = "chambolle" }
```
- `method` of a step:
  - `"resize"`: changes the resolution by area averaging. Options as for
    `[corrections.resize]`: `mode = "scale"` with `scale`, or
    `mode = "target_shape"` with `target_shape = [rows, cols]`.
  - `"volume_average"`: averages over a representative elementary volume; option
    `rev_size` in meters.
  - `"tvd"`: total variation denoising; options `method`, `weight` (a number,
    `"image_porosity"` or `"boolean_porosity"`), `max_num_iter`, `eps`, `omega`,
    `regularization`.
- `active` of a step (optional, default true): false skips the step and keeps its
  values.
- `ignore` (optional): list of named ignore masks applied in additive mode.
  - Currently supported:
    - `"boolean_porosity"` (ignore regions outside the boolean porosity mask)
    - `"image_porosity"` (ignore regions with non-positive image porosity)
    - `"inner_labels"` (ignore band around labels boundaries)
- **Resolution:** after a resize step, all later steps and the rest of the analysis
  (flash, expert knowledge, mass) work on the new grid. The rig data they use
  (porosity, ignore masks, labels of expert knowledge regions, pressure and
  temperature) is taken on that grid with `darsia.transfer`: labels and masks at the
  center of each pixel, fields by area averaging. The REV size of a volume averaging
  step is in meters, so it covers the same area on any grid.
- **Single method:** `[restoration] method = "..."` with `[restoration.options]` is
  read as one step (the form before the list of steps). A file uses either the steps
  or a single method, not both.
- **Color modes** (an analysis `mode` naming a color embedding, e.g. in segmentation,
  thresholding or fingers): the embedding's image is restored with the same steps,
  so all mode images share the result grid.

## Shared data registry (recommended)
Define reusable selections in top-level `[data]` subsections:
- `[data.interval.<key>]`
- `[data.time.<key>]`
- `[data.path.<key>]`

Then reference these keys from workflow sections (for example `color.path.<id>.data = ["calib_a"]`).

## ROI registry
Define reusable ROI entries under top-level `[roi.<key>]` and reference keys from:
- `analysis.mass.roi = ["roi_key"]`
- `analysis.volume.roi = ["roi_key"]`
- `analysis.fingers.roi = ["roi_key"]`
- `expert_knowledge.saturation_g = ["roi_key"]`
- `expert_knowledge.concentration_aq = ["roi_key"]`
- `color.path.<id>.rois = ["roi_key"]`
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

Use these identifiers in each `[analysis.<mode>].formats`:
```toml
[analysis.mass]
data_selection = ["analysis_set"]
formats = ["my_npy", "4k"]
```

Outputs are written to `<type>_<identifier>` subfolders (for example `jpg_4k`).

## Analysis subsections

Each analysis mode owns its own `data_selection` and `formats`:

- `[analysis.cropping]`: cropping image selection and output formats
  - `data_selection`: image set to analyze
  - `formats`: output format identifiers
  - `filter_background`: stretch each color channel to its full range in the pictures
    (JPG/PNG, preview); NPZ/NPY exports keep the original values (default `false`)
- `[analysis.mass]`: mass analysis and optional ROIs
  - `data_selection`: image set to analyze
  - `formats`: output format identifiers
  - Optional `export` controls which scalar products are written to disk.
    - Default (`None`): `["mass"]`
    - Supported values:
      - `mass`
      - `rescaled_mass`
      - `extensive_mass`
      - `extensive_rescaled_mass`
      - `mass_g`
      - `rescaled_mass_g`
      - `extensive_mass_g`
      - `extensive_rescaled_mass_g`
      - `mass_aq`
      - `rescaled_mass_aq`
      - `extensive_mass_aq`
      - `extensive_rescaled_mass_aq`
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
    - `<results>/mass_g/{npz,jpg}/`
    - `<results>/rescaled_mass_g/{npz,jpg}/`
    - `<results>/extensive_mass_g/{npz,jpg}/`
    - `<results>/extensive_rescaled_mass_g/{npz,jpg}/`
    - `<results>/mass_aq/{npz,jpg}/`
    - `<results>/rescaled_mass_aq/{npz,jpg}/`
    - `<results>/extensive_mass_aq/{npz,jpg}/`
    - `<results>/extensive_rescaled_mass_aq/{npz,jpg}/`
    - `<results>/saturation_g/{npz,jpg}/`
    - `<results>/rescaled_saturation_g/{npz,jpg}/`
    - `<results>/concentration_aq/{npz,jpg}/`
    - `<results>/rescaled_concentration_aq/{npz,jpg}/`
- `[analysis.volume]`: volume analysis and optional ROIs
  - `data_selection`: image set to analyze
  - `formats`: output format identifiers
- `[analysis.segmentation]`: contour layer configuration(s)
  - `data_selection`: image set to analyze
  - `formats`: output format identifiers
  - `[analysis.segmentation.aqueous]` and `[analysis.segmentation.gas]` define layer thresholds
  - `filter_background`: draw the contours on a copy of the image with each color channel
    stretched to its full range (display only; default `false`)
- `[analysis.thresholding]`: threshold selected analysis modes and export mask previews
  - `data_selection`: image set to analyze
  - `formats`: output format identifiers
  - `filter_background`: as for segmentation, for the overlays and the live preview
  - `[analysis.thresholding.layers.<name>]` defines each threshold layer
- `[analysis.fingers]`: finger detection mode/threshold and optional ROIs
  - `data_selection`: image set to analyze
  - `formats`: output format identifiers
  - `filter_background`: as for segmentation, for the result plots (tips, fjords,
    skeleton, paths); detected fingers and statistics are unchanged

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
    - `rescaled_mass_g`
    - `rescaled_mass_aq`
    - `rescaled_saturation_g`
    - `rescaled_concentration_aq`
  - `data`: selector key or keys resolved from top-level `[data.*]` registry.
- `[helper.color]`: interactive color histogram helper.
  - `data`: selector key(s) resolved from top-level `[data.*]` registry.
  - if omitted, falls back to `[helper] data = ["data_key"]`.
  - supports RGB/HSV histograms and absolute/relative display.
- `[helper.roi_viewer]`: interactive ROI viewer for ROI registry entries.
  - `data`: selector key or keys resolved from top-level `[data.*]` registry.
  - ROI selector in the viewer supports `all`, `none`, and each ROI registry key.
  - Images are downscaled and preloaded in-memory for fast image/ROI switching.
- `[helper.results]`: interactive result reader for scalar analysis artifacts.
  - `data`: selector key(s) resolved from top-level `[data.*]` registry.
  - `mode`: result mode folder name under `<results>/` (for example `rescaled_mass`).
  - `format`: `npz`/`csv` or format registry key resolving to `npz`/`csv`.
  - `cmap` (optional): `matplotlib.<name>`, or the colours of a calibrated colour path, `color_path.<embedding_id>[.<label>]` (default label: the embedding's `reference_label`).
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
- `mass_g`
- `mass_aq`
- `rescaled_mass`
- `rescaled_mass_g`
- `rescaled_mass_aq`
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
- `color.<id>` (named entry from top-level `[color.path.*]`, `[color.range.*]`, `[color.channel.*]`)
- `rescaled_mass`
- `rescaled_mass_g`
- `rescaled_mass_aq`
- `rescaled_saturation_g`
- `rescaled_concentration_aq`

Supported keys:
- `formats` (list of output formats: `["jpg", "npz"]`)
- `filter_background` (bool, default `false`): overlays on a copy of the image with each
  color channel stretched from its minimum and maximum to the full range (display only;
  masks and extents use the original image)
- `[analysis.thresholding.layers.<name>]` (one mask layer per entry):
  - `mode` (legacy mass modes, rescaled modes, `color.<id>`)
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
- JPG and NPZ outputs are stored in separate subfolders: `<results>/thresholding/jpg/` and
  `<results>/thresholding/npz/` (fixed results layout; there is no `folder` key).
- JPG outputs are source-image overlays using each layer’s `fill` and `stroke` styling.
- Legacy `modes` + `thresholds` is still accepted and mapped to default layers.

## Color embedding registry
Define reusable color embeddings in one registry:

```toml
[color.channel.my_channel]
mode = "absolute" # or "relative"
basis = "global"  # labels/facies for channel are not implemented
color_space = "RGB" # one of RGB, BGR, HSV, HLS, LAB
channel = "r"       # channel name depends on color_space; case-insensitive
```

```toml
[color.range.custom_range]
mode = "absolute" # or "relative"
basis = "global"  # labels/facies for range are not implemented
color_space = "HSV" # one of RGB, BGR, HSV, HLS, LAB
range = [[0.2, 0.4], [0.5, "none"], [0.8, "none"]]
```

```toml
[color.path.my_color_path]
mode = "relative" # or "absolute"
basis = "labels"  # or "facies" (global is not implemented for path)
baseline = "baseline_imgs"
data = "calibration_imgs"
num_segments = 1
ignore_labels = [0, 1]
resolution = 51
threshold_baseline = 0.0
threshold_calibration = 1e-3
reference_label = 3
rois = ["main_roi"]
ignore_baseline_spectrum = "expanded" # none | baseline | expanded
histogram_weighting = "threshold"
calibration_mode = "auto" # auto | manual (legacy alias: mode_calibration)
```

- `range` must contain exactly 3 channel bounds.
- Each channel bound is `[min, max]`.
- Use `"none"` for open bounds.
- Hue wrap-around is supported in HSV/HLS by using `min > max`.
### Expert-knowledge options
Use the top-level `[expert_knowledge]` section to constrain where selected
scalar products may be non-zero. It is not nested under `[analysis]` since it
also governs calibration.

Supported keys (all lists of ROI registry keys, any ROI shape):
- `saturation_g`: positive constraint - only trust `saturation_g` inside
  the union of these ROIs.
- `concentration_aq`: positive constraint - only trust `concentration_aq`
  inside the union of these ROIs.
- `no_saturation_g`: negative constraint - always zero `saturation_g`
  inside the union of these ROIs.
- `no_concentration_aq`: negative constraint - always zero
  `concentration_aq` inside the union of these ROIs.

Any list left empty (the default) applies no constraint for that field. A
single named ROI may appear in both a positive and a negative list; where
a positive and a negative ROI overlap, the negative rule wins. See
`roi-reference.md` for the `shape` options (`box`/`polygon`/`label`/
`facies`) each referenced ROI may take.

Example:
```toml
[expert_knowledge]
saturation_g = ["roi_key_1", "roi_key_2"]
concentration_aq = ["roi_key_3"]
no_saturation_g = ["sand_layer"]
no_concentration_aq = []
```

Notes:
- Both lists default to empty.
- Empty lists are a strict no-op (analysis behavior is unchanged).
- If provided, values outside the union of the configured ROIs are set to zero.

Bounds restrict the range of the values of `concentration_aq` and
`saturation_g`, one table per field under `[expert_knowledge.bounds]`:
- `lower`: smaller values are raised to it (default `0.0`; `"none"` for no
  bound).
- `upper`: larger values are lowered to it (default: no bound).
- At least one of the two is a number.
- A present table bounds its field; an empty table means `[0, no bound]`,
  e.g. to remove negative concentrations. An optional `active` list, as for
  the corrections, restricts this to the fields listed.
- Bounds act after the flash (and thus after any restoration of the signal),
  before the ROI constraints, so zeroed regions stay zero.

Example:
```toml
[expert_knowledge.bounds.concentration_aq]   # [0, no bound]

[expert_knowledge.bounds.saturation_g]
upper = 1.0
```

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
