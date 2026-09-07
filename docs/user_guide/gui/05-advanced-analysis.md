# 5. Advanced Analysis

This chapter covers the features `sample_config.toml` uses only partially (or not
at all), for when the [Quickstart walkthrough](02-quickstart-walkthrough.md)
isn't enough.

## Segmentation contours

`[analysis.segmentation]` draws one or more threshold-contour layers over a
scalar field. Each layer is an entry in `[[analysis.segmentation.config]]`:

```toml
[[analysis.segmentation.config]]
name = "gas"
label = "CO2(g)"
mode = "rescaled_saturation_g"
thresholds = [0.01, 0.3, 0.6, 0.9]
color = [0, 0, 150]
alpha = [0.25, 0.5, 0.75, 1.0]
linewidth = 16
```

Supported `mode` values: `saturation_g`, `concentration_aq`, `mass` (alias
for total mass), `rescaled_mass`, `rescaled_saturation_g`,
`rescaled_concentration_aq`, and `color.<id>` (a named entry from the
[color embedding registry](04-configuration-reference.md#color-embedding-registry)).

Optional contour-value labels (printed directly on the contour lines), either
flat or nested under `[analysis.segmentation.config.values]`:

```toml
[analysis.segmentation.config.values]
"values.value_color" = "0, 0, 0"
"values.value_size" = "0.5"
"values.value_alpha" = "1.0"
"values.value_density" = "0.35"
"values.value_min_distance_px" = "40.0"
"values.value_max_per_contour" = "3"
```

| Key | Meaning | Default |
|---|---|---|
| `show_values` | Print threshold values on contours | `false` |
| `value_color` | Label text color | contour `color` |
| `value_size` | Font scale | `0.5` |
| `value_alpha` | Label opacity | `1.0` |
| `value_density` | How densely to place labels along a contour | `0.35` |
| `value_min_distance_px` | Minimum spacing between labels | `40.0` |
| `value_max_per_contour` | Cap labels per contour | `3` |
| `value_format` | `str.format` spec | `"{:.2f}"` |

Optional smoothing of the drawn contour, shared with `analysis.mass` and
`analysis.fingers` (same block shape as `sample_config.toml`'s
`contour_smoother_selection`):

```toml
[analysis.segmentation.config.contour_smoother_selection]
"contour_smoother_selection.contour_smoother_options.epsilon" = "0.01"
"contour_smoother_selection.contour_smoother_options.use_ratio" = true
"contour_smoother_selection.contour_smoother_options.sigma" = "2.0"
"contour_smoother_selection.contour_smoother_options.window_length" = "21"
"contour_smoother_selection.contour_smoother_options.polyorder" = "3"
```

## Thresholding

`[analysis.thresholding]` produces binary/multi-band mask overlays from a
scalar field — useful for a quick "is CO₂ present here at all" view, distinct
from segmentation's multi-threshold contour lines. Not used by
`sample_config.toml`; minimal shape:

```toml
[analysis.thresholding]
data_selection = ["baseline"]
formats = ["jpg", "npz"]

[analysis.thresholding.layers.detected]
mode = "mass_total"        # or saturation_g, concentration_aq, color.<id>, ...
threshold_min = 0.1
threshold_max = 1.0
label = "CO2 detected"
fill = [120, 210, 255]
stroke = [0, 0, 0]
fill_alpha = 0.4
stroke_width = 2
```

JPG and NPZ outputs land in separate subfolders (`<folder>/jpg/`,
`<folder>/npz/`); JPG output overlays the layer's `fill`/`stroke` styling on
the source image. A `[analysis.thresholding.legend]` table (same shape as
`[video.overlay]`) can draw a text-box legend on the JPG output.

## Finger detection

`[analysis.fingers]` detects viscous-fingering features at a fluid interface:

```toml
[[analysis.fingers.config]]
name = "primary"
mode = "mass"
roi = ["calibration"]
threshold = "0.5"
reduce_to_main_contour = true
fill_holes = true
save_result_plots = true
gradient_mode = "saturation_g"
```

Outputs are written under `fingers/tips`, `fingers/fjords`, and
`fingers/paths`; non-image formats are skipped for finger plots. Practical
guidance: start with a broad ROI, narrow it once you've visually confirmed
where fingers actually occur, and tune `threshold` against a handful of known
snapshots before running the full batch.

## Expert-knowledge ROI constraints

`[analysis.expert_knowledge]` forces selected scalar fields to zero outside a
union of ROIs — useful when you know CO₂ physically cannot reach certain
regions of the rig, and want to suppress spurious signal there:

```toml
[analysis.expert_knowledge]
saturation_g = ["storage"]
concentration_aq = ["storage"]
```

Both lists default to empty, which is a strict no-op. When set, this affects
`saturation_g`/`rescaled_saturation_g` and
`concentration_aq`/`rescaled_concentration_aq`.

## Calibration modes

`color_path.calibration_mode` (used via `[calibration.color.color_path]`)
controls how the color embedding is fit:

- **`auto`** (used by `sample_config.toml`) — fits automatically from the images
  selected via `data`/`data_selection`, using `histogram_weighting` (e.g.
  `"wls"` — weighted least squares) to decide how much each pixel's histogram
  bin contributes. Start here; it needs no manual input beyond good ROIs and
  a representative calibration image set.
- **`manual`** (legacy alias `mode_calibration`) — you supply calibration
  parameters directly rather than fitting them. Use this when auto-fitting
  produces a poor result (e.g. too few clean calibration images) and you have
  known-good values from a previous run or from `[utils.calibration]`
  bundle import.

`calibration.mass.mode` is the analogous manual/automatic switch for the
mass calibration step; `sample_config.toml` uses `manual` with a hand-picked
`threshold = 0.2`.

`ignore_baseline_spectrum` (`none`/`baseline`/`expanded`) controls whether —
and how much of — the baseline (pre-injection) image's own color spectrum is
excluded from the fit; useful when the baseline itself has a slight
non-uniform tint that shouldn't be mistaken for a calibration signal.

## Related reference material

The original CLI-oriented reference this guide draws from lives in
`src/darsia/presets/workflows/doc/` (see `analysis.md`,
`finger_analysis.md`, `corrections.md`) — useful if you need the exact
`user_interface_*` CLI flags rather than the GUI.
