# 4. Configuration Reference

Field-by-field reference for the TOML sections the GUI edits, grounded in
`sample_config.toml`. For registry
concepts (why sections reference each other by key), read
[Core concepts](03-core-concepts.md) first.

## `[data]`

Where images live and where results are written.

```toml
[data]
folders = ["/path/to/experiment/images"]
format = "JPG"
baseline = "/path/to/experiment/images/baseline.JPG"
results = "/path/to/experiment/results"
use_cache = false
```

| Key | Meaning |
|---|---|
| `folders` | List of input image folders. |
| `format` | Image file extension to read. |
| `baseline` | Reference (pre-injection) image, used by most corrections. |
| `results` | Output root — every analysis mode writes under here. |
| `use_cache` | Reuse previously-corrected images if present. |

## `[rig]`

```toml
[rig]
width = 0.92
height = 0.55
dim = "2"
```

Physical dimensions of the imaged domain, in meters. `dim` is `"2"` for a
standard 2D FluidFlower; used to convert pixel ROIs/measurements to physical
units everywhere else in the config.

## Corrections

Applied in the order listed in `active`:

```toml
[corrections]
active = ["curvature", "drift", "patchwise_illumination", "resize", "type"]
```

**`[corrections.type]`** — numeric dtype cast:
```toml
[corrections.type]
target_type = "float32"
```

**`[corrections.drift]`** — colorchecker-based drift correction:
```toml
[corrections.drift]
colorchecker = "upper_right"
```

**`[corrections.curvature]`** — dewarp/crop to rig corners, itself with its
own `active` sub-list (only `crop` is used here) and pixel corner
coordinates from the one-time interactive point-picker:
```toml
[corrections.curvature]
active = ["crop"]

[corrections.curvature.crop]
top_left = ["2", "357"]
bottom_left = ["4432", "341"]
bottom_right = ["4400", "7694"]
top_right = ["18", "7668"]
width = 0.92
height = 0.55
in_meters = true
```

**`[corrections.patchwise_illumination]`** — flattens lighting using one or
more baseline images, divided into an `nw`×`nw` patch grid:
```toml
[corrections.patchwise_illumination]
baseline_paths = [
  "/path/to/experiment/images/baseline.JPG",
  "/path/to/experiment/images/baseline_late.JPG",
]
nw = 100
eps = 1e-6
```

**`[corrections.resize]`** — downscale for speed:
```toml
[corrections.resize]
mode = "scale"
scale = 0.25
```

## `[restoration]`

Optional smoothing applied to scalar fields before export:
```toml
[restoration]
method = "volume_average"
active = false
```
`method` is `volume_average` or `tvd`; `ignore` (not set here) can exclude
`boolean_porosity`, `image_porosity`, or `inner_labels` masks from smoothing.

## `[labeling]` and `[facies]`

Map a manually segmented reference image to named geological layers:
```toml
[labeling]
colored_image = "/path/to/experiment/segmented_reference.png"
rtol = 0.001
ensure_connectivity = true
unite_labels = []

[facies]
props = ".../data/facies.csv"

[facies.0]
labels = [1, 2]
[facies.1]
labels = [3]
[facies.2]
labels = [0]
```
`facies.<n>.labels` groups raw segmentation label integers into named facies
(row `n` of `props`'s CSV) — used by `basis = "facies"` color embeddings.

## `[depth]`

```toml
[depth]
measurements = ".../data/constant_depth_measurements.csv"
target_resolution = [500, 1000]
```
Depth measurements CSV plus the grid resolution used when interpolating a
depth map from them.

## `[protocols]`

```toml
[protocols]
imaging_mode = "exif"                 # or "ctime"
injection = "/path/to/experiment/protocols/injection_protocol.csv"
pressure_temperature = "/path/to/experiment/protocols/pressure_temperature_protocol.csv"

[protocols.imaging]
"/path/to/experiment/images" = "/path/to/experiment/protocols/imaging_protocol.csv"
```
`[protocols.imaging]` is **always** a per-folder table — one row per entry in
`[data].folders`, with folder keys that must match exactly (for multiple
folders, a value can also be `["<file>.xlsx", "Sheet1"]`).

## Color embedding registry

```toml
[[color_path]]
name = "color_path"
basis = "facies"        # or "labels" (global is not implemented for path)
reference_label = 2
```

Full field set for `color_path` entries (not all used by `sample_config.toml`):
`mode` (`relative`/`absolute`), `baseline`, `data`, `num_segments`,
`ignore_labels`, `resolution`, `threshold_baseline`, `threshold_calibration`,
`rois`, `ignore_baseline_spectrum` (`none`/`baseline`/`expanded`),
`histogram_weighting`, `calibration_mode` (`auto`/`manual`). Two sibling,
simpler embedding types exist: `[color.channel.*]` (single RGB/HSV/LAB
channel) and `[color.range.*]` (bounding box in color space, 3 `[min, max]`
bounds, `"none"` for open bounds). See
[Advanced analysis](05-advanced-analysis.md#calibration-modes) for
`calibration_mode` guidance.

## `[calibration]`

```toml
[calibration.color]
embedding = "color_path"
data_selection = ["calibration"]
baseline = ["baseline"]
rois = ["calibration"]

[calibration.color.color_path]
num_segments = 3
resolution = 21
calibration_mode = "auto"
threshold_baseline = 0.0
threshold_calibration = 1e-6
ignore_baseline_spectrum = "none"
histogram_weighting = "wls"
ignore_labels = [0]

[calibration.mass]
mode = "manual"
fluid = "co2"
threshold = 0.2
data_selection = ["all"]
rois = ["calibration"]
embedding = "color_path"
```
Mass calibration currently supports only color-path embeddings.

## `[roi]`

```toml
[[roi]]
name = "full"
corner_1 = [0.0, 0.0]
corner_2 = [0.92, 0.55]

[[roi]]
name = "calibration"
corner_1 = [0.01, 0.45]
corner_2 = [0.9, 0.0]
```
Coordinates are physical (meters); a corner pair is normalized regardless of
which corner is listed first. Reference by `name` from `analysis.*.roi`,
`color_path.rois`, `analysis.expert_knowledge.*`.

## `[format]`

```toml
[[format]]
name = "jpg"
type = "jpg"
resolution = [1080, 4]
cmap = "color_path.from_facies.7"
keep_ratio = true
quality = 100

[[format]]
name = "csv"
type = "csv"
resolution = [220, 368]
float_format = "{:.3e}"

[[format]]
name = "npz"
type = "npz"
```
Supported `type` values: `jpg`, `png`, `npz`, `npy`, `csv`. Reference by
`name` from any `[analysis.<mode>].formats` list; output goes to
`<results>/<mode>/<type>_<identifier>/` (or `<type>/` when identifier equals
type, as here).

## `[analysis]`

Each mode owns its own `data_selection` and `formats`.

**`[analysis.mass]`**
```toml
[analysis.mass]
data_selection = ["calibration"]
formats = ["jpg"]
export = [
  "concentration_aq", "mass", "rescaled_concentration_aq",
  "rescaled_saturation_g", "saturation_g",
]
roi = ["full"]
embedding = "color_path"
color = "color_path"
active = ["contour_smoother_selection"]

[analysis.mass.contour_smoother_selection]
active = false
type = "savitzky_golay"
```
`export` controls which scalar products are written (default: `mass` only).
Full list of supported values:
`mass`, `rescaled_mass`, `extensive_mass`, `extensive_rescaled_mass`,
`saturation_g`, `rescaled_saturation_g`, `concentration_aq`,
`rescaled_concentration_aq`.

**`[analysis.segmentation]`** — see
[Advanced analysis](05-advanced-analysis.md#segmentation-contours) for the
full contour-styling field set (thresholds, color, alpha, linewidth, value
labels).

**`[analysis.fingers]`** — see
[Advanced analysis](05-advanced-analysis.md#finger-detection).

**`[analysis.thresholding]`**, **`[analysis.volume]`**, **`[analysis.cropping]`**,
**`[analysis.expert_knowledge]`** — not configured in `sample_config.toml`; see
[Advanced analysis](05-advanced-analysis.md) and the [DarSIA config
reference](https://github.com/pmgbergen/darsia/blob/dev/src/darsia/presets/workflows/doc/config-reference.md)
for their fields.

## `[helper]`

```toml
[helper]
data_selection = ["baseline"]

[helper.roi]
mode = "none"
data_selection = ["baseline"]

[helper.roi_viewer]
data_selection = ["baseline"]

[helper.results]
mode = "mass"
data_selection = ["baseline"]
format = "npz"
roi = ["calibration"]

[helper.color]
data_selection = ["all"]
```
Top-level `[helper] data_selection` is a fallback used when a sub-section
omits its own. `helper.roi.mode` accepts `none`, `concentration_aq`,
`saturation_g`, `mass`/`mass_total`, `mass_g`, `mass_aq`, `rescaled_mass`,
`rescaled_saturation_g`, `rescaled_concentration_aq`.

## `[options]`

```toml
[options.setup]
show_plots = false
[options.calibration]
show_plots = false
[options.analysis]
show_plots = false
random_traverse = false
stream_preview = true
[options.helper]
show_plots = false
```
Per-phase toggles; `stream_preview` feeds the GUI's Streaming Preview dock.

## `[video]`

Builds protocol-time-ordered MP4/GIF media from stored analysis images:
```toml
[video.source]
folder = "cropping/jpg"
extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
recursive = false
sorting = "name"

[video.output]
formats = ["mp4"]
fps = 6.0
codec = "mp4v"
quality = 100

[video.overlay]
show_elapsed_time = false
show_note = true
note = "FluidFlower BC02"
font_scale = 0.5
text_color = [255, 255, 255]
box_enabled = true
box_color = [0, 0, 0]
box_alpha = 0.4
```
Output is written to `<results>/videos/`. Run from **Utils** in the sidebar.

## `[download]`

```toml
[download]
skip_existing = true
data_selection = ["baseline"]
```
Optional utility for pulling remote image data into `[data].folders` before
running the pipeline.

## Where to go from here

[Advanced analysis](05-advanced-analysis.md) covers the fields left out
above: segmentation value labels, thresholding layers, expert-knowledge ROI
constraints, and calibration-mode tradeoffs.
