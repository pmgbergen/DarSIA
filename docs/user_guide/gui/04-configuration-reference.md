# 4. Configuration Reference

Field-by-field reference for the TOML sections the GUI edits, grounded in
`sample_config.toml`. For registry
concepts (why sections reference each other by key), read
[Core concepts](03-core-concepts.md) first.

Color fields (contour, label, fill, stroke, text and box colors) take `r, g, b` with
values 0-255. Next to the text field, a colored box shows the color; click it to
choose a color in a dialog (color field, RGB and HSV spin boxes, hex code, screen
picker). Typing `r, g, b` works as before.

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
```
The physical size of the cropped region is `[rig]` width/height, the only place for
it. Older configs may still have `width`/`height` in the crop section; they must equal
`[rig]` (otherwise the rig setup stops with an error) and can be removed.

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

Optional restoration of the signal: a list of steps, applied in the given order:
```toml
[restoration]
active = false

[[restoration.steps]]
method = "volume_average"
options = { rev_size = 0.005 }
```
A step's `method` is `resize` (coarser resolution by area averaging, for all later
steps and the rest of the analysis), `volume_average` or `tvd`; `ignore` (not set
here) can exclude `boolean_porosity` or `image_porosity` masks from the averaging and
denoising steps. `image_porosity` excludes the grains and the band at label boundaries
(`[image_porosity] boundary_band`); with `[image_porosity] method = "none"` only the
band. `volume_average` (Mask, default `image_porosity`) and the porosity
weights and masks need **Setup > Image porosity** to have run; `[image_porosity]
method` (default `"none"`: no detection; `"color_statistics"`: patch-wise colour
statistics interpolated smoothly; `"kmeans"`: legacy) sets how the image porosity is
computed, `regions` (`"labels"`, default; `"facies"`; `"global"`: label agnostic) within
which regions pixels are compared, and
the colour calibration and the illumination correction use it only with their
`mask = "boolean_porosity"` (default `"none"`). See the workflows configuration reference
for the options of each method and the `[image_porosity]` section, and the gallery
example "Image porosity: which pixels show the pore space" for the options on an image.

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

The pressure-temperature protocol is used for **every** fluid: densities (and, for
CO₂, solubility) depend on pressure and temperature, including their gradients over
the rig height (hydrostatic pressure, temperature gradient).

## Color embedding registry

```toml
[[color_path]]
name = "color_path"
basis = "facies"        # or "labels" (global is not implemented for path)
reference_label = 2
```

The `name` is the registry key: other keys refer to it (`[calibration.color] embedding`,
`[analysis.mass] color`, a format's `cmap`) and it names the results folder
`<results>/color/color_path/<name>`. It may only contain letters, digits, `_` and `-`
(no spaces or dots). Renaming an embedding renames its folder; move an existing
calibration along.

`reference_label` picks the label whose colour path is the default colormap
(`cmap = "color_path.<name>"`) and against which the start values of a new
interpretation calibration are scaled. Choose a facies where the tracer shows its full
colour range. An existing interpretation calibration does not depend on it. Once the
colour paths are calibrated, the GUI shows a dropdown of the calibrated labels with
their colour bands; a label without a colour path is marked red.

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
threshold = 0.2
data_selection = ["all"]
rois = ["calibration"]
embedding = "color_path"
active = ["contour_smoother_selection"]

[calibration.mass.contour_smoother_selection]
active = false
type = "savitzky_golay"
```
Mass calibration currently supports only color-path embeddings.

**Analysis type.** The top-level `[physics]` section decides how mass is computed in
both mass calibration and mass analysis:

```toml
[physics]
fluid = "co2"    # default when the section is absent; or "tracer"
```

`"co2"` is the multiphase analysis (dissolved concentration, gas saturation, flash);
`"tracer"` is single-phase (no gas phase). The choice selects the interpretation of the
calibration (quantity, closure), which is stored in the calibration folder
(`interpolation/mass` for CO2, `interpolation/concentration_aq` for tracer) and its
metadata, and checked on load. Color paths do not depend on it.

For a tracer, the interactive calibration has no flash sliders and no gas saturation: you
tune the concentration at the key colors (0 to 1) and one contour threshold for
`concentration_aq`, and compare the detected volume with the injected volume. This needs a
volumetric injection rate (`rate_mL/s`, `rate_mL/min` or `rate_mL/hr`) in the injection
protocol; the density column is optional. `volume_unit` (`"auto"`, `"mL"`, `"L"` or `"m3"`)
sets the unit of the volume axis, like `mass_unit` for CO2. The tracer mass is the volume
times the water density, evaluated at the local pressure and temperature.

The sidebar names are the same for every fluid, because the steps do the same thing:
**Calibration > Interpretation** calibrates the color-to-signal interpretation of the
color path (anchored on the injected mass for CO2, the injected volume for tracer), and
**Analysis > Field maps** computes the dense per-pixel fields and their integrals. What a
calibration represents is recorded in its metadata and folder. The fluid only changes the
forms (applied when a config is loaded or saved): for a tracer, the gas-only
`[expert_knowledge]` fields `saturation_g` / `no_saturation_g` are hidden (shown only if
they are set, so that they can be removed).

**Resolution level.** `resolution_level` (1, 2, 4, 8 or 16; default 4) reduces the images
per axis for the calculations of the interactive calibration tool: colours are averaged
over the reduced pixels before they are interpreted. Every Update analyses all calibration
images, so level 4 is about 16 times faster than full resolution (level 1). The mass
curves of a reduced level deviate slightly from full resolution (about 1 % at level 4 in
our test), because the signal is a nonlinear function of the colour. Tune at a reduced
level, then switch to level 1 with the **Resolution** slider of the tool (applied by
**Update**) to check the curves before saving. The stored calibration and the analysis do
not depend on the level. With a restoration, only level 1 is available.

**Ignore mask.** `ignore_mask` (`"norm"` or `"spectrum"`; default `"norm"`) decides which
pixels count as showing no signal; they get the value of the clean state instead of being
projected onto the colour path. `"norm"` ignores pixels whose colour change against the
baseline has a norm below 0.1, the same limit for all labels. `"spectrum"` ignores pixels
whose colour change lies in the colour spectrum of the baseline of their label, which is
measured in the colour path calibration: the limit then follows the actual noise of each
label. The calibration tool uses the choice of the config and stores it with the
calibration; the analysis uses the stored one. The two masks differ in about 1 % of the
pixels, and the detected mass changed by about 1 % in our test, so check (and if needed
tune) a calibration after changing the mask. It applies to linear and spline evaluation
with a baseline spectrum.

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
cmap = "color_path.color_path.7"  # color_path.<embedding_id>[.<label>]
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

`cmap` (jpg/png) is `matplotlib.<name>` or `color_path.<embedding_id>[.<label>]`.
Recommended: sequential `viridis`, `cividis`, `plasma`, `inferno`, `magma`,
`turbo`, `gray`, `gray_r` for non-negative quantities (mass, concentration, saturation);
diverging `coolwarm`, `seismic`, `berlin`, `managua` for signed quantities such
as differences. Any other matplotlib name works too. Calibrated colour paths
reproduce the dye colours seen in the rig. In the GUI, `cmap` is a dropdown
with a colour band per option: the maps above and the calibrated colour paths
of the saved config. Other names can be typed in (unknown names turn red);
empty means the default colormap.

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
```
`export` controls which scalar products are written (default: `mass` only).
Full list of supported values:
`mass`, `rescaled_mass`, `extensive_mass`, `extensive_rescaled_mass`,
`mass_g`, `rescaled_mass_g`, `extensive_mass_g`, `extensive_rescaled_mass_g`,
`mass_aq`, `rescaled_mass_aq`, `extensive_mass_aq`, `extensive_rescaled_mass_aq`,
`saturation_g`, `rescaled_saturation_g`, `concentration_aq`,
`rescaled_concentration_aq`. `mass_g` is the mass in the gas phase and `mass_aq` the
dissolved mass (together the total `mass`); the gas modes need a fluid with a gas phase.

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
`rescaled_mass_g`, `rescaled_mass_aq`, `rescaled_saturation_g`,
`rescaled_concentration_aq`.

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
