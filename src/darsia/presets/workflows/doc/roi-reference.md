# ROI Reference

ROIs are used in mass/volume/fingers workflows and optionally for color-path calibration.
They can also be used by `[expert_knowledge]` to constrain saturation and
concentration fields (for both analysis and calibration).

## Global ROI registry
```toml
[[roi]]
name = "storage"
corner_1 = [0.0, 0.0]
corner_2 = [2.745, 1.05]
```

Then reference by key:
```toml
[analysis.mass]
roi = ["storage"]
```

### ROI shapes
Every `[[roi]]` entry has a `shape` (default `"box"`):
- `"box"` (default): axis-aligned box from `corner_1`/`corner_2`, as above.
- `"polygon"`: an ordered list of `vertices` (>= 3 physical `[x, y]` pairs),
  masked to its true footprint, not just its bounding box:
  ```toml
  [[roi]]
  name = "wedge"
  shape = "polygon"
  vertices = [[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]
  ```
- `"label"`: a raw segmentation label index, with no material association -
  covers wherever that label appears in the run's labels image:
  ```toml
  [[roi]]
  name = "layer_label_3"
  shape = "label"
  label_index = 3
  ```
- `"facies"`: a named material class, covering wherever that facies ID
  appears in the run's resolved facies image (itself built from `[facies]`'s
  `facies_to_labels_map` during setup):
  ```toml
  [[roi]]
  name = "sand_layer"
  shape = "facies"
  facies_id = 7
  ```
  `"label"`/`"facies"` entries require a loaded labels/facies image;
  `corrections.color.roi`/`corrections.drift.roi` accept `shape = "box"`/
  `"polygon"` entries (a polygon is reduced to its bounding box - the same
  "crop stays box-only" rule `Image.subregion` already follows), but not
  `"label"`/`"facies"`/`"intersection"`/`"union"` (those have no
  well-defined bounding box at the point `corrections.drift` runs, before
  any labels/facies image exists).
- `"intersection"`/`"union"`: combine other named ROIs (any shape,
  including other intersections/unions) via `members` (>= 2 registry
  keys). `"intersection"` keeps only pixels inside every member;
  `"union"` keeps pixels inside any member:
  ```toml
  [[roi]]
  name = "sand_layer_in_storage"
  shape = "intersection"
  members = ["storage", "sand_layer"]
  ```
  This is the general replacement for "restrict a box to one label" -
  define the box and the label (or facies) as their own `[[roi]]` entries,
  then intersect them.

Only consumers that build a boolean mask (currently `[expert_knowledge]`)
support `"label"`/`"facies"`/`"intersection"`/`"union"` shapes.
Crop-style consumers stay box-only: `Image.subregion` only ever takes a
plain box, and `corrections.color.roi`/`corrections.drift.roi` accept
`"box"`/`"polygon"` but reduce either to its bounding box before use.

### Expert knowledge
Top-level `[expert_knowledge]` (not nested under `[analysis]`, since it
also governs calibration), with independent positive ("only trust this
field inside these ROIs") and negative ("always zero this field inside
these ROIs") lists per field - any named ROI, any shape, can be referenced
from either list, or both at once:
```toml
[expert_knowledge]
saturation_g = ["storage"]
concentration_aq = ["storage"]
no_saturation_g = ["sand_layer"]
no_concentration_aq = []
```
If a list is empty (default), that rule is not applied for that field. If
a pixel is covered by both a positive and a negative ROI for the same
field, the negative rule wins (it is always zeroed).

## Inline ROI definitions
Inline blocks are also supported (for example under analysis fingers or color paths).

## ROI helper template output
The ROI helper emits a copy-ready template in ROI registry format:

```toml
[roi.roi_name]
name = "roi_name"
corner_1 = [x1, y1]
corner_2 = [x2, y2]
```

## Coordinate convention
- Coordinates are physical domain coordinates (meters).
- ROI corners are normalized from the two opposite corners.
