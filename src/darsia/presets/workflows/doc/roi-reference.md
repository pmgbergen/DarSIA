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

Expert-knowledge usage (top-level `[expert_knowledge]`, not nested under
`[analysis]`, since it also governs calibration):
```toml
[expert_knowledge]
saturation_g = ["storage"]
concentration_aq = ["storage"]
```

If these lists are empty (default), expert knowledge is not applied.

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
