# ROI Reference

ROIs are used in mass/volume/fingers workflows and optionally for color-path calibration.

## Global ROI registry
```toml
[roi.storage]
name = "Storage"
corner_1 = [0.0, 0.0]
corner_2 = [2.745, 1.05]
```

Then reference by key:
```toml
[analysis.mass]
roi = ["storage"]
```

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
