# ROI Reference

ROIs are used in mass/volume/fingers workflows and optionally for color-path calibration.
They can also be used by analysis expert knowledge to constrain saturation and
concentration fields.

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

Expert-knowledge usage:
```toml
[analysis.expert_knowledge]
saturation_g = ["storage"]
concentration_aq = ["storage"]
```

If these lists are empty (default), expert knowledge is not applied.

## Inline ROI definitions
Inline blocks are also supported (for example under analysis fingers or color paths).

## Coordinate convention
- Coordinates are physical domain coordinates (meters).
- ROI corners are normalized from the two opposite corners.
