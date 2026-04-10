# Analysis

This page expands the analysis workflow beyond the CLI flag overview.

## Entry point
Module: `darsia.presets.workflows.user_interface_analysis`

## Typical operations
- Cropping and export of processed images
- Segmentation analysis
- Mass and volume post-processing
- Finger detection and reporting

## Typical command patterns
Mass + segmentation + fingers:
```bash
python -m darsia.presets.workflows.user_interface_analysis \
  --mass --segmentation --fingers \
  --config /abs/path/common.toml /abs/path/run.toml /abs/path/analysis.toml
```

All configured analysis tasks:
```bash
python -m darsia.presets.workflows.user_interface_analysis \
  --all --config /abs/path/common.toml /abs/path/run.toml /abs/path/analysis.toml
```

## Analysis config structure
Commonly used sections:
- `[analysis.data]`
- `[analysis.segmentation]`
- `[analysis.mass]`
- `[analysis.volume]`
- `[analysis.fingers]`

### Segmentation contour labels
Segmentation contour plots can optionally print the contour threshold values directly
on the plot. Configure this per segmentation entry via:
- `show_values`
- `value_color`
- `value_size`
- `value_alpha`
- `value_density`
- `value_min_distance_px`
- `value_max_per_contour`
- `value_format`

These keys support both flat placement in `[analysis.segmentation]` and nested
placement in `[analysis.segmentation.values]` (and similarly for
`[analysis.segmentation.<name>.values]`).

Use registry keys for image selection and ROIs where possible.

## Output expectations
Depending on enabled tasks, analysis typically produces derived crops, plots/maps, and result artifacts in the configured results directory.

## Related guides
- [Workflow analysis](./workflow-analysis.md)
- [Finger analysis](./finger_analysis.md)
- [ROI reference](./roi-reference.md)
- [Config reference](./config-reference.md)
