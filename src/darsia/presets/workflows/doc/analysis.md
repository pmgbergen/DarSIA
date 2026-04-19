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
- `[analysis.thresholding]`
- `[analysis.expert_knowledge]`

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

### Mass export selection
`[analysis.mass]` supports optional `export` (list of modes). If omitted (`None`),
only `mass` is exported to disk.

Supported `export` values:
- `mass`
- `rescaled_mass`
- `extensive_mass`
- `extensive_rescaled_mass`
- `saturation_g`
- `rescaled_saturation_g`
- `concentration_aq`
- `rescaled_concentration_aq`

### Segmentation and thresholding mode notes
- Segmentation `mode` supports: `saturation_g`, `concentration_aq`, `mass`,
  `rescaled_mass`, `rescaled_saturation_g`, `rescaled_concentration_aq`.
- Thresholding layer `mode` supports the standard mass products plus
  `rescaled_mass`, `rescaled_saturation_g`, and `rescaled_concentration_aq`.
- `mass` remains a segmentation alias for total mass (`mass_total`).

### Expert knowledge ROI constraints
Optionally constrain where selected scalar fields may be non-zero:
```toml
[analysis.expert_knowledge]
saturation_g = ["roi_key_1", "roi_key_2"]
concentration_aq = ["roi_key_3"]
```

- Keys reference ROIs from the global ROI registry (`[roi.*]`).
- Both lists default to empty (no-op).
- When configured, values outside the union ROI are forced to zero for:
  - `saturation_g` and `rescaled_saturation_g`
  - `concentration_aq` and `rescaled_concentration_aq`

## Output expectations
Depending on enabled tasks, analysis typically produces derived crops, plots/maps, and result artifacts in the configured results directory.

## Suggested streamed image keys
When GUI streaming is enabled for analysis, the latest image payload can include:
- Cropping: `cropping`
- Segmentation: `segmentation`
- Mass: `mass_source_image`, `mass_total`, `mass_g`, `mass_aq`
  - Stored artifacts (only for modes selected in `[analysis.mass].export`; default only `mass`):
    - `mass/<type>_<identifier>/<stem>.<type>`
    - `rescaled_mass/<type>_<identifier>/<stem>.<type>`
    - `extensive_mass/<type>_<identifier>/<stem>.<type>`
    - `extensive_rescaled_mass/<type>_<identifier>/<stem>.<type>`
    - `saturation_g/<type>_<identifier>/<stem>.<type>`
    - `rescaled_saturation_g/<type>_<identifier>/<stem>.<type>`
    - `concentration_aq/<type>_<identifier>/<stem>.<type>`
    - `rescaled_concentration_aq/<type>_<identifier>/<stem>.<type>`
- Volume: `volume_source_image`, `saturation_g`, `concentration_aq`, `saturation_aq`
- Thresholding: `thresholding_source_image`, `thresholding_<layer_name>`
- Fingers: `fingers_source_image`, `fingers_segmentation`, plus ROI-specific
  `fingers_tips_<roi_key>` and `fingers_paths_<roi_key>`

## GUI batch progress telemetry
- Analysis execution also emits dedicated progress events for GUI batch monitoring
  (separate from image streaming).
- Progress events include analysis step transitions plus per-image progress with:
  current image path, processed/total counters, last-image runtime, and step elapsed time.

## Related guides
- [Workflow analysis](./workflow-analysis.md)
- [Finger analysis](./finger_analysis.md)
- [ROI reference](./roi-reference.md)
- [Config reference](./config-reference.md)
