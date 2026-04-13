# Finger Analysis

This page focuses on configuration and usage of finger detection in analysis.

## Entry point
Finger analysis is run through `user_interface_analysis` with the `--fingers` flag (or `--all`).

## Minimal usage
```bash
python -m darsia.presets.workflows.user_interface_analysis \
  --fingers --config /abs/path/common.toml /abs/path/run.toml /abs/path/analysis.toml
```

## Typical config block
```toml
[analysis.fingers]
# Example values only; choose experiment-appropriate settings.
method = "contour"
threshold = 0.2
roi = ["storage"]
```

## Practical guidance
- Start with broad ROI(s), then narrow after visual inspection.
- Tune threshold values against known snapshots.
- Keep finger settings in `analysis.toml` to avoid cluttering setup/calibration files.

## Related guides
- [Analysis](./analysis.md)
- [Workflow analysis](./workflow-analysis.md)
- [ROI reference](./roi-reference.md)
