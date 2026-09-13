# Preprocessing Workflow

Module: `darsia.presets.workflows.user_interface_preprocessing`

Runs ahead of, and independently from, the main [Setup workflow](./workflow-setup.md)
(depth map, segmentation, facies, rig).

## Main flags
- `--all` (protocol + depth-measurements; excludes crop correction, which is
  always interactive and opt-in)
- `--protocol`
- `--depth-measurements`
- `--crop`
- `--force` (for protocol/depth-measurements overwrite)
- `--show`

## Purpose
- Generate imaging/injection/pressure-temperature protocol CSV templates
- Generate a depth-measurements CSV from a constant value, if
  `[depth].measurements_mode` is `'constant'`
- Interactively set up crop correction (`[corrections.curvature]`)

## Typical command
```bash
python -m darsia.presets.workflows.user_interface_preprocessing --all --config /abs/path/common.toml /abs/path/run.toml
```

See [config reference](./config-reference.md).
