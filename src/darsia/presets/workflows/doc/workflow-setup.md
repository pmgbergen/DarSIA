# Setup Workflow

Module: `darsia.presets.workflows.user_interface_setup`

## Main flags
- `--all`
- `--depth`
- `--segmentation`
- `--facies`
- `--rig`
- `--delete`
- `--show`

## Purpose
- Build depth map artifacts
- Generate/validate labels and facies mapping
- Setup rig and correction pipeline artifacts

## Typical command
```bash
python -m darsia.presets.workflows.user_interface_setup --all --config /abs/path/common.toml /abs/path/run.toml
```

See [config reference](./config-reference.md).
