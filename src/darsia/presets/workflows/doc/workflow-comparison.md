# Comparison Workflow

Module: `darsia.presets.workflows.user_interface_comparison`

## Main flags
- `--events`
- `--wasserstein-compute`
- `--wasserstein-assemble`

Exactly one comparison mode is expected per run.

## Typical commands
```bash
python -m darsia.presets.workflows.user_interface_comparison --events --config /abs/path/multi.toml
python -m darsia.presets.workflows.user_interface_comparison --wasserstein-compute --config /abs/path/multi.toml
```
