# GUI Workflow

Module: `darsia.presets.workflows.user_interface_gui`

The GUI is additive: it does not replace CLI workflow modules.

## Capabilities
- Manage and order multiple config files
- Edit TOML content
- Run setup/calibration/analysis/comparison/actions
- Inject custom Rig class using `module.path:ClassName`
- Abort active workflow process from the UI

## Dependency
`tkinter` is required in the Python environment.

For Rig extensibility, see [custom-rig.md](./custom-rig.md).
