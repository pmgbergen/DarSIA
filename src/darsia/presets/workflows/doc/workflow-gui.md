# GUI Workflow

Module: `darsia.presets.workflows.user_interface_gui`

The GUI is additive: it does not replace CLI workflow modules.

## Capabilities
- Manage and order multiple config files
- Open context-listed config files directly in editor ("Open in editor" and list double-click)
- Edit TOML content and save back to currently opened path
- Switch GUI mode between Config Editor, Dashboard, Batch Monitor, and Streaming monitor
- Switch visual theme (System/Light/Dark) with optional `sv_ttk` support
- Run setup/calibration/analysis/comparison/actions
- Enable prototype segmentation streaming from Analysis and inspect latest streamed images by key
- View richer execution logs (workflow start details and workflow-specific completion)
- Inject custom Rig class using `module.path:ClassName`
- Abort active workflow process from the UI

## Dependency
`tkinter` is required in the Python environment.

Optional:
- `sv_ttk` for richer light/dark theme support.

For Rig extensibility, see [custom-rig.md](./custom-rig.md).
