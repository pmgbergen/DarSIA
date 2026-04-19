# GUI Workflow

Module: `darsia.presets.workflows.user_interface_gui`

The GUI is additive: it does not replace CLI workflow modules.

## Capabilities
- Manage and order multiple config files
- Persist config context between GUI sessions and restore via "Load previous session"
- Open context-listed config files directly in editor ("Open in editor" and list double-click)
- Edit TOML content and save back to currently opened path
- Switch GUI mode between Config Editor, Dashboard, Batch Monitor, and Streaming monitor
- Switch visual theme (System/Light/Dark) with optional `sv_ttk` support
- Run setup/calibration/analysis/comparison/actions
- Run helper actions (ROI helper)
- Run utils actions: download/cache data, export/import calibration bundle, protocol-time media generation (MP4/GIF)
- Enable analysis streaming from Analysis and inspect latest streamed images by key
- Monitor live analysis batch progress (text + progress bar) independently of image streaming
- View richer execution logs (workflow start details and workflow-specific completion)
- Show blocking terminal-state dialogs for workflow completion (`Done`) and failure (`Error`) while still writing terminal-state entries to the execution log
- Inject custom Rig class using `module.path:ClassName`
- Abort active workflow process from the UI
- Warn when cached session config files are no longer available during restore
- Preview calibration import conflicts and choose overwrite-all, skip-all, or abort before import

## Helper tab
- `ROI`: launch interactive ROI helper from `[helper.roi]` config.
- `ROI Viewer`: launch interactive ROI viewer for registry entries from
  `[helper.roi_viewer]` or shorthand `[helper] data = [...]`.
- `ResultReader`: launch interactive scalar result browsing from `[helper.results]`.
- `Show plots`: parity option with other workflow tabs.

## Batch monitor tab
- Focused on **analysis** workflows.
- Displays:
  - current analysis step
  - current image path
  - processed image count (`current/total`) and percentage
  - elapsed time for last processed image
  - elapsed time for current step
  - overall elapsed runtime
  - estimated remaining runtime (rolling average over recent images)
- Includes a horizontal progress bar driven by processed vs total image count.
- For non-analysis workflows (or idle state), the monitor shows inactive status.

## Dependency
`tkinter` is required in the Python environment.

Optional:
- `sv_ttk` for richer light/dark theme support.

For Rig extensibility, see [custom-rig.md](./custom-rig.md).
