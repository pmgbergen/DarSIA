# Custom Rig Support

Workflow entry points accept injectable Rig classes.

## Where injection is supported
- setup
- calibration
- analysis
- comparison
- GUI (`module.path:ClassName`)

## Guidance
- Subclass `darsia.presets.workflows.rig.Rig`
- Keep overrides focused on experiment-specific behavior
- Reuse standard workflow entry points instead of duplicating orchestration logic
