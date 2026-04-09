# Image Selection

DarSIA supports three selection styles. They can be combined, and each style can be
defined inline or through reusable top-level registries.

## Recommended workflow

1. Define reusable selectors under top-level `[data.interval.*]`, `[data.time.*]`, and
   `[data.path.*]`.
2. Reference selector keys from workflow sections (for example `color_paths.data` or
   analysis data selectors).
3. Use direct inline selectors only for one-off experiments.

## Selection styles

### 1) Time interval
Select a uniformly sampled subset in a time window.

```toml
[color_paths.data.interval.calibration]
start = "01:00:00"
end = "05:00:00"
num = 5
tol = "00:05:00"
```

### 2) Explicit times
Pick specific timestamps directly.

```toml
[color_paths.data.time.snapshots]
times = ["01:00:00", "02:30:00", "04:00:00"]
tol = "00:05:00"
```

### 3) Direct paths (supports glob)
Use exact file names and/or wildcard patterns.

```toml
[color_paths.data.path.selection]
paths = ["DSC00160.JPG", "baseline/*.JPG"]
```

## Registry-based style (recommended)
Define selectors once:

```toml
[data.interval.calibration]
start = "01:00:00"
end = "05:00:00"
num = 5
tol = "00:05:00"

[data.time.qa]
times = ["01:00:00", "02:30:00", "04:00:00"]
tol = "00:05:00"

[data.path.manual]
paths = ["DSC00160.JPG", "baseline/*.JPG"]
```

Reference selector keys from a workflow section:

```toml
[color_paths]
data = ["calibration", "qa", "manual"]
```

## Practical notes
- Time strings are interpreted relative to the experiment timeline metadata.
- `tol` controls matching tolerance when mapping requested times to available images.
- Path selectors support glob patterns.
- Keep selector names semantic (`calibration`, `reference`, `qa`) for readability.

## Common pitfalls
- Mixing too many broad glob patterns can unintentionally include unrelated images.
- Very small `tol` values can miss nearby snapshots.
- Duplicating selector definitions across files creates drift; prefer shared registries.

## Related guides
- [Introduction](./introduction.md)
- [Config reference](./config-reference.md)
- [Analysis](./analysis.md)

## Minimal end-to-end example

```toml
[data.interval.calibration]
start = "00:30:00"
end = "04:00:00"
num = 8
tol = "00:05:00"

[color_paths]
data = ["calibration"]
```
The same registry pattern applies across setup/calibration/analysis usage.
