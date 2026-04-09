# Image Selection

DarSIA supports three selection styles and allows combining them.

## 1) Time interval
```toml
[color_paths.data.interval.calibration]
start = "01:00:00"
end = "05:00:00"
num = 5
tol = "00:05:00"
```

## 2) Explicit times
```toml
[color_paths.data.time.snapshots]
times = ["01:00:00", "02:30:00", "04:00:00"]
tol = "00:05:00"
```

## 3) Direct paths (supports glob)
```toml
[color_paths.data.path.selection]
paths = ["DSC00160.JPG", "baseline/*.JPG"]
```

## Recommended reusable registry
Use top-level registry once, then reference keys:
```toml
[data.interval.calibration]
start = "01:00:00"
end = "05:00:00"
num = 5
tol = "00:05:00"

[color_paths]
data = ["calibration"]
```

The same approach applies to analysis sections.
