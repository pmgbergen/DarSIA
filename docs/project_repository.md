# A series project folder as a Git repository

A series' DarSIA project folder (the folder that holds `series.toml`, with
`config/`, `protocols/`, `data/`, `geometry_facies/`, `image_selection/`, ...)
can be a Git repository of its own. It then carries the configuration and small
inputs, while raw photographs and results stay elsewhere. Someone who clones the
project points it at the raw data on their own machine with two per-user files;
nothing in the repository has to be edited.

## Three places

| Place | What it holds | In Git? |
| --- | --- | --- |
| **Project folder** (series root) | `series.toml`, native run configurations, protocols, small data files, geometry work, saved image choices | Yes |
| **Raw data root** | The original photographs | No – a shared drive or a local copy |
| **Results and caches** | `results/`, job outputs, local caches | No |

Keep the project folder outside the DarSIA source checkout. DarSIA refuses to
write per-user or generated project files inside its own checkout; any other
Git repository is fine.

Some files under `results/` are inputs to later steps (for example native setup
products, the geometry store and rig models, an active phase model or a frozen
calibration). Either keep them reachable from the clone, regenerate them, or
track exactly those folders (see `.gitignore` below).

## Two per-user files

Both files belong to one user on one machine and never go into a repository.

**`series_registry.toml`** – which projects this user has, and how old paths map
to this machine. Location: `%APPDATA%\DarSIA\series_registry.toml` on Windows,
otherwise `$XDG_CONFIG_HOME/DarSIA/series_registry.toml` or
`~/.config/DarSIA/series_registry.toml` (`DARSIA_SERIES_REGISTRY` overrides it).
Start → *Add existing series folder…* writes the `[[series]]` entry for you.

- `[[series]]` – `id`, `name` and `root` (the project folder) of each project.
- `[storage.path_aliases]` – `"<prefix written in the files>" = "<where it is here>"`.
  An absolute path in the project's files that starts with such a prefix is
  read at the mapped location: native run configurations, saved image choices
  and the geometry export's photograph. Files are never rewritten, so their
  content (SHA) bindings stay valid. Without an alias nothing changes.
- `[storage] cache_root` – where local caches go (default
  `%LOCALAPPDATA%\DarSIA\cache`; `DARSIA_FLUIDFLOWER_CACHE` overrides it).

**`stores.toml`** – only needed to run shared jobs. Location:
`%LOCALAPPDATA%\DarSIA\stores.toml` on Windows, otherwise
`$XDG_CONFIG_HOME/darsia/stores.toml` or `~/.config/darsia/stores.toml`. It names
this machine (`[machine] id`) and, per series, where its roots are here
(`[stores.<series-id>]`: `series`, `results` and every root declared in
`series.toml [roots]`, e.g. `raw`). It must be on a local disk, outside any
repository and series folder. Workers find job inputs through this file, not
through `[storage.path_aliases]`. Creating a job reads
`series.toml [roots]` as written, so a clone whose raw data lives elsewhere
edits `[roots] raw` locally (without committing it) and regenerates its
execution plans; job folders belong on an SMB/NFS share outside the repository.

## Relative paths

In the native run configurations of a project, relative paths in `[data]`
(`folder`, `folders`, `results`), `[protocols]` (protocol files and their folder
keys), `[facies]`, `[labeling]`, `[depth]`, `[rig]` and `[color_to_mass]` are
read against the project folder, not the working directory. `[data] baseline`
stays relative to the image folders. Absolute paths are read as written (or
through an alias). New image-selection saves store paths inside the project
folder relative to it; older saves with absolute paths still validate.

## Template: a cloned project with raw data elsewhere

```text
git clone <project-repository-url> C:/Users/<you>/<series>-project
```

`series_registry.toml`:

```toml
schema_version = 1

[[series]]
id = "<series-id>"
name = "<Series name>"
root = "C:/Users/<you>/<series>-project"

[storage]
cache_root = "D:/<you>/darsia-cache"

[storage.path_aliases]
# prefix written in the project's files   =   where it is on this machine
"//<nas>/<share>/<series>/DarSIA" = "C:/Users/<you>/<series>-project"
"//<nas>/<share>/<rig>/Raw data" = "D:/<you>/<series>-raw"
```

`stores.toml` (only for shared jobs):

```toml
[machine]
id = "<machine-name>"

[stores.<series-id>]
series = "C:/Users/<you>/<series>-project"
raw = "D:/<you>/<series>-raw"
results = "D:/<you>/darsia-results"
```

## Recommended `.gitattributes` and `.gitignore`

`.gitattributes` – no end-of-line conversion, so configurations, protocols and
manifests that DarSIA binds by SHA keep identical bytes in every clone:

```text
* -text
```

`.gitignore`:

```text
# Outputs, job state and caches stay outside the repository
/results/
/jobs/
/geometry_facies/cache/
# Per-user files
/config/phase_model_studio_reviewer.json
# Temporary files written next to saved files
.*.tmp
*.partial
```

To keep one needed folder under `results/` in Git, replace `/results/` by
`/results/*` and un-ignore each level down to that folder, for example
`!/results/<sub>/`, `/results/<sub>/*`, `!/results/<sub>/<folder>/`. Large
binary inputs (such as `.npz` files under `geometry_facies/`) are better kept
with Git LFS. Keep other repositories (for example series-specific scripts)
as separate repositories, not inside the project folder.
