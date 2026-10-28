# Series setup on a Mac

A Mac with Apple silicon can set up a series (series and project folder,
protocols, image selection, rig geometry and corners, sand map painting and
evidence review) while the Windows computers compute. Everything the Mac saves
lands in the series folder on the shared drive, so a Windows computer opens the
same series afterwards and continues there. Jobs (image preparation, training,
Build run model, Finish setup, calibration and production) stay on Windows: on
a Mac the Jobs page and Finish setup say so instead of starting anything.

## Install and start

1. Install Python 3.12 or newer (for example from python.org).
2. Clone the repository.
3. Connect the shared drive in Finder: Go > Connect to Server, then
   `smb://<server>/<share>`. It opens as `/Volumes/<share>`.
4. Double-click `start_darsia.command` in the clone. The first start creates
   `.venv-fluidflower` next to it and installs DarSIA (a few minutes, needs
   internet); later starts open the window directly and reinstall only after
   `pyproject.toml` changes. `./start_darsia.command studio` opens the Phase
   Model Studio. The output is also kept in `~/Library/Logs/DarSIA`.

## Tell DarSIA where the share is

Saved records name shared files by the share's own path, such as
`\\<server>\<share>\...`. Open Settings > Shared storage on this computer and
add one row per shared root, with the folder where the Mac opens it:

| Shared root | Folder on this computer |
| --- | --- |
| `\\<server>\<share>` | `/Volumes/<share>` |

"Add mounted shares" proposes the connected shares. If the records use more
than one spelling of the same share (a server name and its address, or a drive
letter such as `X:`), add each spelling as its own row with the same folder.
The first row of a folder is the spelling the Mac writes. Then open or register
the series folder on Start from `/Volumes/<share>/...`.

The setting belongs to this user on this computer. Windows computers that open
the shared paths directly need no rows; without rows nothing changes.

## What the Mac writes

Paths below `/Volumes/<share>` are saved with the shared root, so the Windows
computers read them; paths inside the series stay relative. This covers new
series and run files, series locations and storage aliases, protocol workbooks,
image selections, rig frames and crop areas, painting sessions and saved map
versions, review session registries and manifests, and the native settings form.

Do these steps on Windows for now, because their records still hold the full
path of the computer that saved them: saving a painted map as a geometry
revision, Sand map in all runs (transfer, check and adopt), the image
correction source folder and its revisions, the rig point review and static
references, frame quality assessments, scratch reviews, and projects of the
older `series.json` kind.

## Checklist after cloning

Use a test series or a copy, never production data, for the steps that save.

1. `start_darsia.command` creates `.venv-fluidflower` and the DarSIA window opens.
2. Settings > Shared storage on this computer: save the rows, reopen the
   dialog and see them again.
3. Start lists a series registered from `/Volumes/<share>`; choosing a run shows
   its configuration, and Protocols and Image selection read their files.
4. Rig Geometry shows the reference photograph (camera RAW files render).
5. New series on a test folder: open `config/runs/<run>.toml`; image folders,
   results and protocol paths start with `\\<server>\<share>`, and
   `series.toml` holds relative folders.
6. Mark rig corners and save; paint a few strokes in Layers and facies and save
   a map version; save an image selection. Reopen each: no message about a
   changed photograph.
7. Search the test series folder for `/Volumes`; nothing is found.
8. Open an existing evidence review read-only.
9. Jobs and Finish setup say that jobs run on the Windows computers.
10. On a Windows computer, open the test series: the runs, protocols, image
    selection, rig frame and painted map from the Mac are there.
11. Note any message that a file is busy or locked while saving on the share,
    and any window that fails to open; report both with the log from
    `~/Library/Logs/DarSIA`.
