"""Render and export already saved concentration fields and scalar mass tables."""
from __future__ import annotations
import hashlib
import json
import subprocess
from pathlib import Path
import cv2
import numpy as np

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)

def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()

def render_frame(path, rgb, c, av, gv, pv, row, title, limit):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(13, 4.4), layout="constrained"); FigureCanvasAgg(fig)
    photo, field = fig.subplots(1, 2)
    photo.imshow(rgb); photo.set_title("Corrected photograph"); photo.axis("off")
    image = field.imshow(np.ma.array(c, mask=av <= 0), vmin=0, vmax=limit, cmap="viridis")
    field.set_facecolor("#e6edf2")
    gas = np.zeros((*gv.shape, 4)); gas[..., :3] = [.25, .29, .34]
    gas[..., 3] = np.divide(gv, pv, out=np.zeros_like(gv), where=pv > 0)
    field.imshow(gas); field.set_title("Estimated dissolved CO2 · g/L"); field.axis("off")
    fig.colorbar(image, ax=field, fraction=.035, label="g/L · fixed linear scale")
    fig.suptitle(
        f"{title} · {float(row['elapsed_h']):.3f} h\n"
        f"Injected: {row['injected_mass_g']:.3f} g · Estimated: {row['total_mass_g']:.3f} g"
    )
    fig.savefig(path, dpi=120)

def render_mass_curve(path, rows, title):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 4.5), layout="constrained"); FigureCanvasAgg(fig)
    ax = fig.subplots(); times = [row["elapsed_h"] for row in rows]
    for key, label, color in (
        ("injected_mass_g", "Injected reference", "black"),
        ("total_mass_g", "Estimated total", "#175aad"),
        ("aqueous_mass_g", "Dissolved", "#169879"),
        ("gas_mass_g", "Free gas", "#dc398b"),
    ):
        ax.plot(times, [row[key] for row in rows], label=label, color=color)
    ax.fill_between(times, [row["gas_mass_min_g"] for row in rows],
                    [row["gas_mass_max_g"] for row in rows], color="#dc398b", alpha=.14,
                    label="Free-gas partition interval")
    held = [row for row in rows if row["calibration_role"] == "held-out time"]
    ax.scatter([row["elapsed_h"] for row in held], [row["total_mass_g"] for row in held],
               marker="o", facecolors="white", edgecolors="#175aad", zorder=5,
               label="Held-out time")
    ax.set(xlabel="Time since injection start · h", ylabel="CO2 mass · g",
           title=title + " · concentration and gas-partition fit")
    ax.grid(alpha=.2); ax.legend(); fig.savefig(path, dpi=120)

def encode(result_path, target):
    # Video support is optional for numerical results and CSV/PNG exports.
    import imageio_ffmpeg

    result_path, target = Path(result_path), Path(target)
    count = len(read(result_path)["frames"])
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-framerate", "2/3", "-i", str(result_path.parent / "frame_%03d.png"),
        "-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-threads", "2", "-movflags", "+faststart", str(target),
    ], check=True)
    video = cv2.VideoCapture(str(target))
    try:
        frames, fps = int(video.get(cv2.CAP_PROP_FRAME_COUNT)), video.get(cv2.CAP_PROP_FPS)
        if frames != count*36 or fps != 24:
            raise ValueError(f"Unexpected movie length: {frames}/{fps}")
        for index in (0, frames-1):
            video.set(cv2.CAP_PROP_POS_FRAMES, index); ok, image = video.read()
            if not ok or image.shape[:2] != (528, 1560):
                raise ValueError("Movie did not decode")
    finally:
        video.release()
    return {"path": str(target), "sha256": sha(target), "frames": frames, "fps": fps,
            "duration_s": count*1.5, "playback": "one selected image per 1.5 seconds"}


def read_result(path):
    """Read saved mass metadata and resolve its existing product paths."""
    result=read(path);folder=Path(path).resolve().parent
    for row in result.get('frames',[]):
        for key in ('preview','fields'):
            if row.get(key) and not Path(row[key]).is_absolute():row[key]=str(folder/row[key])
    for key in ('mass_curve','overflow_curve','series_calibration','result_source_manifest'):
        if result.get(key) and not Path(result[key]).is_absolute():result[key]=str(folder/result[key])
    return result
