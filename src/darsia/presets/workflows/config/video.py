"""Configuration for video/GIF utility within workflow utils."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .utils import _get_key, _get_section_from_toml


def _normalize_extensions(extensions: list[str] | str) -> list[str]:
    if isinstance(extensions, str):
        extensions = [extensions]
    normalized: list[str] = []
    for ext in extensions:
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        normalized.append(ext)
    if not normalized:
        raise ValueError("Video source extensions must not be empty.")
    return normalized


def _to_rgb(color: list[int] | tuple[int, int, int], name: str) -> tuple[int, int, int]:
    if len(color) != 3:
        raise ValueError(f"{name} must have exactly 3 entries [R, G, B].")
    vals = tuple(int(v) for v in color)
    if any(v < 0 or v > 255 for v in vals):
        raise ValueError(f"{name} entries must be in [0, 255].")
    return vals


@dataclass
class VideoSourceConfig:
    folder: Path | None = field(
        default=None,
        metadata={
            "name": "Video source folder",
            "help": "Directory containing source images/frames for video/GIF creation.",
            "widget": "folder",
            "placeholder": "Select a folder containing images",
        },
    )
    pattern: str | None = field(
        default=None,
        metadata={
            "name": "File name pattern",
            "help": "Glob pattern to match image files (e.g., '*.jpg' or 'frame_*.png').",
        },
    )
    extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"],
        metadata={
            "name": "File extensions",
            "help": (
                "List of file extensions to include (e.g., ['.jpg', '.png']). "
                "Will be normalized automatically."
            ),
        },
    )
    recursive: bool = field(
        default=False,
        metadata={
            "name": "Search recursively",
            "help": "Whether to search subdirectories for image files.",
        },
    )
    sorting: str = field(
        default="protocols",
        metadata={
            "name": "Sorting method",
            "help": "Method to sort images: 'protocols' (default) or 'name'.",
        },
    )

    ALLOWED_SORTING_METHODS = {"protocols", "name"}

    def load(self, sec: dict) -> "VideoSourceConfig":
        src = _get_key(sec, "source", required=True)
        folder = _get_key(src, "folder", required=True, type_=str).strip()
        if not folder:
            raise ValueError("[video.source].folder must not be empty.")
        self.folder = Path(folder)
        self.pattern = _get_key(src, "pattern", required=False, default=None, type_=str)
        self.extensions = _normalize_extensions(
            _get_key(src, "extensions", required=False, default=self.extensions)
        )
        self.recursive = bool(_get_key(src, "recursive", required=False, default=False))
        self.sorting = _get_key(src, "sorting", required=False, default=self.sorting)
        return self


@dataclass
class VideoOutputConfig:
    formats: list[str] = field(
        default_factory=lambda: ["mp4"],
        metadata={
            "name": "Output formats",
            "help": "List of output formats: 'mp4' and/or 'gif'.",
        },
    )
    fps: float = field(
        default=6.0,
        metadata={
            "name": "Frames per second (FPS)",
            "help": "Playback speed for the video/GIF (must be positive).",
        },
    )
    resolution: tuple[int, int] | None = field(
        default=None,
        metadata={
            "name": "Output resolution",
            "help": (
                "Video resolution as [width, height] pixels. "
                "Leave empty to auto-detect."
            ),
        },
    )
    filename: str | None = field(
        default=None,
        metadata={
            "name": "Output filename",
            "help": (
                "Base name for output files (without extension). "
                "Auto-generated if not set."
            ),
        },
    )
    codec: str = field(
        default="mp4v",
        metadata={
            "name": "Video codec",
            "help": "Codec for MP4 output (typically 'mp4v' or 'avc1').",
        },
    )
    quality: int = field(
        default=95,
        metadata={
            "name": "Quality (1-100)",
            "help": "Video/GIF quality on a scale of 1-100 (higher = better).",
        },
    )

    def load(self, sec: dict) -> "VideoOutputConfig":
        out = sec.get("output", {})
        formats = _get_key(out, "formats", required=False, default=self.formats)
        if isinstance(formats, str):
            formats = [formats]
        self.formats = [f.lower().strip() for f in formats if f.strip()]
        invalid = [f for f in self.formats if f not in {"mp4", "gif"}]
        if invalid:
            raise ValueError(f"Unsupported video output format(s): {invalid}")
        self.fps = float(_get_key(out, "fps", required=False, default=self.fps))
        if self.fps <= 0:
            raise ValueError("Video output fps must be positive.")

        resolution = _get_key(out, "resolution", required=False, default=None)
        if resolution is not None:
            if len(resolution) != 2:
                raise ValueError("Video output resolution must be [width, height].")
            self.resolution = (int(resolution[0]), int(resolution[1]))
            if self.resolution[0] <= 0 or self.resolution[1] <= 0:
                raise ValueError("Video output resolution values must be positive.")
        else:
            self.resolution = None
        self.filename = _get_key(out, "filename", required=False, default=None)
        self.codec = _get_key(out, "codec", required=False, default=self.codec)
        self.quality = int(
            _get_key(out, "quality", required=False, default=self.quality)
        )
        if not (1 <= self.quality <= 100):
            raise ValueError("Video output quality must be in [1, 100].")
        return self


@dataclass
class VideoOverlayConfig:
    show_elapsed_time: bool = field(
        default=True,
        metadata={
            "name": "Show elapsed time",
            "help": "Display elapsed time overlay on the video.",
        },
    )
    elapsed_time_format: str = field(
        default="Elapsed: {:.2f} h",
        metadata={
            "name": "Elapsed time format",
            "help": (
                "Format string for elapsed time (Python format, e.g., "
                "'Elapsed: {:.2f} h')."
            ),
        },
    )
    show_note: bool = field(
        default=True,
        metadata={
            "name": "Show note",
            "help": "Display custom note overlay on the video.",
        },
    )
    note: str = field(
        default="",
        metadata={
            "name": "Custom note text",
            "help": "Text to display as overlay (if show_note is enabled).",
        },
    )
    font_scale: float = field(
        default=0.7,
        metadata={
            "name": "Font scale",
            "help": "Font size scale for overlay text.",
        },
    )
    text_color: tuple[int, int, int] = field(
        default=(255, 255, 255),
        metadata={
            "name": "Text color [R, G, B]",
            "help": "RGB color for overlay text (values 0-255).",
        },
    )
    thickness: int = field(
        default=2,
        metadata={
            "name": "Text thickness",
            "help": "Thickness of text strokes.",
        },
    )
    line_spacing: int = field(
        default=8,
        metadata={
            "name": "Line spacing",
            "help": "Spacing between lines in multi-line overlays.",
        },
    )
    position: tuple[int, int] = field(
        default=(20, 20),
        metadata={
            "name": "Overlay position [x, y]",
            "help": "Position of overlay in pixels from top-left corner.",
        },
    )
    box_enabled: bool = field(
        default=True,
        metadata={
            "name": "Enable background box",
            "help": "Draw a semi-transparent background box behind overlay text.",
        },
    )
    box_color: tuple[int, int, int] = field(
        default=(0, 0, 0),
        metadata={
            "name": "Box color [R, G, B]",
            "help": "RGB color for background box (values 0-255).",
        },
    )
    box_alpha: float = field(
        default=0.4,
        metadata={
            "name": "Box alpha (0-1)",
            "help": "Transparency of background box (0=transparent, 1=opaque).",
        },
    )
    box_padding: int = field(
        default=10,
        metadata={
            "name": "Box padding",
            "help": "Padding in pixels around overlay text inside the box.",
        },
    )

    def load(self, sec: dict) -> "VideoOverlayConfig":
        overlay = sec.get("overlay", {})
        self.show_elapsed_time = bool(
            _get_key(
                overlay,
                "show_elapsed_time",
                required=False,
                default=self.show_elapsed_time,
            )
        )
        self.elapsed_time_format = _get_key(
            overlay,
            "elapsed_time_format",
            required=False,
            default=self.elapsed_time_format,
        )
        self.show_note = bool(
            _get_key(overlay, "show_note", required=False, default=self.show_note)
        )
        self.note = _get_key(overlay, "note", required=False, default=self.note)
        self.font_scale = float(
            _get_key(overlay, "font_scale", required=False, default=self.font_scale)
        )
        self.thickness = int(
            _get_key(overlay, "thickness", required=False, default=self.thickness)
        )
        self.line_spacing = int(
            _get_key(overlay, "line_spacing", required=False, default=self.line_spacing)
        )
        position = _get_key(overlay, "position", required=False, default=self.position)
        if len(position) != 2:
            raise ValueError("Video overlay position must be [x, y].")
        self.position = (int(position[0]), int(position[1]))
        self.text_color = _to_rgb(
            _get_key(overlay, "text_color", required=False, default=self.text_color),
            "text_color",
        )

        self.box_enabled = bool(
            _get_key(overlay, "box_enabled", required=False, default=self.box_enabled)
        )
        self.box_color = _to_rgb(
            _get_key(overlay, "box_color", required=False, default=self.box_color),
            "box_color",
        )
        self.box_alpha = float(
            _get_key(overlay, "box_alpha", required=False, default=self.box_alpha)
        )
        if not (0 <= self.box_alpha <= 1):
            raise ValueError("Video overlay box_alpha must be in [0, 1].")
        self.box_padding = int(
            _get_key(overlay, "box_padding", required=False, default=self.box_padding)
        )
        return self


@dataclass
class VideoConfig:
    source: VideoSourceConfig = field(
        default_factory=VideoSourceConfig,
        metadata={
            "name": "Video source",
            "help": "Settings for video/GIF source images.",
        },
    )
    output: VideoOutputConfig = field(
        default_factory=VideoOutputConfig,
        metadata={
            "name": "Video output",
            "help": "Settings for video/GIF output (format, quality, resolution).",
        },
    )
    overlay: VideoOverlayConfig = field(
        default_factory=VideoOverlayConfig,
        metadata={
            "name": "Video overlay",
            "help": "Settings for overlay text/graphics on video.",
        },
    )
    folder: Path = field(
        default_factory=Path,
        metadata={
            "name": "Video output folder",
            "help": (
                "Destination folder for generated videos (auto-set to "
                "[data.results]/videos)."
            ),
        },
    )

    def load(self, path: Path | list[Path], results: Path | None) -> "VideoConfig":
        sec = _get_section_from_toml(path, "video")
        self.source.load(sec)
        self.output.load(sec)
        self.overlay.load(sec)
        if results is None:
            raise ValueError(
                "Video utility requires a results directory from [data.results]."
            )
        self.folder = results / "videos"
        return self

    def error(self):
        raise ValueError("Use [video] in the config file to load video utility config.")
