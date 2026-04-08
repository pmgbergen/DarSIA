"""GUI interface for preset workflows.

This GUI is additive and does not replace the existing command-line
``user_interface_*`` modules.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from typing import Callable

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from darsia.presets.workflows.calibration.calibration_color_paths import (
    calibration_color_paths,
    delete_calibration,
)
from darsia.presets.workflows.calibration.calibration_color_to_mass_analysis import (
    calibration_color_to_mass_analysis,
)
from darsia.presets.workflows.comparison.comparison_events import comparison_events
from darsia.presets.workflows.comparison.comparison_wasserstein import (
    comparison_wasserstein,
)
from darsia.presets.workflows.rig import Rig
from darsia.presets.workflows.setup.setup_depth import setup_depth_map
from darsia.presets.workflows.setup.setup_facies import setup_facies
from darsia.presets.workflows.setup.setup_labeling import segment_colored_image
from darsia.presets.workflows.setup.setup_rig import delete_rig, setup_rig
from darsia.presets.workflows.user_interface_analysis import run_analysis
from darsia.presets.workflows.user_interface_comparison import run_comparison
from darsia.presets.workflows.utils.utils_download import download_data

logger = logging.getLogger(__name__)


def resolve_rig_class(spec: str) -> type[Rig]:
    """Resolve a rig class from ``module:ClassName`` notation."""
    if spec.strip() == "":
        return Rig
    if ":" not in spec:
        raise ValueError("Rig class must be formatted as 'module.path:ClassName'.")

    module_name, class_name = spec.split(":", maxsplit=1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name, None)
    if cls is None:
        raise ValueError(f"Class '{class_name}' not found in module '{module_name}'.")
    if not isinstance(cls, type) or not issubclass(cls, Rig):
        raise ValueError(f"'{spec}' is not a subclass of Rig.")
    return cls


def normalize_paths(paths: list[str]) -> list[Path]:
    """Normalize path strings to unique absolute paths preserving order."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        stripped = raw.strip()
        if not stripped:
            continue
        path = Path(stripped).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


class QueueLogHandler(logging.Handler):
    """Log handler writing to a queue for GUI consumption."""

    def __init__(self, queue: Queue[str]):
        super().__init__()
        self._queue = queue

    def emit(self, record: logging.LogRecord) -> None:
        self._queue.put(self.format(record))


@dataclass
class RunContext:
    config_paths: list[Path]
    rig_cls: type[Rig]


class WorkflowGUI:
    """Tkinter-based GUI for preset workflow execution."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DarSIA Workflows GUI")
        self.root.geometry("1200x800")

        self.current_config_file: Path | None = None
        self.log_queue: Queue[str] = Queue()
        self._worker: threading.Thread | None = None

        self._setup_logging()
        self._build_layout()
        self._poll_logs()

    def _setup_logging(self) -> None:
        self.queue_handler = QueueLogHandler(self.log_queue)
        self.queue_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root_logger = logging.getLogger()
        root_logger.addHandler(self.queue_handler)
        if root_logger.level > logging.INFO:
            root_logger.setLevel(logging.INFO)

    def _build_layout(self) -> None:
        main = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        right = ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)

        self._build_config_manager(left)
        self._build_workflow_controls(left)
        self._build_editor(right)
        self._build_log_view(right)

    def _build_config_manager(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Config paths (merged in listed order)")
        frame.pack(fill=tk.X, padx=8, pady=8)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.X, padx=5, pady=5)

        self.config_list = tk.Listbox(list_frame, height=6)
        self.config_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        buttons = ttk.Frame(list_frame)
        buttons.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        ttk.Button(buttons, text="Add", command=self._add_config_path).pack(fill=tk.X)
        ttk.Button(buttons, text="Remove", command=self._remove_config_path).pack(
            fill=tk.X, pady=4
        )
        ttk.Button(buttons, text="Up", command=lambda: self._move_config(-1)).pack(
            fill=tk.X
        )
        ttk.Button(buttons, text="Down", command=lambda: self._move_config(1)).pack(
            fill=tk.X, pady=4
        )

        rig_frame = ttk.Frame(frame)
        rig_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(rig_frame, text="Custom Rig class (optional):").pack(anchor=tk.W)
        self.rig_spec = tk.StringVar(value="")
        ttk.Entry(rig_frame, textvariable=self.rig_spec).pack(fill=tk.X)
        ttk.Label(
            rig_frame,
            text="Format: package.module:ClassName (must inherit Rig)",
        ).pack(anchor=tk.W)

    def _build_workflow_controls(self, parent: ttk.Frame) -> None:
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.setup_frame = ttk.Frame(notebook)
        self.calibration_frame = ttk.Frame(notebook)
        self.analysis_frame = ttk.Frame(notebook)
        self.comparison_frame = ttk.Frame(notebook)
        self.utils_frame = ttk.Frame(notebook)
        notebook.add(self.setup_frame, text="Setup")
        notebook.add(self.calibration_frame, text="Calibration")
        notebook.add(self.analysis_frame, text="Analysis")
        notebook.add(self.comparison_frame, text="Comparison")
        notebook.add(self.utils_frame, text="Utils")

        self._build_setup_tab()
        self._build_calibration_tab()
        self._build_analysis_tab()
        self._build_comparison_tab()
        self._build_utils_tab()

    def _build_setup_tab(self) -> None:
        self.setup_all = tk.BooleanVar(value=False)
        self.setup_depth = tk.BooleanVar(value=False)
        self.setup_seg = tk.BooleanVar(value=False)
        self.setup_facies_var = tk.BooleanVar(value=False)
        self.setup_rig_var = tk.BooleanVar(value=False)
        self.setup_delete = tk.BooleanVar(value=False)
        self.setup_show = tk.BooleanVar(value=False)
        for label, var in [
            ("All", self.setup_all),
            ("Depth", self.setup_depth),
            ("Segmentation", self.setup_seg),
            ("Facies", self.setup_facies_var),
            ("Rig", self.setup_rig_var),
            ("Delete rig", self.setup_delete),
            ("Show plots", self.setup_show),
        ]:
            ttk.Checkbutton(self.setup_frame, text=label, variable=var).pack(anchor=tk.W)
        ttk.Button(
            self.setup_frame, text="Run setup", command=self._run_setup_clicked
        ).pack(fill=tk.X, pady=6)

    def _build_calibration_tab(self) -> None:
        self.cal_color_paths = tk.BooleanVar(value=False)
        self.cal_mass = tk.BooleanVar(value=False)
        self.cal_default_mass = tk.BooleanVar(value=False)
        self.cal_reset = tk.BooleanVar(value=False)
        self.cal_delete = tk.BooleanVar(value=False)
        self.cal_show = tk.BooleanVar(value=False)
        for label, var in [
            ("Color paths", self.cal_color_paths),
            ("Mass", self.cal_mass),
            ("Default mass", self.cal_default_mass),
            ("Reset", self.cal_reset),
            ("Delete calibration", self.cal_delete),
            ("Show plots", self.cal_show),
        ]:
            ttk.Checkbutton(self.calibration_frame, text=label, variable=var).pack(
                anchor=tk.W
            )
        ttk.Button(
            self.calibration_frame,
            text="Run calibration",
            command=self._run_calibration_clicked,
        ).pack(fill=tk.X, pady=6)

    def _build_analysis_tab(self) -> None:
        self.an_all = tk.BooleanVar(value=False)
        self.an_crop = tk.BooleanVar(value=False)
        self.an_seg = tk.BooleanVar(value=False)
        self.an_fingers = tk.BooleanVar(value=False)
        self.an_mass = tk.BooleanVar(value=False)
        self.an_volume = tk.BooleanVar(value=False)
        self.an_show = tk.BooleanVar(value=False)
        self.an_jpg = tk.BooleanVar(value=False)
        self.an_npz = tk.BooleanVar(value=False)
        for label, var in [
            ("All images", self.an_all),
            ("Cropping", self.an_crop),
            ("Segmentation", self.an_seg),
            ("Fingers", self.an_fingers),
            ("Mass", self.an_mass),
            ("Volume", self.an_volume),
            ("Show plots", self.an_show),
            ("Save JPG", self.an_jpg),
            ("Save NPZ", self.an_npz),
        ]:
            ttk.Checkbutton(self.analysis_frame, text=label, variable=var).pack(
                anchor=tk.W
            )
        ttk.Button(
            self.analysis_frame, text="Run analysis", command=self._run_analysis_clicked
        ).pack(fill=tk.X, pady=6)

    def _build_comparison_tab(self) -> None:
        self.comp_events = tk.BooleanVar(value=False)
        self.comp_w_compute = tk.BooleanVar(value=False)
        self.comp_w_assemble = tk.BooleanVar(value=False)
        for label, var in [
            ("Events", self.comp_events),
            ("Wasserstein compute", self.comp_w_compute),
            ("Wasserstein assemble", self.comp_w_assemble),
        ]:
            ttk.Checkbutton(self.comparison_frame, text=label, variable=var).pack(
                anchor=tk.W
            )
        ttk.Button(
            self.comparison_frame,
            text="Run comparison",
            command=self._run_comparison_clicked,
        ).pack(fill=tk.X, pady=6)

    def _build_utils_tab(self) -> None:
        self.utils_download = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.utils_frame, text="Download/cache data", variable=self.utils_download
        ).pack(anchor=tk.W)
        ttk.Button(
            self.utils_frame, text="Run utils", command=self._run_utils_clicked
        ).pack(fill=tk.X, pady=6)

    def _build_editor(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Config editor")
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(toolbar, text="New from template", command=self._new_from_template).pack(
            side=tk.LEFT
        )
        ttk.Button(toolbar, text="Open", command=self._open_config).pack(
            side=tk.LEFT, padx=4
        )
        ttk.Button(toolbar, text="Save", command=self._save_config).pack(side=tk.LEFT)
        ttk.Button(toolbar, text="Save as", command=self._save_config_as).pack(
            side=tk.LEFT, padx=4
        )

        self.editor_path = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.editor_path).pack(anchor=tk.W, padx=5)

        self.editor = tk.Text(frame, wrap=tk.NONE)
        self.editor.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_log_view(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Execution log")
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.log = tk.Text(frame, height=15, state=tk.DISABLED)
        self.log.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _append_log(self, msg: str) -> None:
        self.log.config(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.config(state=tk.DISABLED)

    def _poll_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except Exception:
            pass
        self.root.after(100, self._poll_logs)

    def _selected_paths(self) -> list[Path]:
        values = [self.config_list.get(i) for i in range(self.config_list.size())]
        return normalize_paths(values)

    def _context(self) -> RunContext:
        paths = self._selected_paths()
        if len(paths) == 0:
            raise ValueError("Please add at least one config file.")
        rig_cls = resolve_rig_class(self.rig_spec.get())
        return RunContext(config_paths=paths, rig_cls=rig_cls)

    def _run_async(self, fn: Callable[[], None]) -> None:
        if self._worker is not None and self._worker.is_alive():
            messagebox.showwarning("Busy", "Another workflow is still running.")
            return

        def _target():
            try:
                fn()
                self.log_queue.put("Done.")
            except Exception as e:
                logger.exception("Workflow failed")
                self.log_queue.put(f"ERROR: {e}")

        self._worker = threading.Thread(target=_target, daemon=True)
        self._worker.start()

    def _run_setup_clicked(self) -> None:
        def _run():
            ctx = self._context()
            show = self.setup_show.get()
            if self.setup_all.get() or self.setup_depth.get():
                setup_depth_map(ctx.config_paths, key="depth", show=show)
            if self.setup_all.get() or self.setup_seg.get():
                segment_colored_image(ctx.config_paths, show=show)
            if self.setup_all.get() or self.setup_facies_var.get():
                setup_facies(ctx.rig_cls, ctx.config_paths, show=show)
            if self.setup_all.get() or self.setup_rig_var.get():
                setup_rig(ctx.rig_cls, ctx.config_paths, show=show)
            if self.setup_delete.get():
                delete_rig(ctx.rig_cls, ctx.config_paths, show=show)

        self._run_async(_run)

    def _run_calibration_clicked(self) -> None:
        def _run():
            ctx = self._context()
            if self.cal_delete.get():
                delete_calibration(ctx.config_paths)
                return
            if self.cal_color_paths.get():
                calibration_color_paths(ctx.rig_cls, ctx.config_paths, self.cal_show.get())
            if self.cal_mass.get() or self.cal_default_mass.get():
                calibration_color_to_mass_analysis(
                    ctx.rig_cls,
                    ctx.config_paths,
                    reset=self.cal_reset.get(),
                    show=self.cal_show.get(),
                    default=self.cal_default_mass.get(),
                )

        self._run_async(_run)

    def _run_analysis_clicked(self) -> None:
        def _run():
            ctx = self._context()
            args = argparse.Namespace(
                config=ctx.config_paths,
                all=self.an_all.get(),
                cropping=self.an_crop.get(),
                segmentation=self.an_seg.get(),
                fingers=self.an_fingers.get(),
                mass=self.an_mass.get(),
                volume=self.an_volume.get(),
                show=self.an_show.get(),
                save_jpg=self.an_jpg.get(),
                save_npz=self.an_npz.get(),
                info=False,
            )
            run_analysis(ctx.rig_cls, args)

        self._run_async(_run)

    def _run_comparison_clicked(self) -> None:
        def _run():
            ctx = self._context()
            if len(ctx.config_paths) != 1:
                raise ValueError("Comparison currently supports exactly one config path.")
            args = argparse.Namespace(
                config=ctx.config_paths[0],
                events=self.comp_events.get(),
                wasserstein_compute=self.comp_w_compute.get(),
                wasserstein_assemble=self.comp_w_assemble.get(),
                info=False,
                show=False,
            )
            run_comparison(ctx.rig_cls, args)

        self._run_async(_run)

    def _run_utils_clicked(self) -> None:
        def _run():
            ctx = self._context()
            if self.utils_download.get():
                download_data(ctx.config_paths)
            else:
                logger.info("No utility option selected.")

        self._run_async(_run)

    def _add_config_path(self) -> None:
        path = filedialog.askopenfilename(
            title="Select config file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if path:
            self.config_list.insert(tk.END, str(Path(path).resolve()))

    def _remove_config_path(self) -> None:
        selection = list(self.config_list.curselection())
        for idx in reversed(selection):
            self.config_list.delete(idx)

    def _move_config(self, direction: int) -> None:
        selected = self.config_list.curselection()
        if len(selected) != 1:
            return
        i = selected[0]
        j = i + direction
        if j < 0 or j >= self.config_list.size():
            return
        value = self.config_list.get(i)
        self.config_list.delete(i)
        self.config_list.insert(j, value)
        self.config_list.selection_set(j)

    def _new_from_template(self) -> None:
        template = Path(__file__).resolve().parent / "templates" / "config.toml"
        if not template.exists():
            messagebox.showerror("Missing template", f"Template not found: {template}")
            return
        self.current_config_file = None
        self.editor_path.set("New file (from template)")
        self.editor.delete("1.0", tk.END)
        self.editor.insert(tk.END, template.read_text())

    def _open_config(self) -> None:
        path = filedialog.askopenfilename(
            title="Open config file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path).resolve()
        self.current_config_file = p
        self.editor_path.set(str(p))
        self.editor.delete("1.0", tk.END)
        self.editor.insert(tk.END, p.read_text())
        if str(p) not in self.config_list.get(0, tk.END):
            self.config_list.insert(tk.END, str(p))

    def _save_config(self) -> None:
        if self.current_config_file is None:
            self._save_config_as()
            return
        self.current_config_file.write_text(self.editor.get("1.0", tk.END))
        self.editor_path.set(str(self.current_config_file))
        if str(self.current_config_file) not in self.config_list.get(0, tk.END):
            self.config_list.insert(tk.END, str(self.current_config_file))

    def _save_config_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save config file",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not path:
            return
        self.current_config_file = Path(path).resolve()
        self._save_config()


def launch_workflows_gui() -> None:
    root = tk.Tk()
    WorkflowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_workflows_gui()
