"""GUI interface for preset workflows.

This GUI is additive and does not replace the existing command-line
``user_interface_*`` modules.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import multiprocessing as mp
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable

from darsia.presets.workflows.rig import Rig

logger = logging.getLogger(__name__)


def _require_tkinter() -> tuple[Any, Any, Any, Any]:
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "tkinter is required for the DarSIA GUI. Install Python Tk support "
            "(e.g. apt install python3-tk on Debian/Ubuntu) and retry."
        ) from e
    return tk, filedialog, messagebox, ttk


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


def _find_template_file() -> Path:
    candidates: list[Path] = []
    try:
        packaged = resources.files("darsia.presets.workflows.templates").joinpath(
            "config.toml"
        )
        candidates.append(Path(str(packaged)))
    except (ModuleNotFoundError, AttributeError, FileNotFoundError):
        pass
    candidates.append(Path(__file__).resolve().parent / "templates" / "config.toml")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


def abort_process(process: mp.Process | None) -> bool:
    """Abort a running process."""
    if process is None or not process.is_alive():
        return False
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)
    return True


def _run_setup_workflow(
    config_paths: list[str], rig_spec: str, options: dict[str, bool]
) -> None:
    from darsia.presets.workflows.setup.setup_depth import setup_depth_map
    from darsia.presets.workflows.setup.setup_facies import setup_facies
    from darsia.presets.workflows.setup.setup_labeling import segment_colored_image
    from darsia.presets.workflows.setup.setup_rig import delete_rig, setup_rig

    paths = normalize_paths(config_paths)
    rig_cls = resolve_rig_class(rig_spec)
    show = options["show"]
    if options["all"] or options["depth"]:
        setup_depth_map(paths, key="depth", show=show)
    if options["all"] or options["segmentation"]:
        segment_colored_image(paths, show=show)
    if options["all"] or options["facies"]:
        setup_facies(rig_cls, paths, show=show)
    if options["all"] or options["rig"]:
        setup_rig(rig_cls, paths, show=show)
    if options["delete_rig"]:
        delete_rig(rig_cls, paths, show=show)


def _run_calibration_workflow(
    config_paths: list[str], rig_spec: str, options: dict[str, bool]
) -> None:
    from darsia.presets.workflows.calibration import (
        calibration_color_to_mass_analysis as c2m_analysis_module,
    )
    from darsia.presets.workflows.calibration.calibration_color_paths import (
        calibration_color_paths,
        delete_calibration,
    )

    paths = normalize_paths(config_paths)
    rig_cls = resolve_rig_class(rig_spec)
    if options["delete"]:
        delete_calibration(paths)
        return
    if options["color_paths"]:
        calibration_color_paths(rig_cls, paths, options["show"])
    if options["mass"] or options["default_mass"]:
        c2m_analysis_module.calibration_color_to_mass_analysis(
            rig_cls,
            paths,
            reset=options["reset"],
            show=options["show"],
            default=options["default_mass"],
        )


def _run_analysis_workflow(
    config_paths: list[str], rig_spec: str, options: dict[str, bool]
) -> None:
    from darsia.presets.workflows.user_interface_analysis import run_analysis

    paths = normalize_paths(config_paths)
    rig_cls = resolve_rig_class(rig_spec)
    args = argparse.Namespace(
        config=paths,
        all=options["all"],
        cropping=options["cropping"],
        segmentation=options["segmentation"],
        fingers=options["fingers"],
        mass=options["mass"],
        volume=options["volume"],
        show=options["show"],
        save_jpg=options["save_jpg"],
        save_npz=options["save_npz"],
        info=False,
    )
    run_analysis(rig_cls, args)


def _run_comparison_workflow(
    config_path: str, rig_spec: str, options: dict[str, bool]
) -> None:
    from darsia.presets.workflows.user_interface_comparison import run_comparison

    path = Path(config_path)
    rig_cls = resolve_rig_class(rig_spec)
    args = argparse.Namespace(
        config=path,
        events=options["events"],
        wasserstein_compute=options["wasserstein_compute"],
        wasserstein_assemble=options["wasserstein_assemble"],
        info=False,
        show=False,
    )
    run_comparison(rig_cls, args)


def _run_utils_workflow(config_paths: list[str], options: dict[str, bool]) -> None:
    from darsia.presets.workflows.utils.utils_download import download_data

    paths = normalize_paths(config_paths)
    if options["download"]:
        download_data(paths)


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

    def __init__(self, root: Any):
        self.tk, self.filedialog, self.messagebox, self.ttk = _require_tkinter()
        self.root = root
        self.root.title("DarSIA Workflows GUI")
        self.root.geometry("1200x800")

        self.current_config_file: Path | None = None
        self.log_queue: Queue[str] = Queue()
        self._worker_process: mp.Process | None = None
        self._abort_requested = False

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
        main = self.ttk.Panedwindow(self.root, orient=self.tk.HORIZONTAL)
        main.pack(fill=self.tk.BOTH, expand=True)

        left = self.ttk.Frame(main)
        right = self.ttk.Frame(main)
        main.add(left, weight=1)
        main.add(right, weight=2)

        self._build_config_manager(left)
        self._build_workflow_controls(left)
        self._build_editor(right)
        self._build_log_view(right)

    def _build_config_manager(self, parent) -> None:
        frame = self.ttk.LabelFrame(
            parent, text="Config paths (merged in listed order)"
        )
        frame.pack(fill=self.tk.X, padx=8, pady=8)

        list_frame = self.ttk.Frame(frame)
        list_frame.pack(fill=self.tk.X, padx=5, pady=5)

        self.config_list = self.tk.Listbox(list_frame, height=6)
        self.config_list.pack(side=self.tk.LEFT, fill=self.tk.BOTH, expand=True)

        buttons = self.ttk.Frame(list_frame)
        buttons.pack(side=self.tk.RIGHT, fill=self.tk.Y, padx=5)
        self.ttk.Button(buttons, text="Add", command=self._add_config_path).pack(
            fill=self.tk.X
        )
        self.ttk.Button(buttons, text="Remove", command=self._remove_config_path).pack(
            fill=self.tk.X, pady=4
        )
        self.ttk.Button(buttons, text="Up", command=lambda: self._move_config(-1)).pack(
            fill=self.tk.X
        )
        self.ttk.Button(
            buttons, text="Down", command=lambda: self._move_config(1)
        ).pack(fill=self.tk.X, pady=4)

        rig_frame = self.ttk.Frame(frame)
        rig_frame.pack(fill=self.tk.X, padx=5, pady=5)
        self.ttk.Label(rig_frame, text="Custom Rig class (optional):").pack(
            anchor=self.tk.W
        )
        self.rig_spec = self.tk.StringVar(value="")
        self.ttk.Entry(rig_frame, textvariable=self.rig_spec).pack(fill=self.tk.X)
        self.ttk.Label(
            rig_frame,
            text="Format: package.module:ClassName (must inherit Rig)",
        ).pack(anchor=self.tk.W)

    def _build_workflow_controls(self, parent) -> None:
        notebook = self.ttk.Notebook(parent)
        notebook.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)

        self.setup_frame = self.ttk.Frame(notebook)
        self.calibration_frame = self.ttk.Frame(notebook)
        self.analysis_frame = self.ttk.Frame(notebook)
        self.comparison_frame = self.ttk.Frame(notebook)
        self.utils_frame = self.ttk.Frame(notebook)
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
        self.setup_all = self.tk.BooleanVar(value=False)
        self.setup_depth = self.tk.BooleanVar(value=False)
        self.setup_seg = self.tk.BooleanVar(value=False)
        self.setup_facies = self.tk.BooleanVar(value=False)
        self.setup_rig = self.tk.BooleanVar(value=False)
        self.setup_delete = self.tk.BooleanVar(value=False)
        self.setup_show = self.tk.BooleanVar(value=False)
        for label, var in [
            ("All", self.setup_all),
            ("Depth", self.setup_depth),
            ("Segmentation", self.setup_seg),
            ("Facies", self.setup_facies),
            ("Rig", self.setup_rig),
            ("Delete rig", self.setup_delete),
            ("Show plots", self.setup_show),
        ]:
            self.ttk.Checkbutton(self.setup_frame, text=label, variable=var).pack(
                anchor=self.tk.W
            )
        self.ttk.Button(
            self.setup_frame, text="Run setup", command=self._run_setup_clicked
        ).pack(fill=self.tk.X, pady=6)

    def _build_calibration_tab(self) -> None:
        self.cal_color_paths = self.tk.BooleanVar(value=False)
        self.cal_mass = self.tk.BooleanVar(value=False)
        self.cal_default_mass = self.tk.BooleanVar(value=False)
        self.cal_reset = self.tk.BooleanVar(value=False)
        self.cal_delete = self.tk.BooleanVar(value=False)
        self.cal_show = self.tk.BooleanVar(value=False)
        for label, var in [
            ("Color paths", self.cal_color_paths),
            ("Mass", self.cal_mass),
            ("Default mass", self.cal_default_mass),
            ("Reset", self.cal_reset),
            ("Delete calibration", self.cal_delete),
            ("Show plots", self.cal_show),
        ]:
            self.ttk.Checkbutton(self.calibration_frame, text=label, variable=var).pack(
                anchor=self.tk.W
            )
        self.ttk.Button(
            self.calibration_frame,
            text="Run calibration",
            command=self._run_calibration_clicked,
        ).pack(fill=self.tk.X, pady=6)

    def _build_analysis_tab(self) -> None:
        self.an_all = self.tk.BooleanVar(value=False)
        self.an_crop = self.tk.BooleanVar(value=False)
        self.an_seg = self.tk.BooleanVar(value=False)
        self.an_fingers = self.tk.BooleanVar(value=False)
        self.an_mass = self.tk.BooleanVar(value=False)
        self.an_volume = self.tk.BooleanVar(value=False)
        self.an_show = self.tk.BooleanVar(value=False)
        self.an_jpg = self.tk.BooleanVar(value=False)
        self.an_npz = self.tk.BooleanVar(value=False)
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
            self.ttk.Checkbutton(self.analysis_frame, text=label, variable=var).pack(
                anchor=self.tk.W
            )
        self.ttk.Button(
            self.analysis_frame, text="Run analysis", command=self._run_analysis_clicked
        ).pack(fill=self.tk.X, pady=6)

    def _build_comparison_tab(self) -> None:
        self.comp_events = self.tk.BooleanVar(value=False)
        self.comp_w_compute = self.tk.BooleanVar(value=False)
        self.comp_w_assemble = self.tk.BooleanVar(value=False)
        for label, var in [
            ("Events", self.comp_events),
            ("Wasserstein compute", self.comp_w_compute),
            ("Wasserstein assemble", self.comp_w_assemble),
        ]:
            self.ttk.Checkbutton(self.comparison_frame, text=label, variable=var).pack(
                anchor=self.tk.W
            )
        self.ttk.Button(
            self.comparison_frame,
            text="Run comparison",
            command=self._run_comparison_clicked,
        ).pack(fill=self.tk.X, pady=6)

    def _build_utils_tab(self) -> None:
        self.utils_download = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(
            self.utils_frame, text="Download/cache data", variable=self.utils_download
        ).pack(anchor=self.tk.W)
        self.ttk.Button(
            self.utils_frame, text="Run utils", command=self._run_utils_clicked
        ).pack(fill=self.tk.X, pady=6)

    def _build_editor(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Config editor")
        frame.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)

        toolbar = self.ttk.Frame(frame)
        toolbar.pack(fill=self.tk.X, padx=5, pady=5)
        self.ttk.Button(
            toolbar, text="New from template", command=self._new_from_template
        ).pack(side=self.tk.LEFT)
        self.ttk.Button(toolbar, text="Open", command=self._open_config).pack(
            side=self.tk.LEFT, padx=4
        )
        self.ttk.Button(toolbar, text="Save", command=self._save_config).pack(
            side=self.tk.LEFT
        )
        self.ttk.Button(toolbar, text="Save as", command=self._save_config_as).pack(
            side=self.tk.LEFT, padx=4
        )

        self.editor_path = self.tk.StringVar(value="")
        self.ttk.Label(frame, textvariable=self.editor_path).pack(
            anchor=self.tk.W, padx=5
        )

        self.editor = self.tk.Text(frame, wrap=self.tk.NONE)
        self.editor.pack(fill=self.tk.BOTH, expand=True, padx=5, pady=5)

    def _build_log_view(self, parent) -> None:
        frame = self.ttk.LabelFrame(parent, text="Execution log")
        frame.pack(fill=self.tk.BOTH, expand=True, padx=8, pady=8)

        controls = self.ttk.Frame(frame)
        controls.pack(fill=self.tk.X, padx=5, pady=(5, 0))
        self.abort_button = self.ttk.Button(
            controls,
            text="Abort running workflow",
            command=self._abort_worker_clicked,
            state=self.tk.DISABLED,
        )
        self.abort_button.pack(side=self.tk.RIGHT)

        self.log = self.tk.Text(frame, height=15, state=self.tk.DISABLED)
        self.log.pack(fill=self.tk.BOTH, expand=True, padx=5, pady=5)

    def _append_log(self, msg: str) -> None:
        self.log.config(state=self.tk.NORMAL)
        self.log.insert(self.tk.END, msg + "\n")
        self.log.see(self.tk.END)
        self.log.config(state=self.tk.DISABLED)

    def _poll_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                self._append_log(msg)
        except Empty:
            pass
        self.root.after(100, self._poll_logs)

    def _selected_paths(self) -> list[Path]:
        values = [self.config_list.get(i) for i in range(self.config_list.size())]
        return normalize_paths(values)

    def _context(self) -> RunContext:
        paths = self._selected_paths()
        if not paths:
            raise ValueError("Please add at least one config file.")
        rig_cls = resolve_rig_class(self.rig_spec.get())
        return RunContext(config_paths=paths, rig_cls=rig_cls)

    def _set_worker_state(self, running: bool) -> None:
        state = self.tk.NORMAL if running else self.tk.DISABLED
        self.abort_button.config(state=state)

    def _abort_worker_clicked(self) -> None:
        if abort_process(self._worker_process):
            self._abort_requested = True
            self.log_queue.put("Abort requested.")
            return
        self.log_queue.put("No active workflow to abort.")

    def _run_async(self, fn: Callable[..., None], *args: Any) -> None:
        if self._worker_process is not None and self._worker_process.is_alive():
            self.messagebox.showwarning("Busy", "Another workflow is still running.")
            return
        self._worker_process = mp.Process(target=fn, args=args, daemon=True)
        self._worker_process.start()
        self._abort_requested = False
        self._set_worker_state(True)
        self.root.after(200, self._poll_worker_completion)

    def _poll_worker_completion(self) -> None:
        process = self._worker_process
        if process is None:
            self._set_worker_state(False)
            return
        if process.is_alive():
            self.root.after(200, self._poll_worker_completion)
            return

        exit_code = process.exitcode
        if self._abort_requested:
            self.log_queue.put("Workflow aborted.")
        elif exit_code == 0:
            self.log_queue.put("Done.")
        else:
            self.log_queue.put(f"ERROR: Workflow failed with exit code {exit_code}.")
        self._worker_process = None
        self._set_worker_state(False)

    def _run_setup_clicked(self) -> None:
        ctx = self._context()
        options = {
            "all": self.setup_all.get(),
            "depth": self.setup_depth.get(),
            "segmentation": self.setup_seg.get(),
            "facies": self.setup_facies.get(),
            "rig": self.setup_rig.get(),
            "delete_rig": self.setup_delete.get(),
            "show": self.setup_show.get(),
        }
        self._run_async(
            _run_setup_workflow,
            [str(path) for path in ctx.config_paths],
            self.rig_spec.get(),
            options,
        )

    def _run_calibration_clicked(self) -> None:
        ctx = self._context()
        options = {
            "delete": self.cal_delete.get(),
            "color_paths": self.cal_color_paths.get(),
            "mass": self.cal_mass.get(),
            "default_mass": self.cal_default_mass.get(),
            "reset": self.cal_reset.get(),
            "show": self.cal_show.get(),
        }
        self._run_async(
            _run_calibration_workflow,
            [str(path) for path in ctx.config_paths],
            self.rig_spec.get(),
            options,
        )

    def _run_analysis_clicked(self) -> None:
        ctx = self._context()
        options = {
            "all": self.an_all.get(),
            "cropping": self.an_crop.get(),
            "segmentation": self.an_seg.get(),
            "fingers": self.an_fingers.get(),
            "mass": self.an_mass.get(),
            "volume": self.an_volume.get(),
            "show": self.an_show.get(),
            "save_jpg": self.an_jpg.get(),
            "save_npz": self.an_npz.get(),
        }
        self._run_async(
            _run_analysis_workflow,
            [str(path) for path in ctx.config_paths],
            self.rig_spec.get(),
            options,
        )

    def _run_comparison_clicked(self) -> None:
        try:
            ctx = self._context()
        except Exception as e:
            self.messagebox.showerror("Invalid configuration", str(e))
            return
        if len(ctx.config_paths) != 1:
            self.messagebox.showerror(
                "Invalid configuration",
                "Comparison currently supports exactly one config path.",
            )
            return

        options = {
            "events": self.comp_events.get(),
            "wasserstein_compute": self.comp_w_compute.get(),
            "wasserstein_assemble": self.comp_w_assemble.get(),
        }
        self._run_async(
            _run_comparison_workflow,
            str(ctx.config_paths[0]),
            self.rig_spec.get(),
            options,
        )

    def _run_utils_clicked(self) -> None:
        ctx = self._context()
        options = {"download": self.utils_download.get()}
        if not options["download"]:
            logger.info("No utility option selected.")
            return
        self._run_async(
            _run_utils_workflow, [str(path) for path in ctx.config_paths], options
        )

    def _add_config_path(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Select config file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if path:
            self.config_list.insert(self.tk.END, str(Path(path).resolve()))

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
        template = _find_template_file()
        if not template.exists():
            self.messagebox.showerror(
                "Missing template", f"Template not found: {template}"
            )
            return
        self.current_config_file = None
        self.editor_path.set("New file (from template)")
        self.editor.delete("1.0", self.tk.END)
        self.editor.insert(self.tk.END, template.read_text())

    def _open_config(self) -> None:
        path = self.filedialog.askopenfilename(
            title="Open config file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not path:
            return
        p = Path(path).resolve()
        self.current_config_file = p
        self.editor_path.set(str(p))
        self.editor.delete("1.0", self.tk.END)
        self.editor.insert(self.tk.END, p.read_text())
        if str(p) not in self.config_list.get(0, self.tk.END):
            self.config_list.insert(self.tk.END, str(p))

    def _save_config(self) -> None:
        if self.current_config_file is None:
            self._save_config_as()
            return
        self.current_config_file.write_text(self.editor.get("1.0", "end-1c"))
        self.editor_path.set(str(self.current_config_file))
        if str(self.current_config_file) not in self.config_list.get(0, self.tk.END):
            self.config_list.insert(self.tk.END, str(self.current_config_file))

    def _save_config_as(self) -> None:
        path = self.filedialog.asksaveasfilename(
            title="Save config file",
            defaultextension=".toml",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
        )
        if not path:
            return
        self.current_config_file = Path(path).resolve()
        self._save_config()


def launch_workflows_gui() -> None:
    tk, _, _, _ = _require_tkinter()
    root = tk.Tk()
    WorkflowGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_workflows_gui()
