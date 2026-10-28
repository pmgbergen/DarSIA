"""GPU device choice in the GUI: the Production page's device list and the
resource advisor's GPU speed table. Offscreen Qt, synthetic devices and
timings; receipts live under pytest's tmp_path and no GPU is measured.
"""

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject
from shiboken6 import isValid

from darsia.fluidflower import device_choice
from darsia.fluidflower import resource_advisor_panel as ui
from darsia.fluidflower.project import FluidFlowerProject

GIB = 1024**3


def inventory(*names):
    return {
        "devices": [
            {
                "family": "opencl",
                "device_id": str(index),
                "name": name,
                "vendor": "Vendor",
                "platform": "Platform",
                "platform_version": "OpenCL 3.0",
                "driver": "1.0",
                "display_driver": "",
                "fp64": True,
                "memory_bytes": 8 * GIB,
                "max_allocation_bytes": 8 * GIB,
                "unified_memory": False,
                "pci_bus_id": "",
                "free_bytes": 8 * GIB,
                "free_source": "test",
            }
            for index, name in enumerate(names)
        ],
        "runtime": {"opencl": "fake"},
        "notes": {},
    }


def measure(seconds):
    """Return a measure stub giving each device its seconds and the CPU 2 s."""

    def runner(request):
        if request["family"] == "cpu":
            return {"status": "measured", "seconds_per_frame_median": 2.0}
        return {
            "status": "measured",
            "seconds_per_frame_median": seconds[int(request["device_id"])],
            "peak_device_bytes": GIB,
        }

    def run(devices, *, precision, shape):
        return device_choice.measure_devices(
            devices, precision=precision, shape=shape, runner=runner
        )

    return run


def store(path, precision="float32", shape=(40, 60)):
    return device_choice.measure_and_store(
        precision=precision,
        shape=shape,
        receipt_path=path,
        inventory=inventory("Slow", "Fast"),
        measure=measure([0.5, 0.1]),
    )


def cells(table, row):
    return [table.item(row, column).text() for column in range(table.columnCount())]


def advisor(qtbot, **options):
    def forbidden():
        pytest.fail("No hardware provider may run from the GPU speed section")

    panel = ui.ResourceAdvisorPanel(inventory_provider=forbidden, **options)
    qtbot.addWidget(panel)
    return panel


def test_gpu_speed_table_shows_the_stored_receipt_without_measuring(qtbot, tmp_path):
    path = tmp_path / "receipt.json"
    store(path)
    panel = advisor(
        qtbot,
        device_receipt_path=path,
        device_benchmark=lambda *_: pytest.fail("opening must not measure"),
    )
    rows = [cells(panel.gpu_table, row) for row in range(panel.gpu_table.rowCount())]
    assert [row[0] for row in rows] == [
        "OpenCL 0: Slow",
        "OpenCL 1: Fast",
        "CPU (NumPy, same mass algebra)",
    ]
    assert rows[1][2:7] == ["float32", "40x60", "0.1000", "1.00 GiB", "Yes"]
    assert rows[2][4:7] == ["2.0000", "Host memory", ""]
    assert "Stored receipt for 2 GPU device(s)" in panel.gpu_label.text()
    assert any(
        "never switched automatically" in label.text()
        for label in panel.findChildren(ui.QLabel)
    )


def test_missing_receipt_says_nothing_is_measured_yet(qtbot, tmp_path):
    panel = advisor(qtbot, device_receipt_path=tmp_path / "none.json")
    assert panel.gpu_table.rowCount() == 0
    assert panel.gpu_label.text() == "No GPU speed measurement is stored on this machine yet."


def test_measure_gpu_speed_runs_in_a_thread_with_the_jobs_precision_and_size(
    qtbot, tmp_path
):
    path = tmp_path / "receipt.json"
    calls, emitted = [], []

    def benchmark(precision, shape):
        calls.append((precision, shape))
        return store(path, precision, shape)

    panel = advisor(qtbot, device_receipt_path=path, device_benchmark=benchmark)
    panel.gpu_speed_measured.connect(emitted.append)
    panel.set_context({"job_id": "j", "precision": "float64", "frame_shape_rc": [30, 50]})
    panel.measure_gpu_button.click()
    assert not panel.can_close() and not panel.measure_gpu_button.isEnabled()
    qtbot.waitUntil(lambda: panel._gpu_thread is None, timeout=10000)
    assert calls == [("float64", (30, 50))]
    assert panel.measure_gpu_button.isEnabled() and panel.can_close()
    assert emitted and emitted[0]["format"] == device_choice.RECEIPT_FORMAT
    assert {cells(panel.gpu_table, row)[3] for row in range(panel.gpu_table.rowCount())} == {
        "30x50"
    }


def test_gpu_worker_is_freed_by_its_owner_in_the_gui_thread(qtbot, tmp_path):
    """The GPU speed worker is never deleted in its own thread.

    Deleted there, Qt calls the Python QThread's disconnectNotify under a
    signal-slot lock, which needs the GIL; a GUI thread that holds the GIL and
    waits for that lock deadlocks. The worker's Python owner frees it in the
    GUI thread after the measurement retires.
    """
    panel = advisor(
        qtbot,
        device_receipt_path=tmp_path / "r.json",
        device_benchmark=lambda precision, shape: {"format": "measured"},
    )
    panel.measure_gpu_speed()
    worker = panel._gpu_worker
    qtbot.waitUntil(panel.can_close, timeout=10000)  # Retired; its thread finished.
    assert panel._gpu_worker is None
    assert isValid(worker), "the GPU speed worker was deleted in its own thread"
    watcher = QObject()
    worker.destroyed.connect(watcher.deleteLater)
    del worker  # The last owner lets go here, in the GUI thread.
    QCoreApplication.sendPostedEvents(watcher, QEvent.DeferredDelete)
    assert not isValid(watcher)


def test_measurement_uses_the_default_size_and_reports_failures(qtbot, tmp_path):
    calls = []

    def failing(precision, shape):
        calls.append((precision, shape))
        raise RuntimeError("no OpenCL runtime")

    panel = advisor(qtbot, device_receipt_path=tmp_path / "r.json", device_benchmark=failing)
    panel.measure_gpu_speed()
    qtbot.waitUntil(lambda: panel._gpu_thread is None, timeout=10000)
    assert calls == [("float32", device_choice.DEFAULT_SHAPE)]
    assert "no OpenCL runtime" in panel.gpu_label.text()
    assert panel.measure_gpu_button.isEnabled()


def test_unstored_receipt_is_reported_after_measuring(qtbot, tmp_path):
    panel = advisor(
        qtbot,
        device_receipt_path=tmp_path / "r.json",
        device_benchmark=lambda precision, shape: {"store_error": "PermissionError: denied"},
    )
    panel.measure_gpu_speed()
    qtbot.waitUntil(lambda: panel._gpu_thread is None, timeout=10000)
    assert "not stored: PermissionError: denied" in panel.gpu_label.text()


def test_default_receipt_path_is_this_machines(qtbot, tmp_path, monkeypatch):
    path = tmp_path / "machine" / "receipt.json"
    store(path)
    monkeypatch.setattr(device_choice, "default_receipt_path", lambda: path)
    panel = advisor(qtbot)
    assert panel._gpu_receipt_path() == path
    assert panel.gpu_table.rowCount() == 3


@pytest.fixture
def page(qtbot, tmp_path):
    from darsia.fluidflower.pages.execution.page import ExecutionPage

    project = FluidFlowerProject.create(
        tmp_path / "project",
        name="Synthetic",
        series_id="synthetic",
        template_id="custom",
        rig_type="custom",
    )
    value = ExecutionPage(project)
    qtbot.addWidget(value)
    return value


def hardware(*names):
    return {
        "available_backends": ["cpu-float64", "cpu-float32", "opencl-float32"],
        "cpu": {"logical_cores": 8, "physical_cores": 4},
        "memory": {"total_bytes": 16 * GIB, "available_bytes": 8 * GIB},
        "accelerators": [
            {
                "backend": "opencl-float32",
                "supported_backends": ["opencl-float32"],
                "family": "opencl",
                "device_id": str(index),
                "name": name,
            }
            for index, name in enumerate(names)
        ],
    }


def select(page, backend, adapter="mass-analysis-accelerated-v1"):
    page.execution_module.setCurrentIndex(page.execution_module.findData("mass_analysis"))
    page.execution_adapter.setCurrentIndex(page.execution_adapter.findData(adapter))
    page.execution_backend.setCurrentIndex(page.execution_backend.findData(backend))


def entries(page):
    combo = page.execution_device
    return [(combo.itemText(i), combo.itemData(i)) for i in range(combo.count())]


def test_production_page_offers_the_measured_choice_or_a_device(page):
    page._hardware_ready(hardware("Integrated", "Discrete"))
    select(page, "opencl-float32")
    assert entries(page) == [
        ("Fastest, measured at job start", ""),
        ("0: Integrated", "0"),
        ("1: Discrete", "1"),
    ]
    assert page.execution_device.isEnabled()
    assert device_choice.requested_device_resources(
        page.inventory, "opencl-float32", page.execution_device.currentData()
    )["device_choice"] == "measured"
    page.execution_device.setCurrentIndex(2)
    page._refresh_device_choices()
    assert page.execution_device.currentData() == "1"
    select(page, "cpu-float32", adapter="mass-analysis-cpu-float32-v1")
    assert entries(page) == [("CPU", "")] and not page.execution_device.isEnabled()


def test_production_page_with_one_device_keeps_the_old_behaviour(page):
    page._hardware_ready(hardware("Only GPU"))
    select(page, "opencl-float32")
    assert entries(page) == [("Only device", ""), ("0: Only GPU", "0")]
    assert not page.execution_device.isEnabled()
    assert device_choice.requested_device_resources(page.inventory, "opencl-float32", "") == {
        "device_id": "0",
        "device_name": "Only GPU",
        "device_choice": "single_device",
    }
