"""Tiny genuine saved results for native calibration-review integration tests."""
import json
import hashlib
from pathlib import Path

import numpy as np
from PySide6.QtGui import QImage, QColor


def add_saved_results(page, root, runs, *, name="Calibration A", reference="fits/frozen.json",
                      times=(0.0, 1.0)):
    from darsia.fluidflower.phase_saved_review_metadata import inspect_result
    from darsia.fluidflower.tools.phase_studio_saved_results import model_browser_metadata

    saved = page.saved_results
    dataset = page.series
    saved.series.blockSignals(True)
    if saved.series.findData(dataset) < 0:
        saved.series.addItem(dataset, dataset)
    saved.series.setCurrentIndex(saved.series.findData(dataset))
    saved.series.blockSignals(False)
    saved._series_catalog_loaded.add(dataset)
    paths = []
    for run in runs:
        folder = root / "results" / name / run
        folder.mkdir(parents=True, exist_ok=True)
        image = QImage(48, 24, QImage.Format.Format_RGB888)
        image.fill(QColor("steelblue"))
        assert image.save(str(folder / "figure.png"))
        np.savez(folder / "fields.npz", photograph_rgb=np.full((6, 12, 3), .4, np.float32),
                 concentration_g_l=np.full((6, 12), .7, np.float32),
                 aqueous_volume_m3=np.full((6, 12), .01, np.float32))
        frames = [dict(frame_index=index, elapsed_h=time, source_name=f"image-{time}.dng",
                       total_mass_g=2., aqueous_mass_g=1.5, gas_mass_g=.5, injected_mass_g=2.,
                       preview="figure.png", fields="fields.npz", color_limit_g_l=2.,
                       source_reference=dict(native_manifest_sha256="manifest", photo_sha256=str(time),
                                             phase_sha256="phase", raw_source_sha256=str(time)))
                  for index, time in enumerate(times)]
        value = dict(complete=True, format="studio-concentration-mass-v2", frames=frames,
                     selected_inputs=dict(candidate=dict(dataset=dataset, run=run, candidate_id="model",
                         params_sha256="params", optical_model_sha256="optical")),
                     calibration=dict(scope="shared", parameters=dict(gain=1., beta=0.),
                                      display_name=name, chemistry_model="titration"),
                     method=dict(settings=dict(chemistry="darsia_titration_flash")),
                     series_calibration=str(root / reference), mass_curve="figure.png")
        if (root / reference).is_file():
            value["bindings"] = dict(calibration_sha256=hashlib.sha256((root / reference).read_bytes()).hexdigest())
        path = (folder / "result.json").resolve()
        path.write_text(json.dumps(value), encoding="utf-8")
        entry = inspect_result(path, value)
        entry.update(model_browser_metadata(value, path))
        saved.entries.append(entry)
        paths.append(path)
    saved.refresh()
    return paths
