"""Saved timeline and mass-candidate metadata for review and library catalogues."""
from __future__ import annotations
from pathlib import Path
from .review_storage import resolve_review_path
from .phase_saved_review_metadata import _read, configured_parameters


def native_source(job):
    from darsia.fluidflower.phase_training.radiometry import analysis_review_root
    run = str(job["run"]).lower()
    root = resolve_review_path(analysis_review_root(resolve_review_path(job.get("analysis_source_review_root") or job["review_output_dir"]), run))
    path = root / "native_frames" / run / "manifest.json"
    return root, path, _read(path)

def mass_candidate(validation, job):
    """Adapt production maps without requiring optional Studio movie/plot files."""
    from darsia.fluidflower.tools.calibration_pairwise_review import Candidate, _candidate_from_job
    if job.get('output_mode', 'review') == 'review':
        return _candidate_from_job(validation, job)
    if job.get('output_mode') != 'production':
        raise ValueError('Unknown phase output mode')
    def path(value):
        value = Path(value)
        return resolve_review_path(value if value.is_absolute() else Path(validation) / value)
    folder = path(job['candidate_folder'])
    summary = _read(folder / 'summary.json')
    if (summary.get('output_mode') != 'production'
            or summary.get('selection_candidate_id') != job['candidate_id']
            or summary.get('params_sha256') != job['params_sha256']):
        raise ValueError('Production mass metadata identifies another phase candidate')
    return Candidate(run=job['run'], role=job['role'], family=job.get('family', 'unknown'),
        variant=job['variant'], candidate_id=job['candidate_id'], grid_order=job.get('grid_order', 1),
        params_sha256=job['params_sha256'], normalized_mae=None,
        # Reserved paths are deliberately absent in production. The mass
        # adapter consumes the native photos, final maps and timeline instead.
        mass_plot=folder/'calibration_mass_fit.png', contour_movie=folder/'calibration_review.mp4',
        per_frame_csv=folder/'calibration_review_per_frame.csv', summary_path=folder/'summary.json',
        plot_seconds=0., image_seconds=0., fps=0., frame_count=int(summary['frame_count']),
        quality_flags=job.get('quality_flags', {}), dataset=job['dataset'],
        review_output_dir=path(job['review_output_dir']), phase_map_output_dir=path(job['phase_map_output_dir']),
        phase_model_only=True, optical_model_sha256=job['optical_model_sha256'],
        presence_diagnostics_dir=path(job['presence_diagnostics_dir']) if job.get('presence_diagnostics_dir') else None,
        phase_support_format=job.get('phase_support_format', ''))

def stamp(path):
    stat = Path(path).stat()
    return [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]

def selection(request, method):
    parameters = configured_parameters("mass_review_cache")
    candidate = request["candidate"]
    return dict(candidate={key: candidate.get(key) for key in
                ("dataset", "run", "candidate_id", "params_sha256", "optical_model_sha256")},
                local_baseline_search=bool(request.get("local_baseline_search")),
                method=dict(method),
                mass_policy=dict(request.get("mass_policy", parameters["default_mass_policy"])),
                analysis_width=int(request.get("analysis_width", parameters["analysis_width"])))
