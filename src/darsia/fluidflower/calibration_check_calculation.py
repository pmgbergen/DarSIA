"""Current-protocol checks from original frozen N1 evaluation, without refitting.

The five scoring functions are an unchanged arithmetic port of the original
Round-4 amendment-1 scorer, retained by Round-5 section D. The old scorer and
its old authority pins are never called or relabelled. All job ownership,
publication and P1 completion checks remain with their original readers.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import statistics

# The accepted protocol is pinned by the series (decisions.calibration_protocol).
PROTOCOL_SETTING = "calibration_protocol"
MAX_PROTOCOL_BYTES = 1024 * 1024
SCORING_SOURCE_SHA256 = "63b38d2ab92143957d3ac3f1fe64685a9788089e69efcae7d2e91dcd1d3abfa7"
HELDOUT = ['ac19', 'ac25', 'ac35', 'ac41', 'ac47', 'ac52', 'ac56', 'ac60', 'ac21', 'ac22', 'ac32', 'ac44']
GROUPS = {'rig_light': ['ac19', 'ac25', 'ac35', 'ac41', 'ac47'],
          'new_lamps': ['ac52', 'ac56', 'ac60'],
          'caveat': ['ac21', 'ac22', 'ac32', 'ac44']}
CHECK1_T = [1.0, 2.25]
CHECK2_T = [4.0, 8.0, 12.0, 24.0, 48.0]
CHECK3_T = [56.0, 64.0, 72.0, 96.0]

def delta(M, Elo, Ehi, Urest, d):
    if Urest is None or Elo is None or Ehi is None:
        return None
    lo, hi = M + Elo, M + Ehi + Urest
    if lo <= d <= hi:
        return 0.0
    return (lo - d) / d if d < lo else (d - hi) / d

def at(rows, t, runs):
    out = {}
    for r in rows:
        if r['t'] is not None and abs(r['t'] - t) < 1e-9 and r['run'] in runs:
            if r['run'] in out:
                raise ValueError(f'Duplicate {r["run"]} t={t}')
            out[r['run']] = r
    return out

def late_check(rows, times, runs, need_assessable, u0):
    per_time, assessable, ok = {}, 0, True
    for t in times:
        sel = at(rows, t, runs)
        def W_of(r):
            if u0:
                return None if (r['Elo'] is None or r['Ehi'] is None) else r['Ehi'] - r['Elo']
            return None if (r['W'] is None or r['Urest'] is None) else r['W']
        def d_of(r):
            return delta(r['M'], r['Elo'], r['Ehi'], 0.0 if u0 else r['Urest'], r['d'])
        counted = {k: r for k, r in sel.items() if W_of(r) is not None and W_of(r) <= 0.05 * r['d']}
        deltas = {k: d_of(r) for k, r in counted.items()}
        entry = dict(runs_with_frame=len(sel), runs_counted=len(counted),
                     per_run={k: dict(delta=d_of(r), M_over_d=r['M'] / r['d'],
                                      W_over_d=None if W_of(r) is None else W_of(r) / r['d'],
                                      counted=k in counted, set=r['set']) for k, r in sorted(sel.items())})
        if len(counted) >= 7:
            assessable += 1
            med = statistics.median(deltas.values())
            entry.update(assessable=True, median_delta=med, time_pass=med <= 0.10)
            ok = ok and med <= 0.10
        else:
            entry.update(assessable=False)
        per_time[str(t)] = entry
    status = 'NOT_ASSESSABLE' if assessable < need_assessable else ('PASS' if ok else 'FAIL')
    return dict(rule='U_rest_zero_fallback' if u0 else 'primary_U_rest_capacity', status=status,
                assessable_times=assessable, need_assessable=need_assessable, per_time=per_time)

def check1(rows):
    res, ok = {}, True
    for t in CHECK1_T:
        sel = at(rows, t, HELDOUT)
        ratios = {k: r['M'] / r['d'] for k, r in sel.items()}
        med_abs = statistics.median(abs(x - 1) for x in ratios.values())
        groups = {g: statistics.median([ratios[k] for k in members if k in ratios]) for g, members in GROUPS.items()}
        t_ok = len(sel) == 12 and med_abs <= 0.10 and all(0.90 <= v <= 1.10 for v in groups.values())
        ok = ok and t_ok
        res[str(t)] = dict(runs=len(sel), median_abs_error=med_abs, group_median_ratio=groups,
                           per_run_ratio=dict(sorted(ratios.items())), time_pass=t_ok,
                           missing=[k for k in HELDOUT if k not in sel])
    return dict(status='PASS' if ok else 'FAIL', per_time=res)

def decide(primary, fallback):
    if primary['status'] == 'NOT_ASSESSABLE':
        return dict(deciding_rule=fallback['rule'], status=fallback['status'])
    return dict(deciding_rule=primary['rule'], status=primary['status'])


def _finite_predictions(predictions, kind, runs):
    """Translate only original evaluated scalar fields, retaining selected times."""
    result = []
    identities = set()
    for value in predictions:
        run, frame = value["run"], value["frame_index"]
        if run not in runs or (run, frame) in identities:
            raise ValueError("Prediction run/frame identity is duplicated or outside its accepted role")
        identities.add((run, frame))
        row = dict(set=kind, run=run, frame=frame, t=value["selected_target_h"],
            elapsed_h=value["elapsed_h"], M=value["total_mass_g"], Elo=value["E_lo_g"],
            Ehi=value["E_hi_g"], Urest=value["U_rest_g"], W=value["W_g"], d=value["injected_mass_g"])
        for key in ("t", "elapsed_h", "M", "Elo", "Ehi", "Urest", "W", "d"):
            number = row[key]
            if number is None and key in ("t", "Elo", "Ehi", "Urest", "W"):
                continue
            if type(number) not in (int, float) or not math.isfinite(number):
                raise ValueError("Original evaluated prediction requires finite scalar " + key)
        if ((row["t"] in CHECK1_T + CHECK2_T + CHECK3_T and row["d"] <= 0)
                or (row["W"] is not None and row["W"] < 0)):
            raise ValueError("Checks need a positive measured/declared dose and nonnegative width")
        result.append(row)
    if {row["run"] for row in result} != set(runs):
        raise ValueError("Predictions omit a required accepted run")
    return result


def score_predictions(heldout_predictions, loro_predictions, *, fit_runs):
    """Unchanged check arithmetic on original fixed-parameter predictions.

    This pure function grants no source/publication acceptance. The public
    producer below separately authenticates every input before invoking it.
    """
    if len(fit_runs) != len(set(fit_runs)) or set(fit_runs) & set(HELDOUT):
        raise ValueError("Distinct fit runs and separate held-out runs are required")
    held = _finite_predictions(heldout_predictions, "heldout", HELDOUT)
    loro = _finite_predictions(loro_predictions, "fit_loro", fit_runs)
    for time in CHECK1_T:
        if set(at(held, time, HELDOUT)) != set(HELDOUT):
            raise ValueError("Check 1 needs each held-out run at the prescribed time: " + str(time))
    c1 = check1(held)
    c2p = late_check(held, CHECK2_T, HELDOUT, 4, u0=False)
    c2f = late_check(held, CHECK2_T, HELDOUT, 4, u0=True)
    pooled = held + loro
    pooled_runs = set(HELDOUT) | set(fit_runs)
    c3p = late_check(pooled, CHECK3_T, pooled_runs, 3, u0=False)
    c3f = late_check(pooled, CHECK3_T, pooled_runs, 3, u0=True)
    d2, d3 = decide(c2p, c2f), decide(c3p, c3f)
    verdict = 'PASS' if (c1['status'] == 'PASS' and d2['status'] == 'PASS' and d3['status'] == 'PASS') else 'FAIL'
    return dict(contract="darsia-calibration-check-scores-v1", verdict=verdict,
        check1=c1, check2=dict(primary=c2p, fallback=c2f, decision=d2),
        check3=dict(primary=c3p, fallback=c3f, decision=d3), integrity_problems=[],
        counts=dict(heldout_rows=len(held), loro_rows=len(loro), heldout_runs=sorted(HELDOUT),
                    loro_runs=sorted(fit_runs)),
        thresholds=["Check 1: median |M/d - 1| <= 0.10 and every lighting-group median 0.90..1.10",
                    "Checks 2/3: W/d <= 0.05; >=7 runs/time; median delta <=0.10; >=4/5 and >=3/4 assessable times",
                    "U_rest=0 may decide only a primary NOT_ASSESSABLE check"],
        main_mass_definition="Original n1_statistics.evaluate_summary total_mass_g, unchanged domain rule",
        numerical_fit_performed=False, calibration_adopted=False, production_started=False)


class _Inputs:
    """One immutable read set, using original metadata and DATA readers."""
    def __init__(self, series_root):
        from .jobs_metadata import JobSnapshotReader
        from .mass.n1_pipeline import DataReader
        from .pipeline.alias_job import _manifest
        from .pipeline.job_queue import descriptor
        self.metadata_reader = JobSnapshotReader()
        self.pins = {}
        self.data = DataReader(unc_only=False)
        self.manifest = self.pin(Path(series_root) / "series.toml")
        _path, self.series, self.roots = _manifest(self.manifest)
        self.recheck(self.manifest)

    def pin(self, path):
        from .pipeline.job_inputs import MAX_METADATA_BYTES
        path = Path(path)
        with path.open("rb") as stream:
            raw = stream.read(MAX_METADATA_BYTES + 1)
        if len(raw) > MAX_METADATA_BYTES:
            raise ValueError("Check input metadata exceeds the original metadata limit")
        value = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw))
        key = str(path.resolve())
        old = self.pins.setdefault(key, value)
        if old != value:
            raise ValueError("Check input changed while being consumed")
        return value

    def recheck(self, expected):
        """Check exact raw bytes; the original consuming reader owns the format."""
        pin = self.pin(expected["path"])
        if (pin["sha256"] != expected["sha256"]
                or pin["size_bytes"] != expected.get("size_bytes", expected.get("bytes"))):
            raise ValueError("Check input differs from its original binding")
        return pin

    def metadata(self, path, expected=None):
        pin = self.pin(path) if expected is None else self.recheck(expected)
        return self.metadata_reader.read(path, pin)

    def reference(self, reference):
        from .store_map import validate_input_reference, input_reference, require_real_containment
        validate_input_reference(reference)
        root = self.roots[reference["root_alias"]]
        path = require_real_containment(root, Path(root).joinpath(*reference["relative_path"].split("/")))
        if input_reference(str(path), self.roots, sha256=reference["sha256"],
                size_bytes=reference["bytes"], role=reference["role"]) != reference:
            raise ValueError("Job input reference differs from the selected registered roots")
        bound = dict(path=str(path), sha256=reference["sha256"], size_bytes=reference["bytes"])
        self.recheck(bound)
        return bound

    def terminal(self, root, run, pipeline):
        """Require original alias ownership, done, indexed publication and lineage."""
        from .pipeline.job_queue import name
        from .pipeline.common_calibration import refit_terminal
        from .published_jobs_snapshot import _phase1_terminal
        root, run = Path(root), name(run)
        job = self.metadata(root / "job.json")
        if (job.get("format") != "darsia-alias-job-v1" or job.get("pipeline") != pipeline
                or job.get("precision") != "float32" or run not in job.get("expected_runs", ())):
            raise ValueError("Checks require an original completed float32 job of the selected kind")
        if self.reference(job["series_manifest"]) != self.manifest:
            raise ValueError("The completed job belongs to another registered series")
        task = self.metadata(root / "results" / run / "task.json")
        done = self.metadata(root / "done" / (run + ".json"))
        function = refit_terminal if pipeline == "n1-calibration-refit" else _phase1_terminal
        result, _machines, _lineage = function(root, task, job, done, self.metadata)
        from .pipeline.job_queue import _own_asset
        index = self.pin(_own_asset(root, done["products_index"]))
        from .pipeline.products_publication import _read_index
        from .published_jobs_snapshot import _SnapshotMetadata
        folder, _document, rows = _read_index(index)
        shared_metadata = _SnapshotMetadata(index, folder, rows)
        metadata_aliases = []
        for leaf, row in rows.items():
            if "metadata_copy" not in row:
                continue
            shared, _raw = shared_metadata.read(leaf)
            physical = dict(path=shared["path"], sha256=shared["sha256"], size_bytes=shared["bytes"])
            logical = dict(path=row["local_path"], sha256=row["sha256"], size_bytes=row["bytes"])
            # Pin all published metadata as bytes, including .json.gz. Its
            # original consuming reader remains responsible for decoding it.
            self.recheck(physical)
            if leaf.endswith(".json"):
                self.metadata(physical["path"], physical)
            metadata_aliases.append(dict(logical=logical, physical=physical))
        shared_metadata.finish()
        def product(leaf):
            pin, _raw = shared_metadata.read(leaf)
            normalized = dict(path=pin["path"], sha256=pin["sha256"], size_bytes=pin["bytes"])
            value = self.metadata(normalized["path"], normalized)
            shared_metadata.finish()
            return value, normalized
        product.metadata_aliases = tuple(deepcopy(metadata_aliases))
        return job, result, product

    def finish(self):
        """Retain original DATA stamp/hash semantics and recheck all metadata pins."""
        for pin in tuple(self.pins.values()):
            if self.pin(pin["path"]) != pin:
                raise ValueError("Check input changed before score publication")
        from .mass.n1_pipeline import _stamp
        consumed = []
        for (path, sha), stamp in self.data.verified.items():
            if _stamp(Path(path)) != stamp:
                raise ValueError("An original consumed phase-one file changed before scoring finished")
            consumed.append(dict(path=path, sha256=sha, size_bytes=stamp[0]))
        unique = {(value["path"], value["sha256"]): value for value in list(self.pins.values()) + consumed}
        return sorted(unique.values(), key=lambda value: value["path"])


def _run_rows(inputs, phase_job, run, plan, rules):
    from .pipeline.staged_data import verify_phase1_completion
    from .pipeline.common_calibration import phase1_statistics
    from .pipeline.refit_runner import read_statistics
    from .mass.n1_statistics import CONTRACT
    _job, result, _product = inputs.terminal(phase_job, run, "native-phase-statistics")
    located = inputs.reference(result["task"]["configuration"])
    configuration = deepcopy(result["configuration"])
    if (located["sha256"] != configuration["sha256"]
            or located["size_bytes"] != configuration["size_bytes"]):
        raise ValueError("The published P1 configuration differs from its owned task")
    configured = inputs.data.json(located)
    # Retain logical descriptor identities; only locate exact shared byte copies.
    from .mass.n1_pipeline import DataReader
    aliases = list(configured.get("data_aliases", ()))
    aliases.append(dict(logical=configuration, physical=located))
    aliases.extend(deepcopy(_product.metadata_aliases))
    own_reader = DataReader(aliases, unc_only=False)
    verified = verify_phase1_completion(result["phase1_completion"], configuration,
        expected_commit=result["commit"], reader=own_reader)
    if verified["configuration"]["run"] != run:
        raise ValueError("The verified phase-one completion belongs to another run")
    entries = phase1_statistics(verified)
    if run in rules.fit_runs and entries != plan["statistics"][run]:
        raise ValueError("Fit statistics differ from the exact frozen plan's original P1 order")
    def rows():
        # The source order was checked above. predict_run sorts by frame index;
        # retain that same output order while decoding at most one row at a time.
        ordered = sorted(zip(entries, verified["configuration"]["frames"]),
                         key=lambda pair: pair[0]["frame_index"])
        for entry, spec in ordered:
            row = read_statistics(own_reader, entry["statistics"])
            if (row.get("contract") != CONTRACT or row.get("run") != run
                    or row.get("frame_index") != entry["frame_index"]
                    or row.get("elapsed_h") != spec["elapsed_h"]
                    or row.get("selected_target_h") != entry["selected_target_h"]
                    or row.get("source_binding", {}).get("source_sha256") != spec["source"]["source_sha256"]):
                raise ValueError("Check statistics differ from their original selected acquisition")
            for identity, stamp in own_reader.verified.items():
                if inputs.data.verified.setdefault(identity, stamp) != stamp:
                    raise ValueError("An original P1 file identity changed between run reads")
            yield row
    return rows


def _predict_rows(rows, fit, run):
    """Original per-row evaluation in predict_run's identical frame order."""
    from .mass.n1_refit import predict_run
    output = []
    for row in rows():
        output.extend(predict_run([row], fit, run))
    return output


def accepted_protocol_pin(decisions):
    """The series' pinned accepted protocol ``{sha256, size_bytes}`` from its decision settings."""
    from .series_decisions import checked_pin, decision_setting
    return checked_pin(decision_setting(decisions, PROTOCOL_SETTING), "accepted calibration protocol")


def protocol_binding(path, *, series_root):
    """Bound the exact small accepted protocol before any generic descriptor read.

    The series names the accepted protocol by its pinned bytes.
    """
    from .series_decisions import read_series_decisions
    pin = accepted_protocol_pin(read_series_decisions(series_root))
    path = Path(path)
    with path.open("rb") as stream:
        raw = stream.read(min(pin["size_bytes"], MAX_PROTOCOL_BYTES) + 1)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != pin["size_bytes"] or digest != pin["sha256"]:
        raise ValueError("Choose the exact current scoring protocol")
    return dict(path=str(path), sha256=digest, size_bytes=len(raw))


def _global_state(inputs, calibration_job, protocol):
    from .pipeline import refit_runner as refit
    from .pipeline.common_calibration import validate_refit_result
    actual_protocol = inputs.pin(protocol["path"])
    pin = accepted_protocol_pin(inputs.series.get("decisions", {}))
    if (actual_protocol != protocol or protocol["sha256"] != pin["sha256"]
            or protocol["size_bytes"] != pin["size_bytes"]):
        raise ValueError("Checks require the exact current protocol with the retained section-D scoring rules")
    job, result, product = inputs.terminal(calibration_job, "refit", refit.PIPELINE)
    plan_bound = inputs.reference(result["task"]["configuration"])
    plan = inputs.data.json(plan_bound)
    fit, _fit_bound = product("fit.json")
    frozen, frozen_bound = product("frozen_calibration.json")
    _expected, rules = validate_refit_result(result, task=result["task"], expected_commit=job["commit"],
        context="global_application", configuration=plan_bound, plan=plan, fit=fit, frozen=frozen)
    rules.require_variant_f_off()
    declared = inputs.series.get("runs", {}).get("ids", ())
    roles = plan["runs"]
    excluded = {"ac20", "ac24", "ac27", "ac51"}
    fit_runs = list(rules.fit_runs)
    if (len(fit_runs) != 26 or set(roles.get("heldout", ())) != set(HELDOUT)
            or set(roles.get("descriptive", ())) != {"ac38"}
            or set(roles.get("excluded", ())) != excluded
            or set(declared) != set(fit_runs) | set(HELDOUT) | excluded | {"ac38"}
            or set(fit_runs) & (set(HELDOUT) | excluded | {"ac38"})
            or job["expected_runs"] != ["refit"] + ["loro-" + run for run in fit_runs]):
        raise ValueError("The accepted current protocol needs its exact 26 fit/12 held-out/descriptive/excluded roles and all LORO folds")
    return job, plan, rules, frozen, frozen_bound, actual_protocol


def check_readiness(series_root, calibration_job, phase_job, *, protocol):
    """Original completed metadata/ownership only; no predictions or fit.

    P1 payload bytes are verified later by the original verifier before Compute.
    This is readiness to attempt calculation, never a check verdict.
    """
    inputs = _Inputs(series_root)
    job, plan, rules, _frozen, _bound, _protocol = _global_state(inputs, calibration_job, protocol)
    for run in list(rules.fit_runs) + HELDOUT:
        inputs.terminal(phase_job, run, "native-phase-statistics")
    for run in rules.fit_runs:
        _job, fold, _product = inputs.terminal(calibration_job, "loro-" + run, "n1-calibration-refit")
        if fold["stop_required"] is not False:
            raise ValueError("A LORO grid check requires a stop; scores cannot be calculated")
    return dict(ready=True, source_bindings=inputs.finish(), verdict=None,
                payload_bytes_verified=False, numerical_fit_performed=False)


def calculate_check_scores(series_root, calibration_job, phase_job, destination, *, protocol, expected_readiness=None):
    """Evaluate completed frozen global/LORO results, score and publish once.

    This starts no fit, phase calculation, scheduler, queue, calibration adoption
    or production task. Original P1 byte verification can be expensive and must
    run on the page's existing finite background-call route.
    """
    from .pipeline import refit_runner as refit
    from .pipeline.common_calibration import validate_refit_result
    from .mass import n1_refit
    from .pipeline.job_queue import write_immutable, descriptor
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Choose a fresh check-score file")
    inputs = _Inputs(series_root)
    # Bind the exact code actually used by this scalar evaluation and score.
    for module in (Path(__file__), Path(n1_refit.__file__),
                   Path(__file__).parent / "mass" / "n1_statistics.py",
                   Path(__file__).parent / "mass" / "n1_arithmetic.py",
                   Path(__file__).parent / "mass" / "concentration_joint.py"):
        inputs.pin(module)
    if expected_readiness is not None:
        if not isinstance(expected_readiness, dict) or expected_readiness.get("ready") is not True:
            raise ValueError("Verify the current completed inputs before calculating checks")
        for pin in expected_readiness["source_bindings"]:
            inputs.recheck(pin)
    job, plan, rules, frozen, frozen_bound, actual_protocol = _global_state(inputs, calibration_job, protocol)
    fit_runs = list(rules.fit_runs)
    held, loro = [], []
    for run in HELDOUT:
        rows = _run_rows(inputs, phase_job, run, plan, rules)
        # These are exactly the fields the original anchor selector consumes;
        # scientific histograms/pair arrays are not retained between frames.
        thin = [{key: value for key, value in row.items() if key in
                 {"run", "elapsed_h", "frame_index", "source_binding", "affected_pore_volume_m3", "analysis_scope"}}
                for row in rows()]
        anchors = n1_refit.anchors_by_run(thin, rules)
        held.extend(_predict_rows(rows, dict(parameters=frozen["parameters"], anchors_by_run=anchors), run))
    for run in fit_runs:
        fold_job, completed, fold_product = inputs.terminal(calibration_job, "loro-" + run, refit.PIPELINE)
        fold_plan_bound = inputs.reference(completed["task"]["configuration"])
        fold_plan = inputs.data.json(fold_plan_bound)
        expected_plan = deepcopy(plan)
        expected_plan.update(run="loro-" + run, fold=run)
        if fold_plan != expected_plan or fold_job != job:
            raise ValueError("A LORO fold changes the global plan's roles, inputs, rules or baseline")
        fold_fit, _ = fold_product("fit.json")
        validate_refit_result(completed, task=completed["task"], expected_commit=job["commit"],
            context="terminal", fit=fold_fit)
        if completed["stop_required"] is not False or run in fold_fit.get("calibration_runs", ()):
            raise ValueError("A grid-stopped or non-held-out fold cannot enter Check 3")
        predictions, _ = fold_product("predictions.json")
        rows = _run_rows(inputs, phase_job, run, plan, rules)
        actual = _predict_rows(rows, fold_fit, run)
        if (predictions.get("format") != "darsia-n1-refit-predictions-v1"
                or predictions.get("heldout_fit_run") != run
                or predictions.get("parameters") != fold_fit["parameters"]
                or predictions.get("predictions") != actual):
            raise ValueError("Saved LORO predictions do not reproduce the original frozen-parameter evaluation")
        loro.extend(actual)
    scores = score_predictions(held, loro, fit_runs=fit_runs)
    scores.update(protocol=deepcopy(actual_protocol), scoring_rules_source_sha256=SCORING_SOURCE_SHA256,
        frozen_calibration=frozen_bound, parameters=deepcopy(frozen["parameters"]),
        heldout_predictions=held, loro_predictions=loro,
        source_bindings=inputs.finish(), phase_image_arrays_decoded=False,
        statistics_decoded_and_evaluated=True,
        evaluation="Original frozen-parameter n1_refit.predict_run; no refitting or new phase inference")
    from .pipeline.job_queue import _immutable_bytes
    raw = _immutable_bytes(scores)
    from .calibration_checks import MAX_SCORE_BYTES, normalise_check_scores
    if len(raw) > MAX_SCORE_BYTES:
        raise ValueError("Calculated score exceeds the existing check-score file limit")
    normalise_check_scores(scores)  # Original public display contract before publication.
    inputs.finish()
    write_immutable(destination, scores)
    return dict(scores=descriptor(destination), verdict=scores["verdict"],
                numerical_fit_performed=False, calibration_adopted=False, worker_started=False)
