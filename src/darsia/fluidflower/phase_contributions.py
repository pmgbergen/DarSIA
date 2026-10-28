"""Explain saved training evidence; counts are not causal model importance."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

LABELS = {
    "gas": "Only free CO2", "aqueous": "Only dissolved CO2", "water": "Water / no CO2",
    "mixed_gas_aqueous": "Free + dissolved CO2",
    "mixed_aqueous_water": "Dissolved CO2 + water",
    "mixed_gas_aqueous_water": "Free + dissolved CO2 + water",
    "gas_absent_region": "No free CO2", "gas_present_region": "Free CO2 present",
    "aqueous_absent_region": "No dissolved CO2", "aqueous_present_region": "Dissolved CO2 present",
    "co2_absent_region": "No CO2", "current_experiment_co2_artifact": "Optical artifact",
}


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _records(document):
    from .phase_training.logistic import canonical_phase_class
    for run, decision in (document.get("decisions") or {}).items():
        for field, source in (("annotations", "manual"), ("phase_quiz_answers", "quiz")):
            for item in decision.get(field) or []:
                category = canonical_phase_class(item, source)
                if not category or category == "ignored":
                    continue
                identifier = item.get("id") or item.get("proposal_id")
                if not identifier:
                    from .phase_training.trainer import _geometry_key
                    identifier = _geometry_key(run.casefold(), item)[:16]
                key = f"{run.casefold()}:{source}:{identifier}"
                # Display timestamps/selection state do not make an annotation new.
                signature = json.dumps({**{k: item.get(k) for k in (
                    "frame_index", "shape", "points", "x", "y", "width", "height",
                    "x0", "y0", "x1", "y1", "confidence", "bbox", "rect",
                    "coordinate_space")}, "phase_class": category}, sort_keys=True)
                yield key, run.casefold(), category, signature


def regional_inventory(model_root):
    """Recover unique used regions from the hash-bound regional fit receipts."""
    from .project import sha256_file
    from .phase_training import logistic
    root = Path(model_root)
    names = ('regional_model.json', 'evidence.json', 'extraction.json', 'regional_training_protocol.json')
    if not all((root / name).is_file() for name in names):
        return None
    model, evidence, extraction, protocol = (_read(root / name) for name in names)
    if not protocol.get('extraction_sha256'):
        return None  # Older fits did not freeze extraction coverage.
    for name, expected in (('evidence.json', model['evidence_sha256']),
                           ('regional_training_protocol.json', model['protocol_sha256']),
                           ('extraction.json', protocol['extraction_sha256'])):
        if sha256_file(root / name) != expected:
            raise ValueError(f'Regional training inventory changed: {name}')
    fit = model['fit_report']
    if fit.get('policy') != 'joint-hierarchical-aqueous-mil-v1':
        raise ValueError('Unsupported regional training inventory policy')
    coverage = {row['region_id']: row for row in extraction['coverage']}
    coupled = {row['region_id'] for row in fit['coupled_regions'] if row['run_normalized_weight'] > 0}
    rows, active_rows = {}, 0
    for value in evidence['regions']:
        region = logistic.ReviewedRegion(**value)
        used = coverage.get(region.region_id, {})
        count = int(used.get('training_rows', 0))
        active = region.region_id in coupled or any(
            logistic.pixel_target(region.phase_class, head) is not None or (
                fit.get('weak_strength', 0) > 0 and
                logistic.regional_presence_target(region.phase_class, head) is not None)
            for head in ('co2_presence', 'gas_within_co2'))
        if count <= 0 or region.supervision_weight <= 0 or not active:
            continue
        if region.region_id in rows:
            raise ValueError('Duplicate regional training identity')
        if (used['run'], used['phase_class']) != (region.run, region.phase_class):
            raise ValueError('Extraction coverage identifies another region')
        active_rows += count
        rows[region.region_id] = dict(region_id=region.region_id, run=region.run.casefold(),
            phase_class=region.phase_class, supervision_weight=region.supervision_weight)
    if active_rows != fit['active_feature_rows']:
        raise ValueError('Regional inventory does not reproduce fitted active feature coverage')
    return list(rows.values())


def publish_regional_inventory(model_root):
    """Also provide the existing CSV contract to already-running Studio windows."""
    rows = regional_inventory(model_root)
    if rows is None:
        return False
    path = Path(model_root) / 'region_inventory.csv'
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('region_id', 'run', 'phase_class', 'supervision_weight'))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)
    return True


def model_contributions(model_root, pending_sessions=()):
    root = Path(model_root)
    trained, signatures = {}, {}
    known = (root / "region_inventory.csv").is_file()
    summary = _read(root / "summary.json") if (root / "summary.json").is_file() else {}
    if known:
        with (root / "region_inventory.csv").open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if float(row.get("supervision_weight") or 0) > 0:
                    trained[row["region_id"]] = (row["run"].casefold(), row["phase_class"])
    else:
        regional = regional_inventory(root)
        if regional is not None:
            known = True
            trained = {row['region_id']: (row['run'], row['phase_class']) for row in regional}
    request = _read(root / "training_request.json") if (root / "training_request.json").is_file() else {}
    for bundle in request.get("bundles") or []:
        path = Path((bundle.get("decisions") or {}).get("snapshot_path") or "")
        if not path.is_file():
            path = root / "input_snapshot" / path.name
        if path.is_file():
            signatures.update((key, signature) for key, _, _, signature in _records(_read(path)))
    current = {}
    errors = []
    # Saved copies of the same region count once; newest saved version wins.
    for session in sorted(pending_sessions, key=lambda s: float(s.get("modified_timestamp") or 0)):
        path = Path(session.get("decision_path") or "")
        try:
            for key, run, category, signature in _records(_read(path)):
                current[key] = (run, category, signature)
        except (OSError, ValueError) as error:
            errors.append(f"{path.name}: {error}")
    counts = defaultdict(lambda: dict(trained=Counter(), pending=Counter()))
    for run, category in trained.values():
        counts[run]["trained"][category] += 1
    for key, (run, category, signature) in current.items():
        if key not in trained or trained[key] != (run, category) or (
                key in signatures and signature != signatures[key]):
            counts[run]["pending"][category] += 1
    return dict(known=known, runs={run: {k: dict(v) for k, v in row.items()} for run, row in counts.items()},
        inherited_from=summary.get("inherited_optical_model_id"), errors=errors,
        description="Unique training regions with positive weight; not pixels or model influence. "
                    "New/changed counts are saved eligible annotations, subject to the trainer's final checks.")
