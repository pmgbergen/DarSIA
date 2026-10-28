"""Read frozen overflow mass targets without deriving or fitting a reference."""
from __future__ import annotations
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
from .phase_saved_review_metadata import configured_parameters

def _sha256(path: Path) -> str:
    from .project import _sha256_file_uncached
    return _sha256_file_uncached(path)


@dataclass(frozen=True)
class OverflowPhaseTarget:
    time_h: float
    injected_mass_kg: float
    gas_mass_kg: float
    aqueous_mass_kg: float
    phase_mass_sigma_kg: float
    cumulative_outflow_g: float
    inferred_gas_volume_ml: float
    target_source: str = ""


@dataclass(frozen=True)
class OverflowPhaseConstraint:
    run: str
    path: Path
    sha256: str
    targets: tuple[OverflowPhaseTarget, ...]
    match_tolerance_h: float
    provenance: Mapping[str, Any]
    constraint_mode: str = field(default_factory=lambda: configured_parameters("overflow_reference")["default_constraint_mode"])

    def target_at(self, time_h: float) -> OverflowPhaseTarget | None:
        if not self.targets:
            return None
        nearest = min(self.targets, key=lambda target: abs(target.time_h - time_h))
        if abs(nearest.time_h - float(time_h)) > self.match_tolerance_h:
            return None
        return nearest

    def to_dict(self) -> dict[str, Any]:
        return {
            "run": self.run,
            "path": str(self.path),
            "sha256": self.sha256,
            "match_tolerance_h": self.match_tolerance_h,
            "target_count": len(self.targets),
            "target_times_h": [target.time_h for target in self.targets],
            "constraint_mode": self.constraint_mode,
            "provenance": dict(self.provenance),
        }


def normalize_constraint_mode(value: Any, *, parameters: Mapping[str, Any]) -> str:
    mode = str(value or parameters["default_constraint_mode"]).strip().lower().replace("_", "-")
    aliases = parameters["constraint_mode_aliases"]
    mode = aliases.get(mode, mode)
    if mode not in set(parameters["valid_constraint_modes"]):
        raise ValueError(
            f"Unsupported overflow constraint mode {value!r}; expected one of "
            f"{sorted(set(parameters["valid_constraint_modes"]))}"
        )
    return mode


def load_overflow_phase_constraint(
    path: str | Path,
    *,
    run: str,
    parameters: Mapping[str, Any],
) -> OverflowPhaseConstraint:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8-sig"))
    if int(payload.get("schema", parameters["schema_absent"])) != parameters["schema"]:
        raise ValueError(f"Unsupported overflow phase-target schema in {source}")
    target_run = str(payload.get("run", "")).strip().lower()
    if target_run != str(run).strip().lower():
        raise ValueError(
            f"Overflow phase targets are for {target_run!r}, not {run!r}: {source}"
        )
    raw_targets = payload.get("targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError(f"Overflow phase target file has no targets: {source}")
    targets = tuple(
        OverflowPhaseTarget(
            time_h=float(row["time_h"]),
            injected_mass_kg=float(row["injected_mass_kg"]),
            gas_mass_kg=float(row["gas_mass_kg"]),
            aqueous_mass_kg=float(row["aqueous_mass_kg"]),
            phase_mass_sigma_kg=float(row["phase_mass_sigma_kg"]),
            cumulative_outflow_g=float(row["cumulative_outflow_g"]),
            inferred_gas_volume_ml=float(row["inferred_gas_volume_ml"]),
            target_source=str(row.get("target_source", "")),
        )
        for row in raw_targets
    )
    times = [target.time_h for target in targets]
    if times != sorted(times) or len(times) != len(set(times)):
        raise ValueError(f"Overflow phase target times must be unique and sorted: {source}")
    for target in targets:
        values = (
            target.time_h,
            target.injected_mass_kg,
            target.gas_mass_kg,
            target.aqueous_mass_kg,
            target.phase_mass_sigma_kg,
            target.cumulative_outflow_g,
            target.inferred_gas_volume_ml,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError(f"Non-finite overflow phase target in {source}: {target}")
        if target.time_h < 0.0 or target.injected_mass_kg < 0.0:
            raise ValueError(f"Invalid overflow phase target in {source}: {target}")
        if not 0.0 <= target.gas_mass_kg <= target.injected_mass_kg + parameters["mass_rounding_floor_kg"]:
            raise ValueError(f"Gas target is outside injected mass in {source}: {target}")
        if target.phase_mass_sigma_kg < 0.0:
            raise ValueError(f"Negative target uncertainty in {source}: {target}")
    tolerance = float(
        payload.get("match_tolerance_h", parameters["default_match_tolerance_h"])
    )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError(f"Invalid match tolerance in {source}: {tolerance}")
    constraint_mode = normalize_constraint_mode(payload.get("constraint_mode"), parameters=parameters)
    return OverflowPhaseConstraint(
        run=target_run,
        path=source.resolve(),
        sha256=_sha256(source),
        targets=targets,
        match_tolerance_h=tolerance,
        constraint_mode=constraint_mode,
        provenance={
            "model": payload.get("model"),
            "model_version": payload.get("model_version"),
            "source": payload.get("source"),
            "physics": payload.get("physics"),
            "derivation": payload.get("derivation"),
        },
    )
