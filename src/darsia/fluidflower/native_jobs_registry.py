"""Local UI associations for explicitly selected UNC jobs/adopted plans.

No NAS discovery, worker, producer, numerical code or series/config write.
Only this UI-owned local registry is saved after an explicit user action.
"""

from copy import deepcopy
import hashlib
import json
import ntpath
import os
from pathlib import Path
import tomllib
import uuid
from datetime import datetime, timezone

from .pipeline.job_queue import descriptor, name, read_json, require_unc


def _path(value):
    return ntpath.normcase(ntpath.normpath(str(value)))


def _asset(path):
    path = Path(path)
    before = descriptor(path)
    raw = path.read_bytes()
    if (
        len(raw) != before["size_bytes"]
        or hashlib.sha256(raw).hexdigest() != before["sha256"]
    ):
        raise ValueError("Selected metadata changed while read")
    return raw, before


class JobsUIRegistry:
    def __init__(self, root):
        self.root = Path(root)

    def _file(self, series_root, series_id):
        identity = json.dumps(
            [_path(series_root), str(series_id)], ensure_ascii=False
        ).encode()
        return self.root / (hashlib.sha256(identity).hexdigest() + ".json")

    def context(self, series_root, series_id):
        path = self._file(series_root, series_id)
        if not path.exists():
            return dict(
                format="darsia-native-jobs-registry-v1",
                series_root=str(series_root),
                series_id=str(series_id),
                jobs=[],
                plans=[],
                active_job=None,
            )
        doc = read_json(path)
        if (
            doc.get("format") != "darsia-native-jobs-registry-v1"
            or _path(doc.get("series_root", "")) != _path(series_root)
            or doc.get("series_id") != series_id
        ):
            raise ValueError("Local Jobs registration belongs to another series")
        if not isinstance(doc.get("jobs"), list) or not isinstance(
            doc.get("plans"), list
        ):
            raise ValueError("Malformed local Jobs registry")
        return deepcopy(doc)

    def _save(self, series_root, series_id, mutation):
        path = self._file(series_root, series_id)
        before = path.read_bytes() if path.exists() else None
        doc = self.context(series_root, series_id)
        mutation(doc)
        data = (
            json.dumps(doc, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            "." + path.name + "." + uuid.uuid4().hex + ".partial"
        )
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        current = path.read_bytes() if path.exists() else None
        if current != before:
            raise RuntimeError(
                "Local Jobs registry changed; registration not replaced. Pending bytes: "
                + str(temporary)
            )
        os.replace(temporary, path)
        return self.context(series_root, series_id)

    def register_job(
        self, series_root, series_id, job_root, *, label="", activate=True
    ):
        require_unc(str(job_root))
        root = Path(job_root)
        raw, bound = _asset(root / "job.json")
        doc = read_json(root / "job.json")
        if descriptor(root / "job.json") != bound:
            raise ValueError("Selected job metadata changed while parsed")
        runs = doc.get("expected_runs")
        if (
            not isinstance(runs, list)
            or not runs
            or len({name(run) for run in runs}) != len(runs)
        ):
            raise ValueError(
                "An existing job with exact distinct expected_runs is required"
            )
        if doc.get("pipeline") == "preset-wasserstein":
            from .preset_wasserstein_protocol import validate_preset_wasserstein_job
            validate_preset_wasserstein_job(doc)
        elif doc.get("precision") not in {"float32", "float64"}:
            raise ValueError("Existing job precision is not explicit")
        row = dict(
            root=str(root),
            name=str(label).strip() or root.name,
            series_label=str(doc.get("series_id") or "Not recorded"),
            job=bound,
        )

        def change(state):
            prior = next(
                (item for item in state["jobs"] if _path(item["root"]) == _path(root)),
                None,
            )
            if prior is not None and prior["job"] != bound:
                raise ValueError("The registered immutable job metadata changed")
            if prior is None:
                state["jobs"].append(row)
            if activate:
                state["active_job"] = str(root)

        return self._save(series_root, series_id, change)

    def register_reprepared_setup(self, series_root, series_id, job_root, *, previous_job, receipt):
        """Advance one explicit Setup association using its retained transition receipt."""
        root, path = Path(job_root), Path(receipt["path"])
        if (not path.resolve().is_relative_to(root.resolve() / "history")
                or path.name != "receipt.json" or descriptor(path) != receipt):
            raise ValueError("Reprepare registration requires its job-owned history receipt")
        document = read_json(path)
        current = descriptor(root / "job.json")
        retained = descriptor(path.parent / "previous" / "job.json")
        old, new = read_json(path.parent / "previous" / "job.json"), read_json(root / "job.json")
        if (document.get("format") != "darsia-setup-job-reprepare-v1"
                or document.get("complete") is not True or document.get("previous_job") != previous_job
                or document.get("job") != current
                or new.get("pipeline") != "series-setup"
                or old.get("commit") == new.get("commit") and (
                    document.get("failed_preparation", {}).get("format") != "darsia-setup-failed-before-apply-v1"
                    and not document.get("plan_refresh") and not document.get("image_preparation_refresh"))
                or any(retained[key] != previous_job[key] for key in ("sha256", "size_bytes"))):
            raise ValueError("Reprepare registration differs from the retained original Setup job")
        from .setup_job_reprepare import validate_setup_reprepare_transition
        validate_setup_reprepare_transition(old, new, document, history_root=path.parent)
        def change(state):
            row = next((row for row in state["jobs"] if _path(row["root"]) == _path(root)), None)
            if row is None or row["job"] != previous_job:
                raise ValueError("The selected Setup registration changed during reprepare")
            row["job"] = current
            state["active_job"] = str(root)
            state.setdefault("history", []).append(dict(action="reprepare-unstarted-setup",
                root=str(root), previous_job=previous_job, job=current, receipt=receipt,
                actor=document["actor"], utc=document["requested_utc"]))
        return self._save(series_root, series_id, change)

    def register_plan(self, series_root, series_id, path):
        require_unc(str(path))
        raw, bound = _asset(path)
        doc = tomllib.loads(raw.decode("utf-8-sig"))
        if (
            doc.get("format") != "darsia-series-v1"
            or doc.get("adopted_for_execution") is not True
        ):
            raise ValueError(
                "Choose an already adopted darsia-series-v1 execution plan"
            )
        if doc.get("job", {}).get("precision") != "float32":
            raise ValueError("New production execution plans require float32")
        row = dict(bound, label=str(doc.get("job_id") or Path(path).stem))

        def change(state):
            prior = next(
                (item for item in state["plans"] if _path(item["path"]) == _path(path)),
                None,
            )
            if (
                prior is not None
                and {key: prior[key] for key in ("path", "sha256", "size_bytes")}
                != bound
            ):
                raise ValueError("The registered immutable execution plan changed")
            if prior is None:
                state["plans"].append(row)

        return self._save(series_root, series_id, change)

    def set_active(self, series_root, series_id, job_root):
        def change(state):
            if not any(_path(row["root"]) == _path(job_root) for row in state["jobs"]):
                raise ValueError("Select an explicitly registered job")
            state["active_job"] = str(job_root)

        return self._save(series_root, series_id, change)

    @staticmethod
    def _folder_removed(series_root, job_root):
        """A registered job folder is gone while its series folder is reachable."""
        return Path(series_root).is_dir() and not Path(job_root).exists()

    def missing_jobs(self, series_root, series_id):
        """Registered job folders that were moved or deleted outside DarSIA."""
        state = self.context(series_root, series_id)
        return [row["root"] for row in state["jobs"]
                if self._folder_removed(series_root, row["root"])]

    def jobs(self, series_root, series_id, *, include_archived=False):
        state = self.context(series_root, series_id)
        result = []
        for row in state["jobs"]:
            if row.get("archived") and not include_archived:
                continue
            require_unc(row["root"])
            if self._folder_removed(series_root, row["root"]):
                continue  # Reported by missing_jobs(); it must not hide the other jobs.
            if descriptor(Path(row["root"]) / "job.json") != row["job"]:
                raise ValueError("Registered job metadata SHA changed: " + row["root"])
            result.append(
                dict(
                    row,
                    name=("Archived · " + row["name"]) if row.get("archived") else row["name"],
                    active=_path(row["root"]) == _path(state.get("active_job") or ""),
                )
            )
        return result

    def set_archived(self, series_root, series_id, job_root, archived, *, actor):
        if not str(actor).strip() or type(archived) is not bool:
            raise ValueError("An action name and explicit archive state are required")
        def change(state):
            row = next((row for row in state["jobs"] if _path(row["root"]) == _path(job_root)), None)
            if row is None:
                raise ValueError("Select a registered job")
            row["archived"] = archived
            state.setdefault("history", []).append(dict(action="archive" if archived else "restore",
                root=row["root"], job=row["job"], actor=str(actor).strip(),
                reason="Archived selected job" if archived else "Restored archived job",
                utc=datetime.now(timezone.utc).isoformat()))
            if archived and _path(state.get("active_job") or "") == _path(job_root):
                state["active_job"] = next((item["root"] for item in state["jobs"] if not item.get("archived")), None)
        return self._save(series_root, series_id, change)

    def remove_job(self, series_root, series_id, job_root, *, actor, deletion_path):
        if not str(actor).strip():
            raise ValueError("An action name is required")
        def change(state):
            row = next((row for row in state["jobs"] if _path(row["root"]) == _path(job_root)), None)
            if row is None:
                raise ValueError("Select a registered job")
            state["jobs"].remove(row)
            state.setdefault("history", []).append(dict(action="delete", root=row["root"], job=row["job"],
                actor=str(actor).strip(), reason="Deleted selected inactive job folder", deletion_path=str(deletion_path),
                utc=datetime.now(timezone.utc).isoformat()))
            if _path(state.get("active_job") or "") == _path(job_root):
                state["active_job"] = next((item["root"] for item in state["jobs"] if not item.get("archived")), None)
        return self._save(series_root, series_id, change)

    def plans(self, series_root, series_id):
        result = self.context(series_root, series_id)["plans"]
        for row in result:
            require_unc(row["path"])
            if descriptor(Path(row["path"])) != {
                key: row[key] for key in ("path", "sha256", "size_bytes")
            }:
                raise ValueError(
                    "Registered execution plan SHA changed: " + row["path"]
                )
        return result

    def known_jobs(self):
        """All explicit local registrations, without discovering NAS folders.

        The consumer revalidates each job's own immutable descriptor/requests.
        A malformed local record is an error, never evidence of a dead worker.
        """
        result = {}
        for path in sorted(self.root.glob("*.json")):
            raw, bound = _asset(path)
            doc = json.loads(raw)
            if (
                doc.get("format") != "darsia-native-jobs-registry-v1"
                or not isinstance(doc.get("series_root"), str)
                or not isinstance(doc.get("series_id"), str)
                or self._file(doc["series_root"], doc["series_id"]) != path
                or not isinstance(doc.get("jobs"), list)
            ):
                raise ValueError("Malformed local Jobs registration: " + str(path))
            for row in doc["jobs"]:
                require_unc(row["root"])
                if not isinstance(row.get("job"), dict):
                    raise ValueError("Registered job lacks its metadata binding")
                if self._folder_removed(doc["series_root"], row["root"]):
                    continue  # A moved or deleted folder can neither run nor use another job.
                key = _path(row["root"])
                if key in result and result[key]["job"] != row["job"]:
                    raise ValueError(
                        "Conflicting local registrations for job: " + row["root"]
                    )
                result[key] = deepcopy(row)
            if descriptor(path) != bound:
                raise ValueError(
                    "Local Jobs registration changed while read: " + str(path)
                )
        return list(result.values())
