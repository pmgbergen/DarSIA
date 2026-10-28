"""Conservative local throughput tuning between original whole-run batches."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import math
import socket


_GIB = 1024 ** 3


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _epoch(value):
    result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("A measured batch needs an aware timestamp")
    return result.timestamp()


def _inventory():
    from .execution import detect_host_resources, _nvidia_devices

    result = detect_host_resources()
    result["accelerators"] = _nvidia_devices()
    if any(device.get("runtime_ready") for device in result["accelerators"]):
        try:
            import cupy
            count = cupy.cuda.runtime.getDeviceCount()
        except Exception:
            count = 0
        for device in result["accelerators"]:
            device["runtime_ready"] = device.get("runtime_ready") is True and count > int(device["device_id"])
    return result


class AutomaticWorkerCapacity:
    """Try one neighboring CPU/GPU allocation, retaining measured improvements.

    Samples are local completed batches from SharedJobService, never aggregate
    job progress. Different immutable jobs are never pooled. Memory increases
    require the existing advisor's matching measured RSS/VRAM evidence.
    """

    def __init__(self, inventory=None, *, machine=None, history=None,
                 reserve_bytes=40 * _GIB, now=None):
        from .device_choice import backend_devices

        self.inventory = deepcopy(inventory if inventory is not None else _inventory())
        self.machine = machine or socket.gethostname()
        self.history = deepcopy(history or [])
        self.reserve_bytes = reserve_bytes
        if type(reserve_bytes) is not int or reserve_bytes < 0:
            raise ValueError("The RAM reserve must be a nonnegative byte count")
        self._now = now or (lambda: datetime.now(timezone.utc).timestamp())
        devices = backend_devices(self.inventory, "cuda-float32")
        self._device = next((device for device in devices if device.get("runtime_ready") is True), None)
        self._initial = (1, int(self._device is not None))
        available = (self.inventory.get("memory") or {}).get("available_bytes")
        if type(available) is int and available <= 0:
            self._initial = (0, 0)
        self._counts = self._initial
        self._seen = set()
        self._key = None
        self._best = None
        self._scores = {}
        self._pending = {}
        self.note = "Automatic: waiting for completed local batches; starts with one worker per available backend."
        if self._counts == (0, 0):
            self.note = "Automatic: no available RAM was reported; free memory before starting workers."

    def counts(self):
        return self._counts

    def _sample(self, sample, snapshot):
        if not isinstance(sample, dict) or not isinstance(sample.get("sample_id"), str):
            return None
        if sample["sample_id"] in self._seen or sample.get("error") or sample.get("stale"):
            return None
        machine = sample.get("machine", snapshot.get("machine"))
        if not isinstance(machine, str) or machine.casefold() != self.machine.casefold():
            return None
        backend = sample.get("backend")
        workers = sample.get("workers")
        if backend not in {"cpu", "cuda"} or type(workers) is not int or not 1 <= workers <= 256:
            return None
        if not _positive(sample.get("completed_frames")) or not _positive(sample.get("elapsed_seconds")):
            return None
        facts = sample.get("job_facts") or {}
        if not all(isinstance(facts.get(field), str) and facts[field] for field in ("pipeline", "precision", "commit")):
            return None
        workload = sample.get("workload_key")
        if not isinstance(workload, str) or not workload:
            return None
        try:
            end = _epoch(sample["completed_utc"])
            start = _epoch(sample["started_utc"])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        age = self._now() - end
        if end <= start or age < -30 or age > 600:
            return None
        configured = sample.get("configured_counts") or {}
        pair = (configured.get("cpu"), configured.get("cuda"))
        if any(type(count) is not int or not 0 <= count <= 256 for count in pair):
            return None
        if pair[0 if backend == "cpu" else 1] != workers:
            return None
        key = (workload, facts["pipeline"], facts["precision"], facts["commit"], str(sample.get("size_class", "immutable-job")))
        return dict(sample, key=key, pair=pair, start=start, end=end)

    def _remember_memory(self, sample):
        memory = sample.get("memory_sample")
        if not isinstance(memory, dict):
            if not isinstance(sample.get("source"), dict):
                return
            memory = dict(source=sample["source"], threads=sample.get("threads"),
                          peak_worker_rss_bytes=sample.get("peak_worker_rss_bytes"),
                          peak_device_memory_bytes=sample.get("peak_device_memory_bytes"),
                          device_id=sample.get("device_id", ""))
        memory = dict(memory, machine=self.machine, backend=sample["backend"],
                      pipeline=sample["job_facts"]["pipeline"], precision=sample["job_facts"]["precision"],
                      git_commit=sample["job_facts"]["commit"], tuning_key=sample["key"])
        device_id = str(memory.get("device_id", ""))
        if self._device is not None and device_id in {
            str(self._device.get("device_id", "")), str(self._device.get("stable_id", ""))
        }:
            memory["device_id"] = str(self._device["device_id"])
        self.history.append(memory)
        self.history = self.history[-256:]
        available = sample.get("available_bytes")
        if type(available) is int and available >= 0:
            self.inventory.setdefault("memory", {})["available_bytes"] = available
        free_device = sample.get("available_device_bytes")
        if (self._device is not None and str(memory.get("device_id", "")) == str(self._device["device_id"])
                and type(free_device) is int and free_device >= 0):
            self._device["available_bytes"] = free_device
            for device in self.inventory.get("accelerators", []):
                if str(device.get("device_id")) == str(self._device.get("device_id")):
                    device["available_bytes"] = free_device

    def _budget(self, facts):
        from .resource_advisor_panel import resource_advice

        cores = self.inventory.get("cpu", {}).get("physical_cores") or self.inventory.get("cpu", {}).get("logical_cores")
        core_limit = max(1, cores - 1) if type(cores) is int and cores > 0 else 1
        limits, memories, threads = [], [], []
        for backend in ("cpu", "cuda"):
            matching = [row for row in self.history if row.get("machine") == self.machine
                        and tuple(row.get("tuning_key", ())) == self._key
                        and row.get("backend") == backend and row.get("pipeline") == facts["pipeline"]
                        and row.get("precision") == facts["precision"] and row.get("git_commit") == facts["commit"]
                        and type(row.get("threads")) is int and row["threads"] > 0]
            per_worker_threads = max([row["threads"] for row in matching], default=1)
            matching = [row for row in matching if row["threads"] == per_worker_threads]
            machine = dict(machine=self.machine, inventory=self.inventory, history=matching,
                           device_id=str((self._device or {}).get("device_id", "")))
            advice = resource_advice(machine, facts, backend, per_worker_threads,
                                     reserve_bytes=self.reserve_bytes)
            limit = advice.get("workers")
            limits.append(min(256, core_limit, limit) if type(limit) is int else self._initial[0 if backend == "cpu" else 1])
            rss = [row["peak_worker_rss_bytes"] for row in matching if type(row.get("peak_worker_rss_bytes")) is int
                   and row["peak_worker_rss_bytes"] > 0 and isinstance(row.get("source"), dict)]
            memories.append(max(rss) if rss and type(limit) is int else None)
            threads.append(per_worker_threads)
        available = self.inventory.get("memory", {}).get("available_bytes")
        budget = max(0, available - self.reserve_bytes) if type(available) is int else None
        return limits, memories, threads, budget, core_limit

    def _safe(self, pair, budget):
        limits, memory, threads, available, cores = budget
        if sum(pair) == 0 or any(count < 0 or count > limit for count, limit in zip(pair, limits)):
            return False
        if sum(count * thread for count, thread in zip(pair, threads)) > cores:
            return False
        # Unknown memory may keep the initial conservative worker, but never
        # consume extra RAM next to an independently increased backend.
        if any(count and rss is None for count, rss in zip(pair, memory)):
            return all(count <= initial for count, initial in zip(pair, self._initial))
        return available is not None and sum(count * rss for count, rss in zip(pair, memory) if count) <= available

    def _choose(self, score, facts, allowed):
        pair = self._counts
        self._scores[pair] = score
        if self._best is None or score > self._best[1] * 1.05:
            self._best = (pair, score)
        best, best_score = self._best
        budget = self._budget(facts)
        if not self._safe(best, budget):
            smaller = [(min(best[0], 1), 0), (0, min(best[1], 1))]
            self._counts = next((item for item in smaller if self._safe(item, budget)), (0, 0))
            self.note = "Automatic: measured memory limits reduced the next worker batch."
            return
        candidates = [(best[0] + 1, best[1]), (best[0], best[1] + 1),
                      (best[0] - 1, best[1]), (best[0], best[1] - 1)]
        for candidate in candidates:
            if any(candidate[index] > 0 and backend not in allowed for index, backend in enumerate(("cpu", "cuda"))):
                continue
            if candidate not in self._scores and self._safe(candidate, budget):
                self._counts = candidate
                self.note = (f"Automatic: measuring CPU {candidate[0]} + GPU {candidate[1]}; "
                             f"best completed throughput {best_score:.3g} frames/s. Counts change at batch boundaries.")
                return
        self._counts = best
        missing = any(value is None for value in budget[1])
        self.note = (f"Automatic: retaining CPU {best[0]} + GPU {best[1]} at {best_score:.3g} measured frames/s. "
                     + ("More workers need matching RAM/VRAM measurements." if missing else "No better safe neighboring allocation measured."))

    def observe(self, snapshot):
        if not isinstance(snapshot, dict) or snapshot.get("errors"):
            return
        for raw in snapshot.get("completed_batches", snapshot.get("samples", [])):
            sample = self._sample(raw, snapshot)
            if sample is None:
                continue
            self._seen.add(sample["sample_id"])
            self._remember_memory(sample)
            if self._key != sample["key"]:
                self._key, self._best, self._scores, self._pending = sample["key"], None, {}, {}
                self._counts = self._initial
            if sample["pair"] != self._counts:
                continue
            from .shared_job_capabilities import job_capabilities
            capabilities = job_capabilities(sample["job_facts"])
            allowed = set(capabilities["supported_backends"])
            eligible = sample.get("eligible_backends")
            if isinstance(eligible, list) and all(value in {"cpu", "cuda"} for value in eligible):
                allowed.intersection_update(eligible)
            expected = {backend for index, backend in enumerate(("cpu", "cuda"))
                        if self._counts[index] > 0 and backend in allowed}
            self._pending.setdefault(sample["backend"], []).append(sample)
            if not expected or not expected.issubset(self._pending):
                continue
            records = [row for backend in expected for row in self._pending[backend]]
            duration = max(row["end"] for row in records) - min(row["start"] for row in records)
            if duration <= 0:
                continue
            score = sum(row["completed_frames"] for row in records) / duration
            self._pending = {}
            self._choose(score, sample["job_facts"], allowed)
