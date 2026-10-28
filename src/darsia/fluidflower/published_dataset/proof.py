"""Proof that a published revision gives the results of the revision that made them.

A record publishes the exact inputs and code commit of the job that made its
expected results, or a clean revision of them. A revision needs a proof
record (format ``darsia-published-revision-proof-v1``): an exact replay of a
sample of frames on the reference machine, bit-identical to the expected
results. The record binds both sides by commit, plan SHA-256 and a digest of
the published input list::

    {"format": "darsia-published-revision-proof-v1",
     "expected": {"commit": "<40 hex>", "plan_sha256": "<64 hex>"},
     "published": {"commit": "<40 hex>", "plan_sha256": "<64 hex>",
                   "inputs_sha256": "<64 hex>"},
     "machine": "reference machine",
     "frames": [{"run": "<run>", "case_id": "<case>", "verdict": "IDENTICAL"}],
     "verdict": "IDENTICAL"}
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROOF_FORMAT = "darsia-published-revision-proof-v1"


def inputs_digest(rows) -> str:
    """A SHA-256 over an input list: one ``<root alias>/<path>\\t<sha256>`` line per input."""
    lines = sorted(f"{alias}/{relative}\t{sha256}\n" for alias, relative, sha256 in rows)
    return hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()


def load_proof(path: str | Path) -> dict:
    """Read a proof record."""
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def proof_problems(proof: dict, *, expected_commit: str, expected_plan_sha256: str,
                   published_commit: str, published_plan_sha256: str,
                   published_inputs_sha256: str) -> list[str]:
    """Why a proof record does not prove this published revision; empty when it does."""
    if not isinstance(proof, dict) or proof.get("format") != PROOF_FORMAT:
        return [f"The proof must be a {PROOF_FORMAT} record"]
    problems = []
    expected, published = proof.get("expected") or {}, proof.get("published") or {}
    if expected.get("commit") != expected_commit:
        problems.append("The proof's expected commit is not the one that made the results")
    if expected.get("plan_sha256") != expected_plan_sha256:
        problems.append("The proof's expected plan is not the job's plan")
    if published.get("commit") != published_commit:
        problems.append("The proof's published commit is not the published commit")
    if published.get("plan_sha256") != published_plan_sha256:
        problems.append("The proof's published plan is not the published plan")
    if published.get("inputs_sha256") != published_inputs_sha256:
        problems.append("The proof covers another set of published inputs")
    if proof.get("machine") != "reference machine":
        problems.append("The proof must come from the reference machine")
    frames = proof.get("frames")
    if not isinstance(frames, list) or not frames:
        problems.append("The proof compares no frames")
    elif any(not isinstance(row, dict) or row.get("verdict") != "IDENTICAL" for row in frames):
        problems.append("A frame of the proof is not bit-identical")
    if proof.get("verdict") != "IDENTICAL":
        problems.append("The proof's verdict is not IDENTICAL")
    return problems
