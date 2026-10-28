"""Build and open ranked annotation checks from the active review window."""

from darsia.fluidflower.series_paths import tool_arguments, local_cache_root
from datetime import UTC, datetime
import json
from pathlib import Path
import uuid

from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QPushButton

from darsia.fluidflower.phase_evidence_audit import load_queue, queue_directory
from darsia.fluidflower.phase_workflow import default_phase_review_roots, phase_python_executable
from .evidence_audit_library import EvidenceAuditLibrary


def review_model_root(workspace):
    """Resolve the source model of a normal or composite frozen review."""
    root = Path(workspace.combined_root)
    receipt = root / 'collection_receipt.json'
    if receipt.is_file():
        value = json.loads(receipt.read_text(encoding='utf-8-sig'))
        identity, path = value.get('source_model_id'), value.get('source_model_root')
    else:
        value = json.loads((root/'validation_binding.json').read_text(encoding='utf-8-sig'))
        identity, path = value.get('model_id'), value.get('model_root')
    if not path or identity != workspace.target_model_id:
        raise ValueError('The review binding does not identify the selected model')
    return Path(path), root
