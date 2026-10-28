"""Exact reviewed three-flag exception, with all remaining writer AST retained."""
import hashlib
import pytest
from darsia.fluidflower.pipeline import capacity as cap

OLD_METADATA = 'def _storage_facts(output):\n    """Report the actual admitted output location; never an assumed NAS copy."""\n    from ..pipeline.run_io import current_binding\n    binding = current_binding()\n    if binding is None:\n        return dict(local_retained=False,direct_NAS_write=True)\n    binding.require_output(output)\n    return dict(local_retained=True,direct_NAS_write=False,direct_local_write=True)'
NEW_METADATA = 'def _storage_facts(output):\n    """Report the actual admitted output location; never an assumed NAS copy."""\n    from ..pipeline.run_io import current_binding\n    binding = current_binding()\n    if binding is None:\n        return dict(local_retained=False,direct_NAS_write=True)\n    binding.require_output(output)\n    from ..job_filesystem import LocalFileAdmission, ResultsRootAdmission, valid_results_admission\n    admitted = binding.local_output\n    if not valid_results_admission(admitted.admission,admitted.results_root):\n        raise ValueError("Storage metadata requires its positive actual results-root admission")\n    local = (isinstance(admitted.admission,LocalFileAdmission)\n             or isinstance(admitted.admission,ResultsRootAdmission) and admitted.admission.local is True)\n    return dict(local_retained=local,direct_NAS_write=not local,direct_local_write=local)'


def writer_fixture(tmp_path, monkeypatch, *, current_extra="return 1", current_function=NEW_METADATA):
    old = (OLD_METADATA + "\n\ndef actual_array_writer(): return 1\n").encode()
    current = (current_function + "\n\ndef actual_array_writer(): " + current_extra + "\n").encode()
    relative = "src/darsia/fluidflower/mass/n1_pipeline.py"
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    target.write_bytes(current)
    sources = {"a" * 40: old, "b" * 40: current}
    def git(args):
        commit, path = args[-1].split(":", 1)
        assert path == relative and commit in sources
        return sources[commit]
    monkeypatch.setattr(cap.subprocess, "check_output", git)
    row = dict(relative_path=relative, sha256=hashlib.sha256(current).hexdigest(),
               measured_sha256=hashlib.sha256(old).hexdigest())
    return row, sources


def test_exact_flags_fix_retains_measured_native_writer_applicability(tmp_path, monkeypatch):
    row, _ = writer_fixture(tmp_path, monkeypatch)
    cap._profile_writer_identity(tmp_path, "a" * 40, "b" * 40, row)
    assert set(row) == {"relative_path", "sha256", "measured_sha256"}


def test_actual_array_writer_change_requires_a_new_measurement(tmp_path, monkeypatch):
    row, _ = writer_fixture(tmp_path, monkeypatch, current_extra="return 2")
    with pytest.raises(ValueError, match="actual writer/executor body changed"):
        cap._profile_writer_identity(tmp_path, "a" * 40, "b" * 40, row)


def test_other_metadata_function_is_not_a_general_compatibility_exemption(tmp_path, monkeypatch):
    row, _ = writer_fixture(tmp_path, monkeypatch, current_function=NEW_METADATA.replace("direct_NAS_write=not local", "direct_NAS_write=False"))
    with pytest.raises(ValueError, match="not the exact reviewed"):
        cap._profile_writer_identity(tmp_path, "a" * 40, "b" * 40, row)


def test_whole_file_identity_remains_required_without_explicit_measured_provenance(tmp_path, monkeypatch):
    row, _ = writer_fixture(tmp_path, monkeypatch)
    row.pop("measured_sha256")
    with pytest.raises(ValueError, match="Measured/current actual writer differs"):
        cap._profile_writer_identity(tmp_path, "a" * 40, "b" * 40, row)


@pytest.mark.parametrize("change", ["other_writer", "wrong_measured_sha", "nonstring_sha"])
def test_exception_rejects_other_file_or_wrong_provenance(tmp_path, monkeypatch, change):
    row, _ = writer_fixture(tmp_path, monkeypatch)
    if change == "other_writer": row["relative_path"] = "src/darsia/fluidflower/native/sidecar.py"
    elif change == "wrong_measured_sha": row["measured_sha256"] = "0" * 64
    else: row["measured_sha256"] = None
    with pytest.raises(ValueError):
        cap._profile_writer_identity(tmp_path, "a" * 40, "b" * 40, row)


def test_unchanged_writer_keeps_the_original_two_field_profile_contract(tmp_path, monkeypatch):
    relative = "src/darsia/fluidflower/native/sidecar.py"
    raw = b"def actual_array_writer(): return 1\n"
    path = tmp_path / relative
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    monkeypatch.setattr(cap.subprocess, "check_output", lambda args: raw)
    row = dict(relative_path=relative, sha256=hashlib.sha256(raw).hexdigest())
    cap._profile_writer_identity(tmp_path, "a" * 40, "b" * 40, row)
