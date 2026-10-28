"""Keep the original first-qualification entry refusal before any source read."""

import json

import pytest

from darsia_cache import bootstrap
from darsia_cache.first_qualification import REQUEST_ENV
from darsia_cache.policy import CachePolicyError


@pytest.mark.parametrize("entry_kind", ("runtime", "tests"))
def test_first_qualification_keeps_original_entry_refusal_before_metadata_read(monkeypatch, entry_kind):
    """Reject both non-worker entries with the original text and no host access."""
    pin = dict(path="C:/synthetic/profile.json", sha256="a" * 64, size_bytes=1)
    purposes = dict(display="display", jobs="jobs", tests="tests")
    document = dict(
        cache=dict(root="C:/synthetic/cache", quota_bytes=1024,
                   profiles={purpose: dict(pin) for purpose in purposes.values()}),
        cache_bootstrap=dict(
            entry_roles=dict(runtime="display", worker="jobs", tests="tests"),
            roles={role: dict(purpose=purpose, profile=dict(pin))
                   for role, purpose in purposes.items()},
            namespace_admission=dict(pin), accounting_entry_limit=1))
    monkeypatch.setenv(REQUEST_ENV, json.dumps(pin))

    def unexpected_read(*args, **kwargs):
        """Make a premature descriptor, owner or native lookup visible."""
        pytest.fail("The original entry refusal must precede metadata and host access")

    monkeypatch.setattr(bootstrap, "_exact", unexpected_read)
    monkeypatch.setattr(bootstrap, "_owner", unexpected_read)
    with pytest.raises(CachePolicyError) as error:
        bootstrap.ImportLease(document, {}, "synthetic-package", entry_kind=entry_kind)
    assert str(error.value) == "Finite first qualification requires the original worker/jobs entry"
