"""Pure cap-policy/comparison fixtures. No VHDX, disk APIs or host installation."""
import copy
from pathlib import PureWindowsPath
from types import SimpleNamespace
import pytest
from darsia_cache.policy import CachePolicyError, cache_settings
from darsia_cache.windows import _match_volume_cap


def identity(serial, byte, directory):
    return dict(volume_serial=serial,file_id_128=byte*32,directory=directory,creation_100ns=123456789)


def document():
    return {"cache":dict(root="Q:/DarSIA-owned-cache",quota_bytes=1024**3,
        profiles={"tests":dict(path="C:/TESTONLY/profile.json",sha256="a"*64,size_bytes=1)},
        volume_cap=dict(format="darsia-fixed-vhdx-cap-binding-v1",
            volume_guid="\\\\?\\Volume{11111111-1111-1111-1111-111111111111}\\",
            virtual_disk_identifier="22222222-2222-2222-2222-222222222222",maximum_size_bytes=1024**3,
            image_path="C:/TESTONLY_VOLUME/image.vhdx",image_parent_identity=identity(1,"1",True),
            image_identity=identity(1,"2",False),cache_identity=identity(2,"3",True)))}


def observed(binding):
    return dict(volume_guid=binding.volume_guid,virtual_disk_identifier=binding.virtual_disk_identifier,
        virtual_size_bytes=binding.maximum_size_bytes,storage_device_id=3,provider_subtype=2,is_loaded=True,
        filesystem="NTFS",extent_count=1,volume_disk_number=7,vhdx_disk_number=7,
        extent_start=1024**2,extent_bytes=binding.maximum_size_bytes-2*1024**2,
        volume_total_bytes=binding.maximum_size_bytes-3*1024**2)


def test_exact_typed_cap_and_logical_only_are_distinct():
    doc=document(); binding=cache_settings(doc).volume_cap
    expected=copy.deepcopy(doc["cache"]["volume_cap"])
    expected["image_path"]=str(PureWindowsPath(expected["image_path"]))
    assert binding.document()==expected
    assert _match_volume_cap(binding,observed(binding))["verified"] is True
    del doc["cache"]["volume_cap"]
    assert cache_settings(doc).volume_cap is None


@pytest.mark.parametrize("key,value",[
    ("volume_guid","Q:/"),("virtual_disk_identifier","0"*36),("maximum_size_bytes",512),("maximum_size_bytes",2*1024**3),
    ("image_path","Q:/self/image.vhdx"),("format","a claimed success"),
    ("cache_identity",identity(1,"3",True)),("image_identity",identity(1,"0",False)),
])
def test_cap_policy_refuses_unbound_aliases_or_size(key,value):
    doc=document();doc["cache"]["volume_cap"][key]=value
    with pytest.raises(CachePolicyError):cache_settings(doc)


@pytest.mark.parametrize("key,value",[
    ("volume_guid","\\\\?\\Volume{33333333-3333-3333-3333-333333333333}\\"),
    ("virtual_disk_identifier","33333333-3333-3333-3333-333333333333"),
    ("virtual_size_bytes",2*1024**3),("storage_device_id",2),("provider_subtype",3),
    ("is_loaded",False),("filesystem","ReFS"),("extent_count",2),("volume_disk_number",8),
    ("extent_start",-1),("extent_bytes",2*1024**3),("volume_total_bytes",2*1024**3),
])
def test_actual_wrong_or_missing_cap_refuses(key,value):
    binding=cache_settings(document()).volume_cap
    value_before=observed(binding);value_before[key]=value
    with pytest.raises(CachePolicyError):_match_volume_cap(binding,value_before)


def test_matching_query_is_not_complete_writer_coverage_or_user_quota():
    binding=cache_settings(document()).volume_cap
    result=_match_volume_cap(binding,observed(binding))
    assert "not a per-user quota" in result["semantics"]
    assert result["binding"]==binding.document()
    assert "complete writer-coverage proof" in result["semantics"]


def test_managed_lease_missing_cap_refuses_before_any_owner_or_profile_probe(monkeypatch):
    from darsia_cache import bootstrap
    doc=document();del doc["cache"]["volume_cap"]
    def forbidden(*args,**kwargs):
        pytest.fail("No owner/profile/runtime/native probing before missing-cap refusal")
    monkeypatch.setattr(bootstrap,"_owner",forbidden)
    monkeypatch.setattr(bootstrap,"_exact",forbidden)
    with pytest.raises(CachePolicyError,match="fixed-volume write-cap binding"):
        bootstrap.ImportLease(doc,{},"TESTONLY_NOT_READ",entry_kind="worker")


def test_real_cap_refusal_precedes_any_host_binding_write_or_reservation(monkeypatch):
    from darsia_cache import control,capacity
    from contextlib import contextmanager
    policy=cache_settings(document());calls=[]
    admitted=dict(host="TESTONLY",authorized_sids=["TESTONLY"],
        cache_identity=policy.volume_cap.cache_identity,volume_cap=policy.volume_cap.document())
    class Namespace_TESTONLY:
        def __init__(self,path,identity):
            self.path=path;calls.append(("open",path))
        def verify_volume_cap(self,binding):
            assert binding is policy.volume_cap
            calls.append(("native-cap-refusal",self.path))
            raise CachePolicyError("TESTONLY native getter refuses")
        def names(self):
            pytest.fail("Host pool was examined before volume-cap refusal")
        def write_once(self,*args):
            pytest.fail("Host binding must not be written before actual cap admission")
        def close(self):calls.append(("close",self.path))
    @contextmanager
    def Lock_TESTONLY(*args):yield
    monkeypatch.setattr(control,"WindowsNamespace",Namespace_TESTONLY)
    monkeypatch.setattr(capacity,"_LocalVolumeLock",Lock_TESTONLY)
    ledger=control.CacheBudgetLedger.__new__(control.CacheBudgetLedger)
    ledger.policy=policy;ledger.admission=admitted
    ledger.capacity=SimpleNamespace(host="TESTONLY")
    ledger.bound=dict(path="C:/TESTONLY_CONTROL",identity={})
    ledger.key="TESTONLY";ledger.lock_active=False;ledger.control=None
    with pytest.raises(CachePolicyError,match="TESTONLY native getter refuses"):
        with ledger.locked():pytest.fail("Refused cap cannot enter a cache transaction")
    assert calls==[("open","C:/TESTONLY_CONTROL"),("open",policy.root),
        ("native-cap-refusal",policy.root),("close",policy.root),("close","C:/TESTONLY_CONTROL")]
    assert ledger.lock_active is False and ledger.control is None
    assert ledger.volume_cap_observation is None
