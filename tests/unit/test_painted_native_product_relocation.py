"""Moved Store publications keep exact byte identities during native planning."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
import pytest,toml
from darsia.fluidflower import painted_native_product as product, geometry_facies


def pin(path):
    raw=path.read_bytes()
    return dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),size_bytes=len(raw))


@pytest.mark.parametrize('change',['none','artifact','hash'])
def test_native_product_follows_only_explicit_moved_store_paths(tmp_path,monkeypatch,change):
    old=tmp_path/'before';new=tmp_path/'after';old.mkdir()
    folder=old/'annotations/map_adoption/prepared';folder.mkdir(parents=True)
    model=folder/'model.json';model.write_text('{"baseline_native_grid":{}}')
    labels=folder/'one/labels.npz';labels.parent.mkdir();labels.write_bytes(b'planning never opens this array')
    source=pin(labels);source['path']='one/labels.npz'
    prepared=dict(format='darsia-fluidflower-painted-map-adoption-preparation-v2',
        runs=[dict(run_id='one',status='prepared',labels=source,target_model=pin(model))])
    prepfile=folder/'index.json';prepfile.write_text(json.dumps(prepared))
    revision=dict(revision_id='a'*32,configuration=dict(painted_adoption_table=dict(preparation=pin(prepfile)),
        provenance=dict(local_map_reuse={})))
    stored=old/'results/geometry'/revision['revision_id']
    stored.mkdir(parents=True);revfile=stored/'revision.json';revfile.write_text(json.dumps(revision))
    stored_labels=stored/'runs/one/labels.npz';stored_labels.parent.mkdir(parents=True)
    stored_labels.write_bytes(labels.read_bytes())
    registered=dict(files=dict(labels=dict(pin(stored_labels))))
    if change=='hash':registered['files']['labels']['sha256']='0'*64
    artifact=stored_labels if change!='artifact' else old/'unrelated/labels.npz'
    monkeypatch.setattr(geometry_facies,'resolve_run_geometry_artifacts',
        lambda *a,**k:dict(scope='run_override',labels_path=str(artifact)))
    before={p.relative_to(old):p.read_bytes() for p in (model,labels,prepfile,revfile,stored_labels)}
    old.rename(new)
    registry=tmp_path/'registry.toml'
    registry.write_text(toml.dumps(dict(schema_version=1,storage=dict(path_aliases={str(old):str(new)}))))
    monkeypatch.setenv('DARSIA_SERIES_REGISTRY',str(registry))
    original=Path.open
    def metadata_only(path,*a,**k):
        if path.suffix=='.npz':raise AssertionError('Planning must not open NPZ payloads')
        return original(path,*a,**k)
    with monkeypatch.context() as patch:
        patch.setattr(Path,'open',metadata_only)
        if change=='none':
            value=product.adopted_run_native_product(revision,'one',project_root=new,registered_sources=registered)
            assert value['source_labels']['path']==str(new/'annotations/map_adoption/prepared/one/labels.npz')
        else:
            with pytest.raises(ValueError,match='Store artifact'):
                product.adopted_run_native_product(revision,'one',project_root=new,registered_sources=registered)
    assert {rel:(new/rel).read_bytes() for rel in before}==before
