"""Collect explicitly indexed products; no discovery, RPC or scientific work."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .pipeline.job_queue import descriptor
from .pipeline.products_publication import collect_products
from .store_map import read_store_map

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    collect=sub.add_parser("collect",help="Hash-verified copies of explicitly listed product IDs")
    collect.add_argument("--index",required=True,type=Path)
    collect.add_argument("--index-sha256",required=True)
    collect.add_argument("--index-bytes",required=True,type=int)
    collect.add_argument("--product-id",action="append",required=True)
    collect.add_argument("--target",required=True,type=Path)
    collect.add_argument("--stores",type=Path)
    collect.add_argument("--source",action="append",default=[],metavar="PRODUCT_ID=ACCESSIBLE_PATH",
        help="Explicit accessible source alias for an original product on another machine")
    collect.add_argument("--published-bundle",action="append",default=[],type=Path,
        help="Explicit immutable publication.json already on the shared filesystem")
    args=parser.parse_args(argv)
    # Only the externally stored machine identity is consulted, never inferred
    # from a product's local drive spelling or from the shared-folder creator.
    repository=Path(__file__).resolve().parents[3]
    stores=read_store_map(args.stores,forbidden_roots=[repository])
    sources={}
    for item in args.source:
        key,separator,path=item.partition("=")
        if not separator or not key or not path or key in sources:
            raise ValueError("Each source must bind one distinct PRODUCT_ID=ACCESSIBLE_PATH")
        sources[key]=Path(path)
    bound=dict(path=str(args.index),sha256=args.index_sha256,bytes=args.index_bytes)
    bundles=[descriptor(path) for path in args.published_bundle]
    result=collect_products(bound,args.product_id,args.target,current_machine_id=stores.machine_id,
                            source_paths=sources,payload_publications=bundles,file_writers_closed=True)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return result

if __name__=="__main__":
    main()
