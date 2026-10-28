"""Exact SDK reviewer/run-order reuse; pure functions and fake imports, no GUI."""
import ast
import re
import subprocess
import sys
import types
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
BASE="65c0e85015700f764b2c4f5fed6bb558490ef7f0"
DESKTOP="src/darsia/fluidflower/tools/calibration_review_desktop.py"
PAIRWISE="src/darsia/fluidflower/tools/calibration_pairwise_review.py"
SORT="src/darsia/fluidflower/tools/phase_review_area_sort.py"

def load(path,name,original=False):
    raw=(subprocess.check_output(["git","show",BASE+":"+path],cwd=ROOT).decode("utf8")
         if original else (ROOT/path).read_bytes().decode("utf8"))
    tree=ast.parse(raw)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    code=ast.Module(body=[ast.ImportFrom(module="__future__",names=[ast.alias(name="annotations")],level=0),node],type_ignores=[])
    ns={"re":re,"__package__":"darsia.fluidflower.tools"}
    exec(compile(ast.fix_missing_locations(code),path,"exec"),ns)
    return ns[name]

def outcome(fn,*a,**kw):
    try:return ("value",fn(*a,**kw))
    except Exception as e:return ("error",type(e).__name__,str(e))

@pytest.mark.parametrize("name",["Reviewer"," Åsa  ÆØÅ ","","___","Alice.BOB  2",None,12])
def test_reviewer_identifier_matches_original_and_shared_owner(monkeypatch,name):
    canonical=load(PAIRWISE,"_reviewer_identifier")
    module=types.ModuleType("darsia.fluidflower.tools.calibration_pairwise_review")
    module._reviewer_identifier=canonical
    monkeypatch.setitem(sys.modules,module.__name__,module)
    assert outcome(load(DESKTOP,"reviewer_identifier"),name)==outcome(load(DESKTOP,"reviewer_identifier",True),name)
    assert outcome(canonical,name)==outcome(load(PAIRWISE,"_reviewer_identifier",True),name)

@pytest.mark.parametrize("run",["ac2","ac10","","none","a001b2","a١٢","a²",None])
def test_run_number_matches_both_original_sdk_owners(monkeypatch,run):
    canonical=load(SORT,"_run_number")
    module=types.ModuleType("darsia.fluidflower.tools.phase_review_area_sort")
    module._run_number=canonical
    monkeypatch.setitem(sys.modules,module.__name__,module)
    expected=outcome(load(DESKTOP,"_run_number",True),run)
    assert outcome(canonical,run)==expected
    assert outcome(load(DESKTOP,"_run_number"),run)==expected
    assert outcome(load(SORT,"run_number",True),run)==expected

def test_full_area_sort_retains_unknown_report_and_stable_ties():
    actual=load(SORT,"sorted_phase_area_runs")
    actual.__globals__["_run_number"]=load(SORT,"_run_number")
    original=load(SORT,"sorted_phase_area_runs",True)
    entries=[("ac10","b","c",1),("ac2","b","c",2),("none","b","c",3),("ac1","b","c",4)]
    areas={"ac10":{"left":{"gas":2.0}},"ac2":{"left":{"gas":2.0}},"ac1":{"left":{"gas":0.0}}}
    for largest in (False,True):
        assert actual(entries,areas,side="left",phase="gas",largest_first=largest)==original(entries,areas,side="left",phase="gas",largest_first=largest)
