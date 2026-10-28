"""Real owned Git fixtures; no monkeypatch of writer guards or scientific import."""
from pathlib import Path
import ast
import hashlib
import subprocess

import pytest

from darsia.fluidflower.pipeline import capacity as cap


WRITER = "src/darsia/fluidflower/mass/n1_pipeline.py"
MEASURED = "1d249265e9e6bbec8caf8e008a7f0083671d7880"
STORAGE_ONLY = "cefad7cf06df4079efee6352bde6aef7b55c0025"
JSON_FIXED = "1f5f0016d7c1a8d3014d8fc490afc8dd5b503b29"
SOURCE_SHA = {
    MEASURED: "cbf74488db1e93d608762f2cc45b5648e033c2dde51a3b291f4185374a259833",
    STORAGE_ONLY: "df74446a61eed48e267dc495ab10585c714c5dfc21c440c1a6a41591c1086f49",
    JSON_FIXED: "23f4ca6748959e150c0850c7f60b54c4aed7dbccb6a9555600b0d6f051e45972",
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(scope="module")
def sources():
    repository = Path(__file__).resolve().parents[3]
    values = {}
    for commit, expected in SOURCE_SHA.items():
        raw = subprocess.check_output(["git", "-C", str(repository), "show", commit + ":" + WRITER])
        assert sha(raw) == expected
        values[commit] = raw
    return values


def fixture_git(tmp_path, measured, current, *, eol="lf", relative=WRITER):
    repository = tmp_path / "writer-git"
    repository.mkdir()

    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(repository), *arguments], stderr=subprocess.STDOUT).decode().strip()

    git("init", "--object-format=sha1")
    git("config", "core.autocrlf", "false")
    git("config", "core.eol", eol)
    (repository / ".gitattributes").write_bytes(b"* text\n")
    path = repository / relative
    path.parent.mkdir(parents=True)
    commits = []
    for label, raw in [("measured", measured), ("current", current)]:
        path.write_bytes(raw.replace(b"\n", b"\r\n") if eol == "crlf" else raw)
        git("add", "--", ".gitattributes", relative)
        git("-c", "user.name=Validation fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "--allow-empty", "-m", label)
        commits.append(git("rev-parse", "HEAD"))
    git("checkout-index", "--force", "--all")
    assert not git("status", "--porcelain=v1", "--untracked-files=all")
    assert path.read_bytes().replace(b"\r\n", b"\n") == current
    row = dict(relative_path=relative, measured_sha256=sha(measured), sha256=sha(current))
    return repository, commits[0], commits[1], row


@pytest.mark.parametrize("eol", ["lf", "crlf"])
def test_exact_named_JSON_and_storage_pairs_pass_real_clean_Git(sources, tmp_path, eol):
    args = fixture_git(tmp_path, sources[MEASURED], sources[JSON_FIXED], eol=eol)
    cap._profile_writer_identity(*args)
    raw = (args[0] / WRITER).read_bytes()
    assert (b"\r\n" in raw) == (eol == "crlf")


def test_original_storage_only_path_remains_valid_without_JSON_exception(sources, tmp_path):
    args = fixture_git(tmp_path, sources[MEASURED], sources[STORAGE_ONLY])
    cap._profile_writer_identity(*args)


def test_original_two_key_identical_writer_path_remains_valid(sources, tmp_path):
    repository, measured, current, row = fixture_git(tmp_path, sources[MEASURED], sources[MEASURED])
    del row["measured_sha256"]
    cap._profile_writer_identity(repository, measured, current, row)


def changed(raw, defect):
    tree = ast.parse(raw)
    reader = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DataReader")
    if defect == "third_import":
        tree.body.append(ast.Import(names=[ast.alias(name="fractions")]))
    elif defect == "third_method":
        next(node for node in reader.body if isinstance(node, ast.FunctionDef) and node.name == "path").body.append(ast.Pass())
    elif defect == "third_executor":
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_n1").body.append(ast.Pass())
    elif defect == "json_pair":
        next(node for node in reader.body if isinstance(node, ast.FunctionDef) and node.name == "json").body.append(ast.Pass())
    elif defect == "storage_pair":
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_storage_facts").body.append(ast.Pass())
    elif defect == "missing_json":
        reader.body = [node for node in reader.body if not isinstance(node, ast.FunctionDef) or node.name != "json"]
    elif defect == "duplicate_reader":
        tree.body.append(ast.parse(ast.unparse(reader)).body[0])
    else:
        raise AssertionError(defect)
    return (ast.unparse(ast.fix_missing_locations(tree)) + "\n").encode()


@pytest.mark.parametrize("defect", ["third_import", "third_method", "third_executor"])
def test_any_third_AST_change_refuses_even_with_correct_whole_SHAs(sources, tmp_path, defect):
    args = fixture_git(tmp_path, sources[MEASURED], changed(sources[JSON_FIXED], defect))
    with pytest.raises(ValueError, match="actual writer/executor body changed"):
        cap._profile_writer_identity(*args)


@pytest.mark.parametrize("side,defect", [("current", "json_pair"), ("measured", "json_pair"),
    ("current", "storage_pair"), ("measured", "storage_pair"),
    ("current", "missing_json"), ("current", "duplicate_reader")])
def test_only_exact_named_function_pairs_and_one_reader_are_admitted(sources, tmp_path, side, defect):
    measured, current = sources[MEASURED], sources[JSON_FIXED]
    if side == "measured": measured = changed(measured, defect)
    else: current = changed(current, defect)
    args = fixture_git(tmp_path, measured, current)
    with pytest.raises(ValueError, match="exact reviewed"):
        cap._profile_writer_identity(*args)


@pytest.mark.parametrize("field", ["sha256", "measured_sha256"])
def test_both_whole_file_provenance_SHAs_are_still_required(sources, tmp_path, field):
    repository, measured, current, row = fixture_git(tmp_path, sources[MEASURED], sources[JSON_FIXED])
    row[field] = "0" * 64
    with pytest.raises(ValueError, match="identity differs|provenance SHA differs"):
        cap._profile_writer_identity(repository, measured, current, row)


def test_current_physical_code_must_still_match_its_canonical_Git(sources, tmp_path):
    repository, measured, current, row = fixture_git(tmp_path, sources[MEASURED], sources[JSON_FIXED])
    path = repository / WRITER
    path.write_bytes(path.read_bytes() + b"# uncommitted physical change\n")
    with pytest.raises(ValueError, match="writer identity differs"):
        cap._profile_writer_identity(repository, measured, current, row)


@pytest.mark.parametrize("defect", ["extra_key", "another_writer"])
def test_no_new_schema_or_other_writer_exception_is_created(sources, tmp_path, defect):
    relative = WRITER if defect == "extra_key" else "src/darsia/fluidflower/another_writer.py"
    repository, measured, current, row = fixture_git(tmp_path, sources[MEASURED], sources[JSON_FIXED], relative=relative)
    if defect == "extra_key": row["any_io_is_safe"] = True
    with pytest.raises(ValueError, match="Only the exact n1_pipeline"):
        cap._profile_writer_identity(repository, measured, current, row)
