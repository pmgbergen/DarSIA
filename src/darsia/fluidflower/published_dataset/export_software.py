"""Explicit software closure for a reproduction export; never install or download.

An inventory supplies the producing job's exact lock, a public code URL and,
for inclusion, a self-contained Git bundle and one wheel per locked package.
The export never substitutes the environment of the machine doing the export.
"""
from __future__ import annotations

from email.parser import BytesParser
import base64
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import urlsplit
import zipfile

from .archives import Member, file_sha256
from .layout import inside, record_relative

FORMAT = "darsia-export-software-input-v1"


def _name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def _url(value):
    parsed = urlsplit(str(value))
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Software references need public HTTPS URLs without credentials")
    return str(value)


def _asset(root, row):
    path = inside(root, row["path"])
    if (type(row.get("size_bytes")) is not int or row["size_bytes"] < 1
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("sha256", "")))
            or path.stat().st_size != row["size_bytes"] or file_sha256(path) != row["sha256"]):
        raise ValueError("Software file differs from its explicit size or SHA-256")
    return path


def _pins(raw):
    pins = {}
    for line in raw.decode("utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match is None or _name(match[1]) in pins:
            raise ValueError("The software lock must contain unique exact name==version pins")
        pins[_name(match[1])] = match[2]
    if not pins:
        raise ValueError("The software lock contains no exact library versions")
    return pins


def _wheel(path, name, version):
    """Check a wheel's distribution identity and complete internal RECORD."""
    try:
        with zipfile.ZipFile(path) as archive:
            files = [item.filename for item in archive.infolist() if not item.is_dir()]
            if len(files) != len(set(files)):
                raise ValueError("Library wheel contains duplicate members")
            for member in files:
                record_relative(member)
            metadata = [item for item in files if item.endswith(".dist-info/METADATA")]
            if len(metadata) != 1:
                raise ValueError("A library wheel must contain exactly one distribution metadata file")
            info = BytesParser().parsebytes(archive.read(metadata[0]))
            if _name(info.get("Name", "")) != name or info.get("Version") != version:
                raise ValueError("Wheel metadata differs from the locked library: " + name)
            prefix = metadata[0].rsplit("/", 1)[0]
            if prefix + "/WHEEL" not in files or prefix + "/RECORD" not in files:
                raise ValueError("Library wheel lacks WHEEL or RECORD")
            wheel = BytesParser().parsebytes(archive.read(prefix + "/WHEEL"))
            if not str(wheel.get("Wheel-Version", "")).startswith("1.") or not wheel.get_all("Tag"):
                raise ValueError("Library wheel format or compatibility tags are missing")
            records = list(csv.reader(io.StringIO(archive.read(prefix + "/RECORD").decode("utf-8"))))
            if any(len(row) != 3 for row in records):
                raise ValueError("Library wheel RECORD is malformed")
            names = [row[0] for row in records]
            signatures = {prefix + "/RECORD.jws", prefix + "/RECORD.p7s"}
            if len(names) != len(set(names)) or set(names) != set(files) - signatures:
                raise ValueError("Library wheel RECORD does not cover its complete contents")
            for member, digest, size in records:
                if member == prefix + "/RECORD":
                    if digest or size:
                        raise ValueError("Wheel RECORD must not hash itself")
                    continue
                algorithm, separator, expected = digest.partition("=")
                if separator != "=" or algorithm not in {"sha256", "sha384", "sha512"}:
                    raise ValueError("Library wheel needs strong hashes for every member")
                raw = archive.read(member)
                actual = base64.urlsafe_b64encode(hashlib.new(algorithm, raw).digest()).rstrip(b"=").decode("ascii")
                if size != str(len(raw)) or actual != expected:
                    raise ValueError("Library wheel member differs from its RECORD")
    except zipfile.BadZipFile as error:
        raise ValueError("Unreadable library wheel") from error


def _bundle(path, commit):
    """Verify the complete Git object closure in a disposable, offline repository.

    A valid advertised head is insufficient: the bundle's pack can still be
    truncated or omit a reachable tree/blob. No source is checked out or run.
    Submodules and LFS pointers need their own supplied payloads and cannot be
    presented as all code by this single-bundle format.
    """
    with path.open("rb") as stream:
        header, header_bytes = [], 0
        while True:
            line = stream.readline(4097)
            header_bytes += len(line)
            if len(line) > 4096 or header_bytes > 1024 * 1024:
                raise ValueError("Invalid Git bundle header")
            if line == b"\n":
                break
            if not line:
                raise ValueError("Incomplete Git bundle header")
            header.append(line)
    if not header or header[0] not in {b"# v2 git bundle\n", b"# v3 git bundle\n"}:
        raise ValueError("Included source must be a Git bundle")
    if any(line.startswith(b"-") for line in header):
        raise ValueError("The source bundle needs unavailable prerequisite commits")
    with tempfile.TemporaryDirectory(prefix="darsia-export-software-") as temporary:
        root = Path(temporary)
        template, repository = root / "template", root / "objects.git"
        template.mkdir()
        # The caller's repository, alternate object stores and injected config
        # must never help prove that the supplied bundle is self-contained.
        environment = {key: value for key, value in os.environ.items()
                       if not key.upper().startswith("GIT_")}
        environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_SYSTEM=os.devnull,
            GIT_CONFIG_GLOBAL=os.devnull, GIT_NO_REPLACE_OBJECTS="1",
            GIT_OPTIONAL_LOCKS="0", GIT_TERMINAL_PROMPT="0", GIT_ALLOW_PROTOCOL="",
            GIT_NO_LAZY_FETCH="1")

        def git(*args, initializing=False, check=True):
            return subprocess.run(["git", "-C", str(root if initializing else repository),
                "-c", "core.hooksPath=" + str(template), "-c", "core.fsmonitor=false", *args],
                text=True, capture_output=True, timeout=60, env=environment, check=check)

        def require_disposable_objects():
            git_dir = Path(git("rev-parse", "--absolute-git-dir").stdout.strip()).resolve()
            objects = Path(git("rev-parse", "--git-path", "objects").stdout.strip())
            if not objects.is_absolute():
                objects = repository / objects
            if (git_dir != repository.resolve() or objects.resolve() != repository.resolve() / "objects"
                    or any((objects / "info" / name).exists() or (objects / "info" / name).is_symlink()
                           for name in ("alternates", "http-alternates"))):
                raise ValueError("Bundle verification must use only its disposable object store")

        try:
            git("init", "--bare", "--template=" + str(template), str(repository), initializing=True)
            require_disposable_objects()
            heads = git("bundle", "list-heads", str(path.resolve())).stdout
            if commit not in [line.split()[0] for line in heads.splitlines()]:
                raise ValueError("The source bundle must advertise the exact published commit")
            git("bundle", "unbundle", str(path.resolve()))
            require_disposable_objects()
            git("cat-file", "-e", commit + "^{commit}")
            git("fsck", "--full", "--strict", "--no-reflogs", commit)
            tree = git("ls-tree", "-r", "--full-tree", commit).stdout
            if any(line.startswith("160000 ") for line in tree.splitlines()):
                raise ValueError("Included code has submodules whose source is absent from the bundle")
            lfs = git("grep", "-I", "-l", "-e", "^version https://git-lfs.github.com/spec/v1$",
                      commit, "--", check=False)
            if lfs.returncode == 0:
                raise ValueError("Included code has Git LFS pointers whose payloads are absent from the bundle")
            if lfs.returncode != 1:
                raise ValueError("Could not verify the included source for external payloads")
        except subprocess.CalledProcessError as error:
            raise ValueError("The source bundle does not contain a complete, readable Git object closure") from error


def folder_inventory(request, job, published_commit):
    """Prepare the same inventory from explicit GUI file choices, without writes.

    The chosen folder holds the job's lock at its original relative name. For
    inclusion it also holds source.bundle and wheels/. Nothing is taken from
    the export machine's installed environment, and no registry is contacted.
    """
    root = Path(request.software_folder)
    lock_name = record_relative((job.job.get("environment_lock") or {}).get("path", ""))
    lock_path = inside(root, lock_name)
    raw = lock_path.read_bytes()
    pins = _pins(raw)

    def asset(path):
        path = inside(root, path.relative_to(root).as_posix())
        return dict(path=path.relative_to(root).as_posix(), sha256=file_sha256(path),
                    size_bytes=path.stat().st_size)

    value = dict(format=FORMAT, python=request.software_python,
                 code=dict(commit=published_commit, url=request.software_code_url),
                 environment_lock=asset(lock_path),
                 libraries=[dict(name=name, version=version) for name, version in sorted(pins.items())])
    if request.software_mode == "include":
        value["code"]["bundle"] = asset(root / "source.bundle")
        candidates = {}
        # Only this explicitly selected folder is inspected. A second build of
        # the same library needs a user choice, never an arbitrary first match.
        for path in sorted((root / "wheels").glob("*.whl")):
            path = inside(root, path.relative_to(root).as_posix())
            with zipfile.ZipFile(path) as archive:
                metadata = [key for key in archive.namelist() if key.endswith(".dist-info/METADATA")]
                if len(metadata) != 1:
                    raise ValueError("A library wheel has no unique distribution metadata")
                info = BytesParser().parsebytes(archive.read(metadata[0]))
            key = _name(info.get("Name", "")), info.get("Version")
            if key[0] in pins and pins[key[0]] == key[1]:
                if key in candidates:
                    raise ValueError("Choose exactly one wheel build for locked library " + key[0])
                candidates[key] = path
        for row in value["libraries"]:
            path = candidates.get((row["name"], row["version"]))
            if path is None:
                raise ValueError("A locked library wheel is missing: " + row["name"] + "==" + row["version"])
            row.update(asset(path))
    return root, value


def software_members(request, job, published_commit):
    """Return verified software archive members and portable version references.

    ``None`` keeps the established builder contract for existing callers.
    The Results and export dialog always selects references or inclusion.
    """
    mode = request.software_mode
    if mode is None:
        return [], None
    if request.record != "R" or mode not in {"references", "include"}:
        raise ValueError("Choose exact software references or included software for a reproduction package")
    if request.software_inventory is not None:
        path = Path(request.software_inventory)
        root = path.parent
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    elif request.software_folder is not None:
        root, value = folder_inventory(request, job, published_commit)
    else:
        raise ValueError("Choose the software folder or an existing inventory binding the producing job's code and libraries")
    if value.get("format") != FORMAT:
        raise ValueError("An explicit export software inventory is required")
    code = value["code"]
    if code.get("commit") != published_commit:
        raise ValueError("Software code commit differs from the published revision")
    code_url = _url(code["url"])
    lock = value["environment_lock"]
    if lock.get("sha256") != (job.job.get("environment_lock") or {}).get("sha256"):
        raise ValueError("Software lock differs from the producing job")
    lock_path = _asset(root, lock)
    raw = lock_path.read_bytes()
    pins = _pins(raw)
    python = value.get("python")
    match = re.search(r"Python\s+(\d+\.\d+)", raw.decode("utf-8-sig"))
    if (not isinstance(python, str) or re.fullmatch(r"\d+\.\d+\.\d+", python) is None
            or match is None or not python.startswith(match[1] + ".")):
        raise ValueError("Record the exact Python version matching the producing job's lock")
    libraries = value.get("libraries")
    if not isinstance(libraries, list):
        raise ValueError("Software inventory must list every locked library")
    by_name = {_name(row["name"]): row for row in libraries}
    if len(by_name) != len(libraries) or set(by_name) != set(pins):
        raise ValueError("Software inventory must contain each locked library exactly once")
    members = [Member("software/requirements-worker.lock", data=raw)]
    references = []
    for name, version in sorted(pins.items()):
        row = by_name[name]
        if row.get("version") != version:
            raise ValueError("Library version differs from the producing job's lock: " + name)
        reference = dict(name=name, version=version)
        if row.get("url") is not None:
            reference["url"] = _url(row["url"])
        if mode == "include":
            wheel = _asset(root, row)
            if wheel.suffix != ".whl":
                raise ValueError("Include an installable wheel for every locked library: " + name)
            _wheel(wheel, name, version)
            target = "software/wheels/" + wheel.name
            members.append(Member(target, wheel, row["sha256"], row["size_bytes"]))
            reference.update(path=target, sha256=row["sha256"], size_bytes=row["size_bytes"])
        references.append(reference)
    document = dict(mode=mode, python=python, code=dict(url=code_url, commit=published_commit),
                    environment_lock=dict(path="software/requirements-worker.lock", sha256=lock["sha256"]),
                    libraries=references)
    if mode == "include":
        bundle = _asset(root, code["bundle"])
        _bundle(bundle, published_commit)
        bound = code["bundle"]
        members.append(Member("software/source.bundle", bundle, bound["sha256"], bound["size_bytes"]))
        document["code"].update(path="software/source.bundle", sha256=bound["sha256"],
                                 size_bytes=bound["size_bytes"])
    names = [member.name.casefold() for member in members]
    if len(names) != len(set(names)):
        raise ValueError("Included software files have conflicting portable names")
    versions = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    members.append(Member("software/versions.json", data=versions))
    return members, dict(document, inventory=dict(path="software/versions.json",
        sha256=hashlib.sha256(versions).hexdigest(), size_bytes=len(versions)))
