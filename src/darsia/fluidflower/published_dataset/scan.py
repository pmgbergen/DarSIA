"""The builder's name scan: no names of people, machines or users in a record.

Every file a record would publish is scanned before anything is written: its
contents, the names of its files, folders and archive members. A hit is a
name (the build machine's host and user names, the host and user names in the
production job's receipts, and the deny-list of the local publication
settings) or an absolute path (a drive letter, a UNC path or a home folder).

One exception: a validation record may name the person who performed the
validation. A name in a ``reviewer`` or ``reviewer_name`` field of a JSON or
TOML document is allowed and listed as allowed; the same name anywhere else
is refused. Raw images are never edited: their EXIF owner, artist and
copyright fields are reported, not refused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import getpass
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import tomllib
import zipfile

# Fields of validation records that may name the validator.
ALLOWED_FIELDS = ("reviewer", "reviewer_name")
# Receipt keys whose string values name a machine or a user.
RECEIPT_NAME_KEYS = frozenset({
    "host", "hostname", "host_name", "machine", "machine_id", "computer",
    "computer_name", "physical_host", "user", "username", "user_name", "login",
    "publishing_machine_id", "created_by", "updated_by", "actor",
})
TEXT_SUFFIXES = frozenset({
    ".csv", ".tsv", ".md", ".txt", ".rst", ".log", ".ini", ".cfg", ".yaml",
    ".yml", ".xml", ".html", ".htm", ".svg", ".py", ".ps1", ".cmd", ".bat", ".sha256",
})
STRUCTURED_SUFFIXES = frozenset({".json", ".toml"})
# Minimum length of an automatically found host or user name that is scanned.
MINIMUM_AUTOMATIC_NAME = 3
# Minimum printable run in which a hit inside an unknown binary file counts.
MINIMUM_BINARY_STRING = 6

_SEPARATOR = r"[\W_]*"
_DRIVE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:(?:\\\\|\\|/)")
_UNC = re.compile(r"(?<![:\w\\/])(?:\\\\\\\\|\\\\|//)[A-Za-z0-9][\w.$-]*(?:\\\\|\\|/)")
_HOME = re.compile(r"(?<![\w.])(?:/home|/Users|/root)(?:/)[^/\s\"']+")
_PATH_PATTERNS = (("drive letter", _DRIVE), ("UNC path", _UNC), ("home folder", _HOME))
_PRINTABLE = re.compile(rb"[\x20-\x7e]{%d,}" % MINIMUM_BINARY_STRING)


@dataclass(frozen=True)
class ScanName:
    """A name to look for and where it came from (for the build report)."""

    text: str
    source: str


@dataclass(frozen=True)
class ScanHit:
    """One finding: a file, a place in it, what was found and whether it is allowed."""

    file: str
    where: str
    kind: str
    source: str
    excerpt: str
    allowed: bool = False

    def describe(self) -> str:
        """One report line: file, place, finding and, if so, that it is allowed."""
        state = "allowed (validation record)" if self.allowed else "refused"
        return f"{self.file} [{self.where}]: {self.kind} ({self.source}) {self.excerpt!r} - {state}"

    def as_dict(self) -> dict:
        """The hit as a JSON-ready dictionary."""
        return asdict(self)


def machine_names() -> tuple[ScanName, ...]:
    """The build machine's host name (full and short) and user names."""
    names = []
    host = socket.gethostname()
    for value in (host, host.split(".")[0]):
        names.append(ScanName(value, "build machine host name"))
    for value in (_safe(getpass.getuser), os.environ.get("USERNAME"), os.environ.get("USER"),
                  Path.home().name):
        if value:
            names.append(ScanName(value, "build machine user name"))
    return tuple(_unique(names))


def _safe(function):
    """``function()``, or an empty string when it fails."""
    try:
        return function()
    except Exception:  # noqa: BLE001 - an unknown login is simply not scanned
        return ""


def _unique(names):
    """Nonempty names, once per (name, source), stripped."""
    seen, result = set(), []
    for item in names:
        key = (item.text.strip().casefold(), item.source)
        if item.text.strip() and key not in seen:
            seen.add(key)
            result.append(ScanName(item.text.strip(), item.source))
    return result


def receipt_names(value, source: str = "production receipt") -> tuple[ScanName, ...]:
    """Machine and user names in a receipt document: string values of name keys."""
    found = []

    def walk(item):
        """Collect name values from one level of the receipt and below."""
        if isinstance(item, dict):
            for key, child in item.items():
                if str(key).casefold() in RECEIPT_NAME_KEYS and isinstance(child, str):
                    found.append(ScanName(child, source))
                elif str(key).casefold() in RECEIPT_NAME_KEYS and isinstance(child, dict):
                    for nested in ("machine", "host", "machine_id", "user"):
                        if isinstance(child.get(nested), str):
                            found.append(ScanName(child[nested], source))
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)
    return tuple(_unique(found))


def _pattern(name: str) -> re.Pattern:
    """Match a name as a word, its parts joined by any separators, and literally.

    Underscores count as word characters at the ends, so a machine name that is
    also a common word part (``x`` in ``x_mass``) does not match an identifier.
    """
    tokens = [token for token in re.split(r"[\W_]+", name) if token]
    flexible = _SEPARATOR.join(re.escape(token) for token in tokens)
    literal = re.escape(name.strip())
    body = flexible if literal == flexible else f"(?:{literal}|{flexible})"
    return re.compile(r"(?<![A-Za-z0-9_])" + body + r"(?![A-Za-z0-9_])", re.IGNORECASE)


class NameScanner:
    """Scan texts, documents, file names and archives for names and absolute paths."""

    def __init__(self, names=(), *, deny=()):
        """Compile the names (short automatic names are skipped) and the deny-list."""
        rows = list(_unique(names)) + [ScanName(text, "deny-list") for text in deny]
        self.names = []
        for item in rows:
            automatic = item.source != "deny-list"
            if automatic and len(item.text) < MINIMUM_AUTOMATIC_NAME:
                continue
            if not re.search(r"[A-Za-z0-9]", item.text):
                continue
            self.names.append((item, _pattern(item.text)))

    def text(self, value: str, *, file: str, where: str, allowed: bool = False) -> list[ScanHit]:
        """Hits in one piece of text; absolute paths are never allowed."""
        hits = []
        for item, pattern in self.names:
            for match in pattern.finditer(value):
                hits.append(ScanHit(file, where, "name", item.source, _excerpt(value, match), allowed))
        for label, pattern in _PATH_PATTERNS:
            for match in pattern.finditer(value):
                hits.append(ScanHit(file, where, "absolute path", label, _excerpt(value, match)))
        return hits

    def lines(self, data: bytes, *, file: str) -> list[ScanHit]:
        """Hits in a text file, reported by line number."""
        text = data.decode("utf-8", errors="replace")
        hits = []
        for number, line in enumerate(text.splitlines(), start=1):
            hits.extend(self.text(line, file=file, where=f"line {number}"))
        return hits

    def document(self, value, *, file: str, pointer: str = "", allowed: bool = False) -> list[ScanHit]:
        """Hits in a parsed JSON or TOML document, reported by key path.

        A string under a ``reviewer`` or ``reviewer_name`` key may name the
        validator: such name hits are listed as allowed.
        """
        hits = []
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{pointer}/{key}"
                hits.extend(self.text(str(key), file=file, where=f"key {path}"))
                hits.extend(self.document(child, file=file, pointer=path,
                                          allowed=str(key) in ALLOWED_FIELDS))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                hits.extend(self.document(child, file=file, pointer=f"{pointer}/{index}",
                                          allowed=allowed))
        elif isinstance(value, str):
            hits.extend(self.text(value, file=file, where=f"key {pointer or '/'}", allowed=allowed))
        return hits

    def path(self, relative: str, *, file: str, where: str = "file name") -> list[ScanHit]:
        """Hits in a record path: file, folder or archive member names."""
        return self.text(relative, file=file, where=where)

    def binary(self, data: bytes, *, file: str) -> list[ScanHit]:
        """Hits in printable runs of an unknown binary file, reported by byte offset."""
        hits = []
        for match in _PRINTABLE.finditer(data):
            text = match.group().decode("ascii")
            hits.extend(self.text(text, file=file, where=f"byte {match.start()}"))
        return hits

    def npz(self, data: bytes, *, file: str) -> list[ScanHit]:
        """Hits in a NumPy archive: member names and text arrays, never numbers."""
        import numpy as np

        hits = []
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.namelist():
                hits.extend(self.path(member, file=file, where="array name"))
        with np.load(io.BytesIO(data), allow_pickle=False) as arrays:
            for key in arrays.files:
                array = arrays[key]
                if array.dtype.kind in "US":
                    for index, item in enumerate(array.reshape(-1)):
                        text = item.decode("latin-1") if isinstance(item, bytes) else str(item)
                        hits.extend(self.text(text, file=file, where=f"array {key}[{index}]"))
        return hits

    def png(self, data: bytes, *, file: str) -> list[ScanHit]:
        """Hits in the text chunks of a PNG image."""
        hits = []
        position = 8
        while position + 8 <= len(data):
            length = int.from_bytes(data[position:position + 4], "big")
            kind = data[position + 4:position + 8]
            body = data[position + 8:position + 8 + length]
            if kind in (b"tEXt", b"iTXt", b"zTXt"):
                text = body.replace(b"\x00", b" ").decode("latin-1")
                hits.extend(self.text(text, file=file, where=f"PNG {kind.decode()} chunk"))
            if kind == b"IEND":
                break
            position += 12 + length
        return hits

    def content(self, data: bytes, *, file: str, suffix: str) -> list[ScanHit]:
        """Hits in a file's bytes, read according to its kind (by suffix)."""
        suffix = suffix.casefold()
        if suffix == ".json":
            try:
                return self.document(json.loads(data.decode("utf-8-sig")), file=file)
            except ValueError:
                return self.lines(data, file=file)
        if suffix == ".toml":
            try:
                return self.document(tomllib.loads(data.decode("utf-8-sig")), file=file)
            except (ValueError, tomllib.TOMLDecodeError):
                return self.lines(data, file=file)
        if suffix in TEXT_SUFFIXES:
            return self.lines(data, file=file)
        if suffix == ".npz":
            try:
                return self.npz(data, file=file)
            except (ValueError, OSError, zipfile.BadZipFile):
                return self.binary(data, file=file)
        if suffix == ".png":
            return self.png(data, file=file)
        if suffix == ".npy":
            return self.npy(data, file=file)
        return self.binary(data, file=file)

    def npy(self, data: bytes, *, file: str) -> list[ScanHit]:
        """Hits in a NumPy array file: its header and any text values."""
        import numpy as np

        array = np.load(io.BytesIO(data), allow_pickle=False)
        hits = []
        if array.dtype.kind in "US":
            for index, item in enumerate(array.reshape(-1)):
                text = item.decode("latin-1") if isinstance(item, bytes) else str(item)
                hits.extend(self.text(text, file=file, where=f"value {index}"))
        return hits

    def file(self, path: str | Path, *, file: str) -> list[ScanHit]:
        """Hits in one source file on disk (its content) and in its record name."""
        source = Path(path)
        hits = self.path(file, file=file)
        hits.extend(self.content(source.read_bytes(), file=file, suffix=source.suffix))
        return hits

    def raw_image(self, path: str | Path, *, file: str) -> tuple[list[ScanHit], list[dict]]:
        """A raw image: its record name is scanned; EXIF identity fields are reported."""
        from .exif import identity_fields

        hits = self.path(file, file=file)
        fields = identity_fields(path)
        reports = [dict(file=file, field=key, value=value) for key, value in fields.items()]
        return hits, reports


def _excerpt(text: str, match: re.Match, width: int = 24) -> str:
    """The matched text with a little context on both sides."""
    start, end = max(0, match.start() - width), min(len(text), match.end() + width)
    return text[start:end]


def refused(hits) -> list[ScanHit]:
    """The hits that stop a build: everything except allowed validator names."""
    return [hit for hit in hits if not hit.allowed]


def record_suffix(name: str) -> str:
    """The suffix used to choose how a record file is read."""
    return PurePosixPath(name).suffix
