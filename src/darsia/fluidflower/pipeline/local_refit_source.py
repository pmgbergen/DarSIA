"""Content identity for the numerical code already loaded by local Main.

No Git, checkout cleanliness, provider or installation qualification is used.
Compilation below only compares code; compiled module bodies are never executed.
"""
from __future__ import annotations

import ast
import builtins
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import types

FORMAT = "darsia-local-refit-source-v1"
MODULES = (
    "darsia.fluidflower.mass.n1_refit",
    "darsia.fluidflower.mass.n1_statistics",
    "darsia.fluidflower.mass.n1_arithmetic",
    "darsia.fluidflower.mass.concentration_joint",
)


def _code_value(code):
    def constant(value):
        if isinstance(value, types.CodeType):
            return _code_value(value)
        if isinstance(value, tuple):
            return ["tuple", [constant(item) for item in value]]
        if isinstance(value, frozenset):
            return ["frozenset", sorted((constant(item) for item in value), key=repr)]
        return [type(value).__name__, repr(value)]
    return [code.co_name, code.co_qualname, code.co_argcount, code.co_posonlyargcount,
            code.co_kwonlyargcount, code.co_nlocals, code.co_stacksize, code.co_flags,
            code.co_code.hex(), [constant(value) for value in code.co_consts],
            list(code.co_names), list(code.co_varnames), list(code.co_freevars),
            list(code.co_cellvars), code.co_exceptiontable.hex()]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _literal_value(value):
    if isinstance(value, (set, frozenset)):
        return [type(value).__name__, sorted((_literal_value(item) for item in value), key=repr)]
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_literal_value(item) for item in value]]
    if isinstance(value, dict):
        return ["dict", sorted(([_literal_value(key), _literal_value(item)]
                               for key, item in value.items()), key=repr)]
    return [type(value).__name__, repr(value)]


def _default(node, module):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        if isinstance(node, ast.Name):
            return vars(module).get(node.id, getattr(builtins, node.id, None))
        if isinstance(node, ast.Attribute):
            return getattr(_default(node.value, module), node.attr)
        raise ValueError("Unsupported numerical default expression: " + ast.unparse(node))


def _module_binding(module, root):
    """Bind exact source bytes to live authored functions, constants and fields."""
    path, root = Path(module.__file__).resolve(), Path(root).resolve()
    if path.suffix != ".py" or not path.is_relative_to(root):
        raise ValueError("Loaded numerical module is outside the selected Main source: " + module.__name__)
    raw = path.read_bytes()
    tree = ast.parse(raw, filename=str(path))
    compiled = compile(raw, str(path), "exec", dont_inherit=True, optimize=sys.flags.optimize)
    codes = {}
    def index(code):
        codes[code.co_qualname] = code
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                index(value)
    index(compiled)
    functions, literals, fields = {}, {}, {}

    def function(node, owner, qualified):
        actual = inspect.getattr_static(owner, node.name)
        if isinstance(actual, (staticmethod, classmethod)):
            actual = actual.__func__
        if (not isinstance(actual, types.FunctionType) or actual.__module__ != module.__name__
                or _code_value(actual.__code__) != _code_value(codes[qualified])):
            raise ValueError("Loaded numerical function differs from its source: " + qualified)
        expected = tuple(_default(value, module) for value in node.args.defaults)
        keyword = {arg.arg: _default(value, module)
                   for arg, value in zip(node.args.kwonlyargs, node.args.kw_defaults)
                   if value is not None}
        if (actual.__defaults__ or ()) != expected or (actual.__kwdefaults__ or {}) != keyword:
            raise ValueError("Loaded numerical defaults differ from source: " + qualified)
        functions[qualified] = _digest(_code_value(actual.__code__))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            function(node, module, node.name)
        elif isinstance(node, ast.ClassDef):
            cls = getattr(module, node.name)
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    function(child, cls, node.name + "." + child.name)
            declared = [child for child in node.body if isinstance(child, ast.AnnAssign)
                        and isinstance(child.target, ast.Name)]
            if hasattr(cls, "__dataclass_fields__"):
                if list(cls.__dataclass_fields__) != [child.target.id for child in declared]:
                    raise ValueError("Loaded numerical dataclass fields differ: " + node.name)
                from dataclasses import MISSING
                values = {}
                for child in declared:
                    field = cls.__dataclass_fields__[child.target.id]
                    default = _default(child.value, module) if child.value is not None else MISSING
                    annotation = ast.unparse(child.annotation)
                    if (field.default != default or field.default_factory is not MISSING
                            or ast.dump(ast.parse(str(field.type), mode="eval").body)
                            != ast.dump(ast.parse(annotation, mode="eval").body)):
                        raise ValueError("Loaded numerical dataclass declaration differs: " + node.name)
                    values[child.target.id] = dict(annotation=annotation,
                        default="MISSING" if default is MISSING else repr(default))
                fields[node.name] = values
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            try:
                expected = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    if getattr(module, target.id) != expected:
                        raise ValueError("Loaded numerical constant differs: " + target.id)
                    literals[target.id] = _literal_value(expected)
    if path.read_bytes() != raw:
        raise ValueError("Numerical source changed while binding: " + str(path))
    return dict(module=module.__name__, relative_path=path.relative_to(root).as_posix(),
                sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw),
                functions=functions, constants=literals, dataclass_fields=fields)


def source_identity(repository=None, *, calibration_sha256, precision="float32", overflow=False):
    """Record the actual loaded numerical implementation; unrelated files are ignored."""
    root = Path(repository or Path(__file__).resolve().parents[4]).resolve()
    if precision != "float32":
        raise ValueError("Local refit uses the original float32 task precision")
    names = MODULES + (("darsia.fluidflower.mass.overflow_steering",
        "darsia.fluidflower.mass.overflow_statistics", "darsia.fluidflower.mass.overflow_measurements",
        "darsia.fluidflower.mass.overflow_method", "darsia.fluidflower.mass.overflow_levels",
        "darsia.fluidflower.mass.overflow_refit", "darsia.fluidflower.mass.overflow_phase_split") if overflow else ())
    modules = [importlib.import_module(name) for name in names]
    rows = []
    for module in modules:
        expected = root / "src" / Path(*module.__name__.split(".")).with_suffix(".py")
        if Path(module.__file__).resolve() != expected.resolve():
            raise ValueError("Main loaded numerical code from another source root: " + module.__name__)
        rows.append(_module_binding(module, root))
    # Imported numerical aliases must still reference the checked implementation.
    by_name = {module.__name__: module for module in modules}
    for module in modules:
        for value in vars(module).values():
            if isinstance(value, types.FunctionType) and value.__module__ in by_name:
                canonical = getattr(by_name[value.__module__], value.__name__, None)
                if value is not canonical:
                    raise ValueError("Loaded numerical alias differs from its source function")
    runtime = {"python": sys.version, "implementation": sys.implementation.name}
    for name in ("numpy", "scipy", "threadpoolctl"):
        runtime[name] = importlib.import_module(name).__version__
    content = dict(format=FORMAT, modules=rows, runtime=runtime,
        scope=("Original refit and overflow numerical modules" if overflow else
               "Four original refit numerical modules; runtime versions recorded, no binary/provider qualification"))
    digest = _digest(content)
    binding = dict(content, sha256=digest)
    return dict(commit="local-source-sha256:" + digest, source_binding=binding,
                precision=precision, calibration_sha256=calibration_sha256,
                created_utc=datetime.now(timezone.utc).isoformat())
