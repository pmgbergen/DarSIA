"""Original serializer partial names and retained members, without numerical imports."""
from pathlib import PureWindowsPath


def partial_name(name, token):
    """Name the original exclusive temporary file or frame directory."""
    return "." + name + "." + token + ".partial"


def retained_member(path, target):
    """Include a committed target and all retained serializer partial descendants."""
    path, target = PureWindowsPath(path), PureWindowsPath(target)
    if path == target or path.is_relative_to(target):
        return True
    if not path.is_relative_to(target.parent):
        return False
    first = path.relative_to(target.parent).parts[0]
    return first.startswith("." + target.name + ".") and first.endswith(".partial")
