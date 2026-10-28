"""The retired UI action cannot start its former calculation backend."""
import ast
from pathlib import Path


def test_retired_optical_preview_has_no_calculator_imports():
    source = Path(__file__).parents[2] / 'src/darsia/fluidflower/ui/optical_cleanup.py'
    tree = ast.parse(source.read_bytes())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert imports == ['PySide6.QtWidgets']
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    active = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'active')
    namespace = {}
    code = ast.Module(body=[ast.FunctionDef(name='active', args=active.args, body=active.body, decorator_list=[], returns=None, type_comment=None)], type_ignores=[])
    ast.fix_missing_locations(code)
    exec(compile(code, '<retired-ui-slot>', 'exec'), namespace)
    assert namespace['active'](object()) is False
    asset = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'asset')
    assert any(isinstance(node, ast.Raise) for node in asset.body)
