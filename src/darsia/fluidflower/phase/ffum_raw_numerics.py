"""SHA-bound numerical functions for the preserved FF_UM training methods."""
import ast
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import cv2
import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load_numerics(snapshot):
    """Import bound source functions without historical dispatch or publication."""
    root = Path(snapshot)
    manifest = json.loads((root/'source_binding.json').read_bytes())
    expected = {'logistic.py', 'coupled_aqueous.py', 'joint_aqueous.py', 'phase_features.py',
                'phase_optical_regions.py', 'source_sampling.py', 'source_annotation.py',
                'source_pooling.py', 'source_gas.py', 'source_guards.py', 'source_freegas.py'}
    if set(manifest) != expected:
        raise ValueError('The complete UM numerical source bundle is required')
    for name, descriptor in manifest.items():
        if sha(root/name) != descriptor['sha256']:
            raise ValueError('Changed UM numerical source: '+name)
    prediction_binding = json.loads((root/'prediction_source_binding.json').read_bytes())
    if (set(prediction_binding) != {'source_learning.py'}
            or sha(root/'source_learning.py') != prediction_binding['source_learning.py']['sha256']):
        raise ValueError('Changed UM CPU64 optical prediction source')
    prefix = '_um_raw_math_'+sha(root/'source_binding.json')[:16]
    package = ModuleType(prefix); package.__path__ = [str(root)]
    sys.modules[prefix] = package
    def module(name):
        spec = importlib.util.spec_from_file_location(prefix+'.'+name, root/(name+'.py'))
        value = importlib.util.module_from_spec(spec); sys.modules[spec.name] = value
        spec.loader.exec_module(value); setattr(package, name, value)
        return value
    logistic = module('logistic'); module('coupled_aqueous')
    joint = module('joint_aqueous'); optical = module('phase_features')
    regional = module('phase_optical_regions')
    def selected(filename, names, globals=None):
        value = ModuleType(prefix+'.'+Path(filename).stem)
        sys.modules[value.__name__] = value
        value.__dict__.update(np=np, cv2=cv2, hashlib=hashlib, dataclass=dataclass)
        value.__dict__.update(globals or {})
        tree = ast.parse((root/filename).read_text(encoding='utf8'))
        chosen = [node for node in tree.body if
            (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names)
            or (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets))]
        found = {node.name for node in chosen if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        found.update(t.id for node in chosen if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name))
        if found != set(names):
            raise ValueError('UM numerical function inventory changed')
        exec(compile(ast.Module(body=chosen, type_ignores=[]), str(root/filename), 'exec'), value.__dict__)
        return value
    annotation = selected('source_annotation.py', ['annotation_window'])
    pooling = selected('source_pooling.py', ['_axis_area_weights', 'pool_annotation'])
    sampling = selected('source_sampling.py', ['small_region_samples', 'region_samples'],
        dict(annotation_window=annotation.annotation_window, pool_annotation=pooling.pool_annotation, regional=regional))
    learning = selected('source_learning.py', ['SerializedPhaseHead'],
        dict(OpticalFeatureContract=optical.OpticalFeatureContract, Mapping=Mapping, Any=Any, Path=Path))
    return SimpleNamespace(logistic=logistic, joint=joint, optical=optical, regional=regional,
        annotation=annotation, sampling=sampling, learning=learning,
        gas=selected('source_gas.py', ['VERSION', 'Settings', 'confirm_gas']),
        guards=selected('source_guards.py', ['VERSION', 'Settings', 'guard_labels']),
        freegas=selected('source_freegas.py', ['VERSION', 'Settings', 'clean_labels']),
        source_binding=manifest, prediction_source_binding=prediction_binding)
