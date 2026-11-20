"""Provided plumbing. Paths are relative to this package, not your working directory."""
from pathlib import Path
import csv, json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
LEVELS=[f'T{i}' for i in range(1,13)]+['L1','L2','L3']
LABELS=list(range(8,23))
OUTPUT=ROOT/'outputs'
OUTPUT.mkdir(exist_ok=True)

def output_relative_path(value, output_root=OUTPUT):
    """Resolve a ShapeWorks path containing POSIX, Windows, or mixed separators."""
    return Path(output_root) / Path(str(value).replace('\\', '/'))

def manifest():
    with (ROOT/'data/cases.csv').open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))
def save_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2),encoding='utf-8')
def read_centres(case):
    points=np.asarray(json.loads((OUTPUT/'centroids'/f'{case}.json').read_text())['aligned_mm'])
    if points.shape!=(15,3) or not np.isfinite(points).all():
        raise ValueError(f'{case}: expected finite (15,3) aligned centres')
    return points
