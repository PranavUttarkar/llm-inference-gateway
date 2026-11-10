"""Verify published capture hashes and recompute the 32B counts and metrics."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmarks.analyze_32b import analyze


def main():
    checked = 0
    manifests = sorted((ROOT / 'results').glob('*/SHA256SUMS'))
    if len(manifests) != 3:
        raise ValueError('expected manifests for all three published studies')
    for manifest in manifests:
        for line in manifest.read_text(encoding='utf-8').splitlines():
            digest, name = line.split('  ', 1)
            path = (manifest.parent / name).resolve()
            if not path.is_relative_to(manifest.parent.resolve()):
                raise ValueError(f'capture path escapes study directory: {name}')
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f'capture hash mismatch: {path.relative_to(ROOT)}')
            checked += 1
    directory = ROOT / 'results' / '32b-admission'
    actual = analyze(directory)
    stored = json.loads((directory / 'analysis-32b.json').read_text(encoding='utf-8'))
    if actual != stored:
        raise ValueError('published 32B analysis differs from recomputed raw results')
    print(f'Verified {checked} captured files across 3 studies; 32B analysis matches raw records.')


if __name__ == '__main__':
    main()
