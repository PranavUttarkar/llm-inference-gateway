"""Stage pinned inputs for an optional model replay; does not download or run a model."""
import argparse
from pathlib import Path
import shutil

from .reproduce import DEFAULT_STUDY, verify_hashes


def stage(study, output):
    study, output = Path(study).resolve(), Path(output).resolve()
    verify_hashes(study)
    if output.is_relative_to(study) or study.is_relative_to(output):
        raise ValueError('staging output must not overlap published evidence')
    output.mkdir(parents=True, exist_ok=False)
    mapping = {'cpu.json': 'cpu-spec-v1.json', 'gpu.json': 'gpu-spec-v3.json',
               'gpu0.json': 'gpu-spec-v3-gpu0.json', 'gpu1.json': 'gpu-spec-v3-gpu1.json',
               'smoke.json': 'gpu-smoke-v3.json'}
    for source, name in mapping.items():
        shutil.copyfile(study/'specs'/source, output/name)
    for source, name in {'upstream-metadata.json': 'upstream-metadata-v1.json',
                         'model-weight-hash.json': 'model-weight-hash-v1.json',
                         'dataset-metadata.json': 'dataset-metadata-v1.json'}.items():
        shutil.copyfile(study/'evidence'/source, output/name)
    shutil.copytree(study/'evidence/upstream', output/'upstream-source')
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--output', type=Path, default=Path('artifacts/cache-repair'))
    args = parser.parse_args()
    print(f'Staged pinned inputs at {stage(args.study, args.output)}. No model execution started.')


if __name__ == '__main__':
    main()
