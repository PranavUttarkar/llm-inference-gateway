"""Package a source/workload snapshot for transfer; never includes credentials."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

from .design import CONDITIONS, schedule
from .run import MODEL, REVISION, ROOT, source_hashes
from .workload import digest, validate_workload


def freeze(workload_path, output):
    if output.exists():
        raise FileExistsError(output)
    workload_bytes = workload_path.read_bytes()
    workload = json.loads(workload_bytes)
    rows = workload['questions']
    validate_workload(rows)
    if digest(rows) != workload['sha256']:
        raise ValueError('workload hash mismatch')
    files = {relative: (ROOT / relative).read_bytes() for relative in source_hashes()}
    files['workloads/batch-probe-pilot-v1.json'] = workload_bytes
    plan = {'purpose': 'exploratory_pilot', 'questions': len(rows), 'model': MODEL,
            'revision': REVISION, 'dtype': 'bfloat16', 'attention': 'sdpa', 'seed': 1701,
            'conditions': CONDITIONS, 'workload_sha256': digest(rows),
            'schedule_sha256': digest(schedule([q['id'] for q in rows], 1701)),
            'files_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in files.items()}}
    files['BUNDLE.json'] = (json.dumps(plan, indent=2) + '\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Stable names, metadata, and gzip timestamp make repeated freezes byte-identical.
    with output.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name, value in sorted(files.items()):
                info = tarfile.TarInfo(name)
                info.size = len(value)
                info.mtime = 0
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(value))
    return {'bundle': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'files': len(files), 'questions': len(rows), 'workload_sha256': digest(rows),
            'schedule_sha256': plan['schedule_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.workload, args.output)
    with Path(str(args.output)+'.sha256.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
