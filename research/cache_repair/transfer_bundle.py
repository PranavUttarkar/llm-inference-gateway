"""Create a reviewable transfer snapshot without credentials or model weights."""
import argparse
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--specs', type=Path, nargs='+', required=True)
    parser.add_argument('--root', type=Path, default=Path('artifacts/cache-repair'), help='staged input directory')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite a transfer bundle')
    root = Path(__file__).resolve().parents[2]
    files = [root/'research/__init__.py', root/'research/batch_probe/__init__.py',
             root/'research/batch_probe/workload.py']
    files += [p for p in (root/'research/cache_repair').iterdir() if p.suffix in ('.py', '.txt', '.sh')]
    contents = {p.relative_to(root).as_posix(): p.read_bytes() for p in files}
    data = args.root.resolve()
    inputs = list((data/'upstream-source').rglob('*.py'))+[data/'upstream-source/LICENSE']
    inputs += [data/'upstream-metadata-v1.json', data/'model-weight-hash-v1.json', data/'dataset-metadata-v1.json']
    for path in inputs:
        contents['artifacts/cache-repair/'+path.relative_to(data).as_posix()] = path.read_bytes()
    for path in args.specs:
        name = 'artifacts/cache-repair/'+path.name
        if name in contents:
            raise ValueError('duplicate input name')
        contents[name] = path.read_bytes()
    manifest = {'sha256': {name: hashlib.sha256(value).hexdigest() for name,value in contents.items()},
                'scope': 'KVarN-reference numerical emulation; not a native packed-attention benchmark',
                'excludes': ['model weights', 'credentials', 'HF authentication/cache files', 'offloaded weights']}
    contents['TRANSFER.json'] = (json.dumps(manifest, indent=2)+'\n').encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('xb') as raw, gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for name,value in sorted(contents.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(value), 0o644, 0
                archive.addfile(info, io.BytesIO(value))
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    Path(str(args.output)+'.sha256').write_text(checksum+'\n', encoding='utf-8')
    print(json.dumps({'file': str(args.output), 'sha256': checksum, 'files': len(contents),
                      'bytes': args.output.stat().st_size}, indent=2))


if __name__ == '__main__':
    main()
