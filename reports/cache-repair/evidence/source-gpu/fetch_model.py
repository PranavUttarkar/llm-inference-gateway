"""Fetch and hash the single pinned pilot model; no remote code is enabled."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/cache-repair'))
    args = parser.parse_args()
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    from huggingface_hub import snapshot_download
    meta = json.loads((args.root/'upstream-metadata-v1.json').read_text())
    expected = json.loads((args.root/'model-weight-hash-v1.json').read_text())
    print('Fetching exact pinned model snapshot via HTTP', flush=True)
    snapshot_download(repo_id=meta['model_id'], revision=meta['model_revision'], max_workers=2,
                      allow_patterns=['*.json', '*.safetensors', 'tokenizer.model', 'LICENSE', 'README.md'],
                      local_dir=args.root/'model-1.5b')
    weights = args.root/'model-1.5b/model.safetensors'
    h = hashlib.sha256()
    with weights.open('rb') as stream:
        while chunk := stream.read(16*1024*1024):
            h.update(chunk)
    if h.hexdigest() != expected['sha256'] or weights.stat().st_size != expected['bytes']:
        raise ValueError('pinned checkpoint verification failed')
    print(json.dumps({'verified': str(weights), 'sha256': h.hexdigest()}), flush=True)


if __name__ == '__main__':
    main()
