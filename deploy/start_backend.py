"""Prepare a pinned Hugging Face snapshot and start NVIDIA's bundled frontend."""
import json
import os
import pathlib
import sys

MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
REVISION = '989aa7980e4cf806f80c7fef2b1adb7bc71aa306'


def model_config(snapshot, tp, max_num_seqs=None):
    if tp not in (1, 2):
        raise ValueError('TP_SIZE must be 1 or 2')
    config = {'model': snapshot, 'tokenizer': snapshot, 'tensor_parallel_size': tp,
              'max_model_len': 2048, 'gpu_memory_utilization': 0.8,
              'disable_log_requests': True}
    if max_num_seqs is not None:
        if max_num_seqs < 1:
            raise ValueError('MAX_NUM_SEQS must be positive')
        config['max_num_seqs'] = max_num_seqs
    return config


def main():
    from huggingface_hub import snapshot_download
    tp = int(os.environ.get('TP_SIZE', '1'))
    model = os.environ.get('MODEL_ID', MODEL)
    revision = os.environ.get('MODEL_REVISION', REVISION)
    max_num_seqs = int(os.environ['MAX_NUM_SEQS']) if os.environ.get('MAX_NUM_SEQS') else None
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('MODEL_REVISION must be a full lowercase commit SHA')
    model_config('/placeholder', tp, max_num_seqs)  # validate before downloading
    snapshot = snapshot_download(repo_id=model, revision=revision)
    repo = pathlib.Path('/tmp/gateway-models')
    model_dir = repo / 'qwen' / '1'
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir.parent / 'config.pbtxt').write_text('backend: "vllm"\ninstance_group [{kind: KIND_MODEL}]\n')
    (model_dir / 'model.json').write_text(json.dumps(model_config(snapshot, tp, max_num_seqs)))
    os.chdir('/opt/tritonserver/python/openai')
    os.execv(sys.executable, [sys.executable, 'openai_frontend/main.py', '--model-repository', str(repo),
                              '--tokenizer', snapshot, '--host', '0.0.0.0'])


if __name__ == '__main__':
    main()
