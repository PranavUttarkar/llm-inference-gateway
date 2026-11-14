"""Fixed-batch PyTorch probe. CUDA runs are evidence; fixtures are plumbing only."""
import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .design import CONDITIONS, padded_batch, schedule
from .workload import LABELS, canonical, digest, validate_workload

ROOT = Path(__file__).resolve().parents[2]
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def atomic_manifest(path, manifest):
    """Bounded retries for transient Windows/OneDrive replacement locks."""
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.05 * (2 ** attempt))


def softmax(logits):
    peak = max(logits)
    values = [math.exp(value - peak) for value in logits]
    return [value / sum(values) for value in values]


def source_hashes():
    files = [path for path in Path(__file__).parent.iterdir() if path.suffix in ('.py', '.md', '.txt')]
    files += [ROOT / 'RESEARCH.md', ROOT / 'tests' / 'test_batch_probe.py']
    import hashlib
    return {str(path.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def environment():
    packages = {}
    for name in ('torch', 'transformers', 'tokenizers', 'numpy', 'scipy', 'scikit-learn', 'safetensors'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {'python': sys.version, 'platform': platform.platform(), 'packages': packages}
    try:
        info['nvidia_smi'] = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=False).stdout
    except OSError:
        info['nvidia_smi'] = None
    return info


class Fixture:
    """Deliberately simulated outcomes. Must never be reported as model evidence."""
    def __init__(self, scenario):
        self.scenario = scenario
        self.metadata = {'backend': 'fixture', 'scenario': scenario, 'scientific_evidence': False}

    def measure(self, question, condition):
        # No gold labels: plumbing fixture signal is numerical, not a claim of error prediction.
        rng = random.Random(int(digest(question['id'])[:12], 16))
        logits = [rng.uniform(-2, 2) for _ in LABELS]
        if self.scenario == 'sensitive' and condition not in ('solo', 'solo_repeat'):
            jitter = random.Random(int(digest([question['id'], condition.replace('_repeat', '')])[:12], 16))
            logits = [value + jitter.uniform(-.2, .2) for value in logits]
        return {'choice_logits': logits, 'raw_top_token_id': logits.index(max(logits)),
                'raw_top_token_text': LABELS[logits.index(max(logits))],
                'raw_top_token_probability': max(softmax(logits)),
                'elapsed_seconds': .001, 'target_tokens': len(question['prompt']),
                'batch_size': CONDITIONS[condition]['batch'],
                'padded_length': len(question['prompt']) + CONDITIONS[condition]['extra_padding'],
                'all_input_tokens': len(question['prompt']) * CONDITIONS[condition]['batch'],
                'peak_allocated_bytes': 0, 'target_input_sha256': digest(question['prompt']),
                'target_choice_mass': 1.0,
                'full_vocab_entropy': -sum(p*math.log(p) for p in softmax(logits)),
                'full_vocab_top2_margin': sorted(softmax(logits), reverse=True)[0] - sorted(softmax(logits), reverse=True)[1]}


class TorchProbe:
    def __init__(self, args):
        # Keep imports lazy: local dry runs and CI need no model packages or GPU.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for model evidence; use --fixture for plumbing only')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one GPU through CUDA_VISIBLE_DEVICES for the first pilot')
        if args.dtype == 'bfloat16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('BF16 is unsupported on this GPU; do not silently change the frozen primary arm')
        self.torch = torch
        torch.manual_seed(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, torch_dtype=getattr(torch, args.dtype),
            attn_implementation=args.attention, trust_remote_code=False).eval().to('cuda')
        self.labels = [self.tokenizer.encode(label, add_special_tokens=False) for label in LABELS]
        if any(len(tokens) != 1 for tokens in self.labels) or len({tokens[0] for tokens in self.labels}) != 4:
            raise ValueError('A-D must be four distinct single tokens for this protocol')
        self.label_ids = [tokens[0] for tokens in self.labels]
        self.pad = self.tokenizer.pad_token_id
        if self.pad is None:
            self.pad = self.tokenizer.eos_token_id
        if not isinstance(self.pad, int):
            raise ValueError('tokenizer requires a single integer padding ID')
        self.max_tokens = args.max_input_tokens
        self.cache = {}
        self.metadata = {'backend': 'transformers_fixed_batch', 'scientific_evidence': True,
                         'gpu': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                         'label_ids': self.label_ids, 'pad_id': self.pad,
                         'model_commit': getattr(self.model.config, '_commit_hash', None),
                         'attention': args.attention, 'dtype': args.dtype,
                         'actual_attention': self.model.config._attn_implementation,
                         'actual_parameter_dtype': str(next(self.model.parameters()).dtype),
                         'bf16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                         'fp16_reduced_precision_reduction': torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                         'numerical_environment': {key: os.environ.get(key) for key in
                             ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE', 'CUDA_VISIBLE_DEVICES')},
                         'prefix_cache': False, 'use_cache': False, 'tf32': False}

    def inputs(self, question):
        if question['id'] not in self.cache:
            target = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': question['prompt']}], tokenize=True, add_generation_prompt=True)
            if len(target) > self.max_tokens:
                raise ValueError(f"prompt {question['id']} has {len(target)} tokens; never silently truncate")
            companions = [self.tokenizer.encode(f'Independent sequence {i}: ' +
                         'Read these unrelated words carefully. ' * 12, add_special_tokens=False)[:len(target)]
                          for i in range(7)]
            self.cache[question['id']] = (target, companions)
        return self.cache[question['id']]

    def measure(self, question, condition):
        torch = self.torch
        target, companions = self.inputs(question)
        ids, masks, position, token_position = padded_batch(target, condition, self.pad, companions)
        input_ids = torch.tensor(ids, dtype=torch.long, device='cuda')
        attention_mask = torch.tensor(masks, dtype=torch.long, device='cuda')
        position_ids = attention_mask.cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                                 position_ids=position_ids, use_cache=False, return_dict=True)
            # Gather at target's last real token, NOT at the end of padded storage.
            logits = outputs.logits[position, token_position].float()
            choice_logits = logits[self.label_ids].cpu().tolist()
            probabilities = torch.softmax(logits, dim=-1)
            choice_mass = float(probabilities[self.label_ids].sum().item())
            top_id = int(logits.argmax().item())
            top_probability = float(probabilities[top_id].item())
            top2 = probabilities.topk(2).values
            full_margin = float((top2[0] - top2[1]).item())
            full_entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum().item())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        return {'choice_logits': choice_logits, 'raw_top_token_id': top_id,
                'raw_top_token_text': self.tokenizer.decode([top_id]),
                'raw_top_token_probability': top_probability, 'elapsed_seconds': elapsed,
                'target_tokens': len(target), 'batch_size': len(ids), 'padded_length': len(ids[0]),
                'all_input_tokens': sum(map(sum, masks)),
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'target_input_sha256': digest(target), 'target_choice_mass': choice_mass,
                'full_vocab_entropy': full_entropy, 'full_vocab_top2_margin': full_margin}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workload', required=True)
    parser.add_argument('--output', required=True, help='new directory; overwriting is refused')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--dtype', choices=('bfloat16', 'float16', 'float32'), default='bfloat16')
    parser.add_argument('--attention', choices=('sdpa', 'eager'), default='sdpa')
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--max-input-tokens', type=int, default=768)
    parser.add_argument('--limit', type=int, default=0, help='smoke test only; never merge with full pilot')
    parser.add_argument('--fixture', choices=('stable', 'sensitive'))
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in '0123456789abcdef' for c in args.revision):
        parser.error('revision must be a full lowercase commit SHA')
    if args.limit < 0 or args.max_input_tokens < 1:
        parser.error('limit must be nonnegative; max-input-tokens must be positive')
    workload = json.loads(Path(args.workload).read_text(encoding='utf-8'))
    rows = workload['questions']
    validate_workload(rows)
    if digest(rows) != workload['sha256']:
        raise ValueError('workload content does not match frozen hash')
    if args.limit:
        rows = rows[:args.limit]
    jobs = schedule([q['id'] for q in rows], args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': 1, 'status': 'initializing', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                'parameters': vars(args), 'purpose': 'fixture' if args.fixture else ('smoke' if args.limit else 'pilot'),
                'workload_sha256': workload['sha256'], 'selected_questions_sha256': digest(rows),
                'source_sha256': source_hashes(), 'environment': environment(),
                'schedule_sha256': digest(jobs), 'conditions': CONDITIONS, 'completed_jobs': 0}
    def save():
        atomic_manifest(output / 'manifest.json', manifest)
    save()
    # Store the exact runner/protocol/tests alongside evidence, not just mutable repo paths.
    for relative in manifest['source_sha256']:
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / relative).read_bytes())
    (output / 'schedule.json').write_text(json.dumps(jobs, indent=2) + '\n', encoding='utf-8')
    (output / 'workload.json').write_text(json.dumps({'questions': rows, 'sha256': digest(rows)}, indent=2) + '\n', encoding='utf-8')
    try:
        probe = Fixture(args.fixture) if args.fixture else TorchProbe(args)
        manifest['backend'] = probe.metadata
        # Warm the largest shape, then singleton; excluded from results but charged separately.
        warm_start = time.perf_counter()
        for condition in ('batch8_padded', 'solo'):
            probe.measure(rows[0], condition)
        manifest['warmup_wall_seconds'] = time.perf_counter() - warm_start
        manifest['status'] = 'running'
        save()
        lookup = {q['id']: q for q in rows}
        started = time.perf_counter()
        with (output / 'records.jsonl').open('x', encoding='utf-8') as stream:
            for order, job in enumerate(jobs):
                result = probe.measure(lookup[job['question_id']], job['condition'])
                result.update(job, order=order)
                stream.write(canonical(result) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                manifest['completed_jobs'] = order + 1
                if (order + 1) % 50 == 0:
                    save()
                    print(f'{order + 1}/{len(jobs)} jobs', flush=True)
        manifest.update(status='complete', measurement_wall_seconds=time.perf_counter() - started)
        import hashlib
        manifest['records_sha256'] = hashlib.sha256((output / 'records.jsonl').read_bytes()).hexdigest()
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        save()
    print(json.dumps({'status': manifest['status'], 'purpose': manifest['purpose'],
                      'completed_jobs': manifest['completed_jobs'], 'output': str(output)}, indent=2))


if __name__ == '__main__':
    main()
