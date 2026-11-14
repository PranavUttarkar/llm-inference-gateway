"""Pinned pretrained-model teacher-forced cache mechanism pilot, CPU or CUDA.

No accuracy, packed-attention speedup or physical cache-memory saving is claimed.
The primary comparison isolates history reconstruction with frozen scale params.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

from research.batch_probe.workload import digest
from .reference_quant import Quantizer, load_references


def prepare(root, output):
    from transformers import AutoTokenizer
    meta = json.loads((root/'upstream-metadata-v1.json').read_text())
    dataset = json.loads((root/'dataset-metadata-v1.json').read_text())
    tokenizer = AutoTokenizer.from_pretrained(root/'model-1.5b', local_files_only=True, trust_remote_code=False)
    rows = [json.loads(line) for line in (root/'math500-test.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = []
    for row in sorted(rows, key=lambda row: digest(row['unique_id'])):
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['problem']}],
                                               tokenize=True, add_generation_prompt=True)
        tokens = prompt + tokenizer.encode(row['solution'], add_special_tokens=False)
        if len(tokens) >= 400:
            selected.append({'id': row['unique_id'], 'token_ids': tokens[:400], 'token_sha256': digest(tokens[:400]),
                             'origin': 'teacher_forced_reference_solution_not_model_generated'})
        if len(selected) == 4:
            break
    if len(selected) != 4:
        raise ValueError('not enough eligible length-filtered reference traces')
    spec = {'study': 'exploratory_mechanistic_cpu_v1', 'model_id': meta['model_id'],
            'model_revision': meta['model_revision'], 'dataset': dataset,
            'dataset_file_sha256': hashlib.sha256((root/'math500-test.jsonl').read_bytes()).hexdigest(),
            'kvarn_commit': meta['kvarn_commit'], 'questions': selected,
            'prefix_tokens': 384, 'probe_tokens': 16, 'pseudo_decode_block': 32,
            'kv_key_bits': 4, 'kv_value_bits': 2, 'group': 128, 'sink_tokens': 128,
            'weight_dtype': 'bfloat16', 'rotation_dtype': 'float32', 'scale_storage_dtype': 'float16',
            'seed': 0, 'attention': 'eager',
            'primary': 'per-question mean KL(reference_stream || frozen_scale_reconstructed_history) '
                       'minus mean KL(reference_stream || keep_compressed_history); negative favors repair',
            'scope': 'reference-text, teacher-forced blockwise numerical emulation; no deployability/accuracy claim'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'sha256': digest(spec), 'specification': spec}, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'prepared': str(output), 'sha256': digest(spec), 'ids': [q['id'] for q in selected]}, indent=2))


def clone_cache(cache):
    from transformers import DynamicCache
    return DynamicCache.from_legacy_cache(tuple((k.clone(), v.clone()) for k, v in cache))


def teacher_force(model, token_ids, block, quantizer=None):
    import torch
    from transformers import DynamicCache
    cache, raw = DynamicCache(), None
    device = model.get_input_embeddings().weight.device
    if device.type == 'meta':
        device = torch.device('cpu')
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start_time = time.perf_counter()
    for start in range(0, len(token_ids), block):
        ids = torch.tensor([token_ids[start:start+block]], device=device)
        output = model(input_ids=ids, past_key_values=cache, use_cache=True, logits_to_keep=1)
        cache = output.past_key_values
        count = ids.shape[-1]
        current = tuple((k[:,:,-count:].clone(), v[:,:,-count:].clone())
                        for k, v in zip(cache.key_cache, cache.value_cache))
        if raw is None:
            raw = current
        else:
            raw = tuple((torch.cat((old[0], new[0]), -2), torch.cat((old[1], new[1]), -2))
                        for old, new in zip(raw, current))
        if quantizer is not None:
            values, _, _ = quantizer.apply(raw)
            cache = clone_cache(values)
        del output
    values = tuple(zip(cache.key_cache, cache.value_cache))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return values, raw, time.perf_counter()-start_time


def probe(model, cache_values, tokens):
    import torch
    cache = clone_cache(cache_values)
    device = cache_values[0][0].device
    out = model(input_ids=torch.tensor([tokens], device=device), past_key_values=cache,
                use_cache=True, logits_to_keep=len(tokens))
    # Position i predicts the token after probe token i; all arms receive identical
    # fixed continuation inputs. Primary KL averages these 16 aligned distributions.
    return out.logits[0].float().log_softmax(-1).cpu()


def kv_errors(values, reference):
    result = []
    for layer, (pair, ref_pair) in enumerate(zip(values, reference)):
        row = {'layer': layer}
        for kind, value, target in zip(('k', 'v'), pair, ref_pair):
            denominator = target.float().square().sum().item()
            row[kind+'_relative_squared_l2'] = (value.float()-target.float()).square().sum().item()/max(denominator, 1e-30)
        result.append(row)
    return result


def kl_positions(reference_logp, candidate_logp):
    """Preserve the CPU-v1 metric; identical aligned full-vocabulary inputs."""
    return (reference_logp.exp().double() *
            (reference_logp.double()-candidate_logp.double())).sum(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/cache-repair'))
    parser.add_argument('--prepare', type=Path)
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--disk-offload', action='store_true')
    args = parser.parse_args()
    # Optional accelerate installation is isolated from the user's environment.
    dependency_path = args.root/'python-deps'
    if dependency_path.exists():
        sys.path.insert(0, str(dependency_path.resolve()))
    if args.prepare:
        prepare(args.root, args.prepare)
        return
    if not args.spec or not args.output:
        parser.error('provide --prepare, or --spec and --output')
    import torch
    from transformers import AutoModelForCausalLM
    if os.environ.get('KVARN_RTN_QUANTILE') not in (None, '', '0'):
        raise ValueError('pilot freezes min/max RTN; unset KVARN_RTN_QUANTILE')
    payload = json.loads(args.spec.read_text())
    spec = payload['specification']
    if digest(spec) != payload['sha256']:
        raise ValueError('specification hash mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'initializing', 'spec_sha256': payload['sha256'], 'specification': spec,
                'mode': 'numerical_emulation', 'device': args.device, 'torch': torch.__version__,
                'cuda_runtime': torch.version.cuda,
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'disk_offload': args.disk_offload, 'results': [],
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}}
    def save():
        (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    save()
    for name, expected in manifest['source_sha256'].items():
        data = (Path(__file__).parent/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('source changed during snapshot')
        destination = args.output/'source'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
    try:
        torch.set_num_threads(4)
        torch.manual_seed(spec['seed'])
        if args.device == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA unavailable')
            torch.backends.cuda.matmul.allow_tf32 = False
            manifest['gpu'] = torch.cuda.get_device_name(0)
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation='eager', trust_remote_code=False,
                      local_files_only=True, low_cpu_mem_usage=True)
        if args.disk_offload:
            # Keep only the small norm resident; offload hooks stream one module
            # at a time. This changes host runtime, never model/caching numerics.
            mapping = {'model.embed_tokens': 'disk', 'model.norm': 'cpu', 'lm_head': 'disk'}
            mapping.update({f'model.layers.{i}': 'disk' for i in range(28)})
            kwargs.update(device_map=mapping, offload_folder=str(args.root/'offloaded-weights'), offload_state_dict=True)
        model = AutoModelForCausalLM.from_pretrained(args.root/'model-1.5b', **kwargs).eval()
        if not args.disk_offload:
            model.to(args.device)
        official = load_references(args.root/'upstream-source')
        manifest['official_source_sha256'] = official.source_sha256
        quantizer = Quantizer(official, spec['kv_key_bits'], spec['kv_value_bits'], spec['group'], spec['sink_tokens'])
        anchor = Quantizer(official, 8, 8, spec['group'], spec['sink_tokens']) if spec.get('higher_precision_anchor') else None
        manifest['status'] = 'running'
        save()
        with torch.inference_mode():
            for question in spec['questions']:
                started = time.perf_counter()
                if args.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                ids = question['token_ids']
                prefix, continuation = ids[:spec['prefix_tokens']], ids[spec['prefix_tokens']:]
                print(f"{question['id']}: uncompressed block reference", flush=True)
                stream_ref, clean_raw, ref_seconds = teacher_force(model, prefix, spec['pseudo_decode_block'])
                print(f"{question['id']}: KVarN compressed block history", flush=True)
                keep, damaged_raw, keep_seconds = teacher_force(model, prefix, spec['pseudo_decode_block'], quantizer)
                print(f"{question['id']}: full-prefill geometry control", flush=True)
                prefill_ref, prefill_raw, prefill_seconds = teacher_force(model, prefix, len(prefix))
                # Independent histories, not merely another probe of one cache.
                stream_repeat, _, stream_repeat_seconds = teacher_force(model, prefix, spec['pseudo_decode_block'])
                prefill_repeat, _, prefill_repeat_seconds = teacher_force(model, prefix, len(prefix))
                anchor_values, anchor_bytes, anchor_seconds = None, None, None
                if anchor is not None:
                    print(f"{question['id']}: fixed K8/V8 streaming numerical anchor", flush=True)
                    anchor_values, anchor_raw, anchor_seconds = teacher_force(model, prefix, spec['pseudo_decode_block'], anchor)
                    _, _, anchor_bytes = anchor.apply(anchor_raw)
                # Scale/state 2x2: C=damaged history, S=clean block reference.
                _, old_scales, keep_bytes = quantizer.apply(damaged_raw)
                _, clean_scales, _ = quantizer.apply(clean_raw)
                self_reencoded, _, _ = quantizer.apply(damaged_raw, frozen=old_scales)
                self_error = max((left-right).abs().max().item()
                                 for a,b in zip(keep,self_reencoded) for left,right in zip(a,b))
                if self_error != 0:
                    raise ValueError('frozen-scale self-encoding failed; do not interpret decomposition')
                state_only, _, _ = quantizer.apply(clean_raw, frozen=old_scales)
                scales_only, _, _ = quantizer.apply(damaged_raw, frozen=clean_scales)
                repaired, _, repair_bytes = quantizer.apply(clean_raw)
                full_rebuilt, _, _ = quantizer.apply(prefill_raw)
                print(f"{question['id']}: fixed-continuation KL and per-layer cache errors", flush=True)
                reference_logp = probe(model, stream_ref, continuation)
                arms = {'keep': keep, 'state_only_frozen_scales': state_only,
                        'scales_only_oracle_diagnostic': scales_only, 'state_and_scales': repaired,
                        'uncompressed_full_prefill_geometry': prefill_ref,
                        'uncompressed_repeat_control': stream_ref,
                        'uncompressed_stream_rebuild_repeat': stream_repeat,
                        'uncompressed_prefill_rebuild_repeat': prefill_repeat,
                        'full_prefill_requant_fresh_scales': full_rebuilt}
                if anchor_values is not None:
                    arms['higher_precision_k8v8_stream_anchor'] = anchor_values
                output = {'question_id': question['id'], 'token_sha256': question['token_sha256'],
                          'reference_seconds': ref_seconds, 'keep_seconds': keep_seconds,
                          'prefill_seconds': prefill_seconds,
                          'stream_repeat_seconds': stream_repeat_seconds,
                          'prefill_repeat_seconds': prefill_repeat_seconds,
                          'anchor_seconds': anchor_seconds,
                          'frozen_scale_self_reencode_max_abs': self_error,
                          'logical_packed_bytes': {'keep': keep_bytes, 'repair': repair_bytes}, 'arms': {}}
                if anchor_bytes is not None:
                    output['logical_packed_bytes']['k8v8_anchor'] = anchor_bytes
                output['raw_damaged_kv_vs_stream_reference'] = kv_errors(damaged_raw, stream_ref)
                output['raw_full_prefill_kv_vs_stream_reference'] = kv_errors(prefill_raw, stream_ref)
                arm_logps = {}
                for name, values in arms.items():
                    logp = probe(model, values, continuation)
                    arm_logps[name] = logp
                    divergence = kl_positions(reference_logp, logp)
                    output['arms'][name] = {'kl_nats_per_position': divergence.tolist(),
                                            'mean_kl_nats': divergence.mean().item(),
                                            'per_layer_kv_errors': kv_errors(values, stream_ref)}
                prefill_repeat_kl = kl_positions(arm_logps['uncompressed_full_prefill_geometry'],
                                                 arm_logps['uncompressed_prefill_rebuild_repeat'])
                output['independent_prefill_repeat'] = {
                    'kl_nats_per_position': prefill_repeat_kl.tolist(),
                    'mean_kl_nats': prefill_repeat_kl.mean().item(),
                    'per_layer_kv_errors': kv_errors(prefill_repeat, prefill_ref)}
                output['paired_primary_delta'] = (output['arms']['state_only_frozen_scales']['mean_kl_nats']-
                                                  output['arms']['keep']['mean_kl_nats'])
                output['wall_seconds'] = time.perf_counter()-started
                if args.device == 'cuda':
                    output['diagnostic_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    output['diagnostic_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                manifest['results'].append(output)
                save()
                print(json.dumps({'id': question['id'], 'delta': output['paired_primary_delta'],
                                  'seconds': output['wall_seconds']}), flush=True)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
