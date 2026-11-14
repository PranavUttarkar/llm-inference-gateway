"""Freeze an independent 96-question numerical GPU pilot after CPU variance check."""
import argparse
import copy
import json
from pathlib import Path

from research.batch_probe.workload import digest
from .numerical_gate import RULE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/cache-repair'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    root = args.root
    cpu = json.loads((root/'cpu-spec-v1.json').read_text())
    results = json.loads((root/'cpu-mechanism-v1/manifest.json').read_text())
    if results['status'] != 'complete':
        raise ValueError('finish the four prespecified CPU cases first')
    old_ids = {q['id'] for q in cpu['specification']['questions']}
    tokenizer = AutoTokenizer.from_pretrained(root/'model-1.5b', local_files_only=True)
    rows = [json.loads(line) for line in (root/'math500-test.jsonl').read_text(encoding='utf-8').splitlines()]
    selected = []
    for row in sorted(rows, key=lambda row: digest(row['unique_id'])):
        if row['unique_id'] in old_ids:
            continue
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': row['problem']}],
                                               tokenize=True, add_generation_prompt=True)
        tokens = prompt+tokenizer.encode(row['solution'], add_special_tokens=False)
        if len(tokens) >= 400:
            selected.append({'id': row['unique_id'], 'token_ids': tokens[:400], 'token_sha256': digest(tokens[:400]),
                             'origin': 'teacher_forced_reference_solution_not_model_generated'})
        if len(selected) == 96:
            break
    if len(selected) != 96:
        raise ValueError('not enough unused length-eligible source questions')
    base = copy.deepcopy(cpu['specification'])
    base.update(study='exploratory_gpu_numerical_v3', questions=selected,
                amendment='F: pre-GPU secondary contrast, robust effect summaries and numerical K8/V8 anchor',
                higher_precision_anchor={'key_bits': 8, 'value_bits': 8, 'path': 'streaming same geometry and sink/tail',
                                         'scope': 'numerical reference; accuracy cost unvalidated; not used to redefine gate'},
                secondary_contrast='mean KL(fresh-state/fresh-scales) minus KL(fresh-state/frozen-scales); '
                                   'paired source-question percentile 95% interval; positive favors frozen scales',
                robustness='median absolute benefit, signs, log(KEEP/frozen) with 1e-9 floor and clipping count; '
                           'leave-one-out mean range; all descriptive, no replacement of primary',
                device_assignment='alternating questions in sorted SHA256(source-id) order: q[0::2], q[1::2]',
                observed_allocation='two RTX 4090 24GB reported by nvidia-smi on supplied host; not the originally offered 5090s',
                numerical_gate=RULE,
                controls='independent blockwise BF16 histories twice; independent full-prefill BF16 histories twice; '
                         'same-cache probe repeat; per-layer KV errors; frozen-scale self-encoding identity',
                design='96 new source questions; within-question paired interventions; no task-accuracy claim',
                planning={'target_absolute_kl_difference_nats': .001, 'nominal_two_sided_alpha': .05,
                          'nominal_power': .8, 'cpu_sd_estimate_n4': .00340958347629935,
                          'normal_approximation_n': 92, 'chosen_n': 96,
                          'warning': 'Four-case variance is uncertain; this remains exploratory, not powered confirmation.'},
                allocation={'gpus': 2, 'maximum_wall_seconds': 7200, 'export_reserve_seconds': 900},
                primary_test='source-question paired mean KL difference; negative favors frozen-scale history reconstruction; '
                             'report all 96 effects and heterogeneity, not just a significance threshold')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for shard in (None, 0, 1):
        spec = copy.deepcopy(base)
        if shard is not None:
            spec.update(questions=selected[shard::2], shard=shard, global_question_count=96)
        path = args.output if shard is None else args.output.with_name(args.output.stem+f'-gpu{shard}'+args.output.suffix)
        with path.open('x', encoding='utf-8') as stream:
            json.dump({'sha256': digest(spec), 'specification': spec}, stream, indent=2)
            stream.write('\n')
        print(json.dumps({'path': str(path), 'sha256': digest(spec), 'questions': len(spec['questions'])}), flush=True)


if __name__ == '__main__':
    main()
