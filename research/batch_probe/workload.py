"""Seeded, exactly scored multiple-choice pilot tasks; no model is involved."""
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

LABELS = 'ABCD'
FAMILIES = ('arithmetic', 'graph', 'state', 'relational')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def solve(family, data):
    """Ground truth derived from structured inputs, never from model text."""
    if family == 'arithmetic':
        return ((data['a'] * data['b'] + data['c']) // data['d']) % data['m']
    if family == 'graph':
        # Enumerate every simple directed path; tiny graphs make this exact.
        edges = {(a, b): w for a, b, w in data['edges']}
        candidates = []
        for n in range(5):
            for middle in itertools.permutations((1, 2, 3, 4), n):
                path = (0,) + middle + (5,)
                if all(pair in edges for pair in zip(path, path[1:])):
                    candidates.append(sum(edges[pair] for pair in zip(path, path[1:])))
        return min(candidates)
    if family == 'state':
        value = data['initial']
        for op, amount in data['operations']:
            if op == 'add':
                value += amount
            elif op == 'multiply':
                value *= amount
            elif value >= amount:
                value -= amount
        return value
    if family == 'relational':
        return sum(row['units'] * row['price'] for row in data['rows']
                   if row['region'] == data['region'] and row['units'] >= data['threshold'])
    raise ValueError('unknown task family: ' + family)


def make_problem(family, rng):
    if family == 'arithmetic':
        data = dict(a=rng.randint(12, 85), b=rng.randint(11, 55), c=rng.randint(1, 80),
                    d=rng.randint(2, 9), m=rng.randint(7, 31))
        text = (f"Compute (({data['a']} * {data['b']} + {data['c']}) // {data['d']}) "
                f"modulo {data['m']}. Here // means floor division of integers.")
    elif family == 'graph':
        edges = [(a, b, rng.randint(1, 20)) for a in range(6) for b in range(a + 1, 6)
                 if b == a + 1 or rng.random() < .45]
        data = {'edges': edges}
        text = ('Find the minimum total weight of a directed path from vertex 0 to vertex 5. '
                'Only these edges exist, listed as (source, destination, weight): ' + str(edges))
    elif family == 'state':
        data = {'initial': rng.randint(2, 15), 'operations': [
            (rng.choice(('add', 'multiply', 'subtract_if_possible')), rng.randint(2, 8))
            for _ in range(7)]}
        text = (f"A register starts at {data['initial']}. Execute these operations in order: "
                f"{data['operations']}. add increases it by the amount; multiply multiplies it "
                'by the amount; subtract_if_possible subtracts the amount only when the '
                'current value is at least that amount, otherwise does nothing. Final value?')
    elif family == 'relational':
        data = {'region': rng.choice(('north', 'south')), 'threshold': rng.randint(3, 7),
                'rows': [dict(region=rng.choice(('north', 'south')), units=rng.randint(1, 9),
                              price=rng.randint(2, 19)) for _ in range(8)]}
        text = (f"Sales rows: {data['rows']}. Sum units * price over rows whose region is "
                f"{data['region']} and whose units are at least {data['threshold']}.")
    else:
        raise ValueError(family)
    return data, text


def build_workload(per_family=60, seed=20261113):
    if per_family < 12 or per_family % 4:
        raise ValueError('per-family must be >=12 and divisible by 4')
    rng = random.Random(seed)
    rows, seen = [], set()
    for family in FAMILIES:
        # Pilot splits share task templates; they are NOT an out-of-template replication.
        splits = ['development'] * (per_family // 2) + ['evaluation'] * (per_family - per_family // 2)
        rng.shuffle(splits)
        for index in range(per_family):
            while True:
                data, question = make_problem(family, rng)
                group = digest({'family': family, 'data': data})
                if group not in seen:
                    seen.add(group)
                    break
            answer = solve(family, data)
            wrong = set()
            while len(wrong) < 3:
                candidate = max(0, answer + rng.randint(-max(5, abs(answer) // 3), max(5, abs(answer) // 3)))
                if candidate != answer:
                    wrong.add(candidate)
            options = sorted(wrong)
            rng.shuffle(options)
            correct_index = index % 4
            options.insert(correct_index, answer)
            prompt = (question + '\n' + '\n'.join(f'{label}. {value}' for label, value in zip(LABELS, options))
                      + '\nReply with exactly one letter: A, B, C, or D.\nAnswer:')
            rows.append({'id': group[:20], 'group_id': group, 'family': family,
                         'split': splits[index], 'prompt': prompt, 'prompt_sha256': digest(prompt),
                         'options': options, 'correct_label': LABELS[correct_index], 'data': data})
    rng.shuffle(rows)
    return rows


def validate_workload(rows):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('empty workload or duplicate IDs')
    groups = {}
    for row in rows:
        if row['split'] not in ('development', 'evaluation'):
            raise ValueError('unknown split')
        if groups.setdefault(row['group_id'], row['split']) != row['split']:
            raise ValueError('source-question group leaked across splits')
        if digest(row['prompt']) != row['prompt_sha256']:
            raise ValueError('prompt hash mismatch')
        options = row['options']
        if len(options) != 4 or len(set(options)) != 4 or row['correct_label'] not in LABELS:
            raise ValueError('invalid choices')
        if options[LABELS.index(row['correct_label'])] != solve(row['family'], row['data']):
            raise ValueError('incorrect oracle label')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--per-family', type=int, default=60)
    parser.add_argument('--seed', type=int, default=20261113)
    args = parser.parse_args()
    rows = build_workload(args.per_family, args.seed)
    validate_workload(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'schema': 1, 'purpose': 'exploratory_pilot', 'seed': args.seed,
                   'sha256': digest(rows), 'questions': rows}, stream, indent=2)
        stream.write('\n')
    print(f'Wrote {len(rows)} questions: {output}; sha256={digest(rows)}')


if __name__ == '__main__':
    main()
