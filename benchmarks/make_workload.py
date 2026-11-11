"""Generate longer synthetic prompts, measured with a local model chat template."""
import argparse
import hashlib
import json
from pathlib import Path


def count_tokens(tokenizer, prompt):
    return len(tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                                             tokenize=True, add_generation_prompt=True))


def extend_prompt(tokenizer, prompt, target, tolerance=8):
    text = prompt + '\n'
    tokens = tokenizer.encode(text, add_special_tokens=False)
    while len(tokens) < target * 2:
        text *= 2
        tokens = tokenizer.encode(text, add_special_tokens=False)
    low, high, best = 1, min(len(tokens), target * 2), None
    while low <= high:
        middle = (low + high) // 2
        candidate = tokenizer.decode(tokens[:middle], skip_special_tokens=True)
        count = count_tokens(tokenizer, candidate)
        if count <= target:
            if best is None or count > best[1]:
                best = candidate, count
            low = middle + 1
        else:
            high = middle - 1
    if best is None or target - best[1] > tolerance:
        raise ValueError('could not meet requested chat-template token range')
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer', required=True, help='local tokenizer snapshot matching the serving model revision')
    p.add_argument('--source', default='workloads/prompts-512.json')
    p.add_argument('--target-tokens', type=int, default=1536)
    p.add_argument('--tolerance', type=int, default=8)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if args.target_tokens < 1 or args.tolerance < 0:
        p.error('target must be positive and tolerance nonnegative')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True, trust_remote_code=False)
    source = Path(args.source)
    prompts = json.loads(source.read_text(encoding='utf-8-sig'))
    if not isinstance(prompts, list) or not prompts or any(not isinstance(s, str) or not s for s in prompts):
        p.error('source must be a nonempty array of nonempty strings')
    generated = [extend_prompt(tokenizer, prompt, args.target_tokens, args.tolerance) for prompt in prompts]
    out = Path(args.output)
    metadata = Path(str(out) + '.metadata.json')
    if out.exists() or metadata.exists():
        p.error('output already exists')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([s for s, _ in generated], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    snapshot = Path(args.tokenizer)
    metadata.write_text(json.dumps({'parameters': vars(args), 'input_token_counts': [n for _, n in generated],
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256(out.read_bytes()).hexdigest(),
        'tokenizer_sha256': {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                             for f in snapshot.iterdir() if f.is_file()}}, indent=2) + '\n', encoding='utf-8')
    print(f'Wrote {len(generated)} prompts; template input counts: {[n for _, n in generated]}')


if __name__ == '__main__':
    main()
