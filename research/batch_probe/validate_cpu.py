"""Check masking/padding on a tiny random Qwen2; never scientific GPU evidence."""
import argparse
import json
from pathlib import Path

from .design import CONDITIONS, padded_batch


def validate():
    import torch
    import transformers
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(571)
    torch.set_num_threads(1)
    target = list(range(3, 23))
    companions = [[30+i, 40+i, 50+i] for i in range(7)]
    report = {'purpose': 'cpu_implementation_validation', 'scientific_evidence': False,
              'model': 'randomly_initialized_tiny_Qwen2_no_download',
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'dtype': 'float32', 'tolerance_max_absolute_logit_drift': 1e-5, 'attention_backends': {}}
    for attention in ('eager', 'sdpa'):
        config = Qwen2Config(vocab_size=128, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=4,
                            num_key_value_heads=2, max_position_embeddings=512,
                            attention_dropout=0.0, pad_token_id=0)
        config._attn_implementation = attention
        torch.manual_seed(571)
        model = Qwen2ForCausalLM(config).eval()
        logits = {}
        for condition in CONDITIONS:
            ids, masks, row, token = padded_batch(target, condition, 0, companions)
            mask = torch.tensor(masks)
            positions = mask.cumsum(-1)-1
            positions.masked_fill_(mask == 0, 0)
            with torch.inference_mode():
                output = model(input_ids=torch.tensor(ids), attention_mask=mask,
                               position_ids=positions, use_cache=False)
            logits[condition] = output.logits[row, token]
        drift = {key: float((value - logits['solo']).abs().max()) for key, value in logits.items()}
        if max(drift.values()) > 1e-5:
            raise AssertionError(f'{attention}: masking/geometry control failed: {drift}')
        report['attention_backends'][attention] = {'passed': True, 'max_absolute_logit_drift': drift}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite existing CPU validation')
    report = validate()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
