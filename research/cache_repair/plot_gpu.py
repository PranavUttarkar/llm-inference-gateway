"""Plot source-question measurements; never imply KL is answer accuracy."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directories', type=Path, nargs='+', required=True)
    parser.add_argument('--analysis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('refusing to overwrite figure')
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [r for p in args.directories for r in json.loads((p/'manifest.json').read_text())['results']]
    report = json.loads(args.analysis.read_text())
    names = ['keep', 'state_only_frozen_scales', 'state_and_scales',
             'higher_precision_k8v8_stream_anchor', 'uncompressed_full_prefill_geometry',
             'full_prefill_requant_fresh_scales']
    labels = ['KEEP\nK4/V2', 'Rebuild\nfrozen params', 'Rebuild\nfresh params', 'Streaming\nK8/V8',
              'BF16\nfull prefill', 'Full prefill\nthen K4/V2']
    values = [np.array([r['arms'][name]['mean_kl_nats'] for r in rows]) for name in names]
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={'width_ratios': [1.3, 1]})
    axes[0].boxplot([np.maximum(v, 1e-9) for v in values], showfliers=False)
    for i, v in enumerate(values):
        jitter = np.random.default_rng(i).uniform(-.14, .14, len(v))
        axes[0].scatter(i+1+jitter, np.maximum(v, 1e-9), s=9, alpha=.35)
    axes[0].set_xticks(range(1, 7), labels, fontsize=9)
    axes[0].set_yscale('log')
    axes[0].set_ylabel('Full-vocabulary KL vs blockwise BF16 (nats/token)')
    axes[0].set_title('All source questions; paired inputs; lower KL is closer')
    benefit = values[0]-values[1]
    order = np.argsort(benefit)
    axes[1].scatter(range(1, len(rows)+1), benefit[order], s=14)
    axes[1].axhline(0, color='black', linewidth=1)
    axes[1].axhline(.001, color='gray', linestyle='--', linewidth=1, label='Planning scale: 0.001')
    axes[1].set_xlabel('Source questions ordered by measured primary effect')
    axes[1].set_ylabel('KEEP KL minus frozen-rebuild KL (nats/token)')
    axes[1].set_title('Primary contrast; positive favors reconstruction')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
    figure.suptitle(f'Exploratory GPU numerical pilot: {len(rows)} fixed reference-text traces', fontsize=13)
    figure.text(.5, .035,
        f"Operational gate T = {report['operational_continue_threshold_nats']:.5g} nats/token; "
        'includes the BF16 geometry floor. No answer-accuracy or packed-memory claim.',
        ha='center', fontsize=9)
    figure.tight_layout(rect=(0, .08, 1, .94))
    figure.savefig(args.output, dpi=180)
    plt.close(figure)


if __name__ == '__main__':
    main()
