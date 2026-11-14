"""Summarize completed numerical pilots without presenting KL as task accuracy."""
import argparse
import json
import math
import statistics
from pathlib import Path
from .effect_reporting import describe


def summarize(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest['status'] != 'complete':
        raise ValueError('only completed pilots can be summarized')
    rows = manifest['results']
    if len(rows) != len(manifest['specification']['questions']):
        raise ValueError('missing source questions')
    differences = [row['paired_primary_delta'] for row in rows]
    n = len(rows)
    means = {arm: statistics.mean(row['arms'][arm]['mean_kl_nats'] for row in rows)
             for arm in rows[0]['arms']}
    sd = statistics.stdev(differences) if n > 1 else None
    planning = {}
    if sd is not None:
        for effect in (.0005, .001, .002):
            planning[str(effect)] = max(2, math.ceil(((1.9599639845+.8416212336)*sd/effect)**2))
    return {'scope': 'pretrained-model numerical emulation; reference-text cohort; not task accuracy or physical KV savings',
            'robustness_and_secondary': describe(rows),
            'questions': n, 'spec_sha256': manifest['spec_sha256'], 'mean_kl_by_arm': means,
            'mean_paired_primary_delta': statistics.mean(differences), 'sd_paired_primary_delta': sd,
            'individual_primary_deltas': differences,
            'fractional_mean_kl_reduction': 1-means['state_only_frozen_scales']/means['keep'],
            'fresh_scale_rebuild_worse_than_keep_questions': sum(
                row['arms']['state_and_scales']['mean_kl_nats'] > row['arms']['keep']['mean_kl_nats'] for row in rows),
            'normal_approximation_planning_N_two_sided_05_power_80': planning,
            'planning_warning': 'Four-case SD is highly uncertain. These N values are sensitivity calculations, '
                                'not validated power, confirmation, or evidence of accuracy gain.',
            'measured_case_seconds': sum(row['wall_seconds'] for row in rows),
            'logical_effective_bytes_per_token': statistics.mean(row['logical_packed_bytes']['keep']/manifest['specification']['prefix_tokens'] for row in rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot', type=Path)
    args = parser.parse_args()
    if args.output.exists() or (args.plot and args.plot.exists()):
        parser.error('refusing to overwrite analysis')
    result = summarize(args.directory)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        manifest = json.loads((args.directory/'manifest.json').read_text())
        names = ['keep', 'state_only_frozen_scales', 'state_and_scales']
        figure, ax = plt.subplots(figsize=(8, 4.5))
        for row in manifest['results']:
            ax.plot(range(3), [row['arms'][name]['mean_kl_nats'] for name in names], marker='o', alpha=.8,
                    label=row['question_id'].rsplit('/', 1)[-1])
        ax.set_xticks(range(3), ['KVarN keep', 'Rebuild states\nfreeze old scales', 'Rebuild states\nfresh scales'])
        ax.set_ylabel('Continuation KL vs BF16 (nats/token)')
        ax.set_title('CPU numerical pilot: 4 pretrained-model reference-text traces')
        ax.grid(axis='y', alpha=.2)
        ax.legend(title='Source question', fontsize=8)
        figure.text(.5, .01, 'BF16 1.5B · prefix 384 · probe 16 · emulated attention · no accuracy or speedup claim',
                    ha='center', fontsize=8)
        figure.tight_layout(rect=(0,.04,1,1))
        figure.savefig(args.plot, dpi=160)
        plt.close(figure)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
