"""Research validity checks: independent oracles, geometry, leakage and nulls."""
import contextlib
import copy
import heapq
import importlib.util
import io
import json
import math
import random
import subprocess
import sys
import tempfile
import tarfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research.batch_probe.analyze import features, load_run, predictive_analysis, summarize
from research.batch_probe.design import CONDITIONS, padded_batch, schedule
from research.batch_probe.freeze import freeze
from research.batch_probe.margin_audit import audit as margin_audit, bound_check, coverage, development_threshold
from research.batch_probe.run import Fixture, atomic_manifest, main as run_main, softmax
from research.batch_probe.workload import LABELS, build_workload, digest, validate_workload

HAS_ANALYSIS = all(importlib.util.find_spec(name) for name in ('numpy', 'sklearn', 'scipy'))


class ResearchDesignTests(unittest.TestCase):
    def test_margin_threshold_fitted_without_flip_labels_and_ties_reported(self):
        threshold = development_threshold([.01, .02, .03, .04], .25)
        self.assertEqual(threshold, .01)
        result = coverage([.005, .01, .01, .4], [True, False, True, True], threshold)
        self.assertEqual(result['flagged_questions'], 3)
        self.assertEqual(result['flips_covered'], 2)
        self.assertEqual(result['flips_missed'], 1)
        self.assertEqual(result['flagged_nonflips'], 1)
        self.assertEqual(result['flip_recall'], 2/3)
        self.assertIsNone(coverage([.02], [False], .01)['flip_recall'])

    def test_same_prefix_margin_bound_and_shift_invariance(self):
        rng = random.Random(9001)
        flips = 0
        for _ in range(1000):
            base = [rng.uniform(-2, 2) for _ in range(4)]
            other = [z+rng.uniform(-.02, .02) for z in base]
            check = bound_check(base, other)
            self.assertFalse(check['bound_violated'])
            flips += check['flipped']
            shifted = bound_check([z+100 for z in base], [z+200 for z in other])
            self.assertEqual(check['flipped'], shifted['flipped'])
            self.assertAlmostEqual(check['delta_logit_span'], shifted['delta_logit_span'])
        self.assertGreater(flips, 0)
        self.assertFalse(bound_check([1, 0, -.5, -1], [1.01, -.01, -.49, -1.01])['flipped'])

    def test_seeded_labels_and_independent_oracles(self):
        rows = build_workload()
        self.assertEqual(rows, build_workload())
        self.assertNotEqual(rows, build_workload(seed=4))
        validate_workload(rows)
        for row in rows:
            d = row['data']
            if row['family'] == 'arithmetic':
                # Alternate quotient/remainder derivation.
                quotient, _ = divmod(d['a'] * d['b'] + d['c'], d['d'])
                _, answer = divmod(quotient, d['m'])
            elif row['family'] == 'graph':
                # Dijkstra, independent of exhaustive paths in the generator.
                queue, distances = [(0, 0)], {0: 0}
                while queue:
                    distance, node = heapq.heappop(queue)
                    if distance != distances[node]:
                        continue
                    for a, b, weight in d['edges']:
                        if a == node and distance + weight < distances.get(b, math.inf):
                            distances[b] = distance + weight
                            heapq.heappush(queue, (distance + weight, b))
                answer = distances[5]
            elif row['family'] == 'state':
                answer = d['initial']
                for op, value in d['operations']:
                    functions = {'add': lambda x: x + value, 'multiply': lambda x: x * value,
                                 'subtract_if_possible': lambda x: x - value * int(x >= value)}
                    answer = functions[op](answer)
            else:
                qualifying = filter(lambda r: r['region'] == d['region'] and r['units'] >= d['threshold'], d['rows'])
                answer = sum(map(lambda r: r['units'] * r['price'], qualifying))
            self.assertEqual(row['options'][LABELS.index(row['correct_label'])], answer)
        for family in ('arithmetic', 'graph', 'state', 'relational'):
            self.assertEqual([sum(r['correct_label'] == label for r in rows if r['family'] == family)
                              for label in LABELS], [15, 15, 15, 15])

    def test_oracle_and_split_tampering_rejected(self):
        rows = build_workload(12)
        bad = copy.deepcopy(rows)
        bad[0]['correct_label'] = LABELS[(LABELS.index(bad[0]['correct_label']) + 1) % 4]
        with self.assertRaisesRegex(ValueError, 'oracle'):
            validate_workload(bad)
        sibling = copy.deepcopy(rows[0])
        sibling.update(id='sibling', split='evaluation' if sibling['split'] == 'development' else 'development')
        with self.assertRaisesRegex(ValueError, 'leaked'):
            validate_workload(rows + [sibling])

    def test_actual_geometry_target_position_and_composition_control(self):
        target, companions = [1, 2, 3, 4], [[10+i, 20+i] for i in range(7)]
        batches = {}
        for condition in CONDITIONS:
            ids, masks, row, last = padded_batch(target, condition, 99, companions)
            self.assertEqual(ids[row][:last+1], target)
            self.assertEqual(masks[row][:last+1], [1] * len(target))
            self.assertTrue(all(x == 0 for x in masks[row][last+1:]))
            self.assertEqual(last, 3)
            self.assertEqual(len(ids), CONDITIONS[condition]['batch'])
            self.assertEqual({len(x) for x in ids}, {len(target)+CONDITIONS[condition]['extra_padding']})
            batches[condition] = (ids, masks, row)
        self.assertEqual(batches['batch8'][1], batches['batch8_content'][1])
        self.assertEqual(batches['solo'], batches['solo_repeat'])
        self.assertEqual(batches['batch8'], batches['batch8_repeat'])
        self.assertEqual(batches['batch8_last'][0][:7], batches['batch8'][0][1:])
        with self.assertRaises(ValueError):
            padded_batch(target, 'batch8', 99, [[1]*5]*7)

    def test_schedule_complete_and_randomized(self):
        jobs = schedule(['a', 'b', 'c'])
        self.assertEqual(jobs, schedule(['a', 'b', 'c']))
        self.assertNotEqual(jobs, schedule(['a', 'b', 'c'], seed=99))
        self.assertEqual(len(jobs), 3 * len(CONDITIONS))
        self.assertEqual(len({(j['question_id'], j['condition']) for j in jobs}), len(jobs))

    def test_features_and_fixture_do_not_read_labels(self):
        question = build_workload(12)[0]
        stripped = {key: value for key, value in question.items() if key not in ('correct_label', 'data', 'options')}
        probe = Fixture('sensitive')
        full = {key: probe.measure(question, key) for key in CONDITIONS}
        hidden = {key: probe.measure(stripped, key) for key in CONDITIONS}
        self.assertEqual(full, hidden)
        fs = features(hidden)
        self.assertEqual(set(fs), set(features(full)))
        self.assertEqual(fs['repeat_l1'], 0)
        self.assertEqual(fs['batch_repeat_l1'], 0)

    def test_analytic_sampling_baseline_and_stable_null(self):
        question = build_workload(12)[0]
        probe = Fixture('stable')
        records = {key: probe.measure(question, key) for key in CONDITIONS}
        fs = features(records)
        self.assertEqual(fs['flip_fraction'], 0)
        self.assertEqual(fs['log_max_l1'], -12)
        p = softmax(records['solo']['choice_logits'])
        self.assertAlmostEqual(fs['analytic_sample_disagreement'], sum(p[i]*p[j] for i in range(4) for j in range(4) if i != j))


class ResearchEvidenceTests(unittest.TestCase):
    def test_frozen_bundle_reproducible_and_contains_only_declared_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = build_workload(12)
            workload = root / 'input.json'
            workload.write_text(json.dumps({'questions': rows, 'sha256': digest(rows)}), encoding='utf-8')
            first = freeze(workload, root/'a.tar.gz')
            second = freeze(workload, root/'b.tar.gz')
            self.assertEqual(first['sha256'], second['sha256'])
            with tarfile.open(root/'a.tar.gz') as archive:
                names = archive.getnames()
                self.assertFalse(any('..' in Path(name).parts or Path(name).is_absolute() for name in names))
                self.assertTrue(all(name.startswith(('research/', 'tests/', 'workloads/')) or
                                    name in ('BUNDLE.json', 'RESEARCH.md') for name in names))
                self.assertIn('workloads/batch-probe-pilot-v1.json', names)
            with self.assertRaises(FileExistsError):
                freeze(workload, root/'a.tar.gz')

    def test_transient_manifest_lock_retried_without_losing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            original = Path.replace
            calls = []
            def transient(source, destination):
                calls.append(source)
                if len(calls) == 1:
                    raise PermissionError('simulated OneDrive lock')
                return original(source, destination)
            with patch.object(Path, 'replace', transient), patch('research.batch_probe.run.time.sleep'):
                atomic_manifest(path, {'status': 'complete'})
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(path.read_text())['status'], 'complete')

    def create_run(self, root, fixture='stable'):
        rows = build_workload(12)
        workload = root / 'input.json'
        workload.write_text(json.dumps({'questions': rows, 'sha256': digest(rows)}), encoding='utf-8')
        output = root / 'run'
        args = ['run', '--workload', str(workload), '--output', str(output), '--fixture', fixture]
        with patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
            run_main()
        return output

    def test_complete_run_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.create_run(Path(directory))
            manifest, questions, grouped = load_run(output)
            self.assertEqual(manifest['purpose'], 'fixture')
            self.assertEqual(len(questions), 48)
            self.assertFalse(manifest['backend']['scientific_evidence'])
            margins = margin_audit(output)
            self.assertFalse(margins['scientific_evidence'])
            self.assertEqual(margins['total_flipped_questions'], 0)
            self.assertIsNone(margins['development_quantile_rules']['probability_margin']['0.25']['flip_recall'])
            path = output / 'records.jsonl'
            original = path.read_text(encoding='utf-8')
            lines = original.splitlines()
            path.write_text('\n'.join(lines[:-1])+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'missing'):
                load_run(output)
            path.write_text(original, encoding='utf-8')
            changed = json.loads(lines[0])
            changed['target_input_sha256'] = 'corrupted'
            path.write_text(json.dumps(changed)+'\n'+'\n'.join(lines[1:])+'\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'tokens changed'):
                load_run(output)

    def test_failure_preserves_evidence_and_is_not_analyzable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(Fixture, 'measure', side_effect=RuntimeError('injected hardware failure')):
                with self.assertRaisesRegex(RuntimeError, 'injected'):
                    self.create_run(root)
            manifest = json.loads((root/'run'/'manifest.json').read_text())
            self.assertEqual(manifest['status'], 'failed')
            self.assertIn('source_sha256', manifest)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                load_run(root/'run')

    @unittest.skipUnless(HAS_ANALYSIS, 'install batch_probe/requirements-analysis.txt')
    def test_fixture_cannot_become_scientific_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            output = self.create_run(Path(directory))
            result = summarize(output, bootstrap=100)
            self.assertFalse(result['scientific_evidence'])
            self.assertFalse(result['triage']['continue_to_confirmation_candidate'])
            self.assertIn('not_a_full_gpu_pilot', result['triage']['failed_gates'])


@unittest.skipUnless(HAS_ANALYSIS, 'install batch_probe/requirements-analysis.txt')
class ResearchStatisticsTests(unittest.TestCase):
    def synthetic(self, signal):
        # Engineered statistical fixtures, not model observations.
        rng = random.Random(401)
        questions, fs = [], []
        for i in range(800):
            confidence = rng.uniform(.35, .95)
            instability = rng.random()
            probability_error = .08 + .84 * instability if signal else 1-confidence
            error = rng.random() < probability_error
            questions.append({'id': str(i), 'group_id': str(i), 'family': 'synthetic',
                              'split': 'development' if i < 400 else 'evaluation',
                              'correct_label': 'B' if error else 'A'})
            p = [confidence] + [(1-confidence)/3]*3
            fs.append({'choice': 'A', 'confidence': confidence, 'margin': p[0]-p[1],
                       'log_margin': math.log10(p[0]-p[1]), 'entropy': -sum(x*math.log(x) for x in p),
                       'analytic_sample_disagreement': 1-sum(x*x for x in p), 'log_tokens': math.log(200),
                       'choice_mass': 1.0, 'raw_top_probability': confidence,
                       'full_vocab_entropy': -sum(x*math.log(x) for x in p), 'full_vocab_top2_margin': p[0]-p[1],
                       'flip_fraction': instability, 'log_max_l1': math.log10(max(instability, 1e-12)),
                       'log_mean_l1': math.log10(max(instability/2, 1e-12))})
        return questions, fs

    def test_can_detect_injected_incremental_signal(self):
        questions, fs = self.synthetic(True)
        result = predictive_analysis(questions, fs, 100)
        self.assertGreater(result['primary_comparison']['delta_log_loss'], .08)
        self.assertGreater(result['primary_comparison']['conditional_group_bootstrap_95'][0], 0)

    def test_confidence_only_null_does_not_look_like_a_discovery(self):
        questions, fs = self.synthetic(False)
        result = predictive_analysis(questions, fs, 100)
        primary = result['primary_comparison']
        self.assertTrue(primary['delta_log_loss'] < .01 or primary['conditional_group_bootstrap_95'][0] <= 0)

    def test_evaluation_labels_cannot_change_fitted_predictions(self):
        questions, fs = self.synthetic(True)
        original = predictive_analysis(questions, fs, 100)
        changed = copy.deepcopy(questions)
        # Flip all evaluation labels. Model scoring must be exactly inverted in AUROC.
        for q in changed:
            if q['split'] == 'evaluation':
                q['correct_label'] = 'A' if q['correct_label'] == 'B' else 'B'
        other = predictive_analysis(changed, fs, 100)
        for method in ('confidence_spline', 'confidence_plus_instability', 'confidence_reference', 'confidence_plus_instability_reference'):
            self.assertAlmostEqual(original[method]['error_auroc'] + other[method]['error_auroc'], 1)
        self.assertEqual(original['reference_model_selection'], other['reference_model_selection'])


if __name__ == '__main__':
    unittest.main()
