import json
from pathlib import Path
import shutil
import tempfile
import unittest

from research.cache_repair.evaluate_gpu import evaluate_cohort
from research.cache_repair.reproduce import DEFAULT_STUDY, reproduce, verify_hashes


class ReproductionTests(unittest.TestCase):
    def test_published_primary_and_secondary_statistics_reproduce(self):
        result = reproduce()
        self.assertEqual(result['gpu']['source_questions'], 96)
        self.assertEqual(result['gpu']['decision'], 'no_material_numerical_target_stop')

    def test_tampered_capture_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            study = Path(temp)/'study'
            shutil.copytree(DEFAULT_STUDY, study)
            with (study/'evidence/gpu0/manifest.json').open('ab') as stream:
                stream.write(b'\n')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                verify_hashes(study)

    def test_partial_cohort_cannot_receive_a_full_cohort_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            for shard in ('gpu0', 'gpu1'):
                directory = Path(temp)/shard
                directory.mkdir()
                m = json.loads((DEFAULT_STUDY/'evidence'/shard/'manifest.json').read_text())
                if shard == 'gpu0':
                    m['results'].pop()
                (directory/'manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'roster differs'):
                evaluate_cohort(DEFAULT_STUDY/'specs/gpu.json', [Path(temp)/'gpu0', Path(temp)/'gpu1'])


if __name__ == '__main__':
    unittest.main()
