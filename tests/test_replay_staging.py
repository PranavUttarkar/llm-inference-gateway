import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from research.cache_repair.stage import stage
from research.cache_repair.reproduce import DEFAULT_STUDY


class ReplayStagingTests(unittest.TestCase):
    def test_clean_staging_and_transfer_exclude_local_state(self):
        with tempfile.TemporaryDirectory() as temp:
            out = stage(DEFAULT_STUDY, Path(temp)/'inputs')
            self.assertFalse((out/'model-1.5b').exists())
            self.assertEqual((out/'gpu-spec-v3.json').read_bytes(), (DEFAULT_STUDY/'specs/gpu.json').read_bytes())
            bundle = Path(temp)/'transfer.tar.gz'
            subprocess.run([sys.executable, '-m', 'research.cache_repair.transfer_bundle', '--root', str(out),
                            '--specs', str(out/'gpu-spec-v3.json'), '--output', str(bundle)],
                           capture_output=True, check=True)
            with tarfile.open(bundle) as archive:
                names = archive.getnames()
                self.assertNotIn('RESEARCH.md', names)
                self.assertFalse(any(name.endswith('.md') for name in names))
                manifest = json.load(archive.extractfile('TRANSFER.json'))
                for name, value in manifest['sha256'].items():
                    self.assertEqual(hashlib.sha256(archive.extractfile(name).read()).hexdigest(), value)
            with self.assertRaises(FileExistsError):
                stage(DEFAULT_STUDY, out)


if __name__ == '__main__':
    unittest.main()
