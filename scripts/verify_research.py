"""Check research evidence and local publication links without network or GPU access."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.cache_repair.reproduce import reproduce
from research.trace_subset import audit
import json


def verify_links():
    documents = [ROOT/'README.md', ROOT/'docs/research-workflow.md', ROOT/'docs/serving.md']
    documents += sorted((ROOT/'reports').glob('*/report.md'))
    checked = 0
    for document in documents:
        content = document.read_text(encoding='utf-8')
        targets = re.findall(r'\]\(([^)]+)\)', content)+re.findall(r'src="([^"]+)"', content)
        for target in targets:
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            relative = target.split('#', 1)[0]
            path = (document.parent/relative).resolve()
            if not path.is_relative_to(ROOT) or not path.exists():
                raise ValueError(f'broken local link in {document.relative_to(ROOT)}: {target}')
            checked += 1
    return checked


def main():
    result = reproduce()
    expected = json.loads((ROOT/'reports/batch-instability/evidence/subset-audit.json').read_text())
    actual = audit([])
    actual['input_sha256'] = None
    if actual != expected:
        raise ValueError('missing-trace audit differs from the published result')
    links = verify_links()
    print(f"Verified {result['verified_files']} research captures, CPU/GPU statistics, arm/cost summaries and {links} local links.")


if __name__ == '__main__':
    main()
