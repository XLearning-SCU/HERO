"""Explicit, process-local evaluation imports; never installed into training."""
from __future__ import annotations

import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
MANIFEST = json.loads((ROOT / 'runtime_manifest.json').read_text())


class EvalRuntimeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        entry = MANIFEST.get(fullname)
        if entry is None:
            return None
        return importlib.util.spec_from_file_location(fullname, ROOT / entry['file'])


def install():
    """Activate before importing HERO CLI or evaluator; fail on mixed imports."""
    sys.dont_write_bytecode = True
    for name, entry in MANIFEST.items():
        expected = ROOT / entry['file']
        if hashlib.sha256(expected.read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Evaluation runtime checksum mismatch: {name}')
        loaded = sys.modules.get(name)
        if loaded is not None and Path(loaded.__file__).resolve() != expected.resolve():
            raise RuntimeError(f'{name} was imported before evaluation bootstrap')
    if not any(isinstance(f, EvalRuntimeFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, EvalRuntimeFinder())


def record_protocol(output):
    """Prevent mixing historical results with the corrected evaluation protocol."""
    install()  # Validate all code and companion assets before starting model servers.
    output = Path(output)
    marker = output / 'settings/eval_protocol.json'
    payload = {
        'protocol': 'count-worker-postrun-single-workspace-20260922',
        'runtime': {name: e['sha256'] for name, e in MANIFEST.items()},
        'driver_sha256': hashlib.sha256((ROOT / 'eval_swebench.py').read_bytes()).hexdigest(),
        'launcher_sha256': hashlib.sha256((ROOT / 'run_eval_swe_bench.sh').read_bytes()).hexdigest(),
    }
    if marker.exists():
        if json.loads(marker.read_text()) != payload:
            raise RuntimeError('Evaluation protocol changed; use a new output directory')
    elif (output / 'results.jsonl').exists() and (output / 'results.jsonl').stat().st_size:
        raise RuntimeError('Historical results have no aligned protocol marker; use a new output directory')
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps(payload, indent=2) + '\n')


if __name__ == '__main__':
    record_protocol(sys.argv[1])
