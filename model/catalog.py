"""Algorithm names and model paths."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
ALGORITHMS = json.loads(Path(__file__).with_name('algorithms.json').read_text(encoding='utf8'))
CATALOG = {a['name']: a for a in ALGORITHMS}
METHODS = tuple(CATALOG)
BASELINES = tuple((n for n, a in CATALOG.items() if a['kind'] == 'controller'))
LEARNERS = tuple((n for n in METHODS if n not in BASELINES))

def canonical_name(name):
    if name in CATALOG:
        return name
    aliases = {a['paper_name'].lower().replace('-', '_').replace(' ', '_'): a['name'] for a in ALGORITHMS}
    key = str(name).lower().replace('-', '_').replace(' ', '_')
    if key in aliases:
        return aliases[key]
    raise ValueError('Unknown paper algorithm: ' + str(name))

def legacy_name(name):
    return CATALOG[canonical_name(name)]['legacy_name']

def paper_name(name):
    return CATALOG[canonical_name(name)]['paper_name']

def checkpoint_path(run_root, name):
    run_root = Path(run_root).resolve()
    relative = run_root.relative_to(ROOT / 'evaluation_results')
    return ROOT / 'model/checkpoints' / relative / (canonical_name(name) + '.pth')

def training_log_path(run_root, name):
    name = canonical_name(name)
    return Path(run_root) / name / (name + '_training.csv')
