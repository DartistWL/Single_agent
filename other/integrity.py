"""integrity."""
import hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def checksum(path, normalize=False):
    data = Path(path).read_bytes()
    return hashlib.sha256(data.replace(b'\r\n', b'\n') if normalize else data).hexdigest()

def verify_runtime_sources():
    path = ROOT / 'other/provenance/source_manifest.json'
    spec = json.loads(path.read_text(encoding='utf8'))
    for rel, value in spec['source_sha256'].items():
        if checksum(ROOT / rel) != value:
            raise ValueError('Reorganized source changed: ' + rel)

def verify_archived_data(p):
    for rel, value in p['inputs'].items():
        if rel.startswith('data/') or rel == 'thermal_revision/weather_2020.csv':
            new = 'other/environment/weather_2020.csv' if rel == 'thermal_revision/weather_2020.csv' else rel
            if checksum(ROOT / new, True) != value:
                raise ValueError('Registered data changed: ' + new)
    from other.environment.config import CONFIG
    if CONFIG['reward_weights'] != p['objective']:
        raise ValueError('Reward parameters differ from the archived experiment')
    expected = json.loads((ROOT / 'other/provenance/effective_config_reference.json').read_text(encoding='utf8'))
    if any((CONFIG[k] != v for k, v in expected.items())):
        raise ValueError('Household parameters differ from the archived experiment')
