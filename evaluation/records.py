"""records."""
import json
from hashlib import sha256
from .protocol import recorded_method
STEPS_PER_DAY = 48

def run_root(root, seed):
    return root / ('controllers' if seed is None else f'seed_{seed}')

def _paths(folder, home, start):
    stem = f'{home}_{start}'
    return (folder / f'{stem}.json', folder / f'{stem}.jsonl.gz')

def _identity(p, method, split, block):
    home, start, days = block
    return dict(id=p['id'], manifest_id=p['manifest_id'], method=recorded_method(p, method), seed=p['seed'], split=split, house=home, start_date=start, days=days, steps=days * STEPS_PER_DAY)

def _read_marker(donefile, identity):
    value = json.loads(donefile.read_text(encoding='utf8'))
    if any((value.get(key) != expect for key, expect in identity.items())):
        raise ValueError(f'Completion record does not match its registered case: {donefile}')
    return value

def _matches_trace(trace, value):
    return trace.is_file() and sha256(trace.read_bytes()).hexdigest() == value.get('trace_sha256')

def _read_completed(donefile, trace, identity):
    if donefile.is_file():
        value = _read_marker(donefile, identity)
        if _matches_trace(trace, value):
            return value
    raise ValueError(f"Incomplete evaluation for {identity['method']} seed {identity['seed']}, {identity['split']} / {identity['house']} / {identity['start_date']}. Run the corresponding model or controller command again.")
