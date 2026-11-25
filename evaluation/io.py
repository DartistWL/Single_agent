"""io."""
import hashlib, json, os
from pathlib import Path

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)

def interaction_budget(schedule, training_blocks):
    training = schedule['train']
    cycles, remaining = divmod(training_blocks, len(training))
    counts = {'train': (training_blocks, cycles * sum((r[2] for r in training)) + sum((r[2] for r in training[:remaining])))}
    counts.update({s: (len(schedule[s]), sum((r[2] for r in schedule[s]))) for s in ('val', 'test')})
    return {s: dict(episodes=n, days=d, steps=d * 48) for s, (n, d) in counts.items()}
