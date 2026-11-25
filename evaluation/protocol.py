"""Register calendars and verify experiment inputs."""
import hashlib, json
from pathlib import Path
from model.catalog import ROOT, METHODS, BASELINES, legacy_name
from .io import digest, save, interaction_budget
from other.integrity import verify_runtime_sources, verify_archived_data
OUTPUT = ROOT / 'evaluation_results/household_multihome_3000'
DEFAULT_EPISODES = 3000
SCHEMA = 'paper-project-v1'

def source_inputs():
    paths = [p for d in ('model', 'evaluation', 'analysis', 'other') for p in (ROOT / d).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.csv') and ('checkpoints' not in p.parts) and ('provenance' not in p.parts) and ('__pycache__' not in p.parts)]
    paths.extend((p for p in (ROOT / 'data').rglob('*') if p.is_file()))
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}

def load(root):
    root = Path(root)
    path = root / 'manifest.json'
    if not path.is_file():
        raise ValueError('Missing ' + str(path) + '; run prepare first')
    p = json.loads(path.read_text(encoding='utf8'))
    if digest({k: v for k, v in p.items() if k != 'id'}) != p.get('id'):
        raise ValueError('Manifest identity differs')
    if p.get('schema') == 'household-multihome-v1':
        verify_runtime_sources()
        verify_archived_data(p)
    elif p.get('schema') == SCHEMA:
        if p['inputs'] != source_inputs():
            raise ValueError('Registered source or data changed')
    else:
        raise ValueError('Unsupported experiment registration')
    if interaction_budget(p['schedule'], p['training_blocks']) != p['budget'] or p['budget']['val'] != p['budget']['test']:
        raise ValueError('Unequal or inconsistent registered budgets')
    return p

def task_protocol(p, seed):
    return {**p, 'manifest_id': p['id'], 'seed': seed, 'id': digest([p['id'], seed])}

def recorded_method(p, name):
    return legacy_name(name) if p.get('schema') == 'household-multihome-v1' else name

def prepare(root, episodes=3000, seeds=(11,)):
    from .calendar import build_schedule, validate_schedule, _calibrate, TRAIN_HOMES, TEST_HOMES, SPLIT_SEED
    from other.environment.config import CONFIG
    root = Path(root)
    if episodes < 1 or not seeds or len(seeds) != len(set(seeds)) or any((not 0 <= s < 2 ** 32 for s in seeds)):
        raise ValueError('Use positive episodes and distinct nonnegative seeds')
    if (root / 'manifest.json').is_file():
        p = load(root)
        if p['training_blocks'] != episodes or p['train_seeds'] != list(seeds):
            raise ValueError('Different budget or training seeds already registered')
        return p
    schedule = build_schedule()
    budget = interaction_budget(schedule, episodes)
    inputs = source_inputs()
    calibration = _calibrate(schedule['train'])
    if source_inputs() != inputs:
        raise ValueError('Source or data changed during calibration')
    p = dict(schema=SCHEMA, train_homes=list(TRAIN_HOMES), val_homes=list(TRAIN_HOMES), test_homes=list(TEST_HOMES), split_seed=SPLIT_SEED, scenario_seed=11, train_seeds=list(seeds), methods=list(METHODS), core_methods=list(METHODS), baseline_methods=list(BASELINES), schedule=schedule, training_blocks=episodes, budget=budget, objective=CONFIG['reward_weights'], mpc_horizon=4, mpc_budget=8, inventory_accounting='block_mean_import_deficit_export_surplus_v1', inputs=inputs, **calibration)
    p['id'] = digest(p)
    save(root / 'manifest.json', p)
    return p
