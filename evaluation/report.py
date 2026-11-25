"""Aggregate household evaluation records."""
from collections import defaultdict
from pathlib import Path
from statistics import fmean, stdev
import csv, os
from model.catalog import METHODS, BASELINES
from .protocol import task_protocol
from .records import run_root, _paths, _identity, _read_completed
BATTERIES = {'ess': 'ess_state', 'ev': 'ev_battery_state'}

def _aggregate(records):
    sums = defaultdict(float)
    days = sum((r['days'] for r in records))
    departures = sum((r['totals']['ev_departures'] for r in records))
    successes = sum((r['totals']['ev_successes'] for r in records))
    for block in records:
        for key, value in block['totals'].items():
            sums[key] += value
        for dev in BATTERIES:
            for key in ('net_depletion_kwh', 'recovery_import_kwh', 'recovery_export_kwh'):
                sums[f'{dev}_{key}'] += block['inventory'][dev][key]
    result = dict(episodes=len(records), days=days, environment_steps=sum((r['steps'] for r in records)), **{k: v / days for k, v in sums.items()})
    result['ev_departure_success_rate'] = successes / departures if departures else None
    for dev in BATTERIES:
        for key in ('start_kwh', 'end_kwh'):
            result[f'{dev}_{key}_per_block'] = fmean((r['inventory'][dev][key] for r in records))
    return result

def _tables(root, p, split, methods):
    summary, per_seed, per_house = ([], [], [])
    budget_fields = ('episodes', 'days', 'environment_steps')
    sd_fields = ('return_total', 'cost_eur', 'inventory_aligned_cost_eur', 'carbon_kg')
    for method in methods:
        seeds = (None,) if method in BASELINES else p['train_seeds']
        evaluations = []
        for seed in seeds:
            task = task_protocol(p, seed)
            folder = run_root(root, seed) / method / split
            cases = [_read_completed(*_paths(folder, home, start), _identity(task, method, split, (home, start, days))) for home, start, days in p['schedule'][split]]
            scores = _aggregate(cases)
            evaluations.append(scores)
            per_seed.append(dict(method=method, seed=seed, **scores))
            by_home = defaultdict(list)
            for case in cases:
                by_home[case['house']].append(case)
            for home, records in by_home.items():
                per_house.append(dict(method=method, seed=seed, house=home, **_aggregate(records)))
        average = {k: evaluations[0][k] if k in budget_fields else fmean((r[k] for r in evaluations)) if evaluations[0][k] is not None else None for k in evaluations[0]}
        average.update({f'{key}_seed_sd': stdev((r[key] for r in evaluations)) if len(evaluations) > 1 else None for key in sd_fields})
        summary.append(dict(method=method, train_seed_count=0 if method in BASELINES else len(seeds), **average))
    summary.sort(key=lambda row: -row['return_total'])
    return (summary, per_seed, per_house)

def _write_csv(path, rows):
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', newline='', encoding='utf8') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)

def available_methods(root, p):
    return tuple((name for name in METHODS if any(((run_root(root, seed) / name).is_dir() for seed in ((None,) if name in BASELINES else p['train_seeds'])))))

def report(root, p, methods=None):
    root = Path(root)
    methods = tuple(methods) if methods else available_methods(root, p)
    if not methods:
        raise ValueError('No paper algorithm results in ' + str(root))
    if len(methods) == 1:
        prefix = methods[0] + '_summary'
    else:
        prefix = 'summary_all' if (root / 'summary_all_test.csv').is_file() else 'summary'
    for split in ('val', 'test'):
        tables = _tables(root, p, split, methods)
        for suffix, rows in zip(('', '_by_seed', '_by_house'), tables):
            if not rows:
                raise ValueError('No completed results for ' + split)
            file = root / (prefix + suffix + '_' + split + '.csv')
            _write_csv(file, rows)
            print(file)
