"""Read result data and render device trajectories."""
import argparse
import csv
import gzip
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo
from model.catalog import BASELINES, legacy_name
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator
ROOT = Path(__file__).resolve().parents[1]
LOCAL_RESULTS = ROOT / 'evaluation_results/household_multihome_3000'
CLOUD_RESULTS = ROOT / 'evaluation_results/household_multihome_3seeds'
CARBON_DATA = ROOT / 'data/carbon_2020/eirgrid_roi_co2_2020_30min.csv'
WIDTH = 7.2
HOURS_PER_STEP = 0.5
STEPS_PER_DAY = 48
DEDUPLICATED = 'ER_PPO'
INDEPENDENT = 'IH_PPO'
PHYSICAL = 'SRB'
BLUE = '#0072B2'
TEAL = '#007C75'
INK = '#202C38'
NEUTRAL = '#53606D'
REFERENCE = '#9BA6B0'
LABELS = {'ER_PPO': 'ER-PPO', 'IH_PPO': 'IH-PPO', 'IC_PPO': 'IC-PPO', 'AR_Class_PPO': 'AR-Class PPO', 'No_Mask': 'IH-PPO w/o action mask', 'No_SN': 'IH-PPO w/o state norm.', 'No_AN': 'IH-PPO w/o advantage norm.', 'No_Share': 'IH-PPO w/o shared encoder', 'SRB': 'SRB', 'CI_RBC': 'CI-RBC', 'TOU_RBC': 'TOU-RBC', 'MPC': 'MPC', 'Reference_PPO': 'Reference PPO', 'CrossQ': 'Adapted CrossQ', 'DroQ': 'Adapted DroQ', 'Branching_Rainbow': 'Branching Rainbow adaptation'}

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def number(row, key):
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"Non-finite {key!r} in {row.get('method', 'input data')}")
    return value

@dataclass(frozen=True)
class Results:
    root: Path
    manifest: dict
    summary: dict
    by_seed: dict

    @classmethod
    def load(cls, root, prefix):
        manifest = read_json(root / 'manifest.json')
        payload = {key: value for key, value in manifest.items() if key != 'id'}
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        if manifest['id'] != sha256(encoded).hexdigest():
            raise ValueError(f'Invalid manifest identity in {root}')
        summary, by_seed = ({}, {})
        for split in ('val', 'test'):
            rows = read_csv(root / f'{prefix}_{split}.csv')
            summary[split] = {row['method']: row for row in rows}
            rows = read_csv(root / f'{prefix}_by_seed_{split}.csv')
            by_seed[split] = {(row['method'], int(row['seed']) if row['seed'] else None): row for row in rows}
            for method in summary[split]:
                seeds = (None,) if method in BASELINES else manifest['train_seeds']
                for seed in seeds:
                    row = by_seed[split][method, seed]
                    budget = manifest['budget'][split]
                    actual = (int(row['episodes']), int(row['days']), int(row['environment_steps']))
                    if actual != (budget['episodes'], budget['days'], budget['steps']):
                        raise ValueError(f'Unequal {split} evaluation budget for {method}, seed {seed}')
        if summary['val'].keys() != summary['test'].keys():
            raise ValueError(f'Validation and test cover different methods in {root}')
        return cls(root, manifest, summary, by_seed)

    def seeds(self, method):
        return (None,) if method in BASELINES else self.manifest['train_seeds']

    def values(self, method, split, metric):
        return [number(self.by_seed[split][method, seed], metric) for seed in self.seeds(method)]

    def run_dir(self, method, seed):
        return self.root / ('controllers' if seed is None else f'seed_{seed}') / method

    def test_block(self, method, seed, case):
        house, start, days = case
        stem = self.run_dir(method, seed) / 'test' / f'{house}_{start}'
        marker = read_json(stem.with_suffix('.json'))
        expected = {'manifest_id': self.manifest['id'], 'method': legacy_name(method) if self.manifest.get('schema') == 'household-multihome-v1' else method, 'seed': seed, 'split': 'test', 'house': house, 'start_date': start, 'days': days, 'steps': days * STEPS_PER_DAY}
        if any((marker[key] != value for key, value in expected.items())):
            raise ValueError(f'Test block differs from its registration in {stem}')
        return (marker, stem.with_suffix('.jsonl.gz'))

def check_comparable(local, cloud):
    fields = ('inputs', 'schedule', 'budget', 'objective', 'scenario_seed', 'threshold_kw', 'mpc_horizon', 'mpc_budget', 'inventory_accounting')
    for key in fields:
        if local.manifest[key] != cloud.manifest[key]:
            raise ValueError(f'The batches must use the same {key!r} for these figures')
    if len(local.manifest['train_seeds']) != 1 or len(cloud.manifest['train_seeds']) < 2:
        raise ValueError('Expected a single-seed screening batch and a multi-seed replication batch')

def panel(ax, letter, title, xlabel=''):
    ax.set_title(f'{letter}  {title}', loc='left', fontweight='bold', pad=6)
    if xlabel:
        ax.set_xlabel(xlabel)

def trace_rows(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)

@dataclass(frozen=True)
class DispatchCase:
    house: str
    start: date
    days: int
    seed: int
    peak_reduction: float
    eligible_runs: int
    primary: list
    reference: list

    @property
    def edges(self):
        return np.arange(len(self.primary) + 1) * HOURS_PER_STEP

def read_dispatch_case(cloud):
    choices = []
    for case in cloud.manifest['schedule']['test']:
        reference, reference_path = cloud.test_block(PHYSICAL, None, case)
        if reference['totals']['planned_trip_kwh'] <= 0:
            continue
        pv_kwh = sum((row['ledger']['pv_generation_kwh'] for row in trace_rows(reference_path)))
        if pv_kwh <= 1e-09:
            continue
        for seed in cloud.seeds(DEDUPLICATED):
            marker, path = cloud.test_block(DEDUPLICATED, seed, case)
            reduction = (reference['totals']['daily_peak_kw'] - marker['totals']['daily_peak_kw']) / case[2]
            choices.append((reduction, tuple(case), seed, path, reference_path))
    if not choices:
        raise ValueError('The device case requires a held-out block with PV production and scheduled EV travel')
    choices.sort(key=lambda row: row[:3])
    reduction, (house, start, days), seed, path, reference_path = choices[(len(choices) - 1) // 2]
    primary, reference = (list(trace_rows(path)), list(trace_rows(reference_path)))
    start_day = date.fromisoformat(start)
    expected = [(house, (start_day + timedelta(days=day)).isoformat(), slot) for day in range(days) for slot in range(STEPS_PER_DAY)]
    for rows in (primary, reference):
        if [(row['house'], row['day'], row['slot']) for row in rows] != expected:
            raise ValueError('Dispatch traces differ from their registered block')
    return DispatchCase(house, start_day, days, seed, reduction, len(choices), primary, reference)

def values_at_steps(rows, group, key):
    return np.array([number(row[group], key) for row in rows])

def state_at_boundaries(rows, key):
    return np.r_[number(rows[0]['state'], key), values_at_steps(rows, 'next_state', key)]

def plot_step_trace(ax, edges, values, **style):
    ax.step(edges, np.r_[values, values[-1]], where='post', **style)

def compare_trace(ax, case, getter, *, stepped=False, style_pair=('-', '--')):
    for rows, color, line, width in zip((case.primary, case.reference), (BLUE, NEUTRAL), style_pair, (1.4, 0.9)):
        style = dict(color=color, ls=line, lw=width, zorder=3)
        values = getter(rows)
        if stepped:
            plot_step_trace(ax, case.edges, values, **style)
        else:
            ax.plot(case.edges, values, **style)

def shared_signal(case, key):
    signal = values_at_steps(case.primary, 'state', key)
    if not np.array_equal(signal, values_at_steps(case.reference, 'state', key)):
        raise ValueError(f'Dispatch traces disagree on {key}')
    return signal

def figure_dispatch(cloud, case, *, pv_color=TEAL, tariff_color=NEUTRAL, include_storage=True):
    rows = 4 if include_storage else 3
    height = 6.64 if include_storage else 5.2
    fig, axes = plt.subplots(rows, 2, sharex=True, figsize=(WIDTH, height), layout='constrained')
    end = case.start + timedelta(days=case.days - 1)
    fig.suptitle(f'{case.house}   {case.start:%d %b}–{end:%d %b %Y}   |   Training seed {case.seed}', fontweight='bold', fontsize=9.2)
    (grid, tariff), (ev, washer), (air, water) = axes[:3]
    specs = (('Grid import', 'kW'), ('Import tariff', '€ / kWh'), ('EV stored energy', 'kWh'), ('Washing machine', 'kW'), ('Indoor temperatures', '°C'), ('Hot-water tank', '°C'), ('Stationary battery and PV', 'kW'), ('ESS stored energy', 'kWh'))
    if not include_storage:
        specs = specs[:6]
    for index, (ax, (title, unit)) in enumerate(zip(axes.flat, specs)):
        panel(ax, 'abcdefgh'[index], title)
        ax.set_ylabel(unit)
        for day in range(1, case.days):
            ax.axvline(day * 24, color='#D8DEE4', lw=0.65, zorder=0)
        ax.set_xlim(case.edges[0], case.edges[-1])
        ax.xaxis.set_major_locator(MultipleLocator(12))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    compare_trace(grid, case, lambda rows: values_at_steps(rows, 'ledger', 'grid_import_kw'), stepped=True)
    grid.axhline(cloud.manifest['threshold_kw'], ls=':', color=REFERENCE, lw=0.9)
    grid.set_ylim(bottom=0)
    plot_step_trace(tariff, case.edges, shared_signal(case, 'electricity_price'), color=tariff_color, lw=1.05)
    away = shared_signal(case, 'ev_at_home') == 0
    edges = np.flatnonzero(np.diff(np.r_[False, away, False]))
    for left, right in edges.reshape(-1, 2):
        ev.axvspan(case.edges[left], case.edges[right], color='#E9EDF1', lw=0)
    compare_trace(ev, case, lambda rows: state_at_boundaries(rows, 'ev_battery_state'))
    active = values_at_steps(case.primary, 'actions', 'ev_shield_intervened').astype(bool)
    ev.scatter(case.edges[1:][active], state_at_boundaries(case.primary, 'ev_battery_state')[1:][active], s=8, marker='o', facecolor='white', edgecolor=BLUE, lw=0.55, zorder=5)
    compare_trace(washer, case, lambda rows: values_at_steps(rows, 'ledger', 'wash_energy_kwh') / HOURS_PER_STEP, stepped=True)
    washer.set_ylim(bottom=0)
    compare_trace(air, case, lambda rows: state_at_boundaries(rows, 'indoor_temp'))
    compare_trace(air, case, lambda rows: state_at_boundaries(rows, 'indoor_temp2'), style_pair=(':', '-.'))
    air.legend(handles=[Line2D([], [], color=INK, lw=1, ls='-', label='Zone 1'), Line2D([], [], color=INK, lw=1, ls=':', label='Zone 2')], loc='lower right', ncols=2, columnspacing=0.65, handlelength=1.25)
    compare_trace(water, case, lambda rows: state_at_boundaries(rows, 'ewh_temp'))

    def ess_power(rows):
        charge = values_at_steps(rows, 'ledger', 'ess_charge_input_kwh')
        discharge = values_at_steps(rows, 'ledger', 'ess_discharge_output_kwh')
        return (charge - discharge) / HOURS_PER_STEP
    if include_storage:
        power, ess = axes[3]
        plot_step_trace(power, case.edges, shared_signal(case, 'pv_generation'), color=pv_color, lw=1.25)
        compare_trace(power, case, ess_power, stepped=True)
        compare_trace(ess, case, lambda rows: state_at_boundaries(rows, 'ess_state'))
        ess.set_ylim(bottom=0)
    for ax in axes[-1]:
        ax.set_xlabel('Elapsed time (h)')
    legend = [Line2D([], [], color=BLUE, lw=1.4, label='ER-PPO'), Line2D([], [], color=NEUTRAL, lw=0.9, ls='--', label='SRB'), Line2D([], [], color=REFERENCE, lw=0.85, ls=':', label='Peak target (a)'), Line2D([], [], marker='o', mfc='white', mec=BLUE, mew=0.7, ms=3.5, ls='none', label='ER-PPO EV safety intervention (c)')]
    if include_storage:
        legend.insert(3, Line2D([], [], color=pv_color, lw=1.25, label='PV generation (g)'))
    fig.legend(handles=legend, loc='outside lower center', ncols=3, columnspacing=1.5)
    return fig
