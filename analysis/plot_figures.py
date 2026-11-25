"""Render Figures 3 to 9."""
from collections import Counter
from dataclasses import dataclass
import json
import re
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo
import numpy as np
from matplotlib.ticker import FixedLocator, MaxNLocator
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.text import Text
from analysis.plot_common import CARBON_DATA, CLOUD_RESULTS, DEDUPLICATED, INDEPENDENT, LOCAL_RESULTS, ROOT, STEPS_PER_DAY, LABELS, PHYSICAL, Results, check_comparable, figure_dispatch, number, plt, read_csv, read_dispatch_case, trace_rows
from other.environment.config import CONFIG
FIGURE = ROOT / 'figure'
DPI = 400
HALF_HOUR = 0.5
TRAIN_AVERAGE_BLOCK = 50
TEST_AVERAGE_BLOCK = 5
ABLATIONS = ('No_SN', 'No_AN', 'No_Share', 'No_Mask')
METHODS = (DEDUPLICATED, INDEPENDENT, *ABLATIONS)
REPRESENTATIONS = (DEDUPLICATED, INDEPENDENT, 'IC_PPO', 'AR_Class_PPO')
LEARNING_METHODS = (DEDUPLICATED, 'Reference_PPO', 'CrossQ', 'DroQ', 'Branching_Rainbow')
BENCHMARK = (*LEARNING_METHODS, PHYSICAL, 'TOU_RBC', 'CI_RBC', 'MPC')
PAPER_LABELS = {DEDUPLICATED: 'ER-PPO', INDEPENDENT: 'IH-PPO', 'IC_PPO': 'IC-PPO', 'AR_Class_PPO': 'AR-Class PPO', 'No_SN': 'No-SN', 'No_AN': 'No-AN', 'No_Share': 'No-Share', 'No_Mask': 'No-Mask', 'Reference_PPO': 'Reference PPO', 'CrossQ': 'CrossQ', 'DroQ': 'DroQ', 'Branching_Rainbow': 'Branching Rainbow', PHYSICAL: 'SRB', 'TOU_RBC': 'TOU-RBC', 'CI_RBC': 'CI-RBC', 'MPC': 'MPC'}
CURVE_STYLES = {DEDUPLICATED: ('#1F78B4', '-'), INDEPENDENT: ('#F29A25', '--'), 'No_SN': ('#33A02C', '-.'), 'No_AN': ('#B878B8', ':'), 'No_Share': ('#5DA5C1', '--'), 'No_Mask': ('#C44E52', '-.'), 'IC_PPO': ('#33A02C', '-.'), 'AR_Class_PPO': ('#B878B8', ':'), 'Reference_PPO': ('#F29A25', '--'), 'CrossQ': ('#33A02C', '-.'), 'DroQ': ('#B878B8', ':'), 'Branching_Rainbow': ('#5DA5C1', '--')}
COMPONENT_MARKERS = {DEDUPLICATED: 'o', INDEPENDENT: 's', 'No_SN': '^', 'No_AN': 'v', 'No_Share': 'D', 'No_Mask': 'x'}
BLUE, ORANGE, GREEN, RED = ('#4C72B0', '#E17C05', '#55A868', '#C44E52')
FONT_STYLE = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'Liberation Serif', 'DejaVu Serif'], 'font.size': 7.0, 'axes.labelsize': 7.4, 'xtick.labelsize': 6.4, 'ytick.labelsize': 6.4, 'legend.fontsize': 5.9, 'axes.linewidth': 0.65, 'pdf.fonttype': 42, 'savefig.facecolor': 'white'}

@dataclass(frozen=True)
class TestBlock:
    total_return: float
    cost_per_step: float
    hourly_cost: float
    satisfaction: float
    comfort: float
    ess_soc: float
    ev_soc: float
    wash_deviation: float
    wash_preference: float
    ac1: float
    ac2: float
    ewh: float
    battery_repair_fraction: float
    health_band_fraction: float
    wash_starts: int
    wash_starts_in_window: int
    physical_excursions: int
    tou_score: float
    charging_kw: float
    discharging_kw: float
    daily_peak_kw: float
    wash_hours: tuple
    ess_tariff_charge: tuple
    ev_tariff_charge: tuple
    ev_balance: tuple

def check_registered_config(batch):
    from other.integrity import verify_runtime_sources, verify_archived_data
    verify_runtime_sources()
    if batch.manifest.get('schema') == 'household-multihome-v1':
        verify_archived_data(batch.manifest)

def read_test_block(batch, method, seed, test_case):
    marker, path = batch.test_block(method, seed, test_case)
    if sha256(path.read_bytes()).hexdigest() != marker['trace_sha256']:
        raise ValueError(f'Test trace does not match its completion record: {path}')
    config = CONFIG['environment']
    ess = config['battery']
    ev = config['ev']
    wash = config['wash']
    tariff = config['tariff']
    cap_ess = ess['ess_capacity']
    cap_ev = ev['ev_capacity']
    health = config['ess_health']
    health_limits = ((health['healthy_soc_low'], health['healthy_soc_high']), (ev['ev_healthy_soc_low'], ev['ev_healthy_soc_high']))
    rated_chg = ess['ess_max_charge_power'] + ev['ev_max_charge_power']
    rated_dis = ess['ess_max_discharge_power'] + ev['ev_max_discharge_power']
    prices = tuple((round(tariff[key], 6) for key in ('offpeak_eur_per_kwh', 'day_eur_per_kwh', 'peak_eur_per_kwh')))
    boundary_start = wash['preferred_time_start']
    latest_start = wash['preferred_time_end'] - wash['duration_hours']
    eta = marker['inventory']['charge_efficiency']
    scores = dict.fromkeys(('ac1_comfort_score', 'ac2_comfort_score', 'ewh_comfort_score', 'combined_comfort_score'), 0.0)
    ess_soc_sum = ev_soc_sum = grid_bill = total_return = 0.0
    charge_energy = discharge_energy = score = 0.0
    repairs = health_outside = physical_excursions = steps = 0
    deviations, wash_hours = ([], [])
    ess_tariff_charge = np.zeros(3)
    ev_tariff_charge = np.zeros(3)
    ev_charge = ev_discharge = trip = 0.0
    for row in trace_rows(path):
        ledger, state, next_state = (row['ledger'], row['state'], row['next_state'])
        before_soc = (state['ess_state'] / cap_ess, state['ev_battery_state'] / cap_ev)
        after_soc = (next_state['ess_state'] / cap_ess, next_state['ev_battery_state'] / cap_ev)
        ess_soc_sum += before_soc[0]
        ev_soc_sum += before_soc[1]
        actions = row['actions']
        repairs += actions['ess_repair_distance_norm'] > 1e-09 or actions['ev_repair_distance_norm'] > 1e-09
        health_outside += sum((soc < low - 1e-09 or soc > high + 1e-09 for soc, (low, high) in zip(after_soc, health_limits)))
        physical_excursions += sum((soc < -1e-09 or soc > 1 + 1e-09 for soc in after_soc))
        total_return += row['reward']
        grid_bill += row['metrics']['cost_eur']
        for key in scores:
            scores[key] += ledger[key]
        if ledger['wash_started']:
            hour = row['slot'] * HALF_HOUR
            deviation = max(boundary_start - hour, hour - latest_start, 0)
            if bool(ledger['wash_on_window']) != (deviation == 0):
                raise ValueError(f'Wash-window reconstruction does not match {path}')
            deviations.append(deviation)
            wash_hours.append(hour)
        charging = ledger['ess_charge_input_kwh'] + ledger['ev_charge_input_kwh']
        discharging = ledger['ess_discharge_output_kwh'] + ledger['ev_discharge_output_kwh']
        charge_energy += charging
        discharge_energy += discharging
        price = round(state['electricity_price'], 6)
        if price not in prices:
            raise ValueError(f'Observed price outside the registered tariff tiers in {path}')
        tier = prices.index(price)
        ess_tariff_charge[tier] += ledger['ess_charge_input_kwh']
        ev_tariff_charge[tier] += ledger['ev_charge_input_kwh']
        ev_charge += ledger['ev_charge_input_kwh']
        ev_discharge += ledger['ev_discharge_output_kwh']
        trip += row['metrics']['trip_kwh']
        for device, field in (('ess', 'ess_state'), ('ev', 'ev_battery_state')):
            change = eta * ledger[f'{device}_charge_input_kwh'] - ledger[f'{device}_discharge_output_kwh'] / eta
            if device == 'ev':
                change -= row['metrics']['trip_kwh']
            if not np.isclose(next_state[field] - state[field], change, atol=1e-07, rtol=0):
                raise ValueError(f'{device} energy balance does not reconcile: {path}')
        power_share = 0.0
        if price == prices[0]:
            power_share = charging / (HALF_HOUR * rated_chg)
        if price == prices[-1]:
            power_share = discharging / (HALF_HOUR * rated_dis)
        score += min(1.0, eta * power_share)
        steps += 1
    if steps != marker['steps'] or len(deviations) != marker['days'] or (not np.isclose(grid_bill, marker['totals']['cost_eur'], rtol=1e-09, atol=1e-10)) or (not np.isclose(total_return, marker['totals']['return_total'], rtol=1e-09, atol=1e-10)):
        raise ValueError(f'Incomplete or inconsistent recorded test block {path}')
    comfort = scores['combined_comfort_score'] / steps
    satisfaction = 0.7 * comfort + 0.3
    return TestBlock(total_return=total_return, cost_per_step=grid_bill / steps, hourly_cost=grid_bill / (HALF_HOUR * steps), satisfaction=satisfaction, comfort=comfort, ess_soc=ess_soc_sum / steps, ev_soc=ev_soc_sum / steps, wash_deviation=fmean(deviations), wash_preference=fmean((d == 0 for d in deviations)), ac1=scores['ac1_comfort_score'] / steps, ac2=scores['ac2_comfort_score'] / steps, ewh=scores['ewh_comfort_score'] / steps, battery_repair_fraction=repairs / steps, health_band_fraction=health_outside / (2 * steps), wash_starts=len(deviations), wash_starts_in_window=sum((value == 0 for value in deviations)), physical_excursions=physical_excursions, tou_score=score / steps, charging_kw=charge_energy / (HALF_HOUR * steps), discharging_kw=discharge_energy / (HALF_HOUR * steps), daily_peak_kw=marker['totals']['daily_peak_kw'] / marker['days'], wash_hours=tuple(wash_hours), ess_tariff_charge=tuple(ess_tariff_charge / marker['days']), ev_tariff_charge=tuple(ev_tariff_charge / marker['days']), ev_balance=(marker['inventory']['ev']['start_kwh'], ev_charge, ev_discharge, trip, (1 - eta) * ev_charge + (1 / eta - 1) * ev_discharge, marker['inventory']['ev']['end_kwh']))

def collect_tests(batch, methods):
    observations = {}
    test_cases = batch.manifest['schedule']['test']
    for method in methods:
        observations[method] = [[read_test_block(batch, method, seed, test_case) for test_case in test_cases] for seed in batch.seeds(method)]
        print(f'Checked test traces for {batch.root.name} / {LABELS[method]}', flush=True)
    return observations

def samples(runs, field):
    return np.mean([[getattr(block, field) for block in run] for run in runs], axis=0)

def mark_panel(ax, letter, xlabel, ylabel, letter_x=-0.19):
    ax.text(letter_x, 1.035, letter, transform=ax.transAxes, fontsize=12, weight='bold', va='bottom')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))

def windows(values, width):
    return np.lib.stride_tricks.sliding_window_view(values, width)

def training_returns(batch, method):
    path = batch.run_dir(method, batch.seeds(method)[0]) / f'{method}_training.csv'
    rows = read_csv(path)
    budget = batch.manifest['budget']['train']
    steps = [int(row['environment_steps']) for row in rows]
    expected_steps = np.cumsum([int(row['days']) for row in rows]) * STEPS_PER_DAY
    if [int(row['block']) for row in rows] != list(range(1, budget['episodes'] + 1)) or not np.array_equal(steps, expected_steps) or steps[-1] != budget['steps']:
        raise ValueError(f'Training history differs from the registered budget: {path}')
    return np.array([number(row, 'return_per_day') for row in rows])

def training_line(ax, batch, method, *, window=TRAIN_AVERAGE_BLOCK):
    values = training_returns(batch, method)
    x = np.arange(window, len(values) + 1)
    averaged = windows(values, window).mean(axis=1)
    color, line = CURVE_STYLES[method]
    ax.plot(x, averaged, color=color, ls=line, lw=0.95 if method == DEDUPLICATED else 0.72, label=PAPER_LABELS[method])

def figure_components(local, local_tests):
    with plt.style.context(WHITE_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), layout='constrained')
        specs = (('a', 'Training block', 'Return / household-day', None), ('b', 'Held-out test block', 'Net bill (EUR / half-hour)', 'cost_per_step'), ('c', 'Held-out test block', 'Comfort-derived satisfaction', 'satisfaction'), ('d', 'Held-out test block', 'Battery health band excursions (%)', 'health_band_fraction'))
        for ax, (tag, xlabel, ylabel, field) in zip(axes.flat, specs):
            mark_panel(ax, tag, xlabel, ylabel, letter_x=-0.16)
            for index, method in enumerate(METHODS):
                if field is None:
                    training_line(ax, local, method)
                else:
                    values = samples(local_tests[method], field)
                    if tag == 'd':
                        values = values * 100
                    x = np.arange(TEST_AVERAGE_BLOCK, len(values) + 1)
                    color, line = CURVE_STYLES[method]
                    ax.plot(x, windows(values, TEST_AVERAGE_BLOCK).mean(axis=1), color=color, ls=line, lw=0.95 if method == DEDUPLICATED else 0.72, label=PAPER_LABELS[method], marker=COMPONENT_MARKERS[method] if tag == 'd' else None, markevery=(2 * index, 12), ms=2.5, mfc='white', mew=0.7)
            ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
            if tag == 'a':
                ax.set_xlim(0, local.manifest['budget']['train']['episodes'])
            if tag == 'd':
                ax.set_ylim(bottom=-0.2)
                zero_methods = [method for method in METHODS if all((block.health_band_fraction == 0 for run in local_tests[method] for block in run))]
                if len(zero_methods) == 5:
                    ax.text(0.97, 0.09, 'Other five methods: 0%', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.2, color='#7D8791', zorder=8, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=0.8))
        handles, labels = axes[1, 1].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncols=3, frameon=False)
        return fig

def figure_training(batch, methods, *, zoom=False):
    with plt.style.context(WHITE_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 3.65), layout='constrained')
        for method in methods:
            training_line(ax, batch, method)
        end = batch.manifest['budget']['train']['episodes']
        ax.set_xlim(end - 500 if zoom else 0, end)
        ax.set_xlabel('Training block')
        ax.set_ylabel('Return / household-day')
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        if zoom:
            retained = np.concatenate([windows(training_returns(batch, m), TRAIN_AVERAGE_BLOCK).mean(axis=1)[-500:] for m in methods])
            ax.set_ylim(retained.min() - 1.0, retained.max() + 1.0)
        handles, _ = ax.get_legend_handles_labels()
        labels = [f'{PAPER_LABELS[m]}  {fmean(training_returns(batch, m)[-TRAIN_AVERAGE_BLOCK:]):+.2f}' for m in methods]
        ax.legend(handles, labels, title='Final 50-block mean', title_fontsize=7.0, loc='lower right', frameon=True, fancybox=False, fontsize=7.1, facecolor='white', framealpha=0.95)
        if not zoom:
            detail = ax.inset_axes([0.33, 0.15, 0.34, 0.32])
            for method in methods:
                training_line(detail, batch, method)
            detail.set_xlim(end - 500, end)
            retained = np.concatenate([windows(training_returns(batch, m), TRAIN_AVERAGE_BLOCK).mean(axis=1)[-500:] for m in methods])
            detail.set_ylim(retained.min() - 1, retained.max() + 1)
            detail.set_title('Final 500 blocks', fontsize=6.2, pad=1.5)
            detail.set_xticks([end - 500, end - 250, end])
            detail.yaxis.set_major_locator(MaxNLocator(3))
            detail.tick_params(labelsize=5.9, length=2, pad=1)
            ax.indicate_inset_zoom(detail, edgecolor='#777777', linewidth=0.55, alpha=0.7)
        return fig

def corner_legend(ax):
    ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=True, facecolor='white', framealpha=0.94, borderpad=0.35, handlelength=1.6)

def histogram(ax, values, color, precision=3):
    ax.hist(values, bins=10, color=color, alpha=0.7, edgecolor='black', linewidth=0.65)
    ax.axvline(fmean(values), color='red', ls='--', lw=0.95, label=f'Mean: {fmean(values):.{precision}f}')
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.ticklabel_format(axis='x', useOffset=False)
    corner_legend(ax)

def marker_line(ax, values, color, marker, legend=None):
    x = np.arange(1, len(values) + 1)
    ax.plot(x, values, color=color, marker=marker, lw=0.72, ms=2.6, label=legend)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    return x
WHITE_STYLE = ['default', {**FONT_STYLE, 'axes.grid': True, 'grid.color': '#E7E7E7', 'grid.linewidth': 0.45, 'axes.axisbelow': True, 'axes.edgecolor': '#555555'}]
EVALUATION_STYLE = ['seaborn-v0_8-darkgrid', {**FONT_STYLE, 'axes.facecolor': '#EAEAF2', 'grid.color': '#F4F4F8', 'grid.linewidth': 0.55, 'axes.axisbelow': True}]

def figure_services(cloud, runs, reference):
    with plt.style.context(EVALUATION_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15), layout='constrained')
        specs = (('a', 'Avg hourly cost (EUR/h)', 'hourly_cost', GREEN), ('c', 'Thermal comfort score', 'comfort', ORANGE))
        for ax, (tag, xlabel, field, color) in zip((axes[0, 0], axes[0, 2]), specs):
            mark_panel(ax, tag, xlabel, 'Count')
            histogram(ax, samples(runs, field), color, precision=3 if tag == 'a' else 4)
        ax_peak = axes[0, 1]
        mark_panel(ax_peak, 'b', 'Held-out household', 'Mean daily peak (kW)')
        home_rows = read_csv(cloud.root / 'summary_by_house_test.csv')
        homes = cloud.manifest['test_homes']
        peaks = [[fmean((number(row, 'daily_peak_kw') for row in home_rows if row['method'] == method and row['house'] == home)) for home in homes] for method in (DEDUPLICATED, PHYSICAL)]
        x = np.arange(len(homes))
        ax_peak.bar(x - 0.18, peaks[0], 0.36, color=BLUE, alpha=0.8, label='ER-PPO')
        ax_peak.bar(x + 0.18, peaks[1], 0.36, color=ORANGE, alpha=0.8, label='SRB')
        ax_peak.set_xticks(x, homes, rotation=30)
        ax_peak.set_ylim(0, max(map(max, peaks)) * 1.25)
        corner_legend(ax_peak)
        ax_soc, ax_wash, ax_comfort = axes[1]
        mark_panel(ax_soc, 'd', 'Held-out test block', 'Mean SOC')
        ess_soc, ev_soc = (samples(runs, key) for key in ('ess_soc', 'ev_soc'))
        marker_line(ax_soc, ess_soc, BLUE, 'o', 'ESS SOC mean')
        marker_line(ax_soc, ev_soc, ORANGE, 's', 'EV SOC mean')
        ax_soc.set_ylim(max(0, min(ess_soc.min(), ev_soc.min()) - 0.04), max(ess_soc.max(), ev_soc.max()) + 0.16)
        mark_panel(ax_wash, 'e', 'Washer start time', 'Starts (%)')
        hours = np.arange(0, 24, HALF_HOUR)
        for records, offset, color, label in ((runs, -0.12, BLUE, 'ER-PPO'), (reference, 0.12, ORANGE, 'SRB')):
            starts = [hour for run in records for block in run for hour in block.wash_hours]
            counts = Counter(starts)
            ax_wash.bar(hours + offset, [100 * counts[h] / len(starts) for h in hours], width=0.24, color=color, alpha=0.85, label=label)
        ax_wash.set_xticks([6, 9, 12, 15, 18], ['06:00', '09:00', '12:00', '15:00', '18:00'], rotation=30)
        ax_wash.set_xlim(4.5, 20.5)
        ax_wash.set_ylim(0, 116)
        corner_legend(ax_wash)
        mark_panel(ax_comfort, 'f', 'Held-out test block', 'Comfort score')
        for key, label, color, marker in (('ac1', 'AC1 comfort', BLUE, 'o'), ('ac2', 'AC2 comfort', ORANGE, 's'), ('ewh', 'EWH comfort', GREEN, '^'), ('comfort', 'Overall comfort', RED, 'v')):
            marker_line(ax_comfort, samples(runs, key), color, marker, label)
        lower = min((samples(runs, key).min() for key in ('ac1', 'ac2', 'ewh', 'comfort')))
        ax_comfort.set_ylim(lower - 0.002, 1 + 0.28 * (1 - lower))
        ax_comfort.set_yticks(np.arange(0.98, 1.001, 0.005))
        ax_comfort.ticklabel_format(axis='y', useOffset=False)
        ax_comfort.legend(loc='upper right', ncols=2, fontsize=5.4, columnspacing=0.6, handlelength=1.15, borderpad=0.3, frameon=True, facecolor='white')
        corner_legend(ax_soc)
        return fig

def mean_filled_trace(ax, values, color, decimals):
    x = marker_line(ax, values, color, 'o')
    ax.fill_between(x, 0.0, values, color=color, alpha=0.16)
    ax.axhline(fmean(values), color='red', ls='--', lw=0.95, label=f'Mean: {fmean(values):.{decimals}f}')
    corner_legend(ax)

def figure_storage(runs, reference):
    with plt.style.context(EVALUATION_STYLE):
        fig = plt.figure(figsize=(7.2, 4.4), layout='constrained')
        grid = fig.add_gridspec(3, 1)
        top = grid[:2].subgridspec(2, 3).subplots()
        bottom = grid[2].subgridspec(1, 3).subplots()
        score, returns, cost = (samples(runs, key) for key in ('tou_score', 'total_return', 'hourly_cost'))
        mark_panel(top[0, 0], 'a', 'TOU dispatch score', 'Count')
        histogram(top[0, 0], score, ORANGE, 3)
        ax_tariff = top[0, 1]
        mark_panel(ax_tariff, 'b', 'Tariff period', 'Charge (kWh/day)')
        ess = np.mean([block.ess_tariff_charge for run in runs for block in run], axis=0)
        ev = np.mean([block.ev_tariff_charge for run in runs for block in run], axis=0)
        ax_tariff.bar(np.arange(3), ess, width=0.58, color=BLUE, alpha=0.8, label='ESS')
        ax_tariff.bar(np.arange(3), ev, bottom=ess, width=0.58, color=ORANGE, alpha=0.75, label='EV')
        ax_tariff.set_xticks(range(3), ['Off-peak', 'Day', 'Peak'])
        ax_tariff.set_ylim(0, max(ess + ev) * 1.3)
        corner_legend(ax_tariff)
        ax_balance = top[0, 2]
        mark_panel(ax_balance, 'c', 'EV energy balance', 'Energy (kWh)')
        start, charge, discharge, journey, loss, end = np.mean([block.ev_balance for run in runs for block in run], axis=0)
        if not np.isclose(start + charge - discharge - journey - loss, end, atol=1e-07):
            raise ValueError('Aggregate EV energy does not reconcile')
        changes = [start, charge, -discharge, -journey, -loss, end]
        levels = [0, start, start + charge, start + charge - discharge, start + charge - discharge - journey, 0]
        for i, (change, base) in enumerate(zip(changes, levels)):
            bar_bottom = base + min(change, 0)
            ax_balance.bar(i, abs(change), bottom=bar_bottom, width=0.64, color=BLUE if i in (0, 5) else GREEN if change >= 0 else RED, alpha=0.75)
        ax_balance.set_xticks(range(6), ['Start', 'Charge', 'Discharge', 'Trip', 'Loss', 'End'], rotation=55, ha='right', fontsize=5.7)
        ax_score, ax_power, ax_return = top[1]
        mark_panel(ax_score, 'd', 'Held-out test block', 'TOU dispatch score')
        mean_filled_trace(ax_score, score, ORANGE, 3)
        ax_score.set_ylim(0, max(0.0001, float(score.max()) * 1.42))
        mark_panel(ax_power, 'e', 'Held-out test block', 'Power (kW)')
        marker_line(ax_power, samples(runs, 'charging_kw'), BLUE, 'o', 'Charging power')
        marker_line(ax_power, samples(runs, 'discharging_kw'), ORANGE, 's', 'Discharging power')
        ax_power.set_ylim(0, max(0.05, float(samples(runs, 'charging_kw').max()) * 1.36, float(samples(runs, 'discharging_kw').max()) * 1.36))
        corner_legend(ax_power)
        mark_panel(ax_return, 'f', 'Held-out test block', 'Return')
        mean_filled_trace(ax_return, returns, BLUE, 2)
        ax_return.margins(y=0.12)
        ax_cost, ax_satisfaction, ax_soc = bottom
        mark_panel(ax_cost, 'g', 'Avg hourly cost (EUR/h)', 'TOU dispatch score')
        points = ax_cost.scatter(cost, score, c=returns, cmap='viridis', s=12.5, alpha=0.7)
        scale = fig.colorbar(points, ax=ax_cost, fraction=0.038, pad=0.035)
        scale.solids.set_rasterized(False)
        scale.set_label('Return')
        scale.locator = MaxNLocator(nbins=4)
        scale.update_ticks()
        mark_panel(ax_satisfaction, 'h', 'Daily peak reduction (kW)', 'Count')
        reduction = samples(reference, 'daily_peak_kw') - samples(runs, 'daily_peak_kw')
        histogram(ax_satisfaction, reduction, GREEN, 2)
        mark_panel(ax_soc, 'i', 'Mean SOC', 'TOU dispatch score')
        ax_soc.scatter(samples(runs, 'ess_soc'), score, color=BLUE, s=12.5, alpha=0.7, label='ESS SOC')
        ax_soc.scatter(samples(runs, 'ev_soc'), score, color=ORANGE, s=12.5, alpha=0.7, label='EV SOC')
        ax_soc.legend(loc='upper right', frameon=False)
        return fig

def test_carbon_profile(cloud):
    digest = sha256(CARBON_DATA.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    if digest != cloud.manifest['inputs'][CARBON_DATA.relative_to(ROOT).as_posix()]:
        raise ValueError('Unregistered carbon intensity file')
    readings = {datetime.fromisoformat(row['timestamp_utc']): number(row, 'carbon_intensity_gco2_per_kwh') for row in read_csv(CARBON_DATA)}
    days = sorted({date.fromisoformat(start) + timedelta(days=i) for _, start, length in cloud.manifest['schedule']['test'] for i in range(length)})
    tz = ZoneInfo('Europe/Dublin')
    values = [[readings[datetime.combine(day, datetime.min.time(), tzinfo=tz).astimezone(timezone.utc) + timedelta(minutes=30 * slot)] for slot in range(STEPS_PER_DAY)] for day in days]
    return (np.mean(values, axis=0), len(days))

def figure_benchmark(local):
    with plt.style.context(WHITE_STYLE):
        fig, ax = plt.subplots(figsize=(7.2, 4.25), layout='constrained')
        x = np.arange(len(BENCHMARK))
        returns = [fmean(local.values(m, 'test', 'return_total')) for m in BENCHMARK]
        emissions = [fmean(local.values(m, 'test', 'carbon_kg')) for m in BENCHMARK]
        bars = ax.bar(x, returns, width=0.58, color=['#5EB2F6' if m == DEDUPLICATED else '#C5E3FB' for m in BENCHMARK], edgecolor='#515151', linewidth=0.5, label='Test return')
        value_labels = ax.bar_label(bars, fmt='%.2f', fontsize=6.0, padding=3)
        value_labels[1].set_position((0, 10))
        ax.axhline(0, color='#555555', lw=0.6)
        ax.set_ylim(min(returns) - 2.0, max(returns) + 2.0)
        ax.set_xlim(-0.65, len(BENCHMARK) - 0.35)
        ax.set_ylabel('Return / household-day', color=BLUE)
        ax.tick_params(axis='y', labelcolor=BLUE)
        ax.set_xticks(x, [PAPER_LABELS[m] for m in BENCHMARK], rotation=34, ha='right')
        ax.xaxis.grid(False)
        carbon_ax = ax.twinx()
        carbon_line, = carbon_ax.plot(x, emissions, color=ORANGE, lw=0.85, marker='s', ms=3.0, label='Grid-import emissions')
        carbon_ax.set_ylabel('kg CO₂ / household-day', color=ORANGE)
        carbon_ax.tick_params(axis='y', labelcolor=ORANGE)
        carbon_ax.set_ylim(0, max(emissions) * 1.7)
        carbon_ax.grid(False)
        fig.legend([bars, carbon_line], ['Test return', 'Grid-import emissions'], loc='outside lower center', ncols=2, frameon=False, fontsize=6.6)
        inset = ax.inset_axes([0.43, 0.76, 0.51, 0.21])
        intensity, _ = test_carbon_profile(local)
        clock = np.arange(STEPS_PER_DAY) * HALF_HOUR
        inset.plot(clock, intensity, color='#4B61A1', lw=0.65)
        inset.axhline(intensity.mean(), color='red', ls='--', lw=0.55)
        inset.set_title('EirGrid test-day mean', fontsize=6.1, pad=1)
        inset.set_ylabel('g CO₂/kWh', fontsize=5.8, labelpad=1)
        inset.set_xlabel('Local time', fontsize=5.8, labelpad=0.5)
        inset.set_xticks([0, 12, 23.5], ['00:00', '12:00', '23:30'])
        inset.tick_params(labelsize=5.5, length=2, pad=1)
        inset.yaxis.set_major_locator(MaxNLocator(3))
        inset.set_xlim(0, 23.5)
        return fig

def figure_washer_checks(runs):
    with plt.style.context(EVALUATION_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 3.7), sharex=True, layout='constrained')
        marker_line(axes[0], samples(runs, 'wash_deviation'), BLUE, 'o')
        marker_line(axes[1], samples(runs, 'wash_preference'), ORANGE, 's')
        axes[0].set_ylabel('Start deviation (h)')
        axes[1].set_ylabel('Preferred-window share')
        axes[1].set_xlabel('Held-out test block')
        axes[0].set_ylim(-0.05, 0.05)
        axes[1].set_ylim(0.95, 1.05)
        return fig

def chinese_font():
    path = Path('C:/Windows/Fonts/msyh.ttc')
    if path.is_file():
        return path
    families = ['Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Zen Hei']
    return Path(findfont(FontProperties(family=families), fallback_to_default=False))

def export_figure(fig, directory, name):
    for suffix in ('pdf', 'png', 'svg'):
        with plt.rc_context({'svg.fonttype': 'path'}):
            fig.savefig(directory / f'{name}.{suffix}', dpi=DPI)
    labels = json.loads(Path(__file__).with_name('figure_labels_ZH.json').read_text(encoding='utf8'))
    font = chinese_font()
    for artist in fig.findobj(Text):
        original = artist.get_text()
        match = re.fullmatch('([a-i]  )(.+)', original)
        translated = labels.get(original)
        if original.startswith('Mean: '):
            translated = '均值 ' + original.removeprefix('Mean: ')
        if match and match[2] in labels:
            translated = match[1] + labels[match[2]]
        if 'Training seed' in original:
            translated = original.replace('19 Nov–21 Nov 2020', '2020 年 11 月 19–21 日')
            translated = translated.replace('Training seed', '训练种子')
        if translated:
            artist.set_text(translated)
            artist.set_fontproperties(FontProperties(fname=font, size=artist.get_fontsize(), weight=artist.get_fontweight()))
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            if isinstance(axis.get_major_locator(), FixedLocator):
                locations = axis.get_majorticklocs()
                formatter = axis.get_major_formatter()
                names = [formatter(value, i) for i, value in enumerate(locations)]
                axis.set_ticklabels([labels.get(name, name) for name in names])
    for suffix in ('pdf', 'png', 'svg'):
        with plt.rc_context({'svg.fonttype': 'path'}):
            fig.savefig(directory / f'{name}_zh.{suffix}', dpi=DPI)
    plt.close(fig)
    print(f'Rendered {name} in English and Chinese', flush=True)
def main(argv=None):
    import argparse
    from shutil import copy2
    p = argparse.ArgumentParser(description='Render the paper result figures.')
    p.add_argument('--figures', nargs='+', type=int, choices=range(3, 10), default=list(range(3, 10)))
    p.add_argument('--language', choices=('EN', 'both'), default='EN')
    p.add_argument('--supplementary', action='store_true')
    args = p.parse_args(argv)
    local = Results.load(LOCAL_RESULTS, 'summary_all')
    cloud = Results.load(CLOUD_RESULTS, 'summary')
    check_comparable(local, cloud)
    check_registered_config(cloud)
    tests = collect_tests(local, METHODS) if 3 in args.figures else None
    records = collect_tests(cloud, (DEDUPLICATED, PHYSICAL)) if set(args.figures) & {5, 6} or args.supplementary else None
    runs, reference = (records[DEDUPLICATED], records[PHYSICAL]) if records else (None, None)
    FIGURE.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(FONT_STYLE), TemporaryDirectory(prefix='.plots-', dir=ROOT / 'analysis') as directory:
        stage = Path(directory)
        for number in args.figures:
            if number == 3:
                fig = figure_components(local, tests)
            elif number == 4:
                fig = figure_training(local, REPRESENTATIONS)
            elif number == 5:
                fig = figure_services(cloud, runs, reference)
            elif number == 6:
                fig = figure_storage(runs, reference)
            elif number == 7:
                with plt.style.context(WHITE_STYLE):
                    fig = figure_dispatch(cloud, read_dispatch_case(cloud), pv_color='#2CA02C', tariff_color='#D62728', include_storage=False)
            elif number == 8:
                fig = figure_training(local, LEARNING_METHODS)
            else:
                fig = figure_benchmark(local)
            name = 'Fig' + str(number)
            if args.language == 'both':
                export_figure(fig, stage, name)
                for lang in ('', '_zh'):
                    for ext in ('pdf', 'png', 'svg'):
                        copy2(stage / (name + lang + '.' + ext), FIGURE / (name + ('_ZH' if lang else '') + '.' + ext))
            else:
                for ext in ('pdf', 'png', 'svg'):
                    with plt.rc_context({'svg.fonttype': 'path'}):
                        fig.savefig(FIGURE / (name + '.' + ext), dpi=DPI)
                plt.close(fig)
            print(FIGURE / (name + '.pdf'))
        if args.supplementary:
            out = FIGURE / 'supplement'
            out.mkdir(exist_ok=True)
            figures = [(figure_training(local, LEARNING_METHODS, zoom=True), 'FigS1'), (figure_washer_checks(runs), 'FigS2'), (figure_training(local, REPRESENTATIONS, zoom=True), 'FigS3')]
            for fig, name in figures:
                if args.language == 'both':
                    export_figure(fig, stage, name)
                    for lang in ('', '_zh'):
                        for ext in ('pdf', 'png', 'svg'):
                            copy2(stage / (name + lang + '.' + ext), out / (name + ('_ZH' if lang else '') + '.' + ext))
                else:
                    for ext in ('pdf', 'png', 'svg'):
                        with plt.rc_context({'svg.fonttype': 'path'}):
                            fig.savefig(out / (name + '.' + ext), dpi=DPI)
                    plt.close(fig)
if __name__ == '__main__':
    main()
