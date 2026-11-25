"""evaluate."""
from collections import defaultdict
import csv, gzip, json, os
from hashlib import sha256
from statistics import fmean, stdev
from model.catalog import BASELINES
from model.factory import controller
from other.environment.household import STEPS_PER_DAY, environment
from .metrics import json_scalar, meter
from .io import save
from .calendar import check_reset
from .records import run_root, _paths, _identity, _read_marker, _matches_trace, _read_completed

BATTERIES = {'ess': 'ess_state', 'ev': 'ev_battery_state'}

def inventory_account(start, end, buy_price, sell_price, charge_efficiency, discharge_efficiency):
    ledger = dict(import_price_eur_per_kwh=buy_price, export_price_eur_per_kwh=sell_price, charge_efficiency=charge_efficiency, discharge_efficiency=discharge_efficiency)
    for battery, field in BATTERIES.items():
        initial, final = (float(start[field]), float(end[field]))
        difference = initial - final
        recharge = max(0.0, difference) / charge_efficiency
        export = max(0.0, -difference) * discharge_efficiency
        ledger[battery] = dict(start_kwh=initial, end_kwh=final, net_depletion_kwh=difference, recovery_import_kwh=recharge, recovery_export_kwh=export, adjustment_eur=recharge * buy_price - export * sell_price)
    return ledger

def evaluate(root, method, p, split, device):
    blocks = p['schedule'][split]
    output = root / method / split
    output.mkdir(parents=True, exist_ok=True)
    env = environment(blocks, p['threshold_kw'])
    if method not in BASELINES:
        from model.train import inference
        policy = inference(root, method, p, env, device)
    for index, (home, start, days) in enumerate(blocks, 1):
        state = env.reset()
        check_reset(env, home, start, days)
        initial = {field: state[field] for field in BATTERIES.values()}
        donefile, trace = _paths(output, home, start)
        identity = _identity(p, method, split, (home, start, days))
        if donefile.is_file():
            previous = _read_marker(donefile, identity)
            if _matches_trace(trace, previous):
                continue
            print(f'[Recover] Rebuilding {method} {split} trace for {home}/{start}', flush=True)
        if method in BASELINES:
            policy = controller(method, env, p)
        else:
            policy.reset()
        totals = defaultdict(float)
        peak = price_sum = export_price_sum = 0.0
        steps = identity['steps']
        temporary = trace.with_name(trace.name + '.tmp')
        with gzip.open(temporary, 'wt', encoding='utf8', compresslevel=1) as handle:
            for step in range(steps):
                stamp, slot = (env.current_time, env.current_time_index)
                action = policy.take_action(state)
                nxt, reward, finished = env.step(state, action)
                if finished != (step + 1 == steps):
                    raise ValueError(f'Evaluation horizon mismatch for {home}/{start}')
                metrics = meter(env, state, nxt, reward)
                for key, value in metrics.items():
                    totals[key] += value
                peak = max(peak, env.current_step_grid_import_kw)
                if (step + 1) % STEPS_PER_DAY == 0:
                    totals['daily_peak_kw'] += peak
                    peak = 0.0
                price_sum += state['electricity_price']
                export_price_sum += state['grid_export_price']
                record = dict(house=home, day=stamp, slot=slot, state=state, next_state=nxt, actions=env._last_action_info, reward=reward, metrics=metrics, ledger=env._last_reward_components, reward_terms={k: v[-1] for k, v in env.reward_components.items() if v})
                handle.write(json.dumps(record, default=json_scalar, allow_nan=False, separators=(',', ':')) + '\n')
                state = nxt
        inventory = inventory_account(initial, state, price_sum / steps, export_price_sum / steps, env.charge_efficiency, env.discharge_efficiency)
        totals['inventory_adjustment_eur'] = sum((inventory[dev]['adjustment_eur'] for dev in BATTERIES))
        totals['inventory_aligned_cost_eur'] = totals['cost_eur'] + totals['inventory_adjustment_eur']
        trace_hash = sha256(temporary.read_bytes()).hexdigest()
        temporary.replace(trace)
        save(donefile, {**identity, 'totals': totals, 'inventory': inventory, 'trace_sha256': trace_hash})
        print(f"[{method} {split}] {index}/{len(blocks)} {home}/{start} | return/day={totals['return_total'] / days:.3f} | bill={totals['cost_eur'] / days:.3f} | aligned={totals['inventory_aligned_cost_eur'] / days:.3f}", flush=True)
