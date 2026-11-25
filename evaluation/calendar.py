"""calendar."""
from calendar import monthrange
from datetime import date, timedelta
from random import Random
import numpy as np
from other.environment.config import CONFIG
from other.environment.household import DATA, ROOT, SEED, STEPS_PER_DAY, carbon_days, environment, measured_template
from other.environment.weather import readings
from model.paper.SRB import PhysicalBaselineController
TRAIN_HOMES = tuple(('H' + str(i) for i in range(1, 16)))
TEST_HOMES = tuple(('H' + str(i) for i in range(16, 21)))
SPLIT_SEED = 42
EVALUATION_DAYS = 3
DAY = timedelta(days=1)
PEAK_KEYS = ('episode_peak_import_norm', 'peak_target_margin_norm', 'peak_pressure_norm')

def _available_days(homes):
    weather = readings()
    covered = {d for d in carbon_days() if all(((d, slot) in weather for slot in range(STEPS_PER_DAY)))}
    covered.discard('2020-01-01')
    result = {}
    for home in homes:
        data = measured_template(home, str(DATA), 'ireland_tou_2020', STEPS_PER_DAY)
        result[home] = {date.fromisoformat(d) for d in covered & data.load_days.keys() & data.pv_days.keys()}
    return result

def _assign_windows(available):
    shared = set.intersection(*available.values())
    rng = Random(SPLIT_SEED)
    validation, test, reserved = ([], [], set())
    per_month = len(TEST_HOMES)
    for month in range(1, 13):
        month_days = monthrange(2020, month)[1]
        starts = [date(2020, month, d) for d in range(1, month_days - EVALUATION_DAYS + 2, EVALUATION_DAYS)]
        eligible = [start for start in starts if all((start + i * DAY in shared for i in range(EVALUATION_DAYS)))]
        if len(eligible) < 2:
            raise ValueError(f'Month {month} lacks two complete shared evaluation windows')
        val_start, test_start = rng.sample(eligible, 2)
        offset = (month - 1) % (len(TRAIN_HOMES) // per_month) * per_month
        validation.extend(((h, val_start.isoformat(), EVALUATION_DAYS) for h in TRAIN_HOMES[offset:offset + per_month]))
        test.extend(((h, test_start.isoformat(), EVALUATION_DAYS) for h in TEST_HOMES))
        reserved.update((start + i * DAY for start in (val_start, test_start) for i in range(EVALUATION_DAYS)))
    return (validation, test, reserved)

def build_schedule():
    available = _available_days(TRAIN_HOMES + TEST_HOMES)
    validation, test, reserved = _assign_windows(available)
    training = []
    for home in TRAIN_HOMES:
        valid = available[home] - reserved
        minimum = CONFIG['cross_day']['BLOCK_MIN_DAYS']
        maximum = CONFIG['cross_day']['BLOCK_MAX_DAYS']
        length = minimum
        for start in sorted(valid):
            if start not in valid:
                continue
            window = []
            for offset in range(length):
                day = start + offset * DAY
                if day not in valid:
                    break
                window.append(day)
            if len(window) < minimum:
                continue
            training.append((home, start.isoformat(), len(window)))
            valid.difference_update(window)
            length = minimum if length == maximum else length + 1
    Random(SPLIT_SEED).shuffle(training)
    schedule = dict(train=training, val=validation, test=test)
    validate_schedule(schedule)
    return schedule

def validate_schedule(schedule):
    expected = dict(train=set(TRAIN_HOMES), val=set(TRAIN_HOMES), test=set(TEST_HOMES))
    calendar_days = {}
    for split, blocks in schedule.items():
        if {h for h, _, _ in blocks} != expected[split]:
            raise ValueError(f'Household coverage differs from the registered {split} split')
        cases = set()
        for home, start, length in blocks:
            if not 2 <= length <= 4 or (split != 'train' and length != EVALUATION_DAYS):
                raise ValueError(f'Invalid block length for {home}/{start}')
            days = {(home, date.fromisoformat(start) + i * DAY) for i in range(length)}
            if cases & days:
                raise ValueError(f'Overlapping blocks for {home}/{start} in {split}')
            cases.update(days)
        calendar_days[split] = {day for _, day in cases}
    if calendar_days['train'] & calendar_days['val'] or calendar_days['train'] & calendar_days['test'] or calendar_days['val'] & calendar_days['test']:
        raise ValueError('Calendar dates overlap across training, validation or test')

def check_reset(env, home, start, days):
    if (env._current_episode_house_id, env.current_time, env._current_block_days) != (home, start, days):
        raise ValueError(f'Simulator opened the wrong block for {home}/{start}')

def _calibrate(schedule):
    env = environment(schedule)
    keys = sorted(env.state_space)
    sums = np.zeros(len(keys))
    squares = np.zeros(len(keys))
    count = 0
    carbon_sum = peak_sum = 0.0
    days_seen = 0
    for index, (home, start, days) in enumerate(schedule, 1):
        state = env.reset()
        check_reset(env, home, start, days)
        policy = PhysicalBaselineController(env)
        stop = days * STEPS_PER_DAY
        inputs = np.empty((stop, len(keys)))
        peak = 0.0
        for step in range(stop):
            inputs[step] = [state[k] for k in keys]
            state, _, done = env.step(state, policy.take_action(state))
            if done != (step + 1 == stop):
                raise ValueError(f'Calibration horizon mismatch for {home}/{start}')
            carbon_sum += env.last_step_carbon_intensity_kg_per_kwh
            peak = max(peak, env.current_step_grid_import_kw)
            if (step + 1) % STEPS_PER_DAY == 0:
                peak_sum += peak
                peak = 0.0
                days_seen += 1
        sums += inputs.sum(axis=0)
        squares += np.square(inputs).sum(axis=0)
        count += stop
        if index % 100 == 0 or index == len(schedule):
            print(f'[Calibration] {index}/{len(schedule)} training blocks', flush=True)
    mean = sums / count
    std = np.sqrt(np.maximum(0.0, squares / count - np.square(mean)))
    for index, key in enumerate(keys):
        if std[index] < 1e-06 or key in PEAK_KEYS:
            low, high = env.state_space[key]
            mean[index], std[index] = ((low + high) / 2.0, max((high - low) / 2.0, 1e-06))
    mean_peak = peak_sum / days_seen
    return dict(norm_keys=keys, norm_mean=mean.tolist(), norm_std=std.tolist(), calibration_days=days_seen, mean_train_peak_kw=mean_peak, threshold_ratio=env.peak_target_ratio, threshold_kw=mean_peak * env.peak_target_ratio, carbon_reference=carbon_sum / count)
