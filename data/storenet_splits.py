"""
StoreNet Ireland 2020 date split utilities.

Goal:
1) Deterministically select fixed days per household for Train/Val/Test.
2) Allocate exact overall weekday/weekend quotas before monthly stratification.
3) Distribute each category across months using largest remainders.

This module does NOT require any code changes in the rest of the project;
it provides a clean data structure to drive `episode_schedule` in `environment.py`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd


DATE_FMT = "%Y-%m-%d"


def _is_weekend(date_str: str) -> bool:
    """Mon-Fri=False, Sat/Sun=True."""
    dt = datetime.strptime(date_str, DATE_FMT)
    return dt.weekday() >= 5


def _month_of(date_str: str) -> int:
    dt = datetime.strptime(date_str, DATE_FMT)
    return dt.month


def _allocate_counts_by_ratio(total_count: int, ratios: List[float], rng: random.Random) -> List[int]:
    """
    Allocate `total_count` items to buckets using expected ratios, but return integers summing exactly.
    Uses largest remainder method with deterministic tie-breaking via RNG.
    """
    if total_count < 0:
        raise ValueError("total_count must be >=0")
    if not ratios:
        return []

    s = sum(ratios)
    if s <= 0:
        # Degenerate: equally distribute
        base = total_count // len(ratios)
        rem = total_count - base * len(ratios)
        out = [base] * len(ratios)
        idxs = list(range(len(ratios)))
        rng.shuffle(idxs)
        for i in idxs[:rem]:
            out[i] += 1
        return out

    expected = [total_count * (r / s) for r in ratios]
    floor_counts = [int(x) for x in expected]
    assigned = sum(floor_counts)
    remainder = total_count - assigned
    # Largest remainder: fractional part
    frac = [x - int(x) for x in expected]

    # Deterministic shuffle for exact ties
    candidates = list(range(len(ratios)))
    rng.shuffle(candidates)
    candidates.sort(key=lambda i: frac[i], reverse=True)

    out = floor_counts[:]
    for i in candidates[:remainder]:
        out[i] += 1
    return out


def _sample_balanced_for_split(
    all_dates: List[str],
    total_days: int,
    seed: int,
) -> List[str]:
    """Sample without replacement with exact overall calendar-category quotas.

    Allocating weekdays separately *inside* every month with ``int(n * p)``
    systematically assigned one-day monthly samples to weekends. Allocate the
    two overall quotas once, then distribute each quota across its months by
    largest remainders. This keeps rounding error below one day globally.
    """
    rng = random.Random(seed)
    if total_days < 0:
        raise ValueError('total_days must be nonnegative')
    if len(set(all_dates)) != len(all_dates):
        raise ValueError('Date pools must contain unique dates')
    if total_days > len(all_dates):
        raise ValueError(f"Requested total_days={total_days} but only {len(all_dates)} available.")
    categories = [[d for d in sorted(all_dates) if _is_weekend(d) == weekend]
                  for weekend in (False, True)]
    quotas = _allocate_counts_by_ratio(total_days, [len(pool) for pool in categories], rng)
    selected: List[str] = []
    for quota, pool in zip(quotas, categories):
        months = [[d for d in pool if _month_of(d) == month] for month in range(1, 13)]
        counts = _allocate_counts_by_ratio(quota, [len(bucket) for bucket in months], rng)
        for count, bucket in zip(counts, months):
            if count > len(bucket):
                raise RuntimeError('Monthly allocation exceeds available dates')
            rng.shuffle(bucket)
            selected.extend(bucket[:count])
    if len(selected) != total_days or len(set(selected)) != total_days:
        raise RuntimeError(f"Split selection size mismatch: got {len(selected)} expected {total_days}")
    rng.shuffle(selected)
    return selected


@dataclass(frozen=True)
class StoreNetSplitConfig:
    seed: int = 42
    train_days: int = 100
    val_days: int = 30
    test_days: int = 100


def make_storenet_splits_2020(
    base_dir: str = "data/storenet_ireland_2020",
    houses_train: Sequence[str] = (),
    houses_val: Sequence[str] = (),
    houses_test: Sequence[str] = (),
    config: StoreNetSplitConfig = StoreNetSplitConfig(seed=42, train_days=100, val_days=30, test_days=100),
) -> Dict[str, Dict[str, List[str]]]:
    """
    Return:
      {house_id: {'train': [...], 'val': [...], 'test': [...]}}
    """
    base = Path(base_dir)
    if not base.exists():
        raise SystemExit(f"base_dir not found: {base}")

    if not houses_train and not houses_val and not houses_test:
        raise ValueError("At least one of houses_train/houses_val/houses_test must be provided.")

    houses_all = list(dict.fromkeys([*houses_train, *houses_val, *houses_test]))

    rng_master = random.Random(config.seed)
    splits: Dict[str, Dict[str, List[str]]] = {}

    for hid in houses_all:
        hid_u = str(hid).strip().upper()
        pivot_path = base / f"daily_pivot_cons_2020_{hid_u}.csv"
        if not pivot_path.exists():
            raise SystemExit(f"Missing pivot file for house {hid_u}: {pivot_path}")

        # Read only the `date` column
        df = pd.read_csv(pivot_path, usecols=["date"])
        dates = df["date"].astype(str).tolist()

        if not dates:
            raise SystemExit(f"No dates found in {pivot_path}")

        # Train
        train_seed = config.seed * 1000 + int(hid_u[1:]) * 7 + 1
        train_selected = _sample_balanced_for_split(
            all_dates=dates,
            total_days=config.train_days,
            seed=train_seed,
        )

        remaining_after_train = [d for d in dates if d not in set(train_selected)]

        # Val
        val_seed = config.seed * 1000 + int(hid_u[1:]) * 7 + 2
        val_selected = _sample_balanced_for_split(
            all_dates=remaining_after_train,
            total_days=config.val_days,
            seed=val_seed,
        )
        remaining_after_val = [d for d in remaining_after_train if d not in set(val_selected)]

        # Test
        test_seed = config.seed * 1000 + int(hid_u[1:]) * 7 + 3
        test_selected = _sample_balanced_for_split(
            all_dates=remaining_after_val,
            total_days=config.test_days,
            seed=test_seed,
        )

        splits[hid_u] = {
            "train": sorted(train_selected),
            "val": sorted(val_selected),
            "test": sorted(test_selected),
        }

    return splits


def flatten_episode_schedule(
    splits: Dict[str, Dict[str, List[str]]],
    house_ids: Sequence[str],
    split_name: str,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Flatten {house->dates} into a shuffled (house_id, date) episode schedule."""
    rng = random.Random(seed)
    schedule: List[Tuple[str, str]] = []
    for hid in house_ids:
        hid_u = str(hid).strip().upper()
        for d in splits[hid_u][split_name]:
            schedule.append((hid_u, d))
    rng.shuffle(schedule)
    return schedule

