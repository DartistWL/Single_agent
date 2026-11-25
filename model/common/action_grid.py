"""action grid."""
from typing import Tuple

def build_wash_schedule_action_tuple(steps_per_day: int, pref: Tuple[int, int], wash_duration_hours: float) -> Tuple[int, ...]:
    spd = int(steps_per_day)
    earliest = int(pref[0] * spd / 24)
    dur_steps = max(1, int(round(float(wash_duration_hours) * spd / 24)))
    latest = int(pref[1] * spd / 24) - dur_steps
    latest = max(earliest, min(spd - 1, latest))
    n_slots = latest - earliest + 1
    if n_slots < 1:
        raise ValueError(f'洗衣机偏好窗过窄：earliest={earliest}, latest={latest}, steps_per_day={spd}')
    return (0,) + tuple(range(1, n_slots + 1))
