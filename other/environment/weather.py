"""weather."""
import csv
from functools import lru_cache
from pathlib import Path
CSV = Path(__file__).with_name('weather_2020.csv')

@lru_cache(maxsize=1)
def readings():
    with CSV.open(encoding='ascii', newline='') as source:
        return {(r['date'], int(r['slot'])): float(r['drybulb_c']) for r in csv.DictReader(source)}
