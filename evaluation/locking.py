"""Prevent duplicate writes to the same task."""
from contextlib import contextmanager
import os

@contextmanager
def task_lock(folder):
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.run.lock').open('a+b') as handle:
        try:
            if os.name == 'nt':
                import msvcrt
                if not handle.tell():
                    handle.write(b'0')
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise SystemExit(f'Another process is already running this same task: {folder}') from exc
        yield
