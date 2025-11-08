import os
from contextlib import contextmanager
from filelock import FileLock, Timeout

LOCK_FILE = os.path.join(os.path.dirname(__file__), "ml_training.lock")

@contextmanager
def ml_training_lock(timeout=10):
    lock = FileLock(LOCK_FILE)
    try:
        lock.acquire(timeout=timeout)
        yield
    except Timeout:
        raise RuntimeError("Training already in progress.")
    finally:
        if lock.is_locked:
            lock.release()
