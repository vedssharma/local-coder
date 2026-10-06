"""Retention limits for the harness's own state: undo records, artifacts, traces and sessions."""
import fcntl
import os
from pathlib import Path
import re
import time

# Days to keep each kind of state, measured from its last modification.
DEFAULT_RETENTION = {'undo': 30, 'artifacts': 30, 'traces': 30, 'sessions': 90}

_PATTERNS = {
    'undo': re.compile(r'\d+-[a-f0-9]{32}\.json'),
    'artifacts': re.compile(r'[a-f0-9]{64}\.txt'),
    'traces': re.compile(r'[a-f0-9]{32}\.jsonl'),
}


def retention_days(overrides=None):
    days = dict(DEFAULT_RETENTION)
    for kind, value in (overrides or {}).items():
        if kind not in days:
            raise ValueError(f'Unknown retention kind {kind!r}; use {", ".join(DEFAULT_RETENTION)}')
        if value is not None and (type(value) not in (int, float) or value < 0):
            raise ValueError(f'Retention for {kind} must be a nonnegative number of days, or null to keep forever')
        days[kind] = value
    return days


def prune(workspace, store, days=None, dry_run=False, now=None):
    """Delete state older than its retention. Sessions are only pruned for this workspace, and a
    session in use by another runtime is skipped. Returns {kind: (files, bytes)} of what was (or
    would be) removed."""
    days = retention_days(days)
    now = time.time() if now is None else now
    harness = Path(workspace) / '.local-coder'
    removed = {kind: [0, 0] for kind in DEFAULT_RETENTION}
    if harness.is_symlink():
        raise ValueError('.local-coder cannot be a symlink')
    for kind, pattern in _PATTERNS.items():
        if days[kind] is None:
            continue
        directory = harness / kind
        if directory.is_symlink() or not directory.is_dir():
            continue
        for path in directory.iterdir():
            if pattern.fullmatch(path.name) and not path.is_symlink():
                _remove_if_old(path, now - days[kind] * 86400, dry_run, removed[kind])
    if days['sessions'] is not None:
        cutoff = now - days['sessions'] * 86400
        for key in store.list():
            path = store.directory / (key + '.json')
            lock = store.directory / (key + '.lock')
            try:
                if path.stat().st_mtime >= cutoff or store.workspace_of(key) != store.workspace:
                    continue
            except (OSError, ValueError):
                continue
            if not _unlocked(lock):
                continue
            _remove_if_old(path, cutoff, dry_run, removed['sessions'])
            if not dry_run:
                lock.unlink(missing_ok=True)
    return {kind: tuple(value) for kind, value in removed.items()}


def describe(removed, dry_run=False):
    files = sum(count for count, _ in removed.values())
    if not files:
        return 'Nothing to clean.'
    size = sum(size for _, size in removed.values())
    parts = [f'{count} {kind}' for kind, (count, _) in removed.items() if count]
    verb = 'Would remove' if dry_run else 'Removed'
    return f'{verb} {files} file{"s" if files != 1 else ""} ({size / 1024:.0f} KB): {", ".join(parts)}.'


def _remove_if_old(path, cutoff, dry_run, tally):
    try:
        stat = path.stat()
        if stat.st_mtime >= cutoff:
            return
        if not dry_run:
            path.unlink()
    except FileNotFoundError:
        return
    tally[0] += 1
    tally[1] += stat.st_size


def _unlocked(lock):
    if not lock.exists():
        return True
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_NOFOLLOW)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)
