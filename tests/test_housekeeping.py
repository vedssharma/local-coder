"""Retention limits and the clean command."""

import os
import time

import pytest

import housekeeping
from session import SessionStore

DAY = 86400


def _age(path, days):
    stamp = time.time() - days * DAY
    os.utime(path, (stamp, stamp))


def _state(tmp_path):
    workspace = tmp_path / 'repo'
    harness = workspace / '.local-coder'
    for kind in ('undo', 'artifacts', 'traces'):
        (harness / kind).mkdir(parents=True)
    files = {
        'old_undo': harness / 'undo' / f'1-{"a" * 32}.json',
        'new_undo': harness / 'undo' / f'2-{"b" * 32}.json',
        'old_artifact': harness / 'artifacts' / f'{"c" * 64}.txt',
        'old_trace': harness / 'traces' / f'{"d" * 32}.jsonl',
        'stray': harness / 'undo' / 'notes.txt',
    }
    for path in files.values():
        path.write_text('x' * 10)
    for name in ('old_undo', 'old_artifact', 'old_trace', 'stray'):
        _age(files[name], 40)
    store = SessionStore(tmp_path / 'state' / 'sessions', workspace)
    other = SessionStore(tmp_path / 'state' / 'sessions', tmp_path / 'elsewhere')
    sessions = {'old': store.save([{'role': 'user', 'content': 'fix the parser'}]),
                'new': store.save([{'role': 'user', 'content': 'add a flag'}]),
                'other': other.save([{'role': 'user', 'content': 'not ours'}])}
    for name in ('old', 'other'):
        _age(store.directory / (sessions[name] + '.json'), 100)
    return workspace, store, files, sessions


def test_prune_removes_only_old_harness_files_and_this_workspaces_old_sessions(tmp_path):
    workspace, store, files, sessions = _state(tmp_path)
    preview = housekeeping.prune(workspace, store, dry_run=True)
    assert preview == {'undo': (1, 10), 'artifacts': (1, 10), 'traces': (1, 10), 'sessions': (1, preview['sessions'][1])}
    assert files['old_undo'].exists()
    housekeeping.prune(workspace, store)
    assert not files['old_undo'].exists() and not files['old_artifact'].exists() and not files['old_trace'].exists()
    assert files['new_undo'].exists() and files['stray'].exists()
    remaining = set(store.list())
    assert sessions['old'] not in remaining and {sessions['new'], sessions['other']} <= remaining


def test_null_retention_keeps_that_kind_forever(tmp_path):
    workspace, store, files, sessions = _state(tmp_path)
    removed = housekeeping.prune(workspace, store, {'undo': None, 'sessions': None})
    assert files['old_undo'].exists() and removed['undo'] == (0, 0)
    assert sessions['old'] in store.list()
    with pytest.raises(ValueError, match='Unknown retention kind'):
        housekeeping.retention_days({'logs': 3})


def test_a_session_in_use_is_not_pruned(tmp_path):
    workspace, store, files, sessions = _state(tmp_path)
    with store.lease(sessions['old']):
        housekeeping.prune(workspace, store)
    assert sessions['old'] in store.list()


def test_session_summaries_show_date_and_first_prompt_for_this_workspace(tmp_path):
    workspace, store, files, sessions = _state(tmp_path)
    store.save([{'role': 'system', 'content': 'sys'},
                {'role': 'user', 'content': "The user has pre-loaded the following files for reference:\n\n"
                                            "<file path='a.py'>\nx\n</file>\n\nUser request: explain   a.py"}])
    summaries = store.summaries()
    assert [s['prompt'] for s in summaries] == ['explain a.py', 'add a flag', 'fix the parser']
    from main import format_sessions
    lines = format_sessions(summaries).splitlines()
    assert lines[-1].startswith(sessions['old']) and lines[-1].endswith('fix the parser')
    assert time.strftime('%Y-%m-%d') in lines[0]


def test_clean_command(tmp_path, monkeypatch, config_dir):
    from typer.testing import CliRunner
    from main import app
    monkeypatch.chdir(tmp_path)
    (tmp_path / '.local-coder' / 'undo').mkdir(parents=True)
    record = tmp_path / '.local-coder' / 'undo' / f'1-{"a" * 32}.json'
    record.write_text('{}')
    _age(record, 3)
    runner = CliRunner()
    assert 'Nothing to clean.' in runner.invoke(app, ['clean']).output
    result = runner.invoke(app, ['clean', '--older-than', '1', '--dry-run'])
    assert 'Would remove 1 file' in result.output and record.exists()
    assert 'Removed 1 file' in runner.invoke(app, ['clean', '--older-than', '1']).output
    assert not record.exists()


def test_retention_values_must_be_nonnegative_numbers():
    for value in (-1, '30', True):
        with pytest.raises(ValueError, match='nonnegative number of days'):
            housekeeping.retention_days({'undo': value})


def test_a_symlinked_harness_directory_is_refused(tmp_path):
    (tmp_path / 'elsewhere').mkdir()
    (tmp_path / '.local-coder').symlink_to(tmp_path / 'elsewhere')
    with pytest.raises(ValueError, match='cannot be a symlink'):
        housekeeping.prune(tmp_path, SessionStore(tmp_path / 'sessions', tmp_path))


def test_unreadable_and_locked_sessions_are_kept(tmp_path):
    import fcntl
    store = SessionStore(tmp_path / 'sessions', tmp_path)
    corrupt, locked = store.save([]), store.save([])
    (store.directory / (corrupt + '.json')).write_text('{broken')
    for key in (corrupt, locked):
        _age(store.directory / (key + '.json'), 200)
    lock = store.directory / (locked + '.lock')
    lock.touch()
    fd = os.open(lock, os.O_WRONLY)
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        removed = housekeeping.prune(tmp_path, store)
    finally:
        os.close(fd)
    assert removed['sessions'] == (0, 0)
    assert sorted(store.list()) == sorted([corrupt, locked])


def test_a_lock_that_cannot_be_opened_keeps_its_session(tmp_path):
    store = SessionStore(tmp_path / 'sessions', tmp_path)
    key = store.save([])
    _age(store.directory / (key + '.json'), 200)
    (tmp_path / 'target').touch()
    (store.directory / (key + '.lock')).symlink_to(tmp_path / 'target')  # O_NOFOLLOW refuses it.
    assert housekeeping.prune(tmp_path, store)['sessions'] == (0, 0)


def test_unlocked_old_sessions_are_removed_with_their_lock(tmp_path):
    store = SessionStore(tmp_path / 'sessions', tmp_path)
    key = store.save([])
    with store.lease(key):
        pass
    _age(store.directory / (key + '.json'), 200)
    assert housekeeping.prune(tmp_path, store)['sessions'][0] == 1
    assert not (store.directory / (key + '.lock')).exists()


def test_files_that_vanish_during_pruning_are_skipped(tmp_path):
    from pathlib import Path
    tally = [0, 0]
    housekeeping._remove_if_old(Path(tmp_path / 'gone.json'), time.time(), False, tally)
    assert tally == [0, 0]
