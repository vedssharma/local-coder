"""Tests for config.py — configuration loading, saving, and model path management."""

import json

import pytest

import config as cfg


# ---------------------------------------------------------------------------
# ensure_config_dir
# ---------------------------------------------------------------------------

class TestEnsureConfigDir:
    def test_creates_directory(self, config_dir):
        assert not config_dir.exists()
        cfg.ensure_config_dir()
        assert config_dir.exists()

    def test_idempotent(self, config_dir):
        cfg.ensure_config_dir()
        cfg.ensure_config_dir()  # calling twice must not raise
        assert config_dir.exists()


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_creates_default_when_missing(self, config_dir):
        result = cfg.load_config()
        assert result == cfg.DEFAULT_CONFIG

    def test_default_config_file_written(self, config_dir):
        cfg.load_config()
        assert cfg.CONFIG_FILE.exists()
        stored = json.loads(cfg.CONFIG_FILE.read_text())
        assert stored == cfg.DEFAULT_CONFIG

    def test_reads_existing_config(self, config_dir):
        custom = {"model_path": "/custom/model.gguf", "n_ctx": 4096, "n_gpu_layers": 0}
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text(json.dumps(custom))
        result = cfg.load_config()
        assert result == {**cfg.DEFAULT_CONFIG, **custom}

    def test_backfills_missing_default_keys(self, config_dir):
        partial = {"model_path": "/partial/model.gguf"}
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text(json.dumps(partial))
        result = cfg.load_config()
        assert result["n_ctx"] == cfg.DEFAULT_CONFIG["n_ctx"]
        assert result["n_gpu_layers"] == cfg.DEFAULT_CONFIG["n_gpu_layers"]
        assert result["model_path"] == "/partial/model.gguf"

    def test_corrupt_json_is_moved_aside_not_overwritten(self, config_dir, capsys):
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text('{ "profiles": {"big": {} } NOT VALID JSON }')
        result = cfg.load_config()
        assert result == cfg.DEFAULT_CONFIG
        backups = list(config_dir.glob('config.json.corrupt-*'))
        assert len(backups) == 1 and 'profiles' in backups[0].read_text()
        assert 'moved it to' in capsys.readouterr().err
        cfg.save_config(result)
        assert backups[0].exists()

    def test_non_object_json_is_treated_as_corrupt(self, config_dir):
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text('[1, 2]')
        assert cfg.load_config() == cfg.DEFAULT_CONFIG
        assert list(config_dir.glob('config.json.corrupt-*'))

    def test_unreadable_config_raises_instead_of_resetting(self, config_dir, monkeypatch):
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text("{}")

        def bad_open(*args, **kwargs):
            raise IOError("permission denied")

        monkeypatch.setattr("builtins.open", bad_open)
        with pytest.raises(OSError, match='Cannot read'):
            cfg.load_config()


class TestAtomicWrite:
    def test_failed_save_keeps_the_previous_file(self, config_dir, monkeypatch):
        cfg.save_config({'profiles': {'keep': {}}})

        def broken_dump(*args, **kwargs):
            args[1].write('{"partial": ')
            raise RuntimeError('disk full')

        with monkeypatch.context() as patch:
            patch.setattr(cfg.json, 'dump', broken_dump)
            with pytest.raises(RuntimeError):
                cfg.save_config({'profiles': {}})
        assert json.loads(cfg.CONFIG_FILE.read_text()) == {'profiles': {'keep': {}}}
        assert not [p for p in config_dir.iterdir() if p.name.endswith('.tmp')]

    def test_mode_is_applied(self, tmp_path):
        target = tmp_path / 'keys.json'
        cfg.write_json_atomic(target, {'a': 'b'}, mode=0o600)
        assert target.stat().st_mode & 0o777 == 0o600
        assert json.loads(target.read_text()) == {'a': 'b'}


# ---------------------------------------------------------------------------
# save_config
# ---------------------------------------------------------------------------

class TestSaveConfig:
    def test_writes_json(self, config_dir):
        data = {"model_path": "/some/model.gguf", "n_ctx": 2048, "n_gpu_layers": 8}
        cfg.save_config(data)
        stored = json.loads(cfg.CONFIG_FILE.read_text())
        assert stored == data

    def test_overwrites_existing(self, config_dir):
        cfg.save_config({"model_path": "/old.gguf", "n_ctx": 1024, "n_gpu_layers": 0})
        new = {"model_path": "/new.gguf", "n_ctx": 8192, "n_gpu_layers": -1}
        cfg.save_config(new)
        stored = json.loads(cfg.CONFIG_FILE.read_text())
        assert stored == new


# ---------------------------------------------------------------------------
# get_model_path
# ---------------------------------------------------------------------------

class TestGetModelPath:
    def test_returns_default_when_no_config(self, config_dir):
        path = cfg.get_model_path()
        assert path == cfg.DEFAULT_CONFIG["model_path"]

    def test_returns_stored_path(self, config_dir, tmp_path):
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text(json.dumps({"model_path": "/stored/model.gguf"}))
        # Temporarily reload — get_model_path always reads fresh
        path = cfg.get_model_path()
        assert path == "/stored/model.gguf"


# ---------------------------------------------------------------------------
# set_model_path
# ---------------------------------------------------------------------------

class TestSetModelPath:
    def test_returns_false_for_nonexistent_file(self, config_dir):
        assert cfg.set_model_path("/nonexistent/model.gguf") is False

    def test_returns_false_for_non_gguf_extension(self, config_dir, tmp_path):
        bad_file = tmp_path / "model.bin"
        bad_file.write_text("data")
        assert cfg.set_model_path(str(bad_file)) is False

    def test_returns_true_and_persists_for_valid_gguf(self, config_dir, sample_gguf_file):
        result = cfg.set_model_path(sample_gguf_file)
        assert result is True
        stored = json.loads(cfg.CONFIG_FILE.read_text())
        assert stored["model_path"] == sample_gguf_file

    def test_preserves_other_config_keys(self, config_dir, sample_gguf_file):
        config_dir.mkdir(parents=True, exist_ok=True)
        initial = {"model_path": "/old.gguf", "n_ctx": 1234, "n_gpu_layers": 5}
        cfg.CONFIG_FILE.write_text(json.dumps(initial))
        cfg.set_model_path(sample_gguf_file)
        stored = json.loads(cfg.CONFIG_FILE.read_text())
        assert stored["n_ctx"] == 1234
        assert stored["n_gpu_layers"] == 5


# ---------------------------------------------------------------------------
# get_model_config
# ---------------------------------------------------------------------------

class TestGetModelConfig:
    def test_returns_full_config_dict(self, config_dir):
        result = cfg.get_model_config()
        assert "model_path" in result
        assert "n_ctx" in result
        assert "n_gpu_layers" in result

    def test_reflects_saved_values(self, config_dir):
        custom = {"model_path": "/x.gguf", "n_ctx": 512, "n_gpu_layers": 2}
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text(json.dumps(custom))
        result = cfg.get_model_config()
        assert result["n_ctx"] == 512


def test_profiles_and_opt_in_routing_preserve_settings(config_dir):
    cfg.save_config({**cfg.DEFAULT_CONFIG, 'n_threads': 2})
    cfg.save_profile('small')
    cfg.update_model_config({'n_threads': 4})
    cfg.save_profile('large')
    cfg.activate_profile('small')
    cfg.update_model_config({'n_ctx': 2048})
    cfg.set_route('code', 'large')
    assert cfg.get_model_config()['n_threads'] == 2
    assert cfg.get_model_config()['n_ctx'] == 2048
    assert cfg.get_model_config(task_kind='code', use_routes=True)['n_threads'] == 4
    assert cfg.get_model_config('small', 'code', True)['n_threads'] == 2
    cfg.activate_profile('default')
    assert cfg.get_model_config()['n_threads'] == 4
    assert 'small' in cfg.load_config()['profiles']
    with pytest.raises(ValueError):
        cfg.get_model_config('unknown')


def test_named_profile_is_a_snapshot_not_affected_by_new_global_tuning(config_dir):
    cfg.save_profile('saved')
    cfg.update_model_config({'n_threads': 8})
    assert 'n_threads' not in cfg.get_model_config('saved')
    cfg.activate_profile('saved')
    assert cfg.get_model_config('default')['n_threads'] == 8
    with pytest.raises(ValueError):
        cfg.save_profile('default')


# ---------------------------------------------------------------------------
# Atomic writes, profile names and routes
# ---------------------------------------------------------------------------

def test_atomic_write_cleans_up_after_a_failure(tmp_path, monkeypatch):
    import os
    target = tmp_path / 'data.json'
    target.write_text('{"old": true}')
    monkeypatch.setattr(os, 'replace', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        cfg.write_json_atomic(target, {'new': True})
    assert target.read_text() == '{"old": true}'
    assert [p.name for p in tmp_path.iterdir()] == ['data.json']


def test_atomic_write_tolerates_a_temporary_file_already_gone(tmp_path, monkeypatch):
    import os

    def vanish(temporary, path):
        os.unlink(temporary)
        raise OSError('lost')
    monkeypatch.setattr(os, 'replace', vanish)
    with pytest.raises(OSError, match='lost'):
        cfg.write_json_atomic(tmp_path / 'data.json', {})


def test_profile_names_and_routes_are_validated(config_dir):
    for name in ('', 'has space', 'x' * 65, None):
        with pytest.raises(ValueError, match='Profile names'):
            cfg.save_profile(name)
    with pytest.raises(ValueError, match='Unknown model profile'):
        cfg.activate_profile('missing')
    with pytest.raises(ValueError, match='Routes apply to'):
        cfg.set_route('everything', 'default')
    with pytest.raises(ValueError, match='Unknown model profile'):
        cfg.set_route('code', 'missing')
    cfg.save_profile('big')
    cfg.set_route('code', 'big')
    assert cfg.load_config()['routes'] == {'code': 'big'}
    cfg.set_route('code', 'default')
    assert cfg.load_config()['routes'] == {}
