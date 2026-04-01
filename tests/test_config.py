"""Tests for config.py — configuration loading, saving, and model path management."""

import json
from pathlib import Path

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
        assert result == custom

    def test_backfills_missing_default_keys(self, config_dir):
        partial = {"model_path": "/partial/model.gguf"}
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text(json.dumps(partial))
        result = cfg.load_config()
        assert result["n_ctx"] == cfg.DEFAULT_CONFIG["n_ctx"]
        assert result["n_gpu_layers"] == cfg.DEFAULT_CONFIG["n_gpu_layers"]
        assert result["model_path"] == "/partial/model.gguf"

    def test_returns_default_on_corrupt_json(self, config_dir):
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text("{ NOT VALID JSON }")
        result = cfg.load_config()
        assert result == cfg.DEFAULT_CONFIG

    def test_returns_default_on_io_error(self, config_dir, monkeypatch):
        config_dir.mkdir(parents=True, exist_ok=True)
        cfg.CONFIG_FILE.write_text("{}")

        def bad_open(*args, **kwargs):
            raise IOError("permission denied")

        monkeypatch.setattr("builtins.open", bad_open)
        result = cfg.load_config()
        assert result == cfg.DEFAULT_CONFIG


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
