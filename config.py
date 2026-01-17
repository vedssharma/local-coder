import json
import os
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path.home() / ".local-coder"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "model_path": "./Qwen_Qwen2.5-Coder-7B-Instruct-GGUF_qwen2.5-coder-7b-instruct-q4_k_m.gguf",
    "n_ctx": 8192,
    "n_gpu_layers": -1
}


def ensure_config_dir():
    """Create config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def load_config() -> dict:
    """Load configuration from file or create default config."""
    ensure_config_dir()

    if not CONFIG_FILE.exists():
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
        # Ensure all default keys exist
        for key, value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = value
        return config
    except (json.JSONDecodeError, IOError):
        # If config is corrupted, return default
        return DEFAULT_CONFIG.copy()


def save_config(config: dict):
    """Save configuration to file."""
    ensure_config_dir()

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)


def get_model_path() -> str:
    """Get the current model path from config."""
    config = load_config()
    return config.get("model_path", DEFAULT_CONFIG["model_path"])


def set_model_path(path: str) -> bool:
    """
    Set a new model path in config.

    Args:
        path: Path to the GGUF model file

    Returns:
        True if successful, False otherwise
    """
    # Check if file exists
    if not os.path.exists(path):
        return False

    # Check if it's a .gguf file
    if not path.lower().endswith('.gguf'):
        return False

    config = load_config()
    config["model_path"] = path
    save_config(config)
    return True


def get_model_config() -> dict:
    """Get the full model configuration."""
    return load_config()
