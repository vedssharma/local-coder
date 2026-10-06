import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

CONFIG_DIR = Path(os.environ.get("LOCAL_CODER_CONFIG_DIR", Path.home() / ".local-coder"))
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "backend": "embedded",
    "supports_tools": True,
    "stream": True,
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
        if not isinstance(config, dict):
            raise ValueError('configuration is not a JSON object')
    except (json.JSONDecodeError, ValueError, UnicodeDecodeError) as exc:
        # Keep the damaged file for recovery instead of letting the next save overwrite its
        # profiles and routes with defaults.
        backup = CONFIG_FILE.with_name(f'{CONFIG_FILE.name}.corrupt-{time.strftime("%Y%m%d-%H%M%S")}')
        os.replace(CONFIG_FILE, backup)
        print(f'local-coder: {CONFIG_FILE} could not be read ({exc}); moved it to {backup} '
              'and started from defaults.', file=sys.stderr)
        return DEFAULT_CONFIG.copy()
    except OSError as exc:
        raise OSError(f'Cannot read {CONFIG_FILE}: {exc}') from exc
    # Ensure all default keys exist
    for key, value in DEFAULT_CONFIG.items():
        if key not in config:
            config[key] = value
    return config


def write_json_atomic(path: Path, data, mode: Optional[int] = None):
    """Write JSON so readers see either the old file or the complete new one, never a partial write."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    try:
        if mode is not None:
            os.fchmod(fd, mode)
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def save_config(config: dict):
    """Save configuration to file."""
    ensure_config_dir()
    write_json_atomic(CONFIG_FILE, config)


def get_model_path() -> str:
    """Get the current model path from config."""
    config = get_model_config()
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

    update_model_config({"model_path": path})
    return True


def get_model_config(name=None, task_kind=None, use_routes=False) -> dict:
    """Get the full model configuration."""
    return resolve_model_config(load_config(), name, task_kind, use_routes)


def _profile_name(name):
    import re
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', name):
        raise ValueError('Profile names use 1-64 letters, numbers, underscores, or hyphens')
    return name


def resolve_model_config(data, name=None, task_kind=None, use_routes=False):
    base = {k: v for k, v in data.items() if k not in ('profiles', 'routes', 'active_profile')}
    if name == 'default':
        return base
    selected = name
    if selected is None and use_routes:
        selected = data.get('routes', {}).get(task_kind)
    selected = selected or data.get('active_profile')
    base = {k: v for k, v in data.items() if k not in ('profiles', 'routes', 'active_profile')}
    if selected:
        _profile_name(selected)
        if selected not in data.get('profiles', {}):
            raise ValueError(f'Unknown model profile: {selected}')
        base = {**DEFAULT_CONFIG, **data['profiles'][selected]}
    return base


def web_enabled(profile, override=None) -> bool:
    """Whether web_search and web_fetch are offered. An explicit choice wins; otherwise they are on
    only for hosted providers, which already send context off the machine."""
    if override is not None:
        return bool(override)
    if profile.get('web') is not None:
        return bool(profile['web'])
    return bool(profile.get('provider'))


def update_model_config(updates):
    data = load_config()
    selected = data.get('active_profile')
    if selected:
        current = resolve_model_config(data)
        current.update(updates)
        data.setdefault('profiles', {})[selected] = current
    else:
        data.update(updates)
    save_config(data)


def save_profile(name):
    name = _profile_name(name)
    if name == 'default':
        raise ValueError('default is reserved for the base configuration')
    data = load_config()
    data.setdefault('profiles', {})[name] = resolve_model_config(data)
    save_config(data)


def activate_profile(name):
    data = load_config()
    if name == 'default':
        data.pop('active_profile', None)
    else:
        _profile_name(name)
        if name not in data.get('profiles', {}):
            raise ValueError(f'Unknown model profile: {name}')
        data['active_profile'] = name
    save_config(data)


def set_route(task_kind, name):
    if task_kind not in ('answer', 'inspect', 'code'):
        raise ValueError('Routes apply to answer, inspect, or code tasks')
    data = load_config()
    if name == 'default':
        data.setdefault('routes', {}).pop(task_kind, None)
    else:
        _profile_name(name)
        if name not in data.get('profiles', {}):
            raise ValueError(f'Unknown model profile: {name}')
        data.setdefault('routes', {})[task_kind] = name
    save_config(data)
