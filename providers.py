"""Hosted model providers and API key storage.

Every provider here exposes an OpenAI-compatible chat completions endpoint, so
they all run through the `openai` backend. Keys are read from the provider's
environment variable first, then from a private 0600 file in the config
directory; they are never written to config.json.
"""
import json
import os

import config

# The window the harness budgets against, not each model's maximum: every step resends the
# whole context, so a larger window costs more. Override with `models --context-window`.
DEFAULT_CONTEXT_WINDOW = 128000

PROVIDERS = {
    'openai': {'label': 'OpenAI', 'base_url': 'https://api.openai.com/v1', 'key_env': 'OPENAI_API_KEY',
               'models': ['gpt-5', 'gpt-5-mini', 'gpt-4.1']},
    'anthropic': {'label': 'Anthropic', 'base_url': 'https://api.anthropic.com/v1', 'key_env': 'ANTHROPIC_API_KEY',
                  'models': ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-4-5-20251001'],
                  'headers': {'anthropic-version': '2023-06-01'}, 'key_header': 'x-api-key'},
    'google': {'label': 'Google Gemini', 'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai',
               'key_env': 'GEMINI_API_KEY', 'models': ['gemini-2.5-pro', 'gemini-2.5-flash']},
    'xai': {'label': 'xAI', 'base_url': 'https://api.x.ai/v1', 'key_env': 'XAI_API_KEY',
            'models': ['grok-4', 'grok-code-fast-1']},
    'mistral': {'label': 'Mistral', 'base_url': 'https://api.mistral.ai/v1', 'key_env': 'MISTRAL_API_KEY',
                'models': ['mistral-large-latest', 'codestral-latest']},
    'deepseek': {'label': 'DeepSeek', 'base_url': 'https://api.deepseek.com/v1', 'key_env': 'DEEPSEEK_API_KEY',
                 'models': ['deepseek-chat', 'deepseek-reasoner'], 'context_window': 64000},
}


def keys_file():
    return config.CONFIG_DIR / 'keys.json'


def _load_keys():
    try:
        with open(keys_file()) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_key(provider, key):
    if provider not in PROVIDERS:
        raise ValueError(f'Unknown provider: {provider}')
    key = key.strip()
    if not key:
        raise ValueError('API key is empty')
    config.ensure_config_dir()
    keys = _load_keys()
    keys[provider] = key
    config.write_json_atomic(keys_file(), keys, mode=0o600)


def resolve_key(profile):
    """Return the API key for a profile, or None."""
    provider = PROVIDERS.get(profile.get('provider'))
    if provider:
        env = os.environ.get(provider['key_env'])
        return env or _load_keys().get(profile['provider'])
    return os.environ.get(profile.get('api_key_env', 'LOCAL_CODER_API_KEY')) or None


def auth_headers(profile):
    key = resolve_key(profile)
    provider = PROVIDERS.get(profile.get('provider'), {})
    headers = dict(provider.get('headers', {}))
    if key:
        headers['Authorization'] = 'Bearer ' + key
        if provider.get('key_header'):
            headers[provider['key_header']] = key
    return headers


def provider_profile(provider, model):
    """Profile updates that point the harness at a hosted model."""
    spec = PROVIDERS[provider]
    return {'backend': 'openai', 'provider': provider, 'base_url': spec['base_url'], 'model': model,
            'supports_tools': True, 'stream': True, 'request_timeout': 120,
            'n_ctx': spec.get('context_window', DEFAULT_CONTEXT_WINDOW)}


def local_profile(profile, keep_context=False):
    """Profile for embedded inference. A hosted window would allocate an enormous llama.cpp context."""
    hosted = bool(profile.get('provider'))
    profile = {**profile, 'backend': 'embedded', 'provider': None}
    if hosted and not keep_context:
        profile['n_ctx'] = config.DEFAULT_CONFIG['n_ctx']
    return profile


def select_provider_interactively(ask=None, say=print, ask_secret=None):
    """Prompt for provider, model and key. Returns profile updates, or None if cancelled."""
    import getpass
    ask = ask or input
    ask_secret = ask_secret or getpass.getpass
    names = list(PROVIDERS)
    say('\nHosted providers:')
    for i, name in enumerate(names, 1):
        say(f"  {i}. {PROVIDERS[name]['label']}")
    choice = ask('Provider (number, Enter to cancel)> ').strip()
    if not choice.isdigit() or not 1 <= int(choice) <= len(names):
        return None
    provider = names[int(choice) - 1]
    spec = PROVIDERS[provider]
    say(f"\n{spec['label']} models:")
    for i, m in enumerate(spec['models'], 1):
        say(f'  {i}. {m}')
    choice = ask('Model (number or custom model id)> ').strip()
    if not choice:
        return None
    model = spec['models'][int(choice) - 1] if choice.isdigit() and 1 <= int(choice) <= len(spec['models']) else choice
    profile = provider_profile(provider, model)
    if resolve_key(profile):
        say(f"Using existing {spec['label']} API key (replace by entering a new one).")
    key = ask_secret(f"{spec['label']} API key (input hidden, Enter to keep existing)> ").strip()
    if key:
        save_key(provider, key)
    elif not resolve_key(profile):
        say('No API key provided; cancelled.')
        return None
    return profile


def estimate_cost(profile, prompt_tokens, completion_tokens, cached_prompt_tokens=0):
    """Estimated USD cost from the profile's price (USD per million tokens: input, output and
    optionally cached_input, which defaults to the input price). None when no price is set."""
    price = profile.get('price')
    if not isinstance(price, dict) or price.get('input') is None or price.get('output') is None:
        return None
    cached = min(cached_prompt_tokens, prompt_tokens)
    cached_price = price['input'] if price.get('cached_input') is None else price['cached_input']
    return ((prompt_tokens - cached) * price['input'] + cached * cached_price
            + completion_tokens * price['output']) / 1_000_000


def describe_usage(profile, performance):
    """One line of token usage and estimated cost for a turn, or None if the server reported no usage."""
    if not performance.get('usage_reports'):
        return None
    prompt, cached = performance['prompt_tokens'], performance['cached_prompt_tokens']
    line = f"Usage: {prompt:,} input tokens"
    if cached:
        line += f" ({cached:,} cached)"
    line += f", {performance['completion_tokens']:,} output tokens"
    cost = estimate_cost(profile, prompt, performance['completion_tokens'], cached)
    if cost is not None:
        line += f"; estimated ${cost:.4f}"
    if performance['usage_reports'] < performance.get('model_calls', 0):
        line += f" ({performance['usage_reports']} of {performance['model_calls']} model calls reported usage)"
    return line
