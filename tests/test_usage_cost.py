"""Token usage and estimated cost per turn for hosted models."""

from unittest.mock import MagicMock

from typer.testing import CliRunner

import providers
from agent import run_agent


def _answer(usage):
    return {'choices': [{'message': {'content': 'done'}, 'finish_reason': 'stop'}], 'usage': usage}


def test_run_totals_reported_usage():
    model = MagicMock()
    model.create_chat_completion.return_value = _answer(
        {'prompt_tokens': 1200, 'completion_tokens': 30, 'prompt_tokens_details': {'cached_tokens': 1000}})
    performance = run_agent(model, [{'role': 'user', 'content': 'hi'}]).performance
    assert (performance['prompt_tokens'], performance['completion_tokens'], performance['cached_prompt_tokens'],
            performance['usage_reports']) == (1200, 30, 1000, 1)


def test_cost_uses_cached_price_and_falls_back_to_input_price():
    profile = {'price': {'input': 3, 'output': 15, 'cached_input': 0.3}}
    assert providers.estimate_cost(profile, 1_000_000, 100_000, 400_000) == 600_000 * 3 / 1e6 + 400_000 * 0.3 / 1e6 + 1.5
    assert providers.estimate_cost({'price': {'input': 2, 'output': 8}}, 500_000, 0, 500_000) == 1.0
    assert providers.estimate_cost({}, 10, 10) is None


def test_usage_line():
    performance = {'usage_reports': 1, 'model_calls': 2, 'prompt_tokens': 12000,
                   'cached_prompt_tokens': 8000, 'completion_tokens': 450}
    line = providers.describe_usage({'price': {'input': 3, 'output': 15}}, performance)
    assert line == ('Usage: 12,000 input tokens (8,000 cached), 450 output tokens; estimated $0.0428 '
                    '(1 of 2 model calls reported usage)')
    assert providers.describe_usage({}, {**performance, 'usage_reports': 0}) is None
    assert 'estimated' not in providers.describe_usage({}, performance)


def test_prices_are_saved_per_profile_and_cleared_on_provider_change(config_dir, monkeypatch):
    import config
    from main import app
    monkeypatch.setenv('OPENAI_API_KEY', 'k')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'k')
    runner = CliRunner()
    assert runner.invoke(app, ['models', '--provider', 'openai', '--model-name', 'gpt-5-mini']).exit_code == 0
    result = runner.invoke(app, ['models', '--price-input', '0.25', '--price-output', '2'])
    assert result.exit_code == 0, result.output
    assert config.get_model_config()['price'] == {'input': 0.25, 'output': 2.0}
    assert 'Price per million tokens: $0.25 input, $2 output' in runner.invoke(app, ['models']).output
    assert runner.invoke(app, ['models', '--provider', 'anthropic']).exit_code == 0
    assert config.get_model_config()['price'] is None


def test_turn_prints_usage_for_hosted_profiles(capsys):
    from main import execute_turn
    runtime = MagicMock()
    runtime.tools.root = '.'
    runtime.session_id = 'abc'
    runtime.profile = {'provider': 'openai', 'price': {'input': 1, 'output': 2}}
    runtime.turn.return_value = MagicMock(status='completed', performance={
        'usage_reports': 1, 'model_calls': 1, 'prompt_tokens': 1000, 'cached_prompt_tokens': 0, 'completion_tokens': 500})
    execute_turn(runtime, 'hi', 64)
    assert 'Usage: 1,000 input tokens, 500 output tokens; estimated $0.0020' in capsys.readouterr().out


def test_saved_keys_must_name_a_provider_and_be_nonempty(config_dir):
    import pytest
    with pytest.raises(ValueError, match='Unknown provider'):
        providers.save_key('nobody', 'k')
    with pytest.raises(ValueError, match='API key is empty'):
        providers.save_key('openai', '   ')


def _select(answers, secret='', said=None):
    answers = iter(answers)
    return providers.select_provider_interactively(ask=lambda _: next(answers), say=(said if said is not None else []).append,
                                                   ask_secret=lambda _: secret)


def test_interactive_provider_selection_can_be_cancelled(config_dir, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert _select(['']) is None
    assert _select(['99']) is None
    assert _select(['1', '']) is None
    said = []
    assert _select(['1', '1'], said=said) is None
    assert 'No API key provided; cancelled.' in said


def test_interactive_selection_keeps_an_existing_key_and_custom_model(config_dir, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'from-env')
    said = []
    profile = _select(['1', 'my-custom-model'], said=said)
    assert profile['provider'] == 'openai' and profile['model'] == 'my-custom-model'
    assert any('Using existing' in line for line in said)
