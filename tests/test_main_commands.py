"""CLI commands in main.py beyond `models` and `ask`: chat, /md, clean, undo, benchmark, profiles, inference-server."""

import io
import json
import re
from unittest.mock import MagicMock, patch

import pytest
import typer
from rich.console import Console
from typer.testing import CliRunner

import config
import main
from main import app

runner = CliRunner()


def _plain(output):
    """Usage errors as plain text: Rich colors and boxes them when it detects CI."""
    return ' '.join(re.sub(r'[│╭╮╰╯─]', ' ', re.sub(r'\x1b\[[0-9;]*m', '', output)).split())


def _model(*texts, finish_reason='stop', usage=None):
    model = MagicMock()
    replies = [{'choices': [{'message': {'content': text}, 'finish_reason': finish_reason}],
                **({'usage': usage} if usage else {})} for text in texts or ('done',)]
    model.create_chat_completion.side_effect = replies + [replies[-1]] * 20
    return model


@pytest.fixture
def workspace(config_dir, tmp_path, monkeypatch):
    root = tmp_path / 'work'
    root.mkdir()
    monkeypatch.chdir(root)
    monkeypatch.setattr(main, 'llm', None)
    return root


# ---------------------------------------------------------------------------
# get_llm and make_runtime
# ---------------------------------------------------------------------------

def test_get_llm_creates_the_model_once(workspace):
    created = []
    with patch('main.create_model', side_effect=lambda profile: created.append(profile) or object()):
        first = main.get_llm()
        assert main.get_llm() is first
    assert len(created) == 1


def test_make_runtime_rejects_unknown_modes(workspace):
    with pytest.raises(typer.BadParameter):
        main.make_runtime('root', 1, 1, 1, False, Console())


def test_make_runtime_uses_a_named_profile_and_persistent_daemon(workspace):
    config.save_profile('small')
    with patch('main.create_model', return_value=MagicMock()) as create, \
            patch('inference_daemon.PersistentModel') as persistent:
        runtime = main.make_runtime('read-only', 1, 1, 1, False, Console(), profile_name='small', persistent=True)
    with runtime:
        assert create.called and runtime.model is persistent.return_value
        assert runtime.profile['n_ctx'] == config.get_model_config('small')['n_ctx']


def test_runtime_events_render_to_the_console(workspace):
    out = io.StringIO()
    console = Console(file=out, width=200)
    with patch('main.get_llm', return_value=MagicMock()):
        runtime = main.make_runtime('read-only', 1, 1, 1, False, console)
    with runtime:
        emit = runtime.sink
        emit({'type': 'assistant_text', 'text': '**rendered** markdown'})
        emit({'type': 'tool_started', 'name': 'read'})
        emit({'type': 'model_retry', 'error': 'HTTP 503', 'delay_seconds': 2.0})
        emit({'type': 'verification_result', 'changed_files': ['a.py'], 'checks': {},
              'outstanding_processes': [], 'verification_status': 'unverified'})
        emit({'type': 'verification_result', 'changed_files': [], 'checks': {},
              'outstanding_processes': [], 'verification_status': 'not_applicable'})
        emit({'type': 'run_finished', 'status': 'blocked', 'reason': 'model_error'})
        emit({'type': 'assistant_delta', 'text': 'streamed text'})
        emit({'type': 'assistant_text', 'text': 'not printed twice'})
        emit({'type': 'run_finished', 'status': 'completed', 'reason': None})
        emit({'type': 'assistant_text', 'text': 'printed after the stream reset'})
    text = out.getvalue()
    assert 'rendered markdown' in text and '**' not in text
    assert 'Tool: read' in text
    assert 'Model request failed (HTTP 503); retrying in 2s' in text
    assert text.count('Verification:') == 1 and 'Verification: unverified' in text
    assert 'Run blocked: model_error' in text
    assert 'streamed text' in text and 'not printed twice' not in text
    assert 'printed after the stream reset' in text


def test_trace_events_are_written_per_session(workspace):
    with patch('main.get_llm', return_value=_model('traced')):
        result = runner.invoke(app, ['ask', 'hello', '--trace'])
    assert result.exit_code == 0, result.output
    traces = list((workspace / '.local-coder' / 'traces').glob('*.jsonl'))
    assert len(traces) == 1
    events = [json.loads(line)['type'] for line in traces[0].read_text().splitlines()]
    assert 'run_finished' in events


# ---------------------------------------------------------------------------
# ask / edit outcomes and usage reporting
# ---------------------------------------------------------------------------

def test_ask_exits_nonzero_and_prints_the_reason_when_a_run_does_not_complete(workspace):
    with patch('main.get_llm', return_value=_model('cut off', finish_reason='content_filter')):
        result = runner.invoke(app, ['ask', 'hello'])
    assert result.exit_code == 1
    assert 'outcome: blocked' in result.output
    assert 'Model could not complete this request.' in result.output


def test_edit_runs_in_workspace_edit_mode(workspace):
    model = _model('edited')
    with patch('main.get_llm', return_value=model):
        result = runner.invoke(app, ['edit', 'change it'])
    assert result.exit_code == 0, result.output
    names = {t['function']['name'] for t in model.create_chat_completion.call_args.kwargs['tools']}
    assert {'write', 'edit'} <= names


def test_edit_exits_nonzero_when_blocked(workspace):
    with patch('main.get_llm', return_value=_model('no', finish_reason='error')):
        result = runner.invoke(app, ['edit', 'change it'])
    assert result.exit_code == 1


def test_hosted_profiles_report_usage_and_cost(workspace, monkeypatch):
    import providers
    monkeypatch.setenv('OPENAI_API_KEY', 'k')
    config.update_model_config({**config.get_model_config(), **providers.provider_profile('openai', 'gpt-5-mini'),
                                'price': {'input': 1.0, 'output': 2.0}})
    model = _model('hi', usage={'prompt_tokens': 1000, 'completion_tokens': 500})
    with patch('main.get_llm', return_value=model):
        result = runner.invoke(app, ['ask', 'hello'])
    assert result.exit_code == 0, result.output
    assert 'Usage: 1,000 input tokens, 500 output tokens; estimated $0.0020' in result.output


def test_format_sessions():
    assert main.format_sessions([]) == 'No saved sessions for this workspace.'
    lines = main.format_sessions([{'id': 'abc', 'modified': 0, 'prompt': 'short'},
                                  {'id': 'def', 'modified': 0, 'prompt': 'x' * 80}]).splitlines()
    assert lines[0].startswith('abc  ') and lines[0].endswith('short')
    assert lines[1].endswith('x' * 57 + '...')


# ---------------------------------------------------------------------------
# chat loop
# ---------------------------------------------------------------------------

def _chat(lines, model, *args):
    with patch('main.get_llm', return_value=model):
        return runner.invoke(app, ['chat', *args], input='\n'.join(lines) + '\n')


def test_chat_runs_turns_and_slash_commands(workspace):
    model = _model('first answer', 'second answer')
    result = _chat(['hello', '/sessions', '/acknowledge-interrupted', '/undo', '/undo turn', '/new', '', 'again', '/exit'],
                   model, '--mode', 'workspace-edit')
    assert result.exit_code == 0, result.output
    assert result.output.count('Session: ') == 2
    assert 'hello' in result.output  # listed by /sessions
    assert 'Interrupted operations acknowledged' in result.output
    assert result.output.count('No harness edits to undo.') == 2
    sessions = list((config.CONFIG_DIR / 'sessions').glob('*.json'))
    assert len(sessions) == 2  # /new started a second session


def test_chat_resumes_a_session_from_the_flag_and_the_command(workspace):
    model = _model('one', 'two', 'three')
    first = _chat(['remember this', '/exit'], model)
    key = first.output.split('Session: ')[1].split(';')[0]
    result = _chat([f'/resume {key}', 'next', '/exit'], model, '--resume', key)
    assert result.exit_code == 0, result.output
    assert f'Session: {key}' in result.output
    sent = model.create_chat_completion.call_args.kwargs['messages']
    assert any(m.get('content') == 'remember this' for m in sent)


def test_chat_reports_errors_and_keeps_going(workspace):
    result = _chat(['/resume not-a-session', '/exit'], _model())
    assert result.exit_code == 0
    assert 'Session: ' not in result.output


def test_chat_stops_on_end_of_input(workspace):
    with patch('main.get_llm', return_value=_model()), patch('typer.prompt', side_effect=EOFError):
        result = runner.invoke(app, ['chat'])
    assert result.exit_code == 0


def test_chat_model_command_swaps_the_runtime_model(workspace, monkeypatch):
    from model_backend import ModelAdapter

    class Counting(ModelAdapter):
        def count_tokens(self, text):
            return 7

    swapped = Counting({'backend': 'embedded'})
    calls = []
    monkeypatch.setattr(main, 'handle_model_command', lambda: calls.append('model'))
    models = iter([_model(), swapped])
    monkeypatch.setattr(main, 'get_llm', lambda: next(models))
    result = runner.invoke(app, ['chat'], input='/model\n/exit\n')
    assert result.exit_code == 0, result.output
    assert calls == ['model']


def test_chat_md_command(workspace, monkeypatch):
    seen = []
    monkeypatch.setattr(main, 'handle_md_command', lambda console, max_tokens: seen.append(max_tokens))
    result = _chat(['/md', '/exit'], _model(), '--max-tokens', '300')
    assert result.exit_code == 0
    assert seen == [300]


# ---------------------------------------------------------------------------
# /md
# ---------------------------------------------------------------------------

def test_md_command_writes_context_after_confirmation(workspace, monkeypatch):
    (workspace / 'README.md').write_text('# Demo\n')
    model = _model('# Demo project\n\nA demo.')
    monkeypatch.setattr(main, 'get_llm', lambda: model)
    monkeypatch.setattr(typer, 'confirm', lambda _: True)
    main.handle_md_command(Console(file=io.StringIO()), 100)
    assert (workspace / 'CONTEXT.md').read_text() == '# Demo project\n\nA demo.'
    assert model.create_chat_completion.call_args.kwargs['max_tokens'] == 2048
    assert '# Demo' in model.create_chat_completion.call_args.kwargs['messages'][1]['content']


def test_md_command_can_be_cancelled(workspace, monkeypatch, capsys):
    monkeypatch.setattr(main, 'get_llm', lambda: _model('# Doc'))
    monkeypatch.setattr(typer, 'confirm', lambda _: False)
    main.handle_md_command(Console(file=io.StringIO()), None)
    assert not (workspace / 'CONTEXT.md').exists()
    assert 'Write cancelled.' in capsys.readouterr().out


def test_md_command_reports_empty_output_and_failed_runs(workspace, monkeypatch, capsys):
    from agent import RunResult
    out = io.StringIO()
    monkeypatch.setattr(main, 'get_llm', lambda: _model())
    monkeypatch.setattr(main, 'run_agent', lambda *a, **k: RunResult(status='blocked', text='', reason='model_error'))
    main.handle_md_command(Console(file=out), None)
    assert 'Failed to generate CONTEXT.md content.' in capsys.readouterr().out
    assert '[blocked]' in out.getvalue()


def test_md_command_reports_write_errors(workspace, monkeypatch, capsys):
    (workspace / 'CONTEXT.md').mkdir()
    monkeypatch.setattr(main, 'get_llm', lambda: _model('# Doc'))
    monkeypatch.setattr(typer, 'confirm', lambda _: True)
    main.handle_md_command(Console(file=io.StringIO()), None)
    assert 'Error writing CONTEXT.md' in capsys.readouterr().out


def test_gather_project_context_skips_paths_outside_the_workspace_and_unreadable_files(workspace, tmp_path, monkeypatch):
    outside = tmp_path / 'outside.md'
    outside.write_text('secret')
    (workspace / 'README.md').symlink_to(outside)
    (workspace / 'config.py').write_text('ok = True\n')
    original = main.WorkspaceTools.path

    def path(self, name):
        if name == 'config.py':
            broken = MagicMock()
            broken.exists.return_value = broken.is_file.return_value = True
            broken.read_text.side_effect = OSError('unreadable')
            return broken
        return original(self, name)

    monkeypatch.setattr(main.WorkspaceTools, 'path', path)
    context = main._gather_project_context()
    assert 'secret' not in context and 'Contents of config.py' not in context


# ---------------------------------------------------------------------------
# handle_model_command branches not covered elsewhere
# ---------------------------------------------------------------------------

def test_model_command_api_flow_can_be_cancelled_or_fail(workspace, monkeypatch, capsys):
    import providers
    monkeypatch.setattr('builtins.input', lambda _: 'api')
    monkeypatch.setattr(providers, 'select_provider_interactively', lambda: None)
    main.handle_model_command()
    assert 'Keeping current model.' in capsys.readouterr().out

    def fail():
        raise ValueError('bad key')
    monkeypatch.setattr(providers, 'select_provider_interactively', fail)
    main.handle_model_command()
    assert 'Error: bad key' in capsys.readouterr().out


def test_model_command_rejects_an_out_of_range_selection(workspace, tmp_path, monkeypatch, capsys):
    gguf = tmp_path / 'alt.gguf'
    gguf.write_bytes(b'GGUF')
    monkeypatch.setattr('main.glob.glob', lambda _: [str(gguf)])
    monkeypatch.setattr('builtins.input', lambda _: '5')
    main.handle_model_command()
    out = capsys.readouterr().out
    assert '1. alt.gguf' in out and 'Invalid selection.' in out


def test_model_command_reports_errors_while_switching(workspace, tmp_path, monkeypatch, capsys):
    gguf = tmp_path / 'alt.gguf'
    gguf.write_bytes(b'GGUF')
    monkeypatch.setattr('builtins.input', lambda _: str(gguf))
    monkeypatch.setattr(config, 'set_model_path', MagicMock(side_effect=OSError('read-only config')))
    main.handle_model_command()
    assert 'Error switching model: read-only config' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# models command branches
# ---------------------------------------------------------------------------

def test_models_rejects_unknown_providers_and_stray_api_key_flag(workspace):
    result = runner.invoke(app, ['models', '--provider', 'nobody'])
    assert result.exit_code != 0 and 'Unknown provider' in _plain(result.output)
    result = runner.invoke(app, ['models', '--api-key'])
    assert result.exit_code != 0 and '--api-key requires --provider' in _plain(result.output)


def test_models_requires_an_api_key_for_a_new_provider(workspace, monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    result = runner.invoke(app, ['models', '--provider', 'openai'], input='\n')
    assert result.exit_code != 0 and 'An API key is required' in _plain(result.output)


def test_models_rejects_invalid_profiles_without_saving(workspace):
    before = config.get_model_config()
    result = runner.invoke(app, ['models', '--speculative-mode', 'sometimes'])
    assert result.exit_code != 0 and 'speculative_mode' in _plain(result.output)
    assert config.get_model_config() == before


def test_models_prices_are_merged_and_shown(workspace, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'k')
    assert runner.invoke(app, ['models', '--provider', 'openai', '--price-input', '1.25']).exit_code == 0
    result = runner.invoke(app, ['models', '--price-output', '10', '--price-cached', '0.125'])
    assert result.exit_code == 0, result.output
    assert config.get_model_config()['price'] == {'input': 1.25, 'output': 10.0, 'cached_input': 0.125}
    assert 'Price per million tokens: $1.25 input, $10 output, $0.125 cached' in result.output
    assert 'Provider: openai (API key: set)' in result.output
    assert 'Server: https://api.openai.com/v1' in result.output


def test_models_switching_provider_drops_the_old_price(workspace, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'k')
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'k')
    runner.invoke(app, ['models', '--provider', 'openai', '--price-input', '1', '--price-output', '2'])
    assert runner.invoke(app, ['models', '--provider', 'anthropic']).exit_code == 0
    assert config.get_model_config().get('price') is None


def test_models_a_custom_server_clears_the_provider(workspace, monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'k')
    runner.invoke(app, ['models', '--provider', 'openai'])
    result = runner.invoke(app, ['models', '--base-url', 'http://127.0.0.1:9000/v1'])
    assert result.exit_code == 0, result.output
    assert config.get_model_config().get('provider') is None


def test_models_set_reports_a_failed_config_write(workspace, sample_gguf_file, monkeypatch):
    monkeypatch.setattr(config, 'set_model_path', lambda path: False)
    result = runner.invoke(app, ['models', '--set', sample_gguf_file])
    assert result.exit_code == 1 and 'Failed to update model configuration' in result.output


# ---------------------------------------------------------------------------
# clean and undo
# ---------------------------------------------------------------------------

def test_clean_dry_run_and_override(workspace):
    result = runner.invoke(app, ['clean', '--dry-run'])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ['clean', '--older-than', '0'])
    assert result.exit_code == 0, result.output


def test_clean_reports_errors(workspace):
    (workspace / 'elsewhere').mkdir()
    (workspace / '.local-coder').symlink_to(workspace / 'elsewhere')
    result = runner.invoke(app, ['clean'])
    assert result.exit_code == 1 and 'Cannot clean' in result.output


def test_undo_command_reverts_the_last_patch_and_turn(workspace):
    from workspace_tools import WorkspaceTools
    tools = WorkspaceTools(mode='workspace-edit')
    tools.turn_id = 'one'
    tools.call_tool('write', {'path': 'a.txt', 'content': 'v1\n'})
    tools.call_tool('write', {'path': 'b.txt', 'content': 'v1\n'})
    result = runner.invoke(app, ['undo'])
    assert result.exit_code == 0, result.output
    assert not (workspace / 'b.txt').exists() and (workspace / 'a.txt').exists()
    result = runner.invoke(app, ['undo', '--turn'])
    assert result.exit_code == 0, result.output
    assert not (workspace / 'a.txt').exists()


def test_undo_command_refuses_changed_files(workspace):
    from workspace_tools import WorkspaceTools
    tools = WorkspaceTools(mode='workspace-edit')
    tools.call_tool('write', {'path': 'a.txt', 'content': 'v1\n'})
    (workspace / 'a.txt').write_text('mine\n')
    result = runner.invoke(app, ['undo'])
    assert result.exit_code == 1 and 'Cannot undo' in result.output


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

def _bench_model(finish_reason='stop'):
    model = MagicMock()
    model.create_chat_completion.return_value = {
        'choices': [{'message': {'content': 'x'}, 'finish_reason': finish_reason}],
        'performance': {'elapsed_seconds': 1.0}}
    return model


def test_benchmark_writes_a_report(workspace):
    model = _bench_model()
    with patch('main.create_model', return_value=model):
        result = runner.invoke(app, ['benchmark', '--output', 'report.json', '--repeats', '2', '--warmups', '0'])
    assert result.exit_code == 0, result.output
    report = json.loads((workspace / 'report.json').read_text())
    assert report['profile'] == 'default' and report['median']['elapsed_seconds'] == 1.0
    assert 'model_path' in report['settings']
    model.close.assert_called_once()


def test_benchmark_compares_profiles_and_flags_truncation(workspace):
    config.save_profile('other')
    with patch('main.create_model', side_effect=[_bench_model('length'), _bench_model()]):
        result = runner.invoke(app, ['benchmark', '--output', 'cmp.json', '--compare', 'default', '--compare', 'other',
                                     '--repeats', '1', '--warmups', '0'])
    assert result.exit_code == 0, result.output
    report = json.loads((workspace / 'cmp.json').read_text())
    assert [r['profile'] for r in report['comparisons']] == ['default', 'other']
    assert 'Some responses were truncated' in result.output


def test_benchmark_uses_the_persistent_daemon(workspace):
    with patch('inference_daemon.PersistentModel', return_value=_bench_model()) as persistent:
        result = runner.invoke(app, ['benchmark', '--output', 'p.json', '--persistent', '--repeats', '1', '--warmups', '0'])
    assert result.exit_code == 0, result.output
    assert persistent.called


def test_benchmark_rejects_compare_with_persistent(workspace):
    result = runner.invoke(app, ['benchmark', '--output', 'x.json', '--compare', 'default', '--persistent'])
    assert result.exit_code == 1 and 'Persistent inference holds one profile' in result.output


# ---------------------------------------------------------------------------
# profiles
# ---------------------------------------------------------------------------

def test_profiles_list_and_errors(workspace):
    config.save_profile('fast')
    result = runner.invoke(app, ['profiles'])
    assert result.exit_code == 0
    listing = json.loads(result.output)
    assert listing['profiles'] == ['fast'] and listing['active'] == 'default'
    result = runner.invoke(app, ['profiles', 'delete', 'fast'])
    assert result.exit_code != 0 and 'Use list, save NAME' in _plain(result.output)
    result = runner.invoke(app, ['profiles', 'use', 'missing'])
    assert result.exit_code != 0 and 'Unknown model profile' in _plain(result.output)


# ---------------------------------------------------------------------------
# inference-server
# ---------------------------------------------------------------------------

def test_inference_server_stop(workspace):
    with patch('inference_daemon.PersistentModel') as persistent:
        result = runner.invoke(app, ['inference-server', '--stop'])
    assert result.exit_code == 0 and 'Daemon stopping.' in result.output
    persistent.return_value.stop.assert_called_once()


def test_inference_server_stop_without_a_daemon(workspace):
    with patch('inference_daemon.PersistentModel') as persistent:
        persistent.return_value.stop.side_effect = ConnectionRefusedError('nobody home')
        result = runner.invoke(app, ['inference-server', '--stop'])
    assert result.exit_code != 0 and 'Daemon unavailable' in _plain(result.output)


def test_inference_server_rejects_server_backends(workspace):
    runner.invoke(app, ['models', '--backend', 'openai', '--base-url', 'http://127.0.0.1:8080/v1'])
    result = runner.invoke(app, ['inference-server'])
    assert result.exit_code != 0 and 'for embedded models' in _plain(result.output)


def test_inference_server_serves_until_interrupted(workspace):
    server = MagicMock()
    server.__enter__.return_value = server
    server.serve_forever.side_effect = KeyboardInterrupt
    with patch('main.create_model', return_value=MagicMock()), \
            patch('inference_daemon.InferenceDaemon', return_value=server) as daemon:
        result = runner.invoke(app, ['inference-server'])
    assert result.exit_code == 0, result.output
    assert 'Inference daemon running' in result.output
    assert daemon.call_args.args[0] == config.CONFIG_DIR / 'inference.sock'
