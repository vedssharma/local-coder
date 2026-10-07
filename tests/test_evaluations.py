import pytest
from evaluations.run import CASES, ScriptedModel, evaluate


@pytest.mark.parametrize('case', CASES)
def test_scripted_harness_evaluations(case):
    result = evaluate(case, scripted=True)
    assert result['passed'], result


def test_evaluation_rejects_unverified_claim_of_success():
    result = evaluate('bug_fix', scripted=True, model=ScriptedModel(['Everything is fixed and tests passed.']))
    assert not result['passed']
    assert not result['checks']['fixture_tests_passed']
    assert not result['checks']['agent_observed_passing_command']


from evaluations.loop import CASES as LOOP_CASES, evaluate as evaluate_loop


@pytest.mark.parametrize('case', LOOP_CASES)
def test_fault_injection_loop_evaluations(case):
    result=evaluate_loop(case)
    assert result['passed'], result


def test_evaluation_baseline_preserves_metric_deltas(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    runner=Path(__file__).resolve().parents[1]/'evaluations/run.py'
    baseline=tmp_path/'before.json'
    after=tmp_path/'after.json'
    command=[sys.executable,str(runner),'--scripted','--case','transient_retry']
    subprocess.run(command+['--output',str(baseline)],check=True,capture_output=True)
    subprocess.run(command+['--baseline',str(baseline),'--output',str(after)],check=True,capture_output=True)
    report=json.loads(after.read_text())
    comparison=report['comparison'][0]
    assert comparison['passed_before'] and comparison['passed_now']
    assert comparison['deltas']['model_calls']==0 and comparison['deltas']['retries']==0


# ---------------------------------------------------------------------------
# The evaluation runner's command line, run in-process
# ---------------------------------------------------------------------------

import json

import evaluations.loop
import evaluations.run as run


def _main(monkeypatch, *args):
    monkeypatch.setattr('sys.argv', ['run.py', *args])
    return run.main()


def test_scripted_model_tolerates_non_json_tool_output():
    model = ScriptedModel(['ok'])
    response = model.create_chat_completion(messages=[{'role': 'tool', 'content': 'plain text'}])
    assert response['choices'][0]['message']['content'] == 'ok'
    assert model.create_chat_completion(messages=[])['choices'][0]['message']['content'] == 'Done.'


def test_runner_writes_a_report_and_compares_with_a_baseline(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(evaluations.loop, 'evaluate', lambda case: {'case': case, 'passed': True, 'seconds': 1.0})
    first = tmp_path / 'out' / 'first.json'
    assert _main(monkeypatch, '--scripted', '--case', 'transient_retry', '--output', str(first)) == 0
    assert 'transient_retry: PASS' in capsys.readouterr().out
    report = json.loads(first.read_text())
    assert report['mode'] == 'scripted_mechanics' and report['model'] is None
    monkeypatch.setattr(evaluations.loop, 'evaluate', lambda case: {'case': case, 'passed': False, 'seconds': 1.5})
    second = tmp_path / 'second.json'
    assert _main(monkeypatch, '--scripted', '--case', 'transient_retry', '--baseline', str(first),
                 '--output', str(second)) == 1
    comparison = json.loads(second.read_text())['comparison']
    assert comparison == [{'case': 'transient_retry', 'passed_before': True, 'passed_now': False,
                           'deltas': {'seconds': 0.5}}]


def test_runner_records_evaluation_errors_as_failures(monkeypatch, tmp_path):
    def broken(case):
        raise RuntimeError('fixture exploded')
    monkeypatch.setattr(evaluations.loop, 'evaluate', broken)
    output = tmp_path / 'report.json'
    assert _main(monkeypatch, '--scripted', '--suite', 'loop', '--output', str(output)) == 1
    results = json.loads(output.read_text())['results']
    assert len(results) == len(LOOP_CASES)
    assert results[0] == {'case': results[0]['case'], 'passed': False, 'outcome': 'evaluation_error',
                          'reason': 'fixture exploded'}


@pytest.mark.parametrize('args, message', [
    (['--suite', 'loop'], 'require --scripted'),
    (['--case', 'bug_fix'], 'require --allow-execution'),
])
def test_runner_refuses_unsafe_or_meaningless_real_model_runs(monkeypatch, tmp_path, capsys, args, message):
    with pytest.raises(SystemExit):
        _main(monkeypatch, *args, '--output', str(tmp_path / 'x.json'))
    assert message in capsys.readouterr().err


def test_runner_evaluates_a_real_profile(monkeypatch, tmp_path, config_dir):
    seen = []
    monkeypatch.setattr(run, 'evaluate', lambda case, scripted, profile: seen.append((case, scripted, profile))
                        or {'case': case, 'passed': True})
    output = tmp_path / 'model.json'
    assert _main(monkeypatch, '--case', 'navigation', '--output', str(output)) == 0
    report = json.loads(output.read_text())
    assert report['mode'] == 'model_evaluation'
    assert report['model']['backend'] == 'embedded' and report['model']['n_ctx'] == seen[0][2]['n_ctx']
    assert seen[0][:2] == ('navigation', False)
    baseline = tmp_path / 'scripted.json'
    baseline.write_text(json.dumps({'mode': 'scripted_mechanics', 'results': []}))
    with pytest.raises(SystemExit):
        _main(monkeypatch, '--case', 'navigation', '--baseline', str(baseline), '--output', str(output))


def test_unknown_loop_case_and_invalid_benchmark_counts():
    from performance import benchmark
    with pytest.raises(ValueError, match='Unknown loop evaluation'):
        evaluate_loop('nonexistent')
    for counts in ((0, 1, 1), (1, -1, 1), (1, 1, 0)):
        with pytest.raises(ValueError, match='Invalid benchmark counts'):
            benchmark(None, 'prompt', *counts)
