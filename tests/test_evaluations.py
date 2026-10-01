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
