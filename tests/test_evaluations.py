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
