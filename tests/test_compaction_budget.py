"""The compaction summary scales with the context window."""

from session import ContextManager


def test_summary_budget_is_a_bounded_share_of_the_window():
    assert ContextManager(window=2048).summary_chars == 800
    assert ContextManager(window=32768).summary_chars == 4915
    assert ContextManager(window=1_000_000).summary_chars == 24000


def _compact(window):
    manager = ContextManager(window=window)
    messages = [{'role': 'system', 'content': 'sys'}]
    for turn in range(40):
        messages += [{'role': 'user', 'content': f'request {turn} ' + 'r' * 200},
                     {'role': 'assistant', 'content': f'result {turn} ' + 'a' * 300}]
    manager.fit(messages, [], 64)
    summary = next(m['content'] for m in messages if m.get('name') == 'working_memory')
    return manager, messages, summary


def test_larger_windows_keep_more_of_the_compacted_work():
    small, _, small_summary = _compact(6000)
    large, messages, large_summary = _compact(20000)
    assert large_summary.count('Request:') > small_summary.count('Request:') > 0
    assert len(small_summary) < small.summary_chars + 500  # Plus the header and the 400-char task line.
    assert large.size(messages, []) <= 20000 - 64
    assert 'Original task: request 0' in large_summary
    # Notes that no longer fit are dropped, so memory doesn't grow without bound.
    assert len('\n'.join(large.memory)) <= large.summary_chars
