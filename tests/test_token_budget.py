import pytest

from utils.token_budget import TokenBudget


@pytest.fixture
def budget():
    return TokenBudget(budget=1000)


# ------------------------------------------------------------------
# estimate()
# ------------------------------------------------------------------

def test_estimate_returns_at_least_one():
    assert TokenBudget.estimate("") == 1


def test_estimate_four_chars_is_one_token():
    assert TokenBudget.estimate("abcd") == 1


def test_estimate_forty_chars():
    assert TokenBudget.estimate("a" * 40) == 10


# ------------------------------------------------------------------
# messages_tokens()
# ------------------------------------------------------------------

def test_messages_tokens_empty(budget):
    assert budget.messages_tokens([]) == 0


def test_messages_tokens_includes_overhead(budget):
    # 8-char content → 2 tokens, plus 4 overhead = 6 per message
    msg = {"role": "user", "content": "12345678"}
    result = budget.messages_tokens([msg])
    assert result == 6


def test_messages_tokens_two_messages(budget):
    msgs = [
        {"role": "user", "content": "12345678"},       # 2 + 4 = 6
        {"role": "assistant", "content": "12345678"},  # 2 + 4 = 6
    ]
    assert budget.messages_tokens(msgs) == 12


# ------------------------------------------------------------------
# used()
# ------------------------------------------------------------------

def test_used_without_summary(budget):
    msgs = [{"role": "user", "content": "a" * 40}]  # 10 + 4 = 14
    assert budget.used(msgs) == 14


def test_used_adds_summary_tokens(budget):
    msgs = [{"role": "user", "content": "a" * 40}]  # 14
    summary = "b" * 40  # 10 more
    assert budget.used(msgs, summary) == 24


def test_used_none_summary_ignored(budget):
    msgs = [{"role": "user", "content": "a" * 40}]
    assert budget.used(msgs, None) == budget.used(msgs)


# ------------------------------------------------------------------
# fraction_used()
# ------------------------------------------------------------------

def test_fraction_used_half(budget):
    # Need 500 tokens of content → 500 * 4 = 2000 chars, plus 4 overhead each
    # Simpler: build a budget where we can predict the fraction easily
    b = TokenBudget(budget=100)
    # 1 message, 40 chars content → 10 tokens + 4 overhead = 14 tokens / 100 = 0.14
    msgs = [{"role": "user", "content": "a" * 40}]
    assert b.fraction_used(msgs) == pytest.approx(0.14)


def test_fraction_used_zero_messages(budget):
    assert budget.fraction_used([]) == 0.0


# ------------------------------------------------------------------
# is_over_threshold()
# ------------------------------------------------------------------

def test_not_over_threshold_with_no_messages(budget):
    assert budget.is_over_threshold([], threshold=0.75) is False


def test_over_threshold_when_above(budget):
    # 800 tokens used out of 1000 → 80% > 75%
    # 800 - 4 overhead = 796 content tokens → 796 * 4 = 3184 chars
    msgs = [{"role": "user", "content": "a" * 3184}]
    assert budget.is_over_threshold(msgs, threshold=0.75) is True


def test_not_over_threshold_when_below(budget):
    # 100 tokens out of 1000 → 10% < 75%
    msgs = [{"role": "user", "content": "a" * 384}]  # 96 + 4 = 100 tokens
    assert budget.is_over_threshold(msgs, threshold=0.75) is False


def test_at_exact_threshold_is_over(budget):
    # 750 tokens out of 1000 → exactly 75% → should trigger (>=)
    msgs = [{"role": "user", "content": "a" * (746 * 4)}]  # 746 + 4 = 750
    assert budget.is_over_threshold(msgs, threshold=0.75) is True
