"""
Tests for Mockingbird — the top-level orchestrator.

All Ollama and ChromaDB calls are mocked so these tests run offline.
"""
from unittest.mock import MagicMock, patch

import pytest

from chatbot import Mockingbird
from utils.token_budget import TokenBudget


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

def _stream_chunk(text: str) -> MagicMock:
    chunk = MagicMock()
    chunk.message.content = text
    return chunk


@pytest.fixture
def bot():
    """
    A Mockingbird instance with Summarizer, VectorMemory, and ollama patched out.
    Yields (bot, mock_vector_memory, mock_summarizer, mock_ollama).
    """
    with (
        patch("chatbot.VectorMemory") as mock_vm_cls,
        patch("chatbot.Summarizer") as mock_sum_cls,
        patch("chatbot.ollama") as mock_ollama,
    ):
        mock_vm = mock_vm_cls.return_value
        mock_vm.retrieve.return_value = []
        mock_vm.count.return_value = 0
        mock_vm.extract_and_store.return_value = []

        mock_sum = mock_sum_cls.return_value

        mock_ollama.chat.return_value = iter([_stream_chunk("Hello"), _stream_chunk(" world")])

        yield Mockingbird(), mock_vm, mock_sum, mock_ollama


# ------------------------------------------------------------------
# _build_messages() — context layer ordering
# ------------------------------------------------------------------

def test_build_messages_starts_with_system_prompt(bot):
    b, *_ = bot
    msgs = b._build_messages("hi")
    assert msgs[0]["role"] == "system"


def test_build_messages_ends_with_user_message(bot):
    b, *_ = bot
    msgs = b._build_messages("my question")
    assert msgs[-1] == {"role": "user", "content": "my question"}


def test_build_messages_no_summary_no_memories(bot):
    b, *_ = bot
    msgs = b._build_messages("hi")
    # system prompt + user message only
    assert len(msgs) == 2


def test_build_messages_injects_summary(bot):
    b, *_ = bot
    b.state.set_summary("Alice prefers Python.")
    msgs = b._build_messages("hi")
    summary_msg = next(m for m in msgs if "summary" in m.get("content", "").lower())
    assert "Alice prefers Python." in summary_msg["content"]


def test_build_messages_injects_vector_memories(bot):
    b, *_ = bot
    b._last_memories = ["User likes Python.", "User dislikes Java."]
    msgs = b._build_messages("hi")
    mem_msg = next(m for m in msgs if "memories" in m.get("content", "").lower())
    assert "User likes Python." in mem_msg["content"]
    assert "User dislikes Java." in mem_msg["content"]


def test_build_messages_layer_order(bot):
    """system → summary → memories → history → new user message"""
    b, *_ = bot
    b.state.set_summary("A summary.")
    b._last_memories = ["a memory"]
    b.state.add("user", "older message")
    b.state.add("assistant", "older reply")

    msgs = b._build_messages("new message")
    roles = [m["role"] for m in msgs]

    # All system-level injections come before the history
    last_system = max(i for i, m in enumerate(msgs) if m["role"] == "system")
    first_user_history = next(
        i for i, m in enumerate(msgs) if m["role"] == "user" and m["content"] == "older message"
    )
    assert last_system < first_user_history

    # New user message is last
    assert msgs[-1]["content"] == "new message"


def test_build_messages_history_included(bot):
    b, *_ = bot
    b.state.add("user", "previous question")
    b.state.add("assistant", "previous answer")
    msgs = b._build_messages("follow up")
    contents = [m["content"] for m in msgs]
    assert "previous question" in contents
    assert "previous answer" in contents


# ------------------------------------------------------------------
# chat() — generator behaviour and state updates
# ------------------------------------------------------------------

def test_chat_yields_tokens(bot):
    b, _, _, mock_ollama = bot
    mock_ollama.chat.return_value = iter([
        _stream_chunk("Hello"),
        _stream_chunk(" world"),
    ])
    tokens = list(b.chat("hi"))
    assert tokens == ["Hello", " world"]


def test_chat_adds_user_and_assistant_to_state(bot):
    b, _, _, mock_ollama = bot
    mock_ollama.chat.return_value = iter([_stream_chunk("response")])
    list(b.chat("my input"))  # consume generator fully
    assert b.state.messages[-2] == {"role": "user", "content": "my input"}
    assert b.state.messages[-1] == {"role": "assistant", "content": "response"}


def test_chat_increments_turn_count(bot):
    b, _, _, mock_ollama = bot
    mock_ollama.chat.return_value = iter([_stream_chunk("ok")])
    list(b.chat("turn one"))
    mock_ollama.chat.return_value = iter([_stream_chunk("ok")])
    list(b.chat("turn two"))
    assert b.state.turn_count == 2


def test_chat_calls_retrieve_with_user_message(bot):
    b, mock_vm, _, mock_ollama = bot
    mock_ollama.chat.return_value = iter([_stream_chunk("ok")])
    list(b.chat("what is Python?"))
    mock_vm.retrieve.assert_called_once_with("what is Python?", k=3)


def test_chat_calls_extract_and_store_after_streaming(bot):
    b, mock_vm, _, mock_ollama = bot
    mock_ollama.chat.return_value = iter([_stream_chunk("It's a language")])
    list(b.chat("what is Python?"))
    mock_vm.extract_and_store.assert_called_once_with(
        "what is Python?", "It's a language", "llama3.2"
    )


# ------------------------------------------------------------------
# _maybe_summarize()
# ------------------------------------------------------------------

def test_maybe_summarize_skipped_with_few_messages(bot):
    b, _, mock_sum, _ = bot
    # Fewer messages than RECENT_MESSAGES_KEEP (6) — never summarizes
    for i in range(4):
        b.state.add("user", f"msg {i}")
        b.state.add("assistant", f"reply {i}")
    b._maybe_summarize()
    mock_sum.summarize.assert_not_called()


def test_maybe_summarize_skipped_when_under_budget(bot):
    b, _, mock_sum, _ = bot
    # Add 8 messages but keep content tiny so token usage stays low
    for i in range(4):
        b.state.add("user", "hi")
        b.state.add("assistant", "ok")
    b._maybe_summarize()
    mock_sum.summarize.assert_not_called()


def test_maybe_summarize_triggered_when_over_threshold(bot):
    b, _, mock_sum, _ = bot
    mock_sum.summarize.return_value = "A rolling summary."

    # Shrink budget so a few long messages trip the threshold
    b.budget = TokenBudget(budget=100)

    # 8 messages with long content to exceed 75% of 100 tokens
    for i in range(4):
        b.state.add("user", "a" * 80)
        b.state.add("assistant", "b" * 80)

    b._maybe_summarize()
    mock_sum.summarize.assert_called_once()


def test_maybe_summarize_stores_new_summary(bot):
    b, _, mock_sum, _ = bot
    mock_sum.summarize.return_value = "Summary text."
    b.budget = TokenBudget(budget=100)
    for i in range(4):
        b.state.add("user", "a" * 80)
        b.state.add("assistant", "b" * 80)

    b._maybe_summarize()
    assert b.state.summary == "Summary text."


def test_maybe_summarize_keeps_recent_messages(bot):
    b, _, mock_sum, _ = bot
    mock_sum.summarize.return_value = "Summary."
    b.budget = TokenBudget(budget=100)
    for i in range(4):
        b.state.add("user", f"msg {i} " + "a" * 80)
        b.state.add("assistant", "b" * 80)

    b._maybe_summarize()
    # RECENT_MESSAGES_KEEP = 6, so 6 messages should remain in state
    assert len(b.state.messages) == 6


# ------------------------------------------------------------------
# stats()
# ------------------------------------------------------------------

def test_stats_returns_expected_keys(bot):
    b, mock_vm, _, _ = bot
    mock_vm.count.return_value = 5
    s = b.stats()
    expected = {
        "turns", "messages_in_context", "tokens_used", "token_budget",
        "pct_used", "has_summary", "vector_memories",
        "last_memories_retrieved", "last_facts_stored",
    }
    assert expected.issubset(s.keys())


def test_stats_reflects_conversation_state(bot):
    b, mock_vm, _, mock_ollama = bot
    mock_ollama.chat.return_value = iter([_stream_chunk("answer")])
    mock_vm.count.return_value = 0
    list(b.chat("question"))
    s = b.stats()
    assert s["turns"] == 1
    assert s["messages_in_context"] == 2  # user + assistant


# ------------------------------------------------------------------
# clear()
# ------------------------------------------------------------------

def test_clear_resets_conversation_state(bot):
    b, _, _, mock_ollama = bot
    mock_ollama.chat.return_value = iter([_stream_chunk("ok")])
    list(b.chat("hello"))
    b.clear()
    assert b.state.messages == []
    assert b.state.turn_count == 0
    assert b.state.summary is None


def test_clear_resets_last_memories_and_facts(bot):
    b, *_ = bot
    b._last_memories = ["some memory"]
    b._last_facts = ["some fact"]
    b.clear()
    assert b._last_memories == []
    assert b._last_facts == []
