from unittest.mock import MagicMock, patch

from memory.summarizer import Summarizer


def _mock_ollama_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.message.content = text
    return resp


MESSAGES = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Nice to meet you, Alice!"},
]


# ------------------------------------------------------------------
# First-time summarization (no existing summary)
# ------------------------------------------------------------------

def test_summarize_returns_llm_output():
    with patch("memory.summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = _mock_ollama_response("  Alice introduced herself.  ")
        result = Summarizer("llama3.2").summarize(MESSAGES)
    assert result == "Alice introduced herself."


def test_summarize_no_existing_prompt_contains_conversation():
    with patch("memory.summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = _mock_ollama_response("summary")
        Summarizer("llama3.2").summarize(MESSAGES)
    prompt = mock_chat.call_args[1]["messages"][0]["content"]
    assert "Alice" in prompt
    assert "Previous summary" not in prompt


def test_summarize_uses_configured_model():
    with patch("memory.summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = _mock_ollama_response("ok")
        Summarizer("mistral").summarize(MESSAGES)
    assert mock_chat.call_args[1]["model"] == "mistral"


# ------------------------------------------------------------------
# Incremental summarization (existing summary provided)
# ------------------------------------------------------------------

def test_summarize_with_existing_includes_prior_summary_in_prompt():
    with patch("memory.summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = _mock_ollama_response("updated summary")
        Summarizer("llama3.2").summarize(MESSAGES, existing_summary="Alice said hello.")
    prompt = mock_chat.call_args[1]["messages"][0]["content"]
    assert "Alice said hello." in prompt
    assert "Previous summary" in prompt


def test_summarize_with_existing_also_includes_new_messages():
    with patch("memory.summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = _mock_ollama_response("updated")
        Summarizer("llama3.2").summarize(MESSAGES, existing_summary="Prior.")
    prompt = mock_chat.call_args[1]["messages"][0]["content"]
    assert "Alice" in prompt


def test_summarize_strips_whitespace_from_response():
    with patch("memory.summarizer.ollama.chat") as mock_chat:
        mock_chat.return_value = _mock_ollama_response("\n  trimmed  \n")
        result = Summarizer("llama3.2").summarize(MESSAGES)
    assert result == "trimmed"
