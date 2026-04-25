from unittest.mock import MagicMock, patch

import pytest

from memory.vector_store import VectorMemory

# A fixed 4-dimensional unit vector used everywhere embeddings are needed.
FAKE_EMBED = [1.0, 0.0, 0.0, 0.0]


def _embed_response(vec=None):
    r = MagicMock()
    r.embeddings = [vec or FAKE_EMBED]
    return r


def _chat_response(text: str):
    r = MagicMock()
    r.message.content = text
    return r


@pytest.fixture
def vm(tmp_path):
    """VectorMemory backed by a temp ChromaDB with mocked Ollama."""
    with patch("memory.vector_store.ollama") as mock_ollama:
        mock_ollama.embed.return_value = _embed_response()
        mock_ollama.chat.return_value = _chat_response("NONE")
        store = VectorMemory("nomic-embed-text", str(tmp_path))
        yield store, mock_ollama


# ------------------------------------------------------------------
# count() / store()
# ------------------------------------------------------------------

def test_count_starts_at_zero(vm):
    store, _ = vm
    assert store.count() == 0


def test_store_increases_count(vm):
    store, _ = vm
    store.store("Alice likes Python.")
    assert store.count() == 1


def test_store_multiple(vm):
    store, _ = vm
    store.store("fact one")
    store.store("fact two")
    assert store.count() == 2


# ------------------------------------------------------------------
# retrieve()
# ------------------------------------------------------------------

def test_retrieve_empty_collection_returns_empty_list(vm):
    store, _ = vm
    assert store.retrieve("anything") == []


def test_retrieve_returns_stored_document(vm):
    store, _ = vm
    store.store("Alice likes Python.")
    results = store.retrieve("Python preferences")
    assert "Alice likes Python." in results


def test_retrieve_respects_k_limit(vm):
    store, _ = vm
    for i in range(5):
        store.store(f"fact {i}")
    results = store.retrieve("query", k=2)
    assert len(results) == 2


def test_retrieve_k_larger_than_collection(vm):
    store, _ = vm
    store.store("only one fact")
    results = store.retrieve("query", k=10)
    assert len(results) == 1


# ------------------------------------------------------------------
# all_memories()
# ------------------------------------------------------------------

def test_all_memories_empty(vm):
    store, _ = vm
    assert store.all_memories() == []


def test_all_memories_returns_every_document(vm):
    store, _ = vm
    store.store("fact A")
    store.store("fact B")
    memories = store.all_memories()
    assert set(memories) == {"fact A", "fact B"}


# ------------------------------------------------------------------
# extract_and_store()
# ------------------------------------------------------------------

def test_extract_none_returns_empty_list(vm):
    store, mock_ollama = vm
    mock_ollama.chat.return_value = _chat_response("NONE")
    facts = store.extract_and_store("hi", "hello there", "llama3.2")
    assert facts == []


def test_extract_stores_returned_facts(vm):
    store, mock_ollama = vm
    mock_ollama.chat.return_value = _chat_response("User prefers Python.\nUser dislikes Java.")
    initial_count = store.count()
    facts = store.extract_and_store("I love Python", "Great choice!", "llama3.2")
    assert len(facts) == 2
    assert store.count() == initial_count + 2


def test_extract_strips_none_lines(vm):
    store, mock_ollama = vm
    mock_ollama.chat.return_value = _chat_response("User likes cats.\nNONE\n")
    facts = store.extract_and_store("I love cats", "Cute!", "llama3.2")
    assert facts == ["User likes cats."]


def test_extract_empty_response_returns_empty(vm):
    store, mock_ollama = vm
    mock_ollama.chat.return_value = _chat_response("")
    facts = store.extract_and_store("test", "test", "llama3.2")
    assert facts == []
