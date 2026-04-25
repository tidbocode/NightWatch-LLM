# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prerequisites & setup

Ollama must be running locally (`ollama serve`) with both models pulled:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
pip install -r requirements.txt
```

## Running

```bash
python main.py
```

`main.py` calls `check_ollama()` on startup, which verifies Ollama is reachable and both models are present before instantiating `NightWatch`.

## Architecture

`NightWatch` in `chatbot.py` is the single orchestrator. It wires together four subsystems, all configured via `config.py`:

### Per-turn flow in `NightWatch.chat()`

`chat()` is a **generator** — it yields tokens as they stream from Ollama, then runs post-processing (state update + memory extraction) after the last token is yielded. The caller must fully consume the generator for memory extraction to run.

Order of operations each turn:
1. **`_maybe_summarize()`** — checks `TokenBudget.is_over_threshold()` against `SUMMARIZE_THRESHOLD` (default 0.75). If over threshold and more than `RECENT_MESSAGES_KEEP` (6) messages exist, pops the oldest `n` messages from `ConversationState` and sends them to `Summarizer.summarize()`, which calls Ollama to produce a rolling third-person summary.
2. **`VectorMemory.retrieve()`** — embeds the user query via `nomic-embed-text` and queries ChromaDB for the top-`MEMORY_TOP_K` (3) semantically similar stored facts.
3. **`_build_messages()`** — assembles the Ollama messages list in this exact layer order: system prompt → rolling summary (if any) → retrieved memories (if any) → recent `state.messages` → new user message. The new user message is **not** in `state.messages` yet at this point.
4. **Stream** from `ollama.chat(..., stream=True)`, yielding each token.
5. **`VectorMemory.extract_and_store()`** — after streaming completes, makes a second non-streaming Ollama call asking the LLM to extract 1–3 memorable facts from the exchange. Each fact is embedded and stored in ChromaDB.

### Subsystems

- **`memory/conversation.py`** — `ConversationState` dataclass. `pop_oldest(n)` removes and returns the n oldest messages for the summarizer.
- **`memory/summarizer.py`** — `Summarizer.summarize()` accepts `messages` + optional `existing_summary`. If a prior summary exists, the prompt asks Ollama to incorporate new messages into it (incremental update), not rewrite from scratch.
- **`memory/vector_store.py`** — `VectorMemory` wraps ChromaDB `PersistentClient`. Embeddings are always provided explicitly (not via a ChromaDB embedding function), so the collection is created with cosine similarity and no built-in EF. The ChromaDB database persists to `./memory_db/` across sessions. `/clear` in the CLI resets `ConversationState` but does **not** wipe ChromaDB — vector memories accumulate indefinitely.
- **`utils/token_budget.py`** — `TokenBudget` uses a 1 token ≈ 4 chars heuristic. No external tokenizer library.

### Key config knobs (`config.py`)

| Variable | Default | Effect |
|---|---|---|
| `CHAT_MODEL` | `llama3.2` | Model for chat and summarization/extraction |
| `EMBED_MODEL` | `nomic-embed-text` | Model for embeddings |
| `CONTEXT_TOKEN_BUDGET` | `3000` | Approximate token ceiling for context |
| `SUMMARIZE_THRESHOLD` | `0.75` | Fraction of budget that triggers summarization |
| `RECENT_MESSAGES_KEEP` | `6` | Messages kept verbatim; older ones are summarized |
| `MEMORY_TOP_K` | `3` | Memories injected per turn |
| `MEMORY_DB_PATH` | `./memory_db` | ChromaDB persistence path |
