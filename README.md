# NightWatch-LLM

A local LLM chatbot that teaches four core memory fundamentals — all running on your machine via Ollama.

| Concept | Where it lives |
|---|---|
| **Conversation state** | `memory/conversation.py` — rolling message window |
| **Summarization loop** | `memory/summarizer.py` — LLM compresses old turns |
| **Vector memory** | `memory/vector_store.py` — ChromaDB + Ollama embeddings |
| **Token budgeting** | `utils/token_budget.py` — usage tracking & threshold triggers |

## Prerequisites

- [Ollama](https://ollama.com) installed and running (`ollama serve`)
- Python 3.11+

## Setup

```bash
# 1. Pull the required models
ollama pull llama3.2
ollama pull nomic-embed-text

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run
python main.py
```

## How it works

```
User message
    │
    ├─► Token budget check ──over threshold?──► Summarize oldest N turns
    │
    ├─► Vector retrieve: top-k memories relevant to this query
    │
    ├─► Build prompt:
    │     [system prompt]
    │     [rolling summary]       ← compressed history
    │     [retrieved memories]    ← long-term facts
    │     [recent messages]       ← last 6 turns verbatim
    │     [user message]
    │
    ├─► Stream response from Ollama
    │
    └─► Extract & store memorable facts → ChromaDB
```

## Chat commands

| Command | Description |
|---|---|
| `/stats` | Token usage, context bar, memory counts |
| `/memories` | List all stored vector memories |
| `/clear` | Reset conversation (vector memories persist) |
| `/help` | Show command list |
| `/quit` | Exit |

## Configuration

Edit `config.py` to change models, token budget, or summarization threshold.
