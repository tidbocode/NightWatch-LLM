from typing import Generator

import ollama

from config import (
    CHAT_MODEL,
    CONTEXT_TOKEN_BUDGET,
    EMBED_MODEL,
    MEMORY_DB_PATH,
    MEMORY_TOP_K,
    RECENT_MESSAGES_KEEP,
    SUMMARIZE_THRESHOLD,
)
from memory.conversation import ConversationState
from memory.summarizer import Summarizer
from memory.vector_store import VectorMemory
from utils.token_budget import TokenBudget

_SYSTEM_PROMPT = (
    "You are NightWatch, a helpful and thoughtful AI assistant with persistent memory. "
    "You remember important details from past conversations and use them to give "
    "personalized, context-aware responses. Be concise and genuinely helpful."
)


class NightWatch:
    """
    Orchestrates four memory mechanisms:

    1. Conversation state  — rolling window of recent messages
    2. Summarization loop  — compresses old turns when context fills up
    3. Vector memory       — semantic retrieval of long-term facts
    4. Token budget        — tracks usage and triggers summarization
    """

    def __init__(self):
        self.state = ConversationState()
        self.budget = TokenBudget(CONTEXT_TOKEN_BUDGET)
        self.summarizer = Summarizer(CHAT_MODEL)
        self.vector_memory = VectorMemory(EMBED_MODEL, MEMORY_DB_PATH)
        self._last_memories: list[str] = []
        self._last_facts: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str) -> Generator[str, None, None]:
        """
        Stream the assistant's reply token-by-token.
        Memory extraction and state updates run after all tokens are yielded.
        """
        # 1. Summarize old turns if approaching the token budget
        self._maybe_summarize()

        # 2. Retrieve relevant long-term memories for this query
        self._last_memories = self.vector_memory.retrieve(user_message, k=MEMORY_TOP_K)

        # 3. Build the full prompt from context layers
        messages = self._build_messages(user_message)

        # 4. Record the user turn
        self.state.add("user", user_message)

        # 5. Stream the response
        full_response = ""
        for chunk in ollama.chat(model=CHAT_MODEL, messages=messages, stream=True):
            token = chunk.message.content
            full_response += token
            yield token

        # 6. Record the assistant turn
        self.state.add("assistant", full_response)

        # 7. Mine memorable facts from this exchange and store them
        self._last_facts = self.vector_memory.extract_and_store(
            user_message, full_response, CHAT_MODEL
        )

    def stats(self) -> dict:
        tokens_used = self.budget.used(self.state.messages, self.state.summary)
        return {
            "turns": self.state.turn_count,
            "messages_in_context": len(self.state.messages),
            "tokens_used": tokens_used,
            "token_budget": self.budget.budget,
            "pct_used": int(tokens_used / self.budget.budget * 100),
            "has_summary": self.state.summary is not None,
            "vector_memories": self.vector_memory.count(),
            "last_memories_retrieved": len(self._last_memories),
            "last_facts_stored": len(self._last_facts),
        }

    def list_memories(self) -> list[str]:
        return self.vector_memory.all_memories()

    def clear(self) -> None:
        """Reset conversation state; vector memories persist across clears."""
        self.state.clear()
        self._last_memories = []
        self._last_facts = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_messages(self, user_message: str) -> list[dict]:
        """
        Layer the context from outermost to innermost:
          system prompt → summary → vector memories → recent history → new message
        """
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        if self.state.summary:
            messages.append({
                "role": "system",
                "content": f"[Conversation summary]\n{self.state.summary}",
            })

        if self._last_memories:
            memories_text = "\n".join(f"- {m}" for m in self._last_memories)
            messages.append({
                "role": "system",
                "content": f"[Relevant memories from past conversations]\n{memories_text}",
            })

        messages.extend(self.state.messages)
        messages.append({"role": "user", "content": user_message})
        return messages

    def _maybe_summarize(self) -> None:
        """Compress the oldest turns when context usage exceeds the threshold."""
        if len(self.state.messages) <= RECENT_MESSAGES_KEEP:
            return
        if not self.budget.is_over_threshold(
            self.state.messages, self.state.summary, SUMMARIZE_THRESHOLD
        ):
            return

        n_to_summarize = len(self.state.messages) - RECENT_MESSAGES_KEEP
        old_messages = self.state.pop_oldest(n_to_summarize)
        new_summary = self.summarizer.summarize(old_messages, self.state.summary)
        self.state.set_summary(new_summary)
