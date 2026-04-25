import uuid

import chromadb
import ollama


class VectorMemory:
    """
    Semantic long-term memory backed by ChromaDB + Ollama embeddings.

    Flow per turn:
      1. retrieve() — pull top-k memories relevant to the user's message
      2. extract_and_store() — ask the LLM to mine facts from the exchange
    """

    def __init__(self, embed_model: str, db_path: str):
        self.embed_model = embed_model
        self.db = chromadb.PersistentClient(path=db_path)
        # We always supply our own embeddings, so no built-in EF needed.
        self.collection = self.db.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        response = ollama.embed(model=self.embed_model, input=text)
        return response.embeddings[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(self, text: str, metadata: dict | None = None) -> None:
        self.collection.add(
            embeddings=[self._embed(text)],
            documents=[text],
            ids=[str(uuid.uuid4())],
            metadatas=[metadata or None],
        )

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Return the k most semantically similar stored memories."""
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_embeddings=[self._embed(query)],
            n_results=min(k, count),
        )
        return results["documents"][0] if results["documents"] else []

    def extract_and_store(
        self, user_msg: str, assistant_msg: str, chat_model: str
    ) -> list[str]:
        """Ask the LLM to extract memorable facts from one exchange."""
        prompt = (
            f"User: {user_msg}\nAssistant: {assistant_msg}\n\n"
            "Extract 1-3 specific facts worth remembering long-term "
            "(names, preferences, goals, decisions, key details). "
            "Return one fact per line. If nothing is memorable, return NONE."
        )
        response = ollama.chat(
            model=chat_model,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 120},
        )
        raw = response.message.content.strip()
        if not raw or raw.upper() == "NONE":
            return []

        facts = [
            line.strip()
            for line in raw.splitlines()
            if line.strip() and line.strip().upper() != "NONE"
        ]
        for fact in facts:
            self.store(fact, {"source": "auto-extract"})
        return facts

    def all_memories(self) -> list[str]:
        results = self.collection.get()
        return results.get("documents") or []

    def count(self) -> int:
        return self.collection.count()
