"""ChromaDB embedded vector store for RAG over the German bureaucracy corpus."""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings

_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def _make_client() -> chromadb.ClientAPI:
    Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
    )


def get_collection() -> chromadb.Collection:
    global _client, _collection
    if _client is None:
        _client = _make_client()
    if _collection is None:
        _collection = _client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"description": "German bureaucratic knowledge corpus for Klar"},
        )
    return _collection


def init_chroma() -> None:
    coll = get_collection()
    if coll.count() == 0:
        from app.rag.seed import seed_corpus

        seed_corpus(coll)


def search(query: str, top_k: int = 4, where: dict | None = None) -> list[dict]:
    coll = get_collection()
    results = coll.query(query_texts=[query], n_results=top_k, where=where)
    docs = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    return [
        {"text": d, "metadata": m or {}, "score": 1.0 - float(s)}
        for d, m, s in zip(docs, metadatas, distances)
    ]


def upsert(documents: list[str], metadatas: list[dict], ids: list[str]) -> None:
    coll = get_collection()
    coll.upsert(documents=documents, metadatas=metadatas, ids=ids)
