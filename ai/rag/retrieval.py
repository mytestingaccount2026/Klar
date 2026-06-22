"""
Klar — RAG Retrieval Module
ai/rag/retrieval.py

Called by the FastAPI backend (Dev 2) to fetch relevant § paragraphs
for a given letter's OCR text and classification.

Usage:
    from ai.rag.retrieval import retrieve_legal_context, retrieve_as_context

    chunks = retrieve_legal_context(
        ocr_text="Ihr Aufenthaltstitel läuft am 31.07.2026 ab...",
        letter_type="Aufenthaltserlaubnis Verlängerung",
        top_k=5
    )
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import chromadb
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent  # ai/
CHROMA_DIR = ROOT / "data" / "chroma"  # ai/data/chroma/
COLLECTION_NAME = "german_laws"

# ── Schema ────────────────────────────────────────────────────────────────────


@dataclass
class LegalChunk:
    section: str  # e.g. "§ 81"
    law: str  # e.g. "AufenthG"
    title: str  # e.g. "§ 81 Beantragung des Aufenthaltstitels"
    text: str  # full paragraph text
    citation: str  # e.g. "§ 81 AufenthG"
    score: float  # cosine similarity, higher = more relevant


# ── Singleton clients ─────────────────────────────────────────────────────────
# Loaded once on first call, reused across all requests

_collection = None
_qwen_client = None


def _get_qwen_client() -> OpenAI:
    global _qwen_client
    if _qwen_client is None:
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            print("ERROR: DASHSCOPE_API_KEY not set.")
            sys.exit(1)
        _qwen_client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        )
    return _qwen_client


def _get_collection():
    global _collection
    if _collection is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                f"ChromaDB not found at {CHROMA_DIR}. "
                "Run `python ai/rag/ingest.py` first."
            )
        db = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = db.get_collection(name=COLLECTION_NAME)
    return _collection


def _embed_query(query: str) -> list[float]:
    """Embed the search query using the same Qwen model used during ingestion."""
    client = _get_qwen_client()
    response = client.embeddings.create(
        model="text-embedding-v3",
        input=[query],
    )
    return response.data[0].embedding


# ── Core retrieval ────────────────────────────────────────────────────────────


def retrieve_legal_context(
    letter_type: str,
    consequence: str,
    top_k: int = 5,
) -> list[LegalChunk]:
    """
    Retrieve the top_k most relevant § paragraphs from ChromaDB.

    Uses the agent's analysis (type + consequence) as the query — this captures
    the legal essence of the letter far better than raw OCR text.

    Args:
        letter_type — classification from the ReAct agent
        consequence — consequence assessment from the ReAct agent
        top_k       — number of chunks to return (default 5)

    Returns:
        List of LegalChunk objects sorted by relevance (most relevant first)
    """
    query = f"{letter_type}. {consequence}"

    query_embedding = _embed_query(query)
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        section = meta.get("paragraph", meta.get("section", "Unknown"))
        chunks.append(
            LegalChunk(
                section=section,
                law=meta["law"],
                title=meta["title"],
                text=doc,
                citation=f"{section} {meta['law']}",
                score=round(1 - distance, 4),
            )
        )

    return chunks


def retrieve_as_context(
    letter_type: str,
    consequence: str,
    top_k: int = 5,
) -> str:
    """
    Same as retrieve_legal_context() but returns a single formatted string
    ready to be injected into the LLM generation prompt.
    """
    chunks = retrieve_legal_context(letter_type, consequence, top_k=top_k)

    if not chunks:
        return "No relevant legal paragraphs found."

    parts = []
    for c in chunks:
        # Truncate very long paragraphs to keep prompt size manageable
        text_preview = c.text[:1000] + "..." if len(c.text) > 1000 else c.text
        parts.append(f"[{c.citation}] {c.title}\n{text_preview}")

    try:
        return "\n\n---\n\n".join(parts)
    except Exception:
        return "Error formatting legal context."
