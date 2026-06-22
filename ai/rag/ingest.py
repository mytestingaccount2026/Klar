"""
Klar — RAG Ingestion Script
ai/rag/ingest.py

Run ONCE before the demo to build the ChromaDB vector store.

Usage:
    cd Klar/
    python ai/rag/ingest.py

Requirements:
    pip install chromadb openai

Env vars:
    DASHSCOPE_API_KEY — your Qwen API key
"""

import os
import re
import sys
import time
from pathlib import Path

import chromadb
from openai import OpenAI

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent  # ai/
LAWS_DIR = ROOT / "data" / "laws"  # ai/data/laws/
CHROMA_DIR = ROOT / "data" / "chroma"  # ai/data/chroma/
COLLECTION_NAME = "german_laws"

# ── Law file → abbreviation map ───────────────────────────────────────────────

LAWS = {
    "aufenthg.md": "AufenthG",
    "aufenthv.md": "AufenthV",
    "beschv.md": "BeschV",
    "vwvfg.md": "VwVfG",
    "bafoeg.md": "BAföG",
    "asylg.md": "AsylG",
    "asylblg.md": "AsylbLG",
    "wogg.md": "WoGG",
    "bmg.md": "BMG",
    "intv.md": "IntV",
    "owig.md": "OWiG",
    "estg.md": "EStG",
    "sgb5.md": "SGB V",
}

# ── Qwen client ───────────────────────────────────────────────────────────────


def get_qwen_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("ERROR: DASHSCOPE_API_KEY environment variable not set.")
        print("  Windows: set DASHSCOPE_API_KEY=your_key_here")
        sys.exit(1)
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    )


# Qwen text-embedding-v3 limits
MAX_BATCH_SIZE = 10  # max texts per API call
MAX_CHARS = 6000  # conservative char limit (~8192 tokens safety margin)


def truncate(text: str) -> str:
    """Truncate text to stay within Qwen's 8192 token limit."""
    return text[:MAX_CHARS] if len(text) > MAX_CHARS else text


def embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """
    Embed a batch of texts using Qwen text-embedding-v3.
    - Max 10 texts per API call
    - Truncates chunks exceeding token limit
    """
    embeddings = []

    # Truncate all texts first
    texts = [truncate(t) for t in texts]

    for i in range(0, len(texts), MAX_BATCH_SIZE):
        batch = texts[i : i + MAX_BATCH_SIZE]
        response = client.embeddings.create(
            model="text-embedding-v3",
            input=batch,
        )
        embeddings.extend([item.embedding for item in response.data])

        # Small delay to avoid hitting rate limits
        if i + MAX_BATCH_SIZE < len(texts):
            time.sleep(0.5)

    return embeddings


# ── Chunking ──────────────────────────────────────────────────────────────────


def parse_paragraphs(text: str, law_abbrev: str) -> list[dict]:
    """
    Split a law's markdown into one chunk per § paragraph.
    Each chunk keeps the full § text including all Absätze (subsections).

    Returns list of dicts with keys:
        id, text, paragraph, title, law
    """
    # Match § headers at any heading level: ### § 1, #### § 4a, etc.
    pattern = r"^#{1,4} (§ \d+[a-z]?\b.*?)$"
    matches = list(re.finditer(pattern, text, re.MULTILINE))

    chunks = []
    seen_ids = set()

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        header = match.group(1).strip()
        para_num_match = re.match(r"(§ \d+[a-z]?)", header)
        para_num = para_num_match.group(1) if para_num_match else header

        body = text[start:end].strip()

        # Skip empty or near-empty chunks
        if len(body) < 80:
            continue

        # Build unique ID
        base_id = f"{law_abbrev}_{para_num.replace(' ', '_').replace('§', 'para')}"

        # Handle rare duplicates
        unique_id = base_id
        counter = 1
        while unique_id in seen_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        seen_ids.add(unique_id)

        chunks.append(
            {
                "id": unique_id,
                "text": body,
                "paragraph": para_num,
                "title": header,
                "law": law_abbrev,
            }
        )

    return chunks


# ── Main ──────────────────────────────────────────────────────────────────────


def ingest_all():
    print("── Klar RAG Ingestion ──────────────────────────────────────")

    if not LAWS_DIR.exists():
        print(f"ERROR: {LAWS_DIR} not found.")
        print("  Make sure you run from the Klar/ project root.")
        sys.exit(1)

    client = get_qwen_client()
    print("✅ Qwen client ready")

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    db = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        db.delete_collection(COLLECTION_NAME)
        print("   Deleted existing collection (re-ingesting fresh)")
    except Exception:
        pass

    collection = db.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"✅ ChromaDB collection '{COLLECTION_NAME}' created")
    print()

    total_chunks = 0

    for filename, law_abbrev in LAWS.items():
        filepath = LAWS_DIR / filename

        if not filepath.exists():
            print(f"  ⚠  MISSING: {filename} — skipping")
            continue

        text = filepath.read_text(encoding="utf-8")
        chunks = parse_paragraphs(text, law_abbrev)

        if not chunks:
            print(f"  ⚠  No paragraphs parsed in {filename}")
            continue

        print(
            f"  Embedding {law_abbrev}: {len(chunks)} paragraphs ...",
            end=" ",
            flush=True,
        )

        texts_to_embed = [c["text"] for c in chunks]
        embeddings = embed_texts(client, texts_to_embed)

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            collection.add(
                ids=[c["id"] for c in batch_chunks],
                documents=[c["text"] for c in batch_chunks],
                metadatas=[
                    {
                        "law": c["law"],
                        "paragraph": c["paragraph"],
                        "title": c["title"],
                    }
                    for c in batch_chunks
                ],
                embeddings=batch_embeddings,
            )

        print("✅")
        total_chunks += len(chunks)

    print()
    print("── Done ────────────────────────────────────────────────────")
    print(f"   Total chunks ingested : {total_chunks}")
    print(f"   ChromaDB saved to     : {CHROMA_DIR.resolve()}")
    print()
    print("RAG retrieval is ready.")


if __name__ == "__main__":
    ingest_all()
