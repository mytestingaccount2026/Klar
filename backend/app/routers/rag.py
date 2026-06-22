"""RAG search over the German bureaucracy knowledge corpus (auth-required)."""

from fastapi import APIRouter, Depends

from app.auth.dependencies import get_current_user
from app.models import User
from app.rag import store
from app.schemas import ErrorResponse, RagQuery, RagResponse
from app.services import ai_bridge

router = APIRouter(prefix="/api/rag", tags=["rag"])


@router.post(
    "/search",
    response_model=RagResponse,
    summary="Search the German bureaucracy knowledge corpus",
    description=(
        "Returns the top-k vector-similarity matches from the seed corpus "
        "(institutions, phrase patterns, deadline conventions). Internal "
        "debug surface — not part of the production frontend's main flow."
    ),
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
    },
)
def search(payload: RagQuery, _: User = Depends(get_current_user)):
    """RAG search over the German legal corpus.

    Backed by the AI team's 120k-line corpus at `ai/data/chroma/` (see
    docs/07 §12). Returns real `§` references from AufenthG, SGB V, EStG,
    BMG, VwVfG, BAföG, OWiG, AsylG, IntV, BeschV, AsylBLG, AufenthV, WoGG.
    """
    from ai.rag.retrieval import retrieve_legal_context

    # AI team's signature changed in 61fd2b5: now (letter_type, consequence, top_k).
    # For an open-ended /rag/search the query IS the letter-type/topic, and the
    # institution (if given) serves as the consequence/context.
    chunks = retrieve_legal_context(
        letter_type=payload.query,
        consequence=payload.institution or "",
        top_k=payload.top_k,
    )
    return RagResponse(hits=[ai_bridge.legal_chunk_to_rag_hit(c) for c in chunks])


@router.post(
    "/reseed",
    summary="Reload the seed corpus into ChromaDB",
    description="Upserts every entry from `app/rag/seed.py::SEED_DOCS`.",
    responses={
        401: {"model": ErrorResponse, "description": "Not authenticated."},
    },
)
def reseed(_: User = Depends(get_current_user)):
    from app.rag.seed import seed_corpus

    coll = store.get_collection()
    seed_corpus(coll)
    return {"seeded": coll.count()}
