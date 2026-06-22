"""
Klar — RAG Schemas
ai/rag/schemas.py
"""

from dataclasses import dataclass


@dataclass
class RAGEvent:
    type: str  # "explanation" | "response_draft" | "checklist" | "citations" | "error"
    data: dict
    confidence: str = "high"  # "high" if RAG matched well, "low" if no strong matches
    # data shapes per type:
    #   explanation:    {"chunk": str}              — streamed token by token
    #   response_draft: {"chunk": str}              — streamed token by token
    #   checklist:      {"items": list[str]}        — emitted once, complete
    #   citations:      {"items": list[dict]}       — emitted once, [{section, text}, ...]
    #   error:          {"message": str}
