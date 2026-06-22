import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ai.schemas import AgentResult, GenerationOutput
from ai.prompts import GENERATION_PROMPT
from ai.rag.retrieval import retrieve_as_context

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_API_BASE = os.environ.get(
    "QWEN_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
QWEN_AGENT_MODEL = os.environ.get("QWEN_AGENT_MODEL", "qwen3.7-plus")

LANGUAGE_NAMES = {
    "en": "English",
    "de": "German",
    "tr": "Turkish",
    "ar": "Arabic",
    "es": "Spanish",
    "fr": "French",
    "zh": "Chinese",
    "fa": "Persian",
}

_model = ChatOpenAI(
    model=QWEN_AGENT_MODEL,
    api_key=DASHSCOPE_API_KEY,
    base_url=QWEN_API_BASE,
    temperature=0,
    max_tokens=4096,
    extra_body={"enable_thinking": False},
).with_structured_output(GenerationOutput, method="json_mode")


async def generate_response(
    ocr_text: str,
    agent_result: AgentResult,
    language: str = "en",
) -> GenerationOutput:
    """Retrieve legal context from ChromaDB, inject into prompt, return structured output."""
    legal_context = retrieve_as_context(
        agent_result.letter_type, agent_result.consequence
    )

    prompt = GENERATION_PROMPT.format(
        ocr_text=ocr_text[:3000],
        letter_type=agent_result.letter_type,
        agency=agent_result.agency,
        deadline_date=agent_result.deadline_date or "Not specified",
        days_remaining=agent_result.days_remaining or "Unknown",
        risk_score=agent_result.risk_score,
        risk_label=agent_result.risk_label,
        consequence=agent_result.consequence,
        legal_context=legal_context,
        language=LANGUAGE_NAMES.get(language, "English"),
    )

    return await _model.ainvoke([HumanMessage(content=prompt)])
