import os
from typing import AsyncGenerator
from datetime import date, datetime

from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from ai.schemas import AgentEvent, AgentResult, AgentAnalysis
from ai.prompts import AGENT_SYSTEM_PROMPT

DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "")
QWEN_API_BASE = os.environ.get(
    "QWEN_API_BASE", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
QWEN_AGENT_MODEL = os.environ.get("QWEN_AGENT_MODEL", "qwen3.7-plus")

_agent = create_agent(
    model=ChatOpenAI(
        model=QWEN_AGENT_MODEL,
        api_key=DASHSCOPE_API_KEY,
        base_url=QWEN_API_BASE,
        temperature=0,
        max_tokens=4096,
        extra_body={"enable_thinking": False},
    ),
    tools=[
        TavilySearch(
            max_results=3,
            description=(
                "Search the web for current information about German bureaucratic "
                "processes, legal requirements, deadlines, and consequences. "
                "Use German keywords for better results."
            ),
        ),
    ],
    system_prompt=AGENT_SYSTEM_PROMPT,
    response_format=AgentAnalysis,
)


async def run_react_agent(ocr_text: str) -> AsyncGenerator[AgentEvent, None]:
    """Run the ReAct agent with structured output via LangGraph response_format."""
    try:
        result = await _agent.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=f"Analyze this German official letter:\n\n{ocr_text}"
                    )
                ],
            }
        )

        analysis: AgentAnalysis = result["structured_response"]

        days_remaining = analysis.deadline.days_remaining
        if analysis.deadline.date and days_remaining is None:
            try:
                dl = datetime.strptime(analysis.deadline.date, "%Y-%m-%d").date()
                days_remaining = (dl - date.today()).days
            except ValueError:
                pass

        yield AgentEvent(
            "classification",
            {
                "type": analysis.classification.type,
                "agency": analysis.classification.agency,
            },
        )
        yield AgentEvent(
            "risk_score",
            {"score": analysis.risk_score.score, "label": analysis.risk_score.label},
        )
        if analysis.deadline.date:
            yield AgentEvent(
                "deadline",
                {"date": analysis.deadline.date, "days_remaining": days_remaining},
            )
        yield AgentEvent("consequence", {"text": analysis.consequence.text})

    except Exception as e:
        yield AgentEvent("error", {"message": f"Agent failed: {e}"})


def get_last_agent_result(events: list[AgentEvent], ocr_text: str) -> AgentResult:
    """Reconstruct an AgentResult from collected events."""
    data = {e.type: e.data for e in events}
    c, d, r, q = (
        data.get("classification", {}),
        data.get("deadline", {}),
        data.get("risk_score", {}),
        data.get("consequence", {}),
    )
    return AgentResult(
        ocr_text=ocr_text,
        letter_type=c.get("type", "Unknown"),
        agency=c.get("agency", "Unknown"),
        deadline_date=d.get("date"),
        days_remaining=d.get("days_remaining"),
        consequence=q.get("text", ""),
        risk_score=r.get("score", 3),
        risk_label=r.get("label", "Medium"),
    )
