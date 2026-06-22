"""Vision + structured extraction + long-form generation via Qwen3.7-Plus.

All calls go through the OpenAI-compatible chat-completions API. The model
running server-side is Qwen — the openai SDK is just the wire-format client.

Two public entry points:

- `extract_from_letter_file(path, mime, lang)` — single non-streaming call that
  returns a fully-populated ExtractedLetter (structured fields). PDFs are
  expanded to one image per page via pdf_pages.split_to_image_bytes.

- `stream_long_form(extracted, lang)` — streaming call that emits chunks of
  language-localized explanation + checklist + (German) response_draft.
"""

import json
import re
from typing import AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.models import DocumentCategory
from app.rag import store
from app.schemas import ExtractedLetter
from app.services.pdf_pages import iter_data_urls, split_to_image_bytes


DOCUMENT_CATEGORIES: list[str] = [c.value for c in DocumentCategory]

# ISO 639-1 codes — matches the frontend's docs/06-frontend-integration-contract.md.
# Qwen3.7-Plus handles all of these out of the box. Quality bar:
#   en/de:           production-grade, the wedge languages
#   fa/tr/ar/uk:    reasonable for the hackathon demo; not evaluated against natives
# RTL (fa, ar) is handled entirely on the frontend — we just produce the text.
SUPPORTED_LANGS = {"en", "de", "fa", "tr", "ar", "uk"}

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.effective_llm_api_key,
            base_url=settings.effective_llm_base_url,
        )
    return _client


def normalize_lang(lang: str | None) -> str:
    if not lang:
        return "en"
    short = lang.strip().lower()[:2]
    return short if short in SUPPORTED_LANGS else "en"


def _lang_label(lang: str) -> str:
    return {
        "en": "English",
        "de": "German (Deutsch)",
        "fa": "Persian/Farsi (فارسی)",
        "tr": "Turkish (Türkçe)",
        "ar": "Arabic (العربية)",
        "uk": "Ukrainian (українська)",
    }.get(lang, "English")


# ---------- structured-extraction tool schema ----------


EXTRACT_FUNCTION = {
    "type": "function",
    "function": {
        "name": "extract_obligations",
        "description": (
            "Extract structured obligations, deadlines, and required actions from a German "
            "bureaucratic letter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "description": (
                        "The specific German document name as it appears on the letter "
                        "(e.g., 'Beitragsrechnung', 'Mahnung', 'Steuerbescheid', "
                        "'Mieterhöhung', 'Bußgeldbescheid')."
                    ),
                },
                "institution": {
                    "type": "string",
                    "description": (
                        "The exact name of the sender institution as printed on the letter "
                        "(e.g., 'AOK Bayern', 'Finanzamt Hamburg-Mitte', 'Beitragsservice "
                        "ARD ZDF Deutschlandradio')."
                    ),
                },
                "category": {
                    "type": "string",
                    "enum": DOCUMENT_CATEGORIES,
                    "description": (
                        "High-level classification. Pick the single best-fitting bucket. "
                        "Use 'other' only when no category fits."
                    ),
                },
                "category_confidence": {
                    "type": "number",
                    "description": "0.0-1.0 confidence in the category assignment.",
                },
                "language_confidence": {"type": "number"},
                "summary": {
                    "type": "string",
                    "description": (
                        "Plain-language summary, max 2 sentences, written in the "
                        "user's chosen output language (see system prompt)."
                    ),
                },
                "ocr_text": {
                    "type": "string",
                    "description": "The verbatim German text content of the letter.",
                },
                "extraction_warnings": {"type": "array", "items": {"type": "string"}},
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "deadline_iso": {
                                "type": "string",
                                "description": "YYYY-MM-DD, or empty string if unknown.",
                            },
                            "deadline_confidence": {"type": "number"},
                            "deadline_source": {
                                "type": "string",
                                "enum": ["explicit", "inferred", "unknown"],
                            },
                            "severity": {
                                "type": "string",
                                "enum": ["critical", "high", "medium", "low"],
                            },
                            "steps": {"type": "array", "items": {"type": "string"}},
                            "reply_needed": {"type": "boolean"},
                            "evidence_span": {
                                "type": "string",
                                "description": "Exact German quote from source doc.",
                            },
                        },
                        "required": ["title", "severity"],
                    },
                },
            },
            "required": [
                "document_type",
                "institution",
                "category",
                "summary",
                "ocr_text",
                "actions",
            ],
        },
    },
}


CATEGORY_GUIDE = """\
Classify the letter into exactly one of these categories:

- health_insurance: Krankenkasse / Krankenversicherung — AOK, TK, BARMER, DAK, private KV.
- other_insurance: Haftpflicht, Hausrat, KFZ, Lebens-, Reise-, Berufsunfähigkeit.
- banking: bank accounts, credit cards, loans, SCHUFA.
- tax: Finanzamt only.
- immigration: Ausländerbehörde.
- education: Universität, Hochschule, Studentenwerk, BAföG-Amt.
- housing: Vermieter, Hausverwaltung.
- utilities: Strom, Gas, Wasser, Internet, Telefon, Mobilfunk, Müllabfuhr.
- employment: Arbeitgeber and HR letters.
- government_benefits: Bundesagentur für Arbeit, Jobcenter, Familienkasse, Wohngeldstelle.
- pension: Deutsche Rentenversicherung.
- broadcast_fee: Beitragsservice ARD ZDF Deutschlandradio (formerly GEZ).
- civic: Bürgeramt, Einwohnermeldeamt, Standesamt, Personalausweisbehörde.
- legal_debt: Amtsgericht, Mahngericht, Inkasso-Dienste, Bußgeldstellen, Anwaltskanzleien.
- other: only when none of the above fits.
"""


def _structured_system_prompt(lang: str) -> str:
    out_lang = _lang_label(lang)
    return (
        "You are a German bureaucratic document interpreter. Read the attached letter "
        "and extract every legal obligation, deadline, and required action. Return your "
        "output ONLY via the extract_obligations function.\n\n"
        f"Output language: write `summary` in {out_lang}. The `ocr_text` field must "
        "contain the verbatim German text. `evidence_span` MUST be the exact German "
        "sentence the action came from. Action `title` and `description` should be in "
        f"{out_lang} so the user can act on them.\n\n"
        "Classification:\n"
        "- You MUST pick exactly one category from the enumerated list.\n"
        "- Provide category_confidence between 0.0 and 1.0.\n"
        "- Use 'other' only as a last resort.\n\n" + CATEGORY_GUIDE + "\n"
        "Rules:\n"
        "- If a deadline is not explicitly stated, you may infer it with low confidence "
        "and set deadline_source='inferred'. If you cannot determine a value, return an "
        "empty string — NEVER invent dates, amounts, or reference numbers.\n"
        "- Every action MUST include an evidence_span: the exact German sentence the "
        "action came from.\n"
        "- Document_type should be the specific German document name (e.g., 'Mahnung', "
        "'Beitragsrechnung', 'Steuerbescheid'), not the category.\n"
        "- You do not provide legal advice."
    )


# ---------- public: structured extraction ----------


async def extract_from_letter_file(
    path: str,
    mime: str,
    lang: str = "en",
) -> ExtractedLetter:
    """Single Qwen call: vision → structured JSON.

    PDFs are split into one image per page (poppler/pdf2image); each page is
    sent as a separate image_url content part so the model sees the whole doc.
    """
    pages = split_to_image_bytes(path, mime)
    image_parts = list(iter_data_urls(pages))

    rag_hits = store.search(
        "German bureaucratic institution identification deadline phrasing",
        top_k=4,
    )
    rag_context = "\n".join(f"- {h['text']}" for h in rag_hits)

    messages = [
        {
            "role": "system",
            "content": _structured_system_prompt(lang)
            + "\n\nReference knowledge:\n"
            + rag_context,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"This is a {len(pages)}-page German bureaucratic letter. "
                        "Extract all obligations."
                    ),
                },
                *image_parts,
            ],
        },
    ]

    client = _get_client()
    response = await client.chat.completions.create(
        model=settings.effective_llm_model,
        messages=messages,
        tools=[EXTRACT_FUNCTION],
        tool_choice={"type": "function", "function": {"name": "extract_obligations"}},
        temperature=0.1,
    )

    tool_calls = response.choices[0].message.tool_calls or []
    if not tool_calls:
        raise RuntimeError(
            "Model returned no tool call; check model + prompt compatibility."
        )
    payload = json.loads(tool_calls[0].function.arguments)

    # Defensive: models can emit `actions` as
    #   - a proper list of dicts (happy path)
    #   - a list of strings (qwen sometimes emits bullets)
    #   - a string that's actually a JSON-encoded list: '[{"title":"..."}, ...]'
    #     (qwen3.7-plus does this regularly — re-parse it as JSON)
    #   - a single freeform string (qwen sometimes emits a paragraph) —
    #     iterating a string yields one char each, which previously gave us
    #     1986 actions
    raw_actions = payload.get("actions")
    if isinstance(raw_actions, str):
        # Try to re-parse if it looks like JSON
        stripped = raw_actions.strip()
        json_parsed = None
        if stripped.startswith(("[", "{")):
            try:
                json_parsed = json.loads(stripped)
            except json.JSONDecodeError:
                # Sometimes the model wraps the JSON in code fences
                fence_match = re.search(
                    r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL
                )
                if fence_match:
                    try:
                        json_parsed = json.loads(fence_match.group(1).strip())
                    except json.JSONDecodeError:
                        pass

        if isinstance(json_parsed, list):
            raw_actions = json_parsed
        elif isinstance(json_parsed, dict):
            raw_actions = [json_parsed]
        else:
            # Last-resort: wrap the freeform string as a single action title.
            raw_actions = [{"title": raw_actions[:200], "severity": "medium"}]
    elif not isinstance(raw_actions, list):
        raw_actions = []

    cleaned_actions: list[dict] = []
    for action in raw_actions:
        if isinstance(action, str):
            cleaned_actions.append({"title": action[:200], "severity": "medium"})
            continue
        if not isinstance(action, dict):
            continue
        if action.get("deadline_iso") in ("", None):
            action["deadline_iso"] = None
        cleaned_actions.append(action)

    # Sanity cap: any extraction with >25 actions on a single letter is almost
    # certainly a model misfire. Drop the tail rather than spam the DB.
    payload["actions"] = cleaned_actions[:25]

    return ExtractedLetter.model_validate(payload)


# ---------- public: streaming long-form generation ----------


def _explanation_prompt(extracted: ExtractedLetter, lang: str) -> str:
    out_lang = _lang_label(lang)
    return (
        f"You are explaining a German bureaucratic letter to a non-native speaker, in "
        f"{out_lang}. Be direct and concrete. 4-7 sentences max. Cover: who sent the "
        "letter, what they want, what the user must do, and what happens if ignored. "
        "Use plain language, no legalese. No greeting, no sign-off — just the "
        "explanation prose.\n\n"
        f"Document: {extracted.document_type} from {extracted.institution} "
        f"(category: {extracted.category.value}).\n"
        f"Summary: {extracted.summary}\n"
        f"Number of obligations: {len(extracted.actions)}.\n"
        f"German text:\n{extracted.ocr_text[:4000]}"
    )


def _response_prompt_from_letter(
    institution: str,
    document_type: str,
    actions_text: str,
    applicant: dict | None,
) -> str:
    """Build the reply prompt directly from persisted letter data (not from
    an ExtractedLetter object) so the sync /reply endpoint can call it after
    the letter is in the DB. Always produces formal German."""
    applicant_lines = ""
    if applicant:
        lines = [f"- {k}: {v}" for k, v in applicant.items() if v]
        if lines:
            applicant_lines = "\n\nAbsenderdaten (Briefkopf):\n" + "\n".join(lines)
    return (
        "Du bist ein deutscher Briefassistent. Verfasse einen kurzen, formellen "
        "deutschen Antwortbrief an die folgende Behörde. Format: postalisch "
        "(Anrede 'Sehr geehrte Damen und Herren,' / Schluss 'Mit freundlichen Grüßen'). "
        "Halte Aktenzeichen oder Referenznummern bei, falls in den Aktionen erwähnt. "
        "Keine erfundenen Daten oder IBANs.\n\n"
        f"Empfänger: {institution}\n"
        f"Dokumenttyp: {document_type}\n"
        f"Aktionen, auf die geantwortet werden muss:\n{actions_text}"
        f"{applicant_lines}\n\n"
        "Schreibe ausschließlich den Brieftext (keine Erklärungen)."
    )


async def generate_reply_text(
    institution: str,
    document_type: str,
    action_titles: list[str],
    applicant: dict | None = None,
) -> str:
    """Synchronous reply generation — one Qwen call, full string returned.

    Used by POST /letters/{id}/reply (frontend contract §4.7).
    """
    actions_text = (
        "\n".join(f"- {t}" for t in action_titles) or "- (keine spezifische Aktion)"
    )
    prompt = _response_prompt_from_letter(
        institution, document_type, actions_text, applicant
    )

    client = _get_client()
    response = await client.chat.completions.create(
        model=settings.effective_llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return (response.choices[0].message.content or "").strip()


def _response_prompt(extracted: ExtractedLetter) -> str:
    # Reply ALWAYS in formal German regardless of user language.
    actions = "\n".join(f"- {a.title}" for a in extracted.actions)
    return (
        "Du bist ein deutscher Briefassistent. Verfasse einen kurzen, formellen "
        "deutschen Antwortbrief an die folgende Behörde. Format: postalisch "
        "(Anrede 'Sehr geehrte Damen und Herren,' / Schluss 'Mit freundlichen Grüßen'). "
        "Halte Aktenzeichen oder Referenznummern bei, falls in den Aktionen erwähnt. "
        "Keine erfundenen Daten oder IBANs.\n\n"
        f"Empfänger: {extracted.institution}\n"
        f"Dokumenttyp: {extracted.document_type}\n"
        f"Aktionen, auf die geantwortet werden muss:\n{actions}\n\n"
        "Schreibe ausschließlich den Brieftext (keine Erklärungen)."
    )


def _checklist_prompt(extracted: ExtractedLetter, lang: str) -> str:
    out_lang = _lang_label(lang)
    actions = "\n".join(f"- {a.title}" for a in extracted.actions)
    return (
        f"List the concrete documents and items the user must gather to comply with "
        f"this letter. Output a JSON array of short strings, in {out_lang}. Max 8 "
        "items. No prose, no markdown — JSON array only.\n\n"
        f"Letter: {extracted.document_type} from {extracted.institution}.\n"
        f"Required actions:\n{actions}"
    )


async def _stream_text(prompt: str) -> AsyncIterator[str]:
    client = _get_client()
    stream = await client.chat.completions.create(
        model=settings.effective_llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content


async def stream_explanation(
    extracted: ExtractedLetter, lang: str
) -> AsyncIterator[str]:
    async for piece in _stream_text(_explanation_prompt(extracted, lang)):
        yield piece


async def stream_response_draft(extracted: ExtractedLetter) -> AsyncIterator[str]:
    async for piece in _stream_text(_response_prompt(extracted)):
        yield piece


async def generate_checklist(extracted: ExtractedLetter, lang: str) -> list[str]:
    """Non-streaming list output. Forces JSON-array response."""
    client = _get_client()
    response = await client.chat.completions.create(
        model=settings.effective_llm_model,
        messages=[{"role": "user", "content": _checklist_prompt(extracted, lang)}],
        temperature=0.2,
    )
    raw = (response.choices[0].message.content or "").strip()
    # Best-effort parse — model may return code-fenced JSON.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip("`").strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(x) for x in parsed][:8]
    except json.JSONDecodeError:
        pass
    return [line.lstrip("-• ").strip() for line in raw.splitlines() if line.strip()][:8]


# ---------- backwards-compat alias for the original entrypoint ----------


async def extract_from_image(
    image_bytes: bytes, mime: str = "image/jpeg"
) -> ExtractedLetter:
    """Legacy entrypoint: write bytes to a temp file then call the new API."""
    import tempfile
    import os

    ext = "pdf" if mime == "application/pdf" else "jpg"
    fd, tmp = tempfile.mkstemp(suffix=f".{ext}")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(image_bytes)
        return await extract_from_letter_file(tmp, mime, lang="en")
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
