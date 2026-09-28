"""
Lab 11 — Helper Utilities
"""
import asyncio
import os

from core.config import get_llm_provider, PROVIDER_OPENROUTER  # noqa: F401
from core.openai_runtime import OpenAIRunner

# Gemini quotas / overload are per model, so when the primary model is busy
# (503 / 429) we retry it briefly, then walk through these fallbacks.
# Override with GEMINI_FALLBACK_MODELS="modelA,modelB" in .env.
DEFAULT_GEMINI_FALLBACKS = (
    "gemini-3.6-flash,gemini-3.7-flash,gemini-3-flash-preview,"
    "gemini-2.5-flash,gemini-3.1-flash-lite,gemini-3.5-flash-lite"
)
_TRANSIENT_MARKERS = (
    "503", "429", "500", "502", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
    "overloaded", "high demand", "quota", "DEADLINE_EXCEEDED",
    "connection", "connect", "timeout", "timed out", "reset by peer",
)
PRIMARY_RETRIES = 2
RETRY_DELAY_S = 5


def _is_transient(exc: BaseException) -> bool:
    msg = f"{type(exc).__name__}: {exc}"
    if not str(exc).strip():  # empty message (e.g. dropped connection) -> retry
        return True
    cause = exc.__cause__ or exc.__context__
    if cause is not None:
        msg += f" {type(cause).__name__}: {cause}"
    return any(m.lower() in msg.lower() for m in _TRANSIENT_MARKERS)


def gemini_fallback_models(primary: str) -> list[str]:
    raw = os.environ.get("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS)
    return [m.strip() for m in raw.split(",") if m.strip() and m.strip() != primary]


async def chat_with_agent(agent, runner, user_message: str, session_id=None):
    """Send a message to the agent and get the response.

    Works with OpenAIRunner (OpenAI Red / OpenRouter Blue) and Google ADK (Gemini Red).
    """
    provider = getattr(runner, "provider", None)
    if isinstance(runner, OpenAIRunner) or provider in ("openrouter", "openai"):
        text = await runner.chat(agent, user_message)
        return text, None

    primary = getattr(agent, "model", None)
    candidates = [primary] + (gemini_fallback_models(primary) if primary else [])
    last_exc: BaseException | None = None
    try:
        for idx, model in enumerate(candidates):
            # Primary gets a couple of retries with backoff; fallbacks get one shot.
            attempts = PRIMARY_RETRIES + 1 if idx == 0 else 1
            for attempt in range(attempts):
                if model and model != getattr(agent, "model", None):
                    agent.model = model
                try:
                    result = await _chat_adk(agent, runner, user_message, session_id)
                    if idx:
                        print(f"  [fallback] answered by {model} (primary {primary} unavailable)")
                    return result
                except Exception as exc:
                    if not _is_transient(exc):
                        raise
                    last_exc = exc
                    if attempt + 1 < attempts:
                        await asyncio.sleep(RETRY_DELAY_S * (attempt + 1))
            print(f"  [fallback] {model} unavailable, trying next model...")
        raise last_exc  # every model failed
    finally:
        if primary:
            agent.model = primary  # never leave the agent on a fallback model


async def _chat_adk(agent, runner, user_message: str, session_id=None):
    """One Google-ADK round trip (Gemini Red / Red Advance)."""
    from google.genai import types

    user_id = "student"
    app_name = runner.app_name

    session = None
    if session_id is not None:
        try:
            session = await runner.session_service.get_session(
                app_name=app_name, user_id=user_id, session_id=session_id
            )
        except (ValueError, KeyError):
            pass

    if session is None:
        try:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )
        except Exception:
            session = await runner.session_service.create_session(
                app_name=app_name, user_id=user_id
            )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    final_response = ""
    async for event in runner.run_async(
        user_id=user_id, session_id=session.id, new_message=content
    ):
        if hasattr(event, "content") and event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_response += part.text

    return final_response, session
