"""
LLM client utilities for PiGollum.

Provides a thin wrapper around the OpenAI client (which is compatible with
local LLM servers such as Ollama, vLLM, LM Studio, etc.) and gracefully
handles missing credentials so that the rest of PiGollum can degrade to
pure BO when no LLM is available.
"""
import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)

# Sentinel that signals "no LLM available"
_UNAVAILABLE = None


def build_llm_client(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """
    Build and return an ``openai.OpenAI`` client instance, or ``None`` if the
    required credentials are not available.

    Resolution order for each parameter:
        1. Explicit argument
        2. Environment variable (PIGOLLUM_LLM_API_KEY / PIGOLLUM_LLM_BASE_URL /
           PIGOLLUM_LLM_MODEL)
        3. Fallback to standard OPENAI_API_KEY / OPENAI_BASE_URL

    Returns
    -------
    client : openai.OpenAI | None
        A configured client, or None if openai is not installed or no key found.
    model_name : str | None
        The resolved model name string.
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed – LLM principle extraction disabled.")
        return _UNAVAILABLE, _UNAVAILABLE

    resolved_key = (
        api_key
        or os.environ.get("PIGOLLUM_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    resolved_url = (
        base_url
        or os.environ.get("PIGOLLUM_LLM_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )
    resolved_model = (
        model_name
        or os.environ.get("PIGOLLUM_LLM_MODEL")
        or "gpt-4o-mini"
    )

    if resolved_key is None:
        logger.warning(
            "No LLM API key found (PIGOLLUM_LLM_API_KEY / OPENAI_API_KEY). "
            "Principle extraction will be disabled; pure BO will run."
        )
        return _UNAVAILABLE, _UNAVAILABLE

    kwargs = {"api_key": resolved_key}
    if resolved_url:
        kwargs["base_url"] = resolved_url

    client = OpenAI(**kwargs)
    logger.info("LLM client initialised: model=%s base_url=%s", resolved_model, resolved_url)
    return client, resolved_model


def chat_complete(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.4,
    max_tokens: int = 512,
) -> Optional[str]:
    """
    Send a single-turn chat completion and return the response text.

    Returns None on error so callers can gracefully fall back.
    """
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return None
