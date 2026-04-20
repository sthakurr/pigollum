"""
LLM client utilities for PiGollum.

Supports three API backends:
  1. OpenAI-compatible  — any endpoint using the OpenAI wire protocol
     (OpenAI, Together, Groq, vLLM, Ollama, LM Studio, …)
  2. Gemini             — Google Gemini via its OpenAI-compatible REST layer
     Activate with  backend="gemini"  or env GEMINI_API_KEY / GOOGLE_API_KEY.
     No extra SDK required — uses the same openai package pointed at:
       https://generativelanguage.googleapis.com/v1beta/openai/
  3. HuggingFace local  — handled directly in extractor.py via transformers

Gracefully degrades to pure BO when no LLM is available.
"""
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

_UNAVAILABLE = None

# Gemini OpenAI-compatible endpoint
_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
_GEMINI_DEFAULT_MODEL = "gemini-2.0-flash"


def build_gemini_client(
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
):
    """
    Build an OpenAI client pointed at Google's Gemini OpenAI-compatible endpoint.

    Resolution order for API key:
        1. Explicit ``api_key`` argument
        2. GEMINI_API_KEY environment variable
        3. GOOGLE_API_KEY environment variable

    Resolution order for model:
        1. Explicit ``model_name`` argument
        2. GEMINI_MODEL environment variable
        3. Default: ``gemini-2.0-flash``

    Returns
    -------
    client : openai.OpenAI | None
    model_name : str | None
    """
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package not installed – Gemini backend unavailable.")
        return _UNAVAILABLE, _UNAVAILABLE

    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    resolved_model = (
        model_name
        or os.environ.get("GEMINI_MODEL")
        or _GEMINI_DEFAULT_MODEL
    )

    if resolved_key is None:
        logger.warning(
            "No Gemini API key found (GEMINI_API_KEY / GOOGLE_API_KEY). "
            "Gemini backend unavailable."
        )
        return _UNAVAILABLE, _UNAVAILABLE

    client = OpenAI(api_key=resolved_key, base_url=_GEMINI_BASE_URL)
    logger.info("Gemini client initialised: model=%s", resolved_model)
    return client, resolved_model


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
