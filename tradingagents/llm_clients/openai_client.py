import os
from contextlib import contextmanager
from typing import Any, Optional
from urllib.parse import urlparse

import httpx
from openai import APIStatusError
from langchain_openai import ChatOpenAI

from .base_client import BaseLLMClient, normalize_content
from .validators import validate_model


class NormalizedChatOpenAI(ChatOpenAI):
    """ChatOpenAI with normalized content output.

    The Responses API returns content as a list of typed blocks
    (reasoning, text, etc.). This normalizes to string for consistent
    downstream handling.
    """

    def invoke(self, input, config=None, **kwargs):
        try:
            return normalize_content(super().invoke(input, config, **kwargs))
        except APIStatusError as exc:
            raise RuntimeError(_format_openai_status_error(self, exc)) from exc


def _format_openai_status_error(llm: ChatOpenAI, exc: APIStatusError) -> str:
    """Format provider-side API errors without exposing credentials."""
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", "unknown")
    body = None

    if response is not None:
        try:
            body = response.text
        except Exception:
            body = None

    details = [
        "OpenAI-compatible provider rejected the request.",
        f"status_code={status_code}",
        f"model={getattr(llm, 'model_name', 'unknown')}",
        f"base_url={getattr(llm, 'openai_api_base', 'unknown')}",
        f"message={exc}",
    ]
    if body:
        details.append(f"response_body={body[:500]}")

    return " ".join(details)

# Kwargs forwarded from user config to ChatOpenAI
_PASSTHROUGH_KWARGS = (
    "timeout", "max_retries", "reasoning_effort",
    "api_key", "callbacks", "http_client", "http_async_client",
)

# Provider base URLs and API key env vars
_PROVIDER_CONFIG = {
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
}

_CUSTOM_OPENAI_ENV_BASE_URL = "CUSTOM_OPENAI_BASE_URL"
_CUSTOM_OPENAI_ENV_API_KEY = "CUSTOM_OPENAI_API_KEY"

_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)
_SUPPORTED_PROXY_SCHEMES = {"http", "https"}


def _has_unsupported_proxy_scheme(proxy_url: str) -> bool:
    """Return True for proxy schemes unsupported by the installed httpx stack."""
    scheme = urlparse(proxy_url).scheme.lower()
    return bool(scheme and scheme not in _SUPPORTED_PROXY_SCHEMES)


@contextmanager
def _without_unsupported_proxy_env():
    """Temporarily remove proxy env vars that break OpenAI/httpx initialization."""
    removed = {}
    for name in _PROXY_ENV_VARS:
        value = os.environ.get(name)
        if value and _has_unsupported_proxy_scheme(value):
            removed[name] = value
            os.environ.pop(name, None)

    try:
        yield
    finally:
        os.environ.update(removed)


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI-compatible providers.

    For native OpenAI models, uses the Responses API (/v1/responses) which
    supports reasoning_effort with function tools across all model families
    (GPT-4.1, GPT-5). Third-party compatible providers (xAI, OpenRouter,
    Ollama, custom relays) use standard Chat Completions.
    """

    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        provider: str = "openai",
        **kwargs,
    ):
        super().__init__(model, base_url, **kwargs)
        self.provider = provider.lower()

    def get_llm(self) -> Any:
        """Return configured ChatOpenAI instance."""
        self.warn_if_unknown_model()
        llm_kwargs = {"model": self.model}

        # Provider-specific base URL and auth
        if self.provider == "custom":
            custom_base_url = self.base_url or os.environ.get(
                _CUSTOM_OPENAI_ENV_BASE_URL
            )
            custom_api_key = self.kwargs.get("api_key") or os.environ.get(
                _CUSTOM_OPENAI_ENV_API_KEY
            )

            if not custom_base_url:
                raise ValueError(
                    "Custom OpenAI-compatible provider requires backend_url "
                    f"or {_CUSTOM_OPENAI_ENV_BASE_URL}."
                )
            if not custom_api_key:
                raise ValueError(
                    "Custom OpenAI-compatible provider requires llm_api_key "
                    f"or {_CUSTOM_OPENAI_ENV_API_KEY}."
                )

            llm_kwargs["base_url"] = custom_base_url
            llm_kwargs["api_key"] = custom_api_key
        elif self.provider in _PROVIDER_CONFIG:
            base_url, api_key_env = _PROVIDER_CONFIG[self.provider]
            llm_kwargs["base_url"] = base_url
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if api_key:
                    llm_kwargs["api_key"] = api_key
            else:
                llm_kwargs["api_key"] = "ollama"
        elif self.base_url:
            llm_kwargs["base_url"] = self.base_url

        trust_env = self.kwargs.get("trust_env")
        if (
            trust_env is False
            and "http_client" not in self.kwargs
            and "http_async_client" not in self.kwargs
        ):
            llm_kwargs["http_client"] = httpx.Client(trust_env=False)
            llm_kwargs["http_async_client"] = httpx.AsyncClient(trust_env=False)

        # Forward user-provided kwargs
        for key in _PASSTHROUGH_KWARGS:
            if self.provider == "custom" and key == "api_key":
                continue
            if key in self.kwargs:
                llm_kwargs[key] = self.kwargs[key]

        # Native OpenAI: use Responses API for consistent behavior across
        # all model families. Third-party providers use Chat Completions.
        if self.provider == "openai":
            llm_kwargs["use_responses_api"] = True

        with _without_unsupported_proxy_env():
            return NormalizedChatOpenAI(**llm_kwargs)

    def validate_model(self) -> bool:
        """Validate model for the provider."""
        return validate_model(self.provider, self.model)
