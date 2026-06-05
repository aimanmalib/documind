"""Configuration management for DocuMind.

DocuMind speaks the OpenAI-compatible ``/chat/completions`` protocol, so it
works with any provider exposing that API: OpenAI, OpenRouter, Ollama, local
llama.cpp servers, Xiaomi MiMo Token Plan, and more. Select a provider via
``provider=`` (or the ``DOCUMIND_PROVIDER`` env var) or point ``mimo_base_url``
at any compatible endpoint.

The historical ``mimo_*`` fields remain the canonical config attributes for
backward compatibility; they simply hold whichever provider's key/url/model is
active.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Provider presets: base_url, auth header style, default model, and the env
# vars used to populate the key / base_url when they aren't set explicitly.
# auth_style is "bearer" (Authorization: Bearer) or "api-key" (api-key header).
PROVIDER_PRESETS: dict[str, dict[str, str]] = {
    "mimo": {
        "base_url": "https://token-plan-sgp.xiaomimimo.com/v1",
        "auth_style": "api-key",
        "model": "mimo-v2.5-pro",
        "env_key": "MIMO_API_KEY",
        "env_base": "MIMO_BASE_URL",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "auth_style": "bearer",
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "env_base": "OPENAI_BASE_URL",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "auth_style": "bearer",
        "model": "openai/gpt-4o-mini",
        "env_key": "OPENROUTER_API_KEY",
        "env_base": "OPENROUTER_BASE_URL",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "auth_style": "bearer",
        "model": "llama3.1",
        "env_key": "OLLAMA_API_KEY",
        "env_base": "OLLAMA_BASE_URL",
    },
}

DEFAULT_PROVIDER = "mimo"


@dataclass
class DocuMindConfig:
    """Environment-based configuration for DocuMind pipeline."""

    provider: str = field(
        default_factory=lambda: os.getenv("DOCUMIND_PROVIDER", DEFAULT_PROVIDER)
    )
    mimo_api_key: str = field(default_factory=lambda: os.getenv("MIMO_API_KEY", ""))
    mimo_base_url: str = field(default_factory=lambda: os.getenv("MIMO_BASE_URL", ""))
    mimo_model: str = field(default_factory=lambda: os.getenv("MIMO_MODEL", ""))
    auth_style: str = ""  # "bearer" | "api-key" — resolved from provider if blank
    max_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("DOCUMIND_CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("DOCUMIND_CHUNK_OVERLAP", "64"))
    )
    max_retrieval_k: int = field(
        default_factory=lambda: int(os.getenv("DOCUMIND_RETRIEVAL_K", "5"))
    )
    citation_format: str = field(
        default_factory=lambda: os.getenv("DOCUMIND_CITATION_FORMAT", "apa")
    )
    output_dir: str = field(
        default_factory=lambda: os.getenv("DOCUMIND_OUTPUT_DIR", "./output")
    )

    def __post_init__(self) -> None:
        preset = PROVIDER_PRESETS.get(self.provider, PROVIDER_PRESETS[DEFAULT_PROVIDER])
        if not self.mimo_api_key:
            self.mimo_api_key = os.getenv(preset["env_key"], "")
        if not self.mimo_base_url:
            self.mimo_base_url = os.getenv(preset["env_base"], preset["base_url"])
        if not self.mimo_model:
            self.mimo_model = preset["model"]
        if not self.auth_style:
            self.auth_style = preset["auth_style"]

    # Generic, provider-neutral aliases for new code.
    @property
    def api_key(self) -> str:
        return self.mimo_api_key

    @property
    def base_url(self) -> str:
        return self.mimo_base_url

    @property
    def model(self) -> str:
        return self.mimo_model

    @property
    def headers(self) -> dict[str, str]:
        """Auth + content headers, matching the provider's expected auth style."""
        headers = {"Content-Type": "application/json"}
        if self.auth_style == "bearer":
            headers["Authorization"] = f"Bearer {self.mimo_api_key}"
        else:
            headers["api-key"] = self.mimo_api_key
        return headers

    def validate(self) -> list[str]:
        """Return list of configuration errors."""
        errors = []
        if not self.mimo_api_key:
            errors.append("MIMO_API_KEY is not set")
        if self.max_chunk_size < 64:
            errors.append("max_chunk_size must be >= 64")
        if self.chunk_overlap >= self.max_chunk_size:
            errors.append("chunk_overlap must be < max_chunk_size")
        if self.max_retrieval_k < 1:
            errors.append("max_retrieval_k must be >= 1")
        if self.citation_format not in ("apa", "mla", "chicago"):
            errors.append(f"Unknown citation format: {self.citation_format}")
        return errors

    @classmethod
    def from_env(cls) -> "DocuMindConfig":
        """Create config from environment variables."""
        cfg = cls()
        errors = cfg.validate()
        if errors:
            import warnings

            for e in errors:
                warnings.warn(f"Config warning: {e}")
        return cfg
