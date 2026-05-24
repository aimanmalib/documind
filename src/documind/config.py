"""Configuration management for DocuMind."""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class DocuMindConfig:
    """Environment-based configuration for DocuMind pipeline."""

    mimo_api_key: str = field(default_factory=lambda: os.getenv("MIMO_API_KEY", ""))
    mimo_base_url: str = field(
        default_factory=lambda: os.getenv(
            "MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"
        )
    )
    mimo_model: str = field(default_factory=lambda: os.getenv("MIMO_MODEL", "mimo-v2.5-pro"))
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
