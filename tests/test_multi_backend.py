"""Tests for multi-backend provider configuration."""

from documind.config import (
    DEFAULT_PROVIDER,
    PROVIDER_PRESETS,
    DocuMindConfig,
)


class TestProviderPresets:
    def test_known_providers_present(self):
        for provider in (
            "mimo",
            "openai",
            "openrouter",
            "ollama",
            "groq",
            "deepseek",
            "together",
            "mistral",
        ):
            assert provider in PROVIDER_PRESETS

    def test_default_provider_is_mimo(self):
        assert DEFAULT_PROVIDER == "mimo"


class TestPresetResolution:
    def test_openai_preset(self, monkeypatch):
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        c = DocuMindConfig(provider="openai", mimo_api_key="sk-test")
        assert c.base_url == "https://api.openai.com/v1"
        assert c.model == "gpt-4o-mini"
        assert c.auth_style == "bearer"

    def test_openrouter_preset(self):
        c = DocuMindConfig(provider="openrouter", mimo_api_key="or-test")
        assert "openrouter.ai" in c.base_url
        assert c.auth_style == "bearer"

    def test_ollama_preset(self):
        c = DocuMindConfig(provider="ollama", mimo_api_key="x")
        assert "localhost:11434" in c.base_url
        assert c.model == "llama3.1"

    def test_groq_preset(self):
        c = DocuMindConfig(provider="groq", mimo_api_key="t")
        assert "api.groq.com" in c.base_url
        assert c.auth_style == "bearer"

    def test_deepseek_preset(self):
        c = DocuMindConfig(provider="deepseek", mimo_api_key="t")
        assert "api.deepseek.com" in c.base_url
        assert c.auth_style == "bearer"

    def test_together_preset(self):
        c = DocuMindConfig(provider="together", mimo_api_key="t")
        assert "api.together.xyz" in c.base_url
        assert c.auth_style == "bearer"

    def test_mistral_preset(self):
        c = DocuMindConfig(provider="mistral", mimo_api_key="t")
        assert "api.mistral.ai" in c.base_url
        assert c.auth_style == "bearer"

    def test_unknown_provider_falls_back_to_default(self):
        c = DocuMindConfig(provider="nope", mimo_api_key="x")
        assert "xiaomimimo.com" in c.base_url

    def test_explicit_values_override_preset(self):
        c = DocuMindConfig(
            provider="openai",
            mimo_api_key="x",
            mimo_base_url="https://proxy.local/v1",
            mimo_model="custom",
        )
        assert c.base_url == "https://proxy.local/v1"
        assert c.model == "custom"


class TestAuthStyles:
    def test_bearer_auth_header(self):
        c = DocuMindConfig(provider="openai", mimo_api_key="sk-abc")
        assert c.headers["Authorization"] == "Bearer sk-abc"
        assert "api-key" not in c.headers

    def test_api_key_auth_header(self):
        c = DocuMindConfig(provider="mimo", mimo_api_key="mimo-secret")
        assert c.headers["api-key"] == "mimo-secret"
        assert "Authorization" not in c.headers

    def test_explicit_auth_style_override(self):
        c = DocuMindConfig(provider="mimo", mimo_api_key="k", auth_style="bearer")
        assert c.headers["Authorization"] == "Bearer k"


class TestEnvResolution:
    def test_openai_env_key(self, monkeypatch):
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
        c = DocuMindConfig(provider="openai")
        assert c.mimo_api_key == "sk-from-env"

    def test_provider_from_env(self, monkeypatch):
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        monkeypatch.setenv("DOCUMIND_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-env")
        c = DocuMindConfig()
        assert c.provider == "openrouter"
        assert c.mimo_api_key == "or-env"


class TestMiMoBackwardCompat:
    def test_mimo_defaults(self, monkeypatch):
        monkeypatch.delenv("MIMO_API_KEY", raising=False)
        c = DocuMindConfig(mimo_api_key="test")
        assert c.provider == "mimo"
        assert c.mimo_model == "mimo-v2.5-pro"
        assert "xiaomimimo.com" in c.base_url
        assert c.headers["api-key"] == "test"

    def test_generic_aliases(self):
        c = DocuMindConfig(provider="openai", mimo_api_key="sk-x")
        assert c.api_key == "sk-x"
        assert c.api_key == c.mimo_api_key
