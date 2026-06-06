# 🧠 DocuMind

**7-Agent Document Q&A Pipeline for any OpenAI-compatible LLM**

> Multi-agent document analysis system — ingest, index, retrieve, answer, verify, cite, export.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![CI](https://github.com/aimanmalib/documind/actions/workflows/ci.yml/badge.svg)](https://github.com/aimanmalib/documind/actions/workflows/ci.yml)
[![Tests: 110](https://img.shields.io/badge/tests-110-brightgreen.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Works with **OpenAI, OpenRouter, Ollama, llama.cpp, Xiaomi MiMo**, or any endpoint that speaks the OpenAI `/chat/completions` protocol. Pick a provider with one config line — no code changes.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DocuMind Pipeline                         │
│                                                             │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐               │
│  │ Ingester │──▶│ Indexer  │──▶│ Retriever │               │
│  │ Agent    │   │ Agent    │   │ Agent     │               │
│  └──────────┘   └──────────┘   └─────┬─────┘               │
│       │                              │                      │
│       ▼                              ▼                      │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐              │
│  │Summarizer│   │FactCheck │   │ Answerer  │              │
│  │ Agent    │   │ Agent    │   │ Agent     │              │
│  └──────────┘   └──────────┘   └─────┬─────┘              │
│                                      │                      │
│       ┌──────────────────────────────┼──────────┐          │
│       ▼                              ▼          ▼          │
│  ┌──────────┐   ┌──────────┐   ┌───────────┐              │
│  │ Citation │   │  Export  │   │  Token    │              │
│  │ Agent    │   │  Agent   │   │  Tracker  │              │
│  └──────────┘   └──────────┘   └───────────┘              │
│                                                             │
│  ═══════════════════════════════════════════════             │
│  Any OpenAI-compatible /chat/completions endpoint          │
│  (OpenAI · OpenRouter · Ollama · MiMo · ...)               │
└─────────────────────────────────────────────────────────────┘
```

## Agents

| Agent | Role | Est. Tokens/Day |
|-------|------|-----------------|
| **Ingester** | Parse PDF/DOCX/TXT/MD, chunk text | 0.15M |
| **Indexer** | Build keyword + TF-IDF index | 0.20M |
| **Retriever** | Find relevant chunks + MiMo reranking | 0.45M |
| **Answerer** | Generate cited answers from context | 0.80M |
| **Summarizer** | Document-level summaries | 0.35M |
| **Fact Checker** | Verify claims against sources | 0.25M |
| **Citation** | APA/MLA/Chicago formatting | 0.10M |
| **Export** | Markdown/JSON export | 0.05M |

**Daily Total: ~2.4M tokens**

## Supported Providers

DocuMind talks to any OpenAI-compatible `/chat/completions` endpoint. Built-in presets:

| Provider | `provider=` | Default model | Auth | Env vars |
|----------|-------------|---------------|------|----------|
| OpenAI | `openai` | `gpt-4o-mini` | Bearer | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| OpenRouter | `openrouter` | `openai/gpt-4o-mini` | Bearer | `OPENROUTER_API_KEY` |
| Ollama (local) | `ollama` | `llama3.1` | Bearer | `OLLAMA_BASE_URL` |
| Groq | `groq` | `llama-3.3-70b-versatile` | Bearer | `GROQ_API_KEY`, `GROQ_BASE_URL` |
| DeepSeek | `deepseek` | `deepseek-chat` | Bearer | `DEEPSEEK_API_KEY`, `DEEPSEEK_BASE_URL` |
| Together | `together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | Bearer | `TOGETHER_API_KEY`, `TOGETHER_BASE_URL` |
| Mistral | `mistral` | `mistral-small-latest` | Bearer | `MISTRAL_API_KEY`, `MISTRAL_BASE_URL` |
| Xiaomi MiMo | `mimo` | `mimo-v2.5-pro` | api-key | `MIMO_API_KEY` |

Select a provider with the `DOCUMIND_PROVIDER` env var (or `DocuMindConfig(provider=...)`), or point `MIMO_BASE_URL` at any other compatible endpoint (llama.cpp, vLLM, LM Studio, a local proxy). The right auth header (bearer vs api-key) is chosen automatically per provider. The Fact Checker agent benefits from models that expose a `reasoning_content` field, but it isn't required.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Pick any provider — set its API key (OpenAI shown here)
export DOCUMIND_PROVIDER=openai
export OPENAI_API_KEY="sk-your-key-here"

# Ingest a document
documind ingest document.pdf

# Ask questions
documind ask document.pdf -q "What is the main conclusion?" -q "Who are the authors?"

# With verification
documind ask document.pdf -q "Summarize the methods" --verify

# View token stats
documind stats
```

## LLM Backend

- **Protocol**: OpenAI-compatible `/chat/completions`
- **Providers**: OpenAI, OpenRouter, Ollama, llama.cpp, Xiaomi MiMo, or any compatible endpoint
- **Auth**: bearer token or `api-key` header, selected automatically per provider
- **Config**: set `DOCUMIND_PROVIDER` + the provider's API key env var

## Tech Stack

- Python 3.10+
- Click + Rich (CLI)
- httpx (async HTTP)
- Pydantic (data validation)
- pytest + pytest-asyncio (testing)

## License

MIT
