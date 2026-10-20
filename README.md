> [!IMPORTANT]
> **documind has moved into [crewforge](https://github.com/aimanmalib/crewforge).** It now lives on as `crewforge ingest`, `crewforge ask` and `crewforge summarize`, on a shared core
> that fails loudly instead of reporting success on empty output. This repository is archived and kept for history;
> new issues and changes go to crewforge.

# DocuMind

**Multi-agent document Q&A for any OpenAI-compatible LLM**

> Ingest, index, retrieve, answer, verify, cite, export: a small pipeline of focused agents that answers questions about your documents with source citations.

[![CI](https://github.com/aimanmalib/documind/actions/workflows/ci.yml/badge.svg)](https://github.com/aimanmalib/documind/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Works with **OpenAI, OpenRouter, Ollama, Groq, DeepSeek, Together, Mistral, Xiaomi MiMo**, or any server that speaks the OpenAI `/chat/completions` protocol (llama.cpp, vLLM, LM Studio, a local gateway). Pick a provider with environment variables, no code changes.

---

## Quick Start

```bash
git clone https://github.com/aimanmalib/documind.git
cd documind
pip install -e .            # add ".[pdf,docx]" for PDF / Word extraction

export OPENAI_API_KEY="sk-..."   # OpenAI is the default provider

# Check how a document is chunked (no LLM calls)
documind ingest report.md

# Ask questions; answers cite [Source N] chunks
documind ask report.md -q "What is the main conclusion?" -q "Who are the authors?"

# Fact-check each answer against the sources and save the session
documind ask report.md -q "Summarize the methods" --verify --export qa.md

# Summarize one or more documents
documind summarize report.md
```

`documind --help` and `documind <command> --help` list every option. `ask` and `summarize` end with a table of the token usage reported by the LLM for that run.

## Choosing a provider

| Provider | `DOCUMIND_PROVIDER` | Default model | API key env var | Base URL env var |
|----------|---------------------|---------------|-----------------|------------------|
| OpenAI (default) | `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` | `OPENAI_BASE_URL` |
| OpenRouter | `openrouter` | `openai/gpt-4o-mini` | `OPENROUTER_API_KEY` | `OPENROUTER_BASE_URL` |
| Ollama (local) | `ollama` | `llama3.1` | not required | `OLLAMA_BASE_URL` |
| Groq | `groq` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` | `GROQ_BASE_URL` |
| DeepSeek | `deepseek` | `deepseek-chat` | `DEEPSEEK_API_KEY` | `DEEPSEEK_BASE_URL` |
| Together | `together` | `meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` | `TOGETHER_BASE_URL` |
| Mistral | `mistral` | `mistral-small-latest` | `MISTRAL_API_KEY` | `MISTRAL_BASE_URL` |
| Xiaomi MiMo | `mimo` | `mimo-v2.5-pro` | `MIMO_API_KEY` | `MIMO_BASE_URL` |

All presets use `Authorization: Bearer` except MiMo, which uses an `api-key` header; the right one is picked automatically.

```bash
# OpenRouter
export DOCUMIND_PROVIDER=openrouter
export OPENROUTER_API_KEY="sk-or-..."

# Any other OpenAI-compatible endpoint (overrides the preset)
export DOCUMIND_BASE_URL="http://localhost:8000/v1"
export DOCUMIND_API_KEY="anything-non-empty"
export DOCUMIND_MODEL="my-model"
```

`DOCUMIND_MODEL` also works on its own to pick a different model from a preset provider.

Other settings: `DOCUMIND_CHUNK_SIZE` (512), `DOCUMIND_CHUNK_OVERLAP` (64), `DOCUMIND_RETRIEVAL_K` (5), `DOCUMIND_CITATION_FORMAT` (`apa` | `mla` | `chicago`), `DOCUMIND_OUTPUT_DIR` (`./output`).

## How it works

```
 document ─▶ Ingester ─▶ Indexer ─▶ Retriever ─▶ Answerer ─▶ Fact Checker ─▶ Export
             parse +     keyword +   top-k chunks  cited       (--verify)      Markdown /
             chunk       TF-IDF      (+ LLM rerank)  answer                     JSON
                                                         Summarizer · Citation
```

| Agent | Role | Uses the LLM |
|-------|------|:---:|
| **Ingester** | Read TXT / MD / RST / HTML / JSON / CSV (PDF and DOCX with the optional extras) and split into overlapping chunks | no |
| **Indexer** | Build a keyword + TF-IDF index over the chunks | no |
| **Retriever** | Return the top-k chunks for a question, optionally reranked by the LLM (`--rerank`) | optional |
| **Answerer** | Answer strictly from the retrieved chunks with `[Source N]` citations | yes |
| **Fact Checker** | Rate each claim in an answer as verified / unverified / contradicted (`--verify`) | yes |
| **Summarizer** | Document-level summary and key points | yes |
| **Citation** | Format references in APA / MLA / Chicago | yes |
| **Export** | Write the Q&A session to Markdown or JSON | no |

Retrieval is lexical (TF-IDF), not embedding-based, so no embedding model or vector database is needed. That keeps the tool small and provider-agnostic, but it works best when questions share vocabulary with the document.

## Use as a library

```python
import asyncio
from documind.pipeline import DocuMindPipeline

async def main():
    pipeline = DocuMindPipeline()          # reads the same env vars as the CLI
    await pipeline.ingest_documents(["report.md"])
    answer = await pipeline.ask("What is the main conclusion?")
    print(answer.answer_text)
    print(pipeline.tracker.by_agent())     # token usage per agent
    await pipeline.client.close()

asyncio.run(main())
```

## Development

```bash
pip install -e ".[dev]"
ruff check src tests && ruff format --check src tests
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Security issues: see [SECURITY.md](SECURITY.md).

## License

[MIT](LICENSE)

---

Built by [Sasa Teguh](https://sasateguh.com) · Malaysia
