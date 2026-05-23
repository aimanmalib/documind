# 🧠 DocuMind

**Document Q&A Pipeline powered by Xiaomi MiMo V2.5 Pro**

> Multi-agent document analysis system — ingest, index, retrieve, answer, verify, cite, export.

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
│  Xiaomi MiMo V2.5 Pro (token-plan-sgp.xiaomimimo.com)      │
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

## Token Consumption Report

| Metric | Value |
|--------|-------|
| Daily token consumption | ~2.4M tokens |
| Per-session (avg 10 Q&A) | ~12K tokens |
| Primary model | mimo-v2.5-pro |
| API endpoint | `token-plan-sgp.xiaomimimo.com/v1` |
| Auth method | `api-key` header |

## Why MiMo?

1. **Cost efficiency** — MiMo V2.5 Pro delivers GPT-4-level reasoning at a fraction of the cost, critical for our high-volume document analysis pipeline
2. **Structured output** — Excellent at generating consistently formatted citations, fact-check verdicts, and summaries
3. **Reasoning depth** — The `reasoning_content` field provides transparent chain-of-thought, essential for the fact-checking agent
4. **Speed** — Low latency enables interactive Q&A sessions without frustrating wait times
5. **API compatibility** — OpenAI-compatible endpoint makes integration seamless

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Set API key
export MIMO_API_KEY="your-key-here"

# Ingest a document
documind ingest document.pdf

# Ask questions
documind ask document.pdf -q "What is the main conclusion?" -q "Who are the authors?"

# With verification
documind ask document.pdf -q "Summarize the methods" --verify

# View token stats
documind stats
```

## API Details

- **Endpoint**: `https://token-plan-sgp.xiaomimimo.com/v1/chat/completions`
- **Model**: `mimo-v2.5-pro`
- **Auth**: `api-key` header (NOT `Authorization: Bearer`)
- **Streaming**: SSE with `reasoning_content` field
- **Rate Limits**: RPM 100, TPM 10M

## Tech Stack

- Python 3.10+
- Click + Rich (CLI)
- httpx (async HTTP)
- Pydantic (data validation)
- pytest + pytest-asyncio (testing)

## License

MIT
