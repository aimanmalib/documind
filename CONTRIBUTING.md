# Contributing to DocuMind

Thanks for your interest in improving DocuMind. This project is a
provider-agnostic, 7-agent document Q&A pipeline that runs on any
OpenAI-compatible LLM endpoint. Contributions of all sizes are welcome — docs
fixes, new provider presets, agent improvements, and bug reports all help.

## Getting started

```bash
git clone https://github.com/aimanmalib/documind.git
cd documind
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest          # 110 tests should pass
```

## Development workflow

1. Fork the repo and create a feature branch: `git checkout -b feat/my-change`
2. Make your change with tests.
3. Run the full local gate before pushing:
   ```bash
   ruff check src/ tests/          # lint
   ruff format src/ tests/         # format
   pytest                          # tests
   ```
4. Commit using [Conventional Commits](https://www.conventionalcommits.org/)
   (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `ci:`).
5. Open a pull request against `main`. CI runs lint + format + tests on
   Python 3.10/3.11/3.12 — keep it green.

## Adding a new LLM provider

DocuMind talks to any OpenAI-compatible `/chat/completions` endpoint, so most
providers need only a preset entry. In `src/documind/config.py`, add to
`PROVIDER_PRESETS`:

```python
"myprovider": {
    "base_url": "https://api.myprovider.com/v1",
    "auth_style": "bearer",          # "bearer" or "api-key"
    "model": "default-model-name",
    "env_key": "MYPROVIDER_API_KEY",
    "env_base": "MYPROVIDER_BASE_URL",
},
```

Then add a test in `tests/test_multi_backend.py` mirroring the existing
provider cases. No client changes are needed unless the provider deviates from
the OpenAI protocol.

## Good first issues

- Add a provider preset (Together, Groq, DeepSeek, Mistral, ...)
- Add a document loader (HTML, EPUB, CSV)
- Add a citation format (IEEE, Harvard)
- Improve an agent's prompt

## Code style

- Python 3.10+ with type hints
- `ruff` for linting and formatting (config in `pyproject.toml`)
- Public APIs get docstrings

## Reporting bugs / requesting features

Use the issue templates (bug report / feature request). Include repro steps,
your provider/model, and the DocuMind version for bugs.

## License

By contributing, you agree your contributions are licensed under the MIT
License, the same as the project.
