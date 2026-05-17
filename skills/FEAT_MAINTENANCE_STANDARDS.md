# Skill: Maintenance & Observability Standards

## Goal
Ensure every pipeline run is traceable, debuggable, and provides a clear audit trail of its success or failure.

## Observability Patterns

### 1. Status Reporting (`status.json`)
Every run must overwrite `status.json` with a structured summary.
- **Fields:** `timestamp`, `papers_found`, `blogs_found`, `stocks_fetched`, `intelligence_enabled`, `errors`, `elapsed_seconds`.
- **Requirement:** If a scanner fails, it MUST append the error message to the `errors` list rather than crashing the orchestrator.

### 2. Structured Logging
- **Standard:** Use the `logger` instance.
- **Levels:** 
  - `DEBUG`: Raw API responses, prompt text, detailed timing.
  - `INFO`: Scanner start/stop, model invocation, key rotation events.
  - `WARNING`: Partial failures (e.g., one RSS feed down), retry events.
  - `ERROR`: Critical failures (e.g., all models failed, config invalid).
- **Log Format:** `%(levelname)s: %(message)s` (as established in `briefing_runner.py`).

### 3. Cost & Performance Tracking
- **Duration:** Track total runtime and report it in the `status.json` and report footer.
- **Token Usage:** Track input/output tokens per tier.
- **Cost Estimation:** Use `get_usage_summary` in `gemini_client.py` to calculate costs based on Pay-as-you-go pricing.

### 4. Environment Robustness
- **Environment Variables:** Always use `python-dotenv` and support both `.env` files and system environment variables.
- **Path Portability:** Use absolute paths resolved via `pathlib` or `os.path.abspath` based on the script's location (see `run_briefing.sh` logic).

## Testing Standards
- **Mocking:** Always mock external API calls in tests (see `tests/test_arxiv_scanner.py` for examples).
- **Integration Tests:** Use `test_minimal.yaml` for fast integration checks of the full pipeline.
- **Regression Testing:** Before committing changes to `intelligence.py`, run `pytest tests/test_intelligence.py` to ensure parsing logic remains intact.
