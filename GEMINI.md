# Atlas Morning Briefing: Agent Mandates

## Core Principles
- **Intelligence Priority:** Always prefer `gemini-cli` for synthesis and summarization. Use `GeminiCLIClient` in `scripts/gemini_client.py`.
- **Graceful Fallback:** Ensure the pipeline remains functional even if AI services are unavailable. Features should have deterministic fallbacks.
- **Safety First:** Support `--dry-run` in `briefing_runner.py`. Do not send emails or update state files during dry runs.
- **Observability:** Every run must update `status.json`. Scanners must report counts (found/processed) and log errors to the `errors` list.

## Project Structure
- `scripts/briefing_runner.py`: The orchestrator. Entry point for the pipeline.
- `scripts/intelligence.py`: The AI logic layer. Contains all prompts and synthesis functions.
- `scripts/*_scanner.py`: Individual data collectors (Arxiv, Blog, News, Stocks).
- `config.yaml`: Central configuration. Use `scripts/config_validator.py` for schema enforcement.

## Workflow Rules
- **Adding Features:** Follow the guides in the `skills/` directory.
- **Testing:** Add new test files to `tests/` for any new logic. Run `pytest` before proposing changes.
- **Git History:** Follow the existing convention of descriptive, prefix-based commit messages (e.g., `feat:`, `fix:`, `refactor:`).
