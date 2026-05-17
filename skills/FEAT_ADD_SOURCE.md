# Skill: Adding a New Data Source

## Goal
Implement a new content scanner and integrate it into the morning briefing pipeline.

## Implementation Steps
1. **Create Scanner:** Create `scripts/[source]_scanner.py`. 
   - Implement a class/method that fetches and returns a list of dictionaries.
   - Standard keys: `title`, `url`, `description`, `date`, `source`.
2. **Update Config:** 
   - Add new section to `config.yaml` (e.g., `reddit_sources:`).
   - Update `scripts/config_validator.py` if new validation logic is needed.
3. **Register in Runner:** 
   - Update `BriefingRunner.run()` in `scripts/briefing_runner.py`.
   - Call the new scanner and add results to the data collection phase.
4. **Update Reporting:**
   - Ensure the `status` dictionary tracks the number of items found by the new source.
5. **Update Briefing Template:**
   - Modify `generate_markdown_briefing` in `scripts/briefing_runner.py` to include the new section in the final output.

## Validation
- Run with `--dry-run` to see the new content in the generated Markdown.
- Add a unit test in `tests/test_[source]_scanner.py`.
