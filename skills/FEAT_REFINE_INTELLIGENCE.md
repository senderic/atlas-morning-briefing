# Skill: Refining Intelligence Logic

## Goal
Modify AI prompts, synthesis behavior, or scoring logic in the intelligence layer.

## Implementation Steps
1. **Locate Logic:** Open `scripts/intelligence.py`.
2. **Modify Prompts:** 
   - Global prompts are at the top (e.g., `SYSTEM_PROMPT`).
   - Task-specific prompts are inside methods (e.g., `generate_editorial_intro`).
3. **Handle Parsing:** 
   - If changing prompt structure, update corresponding parsers like `_parse_ranked_response` or `extract_score`.
4. **Scoring Logic:** 
   - Update `paper_scorer.py` if changing how papers are ranked.
   - Update `BriefingIntelligence.score_papers` for semantic ranking changes.

## Best Practices
- **Token Efficiency:** Be concise in prompts to stay within free tier limits.
- **Robustness:** Use `_sanitize_prompt_input` for any external data included in prompts.
- **Fallback:** Ensure the method returns a sensible default if the LLM call fails.

## Validation
- Run `tests/test_intelligence.py` to verify parsing logic.
- Execute a sample run and inspect the `DEBUG` logs to see raw LLM interactions.
