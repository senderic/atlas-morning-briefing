# Migration Plan: Gemini CLI to Antigravity CLI (`agy`)

This document outlines the necessary steps to migrate the **Atlas Morning Briefing** from the deprecated `gemini-cli` to the new Go-based **Antigravity CLI** (`agy`) before the June 18, 2026 deadline.

## 1. Prerequisites & Installation

The Antigravity CLI is a Go-based binary, replacing the Node.js implementation.

### Installation Steps
Run the following command to install the binary. Since your fast scratch storage is at `/mnt/fast_scratch`, we will place the binary there for faster loading and access.

```bash
# Download and install the agy binary
curl -sSL https://antigravity.google/install | bash

# Move it to your fast_scratch bin directory for faster loading
mkdir -p /mnt/fast_scratch/bin
mv ~/bin/agy /mnt/fast_scratch/bin/

# Add to PATH (ensure this is in your .bashrc or .zshrc)
export PATH="/mnt/fast_scratch/bin:$PATH"
```

### Initial Setup
Verify installation and test the CLI:
```bash
agy --version
agy --help
```

**Note:** Do NOT run `agy plugin import gemini` unless the migration docs explicitly recommend it. Test basic functionality first.

## 2. Configuration Changes

### File Renaming (Optional)
Antigravity uses new naming conventions for workspace context:
- Keep `GEMINI.md` as-is for now (backward compatible)
- Optionally create `.antigravity.md` for `agy`-specific configuration

**Action:** Leave `GEMINI.md` unchanged until you verify `agy` behavior.

### Skill Relocation (Deferred)
Current repo does not have custom skills in `.gemini/skills/`. This step can be skipped unless you add custom skills later.

## 3. Code Changes

### `scripts/gemini_client.py` - Authentication & Environment Variables

**⚠️ CRITICAL:** Before updating code, verify the environment variable `agy` uses for API keys.

**Current Status:** `gemini-cli` uses `GEMINI_API_KEY`. This needs verification for `agy`.

Test with:
```bash
GEMINI_API_KEY="test-key" agy "test prompt" --help 2>&1 | grep -i "key\|auth\|environment"
agy --help | grep -i "environment\|key\|auth"
```

**Once verified**, update the following in `gemini_client.py`:

1. **Update the `available` property** (line 153):
   ```python
   # OLD
   subprocess.run(["which", "gemini"], capture_output=True, check=True)
   
   # NEW
   subprocess.run(["which", "agy"], capture_output=True, check=True)
   ```

2. **Update the warning message** (line 156):
   ```python
   # OLD
   logger.warning("gemini-cli not found in PATH. Gemini features disabled.")
   
   # NEW
   logger.warning("agy (Antigravity CLI) not found in PATH. Gemini features disabled.")
   ```

3. **Update `_execute_command` method** (lines 161-273):

   **Key Flag Mappings** (must be verified with `agy --help`):
   - `gemini` → `agy`
   - `--prompt <text>` → `<text>` (positional argument)
   - `--model <model_id>` → `--model <model_id>` (verify this exists in `agy`)
   - `--output-format json` → `--output=json` (verify exact syntax)
   - `--approval-mode yolo` → `--dangerously-skip-permissions` (verify)
   - `--raw-output --accept-raw-output-risk` → `--quiet` (verify effect)

   **Proposed Updated Command**:
   ```python
   def _execute_command(self, model_id: str, prompt: str, tier: str) -> str:
       """Execute the agy command."""
       import tempfile
       import shutil
       from pathlib import Path

       tmp_config_dir = tempfile.mkdtemp(prefix="atlas_agy_config_")
       
       try:
           # Create config directory if agy uses one similar to gemini-cli
           agy_dir = Path(tmp_config_dir) / ".agy"
           agy_dir.mkdir(parents=True, exist_ok=True)
           settings_path = agy_dir / "settings.json"
           
           # NOTE: Verify if agy respects these config settings
           with open(settings_path, "w") as f:
               json.dump({
                   "general": {"maxAttempts": self.internal_max_attempts, "requestTimeout": 120000},
               }, f)

           process_env = os.environ.copy()
           process_env["AGY_CONFIG_DIR"] = tmp_config_dir  # VERIFY correct env var name
           
           # Set API key (VERIFY which env var agy uses)
           if tier == "heavy":
               api_key = self._get_current_key()
               key_index = self._current_key_index
           else:
               api_key = self._api_keys[0] if self._api_keys else None
               key_index = 0

           if api_key:
               # VERIFY: Does agy use GEMINI_API_KEY or a different variable?
               process_env["GEMINI_API_KEY"] = api_key
               key_preview = api_key[:6] + "..." + api_key[-4:]
               logger.debug(f"Using API Key index {key_index} for tier {tier}: {key_preview}")
           else:
               logger.warning(f"No API key available for tier {tier}!")

           # Add small delay for heavy tier
           if tier == "heavy":
               time.sleep(1)
           
           logger.info(f"Invoking agy model: {model_id} (tier: {tier})")

           # VERIFY all flags below with `agy --help`
           cmd = [
               "agy", prompt, 
               "--model", model_id,
               "--output=json",
               "--quiet",
               "--dangerously-skip-permissions"
           ]

           try:
               result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=900, env=process_env)
           except Exception as e:
               self.usage_stats[tier]["failed_attempts"] += 1
               raise e
           
           try:
               # Parse JSON output from agy
               data = json.loads(result.stdout)
               output = data.get("response", "").strip()
               
               # Extract stats if available
               stats = data.get("stats", {}).get("models", {})
               model_stats = {}
               for k, v in stats.items():
                   if model_id in k or k in model_id:
                       model_stats = v.get("tokens", {})
                       break
               if not model_stats and stats:
                   model_stats = next(iter(stats.values())).get("tokens", {})

               # Update usage metrics
               self.usage_stats[tier]["calls"] += 1
               in_tokens = model_stats.get("input", 0) or model_stats.get("prompt", 0)
               out_tokens = model_stats.get("candidates", 0)
               
               self.usage_stats[tier]["in_tokens"] += in_tokens
               self.usage_stats[tier]["out_tokens"] += out_tokens
               
               self.usage_stats[tier]["in_chars"] += len(prompt)
               self.usage_stats[tier]["out_chars"] += len(output)

               if not output:
                   raise ValueError(f"Empty response from {tier}")
                   
               logger.info(f"agy response received ({len(output)} chars, {out_tokens} tokens) from {tier}")
               return output

           except (json.JSONDecodeError, KeyError) as e:
               logger.warning(f"Failed to parse JSON response from agy: {e}")
               output = result.stdout.strip()
               self.usage_stats[tier]["calls"] += 1
               self.usage_stats[tier]["in_chars"] += len(prompt)
               self.usage_stats[tier]["out_chars"] += len(output)
               return output
       
       finally:
           shutil.rmtree(tmp_config_dir, ignore_errors=True)
   ```

## 4. Key Usage & Authentication

**Environment Variable:** Determine which variable `agy` uses:
- Primary candidate: `GEMINI_API_KEY` (same as gemini-cli)
- Alternate: `AGY_API_KEY` or `ANTIGRAVITY_API_KEY`

**Testing:**
```bash
# Test 1: Basic auth check
agy --help 2>&1 | grep -iE "environment|GEMINI_API_KEY|api.?key"

# Test 2: Verify with dummy key
GEMINI_API_KEY="dummy" agy "what is 2+2?" 2>&1

# Test 3: Check agy config directory (if it has one)
ls -la ~/.agy/ 2>/dev/null || echo "No ~/.agy directory"
```

**Key Rotation:** The existing rotation logic in `gemini_client.py` (lines 121-140) will continue to work once you verify the environment variable name.

## 5. Pre-Migration Testing Checklist

Before updating `gemini_client.py`, complete these steps:

- [ ] Install `agy` binary to `/mnt/fast_scratch/bin/`
- [ ] Verify `agy --version` works
- [ ] Run `agy "test prompt"` to confirm basic functionality
- [ ] Document the exact command-line flags (run `agy --help`)
- [ ] Determine which environment variable `agy` uses for API keys
- [ ] Test API key injection: `GEMINI_API_KEY="key" agy "prompt"` or `AGY_API_KEY="key" agy "prompt"`
- [ ] Verify JSON output format with `agy "prompt" --output=json`
- [ ] Confirm model selection syntax (e.g., `--model pro`)
- [ ] Test quiet/no-header output flag (verify `--quiet` exists)
- [ ] Create a test branch and update only the `_execute_command` method
- [ ] Run existing tests to ensure parsing logic still works

## 6. Timeline

- **NOW:** Install `agy` and run verification tests
- **This week:** Update `gemini_client.py` with verified flag names
- **Before June 18, 2026:** Complete full migration and remove `gemini-cli` dependency

---

*Updated: May 21, 2026*
*Status: AWAITING VERIFICATION - Do not commit code changes until testing complete*
