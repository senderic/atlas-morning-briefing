# Migration Plan: Gemini CLI to Antigravity CLI (`agy`)

This document outlines the necessary steps to migrate the **Atlas Morning Briefing** from the deprecated `gemini-cli` to the new Go-based **Antigravity CLI** (`agy`) before the June 18, 2026 deadline.

## 1. Prerequisites & Installation

The Antigravity CLI is a Go-based binary, replacing the Node.js implementation.

### Installation Steps
Run the following command to install the binary. Since your `.nvm` is symlinked to `/mnt/fast_scratch`, we will place the binary in your existing local bin path to maintain consistency.

```bash
# Download and install the agy binary
curl -sSL https://antigravity.google/install | bash

# Move it to your bin directory (adjust if your path differs)
mv ~/bin/agy /home/eric/.nvm/versions/node/v20.19.5/bin/
```

### Initial Setup
Run the migration tool to import your existing Gemini settings, keys, and extensions:
```bash
agy plugin import gemini
```

## 2. Configuration Changes

### File Renaming
Antigravity uses new naming conventions for workspace context:
- **Rename** `GEMINI.md` to `.antigravity.md` (or keep both, as `agy` respects `GEMINI.md` for backward compatibility, but `.antigravity.md` is preferred).

### Skill Relocation
If you have custom skills in `.gemini/skills/`, they should be moved:
```bash
mkdir -p .agents/skills
cp -r .gemini/skills/* .agents/skills/
```

## 3. Code Changes

### `scripts/gemini_client.py`
The core logic in `GeminiCLIClient` needs to be updated to use `agy`.

**Key Flag Mappings:**
- `gemini` -> `agy`
- `--prompt` -> Passed as a positional argument.
- `--raw-output` -> `--quiet` (to suppress TUI headers).
- `--output-format json` -> `--output=json`.
- `--approval-mode yolo` -> `--dangerously-skip-permissions`.

**Proposed Change in `_execute_command`:**
```python
# OLD
cmd = [
    "gemini", "--model", model_id, "--prompt", prompt,
    "--approval-mode", "yolo", "--raw-output", "--accept-raw-output-risk",
    "--output-format", "json"
]

# NEW
cmd = [
    "agy", prompt, "--print", "--quiet",
    "--output=json", "--dangerously-skip-permissions"
]
```

## 4. Key Usage & Authentication

Antigravity continues to support `GEMINI_API_KEY` for headless execution.

- **Environment Variable:** `GEMINI_API_KEY` is still the primary method.
- **Key Rotation:** The current rotation logic in `gemini_client.py` will continue to work as it injects the key into the environment before calling the subprocess.

## 5. Timeline
- **June 18, 2026:** Gemini CLI will stop serving requests for individual-tier accounts.

---
*Created on: May 20, 2026*
