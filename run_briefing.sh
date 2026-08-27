#!/bin/bash
# Atlas Morning Briefing Runner Wrapper Script
# Runs main briefing first, then local (San Diego/CA) — sequential to avoid
# simultaneous API hits against Brave + Gemini from the same IP.

# Resolve script directory so relative paths work correctly from cron
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PATH to include gemini-cli, opencode (linuxbrew), and other necessary binaries
export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:$HOME/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Navigate to project directory
cd "$DIR" || exit 1

# Pre-flight model availability check (5:45 AM recommended, runs inline here)
# Tests all free models concurrently and writes .model-availability.json
"$DIR/.venv/bin/python3" "$DIR/scripts/preflight_model_check.py" 2>&1 | logger -t preflight-check
RC_PREFLIGHT=$?
if [ $RC_PREFLIGHT -ne 0 ]; then
    logger -t preflight-check "Pre-flight check failed, continuing with config defaults"
fi

# Main briefing (defense/tech) runs first
"$DIR/.venv/bin/python3" "$DIR/scripts/briefing_runner.py" --config "$DIR/config.yaml" --log-level DEBUG "$@" 2>&1 | logger -t atlas-briefing
RC_MAIN=$?

# Local briefing (San Diego / CA) runs after main completes
"$DIR/.venv/bin/python3" "$DIR/scripts/briefing_runner.py" --config "$DIR/config_local.yaml" --log-level DEBUG "$@" 2>&1 | logger -t local-briefing
RC_LOCAL=$?

if [ $RC_MAIN -ne 0 ] || [ $RC_LOCAL -ne 0 ]; then
    exit 1
fi
exit 0
