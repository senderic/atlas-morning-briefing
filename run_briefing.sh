#!/bin/bash
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set PATH to include node/npm binaries for gemini-cli
export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:$PATH"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    set -a
    source .env
    set +a
fi

# Clean up old files
rm -f Atlas-Briefing-*.md Atlas-Briefing-*.pdf status.json

# Run the briefing
# Upstream v0.2 introduces briefing_runner_v2.py for parallel execution
# PYTHONUNBUFFERED ensures log lines flush immediately through the pipe
# Redirect output to logger so it shows up in journalctl
PYTHONUNBUFFERED=1 python3 scripts/briefing_runner_v2.py --config config.yaml 2>&1 | logger -t atlas-briefing
