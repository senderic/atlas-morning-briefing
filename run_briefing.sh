#!/bin/bash
# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "venv" ]; then
    source venv/bin/activate
fi

# Load environment variables
if [ -f ".env" ]; then
    source .env
fi

# Clean up old files
rm -f Atlas-Briefing-*.md Atlas-Briefing-*.pdf status.json

# Run the briefing
# Upstream v0.2 introduces briefing_runner_v2.py for parallel execution
python3 scripts/briefing_runner_v2.py --config config.yaml
