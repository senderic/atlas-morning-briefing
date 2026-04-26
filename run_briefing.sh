#!/bin/bash
# Atlas Morning Briefing Runner Wrapper Script

# Resolve script directory so relative paths work correctly from cron
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PATH to include gemini-cli and other necessary binaries
export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Navigate to project directory
cd "$DIR" || exit 1

# Load environment variables (optional if .env is used by python)
# if [ -f .env ]; then
#   export $(grep -v '^#' .env | xargs)
# fi

# Run the briefing runner with DEBUG log level as requested
# Using the venv python directly
"$DIR/.venv/bin/python3" "$DIR/scripts/briefing_runner.py" --config "$DIR/config.yaml" --log-level DEBUG "$@" 2>&1 | logger -t atlas-briefing
