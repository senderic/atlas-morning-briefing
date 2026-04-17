#!/bin/bash
# Atlas Morning Briefing Runner Wrapper Script

# Set PATH to include gemini-cli and other necessary binaries
export PATH="/home/eric/.nvm/versions/node/v20.19.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Navigate to project directory
cd /home/eric/atlas-morning-briefing

# Load environment variables (optional if .env is used by python)
# if [ -f .env ]; then
#   export $(grep -v '^#' .env | xargs)
# fi

# Run the briefing runner with DEBUG log level as requested
# Using the venv python directly
/home/eric/atlas-morning-briefing/.venv/bin/python3 scripts/briefing_runner.py --config config.yaml --log-level DEBUG "$@" 2>&1 | logger -t atlas-briefing
