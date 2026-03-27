#!/bin/bash
# Atlas Morning Briefing Runner Wrapper Script

# Set PATH to include gemini-cli and other necessary binaries
export PATH="/home/eric/.nvm/versions/node/v20.19.5/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Navigate to project directory
cd /home/eric/atlas-morning-briefing

# Load environment variables from .env if it exists
if [ -f .env ]; then
  set -o allexport
  source .env
  set +o allexport
fi

# Run the briefing runner
# Using the venv python directly and passing all arguments
/home/eric/atlas-morning-briefing/.venv/bin/python3 scripts/briefing_runner.py --config config.yaml "$@" 2>&1 | logger -t atlas-briefing
