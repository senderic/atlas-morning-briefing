#!/bin/bash
# Personal finance briefing. A separate wrapper rather than a third branch in
# run_briefing.sh: its only consumer is ~/sender-finances, which imports the
# report at 07:30, and it sends no email.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:$HOME/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

cd "$DIR" || exit 1

# PIPESTATUS, not $?: the pipe ends in logger, so $? would be logger's status.
"$DIR/.venv/bin/python3" "$DIR/scripts/briefing_runner.py" --config "$DIR/config_finance.yaml" --log-level INFO "$@" 2>&1 | logger -t finance-briefing
exit "${PIPESTATUS[0]}"
