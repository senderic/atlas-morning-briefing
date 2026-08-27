#!/usr/bin/env bash
# Daily quality check — runs after both briefings have shipped.
#
# Reviews what the pipelines actually produced: source health harvested from
# journald, deterministic invariants over the rendered briefings, and an LLM
# judge scoring the editorial goals. See references/quality_monitoring_design.md.
#
# Cron (both briefings finish by ~06:25):
#   40 6 * * 1-6 /home/eric/atlas-morning-briefing/run_quality_check.sh
#   15 7 * * 0   /home/eric/atlas-morning-briefing/run_quality_check.sh --deep
#
# Exit codes: 0 = clean or warnings only, 1 = CRITICAL findings, 2 = the
# checker itself failed. Cron mails on nonzero.

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:$HOME/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
cd "$DIR" || exit 2

"$DIR/.venv/bin/python3" "$DIR/scripts/quality_check.py" \
    --config "$DIR/config.yaml" \
    --config "$DIR/config_local.yaml" \
    "$@" 2>&1 | logger -t quality-check

# logger sits at the end of the pipe, so take the checker's status, not its.
exit "${PIPESTATUS[0]}"
