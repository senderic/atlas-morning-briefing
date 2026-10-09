#!/usr/bin/env bash
# Daily quality check — runs after both briefings have shipped.
#
# Reviews what the pipelines actually produced: source health harvested from
# journald, deterministic invariants over the rendered briefings, and an LLM
# judge scoring the editorial goals. See references/quality_monitoring_design.md.
#
# NOT scheduled in cron. Each briefing wrapper chains this when its own work
# is done, because run length varies with LLM backend health and a clock-based
# schedule raced the pipeline: run_briefing.sh audits main + local after the
# local run, and run_finance_briefing.sh audits finance after the finance run
# (which starts at 07:10, usually after the first audit has already finished).
# Saturday runs pick up --deep from those callers.
#
# With no --config this audits every pipeline; a caller that passes its own
# --config list gets exactly those. Invoke by hand for an ad-hoc audit:
#   ./run_quality_check.sh --dry-run --date YYYY-MM-DD
#
# Exit codes: 0 = clean or warnings only, 1 = CRITICAL findings, 2 = the
# checker itself failed. Cron mails on nonzero.

set -uo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:$HOME/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
cd "$DIR" || exit 2

CONFIG_ARGS=()
case " $* " in
    *" --config "*) ;;
    *) CONFIG_ARGS=(
           --config "$DIR/config.yaml"
           --config "$DIR/config_local.yaml"
           --config "$DIR/config_finance.yaml"
       ) ;;
esac

# The two chained audits can overlap on a slow morning, and both rewrite the
# alert-dedupe and streak files whole. Take turns; if the lock cannot be had
# in 15 minutes, run anyway rather than skip an audit.
mkdir -p "$DIR/logs"
exec 9>"$DIR/logs/.quality-check.lock"
flock -w 900 9 || echo "quality check: lock busy for 15 min, running unlocked" | logger -t quality-check

"$DIR/.venv/bin/python3" "$DIR/scripts/quality_check.py" \
    ${CONFIG_ARGS[@]+"${CONFIG_ARGS[@]}"} \
    "$@" 2>&1 | logger -t quality-check

# logger sits at the end of the pipe, so take the checker's status, not its.
exit "${PIPESTATUS[0]}"
