#!/bin/bash
# Atlas Morning Briefing Runner Wrapper Script
# Runs main briefing first, then local (San Diego/CA) — sequential to avoid
# simultaneous API hits against Brave + Gemini from the same IP — and then
# audits what they produced.

# Resolve script directory so relative paths work correctly from cron
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PATH to include gemini-cli, opencode (linuxbrew), and other necessary binaries
export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:$HOME/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Navigate to project directory
cd "$DIR" || exit 1

# Main briefing (defense/tech) runs first.
# PIPESTATUS, not $?: the pipe ends in logger, so $? is logger's status and a
# failed briefing would look like a success to cron.
"$DIR/.venv/bin/python3" "$DIR/scripts/briefing_runner.py" --config "$DIR/config.yaml" --log-level DEBUG "$@" 2>&1 | logger -t atlas-briefing
RC_MAIN="${PIPESTATUS[0]}"

# Local briefing (San Diego / CA) runs after main completes
"$DIR/.venv/bin/python3" "$DIR/scripts/briefing_runner.py" --config "$DIR/config_local.yaml" --log-level DEBUG "$@" 2>&1 | logger -t local-briefing
RC_LOCAL="${PIPESTATUS[0]}"

# Audit what was just produced. Chained rather than scheduled at a fixed time:
# run length varies with LLM backend health (15 min one morning, 32 the next),
# so a clock-based check raced the pipeline and reported the local briefing
# missing when it was still being written. Running here means the audit starts
# when the work is actually finished, whatever that takes.
if [ "${SKIP_QUALITY_CHECK:-0}" != "1" ]; then
    QC_ARGS=()
    # Briefings run Mon-Sat, so the weekly deep probe rides along on Saturday
    # rather than firing on a Sunday when there is no briefing to audit.
    [ "$(date +%u)" = "6" ] && QC_ARGS+=("--deep")
    # Forward --dry-run so a dry briefing run doesn't email a real alert.
    case " $* " in *" --dry-run "*) QC_ARGS+=("--dry-run") ;; esac

    "$DIR/run_quality_check.sh" "${QC_ARGS[@]}"
    RC_QUALITY=$?
    # Findings are reported by email, not by this exit code. Only a checker
    # malfunction (exit 2) is worth surfacing to cron here; a briefing that
    # shipped with problems is still a briefing that shipped.
    if [ "$RC_QUALITY" = "2" ]; then
        echo "quality check failed to run (exit 2)" | logger -t quality-check
    fi
fi

if [ "$RC_MAIN" -ne 0 ] || [ "$RC_LOCAL" -ne 0 ]; then
    exit 1
fi
exit 0
