#!/bin/bash
# Atlas Morning Briefing Runner Wrapper Script
# By default, runs main then local sequentially. Cron uses RUN_MAIN_ONLY at
# 06:00 and RUN_LOCAL_ONLY at 07:00 so Axios San Diego has arrived before the
# local report. The quality audit follows the local run.

# Resolve script directory so relative paths work correctly from cron
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set PATH to include gemini-cli, opencode (linuxbrew), and other necessary binaries
export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:$HOME/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# Navigate to project directory
cd "$DIR" || exit 1

RUN_MAIN_ONLY="${RUN_MAIN_ONLY:-0}"
RUN_LOCAL_ONLY="${RUN_LOCAL_ONLY:-0}"
if [ "$RUN_MAIN_ONLY" = "1" ] && [ "$RUN_LOCAL_ONLY" = "1" ]; then
    echo "RUN_MAIN_ONLY and RUN_LOCAL_ONLY cannot both be 1" | logger -t atlas-briefing
    exit 2
fi

RC_MAIN=0
RC_LOCAL=0

# A dry run changes nothing cron-facing tools read. Its log lines go to their
# own journald tags, because the quality check harvests per-source yields from
# the real ones; and it skips the pre-flight probe, which rewrites
# .model-availability.json (the dry run still reads the existing file).
DRY_RUN=0
TAG_SUFFIX=""
case " $* " in *" --dry-run "*) DRY_RUN=1; TAG_SUFFIX="-dryrun" ;; esac

# Pre-flight model availability check. Probes the tiered model roster from
# config.yaml and writes .model-availability.json, which both briefings read to
# pin a working model per tier. A non-zero exit means some tier had no reachable
# model; the run continues on the configured defaults either way.
if [ "$RUN_LOCAL_ONLY" != "1" ] && [ "$DRY_RUN" != "1" ]; then
    "$DIR/.venv/bin/python3" "$DIR/scripts/preflight_model_check.py" \
        --config "$DIR/config.yaml" 2>&1 | logger -t preflight-check
    RC_PREFLIGHT="${PIPESTATUS[0]}"
    if [ "$RC_PREFLIGHT" -ne 0 ]; then
        logger -t preflight-check "Pre-flight check failed (rc=$RC_PREFLIGHT), continuing with config defaults"
    fi
fi

# Main briefing (defense/tech) runs first, or alone for the 06:00 cron job.
# PIPESTATUS, not $?: the pipe ends in logger, so $? is logger's status and a
# failed briefing would look like a success to cron.
if [ "$RUN_LOCAL_ONLY" != "1" ]; then
    "$DIR/.venv/bin/python3" "$DIR/scripts/briefing_runner.py" --config "$DIR/config.yaml" --log-level DEBUG "$@" 2>&1 | logger -t "atlas-briefing$TAG_SUFFIX"
    RC_MAIN="${PIPESTATUS[0]}"
fi

# Local briefing runs after main in manual/full mode, or alone at 07:00 in cron.
if [ "$RUN_MAIN_ONLY" != "1" ]; then
    "$DIR/.venv/bin/python3" "$DIR/scripts/briefing_runner.py" --config "$DIR/config_local.yaml" --log-level DEBUG "$@" 2>&1 | logger -t "local-briefing$TAG_SUFFIX"
    RC_LOCAL="${PIPESTATUS[0]}"
fi

# Audit what was just produced. Chained rather than scheduled at a fixed time:
# run length varies with LLM backend health (15 min one morning, 32 the next),
# so a clock-based check raced the pipeline and reported the local briefing
# missing when it was still being written. Running here means the audit starts
# when the work is actually finished, whatever that takes.
if [ "$RUN_MAIN_ONLY" != "1" ] && [ "${SKIP_QUALITY_CHECK:-0}" != "1" ]; then
    # These two pipelines only: finance has not run yet at this point and is
    # audited by run_finance_briefing.sh when it finishes.
    QC_ARGS=(--config "$DIR/config.yaml" --config "$DIR/config_local.yaml")
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
