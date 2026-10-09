#!/bin/bash
# Personal finance briefing. A separate wrapper rather than a third branch in
# run_briefing.sh: its only consumer is ~/sender-finances, which imports the
# report at 07:30, and it sends no email.

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PATH="$HOME/.nvm/versions/node/v20.19.5/bin:$HOME/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

cd "$DIR" || exit 1

# A dry run logs under its own journald tag: the quality check harvests
# per-source yields from the real one.
TAG_SUFFIX=""
case " $* " in *" --dry-run "*) TAG_SUFFIX="-dryrun" ;; esac

# PIPESTATUS, not $?: the pipe ends in logger, so $? would be logger's status.
"$DIR/.venv/bin/python3" "$DIR/scripts/briefing_runner.py" --config "$DIR/config_finance.yaml" --log-level INFO "$@" 2>&1 | logger -t "finance-briefing$TAG_SUFFIX"
RC_FINANCE="${PIPESTATUS[0]}"

# Audit what was just produced. Chained here, like the main+local audit at the
# end of run_briefing.sh and for the same reason: that audit finishes around
# 07:10, before this pipeline has written its briefing, so including finance
# there would report it missing every morning. Its digest gets its own file
# (logs/quality-digest-DATE-finance.md) so it does not replace the first one.
if [ "${SKIP_QUALITY_CHECK:-0}" != "1" ]; then
    QC_ARGS=(--config "$DIR/config_finance.yaml" --digest-suffix finance)
    [ "$(date +%u)" = "6" ] && QC_ARGS+=("--deep")
    # Forward --dry-run so a dry briefing run doesn't email a real alert.
    case " $* " in *" --dry-run "*) QC_ARGS+=("--dry-run") ;; esac

    "$DIR/run_quality_check.sh" "${QC_ARGS[@]}"
    # Findings are reported by email, not by this exit code; only a checker
    # that could not run at all is worth surfacing here.
    if [ "$?" = "2" ]; then
        echo "quality check failed to run (exit 2)" | logger -t quality-check
    fi
fi

# The briefing's own status, which is what cron and ~/sender-finances care about.
exit "$RC_FINANCE"
