#!/usr/bin/env bash
# Copyright (c) 2026 Junjie Tang. MIT License. See LICENSE file for details.
#
# scripts/run_alignment_check.sh
#
# End-to-end alignment-test wrapper, designed for unattended tmux runs.
#
# 1. git pull origin <current branch>
# 2. Run scripts/test_briefing_alignment.py (live mode by default)
# 3. Rename today's briefing with a unique suffix so same-day re-runs don't
#    clobber each other
# 4. git add -f the briefing + alignment log + alignment report (all of which
#    are gitignored — that's why -f is needed)
# 5. git commit with a structured message that captures the run timestamp,
#    test exit code, verdict, and a tail of the alignment report
# 6. git push -u origin <branch>, retrying up to 4× on network failure with
#    exponential backoff (2s, 4s, 8s, 16s)
#
# Usage:
#   ./scripts/run_alignment_check.sh                     # live mode (default, ~22 min)
#   ./scripts/run_alignment_check.sh --mock              # fast structural smoke test
#   ./scripts/run_alignment_check.sh --no-push           # commit but don't push
#   ./scripts/run_alignment_check.sh --suffix=v3-fix     # custom snapshot suffix
#   ./scripts/run_alignment_check.sh --include-binaries  # also commit PDF + EPUB
#   ./scripts/run_alignment_check.sh --help
#
# Run inside tmux:
#   tmux new -s align
#   ./scripts/run_alignment_check.sh
#   # Ctrl-b d to detach; tmux attach -t align to come back

set -euo pipefail

# ── Resolve paths so the script works from any cwd ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ── Args ────────────────────────────────────────────────────────────────
MODE=live
NO_PUSH=0
SUFFIX="$(date +%H%M)"
INCLUDE_BINARIES=0

print_help() {
    sed -n '3,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mock)              MODE=mock; shift;;
        --no-push)           NO_PUSH=1; shift;;
        --suffix=*)          SUFFIX="${1#--suffix=}"; shift;;
        --include-binaries)  INCLUDE_BINARIES=1; shift;;
        -h|--help)           print_help; exit 0;;
        *) echo "Unknown argument: $1" >&2; echo "Try --help" >&2; exit 2;;
    esac
done

# ── Helpers ─────────────────────────────────────────────────────────────
log()  { printf "[%s] %s\n" "$(date '+%H:%M:%S')" "$*"; }
step() { printf "\n[%s] ═══ %s ═══\n" "$(date '+%H:%M:%S')" "$*"; }

# ── Setup ───────────────────────────────────────────────────────────────
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
STARTED="$(date -Iseconds)"
log "Repo:    $REPO_ROOT"
log "Branch:  $BRANCH"
log "Mode:    $MODE"
log "Suffix:  $SUFFIX"
log "Started: $STARTED"

if [[ "$BRANCH" != "claude/funny-fermi-z9U11" ]]; then
    log "WARNING: not on claude/funny-fermi-z9U11 (you're on '$BRANCH'). Continuing anyway."
fi

# Locate Python — prefer the project venv, fall back to system
PYTHON="${PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
    if   [[ -x "$REPO_ROOT/.venv/bin/python" ]];  then PYTHON="$REPO_ROOT/.venv/bin/python"
    elif [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then PYTHON="$REPO_ROOT/.venv/bin/python3"
    elif command -v uv >/dev/null 2>&1;           then PYTHON="uv run python"
    else                                               PYTHON="python3"
    fi
fi
log "Python:  $PYTHON"

# ── Step 1: pull ────────────────────────────────────────────────────────
step "Step 1/5: git pull origin $BRANCH"
for attempt in 1 2 3 4; do
    if git pull origin "$BRANCH"; then break; fi
    if [[ "$attempt" -eq 4 ]]; then log "Pull failed after 4 attempts; aborting."; exit 1; fi
    BACKOFF=$((2 ** attempt))
    log "Pull failed, retrying in ${BACKOFF}s..."
    sleep "$BACKOFF"
done

# ── Step 2: run the alignment test ──────────────────────────────────────
step "Step 2/5: Running alignment test ($MODE mode)"
TEST_ARGS=()
[[ "$MODE" == "mock" ]] && TEST_ARGS+=("--mock")

TEST_EXIT=0
# shellcheck disable=SC2086
$PYTHON scripts/test_briefing_alignment.py "${TEST_ARGS[@]}" || TEST_EXIT=$?
log "Alignment test exit code: $TEST_EXIT"

case "$TEST_EXIT" in
    0) VERDICT="ALIGNED";;
    1) VERDICT="MISALIGNED";;
    *) VERDICT="GENERATION_FAILED";;
esac
log "Verdict: $VERDICT"

# ── Step 3: locate + snapshot artifacts ────────────────────────────────
step "Step 3/5: Collecting artifacts"
TODAY="$(date +%Y.%m.%d)"
BRIEFING_FILE="Atlas-Briefing-$TODAY.md"
SNAPSHOT_MD=""

# Most recent log + report (the test writes timestamps so we pick the newest)
LATEST_LOG="$(ls -t logs/alignment-*.log         2>/dev/null | head -1 || true)"
LATEST_REPORT="$(ls -t logs/alignment-report-*.txt 2>/dev/null | head -1 || true)"

if [[ -f "$BRIEFING_FILE" ]]; then
    SNAPSHOT_MD="Atlas-Briefing-$TODAY-$SUFFIX.md"
    mv "$BRIEFING_FILE" "$SNAPSHOT_MD"
    log "Briefing snapshot: $SNAPSHOT_MD"
else
    log "No $BRIEFING_FILE found (mock mode or runner failed before save)"
fi

ADD_LIST=()
[[ -n "$SNAPSHOT_MD"    ]] && ADD_LIST+=("$SNAPSHOT_MD")
[[ -n "$LATEST_LOG"     ]] && ADD_LIST+=("$LATEST_LOG")
[[ -n "$LATEST_REPORT"  ]] && ADD_LIST+=("$LATEST_REPORT")

if [[ "$INCLUDE_BINARIES" -eq 1 ]]; then
    for ext in pdf epub; do
        F="Atlas-Briefing-$TODAY.$ext"
        if [[ -f "$F" ]]; then
            SNAPSHOT_BIN="Atlas-Briefing-$TODAY-$SUFFIX.$ext"
            mv "$F" "$SNAPSHOT_BIN"
            ADD_LIST+=("$SNAPSHOT_BIN")
            log "Binary snapshot: $SNAPSHOT_BIN"
        fi
    done
fi

if [[ ${#ADD_LIST[@]} -eq 0 ]]; then
    log "No artifacts to commit. Exiting with test exit code $TEST_EXIT."
    exit "$TEST_EXIT"
fi

# ── Step 4: git add -f + commit ─────────────────────────────────────────
step "Step 4/5: git add -f + commit"
git add -f "${ADD_LIST[@]}"
log "Staged ${#ADD_LIST[@]} file(s):"
for f in "${ADD_LIST[@]}"; do log "  + $f"; done

if git diff --cached --quiet; then
    log "Nothing to commit (files unchanged or already tracked at this content). Exiting."
    exit "$TEST_EXIT"
fi

# Build a structured commit message. Use heredoc for safe formatting.
REPORT_TAIL=""
if [[ -n "$LATEST_REPORT" && -f "$LATEST_REPORT" ]]; then
    REPORT_TAIL="$(tail -20 "$LATEST_REPORT")"
fi

ARTIFACT_LIST="$(printf -- '- %s\n' "${ADD_LIST[@]}")"

git commit -m "$(cat <<EOF
chore(alignment): $MODE-mode snapshot ($VERDICT)

Run started:  $STARTED
Run finished: $(date -Iseconds)
Test exit:    $TEST_EXIT ($VERDICT)
Branch:       $BRANCH
Suffix:       $SUFFIX

Artifacts (all force-added past .gitignore):
$ARTIFACT_LIST
Report tail:
$REPORT_TAIL
EOF
)"
log "Commit created: $(git rev-parse --short HEAD)"

# ── Step 5: push ────────────────────────────────────────────────────────
if [[ "$NO_PUSH" -eq 1 ]]; then
    step "Step 5/5: Skipping push (--no-push set)"
    log "To push manually: git push -u origin $BRANCH"
else
    step "Step 5/5: git push -u origin $BRANCH"
    PUSH_OK=0
    for attempt in 1 2 3 4; do
        if git push -u origin "$BRANCH"; then PUSH_OK=1; break; fi
        if [[ "$attempt" -eq 4 ]]; then break; fi
        BACKOFF=$((2 ** attempt))
        log "Push failed, retrying in ${BACKOFF}s..."
        sleep "$BACKOFF"
    done
    if [[ "$PUSH_OK" -ne 1 ]]; then
        log "Push failed after 4 attempts. Commit is in place locally — push manually."
        exit 1
    fi
fi

step "Done"
log "Test exit code: $TEST_EXIT ($VERDICT)"
log "If MISALIGNED or GENERATION_FAILED, check the committed log:"
log "  $LATEST_LOG"
exit "$TEST_EXIT"
