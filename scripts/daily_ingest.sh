#!/usr/bin/env bash
# Incremental re-index of the notes: new/changed files are indexed, deleted ones removed, the rest skipped.
# Meant to be run by Windows Task Scheduler once a day (see README "Daily re-index").
#
#   bash scripts/daily_ingest.sh                     # default folder below
#   bash scripts/daily_ingest.sh /mnt/e/AI_ML/ML ... # or any folders
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/venvs/researchAgent/bin/python}"
FOLDERS=("${@:-/mnt/e/AI_ML/ML}")
LOG_DIR="$HOME/.ml_researcher/logs"
LOG="$LOG_DIR/ingest-$(date +%F).log"
KEEP_DAYS=30

mkdir -p "$LOG_DIR"
exec >>"$LOG" 2>&1                      # everything below goes to today's log
echo "=== $(date '+%F %T') daily ingest: ${FOLDERS[*]}"

# Only one ingest at a time: skip if a manual run (e.g. in tmux) or yesterday's job is still going.
exec 9>"$HOME/.ml_researcher/ingest.lock"
if ! flock -n 9 || pgrep -f "python -m ml_researcher.ingest" >/dev/null; then
    echo "another ingest is running, skipping"
    exit 0
fi

for f in "${FOLDERS[@]}"; do
    if [[ ! -d "$f" ]]; then             # e.g. drive E: not mounted
        echo "folder not found: $f, skipping run"
        exit 1
    fi
done

cd "$PROJECT_DIR"
export TQDM_DISABLE=1                    # no progress bar in a log file; per-file lines still printed
start=$SECONDS
if "$PYTHON" -m ml_researcher.ingest "${FOLDERS[@]}"; then
    echo "=== finished in $(( (SECONDS - start) / 60 )) min"
else
    echo "=== FAILED (exit $?) after $(( (SECONDS - start) / 60 )) min"
fi

find "$LOG_DIR" -name 'ingest-*.log' -mtime +"$KEEP_DAYS" -delete
