#!/usr/bin/env bash
# Generation eval as a detached batch job (survives closing the terminal / editor).
#
#   bash scripts/run_eval.sh          # 30 questions (default)
#   bash scripts/run_eval.sh 100      # all questions
#
# - checks first that Gemini answers (a suspended project fails fast instead of burning time)
# - resumes the latest unfinished run (per_query.jsonl but no summary.json) instead of starting over
# - Gemini calls are paced at BATCH_RPS (default 1/s, see .env); Cohere reranks are not Gemini calls
# Follow it with:  tail -f ~/.ml_researcher/logs/eval-generation.log
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$HOME/venvs/researchAgent/bin/python}"
LIMIT="${1:-30}"
WORKERS="${WORKERS:-2}"
LOG="$HOME/.ml_researcher/logs/eval-generation.log"
cd "$PROJECT_DIR"
mkdir -p "$(dirname "$LOG")"

if pgrep -f "ml_researcher.evaluate generation" >/dev/null; then
    echo "a generation eval is already running; follow it with: tail -f $LOG"
    exit 0
fi

echo "checking Gemini access..."
if ! "$PYTHON" -c "from ml_researcher.llm import FAST_LLM; FAST_LLM.invoke('Say ok.')" >/dev/null 2>&1; then
    echo "Gemini call failed (project suspended, auth or network?). Not starting."
    exit 1
fi

RESUME=()
for d in $(ls -td evals/runs/*-generation 2>/dev/null); do
    if [[ -f "$d/per_query.jsonl" && ! -f "$d/summary.json" ]]; then
        RESUME=(--resume "$d")
        echo "resuming unfinished run $d ($(wc -l < "$d/per_query.jsonl") done)"
        break
    fi
done

setsid nohup "$PYTHON" -m ml_researcher.evaluate generation --limit "$LIMIT" --workers "$WORKERS" "${RESUME[@]}" \
    >>"$LOG" 2>&1 < /dev/null &
echo "started (pid $!): $LIMIT questions, $WORKERS workers"
echo "follow:  tail -f $LOG"
echo "results: evals/runs/<time>-generation/summary.json"
