# ML Researcher

A personal research agent for machine-learning topics. You ask a question, and it searches **arXiv, your own notes, GitHub and YouTube** in parallel. It then writes a study brief with math rendered as LaTeX. Follow-up questions are answered from the conversation, without searching again.

It's built on LangGraph with Gemini (through Vertex AI). Your notes (PDF, Word, PowerPoint, Markdown) are indexed into a local Chroma database, and figures and diagrams are captioned by Gemini so they can be searched too.

```
                    ┌─> arxiv_researcher   ─┐
router ─> planner ──┼─> notes_researcher   ─┼─> reviewer ─> synthesizer ─> study brief
   │                ├─> github_researcher  ─┤     │  ▲
   │                └─> youtube_researcher ─┘     ▼  │
   │                               follow-ups: search a source again, or a Python check
   └─> answer  (follow-ups: answered from the conversation, no search)
```

After the first pass, a **reviewer** looks for gaps. For example, a video might use a term your notes weren't searched for, or a formula could be checked. It can then send follow-ups back to any source, or ask for a **Python check**. The brief gets a `## Checks` section when one runs.

**Loop guards** (constants at the top of `graph.py`):
- **Inside each researcher:**
  - at most 3 tool calls;
  - every call and result is hashed;
  - a repeated call isn't run again;
  - an identical result is flagged, and the prompt is rewritten to list the failed attempts;
  - after 2 repeats, the model must report without tools.
- **In the reviewer cycle:**
  - at most 2 rounds, 3 follow-ups per round, and 2 per source;
  - each (source, query) is hashed with word order ignored, so a reworded repeat is blocked;
  - a round that brings no new raw results ends the cycle.

**Python checks run in the model provider's hosted sandbox** (Gemini or Claude code execution, see `sandbox.py`): no network, none of your files. Code a model writes is never run on your machine; only the task and the findings text are sent. Expect up to about 60 s per check. With `CODE_SANDBOX=none`, Python checks are switched off.

## Setup

Developed on WSL2 (Ubuntu) with Python 3.14.

```bash
python -m venv ~/venvs/researchAgent && source ~/venvs/researchAgent/bin/activate
pip install -r requirements.txt          # or requirements-dev.txt to also get ruff
gcloud auth application-default login
```

The Google Cloud project needs the **Vertex AI API** enabled. If you want logs and metrics in Google Cloud (see [Monitoring](#monitoring)), also enable the **Cloud Logging** and **Cloud Monitoring** APIs. Optional system tools:
- `libreoffice`, to read `.ppt` files
- `antiword`, to read `.doc` files

Copy `.env.example` to `.env` and fill in your project ID and keys. `.env` is in `.gitignore`, so it's never committed. The main settings:

```ini
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_GENAI_USE_VERTEXAI=TRUE
GOOGLE_CLOUD_LOCATION=global
MAIN_MODEL=google_genai:gemini-3.8-flash          # router, planner, reviewer, synthesizer, answers
RESEARCHER_MODEL=google_genai:gemini-3.5-flash-lite   # the 4 researchers + image captions
# Claude instead: MAIN_MODEL=anthropic:claude-sonnet-5-5 and ANTHROPIC_API_KEY=...  (embeddings stay on Gemini)
EMBED_MODEL=gemini-embedding-2
EMBED_DIM=768
LOCAL_DOCS_DIR=/mnt/e/AI_ML              # default folder for ingest
GITHUB_TOKEN=...                         # optional, raises GitHub's rate limit
YOUTUBE_API_KEY=...                      # needed for YouTube search
LOG_TO_GCP=true                          # optional, see Monitoring
```

<details>
<summary>All settings</summary>

| Variable | Default | Meaning |
|---|---|---|
| `VECTOR_BACKEND` | `chroma` | `chroma` (local) or `rag_engine` (Vertex AI RAG Engine) |
| `CHROMA_DIR` | `~/.ml_researcher/chroma` | where the local vector database lives |
| `EMBED_WORKERS` | `4` | parallel embedding calls per file |
| `FILE_WORKERS` | `2` | files indexed in parallel |
| `HYBRID_SEARCH` | `true` | combine BM25 keyword search with vector search |
| `HNSW_EF_SEARCH` | `100` | Chroma search breadth |
| `COHERE_API_KEY` | (none) | turns on reranking; without it, search keeps the fused order |
| `RERANK` | `true` | use the Cohere reranker when a key is set |
| `RERANK_MODEL` | `rerank-v5.0-pro` | Cohere rerank model |
| `RERANK_CANDIDATES` | `15` | candidates from fused vector + keyword search that the reranker orders; the top 5 are returned |
| `MAIN_MODEL`, `RESEARCHER_MODEL` | Gemini 3.8 Flash, 3.5 Flash Lite | `provider:model` (`google_genai:…` or `anthropic:…`); a bare name means Gemini |
| `JUDGE_MODEL`, `VISION_MODEL` | `MAIN_MODEL`, `RESEARCHER_MODEL` | eval judge and question writer; image captions (any multimodal model) |
| `CODE_SANDBOX` | `auto` | hosted Python for checks: `auto` uses `MAIN_MODEL`'s provider; `none` switches checks off |
| `LLM_RPS` | `5` | maximum model calls per second per process (chat, UI), spaced evenly; `0` means no limit (`GEMINI_RPS` still works) |
| `BATCH_RPS` | `1` | stricter pace that `ingest` and `evaluate` switch to, for long runs |
| `CHECKPOINT_DB` | `~/.ml_researcher/checkpoints.sqlite` | saved conversations |
| `LOG_DIR` | `~/.ml_researcher/logs` | local JSON logs |
| `LOG_LEVEL` | `INFO` | |
| `LOG_TO_GCP` | `false` | also send logs to Cloud Logging |
| `LOG_NAME` | `ml-researcher` | Cloud Logging log name |
| `LOG_PROMPTS` | `true` | include prompt text in request logs |
| `RAG_LOCATION`, `RAG_CORPUS_NAME` | `europe-west4`, `ml-notes` | only used with `rag_engine` |

</details>

## 1. Index your notes

```bash
python -m ml_researcher.ingest /mnt/e/AI_ML/ML "E:\AI_ML\Notes"   # Linux or Windows paths
python -m ml_researcher.ingest                                    # no argument: LOCAL_DOCS_DIR
```

- **Incremental:** only new or changed files are indexed, judged by modification time. Files deleted from disk are removed from the index, so you can re-run it any time.
- **Order:** small files go first, so your notes are searchable before the big books are done.
- **Images:** pages with figures, diagrams or little text are rendered and captioned by Gemini. Images in Word and PowerPoint files are captioned too. Captions are cached in `~/.ml_researcher/captions.json`, so re-indexing doesn't pay for them again.
- **Chunking:** text is split into chunks of about 1,500 characters with 300 characters of overlap. Splits fall on paragraph and sentence boundaries.

A large folder takes hours, so run it inside `tmux`. That way it survives closing the terminal:

```bash
tmux new -s ingest
python -m ml_researcher.ingest /mnt/e/AI_ML/ML 2>&1 | tee ~/ingest_$(date +%F).log
# detach: Ctrl+B then D      reattach: tmux attach -t ingest
grep -c '"ingest_file"' ~/.ml_researcher/logs/agent.jsonl            # files done so far
```

Two helper scripts:

```bash
python test_readers.py "E:\AI_ML\ML\Papers\paper.pdf" "search query"  # what one file extracts to, plus a test search
python inspect_db.py "search query"                                    # chunk counts per file and type, plus a test search
```

### Daily re-index

`scripts/daily_ingest.sh` runs the same incremental ingest and logs to `~/.ml_researcher/logs/ingest-YYYY-MM-DD.log`, keeping 30 days of logs. If another ingest is already running, it skips. Schedule it with Windows Task Scheduler rather than cron, because WSL isn't running unless something starts it. In PowerShell:

```powershell
$a = New-ScheduledTaskAction -Execute "conhost.exe" `
     -Argument "--headless wsl.exe -d Ubuntu -u risha -- bash /mnt/e/AI_ML/Prac/researchAgent/scripts/daily_ingest.sh"
$t = New-ScheduledTaskTrigger -Daily -At 3am
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
     -ExecutionTimeLimit (New-TimeSpan -Hours 12)
Register-ScheduledTask -TaskName "ML Researcher daily ingest" -Action $a -Trigger $t -Settings $s
```

`-StartWhenAvailable` runs a missed day as soon as you log in. Run it now with `Start-ScheduledTask "ML Researcher daily ingest"`, and remove it with `Unregister-ScheduledTask "ML Researcher daily ingest"`.

## 2. Chat

**Web UI** (recommended), at http://localhost:8501:

```bash
streamlit run ui.py
```

- Conversations are saved; switch between them or start new ones from the sidebar.
- Replies stream in as they're written, with LaTeX rendered.
- **⏹ Stop** cancels a reply mid-way and puts your prompt back in the box to edit.
- The sidebar shows the search queries used and the raw findings from each source.
- You can download the last reply as Markdown.

**Terminal:**

```bash
python -m ml_researcher.chat --thread flow-matching    # reuse a thread name to continue it
```

Commands: `/state`, `/graph` (prints a Mermaid diagram), `/quit`.

## Monitoring

Every turn writes JSON log lines to `~/.ml_researcher/logs/agent.jsonl`:
- one `step` event per graph node
- one `request` event per turn, with status, error code, route, turn number, latency, time to first token (TTFT), time per output token (TPOT) and token counts

Indexing also writes an `ingest_file` event per file. To summarize the local log:

```bash
python -m ml_researcher.log_report --last 200    # p50 / p95 per metric and per step
```

With `LOG_TO_GCP=true`, the same events go to Cloud Logging. To create the log-based metrics and the **"ML Researcher Agent"** dashboard (requests, errors, p50/p95 latency, TTFT, TPOT, step latency), run this once:

```bash
python scripts/setup_gcp_metrics.py
```

It's safe to re-run: metrics are updated and the dashboard is replaced.

## Evaluation

`ml_researcher/evaluate.py` measures how well the notes search finds the right passages, and how good answers built from those passages are. The golden set (`evals/golden.jsonl`) and every run's results (`evals/runs/<time>-<kind>/summary.json` and `per_query.jsonl`) are saved in the project folder. Generation metrics need DeepEval: `pip install -r requirements-dev.txt`.

```bash
python -m ml_researcher.evaluate build --root /mnt/e/AI_ML/ML --n 100   # golden set, run once (about 10 min)
python -m ml_researcher.evaluate retrieval                               # about 1 s per question per mode
bash scripts/run_eval.sh 30          # generation eval as a detached batch job: 30 questions, resumable
```

**Golden set** (`evals/golden.jsonl`): Gemini writes questions with reference answers from random chunks of random indexed files. It skips tables of contents, reference lists and fragments. The default mix is 75% single-hop and 25% multi-hop:
- **Single-hop:** the question is written from one chunk.
- **Multi-hop:** the question needs one chunk plus its closest chunk from a *different* document. Pairs that are near-copies are skipped.

The set is reproducible with `--seed`, and it won't overwrite an existing set unless you pass `--force`. Read through a sample before trusting it: generated questions sometimes come out too easy or too vague.

**Retrieval:** three modes are run side by side: `rerank` (hybrid + Cohere), `hybrid` and `vector`. Use `--modes` to pick a subset. The `rerank` mode is skipped if no Cohere key is set. Results are reported for all questions, single-hop and multi-hop, at two levels:
- *Chunk level:* the exact gold passage, or an identical copy of it, must be retrieved.
- *File level:* any chunk from the right file counts.

| Metric | Meaning |
|---|---|
| `recall@k` | share of gold passages found in the top k (for multi-hop, the share of hops found) |
| `precision@k` | share of the top k that is relevant |
| `hit@k` | at least one gold passage in the top k |
| `mrr` | 1 / rank of the first relevant result |
| `dcg@k`, `ndcg@k` | rank-discounted gain, and gain divided by the ideal |
| `all@k` | multi-hop: *every* hop is in the top k |
| `mrr_all` | multi-hop: 1 / the rank at which the last hop appears |

**Generation:** answers are produced from the top 5 full passages (the same text the agent's notes tool sees) and judged by DeepEval, with Gemini as the judge:
- **faithfulness:** whether the claims in the answer are supported by the retrieved passages.
- **answer relevancy:** whether the answer addresses the question.
- **answer correctness:** a G-Eval comparison against the reference answer.

Reasons for every score are saved in `per_query.jsonl`, so you can see why a question failed. For example, an answer can be fully faithful but still wrong, because retrieval missed the passage and the model correctly said the notes don't cover it.

Run evaluations after an ingest has finished. While an ingest is running, every search rebuilds the BM25 keyword index, which takes about 7 s, so runs are slow, and the index keeps changing under the evaluation.

### Results (29 October 2026)

Retrieval on the 100-question golden set (75 single-hop, 25 multi-hop), with 42,527 chunks in the index (the questions were sampled when it held 39,154). Chunk level: the exact gold passage must be retrieved.

| Metric | **Rerank** (hybrid + Cohere, 15 → 5) | Hybrid (vector + BM25) | Vector only |
|---|---|---|---|
| hit@1 | **0.44** | 0.40 | 0.35 |
| hit@3 | **0.67** | 0.59 | 0.61 |
| hit@5 (what the agent sees) | **0.82** | 0.67 | 0.68 |
| hit@10 | **0.85** | 0.82 | 0.80 |
| MRR | **0.58** | 0.52 | 0.49 |
| nDCG@5 | **0.60** | 0.50 | 0.50 |
| recall@5 | **0.76** | 0.62 | 0.64 |
| Multi-hop: both hops in top 5 / top 10 | 0.44 / **0.80** | 0.40 / **0.80** | **0.48** / 0.76 |
| File level: hit@5 / MRR | **0.85 / 0.65** | 0.78 / 0.62 | 0.77 / 0.58 |

- **Reranking adds 15 points to hit@5,** well beyond the noise for 100 questions (about ±9 points).
- **Multi-hop is the weak spot:** both pieces of evidence reach the top 5 less than half the time. The reviewer's follow-up searches are meant to cover this.
- **A rerank call takes about 0.5 s on its own.** The eval measured 5.5 s per query because Cohere throttled the parallel workers.

**Generation** (30 October 2026, the first 30 golden questions: 24 single-hop and 6 multi-hop). Answers come from the top 5 reranked full chunks, with `gemini-3.8-flash` as both answerer and DeepEval judge, paced at 1 call per second. Run: `evals/runs/20261030-153656-generation/`.

| Metric | All | Single-hop | Multi-hop | Pass rate at 0.5 |
|---|---|---|---|---|
| Faithfulness | **0.99** | 1.00 | 0.96 | 100% |
| Answer relevancy | **1.00** | 1.00 | 1.00 | 100% |
| Answer correctness | **0.95** | 0.95 | 0.97 | 100% |

- **22 of 30 answers are fully correct.** The lowest scores (0.7) are an answer that added an incorrect variant of the Bellman target, and one that missed some of the reference's key points.
- **Only one answer has an unsupported claim:** a multi-hop question, with faithfulness 0.75.
- **No answer fell back to "the notes don't cover this".** Even when the exact gold chunk missed the top 5, neighbouring or duplicate chunks carried enough of the answer.

**Read these as an upper bound:**
- **The same model wrote the golden questions, answered them and judged them,** which tends to inflate scores (self-preference).
- **Scores near 1.0 leave little room to show improvements.**
- **30 questions give wide error bars,** and only 6 of them are multi-hop.

A judge from another provider (`JUDGE_MODEL=anthropic:…`) and harder, human-checked questions would give a more trustworthy number.

## Performance

Measured on 29 October 2026 with `log_report`, on a WSL2 laptop, with `gemini-3.8-flash` as the main model and `gemini-3.5-flash-lite` for the researchers. These are small samples, so treat them as rough guides, not benchmarks.

**Chat turns** (latency is the whole turn; TTFT is time to the first token of the reply; TPOT is time per reply token after that):

| Route | Turns | Latency p50 / p95 | TTFT p50 / p95 | TPOT p50 / p95 |
|---|---|---|---|---|
| Research (new topic, all 4 sources) | 4 | 63 s / 111 s | 51 s / 100 s | 7.4 / 8.8 ms |
| Answer (follow-up, no search) | 14 | 15 s / 164 s* | 14 s / 164 s* | 6.4 / 11.4 ms |

\* The follow-up p95 comes from the WSL network stall described under [Troubleshooting](#troubleshooting): the router waited 60–120 s for a connection. Without stalls, follow-ups take 4–15 s.

**Time per step**, across all turns:

| Step | p50 | p95 |
|---|---|---|
| router | 3.4 s | 61 s* |
| planner | 3.2 s | 5.6 s |
| arXiv / GitHub / YouTube researcher | 5.7 / 6.3 / 6.3 s | 9.0 / 8.0 / 9.7 s |
| notes researcher | 10.6 s | 14.8 s |
| synthesizer | 42 s | 93 s |
| answer | 10 s | 104 s* |

The four researchers run in parallel, so the research phase takes about as long as the slowest one, roughly 10–15 s. Most of a research turn is the synthesizer reasoning before it writes, which accounts for about 90% of its output tokens.

**Indexing** (`FILE_WORKERS=3`, `EMBED_WORKERS=4`): in the first 90 minutes of a full run, 644 files were processed: 640 indexed, 4 with no text, none failed.
- p50: 5.8 s per file. Most are short papers and notes.
- p95: 158 s per file.
- max: 286 s per file, for long PDFs with many figures sent for captioning.

Files are processed smallest first, so the files left at the end take longest. Daily incremental runs only touch changed files and usually finish in about a minute.

## Linting

Configured in `pyproject.toml`: [ruff](https://docs.astral.sh/ruff/), with a 120-character line limit. Besides style, it checks for common bugs, unused or unsorted imports, and outdated syntax.

```bash
ruff check .          # report problems
ruff check . --fix    # apply the safe automatic fixes
ruff format --diff .  # preview formatting changes (ruff format . applies them)
```

## Project layout

| Path | What it does |
|---|---|
| `ui.py` | Streamlit app |
| `ml_researcher/graph.py` | the agent graph: router, planner, guarded researchers, reviewer and follow-ups, Python checks, synthesizer, answer |
| `ml_researcher/cache.py` | prompt caching per provider: Gemini explicit caches (only above Vertex's 4,096-token minimum) or Claude `cache_control` breakpoints, for the findings shared by reviewer and synthesizer and the conversation up to the latest brief |
| `ml_researcher/guards.py` | hashes for tool calls, results and follow-ups (repeat and progress detection) |
| `ml_researcher/tools.py` | arXiv, notes, GitHub and YouTube search tools |
| `ml_researcher/runner.py` | runs one turn: streaming, cancellation, telemetry |
| `ml_researcher/chat.py` | terminal chat |
| `ml_researcher/llm.py` | provider-agnostic chat models: `chat_model("provider:model", effort)`, one `effort` setting mapped to each provider|
| `ml_researcher/sandbox.py` | hosted Python sandboxes behind one interface (Gemini or Claude code execution) |
| `ml_researcher/ratelimit.py` | one shared pace for all model calls; batch jobs drop to `BATCH_RPS` |
| `ml_researcher/ingest.py` | indexing pipeline (LangGraph map-reduce over files) |
| `ml_researcher/readers.py` | PDF, DOCX, PPTX and Markdown readers; chunking |
| `ml_researcher/vision.py` | image captions with any multimodal model (`VISION_MODEL`), with cache |
| `ml_researcher/embeddings.py` | Gemini embeddings, with retry |
| `ml_researcher/store_chroma.py`, `hybrid.py` | local vector store, BM25 keyword search, rank fusion |
| `ml_researcher/rerank.py` | Cohere reranker: 15 fused candidates down to the top 5 |
| `ml_researcher/store_rag.py`, `vectordb.py` | Vertex AI RAG Engine backend; backend switch |
| `ml_researcher/render.py` | converts model Markdown and LaTeX into what Streamlit renders |
| `ml_researcher/logging_config.py`, `log_report.py` | logging setup; local log summary |
| `scripts/setup_gcp_metrics.py` | Cloud Monitoring metrics and dashboard |

All runtime data lives in `~/.ml_researcher/`: the vector database, saved conversations, the caption cache and logs. Nothing is written to the project folder.

## Troubleshooting

- **The first request after the app has been idle stalls for 60 or 120 s (WSL).** The stall happens while connecting over IPv6. Make the system prefer IPv4:
  ```bash
  echo "precedence ::ffff:0:0/96  100" | sudo tee -a /etc/gai.conf
  ```
- **`403 PERMISSION_DENIED`:** check `GOOGLE_CLOUD_PROJECT` and that the Vertex AI API is enabled.
- **Indexing is slow or runs out of memory:** close VS Code; its language server can use several GB of WSL's memory. Lower `FILE_WORKERS` if you see many `429` (rate limit) errors.
- **`chroma browse` shows nothing:** it can't read the database while an ingest is writing to it. Use `inspect_db.py`, or wait for the ingest to finish.
