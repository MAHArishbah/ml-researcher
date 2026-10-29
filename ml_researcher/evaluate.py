"""Offline evaluation of notes retrieval and of answers grounded in it.

    python -m ml_researcher.evaluate build --root /mnt/e/AI_ML/ML --n 100   # golden set (LLM-written Q/A)
    python -m ml_researcher.evaluate retrieval                               # recall/precision/MRR/DCG/nDCG @k
    python -m ml_researcher.evaluate generation --limit 30                   # DeepEval: faithfulness, relevancy,
                                                                             #   correctness
Golden items come from random indexed files under --root:
  single-hop: a question written from one chunk;
  multi-hop:  a question that needs one chunk AND its closest chunk from a *different* file.
Relevance is chunk-level (identical copies of a gold chunk elsewhere count too) and, more leniently, file-level.
The golden set (evals/golden.jsonl) and run results (evals/runs/<time>-<kind>/) are kept in the project.
"""
import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from tqdm import tqdm

from . import config, rerank, store_chroma
from .config import DOCS_DIR, MODEL
from .llm import gemini, text_of
from .readers import to_wsl_path

EVAL_DIR = Path(__file__).resolve().parent.parent / "evals"      # golden set + runs/, kept in the project
GOLDEN = EVAL_DIR / "golden.jsonl"
KS = (1, 3, 5, 10)
MIN_CHARS = 500                  # skip tiny chunks (titles, captions) as question sources
DUPLICATE_SIM = 0.97             # a multi-hop partner this similar is a copy, not a second hop
WORKERS = 4                      # default parallel LLM/search calls (--workers)

WRITER = gemini(MODEL, thinkin_level="low")   # writes golden Q/A and the answers under test
JUDGE = gemini(MODEL, thinkin_level="low")    # DeepEval judge


# ---------------------------------------------------------------- golden set

class QA(BaseModel):
    skip: bool = Field(description="true if no good question can be written (table of contents, references, "
                                   "index, boilerplate, code only, too fragmentary, or passages unrelated)")
    question: str = Field(description="the question (empty string only if skip)")
    answer: str = Field(description="reference answer to the question in 1-3 sentences (empty string only if skip)")


RULES = """- Ask it as a standalone question a student revising machine learning would type into a search box.
- Never refer to "the passage", "the text", "the notes", "the document" or the author.
- Paraphrase: do not copy distinctive phrases from the passage(s) into the question.
- Name the concept(s) involved so the question is unambiguous without the passage.
- The reference answer is 1-3 sentences and fully supported by the passage(s)."""

SINGLE_PROMPT = f"""You write evaluation questions for a search engine over a student's ML notes and books.
From the passage, write ONE question answerable from this passage alone, plus its reference answer.
{RULES}
Set skip=true if the passage is a table of contents, references, an index, boilerplate, code only,
or too fragmentary to support a meaningful question."""

MULTI_PROMPT = f"""You write multi-hop evaluation questions for a search engine over a student's ML notes and books.
You get two passages from DIFFERENT documents. Write ONE question whose answer needs facts from BOTH passages
(e.g. compare them, connect them, or apply one to the other), plus a reference answer that uses both.
{RULES}
Set skip=true if the passages can't be meaningfully combined into one question."""


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()


def _load_chunks() -> tuple[list, dict]:
    """All chunks in the index, plus text-hash -> ids (to treat identical copies of a chunk as equivalent)."""
    data = store_chroma.get_all(["documents", "metadatas"])
    chunks, twins = [], defaultdict(list)
    for i, doc, meta in zip(data["ids"], data["documents"], data["metadatas"], strict=True):
        chunks.append({"id": i, "text": doc, **meta})
        twins[_text_hash(doc)].append(i)
    return chunks, twins


def _partner(chunk: dict) -> tuple:
    """Closest chunk from another file (the second hop), skipping near-copies. Returns (chunk, similarity)."""
    col = store_chroma._collection()
    emb = col.get(ids=[chunk["id"]], include=["embeddings"])["embeddings"][0]
    res = col.query(query_embeddings=[emb], n_results=10, where={"path": {"$ne": chunk["path"]}},
                    include=["documents", "metadatas", "distances"])
    for i, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0],
                                  strict=True):
        sim = 1 - dist
        if (sim < DUPLICATE_SIM and len(doc) >= MIN_CHARS and meta["name"] != chunk["name"]   # same name = copy
                and _text_hash(doc) != _text_hash(chunk["text"])):
            return {"id": i, "text": doc, **meta}, round(sim, 3)
    return None, None


def _write_item(cand: dict) -> dict | None:
    """Ask the LLM for a Q/A pair from the candidate's chunk(s); None if it declines or fails."""
    hops = cand["chunks"]
    if len(hops) == 1:
        msgs = [SystemMessage(SINGLE_PROMPT), HumanMessage(hops[0]["text"])]
    else:
        msgs = [SystemMessage(MULTI_PROMPT),
                HumanMessage(f"Passage A:\n{hops[0]['text']}\n\nPassage B:\n{hops[1]['text']}")]
    qa = None
    for attempt in (1, 2):                                   # occasional empty/invalid JSON: retry once
        try:
            qa = WRITER.with_structured_output(QA).invoke(msgs)
            break
        except Exception as e:
            if attempt == 2:
                tqdm.write(f"  writer failed: {type(e).__name__}: {str(e)[:120]}")
                return None
    if qa is None or qa.skip or not qa.question.strip() or not qa.answer.strip():
        return None
    return {"type": "single" if len(hops) == 1 else "multi_hop",
            "question": qa.question.strip(), "reference_answer": qa.answer.strip(),
            "evidence": [{"id": h["id"], "path": h["path"], "name": h["name"], "chunk": h["chunk"],
                          "twins": h["twins"], "text": h["text"]} for h in hops],
            **({"hop_similarity": cand["sim"]} if len(hops) > 1 else {})}


def build(args):
    out = Path(args.out)
    if out.exists() and not args.force:
        sys.exit(f"{out} exists; pass --force to overwrite (or --out elsewhere)")
    root = str(to_wsl_path(args.root)).rstrip("/") + "/"
    rng = random.Random(args.seed)
    chunks, twins = _load_chunks()
    by_file = defaultdict(list)
    for c in chunks:
        if c["path"].startswith(root) and len(c["text"]) >= MIN_CHARS:
            c["twins"] = twins[_text_hash(c["text"])]
            by_file[c["path"]].append(c)
    files = sorted(by_file)
    rng.shuffle(files)
    if not files:
        sys.exit(f"no indexed chunks under {root}")
    n_multi = round(args.n * args.multi_hop)
    n_single = args.n - n_multi
    print(f"{len(chunks)} chunks indexed, {len(files)} files under {root} | "
          f"target {n_single} single-hop + {n_multi} multi-hop")

    # spare candidates: the writer skips unsuitable chunks (TOCs, references, ...) and unrelated pairs (more often)
    n_single_cands, n_multi_cands = math.ceil(n_single * 1.6), math.ceil(n_multi * 3)
    picks, seen = [], set()
    for i in range(n_single_cands + n_multi_cands):
        c = rng.choice(by_file[files[i % len(files)]])   # one random chunk per random file (files reused if few)
        if c["id"] not in seen:
            seen.add(c["id"])
            picks.append(c)
    cands = [{"chunks": [c]} for c in picks[:n_single_cands]]
    for c in tqdm(picks[n_single_cands:], desc="finding second hops", dynamic_ncols=True):
        p, sim = _partner(c)
        if p:
            p["twins"] = twins[_text_hash(p["text"])]
            cands.append({"chunks": [c, p], "sim": sim})

    with ThreadPoolExecutor(args.workers) as ex:
        items = list(tqdm(ex.map(_write_item, cands), total=len(cands), desc="writing Q/A", dynamic_ncols=True))
    singles = [it for it in items if it and it["type"] == "single"][:n_single]
    multis = [it for it in items if it and it["type"] == "multi_hop"][:n_multi]
    golden = singles + multis
    rng.shuffle(golden)                                      # so --limit N still gets a mix of both types
    for k, it in enumerate(golden, 1):
        it["qid"] = f"q{k:03d}"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for it in golden:
            f.write(json.dumps({"qid": it.pop("qid"), **it}, ensure_ascii=False) + "\n")
    print(f"wrote {len(singles)} single-hop + {len(multis)} multi-hop questions to {out}")
    if len(golden) < args.n:
        print("  (fewer than requested: the writer skipped many chunks; re-run with a larger --n or another --seed)")


def load_golden(path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------- retrieval metrics

def score_ranking(matches: list, n_gold: int, ks=KS) -> dict:
    """matches[r] = index of the gold evidence that result r+1 matches, or None if irrelevant.

    recall@k     distinct gold items found in top k / all gold items
    precision@k  relevant results in top k / k
    hit@k        at least one gold item in top k
    mrr          1 / rank of the first relevant result (0 if none)
    dcg@k        sum over distinct gold items found of 1/log2(rank+1);  ndcg@k = dcg / ideal dcg
    multi-hop only: all@k = every hop found in top k;  mrr_all = 1 / rank at which the last hop appears
    """
    first = {}
    for rank, g in enumerate(matches, 1):
        if g is not None and g not in first:
            first[g] = rank
    out = {"mrr": 1 / min(first.values()) if first else 0.0}
    if n_gold > 1:
        out["mrr_all"] = 1 / max(first.values()) if len(first) == n_gold else 0.0
    for k in ks:
        ranks = [r for r in first.values() if r <= k]
        dcg = sum(1 / math.log2(r + 1) for r in ranks)
        ideal = sum(1 / math.log2(r + 1) for r in range(1, min(n_gold, k) + 1))
        out[f"recall@{k}"] = len(ranks) / n_gold
        out[f"precision@{k}"] = sum(g is not None for g in matches[:k]) / k
        out[f"hit@{k}"] = float(bool(ranks))
        out[f"dcg@{k}"] = dcg
        out[f"ndcg@{k}"] = dcg / ideal
        if n_gold > 1:
            out[f"all@{k}"] = float(len(ranks) == n_gold)
    return out


def _judge_results(item: dict, results: list) -> dict:
    """Score one ranked result list at chunk level and file level."""
    chunk_of = {t: g for g, ev in enumerate(item["evidence"]) for t in ev["twins"] + [ev["id"]]}
    paths = list(dict.fromkeys(ev["path"] for ev in item["evidence"]))   # distinct gold files, in order
    file_of = {p: g for g, p in enumerate(paths)}
    return {"chunk": score_ranking([chunk_of.get(r["id"]) for r in results], len(item["evidence"])),
            "file": score_ranking([file_of.get(r["path"]) for r in results], len(paths))}


def _mean(rows: list) -> dict:
    keys = {k for r in rows for k in r}
    return {k: sum(r[k] for r in rows if k in r) / sum(k in r for r in rows) for k in sorted(keys)}


def _print_table(title: str, summary: dict, modes: list):
    groups = ["all", "single", "multi_hop"]
    cols = [(m, g) for m in modes for g in groups]
    print(f"\n{title}")
    print(f"{'':<14}" + "".join(f"{m[:6] + '/' + {'all': 'all', 'single': '1hop', 'multi_hop': 'mhop'}[g]:>13}"
                                for m, g in cols))
    order = ["mrr", "mrr_all"] + [f"{m}@{k}" for m in ("hit", "recall", "precision", "ndcg", "dcg", "all")
                                  for k in KS]
    for key in order:
        vals = [summary[m][g].get(key) for m, g in cols]
        if any(v is not None for v in vals):
            print(f"{key:<14}" + "".join(f"{'-' if v is None else f'{v:.3f}':>13}" for v in vals))


def retrieval(args):
    golden = load_golden(args.golden)[: args.limit]
    modes = args.modes.split(",")
    if "rerank" in modes and not rerank.available():
        print("COHERE_API_KEY not set: skipping the rerank mode")
        modes.remove("rerank")
    run = _run_dir("retrieval")
    per_query, summary = [], {}
    kmax = max(KS)
    for mode in modes:
        store_chroma.HYBRID = mode in ("hybrid", "rerank")   # search() reads these module flags per call
        store_chroma.RERANK = mode == "rerank"
        if store_chroma.HYBRID:
            store_chroma._keyword_index()                 # build BM25 once up front, not inside the threads
        t0 = time.monotonic()
        with ThreadPoolExecutor(args.workers) as ex:
            ranked = list(tqdm(ex.map(lambda it: store_chroma.search(it["question"], top_k=kmax), golden),
                               total=len(golden), desc=f"retrieval ({mode})", dynamic_ncols=True))
        secs = time.monotonic() - t0
        scores = [_judge_results(it, res) for it, res in zip(golden, ranked, strict=True)]
        for it, res, sc in zip(golden, ranked, scores, strict=True):
            per_query.append({"qid": it["qid"], "type": it["type"], "mode": mode, "question": it["question"],
                              "retrieved": [f"{r['name']}#{r['id'].split(':')[1]}" for r in res], **sc})
        summary[mode] = {"seconds_per_query": round(secs / max(len(golden), 1), 3)}
        for level in ("chunk", "file"):
            summary[mode][level] = {
                g: _mean([sc[level] for it, sc in zip(golden, scores, strict=True) if g in ("all", it["type"])])
                for g in ("all", "single", "multi_hop")}
    store_chroma.HYBRID, store_chroma.RERANK = config.HYBRID, config.RERANK

    n_multi = sum(it["type"] == "multi_hop" for it in golden)
    print(f"\n{len(golden)} questions ({len(golden) - n_multi} single-hop, {n_multi} multi-hop) | "
          + " | ".join(f"{m}: {summary[m]['seconds_per_query']}s/query" for m in modes))
    for level, title in (("chunk", "CHUNK level (the exact passage, or an identical copy, must be retrieved)"),
                         ("file", "FILE level (any chunk from the right file counts)")):
        _print_table(title, {m: summary[m][level] for m in modes}, modes)
    _save(run, per_query, summary, vars(args))


# ---------------------------------------------------------------- generation metrics (DeepEval)

ANSWER_PROMPT = """Answer the question using ONLY the numbered excerpts from the user's notes below.
Be concise (2-5 sentences), keep LaTeX math as $...$, and say so if the excerpts don't contain the answer."""


def _answer(item: dict, top_k: int) -> dict:
    hits = store_chroma.search(item["question"], top_k=top_k)
    contexts = [h["text"] for h in hits]                  # exactly what the agent's notes tool sees
    notes = "\n\n".join(f"[{n}] ({h['name']})\n{h['text']}" for n, h in enumerate(hits, 1))
    reply = WRITER.invoke([SystemMessage(ANSWER_PROMPT),
                           HumanMessage(f"Excerpts:\n{notes}\n\nQuestion: {item['question']}")])
    return {"contexts": contexts, "answer": text_of(reply)}


def _deepeval_judge():
    from deepeval.models import DeepEvalBaseLLM

    class GeminiJudge(DeepEvalBaseLLM):
        """DeepEval judge backed by the project's Vertex AI Gemini chat model."""

        def __init__(self, llm):
            self.llm = llm
            super().__init__(llm.model)

        def load_model(self):
            return self.llm

        def generate(self, prompt: str, schema=None):
            if schema is not None:
                try:
                    out = self.llm.with_structured_output(schema).invoke(prompt)
                    if out is not None:
                        return out
                except Exception:
                    pass                                     # fall back to raw JSON text, parsed by DeepEval
            return text_of(self.llm.invoke(prompt))

        async def a_generate(self, prompt: str, schema=None):
            return self.generate(prompt, schema)

        def get_model_name(self):
            return f"vertex/{self.llm.model}"

    return GeminiJudge(JUDGE)


def _metrics(judge) -> dict:
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
    from deepeval.test_case import SingleTurnParams

    common = {"model": judge, "async_mode": False, "include_reason": True}
    return {
        "faithfulness": FaithfulnessMetric(**common),          # claims in the answer supported by the contexts
        "answer_relevancy": AnswerRelevancyMetric(**common),   # answer statements relevant to the question
        "answer_correctness": GEval(                           # agreement with the reference answer
            name="Answer correctness", model=judge, async_mode=False,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT,
                               SingleTurnParams.EXPECTED_OUTPUT],
            evaluation_steps=[
                "Compare the facts in the actual output with the expected output.",
                "Penalise facts that contradict the expected output heavily.",
                "Penalise missing key facts from the expected output; extra correct detail is fine.",
                "Ignore wording, formatting and LaTeX notation differences."]),
    }


def _evaluate_one(item: dict, top_k: int, judge) -> dict:
    from deepeval.test_case import LLMTestCase

    row = {"qid": item["qid"], "type": item["type"], "question": item["question"]}
    try:
        row.update(_answer(item, top_k))
    except Exception as e:
        return {**row, "error": f"answer: {type(e).__name__}: {e}"}
    case = LLMTestCase(input=item["question"], actual_output=row["answer"],
                       expected_output=item["reference_answer"], retrieval_context=row["contexts"])
    for name, metric in _metrics(judge).items():               # fresh metric objects: they hold state
        try:
            metric.measure(case, _show_indicator=False)
            row[name] = metric.score
            row[f"{name}_reason"] = metric.reason
        except Exception as e:
            row[name] = None
            row[f"{name}_reason"] = f"error: {type(e).__name__}: {str(e)[:200]}"
    return row


def generation(args):
    golden = load_golden(args.golden)[: args.limit]
    judge = _deepeval_judge()
    run = _run_dir("generation")
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(tqdm(ex.map(lambda it: _evaluate_one(it, args.top_k, judge), golden),
                         total=len(golden), desc="answer + judge", dynamic_ncols=True))
    names = ["faithfulness", "answer_relevancy", "answer_correctness"]
    summary = {}
    for g in ("all", "single", "multi_hop"):
        sub = [r for r in rows if g in ("all", r["type"])]
        summary[g] = {"n": len(sub)}
        for name in names:
            vals = [r[name] for r in sub if r.get(name) is not None]
            summary[g][name] = round(sum(vals) / len(vals), 3) if vals else None
            summary[g][f"{name}_pass_rate"] = round(sum(v >= 0.5 for v in vals) / len(vals), 3) if vals else None
    summary["errors"] = sum("error" in r or any(r.get(n) is None for n in names) for r in rows)

    print(f"\n{len(rows)} questions | top_k={args.top_k} | judge {JUDGE.model} | errors {summary['errors']}")
    print(f"{'metric':<22}{'all':>10}{'1-hop':>10}{'multi-hop':>11}   (mean score, pass rate at 0.5)")
    for name in names:
        cells = [summary[g][name] for g in ("all", "single", "multi_hop")]
        rates = [summary[g][f"{name}_pass_rate"] for g in ("all", "single", "multi_hop")]
        print(f"{name:<22}" + "".join(f"{'-' if v is None else f'{v:.3f}':>10}" for v in cells)
              + "   pass " + " / ".join("-" if r is None else f"{r:.0%}" for r in rates))
    _save(run, rows, summary, vars(args))


# ---------------------------------------------------------------- helpers + CLI

def _run_dir(kind: str) -> Path:
    d = EVAL_DIR / "runs" / f"{time.strftime('%Y%m%d-%H%M%S')}-{kind}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save(run: Path, rows: list, summary: dict, params: dict):
    with (run / "per_query.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (run / "summary.json").write_text(json.dumps({"params": params, "summary": summary}, indent=2, default=str))
    print(f"\nsaved {run}/summary.json and per_query.jsonl")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="create the golden question set")
    b.add_argument("--root", default=str(DOCS_DIR), help="sample files under this folder")
    b.add_argument("--n", type=int, default=100, help="number of questions")
    b.add_argument("--multi-hop", type=float, default=0.25, help="fraction of multi-hop questions")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--out", default=str(GOLDEN))
    b.add_argument("--force", action="store_true", help="overwrite an existing golden set")
    for name, helptext in (("retrieval", "retrieval metrics"), ("generation", "DeepEval answer metrics")):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--golden", default=str(GOLDEN))
        p.add_argument("--limit", type=int, default=None, help="only the first N questions")
    for p in sub.choices.values():
        p.add_argument("--workers", type=int, default=WORKERS, help="parallel LLM / search calls")
    sub.choices["retrieval"].add_argument("--modes", default="rerank,hybrid,vector",
                                          help="comma list of: rerank (hybrid + Cohere), hybrid, vector")
    sub.choices["generation"].add_argument("--top-k", type=int, default=5, help="passages given to the answerer")
    args = ap.parse_args()
    {"build": build, "retrieval": retrieval, "generation": generation}[args.cmd](args)


if __name__ == "__main__":
    main()
