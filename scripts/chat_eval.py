"""Evaluate Ask the Copilot (D45) on data/evaluation/chat_eval_v1.jsonl: accuracy, grounding and a failure taxonomy.

    python -m scripts.chat_eval                      # dev split -> reports/chat_eval_v1.json + docs/chat_eval.md
    python -m scripts.chat_eval --split test         # the held-out split, ONCE (as D12 does for the agent)
    python -m scripts.chat_eval --split all --ids CHAT-031,CHAT-032
    LLM_BACKEND=mock python -m scripts.chat_eval     # offline pipeline check: reports/mock/ (numbers mean nothing for LLM text)

Per query: route, behaviour (and refusal category), gold facts in the answer (or its table), the exact vendor set for
portfolio questions, valid citations (every cited id exists), a gold source among the citations (system / policy
questions), retrieval recall@6 measured separately from the answer (a gold source among the passages retrieved, before
the writer), and an independent hallucination check: every number in an answer must appear in the full text of the
sources it cites (documents) or in the tool data (computed answers). A query passes when route, behaviour, facts,
vendor set and citations are all right. Failed queries get one class: routing error, tool-call error, retrieval miss,
hallucination, reasoning error. Cost and latency per query come from the turn itself.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from app import llm_client
from app.retrieval.chunking import CORPUS, ROOT

load_dotenv()
EVAL = ROOT / "data" / "evaluation" / "chat_eval_v1.jsonl"


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def _alts(fact: str) -> list[str]:
    return [a.strip().lower() for a in fact.split("|") if a.strip()]


def visible_text(turn: dict) -> str:
    return (turn.get("answer") or "") + " " + json.dumps(turn.get("table") or [], ensure_ascii=False)


def source_doc(source_id: str) -> str:
    return source_id.split("#", 1)[0]


class Sources:
    """Full text of every citable source: knowledge-base chunks, corpus documents and tool outputs."""

    def __init__(self, store):
        from app.chatbot import knowledge
        self.kb = knowledge.get_kb(store)
        self.docs = {}
        if CORPUS.exists():
            for line in open(CORPUS, encoding="utf-8"):
                d = json.loads(line)
                self.docs[d["doc_id"]] = d["text"]

    def exists(self, sid: str) -> bool:
        return sid.startswith("tool:") or sid in self.kb.by_id or sid in self.docs

    def text(self, sid: str) -> str:
        if sid in self.kb.by_id:
            return self.kb.by_id[sid].text
        return self.docs.get(sid, "")


def grade(row: dict, turn: dict, src: Sources) -> dict:
    from app.chatbot import grounding
    routes, behaviours = row["expected_route"].split("|"), row["expected_behaviour"].split("|")
    route_ok = turn["route"] in routes
    beh_ok = turn["behaviour"] in behaviours and (row["refusal_category"] is None or turn.get("refusal_category") == row["refusal_category"])
    text = visible_text(turn).lower()
    facts_hit = [any(a in text for a in _alts(f)) for f in row["gold_facts"]]
    vendors = [r.get("vendor_id") for r in (turn.get("table") or []) if r.get("vendor_id")]
    set_ok = True
    if row["gold_vendor_ids"] is not None:
        set_ok = set(vendors) == set(row["gold_vendor_ids"]) if (vendors or row["gold_vendor_ids"]) else True
        if row["gold_count"] == 0:
            set_ok = not vendors
    count_ok = row["gold_count"] is None or len(vendors) == row["gold_count"] or row["gold_count"] == 0 and not vendors
    cits = [c["source_id"] for c in turn.get("citations") or []]
    cit_valid = all(src.exists(c) for c in cits)
    gold_src = set(row.get("gold_sources") or [])
    src_hit = None
    recall = None
    if gold_src and turn["route"] in ("SYSTEM", "POLICY"):
        src_hit = any(source_doc(c) in gold_src or c in gold_src for c in cits)
        cands = (turn.get("retrieval") or {}).get("candidates") or []
        recall = any(source_doc(c) in gold_src or c in gold_src for c in cands)
    method = (turn.get("grounding") or {}).get("method", "")
    ungrounded: list[str] = []
    if turn["behaviour"] == "ANSWER" and method.startswith(("llm", "extractive")):
        evidence = " ".join(src.text(c) for c in cits) + " " + row["turns"][-1]["message"]
        ungrounded = sorted(grounding.numbers(turn.get("answer", "")) - grounding.numbers(evidence) - grounding.SMALL)
    answered_unanswerable = row["category"] == "not_found" and turn["behaviour"] == "ANSWER"
    passed = route_ok and beh_ok and all(facts_hit) and set_ok and count_ok and cit_valid and not ungrounded and not answered_unanswerable
    cls = None
    if not passed:
        if not route_ok:
            cls = "routing error"
        elif turn["behaviour"] == "PARTIAL" and any("tool" in p for p in (turn.get("grounding") or {}).get("problems", [])):
            cls = "tool-call error"
        elif ungrounded or answered_unanswerable:
            cls = "hallucination"
        elif recall is False or (turn["behaviour"] == "NOT_FOUND" and "ANSWER" in behaviours):
            cls = "retrieval miss"
        else:
            cls = "reasoning error"
    return dict(route_ok=route_ok, behaviour_ok=beh_ok, facts=facts_hit, set_ok=set_ok and count_ok, citations_valid=cit_valid,
                source_hit=src_hit, recall_at_6=recall, ungrounded_numbers=ungrounded, passed=passed, failure_class=cls)


def run(rows: list[dict], ctx) -> list[dict]:
    from app.chatbot.engine import ChatSession, ask
    out = []
    for row in rows:
        s = ChatSession()
        for t in row["turns"][:-1]:
            ask(t["message"], s, ctx=ctx)
        t0 = time.time()
        turn = ask(row["turns"][-1]["message"], s, ctx=ctx)
        turn["wall_ms"] = int((time.time() - t0) * 1000)
        out.append(turn)
        print(f"  {row['id']} {row['category']:<10} {turn['route']:<10} {turn['behaviour']:<9} {turn['grounding'].get('method')}", flush=True)
    return out


def metrics(rows, turns, grades) -> dict:
    n = len(rows)
    rate = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    facts = [f for g in grades for f in g["facts"]]
    answered = [(r, t) for r, t in zip(rows, turns) if r["expected_behaviour"].split("|")[0] == "ANSWER"]
    nf_rows = [(r, t) for r, t in zip(rows, turns) if r["category"] == "not_found"]
    refusals = [(r, t) for r, t in zip(rows, turns) if r["category"] == "refusal"]
    lat = [t["latency_ms"] for t in turns]
    llm_turns = [t for t in turns if (t.get("grounding") or {}).get("method", "").startswith(("llm", "extractive", "agent"))]
    cats = Counter(r["category"] for r in rows)
    by_cat = {c: f"{sum(g['passed'] for r, g in zip(rows, grades) if r['category'] == c)}/{k}" for c, k in sorted(cats.items())}
    return {
        "queries": n, "accuracy": rate([g["passed"] for g in grades]),
        "route_accuracy": rate([g["route_ok"] for g in grades]), "behaviour_accuracy": rate([g["behaviour_ok"] for g in grades]),
        "fact_recall": rate(facts), "portfolio_set_accuracy": rate([g["set_ok"] for r, g in zip(rows, grades) if r["gold_vendor_ids"] is not None]),
        "citation_validity": rate([g["citations_valid"] for g in grades]),
        "source_hit_rate": rate([g["source_hit"] for g in grades if g["source_hit"] is not None]),
        "retrieval_recall_at_6": rate([g["recall_at_6"] for g in grades if g["recall_at_6"] is not None]),
        "task_completion_rate": rate([t["behaviour"] != "PARTIAL" for t in turns]),
        "answered_when_answerable": rate([t["behaviour"] == "ANSWER" for r, t in answered]),
        "abstention_on_unanswerable": rate([t["behaviour"] in ("NOT_FOUND", "REFUSE") for r, t in nf_rows]),
        "refusal_accuracy": rate([t["behaviour"] == "REFUSE" and t.get("refusal_category") == r["refusal_category"] for r, t in refusals]),
        "hallucinated_answers": sum(1 for g in grades if g["ungrounded_numbers"] or g["failure_class"] == "hallucination"),
        "grounding_methods": dict(Counter((t.get("grounding") or {}).get("method") for t in turns)),
        "llm_answers_failing_first_check": sum(1 for t in llm_turns if (t.get("grounding") or {}).get("first_attempt_problems")),
        "avg_tool_calls": round(statistics.mean(t.get("tool_calls", 0) for t in turns), 2),
        "avg_llm_calls": round(statistics.mean(t.get("llm_calls", 0) for t in turns), 2),
        "tokens_median": statistics.median(t.get("tokens", 0) for t in turns),
        "latency_p50_ms": pct(lat, 50), "latency_p95_ms": pct(lat, 95),
        "by_category": by_cat, "failure_classes": dict(Counter(g["failure_class"] for g in grades if g["failure_class"])),
    }


def markdown(m: dict, rows, turns, grades, split: str, model: str) -> str:
    lines = [f"# Ask the Copilot: evaluation ({split})", "",
             f"Model `{model}` · `data/evaluation/chat_eval_v1.jsonl` {split} split ({m['queries']} queries) · generated by "
             "`python -m scripts.chat_eval`. Gold answers come from the ground truth and the documents (D45), never from the chatbot.", "",
             "| Metric | Value |", "|---|---|"]
    for k in ("accuracy", "route_accuracy", "behaviour_accuracy", "fact_recall", "portfolio_set_accuracy", "citation_validity",
              "source_hit_rate", "retrieval_recall_at_6", "task_completion_rate", "answered_when_answerable",
              "abstention_on_unanswerable", "refusal_accuracy", "hallucinated_answers", "llm_answers_failing_first_check",
              "avg_tool_calls", "avg_llm_calls", "tokens_median", "latency_p50_ms", "latency_p95_ms"):
        lines.append(f"| {k.replace('_', ' ')} | {m[k]} |")
    lines += ["", "By category (passed / queries): " + ", ".join(f"{c} {v}" for c, v in m["by_category"].items()), "",
              "Grounding method of each answer: " + ", ".join(f"{k} {v}" for k, v in m["grounding_methods"].items()), "",
              "Failure classes: " + (", ".join(f"{k} {v}" for k, v in m["failure_classes"].items()) or "none"), "",
              "## Queries", "", "| id | category | route | behaviour | method | pass | failure |", "|---|---|---|---|---|---|---|"]
    for r, t, g in zip(rows, turns, grades):
        lines.append(f"| {r['id']} | {r['category']} | {t['route']} | {t['behaviour']} | {(t.get('grounding') or {}).get('method')} | "
                     f"{'yes' if g['passed'] else 'no'} | {g['failure_class'] or ''} |")
    lines += ["", "## Team notes", "", "*(Observations here; the generator keeps everything below this heading.)*", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--ids", default="")
    a = ap.parse_args(argv)
    from app.chatbot.engine import clear_cache
    from app.tools.base import ToolContext
    rows = [json.loads(l) for l in open(EVAL, encoding="utf-8")]
    if a.ids:
        rows = [r for r in rows if r["id"] in set(a.ids.split(","))]
    elif a.split != "all":
        rows = [r for r in rows if r["split"] == a.split]
    ctx = ToolContext()
    clear_cache()
    mock = llm_client.backend() == "mock"
    model = "mock" if mock else os.getenv("LLM_MODEL", "?")
    print(f"chat eval: {len(rows)} queries ({a.split}), model {model}", flush=True)
    turns = run(rows, ctx)
    src = Sources(ctx.get_store())
    grades = [grade(r, t, src) for r, t in zip(rows, turns)]
    m = metrics(rows, turns, grades)
    out_dir = ROOT / "reports" / ("mock" if mock else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "" if a.split == "dev" and not a.ids else f"_{a.split}" if not a.ids else "_ids"
    rep = {"model": model, "split": a.split, "metrics": m,
           "queries": [{"id": r["id"], "category": r["category"], "question": r["turns"][-1]["message"], **g,
                        "route": t["route"], "behaviour": t["behaviour"], "method": (t.get("grounding") or {}).get("method"),
                        "answer": t["answer"], "citations": [c["source_id"] for c in t.get("citations") or []],
                        "retrieval": t.get("retrieval"), "tokens": t.get("tokens"), "latency_ms": t.get("latency_ms"),
                        "steps": t.get("steps"), "llm_calls": t.get("llm_calls")} for r, t, g in zip(rows, turns, grades)]}
    (out_dir / f"chat_eval_v1{tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    doc = out_dir / f"chat_eval{tag}.md" if (mock or a.ids) else ROOT / "docs" / f"chat_eval{tag}.md"
    notes = ""
    if doc.exists() and "## Team notes" in doc.read_text(encoding="utf-8"):
        notes = doc.read_text(encoding="utf-8").split("## Team notes", 1)[1].split("\n", 1)[1]
    md = markdown(m, rows, turns, grades, a.split, model)
    if notes.strip():
        md = md.rsplit("## Team notes", 1)[0] + "## Team notes\n" + notes
    doc.write_text(md, encoding="utf-8")
    print(json.dumps({k: v for k, v in m.items() if k not in ("by_category",)}, indent=1))
    print("by category:", m["by_category"])
    print(f"wrote {Path(out_dir / f'chat_eval_v1{tag}.json').relative_to(ROOT)} and {doc.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
