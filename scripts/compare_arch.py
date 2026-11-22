"""single agent vs multi-agent on eval_v2, once each, judged by an adoption rule fixed before the run.

    python -m scripts.compare_arch                      # both architectures on eval_v2 (45 queries each)
    python -m scripts.compare_arch --archs multi        # one architecture; the other's earlier result is kept
    python -m scripts.compare_arch --limit 6            # quick check (reports/smoke/, not for the report)
    LLM_BACKEND=mock python -m scripts.compare_arch     # offline pipeline check (numbers meaningless)

Writes reports/arch_comparison_v2.json and docs/multi_agent.md (Team notes kept). LLM_CACHE=1 makes a rerun free; the two
architectures send different prompts, so neither replays the other's answers.

Adoption rule (D44, fixed before either architecture was run on eval_v2): adopt the multi-agent architecture only if ALL hold
  1. accuracy rises by at least 0.05 (3 of 45 queries; one query is 2.2 points, so 1-2 queries is noise);
  2. citation precision is not lower by more than 0.02;
  3. refusal accuracy and injection resistance are not lower (safety must not regress);
  4. p95 latency stays under the 60 s target;
  5. median tokens per query are at most 2.5x the single agent's.
Otherwise the single agent stays the default and the multi-agent graph stays in the code as a measured alternative.
"""
from __future__ import annotations

import argparse
import json
import os
import time

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("EVAL_FILE", "eval_v2.jsonl")      # before the eval scripts read it

from app import llm_client  # noqa: E402
from app.retrieval.chunking import ROOT  # noqa: E402
from scripts.agent_eval import evaluate, load_eval, run_one, warm_up  # noqa: E402
from scripts.grade import eval_tag  # noqa: E402

DOC = ROOT / "docs" / "multi_agent.md"
RULE = {"accuracy_gain": 0.05, "citation_precision_drop": 0.02, "p95_ms": 60_000, "token_ratio": 2.5}
LABEL = {"single": "Single agent (plan-and-execute, D30-D43)", "multi": "Multi-agent (supervisor + 3 specialists + verifier, D44)"}
COLS = [("accuracy", "Accuracy", "{:.3f}"), ("task_completion_rate", "Completion", "{:.0%}"), ("fact_recall", "Fact recall", "{:.2f}"),
        ("citation_hit_rate", "Citation hit", "{:.2f}"), ("citation_precision", "Citation precision", "{:.2f}"),
        ("refusal_accuracy", "Refusals", "{:.2f}"), ("injection_resistance", "Injection reported", "{:.2f}"),
        ("avg_tool_calls_per_task", "Tool calls", "{:.2f}"), ("tokens_median", "Tokens (median)", "{:,}"),
        ("latency_p50_ms", "p50", "{}"), ("latency_p95_ms", "p95", "{}")]


def run_arch(arch: str, qs: list[dict]) -> dict:
    os.environ["AGENT_ARCH"] = arch
    warm_up()
    runs = []
    for i, q in enumerate(qs, 1):
        r = run_one(q)
        o = r.get("output") or {}
        print(f"  [{arch} {i}/{len(qs)}] {q['id']} {q['category']:<28} -> {o.get('behaviour', 'CRASH'):<8} steps={r['steps']:<2} "
              f"llm={r.get('llm_calls', 0)} tokens={r['tokens']}", flush=True)
        runs.append(r)
    res = evaluate(qs, runs, f"{eval_tag()} · {arch}")
    res["label"], res["run_at"] = LABEL[arch], time.strftime("%Y-%m-%d %H:%M")
    return res


def verdict(report: dict) -> dict:
    s, m = (report["archs"].get(k, {}).get("metrics") for k in ("single", "multi"))
    if not (s and m):
        return {"decision": "incomplete", "checks": {}, "reason": "run both architectures first"}
    n = lambda x: x or 0                                           # noqa: E731
    checks = {
        "accuracy gain >= 0.05": n(m["accuracy"]) - n(s["accuracy"]) >= RULE["accuracy_gain"] - 1e-9,
        "citation precision not lower by > 0.02": n(m["citation_precision"]) >= n(s["citation_precision"]) - RULE["citation_precision_drop"] - 1e-9,
        "refusals not lower": n(m["refusal_accuracy"]) >= n(s["refusal_accuracy"]),
        "injection resistance not lower": n(m["injection_resistance"]) >= n(s["injection_resistance"]),
        "p95 latency < 60 s": n(m["latency_p95_ms"]) < RULE["p95_ms"],
        "tokens <= 2.5x single": n(m["tokens_median"]) <= RULE["token_ratio"] * max(1, n(s["tokens_median"])),
    }
    ok = all(checks.values())
    return {"decision": "adopt multi-agent" if ok else "keep single agent", "checks": checks,
            "accuracy_delta": round(n(m["accuracy"]) - n(s["accuracy"]), 3),
            "token_ratio": round(n(m["tokens_median"]) / max(1, n(s["tokens_median"])), 2)}


def fmt(key, val, spec):
    if val is None:
        return "–"
    if key.startswith("latency"):
        return f"{val / 1000:.1f} s"
    try:
        return spec.format(val)
    except (ValueError, TypeError):
        return str(val)


def write_markdown(report: dict):
    v = report["verdict"]
    L = ["# Single agent vs multi-agent (D44)", "",
         f"Model `{report['model']}` · eval_v2 ({report['queries']} held-out queries, frozen before the multi-agent code; "
         "`data/evaluation/CHANGELOG.md`) · one run per architecture · generated by `python -m scripts.compare_arch`.", "",
         "Both architectures share the tools, rubric engine, retrieval, refusal rules, writer, code critic and finalize step; "
         "only the orchestration differs (`app/agents/multi_agent.py`).", "",
         "| Architecture | " + " | ".join(c[1] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for k in ("single", "multi"):
        r = report["archs"].get(k)
        if r:
            L.append(f"| {r['label']} | " + " | ".join(fmt(c, r["metrics"].get(c), s) for c, _, s in COLS) + " |")
    cats = sorted({c for r in report["archs"].values() for c in r["metrics"]["by_category"]})
    if cats:
        L += ["", "By category (passed / queries):", "", "| Category | " + " | ".join(k for k in ("single", "multi") if k in report["archs"]) + " |",
              "|---|" + "---|" * len(report["archs"])]
        for c in cats:
            L.append(f"| {c} | " + " | ".join(report["archs"][k]["metrics"]["by_category"].get(c, "–")
                                              for k in ("single", "multi") if k in report["archs"]) + " |")
        L += ["", "Failure classes (primary class per failed query):", "",
              "| Architecture | Tool-call error | Retrieval miss | Hallucination | Reasoning error |", "|---|---|---|---|---|"]
        for k in ("single", "multi"):
            if k in report["archs"]:
                t = report["archs"][k]["metrics"]["failure_taxonomy"]
                L.append(f"| {k} | {t['tool-call error']} | {t['retrieval miss']} | {t['hallucination']} | {t['reasoning error']} |")
    L += ["", "## Adoption rule (fixed before the run)", "",
          "| Check | Result |", "|---|---|"] + [f"| {k} | {'pass' if ok else 'FAIL'} |" for k, ok in v["checks"].items()]
    L += ["", f"**Decision: {v['decision']}**" + (f" (accuracy {v['accuracy_delta']:+.3f}, tokens x{v['token_ratio']})"
                                                   if "accuracy_delta" in v else f" ({v.get('reason', '')})"), "",
          "## Team notes", "", "*(Add the reading of the results here; the generator keeps everything below this heading.)*", ""]
    old = DOC.read_text(encoding="utf-8") if DOC.exists() else ""
    text = "\n".join(L)
    if "## Team notes" in old:
        text = text[:text.index("## Team notes")] + old[old.index("## Team notes"):]
    DOC.write_text(text, encoding="utf-8", newline="\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", default="single,multi")
    ap.add_argument("--limit", type=int, default=0, help="first N queries spread over categories (a quick check)")
    a = ap.parse_args(argv)
    archs = [x.strip() for x in a.archs.split(",") if x.strip()]
    if not set(archs) <= {"single", "multi"}:
        raise SystemExit("--archs takes single and/or multi")
    qs = load_eval()
    if a.limit:
        qs = qs[:: max(1, len(qs) // a.limit)][: a.limit]
    mock = llm_client.backend() == "mock"
    model = "mock" if mock else os.getenv("LLM_MODEL", "?")
    path = ROOT / "reports" / ("mock" if mock else "smoke" if a.limit else "") / f"arch_comparison_{eval_tag()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"model": model, "eval_set": eval_tag(), "queries": len(qs), "rule": RULE, "archs": {}}
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            if old.get("model") == model and old.get("queries") == len(qs):
                report["archs"] = old.get("archs", {})
        except ValueError:
            pass
    print(f"{len(archs)} architecture(s) x {len(qs)} {eval_tag()} queries. The multi-agent run costs roughly 2-3x the single "
          f"agent's tokens. LLM_CACHE={os.getenv('LLM_CACHE', '0')}.", flush=True)
    for arch in archs:
        print(f"\n== {arch}: {LABEL[arch]}", flush=True)
        report["archs"][arch] = run_arch(arch, qs)
        m = report["archs"][arch]["metrics"]
        print(f"   accuracy {m['accuracy']} | completion {m['task_completion_rate']} | citation precision {m['citation_precision']} | "
              f"tokens median {m['tokens_median']:,} | p95 {(m['latency_p95_ms'] or 0) / 1000:.1f}s | taxonomy {m['failure_taxonomy']}", flush=True)
        report["verdict"] = verdict(report)
        path.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str), encoding="utf-8", newline="\n")
    print(f"\nverdict: {report['verdict']['decision']}  {report['verdict'].get('checks', '')}")
    if not (mock or a.limit):
        write_markdown(report)
        print(f"wrote {path.relative_to(ROOT)} and {DOC.relative_to(ROOT)}")
    else:
        print(f"wrote {path.relative_to(ROOT)} (check run; docs/multi_agent.md is written by full runs only)")


if __name__ == "__main__":
    main()
