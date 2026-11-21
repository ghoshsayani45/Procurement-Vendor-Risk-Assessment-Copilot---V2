"""The system knowledge base: the project's own documents, chunked by heading and searched with BM25 + dense (D45).

What it holds (CHAT_KB_PATHS adds more globs, comma-separated, relative to the repository root):
  README.md, CONTRIBUTING.md, docs/**/*.md, the data READMEs, app/scoring/rubric.yaml, and
  "live system facts": a short document generated from the running code (tools, refusal categories, budgets, rubric
  weights and bands, retrieval settings, store counts), so questions like "which tools does the copilot have?" are
  answered from the code itself rather than from a document that may be out of date.

Chunking: markdown sections (heading path kept as the chunk's header), packed to ~CHAT_KB_CHUNK_WORDS words at
paragraph, list and table-row boundaries; a table split across chunks repeats its header row, so every chunk is
readable on its own. Chunk ids are "<path>#<section-slug>" so a citation names the file and the section.

Retrieval: BM25 always; dense too when an embedder is configured (CHAT_KB_DENSE=auto|on|off, EMBED_BACKEND as for the
corpus), fused with reciprocal rank fusion. Vectors are cached on disk by content hash. `coverage()` measures how many of
the question's content words the best passage contains: the chatbot does not answer when it is low (abstention).
"""
from __future__ import annotations

import glob
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.chunking import ROOT, Chunk, tokenize
from app.retrieval.retrievers import BM25Retriever, rrf

DEFAULT_PATHS = ["README.md", "CONTRIBUTING.md", "docs/*.md", "docs/experiments/*.md", "data/evaluation/README.md",
                 "data/evaluation/CHANGELOG.md", "data/ground_truth/README.md", "data/manifests/README.md",
                 "data/manifests/SOURCE_REGISTER_README.md", "observability/README.md", "app/scoring/rubric.yaml"]
LIVE_DOC = "system:live-facts"
LIVE_TITLE = "Live system facts (read from the running code)"
# words that say nothing about WHICH passage answers ("how does the copilot ..."): left out of coverage
GENERIC = set("""copilot system project team please tell explain describe know use used uses using work works
much many get got give show list current currently does do did happen happens happened mean means whats what's
about vendor-risk procurement assistant chatbot bot""".split())


def stem(t: str) -> str:
    """Crude suffix stripping so 'retrieves' / 'retrieval' / 'retrieved' meet at 'retriev' (for coverage and support)."""
    for suf in ("ations", "ation", "ings", "ing", "ions", "ion", "ers", "er", "ies", "ied", "es", "ed", "al", "ly", "s"):
        if len(t) - len(suf) >= 4 and t.endswith(suf):
            return t[: -len(suf)]
    return t


def content_terms(text: str) -> set[str]:
    return {stem(t) for t in tokenize(text) if t not in GENERIC and len(t) > 1}


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:60] or "top"


# ------------------------------------------------------------------ chunking
@dataclass
class _Block:
    kind: str          # para | table | code
    lines: list[str]

    @property
    def words(self) -> int:
        return sum(len(l.split()) for l in self.lines)


def _blocks(lines: list[str]) -> list[_Block]:
    out: list[_Block] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            out.append(_Block("code", lines[i:j + 1]))
            i = j + 1
        elif line.lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            out.append(_Block("table", lines[i:j]))
            i = j
        elif not line.strip():
            i += 1
        else:
            j = i
            while j < len(lines) and lines[j].strip() and not lines[j].lstrip().startswith(("|", "```")):
                j += 1
            out.append(_Block("para", lines[i:j]))
            i = j
    return out


def _split_big(b: _Block, target: int) -> list[_Block]:
    """A table longer than the target becomes row groups that each repeat the header; a long paragraph is split at
    sentence ends. A single row is never cut: it is one fact (e.g. one decision)."""
    if b.words <= target * 1.6:
        return [b]
    if b.kind == "table":
        head = b.lines[:2] if len(b.lines) > 1 and re.match(r"^\s*\|[\s:|-]+\|\s*$", b.lines[1]) else b.lines[:1]
        rows, groups, cur = b.lines[len(head):], [], []
        for r in rows:
            if cur and sum(len(x.split()) for x in head + cur + [r]) > target:
                groups.append(cur)
                cur = []
            cur.append(r)
        if cur:
            groups.append(cur)
        return [_Block("table", head + g) for g in groups]
    if b.kind == "para":
        sents = re.split(r"(?<=[.!?])\s+", " ".join(l.strip() for l in b.lines))
        groups, cur = [], []
        for s in sents:
            if cur and len(" ".join(cur + [s]).split()) > target:
                groups.append(cur)
                cur = []
            cur.append(s)
        if cur:
            groups.append(cur)
        return [_Block("para", [" ".join(g)]) for g in groups]
    return [b]


def _pack(blocks: list[_Block], target: int) -> list[list[_Block]]:
    out, cur, words = [], [], 0
    for b in [x for blk in blocks for x in _split_big(blk, target)]:
        if cur and words + b.words > target:
            out.append(cur)
            cur, words = [], 0
        cur.append(b)
        words += b.words
    if cur:
        out.append(cur)
    return out


def _md_sections(text: str) -> tuple[str, list[tuple[str, list[str]]]]:
    """(title, [(heading path, body lines)]) for a markdown file; text before the first heading is its own section."""
    title, stack, sections, body = "", [], [], []
    in_code = False

    def flush():
        if any(l.strip() for l in body):
            path = " > ".join(h for _, h in stack if h != title) or (title or "")
            sections.append((path, list(body)))
        body.clear()
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        m = None if in_code else re.match(r"^(#{1,4})\s+(.+?)\s*#*\s*$", line)
        if m:
            flush()
            level, head = len(m.group(1)), m.group(2).strip()
            if level == 1 and not title:
                title = head
            stack = [(lv, h) for lv, h in stack if lv < level] + [(level, head)]
        else:
            body.append(line)
    flush()
    return title, sections


def _yaml_sections(text: str) -> list[tuple[str, list[str]]]:
    """Top-level keys of a YAML file, each with its comment lines (rubric.yaml documents its numbers in comments)."""
    sections, cur, head = [], [], "header"
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z_][\w-]*):", line)
        if m:
            if any(l.strip() for l in cur):
                sections.append((head, cur))
            head, cur = m.group(1), []
        cur.append(line)
    if any(l.strip() for l in cur):
        sections.append((head, cur))
    return sections


def chunk_file(rel: str, text: str, target: int | None = None) -> list[Chunk]:
    target = target or int(os.getenv("CHAT_KB_CHUNK_WORDS", "180"))
    if rel.endswith((".yaml", ".yml")):
        title = "Scoring rubric (app/scoring/rubric.yaml)" if rel.endswith("rubric.yaml") else rel
        sections = [(h, [l for l in lines]) for h, lines in _yaml_sections(text)]
        groups = [(h, [[_Block("code", lines)]]) for h, lines in sections]
    else:
        title, sections = _md_sections(text)
        title = title or Path(rel).stem.replace("_", " ")
        groups = [(h, _pack(_blocks(lines), target)) for h, lines in sections]
    out: list[Chunk] = []
    seen: dict[str, int] = {}
    for heading, packs in groups:
        for blocks in packs:
            body = "\n\n".join("\n".join(b.lines).strip() for b in blocks).strip()
            if len(body.split()) < 4:
                continue
            base = f"{rel}#{slug(heading)}"
            n = seen[base] = seen.get(base, 0) + 1
            cid = base if n == 1 else f"{base}-{n}"
            out.append(Chunk(chunk_id=cid, doc_id=rel, seq=len(out), text=body, vendor_id=None, slice="system_docs",
                             title=title, strategy="chat_kb", heading=heading, doc_type="system_doc",
                             access_class="internal", status="CURRENT", source_id="SYS",
                             index_text=f"[{title} | {heading}] {body}"))
    return out


# ------------------------------------------------------------------ live facts from the code
def live_facts(store=None) -> list[tuple[str, str]]:
    """(section, text) pairs computed from the code and the store at build time. No secrets, URLs or keys."""
    facts: list[tuple[str, str]] = []
    try:
        from app.tools.base import DEFAULT_AS_OF, describe_tools
        tools = describe_tools()
        facts.append(("Tools", f"The copilot has {len(tools)} read-only tools: " + "; ".join(
            f"{t['name']} ({t['description'].split('. ')[0].strip().rstrip('.')})" for t in tools)
            + ". Every tool call goes through run_tool(), which validates arguments, applies a timeout, traces the call "
              f"and writes a TOOL_CALL audit event. Reviews are judged as of {DEFAULT_AS_OF.isoformat()} by default."))
    except Exception:
        pass
    try:
        from app.agents import rules
        facts.append(("Refusal categories", "The copilot refuses these request categories before any model call: "
                      + "; ".join(f"{c} (routed to {rules.ROUTE_TO.get(c) or 'nobody'}): {m}" for c, m in rules.MESSAGES.items())))
    except Exception:
        pass
    try:
        from app.agents.state import MAX_STEPS, MULTI_MAX_STEPS, MULTI_TOKENS, TOKEN_CAPS, arch
        facts.append(("Agent budgets", f"Agent step cap: {MAX_STEPS} steps per query (tool calls plus LLM calls). Token budgets: "
                      f"{TOKEN_CAPS['FULL_REVIEW']:,} tokens per full review, {TOKEN_CAPS['QUESTION']:,} per question or "
                      f"comparison, {TOKEN_CAPS['REFUSE']:,} per refusal. Agent architecture in use: {arch()} "
                      f"(the multi-agent graph has its own caps: {MULTI_MAX_STEPS} steps, {MULTI_TOKENS:,} tokens)."))
    except Exception:
        pass
    try:
        from app.scoring.engine import load_rubric
        r = load_rubric()
        bands = ", ".join(f"{b['band']} from {b['min']}" for b in r["bands"])
        facts.append(("Rubric", f"Scoring rubric version {r['version']}. Dimension weights: "
                      + ", ".join(f"{k} {v}" for k, v in r["weights"].items())
                      + f". Bands by overall score (lower bound inclusive): {bands}. Knockouts: "
                      + "; ".join(f"{k['code']} sets at least {k['min_band']} ({k['when']})" for k in r.get("knockouts", []))
                      + f". Limits: " + ", ".join(f"{k} {v}" for k, v in r.get("limits", {}).items()) + "."))
    except Exception:
        pass
    settings = {"retrieval strategy": os.getenv("RETRIEVAL_STRATEGY", "fixed512_ctx"),
                "retrieval mode": os.getenv("RETRIEVAL_MODE", "auto"), "reranker": os.getenv("RERANK_BACKEND", "cross_encoder"),
                "embedding backend": os.getenv("EMBED_BACKEND", "gemini"), "LLM backend": os.getenv("LLM_BACKEND", "endpoint")}
    if os.getenv("LLM_MODEL"):
        settings["LLM model"] = os.getenv("LLM_MODEL", "").strip()
    if os.getenv("LLM_CONTEXT_TOKENS"):
        settings["LLM context window (tokens)"] = os.getenv("LLM_CONTEXT_TOKENS", "")
    facts.append(("Runtime settings", "Settings of this running instance: "
                  + "; ".join(f"{k}: {v}" for k, v in settings.items()) + "."))
    try:
        if store is not None:
            c = store.counts()
            facts.append(("Store contents", f"The {getattr(store, 'backend', '')} store holds {c.get('vendor_records', 0)} vendor "
                          f"records, {c.get('documents', 0)} corpus documents and chunks per strategy: "
                          + ", ".join(f"{k} {v}" for k, v in (c.get("chunks") or {}).items()) + "."))
    except Exception:
        pass
    facts.append(("Chat assistant", "Ask the Copilot (D45) answers questions about single vendors (through the agent and "
                  "its tools), the whole vendor portfolio (computed by the compliance and risk tools for every vendor), "
                  "internal policies and regulations (corpus search), and how the copilot was built and evaluated (this "
                  "knowledge base). It checks every LLM answer in code and answers 'not found' instead of guessing."))
    return facts


def live_chunks(store=None) -> list[Chunk]:
    out = []
    for i, (head, text) in enumerate(live_facts(store)):
        out.append(Chunk(chunk_id=f"{LIVE_DOC}#{slug(head)}", doc_id=LIVE_DOC, seq=i, text=text, vendor_id=None,
                         slice="system_docs", title=LIVE_TITLE, strategy="chat_kb", heading=head, doc_type="system_doc",
                         access_class="internal", status="CURRENT", source_id="SYS", index_text=f"[{LIVE_TITLE} | {head}] {text}"))
    return out


SKIPPED: list[str] = []          # files left out because they are code, not documentation (reported by /chat/kb)
SKIPPED_SECTIONS: list[str] = []  # chunks left out because they quote raw model outputs (failure examples), not facts
MODEL_OUTPUT = re.compile(r"\*\*Got:\*\*|^\s*-?\s*Got:\s*`", re.M)


def records_model_output(text: str) -> bool:
    """Failure-log examples quote what the model WRONGLY answered ("**Got:** {... tier 2 ...}"). They document a
    failure; repeated by the chat they would read as facts. The failure counts and notes around them stay."""
    return bool(MODEL_OUTPUT.search(text))


def looks_like_code(rel: str, text: str) -> bool:
    """A .md file that is really a Python script (e.g. a report generator saved under the report's name) is not
    documentation: its f-strings would be quoted as facts. Left out and reported."""
    if not rel.endswith(".md"):
        return False
    first = next((l.strip() for l in text.splitlines() if l.strip()), "")
    return first.startswith(('"""', "from __future__", "import ", "#!/usr/bin/env python"))


def kb_files(root: Path = ROOT) -> list[str]:
    pats = DEFAULT_PATHS + [p.strip() for p in os.getenv("CHAT_KB_PATHS", "").split(",") if p.strip()]
    out: list[str] = []
    for p in pats:
        for f in sorted(glob.glob(str(root / p))):
            rel = Path(f).resolve().relative_to(root.resolve()).as_posix()
            if rel not in out and Path(f).is_file():
                out.append(rel)
    return out


# ------------------------------------------------------------------ search
@dataclass
class KBHit:
    chunk: Chunk
    score: float
    rank: int


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], embedder=None):
        if not chunks:
            raise RuntimeError("the system knowledge base is empty: README.md and docs/ were not found")
        self.chunks = chunks
        self.by_id = {c.chunk_id: c for c in chunks}
        self.bm25 = BM25Retriever(chunks)
        self.dense = None
        self.note = ""
        if embedder is not None:
            try:
                from app.retrieval.retrievers import DenseRetriever
                self.dense = DenseRetriever(chunks, embedder)
            except Exception as e:                                  # never fail the chat because embeddings are down
                self.note = f"knowledge base: BM25 only (dense index failed: {type(e).__name__}: {str(e)[:100]})"
        self.mode = "hybrid" if self.dense else "bm25"

    def search(self, query: str, k: int = 6) -> list[KBHit]:
        a = [c for c, s in self.bm25.search(query, 20) if s > 0]
        if self.dense is not None:
            try:
                b = [c for c, _ in self.dense.search(query, 20)]
                fused = rrf([a, b])
            except Exception:
                fused = [(c, 1.0 / (61 + i)) for i, c in enumerate(a)]
        else:
            fused = [(c, 1.0 / (61 + i)) for i, c in enumerate(a)]
        out, per_doc = [], {}
        for c, s in fused:
            if per_doc.get(c.doc_id, 0) >= 3:                      # one long document cannot fill every slot
                continue
            per_doc[c.doc_id] = per_doc.get(c.doc_id, 0) + 1
            out.append(KBHit(c, s, len(out) + 1))
            if len(out) >= k:
                break
        return out


def coverage(question: str, texts: list[str]) -> float:
    """Best single-passage share of the question's content words (stemmed). 0 when the question has none."""
    q = content_terms(question)
    if not q:
        return 0.0
    return max((len(q & content_terms(t)) / len(q) for t in texts), default=0.0)


_lock = threading.Lock()
_KB: dict[tuple, KnowledgeBase] = {}


def _embedder():
    mode = os.getenv("CHAT_KB_DENSE", "auto").strip().lower()
    if mode == "off":
        return None, ""
    try:
        from app.retrieval.retrievers import get_embedder
        return get_embedder(), ""
    except Exception as e:
        if mode == "on":
            raise
        return None, f"knowledge base: BM25 only (no embedder: {type(e).__name__})"


def get_kb(store=None) -> KnowledgeBase:
    """One knowledge base per process and settings, built on first use (docs + live facts)."""
    key = (os.getenv("CHAT_KB_PATHS", ""), os.getenv("CHAT_KB_DENSE", "auto"), os.getenv("EMBED_BACKEND", "gemini"),
           os.getenv("CHAT_KB_CHUNK_WORDS", "180"), id(store) if store is not None else None)
    with _lock:
        if key not in _KB:
            chunks: list[Chunk] = []
            SKIPPED.clear()
            SKIPPED_SECTIONS.clear()
            for rel in kb_files():
                try:
                    text = (ROOT / rel).read_text(encoding="utf-8")
                except Exception:
                    continue
                if looks_like_code(rel, text):
                    SKIPPED.append(rel)
                    continue
                for c in chunk_file(rel, text):
                    if records_model_output(c.text):
                        SKIPPED_SECTIONS.append(c.chunk_id)
                    else:
                        chunks.append(c)
            chunks += live_chunks(store)
            emb, note = _embedder()
            kb = KnowledgeBase(chunks, emb)
            kb.note = kb.note or note
            _KB[key] = kb
        return _KB[key]


def reset():
    with _lock:
        _KB.clear()


if __name__ == "__main__":                                        # python -m app.chatbot.knowledge  (prebuild + stats)
    from collections import Counter
    kb = get_kb()
    print(f"{len(kb.chunks)} chunks from {len({c.doc_id for c in kb.chunks})} files, mode {kb.mode}. {kb.note}")
    for d, n in Counter(c.doc_id for c in kb.chunks).most_common():
        print(f"  {n:4d}  {d}")
