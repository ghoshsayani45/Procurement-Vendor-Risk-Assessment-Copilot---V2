"""Chunking strategies

    fixed512        naive baseline (step 1.8): ~512-token word windows, 64-token overlap, no title, no structure
    fixed512_ctx    the same windows, each prefixed with a contextual header "[title | section]" (isolates the header's effect)
    structure       heading/clause-aware: split at markdown headings, numbered clauses, Rule/Article/Section/Chapter lines
                    and ALL-CAPS headings; pack whole sections up to ~400 tokens; contextual header on every chunk
    small2big       small ~128-token child chunks are searched; a hit returns its parent `structure` chunk as the passage

A chunk carries the document metadata needed for pre-search filtering (vendor_id, access_class, status) and for citations.
`text` is what the copilot reads and cites; `index_text` (header + text) is what BM25 and the embedder see.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "processed" / "corpus_v1.jsonl"
WORDS_PER_TOKEN = 0.75   # rough English average; 512 tokens ~ 384 words
STRATEGIES = ("fixed512", "fixed512_ctx", "structure", "small2big")
CHUNKER_VERSION = "2026-11-09.1"   # bump when any strategy's output changes (the store re-chunks on a new version)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    seq: int
    text: str
    vendor_id: str | None
    slice: str
    title: str
    strategy: str = "fixed512"
    heading: str = ""
    parent_id: str | None = None          # small2big: the structure chunk returned in place of this child
    doc_type: str = ""
    access_class: str = "internal"
    status: str = "CURRENT"
    document_date: str | None = None
    source_id: str = ""
    index_text: str = field(default="", repr=False)

    def __post_init__(self):
        if not self.index_text:
            self.index_text = self.text

    @property
    def tokens(self) -> int:
        return int(len(self.text.split()) / WORDS_PER_TOKEN)


def load_corpus(path: Path = CORPUS) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def _meta(doc: dict) -> dict:
    return dict(vendor_id=doc.get("vendor_id"), slice=doc["slice"], title=doc["title"], doc_type=doc.get("doc_type", ""),
                access_class=doc.get("access_class", "internal"), status=doc.get("status", "CURRENT"),
                document_date=doc.get("document_date"), source_id=doc.get("source_id", ""))


def _header(doc: dict, heading: str = "") -> str:
    parts = [doc["title"]]
    if doc.get("vendor_id"):
        parts.append(doc["vendor_id"])
    if heading and heading.lower() not in doc["title"].lower():
        parts.append(heading)
    return "[" + " | ".join(parts) + "]"


# ------------------------------------------------------------------ fixed windows
def _windows(words: list[str], size: int, step: int):
    i = 0
    while True:
        piece = words[i:i + size]
        if not piece:
            return
        yield piece
        if i + size >= len(words):
            return
        i += step


def fixed_chunks(doc: dict, size_tokens: int = 512, overlap_tokens: int = 64, header: bool = False) -> list[Chunk]:
    """Naive baseline: split on whitespace into ~size_tokens windows with overlap. No structure."""
    size = max(1, int(size_tokens * WORDS_PER_TOKEN))
    step = max(1, size - int(overlap_tokens * WORDS_PER_TOKEN))
    strategy = "fixed512_ctx" if header else "fixed512"
    out = []
    for seq, piece in enumerate(_windows(doc["text"].split(), size, step)):
        text = " ".join(piece)
        cid = f"{doc['doc_id']}#{'x' if header else 'c'}{seq:03d}"     # distinct ids: embeddings are keyed by chunk id
        out.append(Chunk(cid, doc["doc_id"], seq, text, strategy=strategy,
                         index_text=f"{_header(doc)} {text}" if header else text, **_meta(doc)))
    return out


# ------------------------------------------------------------------ structure-aware
HEADING_RE = re.compile(
    r"""^(
        \#{1,4}\s+\S.*                                             # markdown heading
      | (?:\d{1,2}\.){1,3}\d{0,2}\s+[A-Z][^.]{2,80}$               # 4.1 Required evidence / 12. Limitation of liability
      | \d{1,2}\.?\s+[A-Z][A-Za-z ,&/'()-]{2,60}$                  # 5 Limitation of Liability (title only, no full stop)
      | (?:Rule|RULE|Article|ARTICLE|Section|SECTION|Chapter|CHAPTER|Clause|CLAUSE|Schedule|SCHEDULE|Annex|ANNEX|Appendix|APPENDIX)
        \s+[\dIVXLC]+[A-Za-z]?\b.{0,90}                            # Rule 144 / Article 5 / Chapter 3: ...
      | Ch\.-?\s*\d+.{0,80}                                        # GFR "Ch.-6 - PROCUREMENT OF GOODS"
      | [A-Z][A-Z0-9 ,&/'()-]{3,70}                                # ALL-CAPS heading (DEFINITIONS, TERM AND TERMINATION)
      | [A-Z&]{2,4}-\d{2}(?:\.\d+)?\b.{0,80}                       # CCM control ids: IAM-02, A&A-01, DSP-07.1
    )$""", re.X)


def is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 100 or len(s.split()) > 14:
        return False
    if s.startswith("|") or s.startswith("**"):
        return False
    if HEADING_RE.match(s) is None:
        return False
    if s.isupper() and len(re.sub(r"[^A-Z]", "", s)) < 4:     # "A.", "II", page furniture
        return False
    return True


def sections(text: str) -> list[tuple[str, str]]:
    """Split text into (heading, body) sections at heading lines. Text before the first heading has heading ''."""
    out, head, buf = [], "", []
    for line in text.splitlines():
        if is_heading(line) and buf and sum(len(l.split()) for l in buf) >= 8:
            out.append((head, "\n".join(buf).strip()))
            head, buf = line.strip().lstrip("#").strip(), [line]
        elif is_heading(line) and not buf:
            head, buf = line.strip().lstrip("#").strip(), [line]
        else:
            if is_heading(line) and not head:
                head = line.strip().lstrip("#").strip()
            buf.append(line)
    if buf:
        out.append((head, "\n".join(buf).strip()))
    return [(h, b) for h, b in out if b]


def _split_long(body: str, max_words: int) -> list[str]:
    """Split one oversized section at paragraph breaks, then (if a paragraph is still too long) into word windows."""
    paras = [p for p in re.split(r"\n\s*\n", body) if p.strip()]
    pieces, cur, n = [], [], 0
    for p in paras:
        w = len(p.split())
        if w > max_words:
            if cur:
                pieces.append("\n\n".join(cur))
                cur, n = [], 0
            words = p.split()
            pieces += [" ".join(x) for x in _windows(words, max_words, max_words - max_words // 8)]
            continue
        if n + w > max_words and cur:
            pieces.append("\n\n".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += w
    if cur:
        pieces.append("\n\n".join(cur))
    if len(pieces) > 1 and len(pieces[-1].split()) < max_words // 6:      # fold a tiny tail into the previous piece
        tail = pieces.pop()
        pieces[-1] = pieces[-1] + "\n\n" + tail
    return pieces


def structure_chunks(doc: dict, max_tokens: int = 400, min_tokens: int = 60) -> list[Chunk]:
    """Pack whole sections into chunks of at most ~max_tokens; merge tiny trailing sections into the previous chunk."""
    max_words, min_words = int(max_tokens * WORDS_PER_TOKEN), int(min_tokens * WORDS_PER_TOKEN)
    packed: list[tuple[str, str]] = []           # (heading of the first section in the chunk, text)
    cur_head, cur, n = "", [], 0
    for head, body in sections(doc["text"]):
        w = len(body.split())
        if w > max_words:
            if cur:
                packed.append((cur_head, "\n\n".join(cur)))
                cur, n = [], 0
            packed += [(head, piece) for piece in _split_long(body, max_words)]
            continue
        if n + w > max_words and n >= min_words:
            packed.append((cur_head, "\n\n".join(cur)))
            cur, n = [], 0
        if not cur:
            cur_head = head
        cur.append(body)
        n += w
    if cur:
        if packed and n < min_words and len(packed[-1][1].split()) + n <= max_words * 1.25:
            h, t = packed[-1]
            packed[-1] = (h, t + "\n\n" + "\n\n".join(cur))
        else:
            packed.append((cur_head, "\n\n".join(cur)))
    out = []
    for seq, (head, text) in enumerate(packed):
        out.append(Chunk(f"{doc['doc_id']}#s{seq:03d}", doc["doc_id"], seq, text, strategy="structure", heading=head,
                         index_text=f"{_header(doc, head)} {text}", **_meta(doc)))
    return out


def small2big_chunks(doc: dict, child_tokens: int = 128, overlap_tokens: int = 16) -> tuple[list[Chunk], list[Chunk]]:
    """Children (searched) and parents (returned). Each child lies inside exactly one parent structure chunk."""
    parents = structure_chunks(doc)
    size = int(child_tokens * WORDS_PER_TOKEN)
    step = size - int(overlap_tokens * WORDS_PER_TOKEN)
    children = []
    for p in parents:
        for j, piece in enumerate(_windows(p.text.split(), size, step)):
            text = " ".join(piece)
            children.append(Chunk(f"{p.chunk_id}.{j:02d}", doc["doc_id"], len(children), text, strategy="small2big",
                                  heading=p.heading, parent_id=p.chunk_id, index_text=f"{_header(doc, p.heading)} {text}",
                                  **_meta(doc)))
    return children, parents


def chunk_corpus(docs: list[dict], strategy: str = "fixed512") -> list[Chunk]:
    """Chunks to index. For small2big these are the children; use `parents_for` to get the passages they map to."""
    if strategy == "fixed512":
        return [c for d in docs for c in fixed_chunks(d)]
    if strategy == "fixed512_ctx":
        return [c for d in docs for c in fixed_chunks(d, header=True)]
    if strategy == "structure":
        return [c for d in docs for c in structure_chunks(d)]
    if strategy == "small2big":
        return [c for d in docs for c in small2big_chunks(d)[0]]
    raise ValueError(f"unknown chunking strategy {strategy!r}; choose from {STRATEGIES}")


def parents_for(docs: list[dict]) -> dict[str, Chunk]:
    """parent_id -> parent chunk (the structure chunks), for small2big."""
    return {p.chunk_id: p for d in docs for p in structure_chunks(d)}


def save_chunks(chunks: list[Chunk], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-/][a-z0-9]+)*")
STOP = set("""a an the of to in and or for on by with is are was were be been as at this that these those from any all its it
shall may will can not no our we you your their they them he she his her i me my do does did has have had what which who whom
when where how why than then there here into onto over under about above below up down out off""".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP]
