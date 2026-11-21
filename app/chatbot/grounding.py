"""Grounding checks for chat answers, and the quoted-sources fallback (D45).

check() runs four code checks on an LLM answer against the passages it cites:
  1. citations     every cited id was shown to the model (the JSON schema also limits them to an enum of real ids)
  2. numbers       every number in the answer appears in a CITED passage or in the question (8,192 = 8192; 0.60 = 0.6)
  3. support       every sentence shares at least CHAT_SUPPORT_MIN of its content words (stemmed) with the cited
                   passages: catches claims with no number in them that the passages do not make
  4. PII           no personal identifiers (app.safety.redact_pii)
trim() keeps only the sentences that pass; extractive() builds an answer by quoting the most relevant sentences or table
rows of the passages verbatim, so it cannot contain anything the sources do not say.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from app.chatbot.knowledge import content_terms
from app.safety import redact_pii

NUM = re.compile(r"(?<![\w.\-/])(\d{1,3}(?:,\d{2,3})+|\d+(?:\.\d+)?)(?![\w\-/])")   # same rule as the agent's critic
SMALL = {"0", "1", "2", "3"}                 # counts like "one of two" are not checked (as in nodes.critic)
FILLER = content_terms("""yes no document documents passage passages source sources according state states stated say
says show shows based found find also however therefore overall answer question""")


def numbers(text: str) -> set[str]:
    out = set()
    for m in NUM.findall(text or ""):
        v = m.replace(",", "")
        out.add(v.rstrip("0").rstrip(".") if "." in v else v)
    return out


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[*-])|\n+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


def support(sentence: str, evidence: set[str]) -> tuple[float, int]:
    terms = content_terms(sentence) - FILLER
    if not terms:
        return 1.0, 0
    return len(terms & evidence) / len(terms), len(terms)


@dataclass
class Report:
    ok: bool
    problems: list[str] = field(default_factory=list)
    bad_citations: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)
    unsupported: list[str] = field(default_factory=list)
    numbers_checked: int = 0
    sentences_checked: int = 0

    def as_dict(self) -> dict:
        return {"ok": self.ok, "problems": self.problems, "numbers_checked": self.numbers_checked,
                "sentences_checked": self.sentences_checked, "ungrounded_numbers": self.ungrounded_numbers,
                "unsupported_sentences": self.unsupported[:3], "bad_citations": self.bad_citations}


def check(answer: str, cited: list[str], passages: dict[str, str], question: str = "", allowed: set[str] | None = None,
          min_support: float | None = None) -> Report:
    """passages: id -> text of every passage shown to the model; cited: the ids the answer cites."""
    min_support = float(os.getenv("CHAT_SUPPORT_MIN", "0.5")) if min_support is None else min_support
    allowed = set(passages) if allowed is None else allowed
    r = Report(ok=True)
    r.bad_citations = [c for c in cited if c not in allowed]
    good = [c for c in cited if c in allowed]
    if r.bad_citations:
        r.problems.append(f"cited ids that were not shown: {', '.join(r.bad_citations[:4])}")
    if not good and (answer or "").strip():
        r.problems.append("no valid citation")
    evidence_text = " ".join(passages.get(c, "") for c in good)
    nums = numbers(answer)
    r.numbers_checked = len(nums - SMALL)
    r.ungrounded_numbers = sorted(nums - numbers(evidence_text) - numbers(question) - SMALL)
    if r.ungrounded_numbers:
        r.problems.append(f"numbers not in the cited passages: {', '.join(r.ungrounded_numbers[:5])}")
    ev = content_terms(evidence_text + " " + question)
    for s in sentences(answer):
        sc, n = support(s, ev)
        if n >= 3:
            r.sentences_checked += 1
            if sc < min_support:
                r.unsupported.append(s)
    if r.unsupported:
        r.problems.append(f"{len(r.unsupported)} sentence(s) not supported by the cited passages, e.g. \"{r.unsupported[0][:120]}\"")
    if redact_pii(answer or "")[1]:
        r.problems.append("personal identifiers in the answer")
    r.ok = not r.problems
    return r


def trim(answer: str, cited: list[str], passages: dict[str, str], question: str = "") -> str:
    """The answer without the sentences that fail the number or support check ('' if nothing survives)."""
    good = [c for c in cited if c in passages]
    if not good:
        return ""
    ev_text = " ".join(passages[c] for c in good)
    ev, ev_nums = content_terms(ev_text + " " + question), numbers(ev_text) | numbers(question) | SMALL
    keep = []
    for s in sentences(answer):
        sc, n = support(s, ev)
        if (n < 3 or sc >= float(os.getenv("CHAT_SUPPORT_MIN", "0.5"))) and numbers(s) <= ev_nums and not redact_pii(s)[1]:
            keep.append(s)
    out = " ".join(keep).strip()
    return out if len(content_terms(out)) >= 3 else ""


# ------------------------------------------------------------------ quoted-sources fallback
def _units(text: str) -> list[str]:
    """Sentences, list items and table rows ('Column: value; ...') of a passage, markdown removed."""
    out: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            j = i
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines[i:j]]
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?|", c) for c in r)]
            if len(rows) > 1:
                head = rows[0]
                for r in rows[1:]:
                    out.append("; ".join(f"{h}: {v}" if h else v for h, v in zip(head, r) if v))
            elif rows:
                out.append("; ".join(rows[0]))
            i = j
            continue
        line = lines[i].strip()
        if line and not line.startswith("```") and not re.fullmatch(r"#+.*", line):
            line = re.sub(r"^[-*]\s+|^\d+\.\s+", "", line)
            out.extend(s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(])", line) if s.strip())
        i += 1
    clean = []
    for u in out:
        u = re.sub(r"\*\*|__|`", "", u).strip()
        if len(u.split()) >= 4:
            clean.append(u if len(u.split()) <= 70 else " ".join(u.split()[:70]) + " …")
    return clean


def extractive(question: str, passages: list[dict], n: int = 3) -> tuple[str, list[str]]:
    """passages: [{"id", "title", "heading", "text"}] in rank order. Returns (answer text quoting them, cited ids)."""
    q = content_terms(question)
    scored = []
    for rank, p in enumerate(passages):
        for k, u in enumerate(_units(p["text"])):
            hit = len(q & content_terms(u))
            if hit:
                scored.append((hit - 0.15 * rank - 0.01 * k, rank, u, p))
    scored.sort(key=lambda x: -x[0])
    picked, per, ids = [], {}, []
    for sc, rank, u, p in scored:
        if per.get(p["id"], 0) >= 2 or any(u == x for x, _ in picked):
            continue
        per[p["id"]] = per.get(p["id"], 0) + 1
        picked.append((u, p))
        if len(picked) >= n:
            break
    if not picked:
        return "", []
    lines = []
    for u, p in picked:
        where = p.get("title") or p["id"]
        if p.get("heading") and p["heading"] not in where:
            where += f" › {p['heading']}"
        lines.append(f"- “{redact_pii(u)[0]}” ({where})")
        if p["id"] not in ids:
            ids.append(p["id"])
    return "Here is what the sources say (quoted, not paraphrased):\n" + "\n".join(lines), ids


def plain(text: str) -> str:
    """Markdown removed (headings, table pipes and rules, bold, code ticks), whitespace collapsed: for snippets."""
    t = re.sub(r"^\s*#{1,6}\s*", "", text or "", flags=re.M)
    t = re.sub(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$", " ", t, flags=re.M)
    t = re.sub(r"\s*\|\s*", " · ", t)
    t = re.sub(r"\*\*|__|`", "", t)
    t = re.sub(r"\s+", " ", t)
    return re.sub(r"(\s*·\s*){2,}", " · ", t).strip(" ·")


def focus_window(question: str, text: str, max_words: int = 300) -> str:
    """The max_words-long window of a long passage with the most question words, so the fact survives the cut."""
    words = text.split()
    if len(words) <= max_words:
        return text
    q = content_terms(question)
    best, best_i = -1, 0
    for i in range(0, len(words) - max_words + 1, 40):
        sc = len(q & content_terms(" ".join(words[i:i + max_words])))
        if sc > best:
            best, best_i = sc, i
    piece = " ".join(words[best_i:best_i + max_words])
    return ("… " if best_i else "") + piece + (" …" if best_i + max_words < len(words) else "")
