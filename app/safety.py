"""Input-side safety controls applied to retrieved text before any model sees it (step 2.7; threat model T1, T2, T5).

  detect_injection(text) -> list of matched patterns: instruction-like text aimed at an AI / automated reviewer
  redact_pii(text)       -> (text with personal identifiers masked, count)
  NON_EVIDENCE_TYPES     -> document types that can never verify a control (decision D11)

These are deterministic heuristics: a first filter, not a guarantee. The real protection is structural (D2, D24):
scores, checklist and approvers come from vendor records through tools, so text in a document cannot change them.
Detection makes attempts visible (flag on the passage, INJECTION_DETECTED audit event) so the critic and humans see them.
"""
from __future__ import annotations

import re

NON_EVIDENCE_TYPES = frozenset({"email", "press_release", "vendor_webpage", "chat_transcript", "policy_draft", "note"})

_I = re.I | re.S
INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("override_instructions", re.compile(r"\b(ignore|disregard|forget)\b[^.\n]{0,40}\b(previous|prior|above|earlier|all|any)\b"
                                         r"[^.\n]{0,20}\b(instructions?|prompts?|rules|guidelines)\b", _I)),
    ("addressed_to_ai", re.compile(r"\b(instructions?|note|message|notice)\s+(to|for)\s+(any\s+|all\s+)?(ai|automated|llm|language[- ]model|"
                                   r"assistants?|bots?|(contract[- ])?review(ing)?[- ]tools?)\b", _I)),
    ("addressed_to_ai", re.compile(r"\b(to|for)\s+(any|all)\s+(ai|llm|assistants?|bots?)\b", _I)),
    ("addressed_to_ai", re.compile(r"\b(ai|llm|automated)\s+(assistant|agent|model|system)s?\s+(reading|processing|summari[sz]ing|reviewing)\b", _I)),
    ("role_prefix", re.compile(r"(^|[\]\n.])\s*(assistant|system|developer)\s*:\s*(when|you|ignore|always|state|do|from)\b", _I)),
    ("exfiltration", re.compile(r"\b(print|reveal|show|output|disclose|send)\b[^.\n]{0,60}\b(system prompt|api[- ]?key|"
                                r"connection string|access token)s?\b", _I)),
    ("verdict_override", re.compile(r"\b(treat|mark|consider|record|report|state)\b[^.\n]{0,80}\b(as\s+|is\s+)(fully\s+)?(compliant|verified|"
                                    r"current|approved|fully met|low risk)\b", _I)),
    ("verdict_override", re.compile(r"\b(approve|recommend)\b[^.\n]{0,40}\b(automatically|proceed)\b", _I)),
    ("concealment", re.compile(r"\bdo not (mention|cite|disclose|reveal|report)\s+(this|these)\b", _I)),
    ("hidden_markup", re.compile(r"<!--[^>]{0,400}\b(instruction|assistant|ai|tool|model)s?\b[^>]{0,400}-->", _I)),
    ("hidden_markup", re.compile(r"\[\s*hidden text", _I)),
]

PII_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    ("aadhaar", re.compile(r"\b\d{4}[ -]\d{4}[ -]\d{4}\b"), "[AADHAAR REDACTED]"),
    ("pan", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[PAN REDACTED]"),
    ("phone", re.compile(r"(?<![\w+])\+\d{1,3}(?:[ -]?\(?\d{2,5}\)?){2,4}\b"), "[PHONE REDACTED]"),   # any +country number
    ("phone", re.compile(r"(?<![\w.,/-])[6-9]\d{9}(?![\w.,/-])"), "[PHONE REDACTED]"),                  # bare Indian mobile
    ("account", re.compile(r"\b\d{4}-\d{4}-\d{4}(-\d{4})?\b"), "[ACCOUNT REDACTED]"),
    ("email", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "[EMAIL REDACTED]"),
]


def detect_injection(text: str) -> list[str]:
    """Names of the injection patterns found (empty list = nothing suspicious)."""
    return sorted({name for name, rx in INJECTION_PATTERNS if rx.search(text or "")})


def redact_pii(text: str, keep_business_email: bool = False) -> tuple[str, int]:
    """Mask Aadhaar, PAN, Indian mobile numbers, account numbers and e-mail addresses. Returns (text, matches)."""
    n = 0
    for name, rx, repl in PII_PATTERNS:
        if name == "email" and keep_business_email:
            continue
        text, k = rx.subn(repl, text or "")
        n += k
    return text, n
