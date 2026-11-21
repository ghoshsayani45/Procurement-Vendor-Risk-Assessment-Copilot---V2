"""Single wrapper for every LLM / embeddings call (Day 0, step 2).

All modules call `chat()` / `embed()` here, never the SDK directly, so that:
  * every call's tokens and latency are logged to a ledger (cost per query, Week 4),
  * the team's token quota is never exceeded (hard stop with a safety margin),
  * JSON-mode / schema-validated calls with one repair retry live in one place (Week 3).
"""
from __future__ import annotations

import json
import re
import os
import time
from pathlib import Path
from typing import Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

load_dotenv()


class QuotaExceeded(RuntimeError):
    pass


def _verify(prefix: str = "LLM"):
    """TLS trust for the endpoint, without ever disabling verification.
    1. <prefix>_CA_BUNDLE=path/to/ca.pem -> trust that CA file (EMBED_CA_BUNDLE for the embeddings endpoint, falling
       back to LLM_CA_BUNDLE; e.g. the course's vm02-ca.crt)
    2. otherwise, if `truststore` is installed (default) -> use the OS certificate store (Windows/macOS), so whatever
       your browser trusts - including antivirus or corporate-proxy root certificates - Python trusts too
    3. otherwise -> the certifi bundle that ships with httpx
    """
    ca = (os.getenv(f"{prefix}_CA_BUNDLE", "") or os.getenv("LLM_CA_BUNDLE", "")).strip().strip('"')
    if ca:
        path = Path(os.path.expandvars(os.path.expanduser(ca)))
        if not path.is_absolute():                            # certs/vm02-ca.crt: relative to the project folder, so the
            path = Path(__file__).resolve().parents[1] / path   # same .env line works on the laptop and in Docker (/app)
        if not path.is_file():
            raise FileNotFoundError(f"{prefix}_CA_BUNDLE points to {path}, which does not exist (save the course "
                                    "certificate there, or fix the path in .env; use forward slashes on Windows)")
        return str(path)
    if os.getenv("USE_SYSTEM_CERTS", "1") == "1":
        try:
            import ssl
            import truststore
            return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        except ImportError:
            pass
    return True


_CLIENTS: dict = {}


def _client(base_url_env: str):
    from openai import OpenAI  # imported lazily so offline scripts don't need the package
    import httpx
    if base_url_env not in _CLIENTS:
        key = (os.getenv("EMBED_API_KEY") if base_url_env == "EMBED_BASE_URL" else None) or os.environ["LLM_API_KEY"]
        embed = base_url_env == "EMBED_BASE_URL"
        # embeddings: short timeout, no hidden SDK retries (default 2 x 120 s could hold a search thread for minutes);
        # embed() retries once itself, and search falls back to BM25 if the endpoint stays slow
        timeout = (httpx.Timeout(float(os.getenv("EMBED_TIMEOUT_S", "8")), connect=5.0) if embed
                   else httpx.Timeout(120.0, connect=20.0))
        _CLIENTS[base_url_env] = OpenAI(base_url=os.environ[base_url_env].strip(), api_key=key.strip(),
                                        max_retries=0 if embed else 2,
                                        http_client=httpx.Client(verify=_verify("EMBED" if embed else "LLM"), timeout=timeout))
    return _CLIENTS[base_url_env]


LEDGER = Path(os.getenv("TOKEN_LEDGER_PATH", "observability/token_ledger.jsonl"))


def tokens_used(kind: str | None = None) -> int:
    """Course-billed tokens in the ledger; kind='chat' or 'embed' for one endpoint (they have separate quotas)."""
    if not LEDGER.exists():
        return 0
    rows = (json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip())
    return sum(r["total_tokens"] for r in rows if r.get("billing", "course") == "course"   # Gemini is billed elsewhere
               and (kind is None or r.get("kind", "chat") == kind))


def _guard(estimated: int = 0, kind: str = "chat"):
    """TEAM_TOKEN_QUOTA guards the chat model; EMBED_TOKEN_QUOTA (optional) guards the embeddings endpoint."""
    quota = int(os.getenv("TEAM_TOKEN_QUOTA" if kind == "chat" else "EMBED_TOKEN_QUOTA", "0") or 0)
    if quota:
        margin = float(os.getenv("QUOTA_SAFETY_MARGIN", "0.05"))
        used = tokens_used(kind)
        if used + estimated > quota * (1 - margin):
            raise QuotaExceeded(f"Token quota guard ({kind}): {used:,} used of {quota:,}")


def _log(kind: str, model: str, usage, latency_ms: int, tag: str):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = dict(ts=time.time(), kind=kind, model=model, tag=tag, latency_ms=latency_ms,
               prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
               completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
               total_tokens=getattr(usage, "total_tokens", 0) or 0)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


LAST_CALL: dict = {}   # usage record of the most recent chat() call (tokens, latency) for per-query cost accounting


def backend() -> str:
    """LLM_BACKEND=endpoint (default, the course API) or mock (offline, deterministic; for tests and CI)."""
    return os.getenv("LLM_BACKEND", "endpoint")


class ContextTooLong(RuntimeError):
    """The prompt alone does not fit LLM_CONTEXT_TOKENS: the caller must shorten it (fewer passages, smaller tables)."""


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Conservative estimate (characters / 3.6, calibrated on the course Qwen endpoint in Week 1)."""
    chars = sum(len(str(m.get("content", ""))) for m in messages) + 12 * len(messages)
    return int(chars / float(os.getenv("PROMPT_CHARS_PER_TOKEN", "3.6"))) + 8


def fit_max_tokens(messages: list[dict], max_tokens: int) -> int:
    """Shrink max_tokens so prompt + answer fit LLM_CONTEXT_TOKENS (e.g. 8192). Raises ContextTooLong when fewer than
    LLM_MIN_ANSWER_TOKENS (default 200) would be left for the answer."""
    window = int(os.getenv("LLM_CONTEXT_TOKENS", "0") or 0)
    if not window:
        return max_tokens
    left = window - estimate_prompt_tokens(messages) - 48
    if left < int(os.getenv("LLM_MIN_ANSWER_TOKENS", "200")):
        raise ContextTooLong(f"prompt ~{estimate_prompt_tokens(messages):,} tokens leaves {left} of {window:,} for the answer")
    return min(max_tokens, left)


CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", "data/processed/llm_cache"))


def _cache_key(messages, temperature, max_tokens, json_mode, schema, kw) -> str:
    import hashlib
    salt = {"salt": os.getenv("LLM_CACHE_SALT")} if os.getenv("LLM_CACHE_SALT") else {}   # repeat runs (4.1); unset = old keys
    # the decoding mode changes the reply, so it is part of the key; json_schema (the project's mode, D32) keeps the
    # original keys so existing caches stay valid, every other mode (JSON mode only, guided_json, no JSON) gets its own
    mode = os.getenv("LLM_CONSTRAINED", "").strip().lower() or ("json_object" if os.getenv("LLM_JSON_MODE", "1") == "1" else "none")
    if mode != "json_schema":
        salt["decoding"] = mode
    blob = json.dumps({"m": os.getenv("LLM_MODEL", "") + "|" + backend(), **salt, "msgs": messages, "t": temperature, "n": max_tokens,
                       "j": json_mode, "s": schema, "kw": {k: v for k, v in kw.items() if k != "extra_body"}},
                      sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def chat(messages: list[dict], *, tag: str = "", temperature: float = 0.0, max_tokens: int = 1024,
         json_mode: bool = False, schema: dict | None = None, **kw) -> str:
    """One model call, traced as a Langfuse generation (prompt, output, tokens, latency, errors, retries).

    schema: a JSON schema for constrained decoding when LLM_CONSTRAINED is set (json_schema | guided_json, see
    scripts/check_constrained.py); otherwise json_mode applies. LLM_CACHE=1 caches temperature-0 answers on disk
    (identical prompt -> identical answer, zero tokens), which keeps reruns of unchanged eval queries free."""
    global LAST_CALL
    from app.observability.tracing import span
    max_tokens = fit_max_tokens(messages, max_tokens)
    use_cache = os.getenv("LLM_CACHE", "0") == "1" and temperature == 0
    key = _cache_key(messages, temperature, max_tokens, json_mode, schema, kw) if use_cache else None
    with span(tag or "chat", kind="generation", input=messages,
              metadata={"temperature": temperature, "max_tokens": max_tokens, "json_mode": json_mode,
                        "constrained": bool(schema and os.getenv("LLM_CONSTRAINED"))}) as s:
        if key and (CACHE_DIR / f"{key}.json").exists():
            hit = json.loads((CACHE_DIR / f"{key}.json").read_text(encoding="utf-8"))
            # Replay the original call's usage so cost/latency metrics are the same with or without the cache.
            # Nothing is written to the ledger: the quota is not charged again. Old cache files have no usage: estimate.
            u = hit.get("usage") or {}
            pt = u.get("prompt_tokens", estimate_prompt_tokens(messages))
            ct = u.get("completion_tokens", int(len(hit["content"]) / 3.6))
            LAST_CALL = dict(kind="chat", model=hit.get("model"), tag=tag, latency_ms=hit.get("latency_ms", 0), prompt_tokens=pt,
                             completion_tokens=ct, total_tokens=u.get("total_tokens", pt + ct), cached=True,
                             usage_estimated=not u, latency_known="latency_ms" in hit, finish_reason=hit.get("finish_reason"),
                             attempts=1)
            s.end(output=hit["content"], model=hit.get("model"), usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                  finish_reason=hit.get("finish_reason"), attempts=0, cached=True)
            return hit["content"]
        try:
            out = _chat(messages, tag=tag, temperature=temperature, max_tokens=max_tokens, json_mode=json_mode, schema=schema, **kw)
        except Exception as e:
            s.error(e)
            raise
        if key and LAST_CALL.get("finish_reason") != "length":
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            (CACHE_DIR / f"{key}.json").write_text(json.dumps({"content": out, "model": LAST_CALL.get("model"),
                                                               "finish_reason": LAST_CALL.get("finish_reason"),
                                                               "latency_ms": LAST_CALL.get("latency_ms", 0),
                                                               "usage": {k: LAST_CALL.get(k, 0) for k in
                                                                         ("prompt_tokens", "completion_tokens", "total_tokens")}}), encoding="utf-8")
        s.end(output=out, model=LAST_CALL.get("model"),
              usage={k: LAST_CALL.get(k, 0) for k in ("prompt_tokens", "completion_tokens", "total_tokens")},
              finish_reason=LAST_CALL.get("finish_reason"), attempts=LAST_CALL.get("attempts", 1))
        return out


def _chat(messages: list[dict], *, tag: str = "", temperature: float = 0.0, max_tokens: int = 1024,
          json_mode: bool = False, schema: dict | None = None, **kw) -> str:
    global LAST_CALL
    if backend() == "mock":
        from app.mock_llm import mock_chat
        t0 = time.time()
        content, usage = mock_chat(messages, temperature=temperature, json_mode=json_mode, tag=tag)
        LAST_CALL = dict(kind="chat", model="mock", tag=tag, latency_ms=int((time.time() - t0) * 1000), **usage)
        return content
    _guard(max_tokens)
    model = os.environ["LLM_MODEL"].strip()
    extra = {}
    body = dict(kw.pop("extra_body", {}) or {})
    constrained = os.getenv("LLM_CONSTRAINED", "").strip().lower()
    if schema and constrained == "json_schema":                 # OpenAI-style structured output (vLLM >= 0.6)
        extra["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}}
    elif schema and constrained == "guided_json":               # vLLM guided decoding through extra_body
        body["guided_json"] = schema
    elif json_mode and os.getenv("LLM_JSON_MODE", "1") == "1":  # set LLM_JSON_MODE=0 if the server rejects response_format
        extra["response_format"] = {"type": "json_object"}
    if _no_thinking(model):                                     # Qwen3: skip the <think> block (vLLM / SGLang chat template flag)
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if body:
        extra["extra_body"] = body
    cap = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "0") or 0)   # e.g. 1024 if the gateway limits answer length
    if cap:
        max_tokens = min(max_tokens, cap)
    t0 = time.time()
    fitted_once = schema_fallback = False
    for attempt in (1, 2, 3):
        try:
            r = _client("LLM_BASE_URL").chat.completions.create(model=model, messages=messages, temperature=temperature,
                                                                max_tokens=max_tokens, **extra, **kw)
            break
        except Exception as e:
            status = getattr(e, "status_code", None)
            fitted = _fit_to_window(e) if not fitted_once else None
            if fitted:                       # the server told us exactly how many prompt tokens it counted: retry once
                max_tokens, fitted_once = fitted, True
                continue
            if (status and 400 <= status < 500 and not schema_fallback
                    and (extra.get("response_format") or {}).get("type") == "json_schema"):
                # a schema the server cannot enforce (e.g. an enum it does not support): plain JSON mode once; the
                # reply is still validated, and the critic still checks citations
                extra["response_format"], schema_fallback = {"type": "json_object"}, True
                continue
            if status and 400 <= status < 500:
                raise EndpointRejected(_explain_rejection(e, messages, max_tokens, extra)) from e
            raise
    LAST_CALL = _log("chat", model, r.usage, int((time.time() - t0) * 1000), tag)
    LAST_CALL["finish_reason"] = r.choices[0].finish_reason   # "length" = answer cut off by max_tokens
    LAST_CALL["attempts"] = attempt
    LAST_CALL["schema_fallback"] = schema_fallback
    return strip_thinking(r.choices[0].message.content or "")


def _fit_to_window(e) -> int | None:
    """If the gateway caps output tokens ('output token limit must be between 1 and N'), return N.
    For vLLM-style 'maximum context length is N ... prompt contains at least M input tokens' errors, return a max_tokens
    that fits (N - M - 16), if at least 200 tokens are left for the answer; otherwise None (the prompt itself is too long)."""
    text = str(getattr(e, "body", None) or e)
    m_cap = re.search(r"output token limit must be between \d+ and (\d+)", text)   # gateway cap on max_tokens
    if m_cap:
        return int(m_cap.group(1))
    m_ctx = re.search(r"maximum context length is (\d+)", text)
    m_in = re.search(r"(?:at least|contains) (\d+) input tokens|(\d+) in the messages", text)
    if not (m_ctx and m_in):
        return None
    left = int(m_ctx.group(1)) - int(next(g for g in m_in.groups() if g)) - 48   # "at least" is a lower bound: keep a margin
    return left if left >= 200 else None


class EndpointRejected(RuntimeError):
    """The server refused the request (HTTP 4xx). The message carries the server's own reason and a likely fix."""


def _explain_rejection(e, messages, max_tokens, extra) -> str:
    body = getattr(e, "body", None)
    if not body:
        try:
            body = e.response.text
        except Exception:
            body = str(e)
    body = json.dumps(body)[:600] if isinstance(body, (dict, list)) else str(body)[:600]
    words = sum(len(str(m.get("content", "")).split()) for m in messages)
    est = int(words * 1.4)
    low = body.lower()
    if "output token limit" in low:
        fix = "The gateway caps answer length: set LLM_MAX_OUTPUT_TOKENS (and PROMPT_MAX_TOKENS) to the limit it states in .env."
    elif any(k in low for k in ("context length", "maximum context", "max_model_len", "too long", "max_tokens", "input tokens", "exceed")):
        fix = (f"PROMPT TOO LONG for the model's context window (this prompt is ~{est:,} tokens + {max_tokens} for the answer). "
               "Lower PROMPT_CONTEXT_WORDS (e.g. 2500) and/or PROMPT_MAX_TOKENS (e.g. 1000) in .env.")
    elif "response_format" in low or "json" in low and "schema" not in low:
        fix = "The server does not accept JSON mode: set LLM_JSON_MODE=0 in .env."
    elif "chat_template_kwargs" in low or "enable_thinking" in low or "extra" in low:
        fix = "The server rejects the Qwen3 thinking switch: set LLM_DISABLE_THINKING=0 in .env."
    elif "model" in low and ("not found" in low or "does not exist" in low):
        fix = "LLM_MODEL does not match a served model id: run python -m scripts.check_endpoint to list them."
    else:
        fix = (f"Unrecognised. Prompt was ~{est:,} tokens with max_tokens={max_tokens}, extras={list(extra)}. "
               "Try PROMPT_CONTEXT_WORDS=2500 first; if that does not help, send this message to the team.")
    return f"HTTP {getattr(e, 'status_code', '?')} from endpoint: {body}\n  -> {fix}"


def _no_thinking(model: str) -> bool:
    """LLM_DISABLE_THINKING=1/0 forces it; default 'auto' = on for Qwen3-family models only."""
    flag = os.getenv("LLM_DISABLE_THINKING", "auto").lower()
    return flag in ("1", "true") or (flag == "auto" and "qwen3" in model.lower())


def strip_thinking(text: str) -> str:
    """Remove reasoning blocks some models emit before the answer (<think>...</think>), even if unterminated."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    if "<think>" in text:                     # truncated reasoning with no closing tag: keep what follows the last '{'
        text = text[text.rfind("{"):] if "{" in text else ""
    return text.strip()


def chat_structured(messages: list[dict], schema: Type[BaseModel], *, tag: str = "", retries: int = 1, **kw) -> BaseModel:
    """Ask for JSON matching `schema`; on validation failure retry once with the error in the prompt."""
    msgs = list(messages)
    last_err: Optional[str] = None
    for _ in range(retries + 1):
        raw = chat(msgs, tag=tag, json_mode=True, **kw)
        try:
            return schema.model_validate_json(raw)
        except ValidationError as e:
            last_err = str(e)[:1500]
            msgs = msgs + [{"role": "assistant", "content": raw},
                           {"role": "user", "content": f"Your JSON did not match the schema. Errors:\n{last_err}\nReturn only corrected JSON."}]
    raise ValueError(f"Structured output failed after {retries + 1} attempts: {last_err}")


_TOKENIZER: dict = {}
TOO_LONG = re.compile(r"less than (\d+) tokens\. Given: (\d+)")


def _embed_tokenizer():
    """bge's own tokenizer (from the local Hugging Face cache, or EMBED_TOKENIZER=path/to/tokenizer.json), so text is cut
    exactly where sentence-transformers cut it in the local run. None if unavailable (e.g. the Docker image)."""
    model = os.getenv("EMBED_MODEL", "")
    name = os.getenv("EMBED_TOKENIZER", "") or ("BAAI/" + model if model.startswith("bge-") else model)
    if name not in _TOKENIZER:
        _TOKENIZER[name] = None
        try:
            from tokenizers import Tokenizer
            if os.path.isfile(name):
                tok = Tokenizer.from_file(name)
            else:
                from huggingface_hub import hf_hub_download
                tok = Tokenizer.from_file(hf_hub_download(name, "tokenizer.json"))
            tok.no_truncation()
            tok.no_padding()
            _TOKENIZER[name] = tok
        except Exception:
            pass
    return _TOKENIZER[name]


def clip_for_embedding(texts: list[str]) -> list[str]:
    """Keep the first EMBED_MAX_TOKENS - 2 word pieces of each text ([CLS] and [SEP] take the other two), cut at the
    original characters: what sentence-transformers keeps for bge (max_seq_length 512)."""
    tok = _embed_tokenizer()
    if tok is None:
        return texts
    keep = int(os.getenv("EMBED_MAX_TOKENS", "512")) - 2
    out = []
    for t, enc in zip(texts, tok.encode_batch(texts, add_special_tokens=False)):
        out.append(t[:enc.offsets[keep - 1][1]] if len(enc.ids) > keep else t)
    return out


def _embed_request(model: str, batch: list[str], tag: str) -> list[list[float]]:
    _guard(kind="embed")
    t0 = time.time()
    try:
        r = _client("EMBED_BASE_URL").embeddings.create(model=model, input=batch)
    except Exception as e:                                    # one retry on a timeout or dropped connection, never more
        if not any(w in type(e).__name__.lower() for w in ("timeout", "connection")):
            raise
        r = _client("EMBED_BASE_URL").embeddings.create(model=model, input=batch)
    _log("embed", model, r.usage, int((time.time() - t0) * 1000), tag)
    return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]


def _embed_fitting(model: str, batch: list[str], tag: str) -> list[list[float]]:
    """One request; if the server says an input is too long (no tokenizer here to clip it exactly), retry the texts one by
    one and shorten each over-long text in proportion to the token count the server reports."""
    try:
        return _embed_request(model, batch, tag)
    except Exception as e:
        if not TOO_LONG.search(str(e)):
            raise
    if len(batch) > 1:
        return [v for t in batch for v in _embed_fitting(model, [t], tag)]
    text = batch[0]
    for _ in range(4):
        try:
            return _embed_request(model, [text], tag)
        except Exception as e:
            m = TOO_LONG.search(str(e))
            if not m:
                raise
            limit, given = int(m.group(1)), int(m.group(2))
            text = text[:max(1, int(len(text) * (limit - 4) / given * 0.98))]
    raise RuntimeError("could not shorten a text enough for the embeddings endpoint")


def embed(texts: list[str], *, tag: str = "") -> list[list[float]]:
    """Embed texts: clipped to the model's 512 tokens like sentence-transformers does (the course endpoint rejects longer
    inputs instead of truncating) and sent in requests of at most EMBED_MAX_BATCH (the endpoint allows 16)."""
    model = os.environ["EMBED_MODEL"]
    size = max(1, int(os.getenv("EMBED_MAX_BATCH", "16")))
    texts = clip_for_embedding(list(texts))
    out: list[list[float]] = []
    for s in range(0, len(texts), size):
        out.extend(_embed_fitting(model, texts[s:s + size], tag))
    return out
