"""Find out why traces do not reach Langfuse. Checks each link in order and stops at the first broken one.

    python -m scripts.check_langfuse

  1. settings     LANGFUSE_HOST, key shapes (pk-lf-... / sk-lf-...), placeholders, swapped keys. Keys are never printed.
  2. proxy        an HTTP(S)_PROXY without localhost in NO_PROXY sends local traffic to the proxy (a common
                  "Unexpected error" cause on office laptops)
  3. server       GET /api/public/health: is Langfuse running and reachable at LANGFUSE_HOST?
  4. keys         GET /api/public/projects with the keys: do they belong to this Langfuse instance?
  5. round trip   send one test trace with the SDK, flush, and read it back through the API

The SDK's "Unexpected error occurred. Please check your request and contact support" can mean any of 2-5; this
script tells you which.
"""
from __future__ import annotations

import os
import sys
import time
import uuid

from dotenv import load_dotenv

OK, BAD, INFO = "OK  ", "FAIL", "    "


def say(tag: str, msg: str):
    print(f"[{tag}] {msg}", flush=True)


def check_settings(env=os.environ) -> tuple[bool, str, str, str]:
    host = (env.get("LANGFUSE_HOST") or "http://localhost:3000").rstrip("/")
    pk, sk = env.get("LANGFUSE_PUBLIC_KEY", "").strip(), env.get("LANGFUSE_SECRET_KEY", "").strip()
    say(INFO, f"LANGFUSE_HOST = {host}")
    say(INFO, f"public key: {len(pk)} chars, starts {pk[:6]!r}; secret key: {len(sk)} chars, starts {sk[:6]!r}")
    problems = []
    if not pk or not sk:
        problems.append("a key is empty: copy both from Langfuse > your project > Settings > API Keys into .env")
    elif "REPLACE" in pk or "REPLACE" in sk:
        problems.append("the keys are still the REPLACE-ME placeholders from .env.example")
    elif pk.startswith("sk-lf-") and sk.startswith("pk-lf-"):
        problems.append("the keys are swapped: LANGFUSE_PUBLIC_KEY must start pk-lf-, LANGFUSE_SECRET_KEY sk-lf-")
    elif not pk.startswith("pk-lf-") or not sk.startswith("sk-lf-"):
        problems.append("unexpected key shape: public keys start pk-lf-, secret keys sk-lf- (check for quotes or spaces in .env)")
    if pk != env.get("LANGFUSE_PUBLIC_KEY", "") or sk != env.get("LANGFUSE_SECRET_KEY", ""):
        problems.append("a key has spaces around it in .env: remove them")
    if "langfuse:3000" in host:
        problems.append("http://langfuse:3000 only works inside Docker; on your laptop use http://localhost:3000 "
                        "(the compose file sets the container value itself)")
    if host.startswith("https://localhost") or host.startswith("https://127."):
        problems.append("the local Langfuse serves http, not https: use http://localhost:3000")
    for p in problems:
        say(BAD, p)
    if not problems:
        say(OK, "settings look right")
    return not problems, host, pk, sk


def check_proxy(host: str, env=os.environ) -> bool:
    proxy = env.get("HTTPS_PROXY") or env.get("https_proxy") or env.get("HTTP_PROXY") or env.get("http_proxy")
    if not proxy or not any(h in host for h in ("localhost", "127.0.0.1")):
        say(OK, "no proxy in the way" if not proxy else "proxy set, host is not local")
        return True
    no_proxy = (env.get("NO_PROXY") or env.get("no_proxy") or "").lower()
    if "localhost" in no_proxy or "127.0.0.1" in no_proxy:
        say(OK, "proxy set, but localhost is excluded (NO_PROXY)")
        return True
    say(BAD, "an HTTP(S)_PROXY is set and NO_PROXY does not include localhost, so traces to Langfuse go to the proxy. "
             "Add to .env:  NO_PROXY=localhost,127.0.0.1")
    return False


def check_server(client, host: str) -> bool:
    try:
        r = client.get(f"{host}/api/public/health", timeout=10)
    except Exception as e:
        say(BAD, f"cannot reach {host} ({type(e).__name__}). Is Langfuse running? Start it with\n"
                 "        docker compose -f docker/docker-compose.yml --env-file .env up -d langfuse langfuse-db\n"
                 "       wait ~30 s, open http://localhost:3000 in a browser, and check `... logs --tail 50 langfuse` if it does not load")
        return False
    if r.status_code != 200:
        say(BAD, f"{host}/api/public/health answered {r.status_code}: {r.text[:200]} (still starting? wait 30 s and retry; "
                 "else see the langfuse container logs)")
        return False
    try:
        version = r.json().get("version")
    except ValueError:
        version = None
    say(OK, f"Langfuse is up at {host}" + (f" (server {version})" if version else ""))
    return True


def check_keys(client, host: str, pk: str, sk: str) -> bool:
    r = client.get(f"{host}/api/public/projects", auth=(pk, sk), timeout=10)
    if r.status_code == 200:
        names = [p.get("name") for p in (r.json().get("data") or [])]
        say(OK, f"keys accepted: project {', '.join(map(str, names)) or '?'}")
        return True
    if r.status_code in (401, 403):
        say(BAD, f"keys rejected ({r.status_code}). They do not belong to this Langfuse instance. Usual causes: keys from "
                 "Langfuse Cloud while LANGFUSE_HOST is local (or the reverse), or the langfuse_data volume was reset so "
                 "the old project is gone. Fix: open the Langfuse UI at LANGFUSE_HOST, sign up / log in, create a project, "
                 "Settings > API Keys > Create, and paste both keys into .env")
        return False
    say(BAD, f"/api/public/projects answered {r.status_code}: {r.text[:200]}")
    return False


def check_round_trip(client, host: str, pk: str, sk: str) -> bool:
    from langfuse import Langfuse
    lf = Langfuse(public_key=pk, secret_key=sk, host=host, release="check_langfuse")
    tid = str(uuid.uuid4())
    lf.trace(id=tid, name="check_langfuse", input={"ping": True}, metadata={"source": "scripts.check_langfuse"})
    lf.flush()
    for _ in range(10):                                      # ingestion is asynchronous on the server
        r = client.get(f"{host}/api/public/traces/{tid}", auth=(pk, sk), timeout=10)
        if r.status_code == 200:
            say(OK, f"test trace stored and read back: {host}/trace/{tid}")
            return True
        time.sleep(1)
    say(BAD, f"the SDK sent a trace but the server did not store it (last answer {r.status_code}). Check the langfuse "
             "container logs for ingestion errors; a langfuse SDK 3.x against this v2 server also causes this "
             "(pip install \"langfuse>=2.50,<3\")")
    return False


def main() -> int:
    load_dotenv(override=False)
    import httpx
    print("Langfuse check\n")
    good, host, pk, sk = check_settings()
    if not good:
        return 1
    if not check_proxy(host):
        return 1
    with httpx.Client(trust_env=False) as client:            # talk to Langfuse directly, as the fixed setup will
        if not check_server(client, host) or not check_keys(client, host, pk, sk):
            return 1
        try:
            import langfuse
            say(INFO, f"langfuse SDK {langfuse.__version__}")
        except Exception:
            say(BAD, "the langfuse SDK is not installed: pip install \"langfuse>=2.50,<3\"")
            return 1
        if not check_round_trip(client, host, pk, sk):
            return 1
    print("\nAll good. Run any command (e.g. python -m scripts.agent_eval --ids EVAL-069) and its trace appears in Langfuse.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
