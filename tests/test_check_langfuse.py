"""scripts/check_langfuse.py: each check names the broken link (simulated server, no Langfuse needed)."""
import httpx

from scripts import check_langfuse as cl

GOOD = {"LANGFUSE_HOST": "http://localhost:3000", "LANGFUSE_PUBLIC_KEY": "pk-lf-123", "LANGFUSE_SECRET_KEY": "sk-lf-456"}


def client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_settings_catch_placeholders_swaps_and_docker_host(capsys):
    assert cl.check_settings(GOOD)[0]
    assert not cl.check_settings({**GOOD, "LANGFUSE_PUBLIC_KEY": "REPLACE-ME"})[0]
    assert not cl.check_settings({**GOOD, "LANGFUSE_PUBLIC_KEY": "sk-lf-456", "LANGFUSE_SECRET_KEY": "pk-lf-123"})[0]
    assert not cl.check_settings({**GOOD, "LANGFUSE_HOST": "http://langfuse:3000"})[0]
    out = capsys.readouterr().out
    assert "swapped" in out and "only works inside Docker" in out
    assert "sk-lf-456" not in out and "pk-lf-123" not in out          # keys are never printed


def test_proxy_without_no_proxy_is_flagged():
    assert cl.check_proxy("http://localhost:3000", {"HTTPS_PROXY": "http://corp:8080"}) is False
    assert cl.check_proxy("http://localhost:3000", {"HTTPS_PROXY": "http://corp:8080", "NO_PROXY": "localhost,127.0.0.1"})
    assert cl.check_proxy("http://localhost:3000", {})


def test_server_down_and_keys_rejected_are_told_apart(capsys):
    def down(req):
        raise httpx.ConnectError("refused")
    assert not cl.check_server(client(down), "http://localhost:3000")
    assert "Is Langfuse running?" in capsys.readouterr().out
    up = client(lambda req: httpx.Response(200, json={"status": "OK", "version": "2.95.0"}))
    assert cl.check_server(up, "http://localhost:3000")
    assert not cl.check_keys(client(lambda req: httpx.Response(401, json={"message": "Invalid credentials"})),
                             "http://localhost:3000", "pk-lf-1", "sk-lf-2")
    assert "do not belong to this Langfuse instance" in capsys.readouterr().out
    assert cl.check_keys(client(lambda req: httpx.Response(200, json={"data": [{"name": "pvrc"}]})),
                         "http://localhost:3000", "pk-lf-1", "sk-lf-2")
