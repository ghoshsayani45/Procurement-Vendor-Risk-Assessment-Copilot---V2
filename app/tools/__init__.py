"""The copilot's five read-only tools (step 2.5). Call them through app.tools.base.run_tool()."""


def load_all():
    """Import the tool modules so they register themselves (idempotent)."""
    from app.tools import document_search, vendor_tools  # noqa: F401


def run_tool(name, args=None, ctx=None):
    from app.tools.base import run_tool as _run
    return _run(name, args, ctx)
