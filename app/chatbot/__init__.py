"""Ask the Copilot: a grounded chat assistant over the whole system (D45).

    from app.chatbot import ask, ChatSession
    session = ChatSession()
    turn = ask("Which vendors are rated CRITICAL?", session)
    turn["answer"], turn["citations"], turn["grounding"]

It answers questions about vendors, the vendor portfolio, internal policies and regulations, and how the copilot itself
was built and evaluated, using only the system's own records: the five tools, the corpus and the project documents.
Every answer either comes from a deterministic tool, or is written by the LLM from retrieved passages and then checked
in code (citations, numbers, sentence support, PII); an answer that fails the checks is replaced by quoted sources,
and a question the records do not cover gets "not found", never a guess. See docs/chatbot.md.
"""


def __getattr__(name):            # lazy: importing app.chatbot.knowledge alone must not load the agent stack
    if name in ("ask", "ChatSession"):
        from app.chatbot import engine
        return getattr(engine, name)
    raise AttributeError(name)
