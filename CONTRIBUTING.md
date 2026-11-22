# Contributing

1. Branch from `main`: `git checkout -b <member>/<step>-<short-name>` (e.g. `m2/1.4-ingestion`).
2. Keep PRs small; one step or one tool per PR. Fill in the PR template.
3. `make test` must pass. Rubric or ground-truth changes need a line in `docs/decisions.md`.
4. Never commit `.env`, anything under `data/raw/` or `data/processed/`, or real vendor records.
5. Every LLM call goes through `app/llm_client.py` so tokens are logged and the quota guard applies.
6. After the Week 1 gate, `data/evaluation/eval_v1.jsonl` is read-only. Fixes go into `eval_v2` with a note.

## Owners
| Member | Area | 
|---|---|
| Sayani Ghosh | Corpus gathering and Eval datasets creation , Single Agent vs Multi Agent Achitecture comparison, Observability
| Rajat Bodankar | RAG pipeline and AWS deployment of application  
| Sounak Moulik | Agentic orchestratio and tool calling , FastApi Service, FrontEnd and UX
| Shakya Acharya  | Frontend and UX , Inbuilt Chatbot

