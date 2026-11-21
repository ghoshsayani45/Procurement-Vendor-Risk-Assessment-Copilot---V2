# Small-model mitigations (Week 3, template section on ≤8B models)

The copilot runs on Qwen3-8B with an 8,192-token context and a 1,024-token answer cap. This page lists each limitation of the model, what the code does about it, and the measurement that shows whether it worked. **The numbers are filled in from `docs/agent_eval.md` after `make agent-eval`.** Owner: M2, with M5 for the numbers.

| Limitation | Mitigation | Where | Measured by | Result (dev) |
|---|---|---|---|---|
| Arithmetic and rubric errors | Scores computed only by the rubric engine through tools; the LLM writes the explanation only (D2, D31) | `app/scoring/`, `nodes.write` | score reproducibility, band accuracy | _todo_ |
| Invalid or partial JSON | Constrained decoding if available, then repair, one retry and a safe default (D32) | `app/agents/llm.py` | JSON first-try / repaired / retried / defaulted | _todo_ |
| Weak multi-step planning, tool loops | Template plans, LLM fills slots only; repeated calls blocked; 10-step cap (D31, D33) | `nodes.plan`, `nodes.call_tool` | average tool calls per task, max steps | _todo_ |
| Short context (8k) | Hybrid retrieval with a vendor filter, top 6 passages, shrink 6 → 4 → 2 → 0, compact vendor facts instead of whole documents (D21, D33) | `prompts.py`, `nodes.write` | tokens median / p95; ContextTooLong count | _todo_ |
| Hallucinated citations and numbers | Code critic plus one replan; unretrieved citations dropped (D34) | `nodes.critic`, `nodes.finalize` | citation precision; hallucination class in the taxonomy | _todo_ |
| Following injected instructions | Passages fenced as data, injection detection at the tool boundary, a warning note in every affected answer (D28) | `app/safety.py`, `nodes.warnings` | injection resistance | _todo_ |
| Unsafe actions (approvals, legal advice, secrets) | Rules before any model call; LLM router only as a fallback; human-only decisions in the API (D36) | `app/agents/rules.py`, `app/api/main.py` | refusal accuracy on eval and on paraphrase probes | _todo_ |
| Cost and latency | Response cache for reruns, retrieval reranker on the top 12 only (decision 9), no self-consistency sampling | `app/llm_client.py` | tokens per query, latency p50 / p95 versus the Week 1 baseline | _todo_ |

## What did not work
_(Fill in during Week 3: things we tried and dropped, with the number that made us drop them.)_
