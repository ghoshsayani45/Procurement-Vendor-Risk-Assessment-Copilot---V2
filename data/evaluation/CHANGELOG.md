# eval set changelog

## v1 (frozen 2026-11-03, tag `eval-v1`)
- 80 queries (30 dev / 50 test) over corpus v1 (1,053 docs) and rubric v1.0, review as-of 2026-11-01.
- SHA-256: see `eval_v1.sha256`.

## v2 (frozen 2026-11-21, tag `eval-v2`)
- **A new, separate set, not an edit of v1** (v1 is unchanged). 45 queries EV2-001 to EV2-045, all held out (`split: test`), over the
  same corpus v1 and rubric v1.0, review as-of 2026-11-01. Built by `scripts/build_eval_v2.py`; SHA-256 in `eval_v2.sha256`.
- **Why:** v1 dev is at ceiling (30/30) and v1 test was spent at the Week 3 gate (D12), so neither can measure the multi-agent
  experiment (D44). v2 targets the Week 3 test failure patterns with questions v1 never asked: 8 verdict lookups, 6 missing/expired/
  contradictory evidence, 7 internal policy (several conditions, nearby rules), 9 external regulation (new rules, a superseded manual),
  4 adversarial (documents v1 did not use), 3 comparisons, 3 full reviews, 3 reworded refusals, 2 clarifications.
- **Gold** from ground truth only (vendors.json, expected_outcomes.json, verbatim corpus quotes); the build fails on a missing doc or
  quote, and the tests check the oracle scores 100% and a naive system under 15%.
- **Disclosure:** written by the same assistant that builds the multi-agent agent, before any multi-agent code and before either
  agent was run on it. Run each architecture on it once; don't tune on its failures.
