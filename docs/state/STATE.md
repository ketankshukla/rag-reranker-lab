# Project State: rag-reranker-lab

**Last updated:** 2026-10-12 06:35 UTC by the floor kit (the portfolio session's scripts/kit.py, AUD-013)

## Classification

```yaml classification
tier: 1
profiles: [cli]
ai_modes: [local-model]
conditions: [long-running-work]
outcome: maintain-at-floor
```

- **Owner:** the owner of this repository, the only human in the project, who decides Type 1 questions, approves specs and grants waivers.
- **Confirmed by the owner:** the portfolio triage's decision log, decision #21, 2026-10-03 (AUD-001); no approvals.md row exists yet.
- **Portfolio outcome:** the `outcome` key above, from the triage of 2026-10-02 (AUD-001).
- **Reasons:** Not yet: adoption fills this (AUDIT.md §10.2).

### Classification history

| Date | From | To | Trigger | Tier-delta plan | Approval |
|---|---|---|---|---|---|
| 2026-10-03 | none | tier 1, cli, local-model, long-running-work, maintain-at-floor | the portfolio triage (decision #21) | none | decision-log 2026-10-03 |

## Project

```yaml project
name: "rag-reranker-lab"
purpose: "not recorded at the kit; adoption records it (DEV-002)"
setup: "not recorded at the kit; adoption records it (DEV-002)"
run: "not recorded at the kit; adoption records it (DEV-002)"
test: "not recorded at the kit; adoption records it (DEV-002)"
check: "uv run --isolated --no-project --no-config python -I -B .standard/tier1.py"
standard_version: "2.0.0-rc.2"
standard_commit: "be1cfe8b496441eb550e6b85ec3d8d9308a21040"
standard_path: "E:/standard-tags/v2.0.0-rc.2"
check_settings:
  install: {".": {list: [requirements.txt], cites: "decision-log 2026-10-11"}}
  tests_declined: {".": {cites: "decision-log 2026-10-11"}}
```

## Phase

Not yet: adoption fills this (AUDIT.md §10.2).

## Standard

- **Pin:** the `standard_version`, `standard_commit` and `standard_path` keys of the project block, recorded by the floor kit (AUD-013).
- **Location:** referenced: the checkout of tag `v2.0.0-rc.2` at `E:/standard-tags/v2.0.0-rc.2` (STANDARD.md §9)
- **Standard repository:** unknown (not part of the pin)
- **Last upgrade:** none
- **CLAUDE.md last generated:** 2026-10-11, from this version and these blocks

## In-flight task

Not yet: adoption fills this (AUDIT.md §10.2).

## Completed slices

Not yet: adoption fills this (AUDIT.md §10.2).

## Exact next step

Not yet: adoption fills this (AUDIT.md §10.2).

## What was learned

Not yet: adoption fills this (AUDIT.md §10.2).

## Reported embedded instructions (untrusted)

none

## Gate receipts

Receipts are not copied here. Without a runner, and in a `maintain-at-floor` repository, each task's report states the check command run and its actual result, the maintenance check's where the floor kit's baseline is in force (AGT-001, DEV-008, AUD-015).

## Open waivers

| Approval | Rule IDs or clause | Scope | Review date |
|---|---|---|---|
| none | | | |

## Promotion triggers

Not yet: adoption fills this (AUDIT.md §10.2).

## Project records

- **Agent baseline:** confirmed in the portfolio records at E:\portfolio-records\agent-baseline.md (AGT-021; not copied here).
- **Kit baseline:** `docs/state/check-baseline.json`, recorded with `--record` on 2026-10-11 in the kit commit, and changed only as `tools/tier1_check.md` §8 allows until adoption retires it (AUD-015; not copied here).

## Read first

1. `CLAUDE.md`
2. `docs/state/STATE.md` (this file)
3. `docs/adr/README.md`, where it exists
