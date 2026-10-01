# Project roadmap: finish the local RTX 4060 study

Status snapshot: **2026-10-01, 20:47 Asia/Singapore (UTC+08:00)**.

The [remaining-task checklist](remaining-tasks-to-complete-study.md) is the
authoritative ordered work list from this checkpoint through final delivery.
The [detailed local project plan](local-rtx4060-project-plan.md) defines the
experiment, the [runbook](runbook.md#full-study-on-the-local-rtx-4060) contains
operator commands, and [DEVLOG.md](../DEVLOG.md) records process evidence.

## Goal

Complete the experiment on all **12,221 selected questions**:

| Dataset | Required | A100 archive | RTX 4060 local |
| --- | ---: | ---: | ---: |
| GSM8K test | 1,319 | 110 | 1,209 |
| MATH test, seven subjects | 350 | 0 | 350 |
| GSM-Plus test | 10,552 | 0 | 10,552 |
| **Total** | **12,221** | **110** | **12,111** |

Preserve Qwen2.5-Math-7B-Instruct, NF4 double quantization, FP16 compute,
the pinned questions and references, one greedy answer, eight independently
seeded samples, temperature 0.7, top-k 50, top-p 1.0, and the 1,024-token
output cap. The final report must label the execution as mixed A100 / RTX 4060
and provide combined and execution-origin metrics.

## Current checkpoint

- A historical unittest discovery passed **256 tests**. Compilation,
  dependency consistency, CLI help, and diff checks also passed. The source
  snapshot tested was
  **ae37dee3b84ecd691e46815c80bda48b34197c9bbd53dc83f3d1385a50e96d91**.
  Reviewer call 1 found that later documentation edits were outside that
  receipt; the next frozen candidate needs a fresh complete pass.
- The fixed-input candidate has now passed **262 tests in 192.218 seconds**.
  Compilation, dependency consistency, runner/report help, diff, and
  whitespace checks passed. The final candidate receipt is rebound after
  this documentation update in the assurance record.
- The current RTX 4060 smoke completed **9/9 rows**, zero failed and zero
  pending, across GSM8K, MATH (all seven subjects), and GSM-Plus. It ran for
  **22.60 minutes** with two samples per row; **23.87 rows/hour** describes
  this smoke only, not the full study. Three generated samples reached the
  1,024-token limit.
- The smoke saved 4,816 MiB host RAM available after model load, 5,311 MiB
  allocated / 6,082 MiB reserved VRAM, and 1,314 MiB free VRAM. Recovery
  persisted greedy batch 1 and sample batch 1 with offloaded KV cache. Five
  memory adaptations were recorded. The smoke output is under
  %LOCALAPPDATA%\TeacherReliability\runs\smoke-rtx4060-20261001-173011.
- Its final progress heartbeat recorded 4,143 MiB available host RAM,
  5,324 MiB allocated / 6,094 MiB reserved VRAM, and 896 MiB free VRAM. This
  was below the normal 1,024 MiB free-VRAM target but above the 512 MiB
  offloaded-cache emergency floor; the smoke does not qualify sustained use.
- At 18:51 SGT the study worker was stopped, GPU use was 0 MiB, and host RAM
  availability was 6,626 MiB. Recheck before a new model load.
- The supplied ZIP passed file-backed validation for 110 complete rows,
  eight samples per row, and matching selected content, revisions, and seeds.
  SHA-256:
  **7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82**.
  It has not been imported. Its missing legacy counters remain unknown.
- Branch **codex/local-rtx4060** is not committed or pushed. Reviewer call 1
  returned **fix-first** because full-study launch/report can accept a
  self-consistent archive or revisions that differ from the declared pins.
  Focused TDD now covers and closes that behavior: full launch checks the
  exact archive hash, seed, model revision, and dataset revisions, and reports
  reject a self-consistent alternate. The 36-test report module and 262-test
  candidate-wide suite passed after the fix; re-review remains pending.
  The assurance unit is
  **Teacher-reliability-Distill-LLM/inference-optimization/vllm-hybrid**,
  open and not review-ready, with **1/3 calls used** and no reservation. This
  continuation has a verified GPT-6-Sol/max parent runtime for the next pass.
- The 12-profile comparison and 24-question qualification were waived by the
  owner. The full run must disclose its **fast-start, unqualified** status.
  The earlier incomplete qualification remains historical evidence only.

## Execution decision

Run native Windows Transformers with one model resident on the RTX 4060,
conservative batching, checkpointing, and OOM recovery. Do not use vLLM or Ray
for this Windows execution; the prepared vLLM/hybrid path remains experimental.
Keep runtime outputs under %LOCALAPPDATA%\TeacherReliability\runs, outside
OneDrive. Do not infer the full-study ETA from the smoke or prior A100 rate.

## Remaining gates

1. Refreeze the candidate, run candidate-bound checks and a new adversarial
   pass, complete the re-review matrix and machine readiness, then obtain an
   accepted final-strict verdict within the remaining budget.
2. Publish the reviewed code and reproducible docs on codex/local-rtx4060 and
   verify the remote SHA.
3. Import the 110 validated A100 rows into a new run, verify 110 imported plus
   12,111 pending with 12,221 unique IDs, then launch the resumable full study.
4. Monitor and resume until every required row is complete, calculate ETA from
   measured local rates, validate complete combined and per-origin reports,
   and document recommendation eligibility and scientific limits.
5. Create a SHA-256-verified backup, test restoring it, update the process and
   result docs, and push the final summary.

See the [remaining-task checklist](remaining-tasks-to-complete-study.md) for
the individual acceptance checks.
