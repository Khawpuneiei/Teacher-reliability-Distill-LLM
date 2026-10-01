# Remaining tasks to complete the local RTX 4060 study

Status snapshot: **2026-10-01, 20:47 Asia/Singapore (UTC+08:00)**. This is the
current actionable checklist after review call 1 through final delivery.
Refresh the status from persisted run evidence as work advances.

See the [detailed local project plan](local-rtx4060-project-plan.md) for fixed
experiment requirements, the [project roadmap](project-roadmap.md) for scope,
the [runbook](runbook.md#full-study-on-the-local-rtx-4060) for PowerShell
commands, and [DEVLOG.md](../DEVLOG.md) for chronological evidence.

## Immediate path from this checkpoint

1. Refreeze the changed candidate. Run the complete candidate checks against
   those exact bytes, repeat the parent adversarial pass, prepare the neutral
   re-review closure matrix, and pass machine readiness. Preserve the existing
   assurance unit and its **1/3 calls used**; two calls remain.
2. Obtain an accepted fresh final-strict verdict before sustained inference.
   Then commit and push the reviewed branch and verify its remote SHA.
3. Create a fresh full-run directory, import the exact validated 110-row A100 ZIP
   once, and confirm 110 imported + 12,111 pending before generation.
4. Start the resumable RTX 4060 run with native Windows Transformers,
   `--fast-start-unqualified`, and no runtime deadline. Do not use vLLM or Ray
   for this approved Windows run.
5. Resume and monitor until all 12,221 rows are valid, then generate reports,
   verify a hash-checked backup restore, publish final documentation, and close
   the checklist.

The detailed checkboxes below are the execution record. Do not mark a phase
complete until its listed evidence exists.

## Goal and fixed settings

Account for all **12,221 selected questions**:

| Origin | Rows | Current status |
| --- | ---: | --- |
| A100 archive import | 110 | Validated; not imported into the local run |
| Local RTX 4060 | 12,111 | Not started |
| **Total** | **12,221** | **Incomplete** |

Preserve Qwen2.5-Math-7B-Instruct revision
**ef9926d75ab1d54532f6a30dd5e760355eb9aa4d**, NF4 double quantization with
FP16 compute, one greedy answer, eight independently seeded samples, temperature
0.7, top-k 50, top-p 1.0, full question/reference content, and a 1,024-token
output cap.

## Selected local execution path

Use native Windows Transformers with one model resident on the 8 GiB RTX 4060
Laptop GPU, a CPU coordinator, small batches, durable checkpoints, and OOM
recovery. Keep the run resumable with no default deadline. Batch reduction and
offloaded KV cache may recover from memory pressure; do not change precision,
lower memory reserves, truncate prompts, reduce samples, or change the output
cap.

**Do not use vLLM or Ray for this Windows run.** That path remains experimental
and unqualified. The approved goal is safe completed-row throughput, not
filling all VRAM. The smoke rate is not a full-run ETA.

## Verified current state

- The historical unittest suite passed **256 tests** for source snapshot
  **ae37dee3b84ecd691e46815c80bda48b34197c9bbd53dc83f3d1385a50e96d91**.
  Reviewer call 1 found that the receipt predates later in-scope documentation
  edits. It is not a full-gate receipt for the frozen candidate; rerun the
  complete checks after the fixed-input change and documentation are stable.
- The current fixed-input source passed **262 tests in 192.218 seconds**;
  compileall, pip check, runner/report help, diff check, and whitespace scan
  passed. Source snapshot:
  **86294937b66f00a0dc729435e1e243292d257ec840fa5a46c02f7a37c513d1b8**.
  The final documentation bytes are being rebound to the candidate before
  the next review packet.
- The current smoke completed **9/9 rows**, zero failed, zero pending, across
  GSM8K, MATH, and GSM-Plus, including all seven MATH subjects. It produced 18
  samples (two per smoke row) and exercised the 1,024-token cap; three samples
  reached the cap.
- Smoke directory:
  %LOCALAPPDATA%\TeacherReliability\runs\smoke-rtx4060-20261001-173011.
  Elapsed time was **22.60 minutes**, about **23.87 rows/hour** for this
  two-sample smoke only. No full-study ETA is available.
- Saved after-load telemetry: **4,816 MiB available host RAM**, **5,311 MiB
  allocated / 6,082 MiB reserved GPU memory**, and **1,314 MiB GPU free**.
  Final persisted settings were greedy batch 1 and sample batch 1, both using
  offloaded KV cache. Five adaptations were recorded; one low-host-RAM attempt
  stopped safely and later resumed successfully.
- The final smoke progress heartbeat at 17:52:56 SGT recorded **4,143 MiB
  available host RAM**, **5,324 MiB allocated / 6,094 MiB reserved GPU
  memory**, and **896 MiB GPU free**. This was below the normal 1,024 MiB GPU
  headroom target but above the 512 MiB emergency floor for offloaded KV
  cache. It does not qualify a long run; watch both memory sources during
  fast-start execution.
- At 18:51 SGT the GPU was idle at **0 MiB used** with no study worker; host RAM
  was **6,626 MiB available**. Recheck both immediately before model loading.
- Archive:
  C:\Users\user\Downloads\full-a10080-20260930T131901Z.zip.
  SHA-256:
  **7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82**.
  File-backed validation accepted **110 complete rows**, eight samples per row,
  and matching selected content, revisions, and seeds. It is not imported yet.
  Three counters absent from its historical schema remain unknown, not zero.
  Its A100 metadata is reported provenance, not independent hardware
  attestation.
- Assurance unit:
  **Teacher-reliability-Distill-LLM/inference-optimization/vllm-hybrid**.
  Generation 0 is open and not review-ready; **1/3** reviewer calls is used,
  with no active reservation. Call 1 returned **fix-first** with an input
  identity blocker. Its completion was recorded under the exclusive sidecar
  lock. This continuation's parent runtime is verified as GPT-6-Sol/max; call
  1's earlier parent runtime remains historical. The call 1 candidate was
  frozen, but is not accepted; nothing is committed or pushed. The sustained
  run has not started.
- The owner-approved fast path waives the 12-profile comparison and
  24-question benchmark. Label the full run **fast-start, unqualified**.
  Preserve earlier failed qualifications as historical evidence.

## Tasks from now to completion

### 0. Close reviewer call 1

- [x] Record the fix-first verdict, reviewer runtime, consumed call, and
      reservation release in the existing durable assurance unit. The unit is
      open, review-ready is no, and two predeclared calls remain.
- [x] Add failing behavioral tests for an alternate self-consistent A100 ZIP,
      wrong model or dataset revisions, and a non-42 full-study seed. Observe
      assertion RED before changing production code.
- [x] Enforce the validated archive SHA-256 and fixed study revisions/seed at
      full local launch. Make full reports reject altered study inputs. Run
      focused GREEN/REFACTOR checks and record the evidence in DEVLOG.md.
- [x] Refresh the assurance ledger's historical header for the current local
      scope and record the call 1 parent-runtime limitation with the new Sol
      continuation proof. Keep the original call 1 history intact.
- [x] Update this checklist, roadmap, runbook, and plan with the final fixed
      launch command, exact source identity, and current review accounting.

### 1. Prepare the next candidate

- [x] Reconcile smoke-running statements in the roadmap, runbook, and process
      log with the completed smoke and this checklist.
- [x] Inspect all staged, unstaged, and untracked in-scope files. Keep the
      experimental A100/vLLM path separate from the selected local route.
      Preserve the smoke manifest, predictions, checkpoints, and archive
      validation evidence.
- [x] Complete and record the parent adversarial pass: inference and sampling,
      OOM and host-memory recovery, archive provenance, worker crash/restart,
      locks, interrupted writes, resume, retry-failed, deadlines, incomplete
      reports, and changed-to-unchanged interactions. Include counterexamples,
      negative paths, and test-sensitivity evidence.
- [x] Freeze the call 1 candidate and acceptance evidence in the existing
      assurance unit. Its review found a behavior blocker and stale full-gate
      receipt; preserve its identity and review budget.
- [x] After all behavior and documentation edits, refreeze the complete
      candidate and run one current candidate-bound full check pass. Resolve
      any failures with focused checks before another complete pass.

### 2. Pass the next final-strict review

- [ ] Prepare a neutral closure matrix for all four call 1 findings, including
      the fixed-input code path, current full-gate receipt, Sol parent proof,
      and internally consistent ledger state. Repeat the parent adversarial
      pass and complete the canonical readiness record, candidate manifest,
      acceptance map, and applicable delivery-manifest decision. Run the
      machine validator and save its passing proof.
- [ ] Under the existing exclusive coordination lock, reserve the next review
      attempt and obtain a fresh read-only Solweaver reviewer verdict with the
      required GPT-6 Sol runtime.
- [ ] Resolve findings within the remaining two-call budget. Refreeze and
      rerun readiness after any candidate change. The sustained run and branch
      push remain behind an accepted final-strict verdict.

### 3. Publish the reviewed implementation

- [ ] Commit the reviewed code, tests, setup/runbook, and process documents on
      codex/local-rtx4060.
- [ ] Push the branch to
      https://github.com/Khawpuneiei/Teacher-reliability-Distill-LLM and verify
      the remote commit SHA.
- [ ] Record the commit and remote SHA in the plan and process log. Keep model
      files, virtual environments, and live runtime directories out of Git;
      publish reproducible commands and a compact result summary.

### 4. Import the archive and launch the full run

- [ ] Confirm no study worker is active. Recheck system RAM and nvidia-smi.
      Keep at least **2.8 GiB available host RAM while the model is loaded**,
      the existing 2 GiB offloaded-cache reserve, and GPU admission rules.
- [ ] Create a new run directory under
      %LOCALAPPDATA%\TeacherReliability\runs. Do not reuse the smoke directory
      or an old qualification.
- [ ] Import the exact validated ZIP once. Before inference, confirm **110
      imported, 12,111 pending, zero failed, and 12,221 unique IDs**. Check
      that the run preserves archive/source hashes, original manifest, row
      provenance, seeds, and a distinct local run identity.
- [ ] Use the runbook command for native transformers-local, seed 42,
      --import-run-zip, and --fast-start-unqualified. Leave the runtime
      deadline unset.
- [ ] Save the run ID, exact command, code revision, initial memory readings,
      logs, and output directory in the process record.

### 5. Monitor and resume to completion

- [ ] Monitor imported/completed/failed/pending counts, active request, tokens,
      per-dataset throughput, checkpoints, host RAM, and GPU memory. Calculate
      a dataset-weighted ETA only from measured local rates.
- [ ] Pause gracefully if memory or run health requires it. Resume the same
      directory with the same archive, seed, and run identity. Do not start a
      second worker against the run lock.
- [ ] Diagnose failures before using --retry-failed. Preserve original IDs and
      seeds, and verify retries do not duplicate completed rows or samples.
- [ ] Record automatic batch/cache changes, worker restarts, interruptions,
      recoveries, and unrecoverable example failures.
- [ ] Continue until every required ID is accounted for. Any missing, failed,
      or pending required row keeps the study incomplete.

### 6. Validate results and reports

- [ ] Verify **110 imported A100 rows plus 12,111 local rows**, unique IDs, and
      zero missing, failed, or pending examples.
- [ ] Validate source question/reference, one greedy answer, eight sample
      indices and original seeds, token limits, aligned token/log-probability/
      entropy vectors, confidence values, consistency labels, and execution
      provenance for every row.
- [ ] Generate a complete combined report and separate A100-import and RTX
      4060 metrics. Verify dataset and execution-origin counts.
- [ ] Apply scorable-coverage and recommendation-eligibility rules. Keep old
      unknown counters unknown; do not claim statistical superiority without
      uncertainty intervals.
- [ ] Record measured per-dataset throughput, active time, wall time, and the
      limitation that sampled text may differ across GPU environments.

### 7. Back up, publish results, and close

- [ ] Back up the raw run, imported archive, manifests, checkpoints, logs, and
      reports. Record SHA-256 hashes.
- [ ] Restore that backup to a separate location and verify manifests,
      checkpoints, prediction counts, and report readability.
- [ ] Update this checklist, project-roadmap.md, runbook.md, and DEVLOG.md with
      final counts, runtime, throughput, recovery history, report/backup
      locations and hashes, remote SHA, and remaining limitations.
- [ ] Push the final documentation and compact result summary to the same
      branch. Keep large raw runtime artifacts in the verified backup unless
      the repository defines a suitable data-artifact location and size policy.
- [ ] Mark the project complete only after row accounting, complete reports,
      backup restore, and final status all pass.

## Definition of done

All **12,221** selected questions are accounted for as **110 validated A100
imports plus 12,111 RTX 4060 generations**, with no missing, failed, or pending
rows. Every row has one greedy result and eight valid seeded samples. Combined
and execution-origin reports are complete, the raw run has a hash-verified
tested restore, and reviewed code, reproducible instructions, process log, and
final result summary are pushed.

The smoke is complete. Candidate review, publication, archive import, the
sustained run, full reports, and verified backup remain outstanding.
