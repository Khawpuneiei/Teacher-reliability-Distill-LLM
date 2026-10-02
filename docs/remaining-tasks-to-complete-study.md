# Mini study closeout checklist

This checklist covers the 111-row teacher run and the 640-question
distillation comparison, which are the whole project scope.

## Completed experiment

- [x] Run `mini_12h` with seed 42, Qwen2.5-Math-7B-Instruct NF4/FP16, one
      greedy output, eight samples, and a 1,024-token cap.
- [x] Account for all 111 teacher rows: 40 GSM8K, 21 MATH (three in each
      subject), and 50 GSM-Plus; zero failures and zero pending rows.
- [x] Resume the two low-host-memory failures and save the final run.
- [x] Create and restore-check the raw teacher backup; record its SHA-256.
- [x] Train and evaluate `base`, `kd_all`, `kd_gated`, and `kd_oracle` on the
      same 640 held-out questions.
- [x] Save per-question predictions, summaries, paired bootstrap results,
      reports, manifests, and teacher calibration artifacts under `results/`.
- [x] State the gate result as inconclusive and disclose the single-seed,
      small-data, student-size, and output-truncation limits.

## Code and delivery closeout

- [x] Remove the 12,221-row profile, A100 archive import, qualification
      matrix, fast-start path, and vLLM/hybrid backend from code, tests, and
      docs; keep only the `smoke` and `mini_12h` profiles.
- [x] Confirm the distillation input guard requires the exact completed
      `mini_12h` manifest, pinned model/dataset revisions, 111 unique selected
      predictions, the expected dataset mix, and all eight original sample
      indices and seeds.
- [x] Run the repository-wide unittest suite (161 tests pass) and regenerate
      the teacher report from a copy of the real run with identical metrics.
- [x] Audit the README, roadmap, plan, runbook, data contract, experiment
      design, and session report for consistent mini-scope commands.
- [x] Commit and push the code, tests, docs, and compact outputs to `main`
      and verify the GitHub SHA.
- [x] Record the scope cleanup and verification in `DEVLOG.md`.

## Definition of done

The project is delivered when the mini-teacher input contract is enforced,
the checked-in code and runbook support the documented teacher and student
protocols, committed outputs account for all required examples, validation
are recorded, and the repository is pushed. Extra training seeds and
additional student sizes are possible extensions, not requirements.
