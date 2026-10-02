# Session report: mini teacher-reliability study and mini distillation

Period: 2026-10-01 20:57 – 2026-10-02 05:50 (Asia/Singapore, UTC+08:00).
Branch: `codex/local-rtx4060` on
[Teacher-reliability-Distill-LLM](https://github.com/Khawpuneiei/Teacher-reliability-Distill-LLM).
Operator: Claude Opus 5.5 (Claude Code). The session could not run as GPT-6-Luna,
and no GPT-6-Sol reviewer call was made.

## 1. Summary

| Step | Result |
| --- | --- |
| Candidate code committed and pushed | `df74efe`; remote SHA verified |
| Full 12,221-row run | Not started. Estimated at ~1,000 GPU-hours; the launch was blocked by the session's permission system |
| Downscaled `mini_12h` profile added | `05b947a` |
| Mini teacher run (111 questions) | **111/111 complete**, 0 failed, about 5.6 h |
| Mini teacher report, backup, restore check | `548a7ca` |
| Mini distillation (student training) | 4 conditions on 640 held-out questions, `f88a79a` |

Headline results:

- **Teacher:** on the mini run, the best predictor of whether the teacher's
  answer is correct was self-consistency agreement (how often the 8 samples
  agree with the greedy answer) on GSM8K and MATH (AUROC 0.895 / 0.950).
  Sequence probability was best on GSM-Plus (0.858).
- **Student:** training a 0.5B student on 111 teacher-answered questions raised
  held-out accuracy from **25.5% to 37.8–38.8%**. The self-consistency gate
  scored +0.9 points over no gate, but the 95% interval [−2.0, +4.1] includes
  zero.

## 2. Starting point

The plan ([local-rtx4060-project-plan.md](local-rtx4060-project-plan.md)) called
for 12,221 questions: 110 imported A100 rows plus 12,111 local rows on the RTX
4060 Laptop GPU. Each local row needs one greedy answer and eight samples
(about 109,000 generations). The nine-row smoke run had passed. Final-strict
review, commit/push, archive import, the full run, reports, and backup remained.

## 3. What was done

### 3.1 Verification and publication (skipping long gates by owner direction)

- Ran the full unittest suite: **261/262 passed** in 241.7 s. The one error
  (`test_local_worker`) was environmental: the test checks live host RAM, and
  only 2.47 GiB was available against a 2.44 GiB requirement plus reserve.
- Skipped, as directed: the GPT-6-Sol final-strict re-review, the readiness
  packet, and the 12-profile qualification matrix.
- Committed the whole candidate (47 files) and pushed `codex/local-rtx4060`.
  Remote SHA `df74efe7486046b752793fdfc2e58c6cf9adf1b1` matches the local one.
- Confirmed the A100 ZIP SHA-256
  (`7fa37f4a…4d82`), an idle GPU (0 MiB used), and no active worker.

### 3.2 Full-run attempt

- The detached full-run launch (`--fast-start-unqualified`) was denied by the
  Claude Code permission classifier. It was not retried another way.
- Throughput estimate: about 11.3 rows/hour at batch one, so 12,111 rows need
  roughly **1,000 GPU-hours (about 6 weeks)**. Fixed protocol settings (model,
  precision, 8 samples, 1,024-token cap) cannot be relaxed to shorten it.

### 3.3 Downscaled 12-hour mini study

The owner asked for a 12-hour mini project. Changes:

- Added profile `mini_12h` in `teacher_reliability/selection.py`: 40 GSM8K,
  21 MATH (3 per subject), and 50 GSM-Plus (one per seed question), for
  111 rows. The protocol is unchanged: same model revision, NF4/FP16, seed 42,
  8 samples, temperature 0.7, top-k 50, 1,024-token cap.
- `report.py` labels `mini_12h` exploratory, with no full-study gate
  recommendation.
- Added the launch commands to `docs/runbook.md`.
- Selection, report, and CLI tests: **53 passed**. Pushed as `05b947a`.
- The A100 rows were not imported, because the importer accepts only the
  `full` profile.

### 3.4 Mini teacher run

| Item | Value |
| --- | --- |
| Run ID | `mini12h-rtx4060-20261001-230236` |
| Command | `teacher_reliability.run --backend transformers-local --device-preset rtx4060-8gb --profile mini_12h --seed 42 --local-files-only --max-runtime-hours 12` |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU (CUDA). About 5.3 GiB allocated, 7.07 GiB total used, 65–68% utilization, 85–86 °C with thermal slowdown |
| Settings in use | Greedy batch 1, sample batch 1, offloaded KV cache. Three automatic memory adaptations, because free VRAM was about 900 MiB, under the 1 GiB target |
| First pass | 23:02 → 04:40. 109/111 complete; 2 rows failed with `InsufficientHostMemory` (offloaded cache needs 2.22 GiB; 1.98–2.14 GiB was available) |
| Retry | `--resume --retry-failed` completed both failed rows → **111/111, 0 failed, 0 pending** |
| Throughput | About **19.4 rows/hour**, roughly 1.7× the earlier qualification estimate |

Teacher report (greedy correctness as the outcome):

| Dataset | n scored | Greedy acc. | Majority acc. | AUROC: entropy | AUROC: seq. prob. | AUROC: self-consistency | ECE: self-consistency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GSM8K | 40 | 85.0% | 90.0% | 0.765 | 0.789 | **0.895** | 0.103 |
| GSM-Plus | 48 | 81.2% | 83.3% | 0.846 | **0.858** | 0.736 | 0.102 |
| MATH | 21 | 71.4% | 81.0% | 0.900 | 0.867 | **0.950** | 0.131 |

Two GSM-Plus rows have no parseable reference answer. Grading had zero
timeouts or unknown results.

Outputs and backup:

- Report, metrics CSVs, reliability plots, and manifest copied to
  `results/mini12h-rtx4060-20261001/`.
- Raw run backup: `mini12h-rtx4060-20261001-230236.tar.gz` (3.6 MB), SHA-256
  `06ed0976394cc485ee5425160fb08ea0ab59e28ce14e41a42d7bc599a4cd1014`.
- Restore check: the backup was extracted to a separate folder. Its
  `predictions.jsonl` has 111 lines and the same SHA-256 as the original
  (`1e62bc4a…a5d6`).
- Pushed as `548a7ca`.

### 3.5 Mini distillation

Script: `scripts/mini_distill.py`.

| Setting | Value |
| --- | --- |
| Teacher data | The 111 mini-run rows: greedy + 8 samples each. Truncated or answerless responses dropped, leaving 955 responses |
| Student | `Qwen/Qwen2.5-0.5B` base, revision `060db649…`, LoRA r=16 (alpha 32) on all linear layers |
| Training | 2 epochs, lr 2e-4 with cosine schedule, effective batch 16, bf16, gradient checkpointing, seed 0 |
| Evaluation | 640 held-out questions (GSM8K 300, MATH 140, GSM-Plus 200). Disjoint from training questions and from any GSM8K/GSM-Plus seed question used in training. Greedy decoding, 512 new tokens, graded by the repo's Math-Verify grader |
| Statistics | Paired bootstrap, 5,000 resamples, 95% intervals |

Conditions:

- **base:** no training.
- **kd_all:** all 955 responses.
- **kd_gated:** 824 responses from 95 questions. A response is kept only when
  its answer equals the majority answer and the majority vote share is at
  least 0.75. No gold labels are used.
- **kd_oracle:** 797 responses graded correct against gold. This is a
  reference only; it could not be used without labels.

Results:

| Condition | GSM8K | GSM-Plus | MATH | Overall | Macro |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 28.3% | 22.5% | 23.6% | 25.5% | 24.8% |
| kd_all | 46.7% | 33.0% | 25.7% | 37.8% | 35.1% |
| kd_gated | 46.7% | **36.0%** | 25.7% | **38.8%** | **36.1%** |
| kd_oracle | 44.3% | 33.0% | 27.1% | 37.0% | 34.8% |

| Comparison | Δ overall | 95% CI |
| --- | ---: | --- |
| kd_all − base | +12.3 pts | [+8.3, +16.3] |
| kd_gated − base | +13.3 pts | [+9.4, +17.2] |
| kd_gated − kd_all | +0.9 pts | [−2.0, +4.1] |
| kd_oracle − kd_all | −0.8 pts | [−3.9, +2.3] |
| kd_oracle − kd_gated | −1.7 pts | [−4.8, +1.4] |

Issues found and fixed:

1. **Empty evaluation pool:** `load_examples` returns only the profile's
   selection. The held-out pool now loads with the `full` profile.
2. **Student never stopped:** the base student never learned `<|im_end|>`,
   and LoRA cannot change that token's embedding. Training now ends each
   response with the model's own `<|endoftext|>`. Every condition is graded
   on text through its first `\boxed{}`, the same rule for all.
3. **CUDA OOM on `kd_oracle`:** this happened after three conditions had run
   in one process. Results are saved per condition, so a fresh-process
   resume with identical settings completed it.
4. **`peft` not in the pinned environment:** installed `peft==0.15.2` with
   `--no-deps` into `%LOCALAPPDATA%\TeacherReliability\pylib`. The project
   `.venv` is unchanged.

Outputs are in `results/mini-distill-20261002/`: `report.md`, `results.json`,
`distill.log`, and per-question `eval_<condition>.jsonl`. Pushed as `f88a79a`.

## 4. Conclusions

1. The teacher's agreement-based confidence (self-consistency) separated right
   from wrong answers well on GSM8K and MATH, and entropy did well on MATH.
   With 21–50 questions per dataset these are indications only.
2. Distillation from just 111 teacher-answered questions clearly improves a
   0.5B student, by about 12–13 points.
3. The self-consistency gate is no worse than using all data and slightly
   better in point estimate, but this mini study cannot establish a gate
   effect. The teacher was mostly correct, so the gates removed only 14–17%
   of responses.

## 5. Limitations

- The study is a 111-question subset, not the planned 12,221-question study,
  and the A100 rows were not used.
- The run was fast-start and unqualified: no profile matrix and no final-strict
  review. One environmental test error was accepted.
- Distillation used one seed and one student size. Training-set sizes differ
  across conditions while epochs were fixed. The 512-token evaluation cap
  truncated 11–20% of student answers.
- A file at `%LOCALAPPDATA%` lands in an app-virtualized folder on this
  machine. Under Claude Code the physical folder is
  `C:\Users\user\AppData\Local\Packages\Claude_pzs8sxrjxfjjc\LocalCache\Local\TeacherReliability\`.
  Earlier Codex runs are under
  `...\Packages\OpenAI.Codex_2p2nqsd0c76g0\LocalCache\Local\TeacherReliability\`.

## 6. Suggested next steps

1. To test the gate properly, generate more teacher data (for example
   `a100_12h`-sized or larger) and run distillation with 3–5 seeds per
   condition.
2. Raise the student evaluation cap to 1,024 tokens, or use
   Qwen2.5-0.5B-Instruct as the student to reduce truncation.
3. If the full 12,221-row study is still wanted, run it from the runbook
   command in an interactive terminal. Free host RAM first (close ChatGPT,
   Codex, Epic Games) and keep the laptop cool.

## 7. Commits

| Commit | Content |
| --- | --- |
| `df74efe` | Local RTX 4060 backend, archive import, fast-start path (candidate) |
| `05b947a` | `mini_12h` profile, report label, runbook section |
| `548a7ca` | Mini teacher-run results, DEVLOG entry, backup hash |
| `f88a79a` | Mini distillation script, results, DEVLOG entry |
