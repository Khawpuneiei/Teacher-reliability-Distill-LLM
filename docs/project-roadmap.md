# Project roadmap: mini teacher reliability and distillation

Status: complete on 2026-10-02. The project scope is the mini teacher run and
the mini distillation comparison; no larger benchmark is planned in this
repository.

## Goal

Deliver and document the completed local experiment:

| Stage | Required scope | Status |
| --- | --- | --- |
| Teacher reliability | 111 questions: GSM8K 40, MATH 21 (3 per subject), GSM-Plus 50; 8 samples per question; 1,024-token cap | Complete: 111/111, no failures or pending rows |
| Student distillation | Four conditions evaluated on the same 640 held-out questions: GSM8K 300, MATH 140, GSM-Plus 200 | Complete; summaries and per-question outputs are committed |
| Reproducibility and delivery | Scope-specific input validation, tested commands, documented limitations, and remote verification | Complete |

The teacher uses Qwen2.5-Math-7B-Instruct at the pinned revision, NF4 double
quantization, FP16 compute, and seed 42. Distillation uses the Qwen2.5-0.5B
base checkpoint, LoRA, training seed 0, and the four conditions `base`,
`kd_all`, `kd_gated`, and `kd_oracle`. See the
[experiment design](experiment-design.md) for the exact protocol.

## Current results

- Teacher-run throughput was about 19.4 completed rows/hour on the RTX 4060.
  The run completed within its 12-hour deadline after resuming two
  low-host-memory failures. The raw-run backup restore reproduced the saved
  111-line prediction file with the same SHA-256.
- Student overall accuracy was 25.5% for base, 37.8% for `kd_all`, 38.8% for
  `kd_gated`, and 37.0% for `kd_oracle`.
- `kd_gated` minus `kd_all` was +0.9 percentage points, with a paired
  95% bootstrap interval from −2.0 to +4.1 points. The gate effect is
  inconclusive.
- The teacher metrics and student results are exploratory: one small teacher
  subset, one training seed, one student size, and 11–20% evaluation-answer
  truncation at the 512-token student limit.

## Closeout

The full-scale benchmark code (12,221-row profile, A100 archive import,
profile qualification, and the vLLM/hybrid backend) was removed on
2026-10-02; it remains only in Git history. The closeout list is in
[`remaining-tasks-to-complete-study.md`](remaining-tasks-to-complete-study.md),
and run commands are in the [runbook](runbook.md).
