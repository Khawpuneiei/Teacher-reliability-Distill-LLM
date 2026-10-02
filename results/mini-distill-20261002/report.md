# Mini distillation follow-up (2026-10-02)

This four-condition, 640-question comparison is the student portion of the
accepted mini-study scope; it is exploratory and does not establish a gate
effect.

Teacher data: run `mini12h-rtx4060-20261001-230236` (Qwen2.5-Math-7B-Instruct, 111 questions, greedy + 8 samples each).
Student: `Qwen/Qwen2.5-0.5B` base (revision `060db6499f32faf8b98477b0a26969ef7d8b9987`), LoRA r=16 on all linear layers, 2 epochs, lr 2e-4, effective batch 16, seed 0.
Evaluation: 640 held-out test questions (GSM8K 300, MATH 140 = 20 per subject, GSM-Plus 200 one per seed), disjoint from training questions and from any GSM8K/GSM-Plus seed question used in training. Greedy decoding, 512 new tokens, graded with the repository's Math-Verify grader on the first `\boxed{}` answer (same rule for every condition).

| Condition | Training responses (questions) | GSM8K | GSM-Plus | MATH | Overall | Macro |
|---|---:|---:|---:|---:|---:|---:|
| base (no training) | 0 | 28.3% | 22.5% | 23.6% | 25.5% | 24.8% |
| kd_all | 955 (111) | 46.7% | 33.0% | 25.7% | 37.8% | 35.1% |
| kd_gated (majority share ≥ 0.75, answer = majority; no gold labels) | 824 (95) | 46.7% | **36.0%** | 25.7% | **38.8%** | **36.1%** |
| kd_oracle (gold-correct responses only; reference) | 797 (97) | 44.3% | 33.0% | 27.1% | 37.0% | 34.8% |

Paired bootstrap (5,000 resamples over the 640 questions), overall accuracy difference with 95% interval:

| Comparison | Δ | 95% CI |
|---|---:|---|
| kd_all − base | +12.3 pts | [+8.3, +16.3] |
| kd_gated − base | +13.3 pts | [+9.4, +17.2] |
| kd_gated − kd_all | +0.9 pts | [−2.0, +4.1] |
| kd_oracle − kd_all | −0.8 pts | [−3.9, +2.3] |
| kd_oracle − kd_gated | −1.7 pts | [−4.8, +1.4] |

## Findings

- Distillation works: training on only 111 teacher-answered questions raises held-out student accuracy by about 12–13 points, mostly on GSM8K and GSM-Plus; MATH barely moves.
- The self-consistency gate is directionally better than no gate (+0.9 pts) and slightly fewer training responses, but the interval includes zero; this mini study cannot show a gate effect.
- Gold-label filtering did not beat the label-free gate, consistent with the teacher already being mostly right on these questions (the gate drops 14% of responses; the oracle drops 17%).

## Limits

One training seed, one student size, tiny training set (111 questions), conditions have different training-set sizes with fixed epochs, and the 512-token cap truncated 11–20% of student answers. Treat as exploratory. Raw per-question outputs: `eval_<condition>.jsonl`. Script: `scripts/mini_distill.py`.
