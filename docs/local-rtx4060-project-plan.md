# Mini teacher reliability and distillation plan

## Accepted scope

The project is a 111-question teacher-reliability run on the local RTX 4060
and a four-condition student distillation comparison on 640 held-out
questions. The code supports only this mini study (`mini_12h`) plus a nine-row
`smoke` pipeline check.

## Teacher experiment

| Setting | Value |
| --- | --- |
| Hardware | NVIDIA RTX 4060 Laptop GPU, 8 GiB VRAM |
| Teacher | `Qwen/Qwen2.5-Math-7B-Instruct` |
| Teacher revision | `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d` |
| Quantization | NF4 double quantization, FP16 compute |
| Profile | `mini_12h`; seed 42 |
| Selected data | GSM8K 40; MATH 21 (3 from each of 7 subjects); GSM-Plus 50 (one per seed question) |
| Per-question output | One greedy answer plus 8 independently seeded samples |
| Output limit | 1,024 new tokens per generation |
| Result | 111 complete rows, 0 failed, 0 pending; RTX 4060 execution only |

The output retains token-level entropy and selected-token log probability,
answer alignment, correctness and parse state, self-consistency agreement,
majority-answer results, and generation termination. The run completed after
resuming two examples that had failed the available-host-RAM check. Its
measured throughput was about 19.4 rows/hour; this is a measurement of this
mini run, not an ETA for a larger profile.

## Student distillation experiment

- Student: `Qwen/Qwen2.5-0.5B`, revision
  `060db6499f32faf8b98477b0a26969ef7d8b9987`.
- LoRA rank 16 on all linear layers; two epochs; learning rate `2e-4`; train
  seed 0; effective batch size 16.
- `kd_all` uses all complete teacher responses. `kd_gated` keeps responses
  matching the self-consistency majority when its share is at least 0.75; it
  does not use gold labels. `kd_oracle` keeps gold-correct responses as a
  label-using reference. `base` has no student training.
- All four conditions are evaluated on the same 640 held-out questions:
  GSM8K 300, MATH 140 (20 per subject), and GSM-Plus 200. Training questions
  and GSM8K/GSM-Plus seed-question groups are excluded from evaluation.
- Evaluation is greedy, capped at 512 new tokens, and graded on the first
  closed `\\boxed{}` answer. Paired accuracy differences use 5,000 bootstrap
  resamples over the shared question IDs.

## Results and interpretation

| Condition | Questions used for training | Training responses | Overall held-out accuracy |
| --- | ---: | ---: | ---: |
| `base` | 0 | 0 | 25.5% |
| `kd_all` | 111 | 955 | 37.8% |
| `kd_gated` | 95 | 824 | 38.8% |
| `kd_oracle` | 97 | 797 | 37.0% |

The `kd_gated` minus `kd_all` estimate is +0.9 percentage points (95% paired
bootstrap interval: −2.0 to +4.1). The mini study does not establish a benefit
from the self-consistency gate. Distillation results are exploratory because
there is one training seed, one student size, a small teacher set, different
training-set sizes across conditions, and 11–20% student-answer truncation.
The teacher reliability report also remains exploratory and does not select a
deployment gate.

## Artifacts and provenance

- Teacher summary, manifest, metrics, plots, and hash receipt:
  [`../results/mini12h-rtx4060-20261001/`](../results/mini12h-rtx4060-20261001/).
- Student script, four per-question evaluation files, results, and report:
  [`../scripts/mini_distill.py`](../scripts/mini_distill.py) and
  [`../results/mini-distill-20261002/`](../results/mini-distill-20261002/).
- Raw teacher run archive:
  `mini12h-rtx4060-20261001-230236.tar.gz`; SHA-256
  `06ed0976394cc485ee5425160fb08ea0ab59e28ce14e41a42d7bc599a4cd1014`.
  The session report records the separate-location restore check and the
  matching `predictions.jsonl` SHA-256.
- Raw teacher data, model weights, and caches stay outside the Git repository.
  The input guard checks the pinned model and dataset commits, complete status,
  exact profile, selected IDs, per-row revisions, dataset mix, and eight
  expected seeded samples before distillation starts.

## Completion criteria

The mini scope is complete when:

1. The 111-row teacher run and all four 640-question student conditions are
   accounted for in committed reports and outputs.
2. Automated checks reject incomplete or mismatched teacher inputs and pass
   on the archived run.
3. The runbook reproduces the teacher and student settings from saved
   data/model caches and records that generated text or metrics can vary with
   the hardware and software stack.
4. Tests, the process log, and the GitHub push are complete.
