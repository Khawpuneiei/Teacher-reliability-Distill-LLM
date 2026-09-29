# Teacher Reliability and Calibration

This repository implements the attached Apply 1 study: **when should a
distillation pipeline trust a math teacher?** It measures whether the
confidence signals of `Qwen/Qwen2.5-Math-7B-Instruct` predict answer
correctness on GSM8K, a deterministic MATH test subset, and GSM-Plus OOD.
The [original supplied brief](docs/apply-1-teacher-reliability.md) is preserved
for provenance; the detailed protocol and runbook record the implementation
decisions needed to make it executable.

## Status

The project includes a pinned local CUDA environment, deterministic run
profiles, resumable per-example checkpoints, and a report generator. The local
RTX 4060 Laptop GPU (8 GiB VRAM) passed the Python/CUDA/bitsandbytes environment
check, including the pinned Hugging Face Xet transfer helper. Dataset loading
was verified against current Hub commits. A 9-row local smoke run completed
with the 4-bit NF4 teacher, and its report deliberately gives no KD-gating
recommendation because smoke is not study evidence. The full evaluation has not
run. In the recorded setup session, the local Vast.ai CLI authenticated and the
local SSH public key was registered. An A100 instance and either A100 workload
have not yet run.

## Study at a glance

1. Load the requested Qwen math teacher in 4-bit inference mode.
2. Greedily solve held-out GSM8K test, a stratified MATH test subset, and
   GSM-Plus OOD examples.
3. Save generated token IDs, per-token entropy and selected-token log
   probabilities, sequence log-probability, answer-token probability, parsed
   answer, and correctness. Full vocabulary probability matrices are used
   transiently during scoring and are not stored.
4. Draw `k=8` stochastic answers at temperature `0.7`; measure the share
   agreeing with the greedy answer and the ordinary majority-vote result.
5. Produce reliability diagrams, an ECE table, AUROC comparisons, coverage
   and parse diagnostics, and a result-grounded signal recommendation.

The primary AUROC comparison uses one shared label: **whether the greedy
answer is correct**. This makes normalized entropy confidence and
self-consistency agreement directly comparable. The majority-vote answer is
also graded and summarized separately. See
[`docs/experiment-design.md`](docs/experiment-design.md) for exact formulas
and limitations.

| Profile | Rows | Samples per row | Max new tokens | Purpose |
|---|---|---:|---:|---|
| `smoke` | 1 GSM8K, 1 per MATH subject, 1 GSM-Plus | 2 | 1,024 | Pipeline check; not evidence |
| `pilot` | 50 GSM8K, 8 per MATH subject, 100 GSM-Plus | 8 | 512 | Runtime and coverage check |
| `a100_12h` | 150 GSM8K, 50 per MATH subject, 220 GSM-Plus | 8 | 1,024 | About 12 hours under the runbook estimate |
| `a100_24h` | 300 GSM8K, 50 per MATH subject, 790 GSM-Plus | 8 | 1,024 | About 24 hours under the runbook estimate |
| `full` | All GSM8K test, 50 per MATH subject, all GSM-Plus test | 8 | 1,024 | Intended evaluation |

The A100 profiles retain all seven MATH subject subsets and eight samples per
example while reducing GSM8K and GSM-Plus rows. They select one GSM-Plus
variation per seed question. Their reports are exploratory and never make a
full-study KD-gating recommendation. See the [runbook](docs/runbook.md) for
the measured smoke baseline and runtime assumptions. NVIDIA A100s come in
40 GB and 80 GB variants; the actual GPU model and memory are recorded in each
run manifest ([NVIDIA A100 specifications](https://www.nvidia.com/en-us/data-center/a100/)).

The current GSM-Plus Hub snapshot has one `test` split; pilot rows are selected
from that split by stable hash and grouped by seed question. It does not use
the older `testmini` layout. See the [runbook](docs/runbook.md) for exact
counts and the later Vast.ai steps. A smoke or pilot run never receives a
final-study KD gate recommendation.

Loading rejects an empty requested split, including any individual MATH
subject. A full report requires GSM8K, MATH, and GSM-Plus even if its saved
selection contains fewer benchmarks; `summary.json` lists missing datasets
and suppresses the recommendation.

## Local setup

On Windows PowerShell with Python 3.10+ and a CUDA-capable NVIDIA driver:

```powershell
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
python -m teacher_reliability.env_check
```

The pinned `hf_xet` package enables Hugging Face's Xet-backed large-file
transfers, including this model's checkpoint shards. See the [Hugging Face Xet
storage guide](https://huggingface.co/docs/hub/xet/using-xet-storage).

The local reproduction was exercised on an 8-GiB RTX 4060 Laptop GPU. See the
A100 runbook estimates before choosing the larger time-targeted workloads;
they preserve protocol settings but evaluate smaller samples.

Start with a small local smoke run; public data and model files download to
the configured Hugging Face cache on first use:
Use an unused run directory; choose a different name if the example path
already exists locally.

```powershell
python -m teacher_reliability.run --profile smoke --run-dir outputs\smoke-reviewed
python -m teacher_reliability.report --run-dir outputs\smoke-reviewed
```

On an A100, choose one bounded profile and a new run directory:

```bash
python -m teacher_reliability.run --profile a100_12h --run-dir outputs/a100-12h
python -m teacher_reliability.report --run-dir outputs/a100-12h
```

For the longer workload, use `a100_24h` and a distinct directory. The profiles
are workload estimates, not enforced wall-clock timeouts. Host speed, long
answers, or first-time downloads can extend a run. Completed rows remain
resumable only under the recorded profile and machine identity.

The run command prints its exact output directory. To resume an interrupted
run, pass that same path with `--resume`. Resume loads the recorded model and
dataset commits even if upstream `main` has advanced. The manifest also binds
selected IDs and input content, decoding settings, package versions, source
code, hardware, and run status. A protocol, package, code, or recorded
Python/platform/GPU change requires a new run directory.
This includes the pinned `tokenizers==0.21.4` implementation; its version is
recorded alongside Transformers in the manifest and resume signature.

## Checks

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
python -m teacher_reliability.env_check
```

Unit tests use small fixtures and do not download model weights or benchmark
rows.

## Data and generated artifacts

Per-example generations, score vectors, figures, and result tables live under
ignored `outputs/`. Hugging Face model and dataset caches use the configured
Hugging Face cache path outside this repository. The repository contains code,
protocol, synthetic fixtures, and schemas only. Review
[`docs/data-contracts.md`](docs/data-contracts.md) before importing any data.

## Primary sources

- [Qwen2.5-Math-7B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
- [GSM8K dataset card](https://huggingface.co/datasets/openai/gsm8k)
- [Hendrycks MATH dataset card](https://huggingface.co/datasets/EleutherAI/hendrycks_math) and [original MATH repository](https://github.com/hendrycks/math)
- [GSM-Plus dataset](https://huggingface.co/datasets/qintongli/GSM-Plus) and [official GSM-Plus repository](https://github.com/qtli/GSM-Plus)
- [Transformers bitsandbytes quantization](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
- [Transformers generation outputs](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Hugging Face Math-Verify](https://github.com/huggingface/Math-Verify)
- [GRACE confidence-gated VLM distillation](https://arxiv.org/abs/2601.22709); see the design note for the scope distinction from [In Good GRACEs](https://arxiv.org/abs/2511.02833)
