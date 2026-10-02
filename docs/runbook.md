# Mini study runbook (Windows, RTX 4060)

This is the operating guide for the accepted project scope: a 111-row teacher
run and its four-condition, 640-question distillation comparison. The runner
supports the `mini_12h` profile and a nine-row `smoke` check. The measured runs and results are already committed; use a fresh output folder
only when intentionally reproducing them.

## 1. Environment

Use Windows PowerShell, Python 3.10+, an NVIDIA driver, and the RTX 4060 Laptop
GPU. Setup and inspect the pinned environment:

```powershell
.\scripts\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
python -m teacher_reliability.env_check
nvidia-smi
```

The first run needs the pinned Qwen2.5-Math-7B-Instruct model and dataset
commits in the Hugging Face cache. The commands below use `--local-files-only`
so an experiment will stop instead of silently resolving newer remote files.

## 2. Teacher run (`mini_12h`)

The completed reference run is `mini12h-rtx4060-20261001-230236`. It selected
40 GSM8K, 3 examples from each of the 7 MATH subjects, and 50 GSM-Plus seed
questions. It uses seed 42, NF4 double quantization with FP16 compute, one
greedy answer, eight independently seeded samples, and a 1,024-token limit.
It completed 111/111 rows with zero failures or pending rows. There was no A100
row import.

The 12-hour deadline is explicit and persisted; `mini_12h` alone only describes
the selection profile and does not enforce wall time. Output and checkpoints
stay outside OneDrive under `%LOCALAPPDATA%\TeacherReliability\runs`.

```powershell
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$RunDir = Join-Path $env:LOCALAPPDATA ('TeacherReliability\runs\mini12h-rtx4060-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$null = New-Item -ItemType Directory -Path $RunDir -Force
& $Python -u -m teacher_reliability.run `
  --backend transformers-local `
  --device-preset rtx4060-8gb `
  --profile mini_12h `
  --seed 42 `
  --local-files-only `
  --max-runtime-hours 12 `
  --run-dir $RunDir
```

Run in the foreground so Ctrl+C can request a checkpointed stop. For an
interrupted run, keep the same folder and profile:

```powershell
& $Python -m teacher_reliability.run --backend transformers-local --device-preset rtx4060-8gb --profile mini_12h --seed 42 --local-files-only --run-dir $RunDir --status
& $Python -m teacher_reliability.run --backend transformers-local --device-preset rtx4060-8gb --profile mini_12h --seed 42 --local-files-only --run-dir $RunDir --stop
& $Python -u -m teacher_reliability.run --backend transformers-local --device-preset rtx4060-8gb --profile mini_12h --seed 42 --local-files-only --max-runtime-hours 12 --run-dir $RunDir --resume
```

Retry recorded failed rows only after checking the error and available system
RAM:

```powershell
& $Python -u -m teacher_reliability.run --backend transformers-local --device-preset rtx4060-8gb --profile mini_12h --seed 42 --local-files-only --run-dir $RunDir --resume --retry-failed
```

Create the teacher calibration report and check row accounting:

```powershell
& $Python -m teacher_reliability.report --run-dir $RunDir
Get-Content (Join-Path $RunDir 'summary.json') -Raw
```

The 111-row report is exploratory. Its metrics do not identify a deployable
teacher-confidence gate.

## 3. Distillation run

The committed distillation used the completed teacher predictions from
`mini12h-rtx4060-20261001-230236`. The local raw teacher run is not stored in
Git. Set `$TeacherRun` to the directory containing that run's
`run_manifest.json` and `predictions.jsonl`. The session report records the
original app-local location and backup checksum.

The script validates the run manifest, exact `mini_12h` profile, all 111 unique
selected IDs, 40/21/50 dataset counts, complete status, pinned teacher and
dataset revisions, each prediction row's revision, and eight correctly seeded
samples per question before it starts training.

`peft==0.15.2` was installed separately from the pinned inference environment:

```powershell
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$Pylib = Join-Path $env:LOCALAPPDATA 'TeacherReliability\pylib'
& $Python -m pip install peft==0.15.2 --no-deps --target $Pylib
$env:PYTHONPATH = if ($env:PYTHONPATH) { $Pylib + [IO.Path]::PathSeparator + $env:PYTHONPATH } else { $Pylib }
```

Use a new output directory to reproduce the experiment rather than overwriting
the committed summary. Defaults match the saved run: evaluation sizes 300/20
per MATH subject/200, eval seed 43, student train seed 0, two epochs, LoRA
rank 16, micro-batch 2 with gradient accumulation 8, and 512 new evaluation
tokens.

```powershell
$TeacherRun = Read-Host 'Path to the completed mini12h-rtx4060-20261001-230236 run directory'
$DistillDir = Join-Path $env:LOCALAPPDATA ('TeacherReliability\mini-distill-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$null = New-Item -ItemType Directory -Path $DistillDir -Force
& $Python scripts\mini_distill.py --teacher-run $TeacherRun --out-dir $DistillDir
```

The script evaluates `base`, `kd_all`, `kd_gated`, and `kd_oracle` on the same
640 held-out questions and writes one `eval_<condition>.jsonl` file per
condition plus `results.json`. A resumed run skips conditions already present
in its `results.json`; use the same output directory to continue after an
interruption.

## 4. Results and interpretation

The checked-in outputs are under:

- [`../results/mini12h-rtx4060-20261001/`](../results/mini12h-rtx4060-20261001/)
- [`../results/mini-distill-20261002/`](../results/mini-distill-20261002/)

Student overall accuracy was 25.5% for base, 37.8% for `kd_all`, 38.8% for
`kd_gated`, and 37.0% for `kd_oracle`. The `kd_gated` minus `kd_all` estimate
was +0.9 percentage points, with a 95% paired-bootstrap interval from −2.0
to +4.1 points. Treat the gate effect as inconclusive. The experiment used one
training seed and student size, a small teacher set, differently sized
training sets, and a 512-token evaluation cap that truncated 11–20% of student
answers.

Re-running on a different GPU, driver, or library stack may change sampled
text or student metrics even when the recorded seeds and settings match.

## 5. Backup and code checks

The raw teacher backup is
`mini12h-rtx4060-20261001-230236.tar.gz`, SHA-256
`06ed0976394cc485ee5425160fb08ea0ab59e28ce14e41a42d7bc599a4cd1014`. The
session report records that restoration reproduced the 111-line
`predictions.jsonl` and its SHA-256. Keep raw generations and model/data caches
outside Git.

Run the focused mini-input contract tests and the full suite from the
repository root:

```powershell
python -m unittest discover -s tests -p "test_mini_distill.py" -v
python -m unittest discover -s tests -p "test_*.py" -v
```

See the [project plan](local-rtx4060-project-plan.md),
[closeout checklist](remaining-tasks-to-complete-study.md), and
[session report](session-report-2026-10-02.md) for protocol, measured results,
provenance, and limits.
