# Local and Vast.ai runbook

This runbook separates a quick pipeline smoke test, a bounded pilot, and the
full benchmark. The earlier 4–8 GPU-hour T4/L4 estimate was not validated and
does not apply to the current local RTX 4060 plan. With the current Hub sizes,
the full profile evaluates about
12,221 rows (1,319 GSM8K, 350 MATH, and 10,552 GSM-Plus) and makes 109,989
greedy/sample generations at `k=8`, before accounting for the output-token
length. Plan for the possibility that it takes materially longer than the
estimate. The 8-GiB RTX 4060 Laptop GPU passed the environment check, and an
earlier baseline runner completed a nine-row smoke near 7.8 GiB reported VRAM.
That smoke predates the current resumable local backend. Its worker recovery is
still being qualified; do not treat baseline throughput as an ETA or an OOM
attempt as a benchmark result.

## 12-hour mini study on the local RTX 4060 (`mini_12h`)

The full 12,111-row local remainder needs roughly 1,000 GPU-hours at the
measured batch-one rate (about 11.3 rows/hour). The `mini_12h` profile keeps
the full protocol (same model revision, NF4/FP16, seed 42, eight samples,
temperature 0.7, top-k 50, 1,024-token cap) on a deterministic 111-row subset:
40 GSM8K, 21 MATH (3 per subject), and 50 GSM-Plus (one per seed question).
At the measured per-dataset rates this is about 9.3 active hours; a persisted
`--max-runtime-hours 12` deadline bounds it. A100 rows are not imported
(the importer accepts only the `full` profile), and the report labels the
result exploratory, with no full-study gate recommendation.

```powershell
$RunDir = Join-Path $env:LOCALAPPDATA ('TeacherReliability\runs\mini12h-rtx4060-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$null = New-Item -ItemType Directory -Path $RunDir -Force
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$RunnerArgs = @('-u', '-m', 'teacher_reliability.run', '--backend', 'transformers-local', '--device-preset', 'rtx4060-8gb', '--profile', 'mini_12h', '--seed', '42', '--local-files-only', '--max-runtime-hours', '12', '--run-dir', $RunDir)
$RunProcess = Start-Process -FilePath $Python -ArgumentList $RunnerArgs -WorkingDirectory (Get-Location).Path -RedirectStandardOutput (Join-Path $RunDir 'runner.log') -RedirectStandardError (Join-Path $RunDir 'runner-error.log') -WindowStyle Hidden -PassThru
$RunProcess.Id | Set-Content (Join-Path $RunDir 'runner.pid')
```

Status, pause, and resume use the same commands as the full study below
(add `--resume` with the same profile and run directory).

## Full study on the local RTX 4060

The active plan is to finish all 12,221 selected questions on the Windows RTX
4060 Laptop GPU. Import the 110 complete A100 rows from the supplied ZIP and
generate the other 12,111 rows locally. The importer verifies source revisions,
question and reference text, sample indices and seeds, score alignment, and
the archive signature. It copies the original ZIP and exact source manifest
into the new local run's `provenance` directory and gives the local run its own
manifest identity. The source archive SHA-256 is
`7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82`. The
source manifest's completed counter says 100, but the archive contains 110
valid complete rows; the import count comes from validated prediction records.

This backend preserves the Qwen2.5-Math-7B-Instruct commit
`ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`, NF4 double quantization with FP16
compute, eight independently seeded samples at temperature 0.7, top-k 50,
top-p 1.0, and a 1,024-token output cap. Confidence normalization uses the
model's 152,064-output vocabulary. The worker uses native Windows Transformers;
vLLM remains a separate experimental Linux/WSL path. One model stays resident
on the GPU. The CPU coordinator writes checkpoints under
`%LOCALAPPDATA%\TeacherReliability\runs`, outside OneDrive.

### Current owner-directed fast path

The owner approved skipping the exhaustive 12-profile matrix and 24-question
throughput comparison. The native Transformers smoke has already completed;
there is no active study worker.

The smoke run is
%LOCALAPPDATA%\TeacherReliability\runs\smoke-rtx4060-20261001-173011.
It completed **9/9 rows, zero failed, zero pending** at 17:52:56 SGT after
22.60 minutes. It covered GSM8K, MATH across all seven subjects, and GSM-Plus;
it generated 18 samples (two per row), with three samples reaching the
1,024-token cap. Its 23.87 rows/hour rate is smoke-only and is not a full-run
ETA.

The saved manifest records 4,816 MiB host RAM after model load, 5,311 MiB
allocated / 6,082 MiB reserved GPU memory, and 1,314 MiB free GPU memory.
Five memory adaptations ended at greedy batch 1 and sampling batch 1, both
using offloaded KV cache. A low-host-RAM sample attempt stopped safely and
completed after resume. Its final progress heartbeat at 17:52:56 SGT recorded
4,143 MiB available host RAM, 5,324 MiB allocated / 6,094 MiB reserved GPU
memory, and 896 MiB free GPU memory. That was below the normal 1,024 MiB
headroom target but above the 512 MiB emergency floor used with offloaded KV
cache; it does not qualify sustained use. At 18:51 SGT the GPU was idle at 0
MiB used, and host RAM availability was 6,626 MiB; recheck immediately before
full-run model loading. Keep the cache reserve and precision unchanged.

The source ZIP was validated as 110 complete rows with eight samples each and
matching selected content, revisions, and seeds. Its SHA-256 is
7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82. The rows
have not yet been imported. Final-strict review call 1 returned `fix-first` on
the exact-archive and revision pinning gate; the candidate is being corrected.
The branch push, archive import, and sustained run remain pending. See the
[remaining-task checklist](remaining-tasks-to-complete-study.md) for current
gates and the next actions.

Launch the full run in a new directory. `--fast-start-unqualified` records in
the manifest and final report that the long profile matrix was skipped; it
starts with greedy batch 1 and sample batch 2, then uses the existing OOM
backoff. The launch ETA is deliberately empty until measured throughput exists.

```powershell
$Archive = Join-Path $env:USERPROFILE 'Downloads\full-a10080-20260930T131901Z.zip'
$ExpectedArchiveSha = '7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82'
$ActualArchiveSha = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualArchiveSha -ne $ExpectedArchiveSha) { throw 'A100 ZIP SHA-256 differs from approved study archive' }
$RunDir = Join-Path $env:LOCALAPPDATA ('TeacherReliability\runs\full-fast-rtx4060-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$null = New-Item -ItemType Directory -Path $RunDir -Force
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$RunnerArgs = @('-u', '-m', 'teacher_reliability.run', '--backend', 'transformers-local', '--device-preset', 'rtx4060-8gb', '--profile', 'full', '--seed', '42', '--import-run-zip', $Archive, '--fast-start-unqualified', '--run-dir', $RunDir)
$RunProcess = Start-Process -FilePath $Python -ArgumentList $RunnerArgs -WorkingDirectory (Get-Location).Path -RedirectStandardOutput (Join-Path $RunDir 'runner.log') -RedirectStandardError (Join-Path $RunDir 'runner-error.log') -WindowStyle Hidden -PassThru
$RunProcess.Id | Set-Content (Join-Path $RunDir 'runner.pid')
Write-Host "Full run: $RunDir"
```

Check status and request a clean pause from another PowerShell window:

```powershell
$RunDir = (Get-Content (Join-Path $env:LOCALAPPDATA 'TeacherReliability\runs\current_run_dir.txt')).Trim()
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
& $Python -m teacher_reliability.run --status --run-dir $RunDir
& $Python -m teacher_reliability.run --stop --run-dir $RunDir
```

Resume with the same archive, run directory, seed, and fast-start flag:

```powershell
$Archive = Join-Path $env:USERPROFILE 'Downloads\full-a10080-20260930T131901Z.zip'
$ExpectedArchiveSha = '7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82'
$ActualArchiveSha = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ActualArchiveSha -ne $ExpectedArchiveSha) { throw 'A100 ZIP SHA-256 differs from approved study archive' }
$RunDir = (Get-Content (Join-Path $env:LOCALAPPDATA 'TeacherReliability\runs\current_run_dir.txt')).Trim()
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$RunnerArgs = @('-u', '-m', 'teacher_reliability.run', '--backend', 'transformers-local', '--device-preset', 'rtx4060-8gb', '--profile', 'full', '--seed', '42', '--import-run-zip', $Archive, '--fast-start-unqualified', '--run-dir', $RunDir, '--resume')
$RunProcess = Start-Process -FilePath $Python -ArgumentList $RunnerArgs -WorkingDirectory (Get-Location).Path -RedirectStandardOutput (Join-Path $RunDir 'runner.log') -RedirectStandardError (Join-Path $RunDir 'runner-error.log') -WindowStyle Hidden -PassThru
$RunProcess.Id | Set-Content (Join-Path $RunDir 'runner.pid')
```

### Optional exhaustive qualification path (waived for the current run)

The longer procedure below preserves the original 12-profile workflow for
future runs that require profile comparison. It is not required for the
owner-directed fast path.

Before the sustained run, qualify a fixed 24-question cohort: eight GSM8K
questions (including archived A100 examples), eight MATH questions covering
all seven subjects, and eight GSM-Plus questions. The harness measures greedy
batch sizes 1, 2, and 4 against sample batch sizes 1, 2, 4, and 8; it records
score and text differences, per-stage time, GPU/RAM use, and a separate forced
1,024-token memory stress check. Start with only `g1-s1` to check whether the
offloaded-cache path can finish the cohort and keep host RAM above its reserve.
The qualification identity still binds the full 12-profile matrix. After
`g1-s1` completes with all 24 greedy outputs, all 192 samples, and no active
failures, continue in the same output directory without a configuration filter
to run or resume the remaining profiles. If `g1-s1` is incomplete, investigate
its failures and minimum available RAM before starting more profiles. Do not
infer a local ETA from the prior A100 throughput of 110 rows in three hours.

### Historical incomplete qualification (do not resume under the changed source)

The attempt `rtx4060-20261001T073411Z-5e1c5132` ended `incomplete`: all twelve
profiles recorded `InsufficientHostMemory` failures, and no configuration was
recommended. The offloaded-cache admission check needs 2,382,364,672 available
bytes including the approved 2 GiB reserve. During inference, available RAM
fell below that threshold; after the worker exited, 4,624 MiB was free. Do not
repeat the attempt until available RAM can stay above the threshold with the
model loaded; aim for at least 2.8 GiB for fluctuation headroom. Keep the 2 GiB
reserve and experiment settings unchanged.

Do not resume this identity: its source snapshot predates the current code and
it is preserved as historical evidence only. If the optional exhaustive path
is needed for a future run, create a fresh qualification directory and source
identity.

```powershell
$Archive = Join-Path $env:USERPROFILE 'Downloads\full-a10080-20260930T131901Z.zip'
$QualificationDir = Join-Path $env:LOCALAPPDATA ('TeacherReliability\qualifications\rtx4060-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$null = New-Item -ItemType Directory -Path $QualificationDir -Force
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$QualificationArgs = @('-u', '-m', 'teacher_reliability.local_benchmark', '--import-run-zip', $Archive, '--output-dir', $QualificationDir, '--seed', '42', '--configuration', 'g1-s1')
$QualificationProcess = Start-Process -FilePath $Python -ArgumentList $QualificationArgs -WorkingDirectory (Get-Location).Path -RedirectStandardOutput (Join-Path $QualificationDir 'qualification.log') -RedirectStandardError (Join-Path $QualificationDir 'qualification-error.log') -WindowStyle Hidden -PassThru
$QualificationProcess.Id | Set-Content (Join-Path $QualificationDir 'qualification.pid')
Write-Host "Qualification directory: $QualificationDir"
Get-Content (Join-Path $QualificationDir 'qualification.log') -Tail 20
```

The filtered first pass intentionally leaves the overall manifest
`incomplete`, because the other 11 profiles have not run yet. Wait for that
process to exit, then inspect the first profile:

```powershell
$G1Config = Get-Content (Join-Path $QualificationDir 'g1-s1\config.json') -Raw | ConvertFrom-Json
[pscustomobject]@{
    Status = $G1Config.status
    Greedy = @($G1Config.greedy.PSObject.Properties).Count
    Samples = @($G1Config.samples.PSObject.Properties).Count
    ActiveFailures = @($G1Config.failures).Count
    PeakAvailableRamMiB = $G1Config.memory.peak_available_ram_mib
    MinimumAvailableRamMiB = $G1Config.memory.minimum_available_ram_mib
} | Format-List
```

Proceed only if `g1-s1` reports `complete`, 24 greedy outputs, 192 samples,
and zero active failures. Otherwise, use its failure records and RAM
measurements to investigate the problem before starting more profiles.

Then run the full matrix by reusing the exact archive and `$QualificationDir`
and omitting `--configuration`. The same identity resumes unfinished stages
with their original request IDs and seeds:

```powershell
$QualificationArgs = @('-u', '-m', 'teacher_reliability.local_benchmark', '--import-run-zip', $Archive, '--output-dir', $QualificationDir, '--seed', '42')
$QualificationProcess = Start-Process -FilePath $Python -ArgumentList $QualificationArgs -WorkingDirectory (Get-Location).Path -RedirectStandardOutput (Join-Path $QualificationDir 'qualification.log') -RedirectStandardError (Join-Path $QualificationDir 'qualification-error.log') -WindowStyle Hidden -PassThru
$QualificationProcess.Id | Set-Content (Join-Path $QualificationDir 'qualification.pid')
```

The full matrix may take many hours. If it is interrupted, rerun the same
unfiltered command with the same `$QualificationDir`; completed per-configuration
checkpoints are reused and saved failures are retried. Start the full study only
when its manifest reports `status: complete`, a recommended configuration,
and a passing 1,024-token stress check. Save the exact manifest path:

Its identity binds the model and data revisions, ZIP hash, package versions,
RTX 4060 hardware, selected 24 questions, and source-code snapshot. Changes to
Python code or pinned requirements require a fresh qualification.

```powershell
$QualificationManifest = Join-Path $QualificationDir 'qualification_manifest.json'
Get-Content $QualificationManifest
```

Create a new unique run directory and start the sustained worker detached.
Pass the same archive and qualification manifest every time you resume:

```powershell
$Archive = Join-Path $env:USERPROFILE 'Downloads\full-a10080-20260930T131901Z.zip'
$QualificationManifest = Join-Path $QualificationDir 'qualification_manifest.json'
$RunDir = Join-Path $env:LOCALAPPDATA ('TeacherReliability\runs\full-rtx4060-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$null = New-Item -ItemType Directory -Path $RunDir -Force
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$RunnerArgs = @('-u', '-m', 'teacher_reliability.run', '--backend', 'transformers-local', '--device-preset', 'rtx4060-8gb', '--profile', 'full', '--seed', '42', '--import-run-zip', $Archive, '--qualification-manifest', $QualificationManifest, '--run-dir', $RunDir)
$RunProcess = Start-Process -FilePath $Python -ArgumentList $RunnerArgs -WorkingDirectory (Get-Location).Path -RedirectStandardOutput (Join-Path $RunDir 'runner.log') -RedirectStandardError (Join-Path $RunDir 'runner-error.log') -WindowStyle Hidden -PassThru
$RunProcess.Id | Set-Content (Join-Path $RunDir 'runner.pid')
```

There is no runtime cutoff by default. To set a finite persisted deadline,
append `--max-runtime-hours <positive-hours>` to `$RunnerArgs`, for example
`$RunnerArgs += @('--max-runtime-hours', '12')`. Resuming
does not reset that deadline. Check progress or request a clean pause from
another PowerShell window:

```powershell
$RunDir = (Get-Content (Join-Path $env:LOCALAPPDATA 'TeacherReliability\runs\current_run_dir.txt')).Trim()
$Archive = Join-Path $env:USERPROFILE 'Downloads\full-a10080-20260930T131901Z.zip'
# Set this to the exact qualification directory printed when the benchmark started.
$QualificationDir = 'C:\path\to\your\rtx4060-qualification-folder'
$QualificationManifest = Join-Path $QualificationDir 'qualification_manifest.json'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$Python = (Resolve-Path '.\.venv\Scripts\python.exe').Path
python -m teacher_reliability.run --status --run-dir $RunDir
python -m teacher_reliability.run --stop --run-dir $RunDir
$RunnerArgs = @('-u', '-m', 'teacher_reliability.run', '--backend', 'transformers-local', '--device-preset', 'rtx4060-8gb', '--profile', 'full', '--seed', '42', '--import-run-zip', $Archive, '--qualification-manifest', $QualificationManifest, '--run-dir', $RunDir, '--resume')
$RunProcess = Start-Process -FilePath $Python -ArgumentList $RunnerArgs -WorkingDirectory (Get-Location).Path -RedirectStandardOutput (Join-Path $RunDir 'runner.log') -RedirectStandardError (Join-Path $RunDir 'runner-error.log') -WindowStyle Hidden -PassThru
$RunProcess.Id | Set-Content (Join-Path $RunDir 'runner.pid')
```

`--stop` takes effect after the current GPU batch; completed greedy results and
individual samples are already checkpointed. Resume skips them and preserves
the original sample seeds. A CUDA OOM reduces the affected batch, restarts the
worker when needed, and only retries unfinished requests. At batch one it tries
the offloaded KV cache only when available system RAM exceeds the cache estimate
plus a 2-GiB reserve. The normal GPU free-memory target is 1 GiB; the recorded
offloaded-cache recovery profile can attempt a request with 512 MiB free because
its KV cache is offloaded to RAM. The worker releases unused CUDA allocator
blocks before checking headroom and records memory at runtime. These guards
reduce OOM risk but cannot guarantee every request will fit. If the minimum
request profile fails, that example is recorded as failed and other examples
continue. If the model itself cannot load after recovery profiles are
exhausted, the coordinator stops with checkpoints intact.
Use `--retry-failed` with `--resume` only
after fixing or deliberately retrying such failures. A report with any failed
or pending row stays incomplete:

```powershell
python -m teacher_reliability.run --status --run-dir $RunDir
python -m teacher_reliability.run --stop --run-dir $RunDir
python -m teacher_reliability.report --run-dir $RunDir
$RunnerArgs = @('-u', '-m', 'teacher_reliability.run', '--backend', 'transformers-local', '--device-preset', 'rtx4060-8gb', '--profile', 'full', '--seed', '42', '--import-run-zip', $Archive, '--qualification-manifest', $QualificationManifest, '--run-dir', $RunDir, '--resume', '--retry-failed')
$RunProcess = Start-Process -FilePath $Python -ArgumentList $RunnerArgs -WorkingDirectory (Get-Location).Path -RedirectStandardOutput (Join-Path $RunDir 'runner.log') -RedirectStandardError (Join-Path $RunDir 'runner-error.log') -WindowStyle Hidden -PassThru
$RunProcess.Id | Set-Content (Join-Path $RunDir 'runner.pid')
```

The run also stores its path in
`%LOCALAPPDATA%\TeacherReliability\runs\current_run_dir.txt` for later
PowerShell sessions. Replace `$RunDir` with that saved path if needed. Local
and imported records retain `execution_origin`; reports contain combined
metrics and separate A100-import and RTX 4060 metrics and row counts.
The standard full-run mode requires a completed qualification manifest. The
owner-directed `--fast-start-unqualified` mode records that no profile matrix
was completed and begins with the conservative default profiles. It makes no
launch ETA claim; calculate one only after actual per-dataset results are
recorded.

## Historical A100 12–24 hour profiles

These profiles and rough durations belong to the earlier cloud-A100 plan; they
are not part of the active local RTX 4060 workflow. Their time estimate is
unvalidated and must not be used as a local runtime estimate.

The earlier baseline local smoke manifest recorded nine examples, 27 generations (one greedy
plus two samples each), and 673 seconds from start to completion. This includes
a cached-model load; it does not measure downloading the model. Dataset loading
and selection happen before `run_manifest.json` records `started_at`, and
reporting happens after the run. The planning estimate assumes an A100 averages
four times the local run's effective generation throughput. That assumption has
not been measured for this model, quantization, and host. It is not derived from
peak-throughput figures: A100 cards come in 40 GB and 80 GB variants, and their
memory specifications do not predict this decode workload ([NVIDIA A100
specifications](https://www.nvidia.com/en-us/data-center/a100/)).

Both profiles keep the original eight samples, temperature 0.7, and 1,024-token
answer cap. Each selected example therefore uses nine generations:

| Profile | GSM8K | MATH | GSM-Plus | Total rows | Generations | Estimated run time* |
|---|---:|---:|---:|---:|---:|---:|
| `a100_12h` | 150 | 50 per each of 7 subjects (350) | 220 distinct seed questions | 720 | 6,480 | about 11.2 hours |
| `a100_24h` | 300 | 50 per each of 7 subjects (350) | 790 distinct seed questions | 1,440 | 12,960 | about 22.4 hours |

*Estimate: `(profile generations / 27 smoke generations) × 673 seconds / 4`.
This estimates manifest run time from the smoke's elapsed time, including its
cached model load; it is not a direct decode-throughput measurement. The
estimate excludes cold model downloads, dataset loading, and report generation.
It leaves roughly 45–95 minutes for setup and those tasks if the assumption
holds. Shared GPU load or longer generation lengths can use that margin or
extend the run. A target-hour value is metadata, not a hard runtime cutoff.
These deterministic subsets cover all three datasets but are smaller than the
full benchmark, so their reports are exploratory and cannot recommend a
full-study KD gate.

Use Python 3.10, one physical A100 with at least 40 GB visible memory, and a
CUDA 12.1-compatible driver. Check the GPU and available memory with
`nvidia-smi`, then run the environment check before inference. Start a chosen
profile in its own directory:

```bash
python -m teacher_reliability.run --profile a100_12h --run-dir outputs/a100-12h
python -m teacher_reliability.report --run-dir outputs/a100-12h
# Or choose this larger workload in a distinct directory:
python -m teacher_reliability.run --profile a100_24h --run-dir outputs/a100-24h
python -m teacher_reliability.report --run-dir outputs/a100-24h
```

Each manifest records its profile target, selected IDs, commits, software,
GPU, start/end times, and status. Preserve the run folder and resume only with
the same profile and hardware identity. After the first A100 run, update
runtime expectations from the actual elapsed time before launching another.

## Windows local setup

1. Use Python 3.10 and check `nvidia-smi` first. The setup was exercised with
   Python 3.10.4, driver 537.53, and an NVIDIA RTX 4060 Laptop GPU with 8 GiB
   VRAM.
2. Run `scripts/setup_windows.ps1`. It creates a repository-local `.venv`,
   installs pinned PyTorch 2.5.1 for CUDA 12.1 and the pinned research
   dependencies, then installs the project without changing global Python.
   The requirements include pinned `hf_xet` for the model repository's
   Xet-backed weight downloads.
3. Activate `.venv` and run `python -m teacher_reliability.env_check`.
4. Run `python -m unittest discover -s tests -p "test_*.py" -v`.
5. Run the local smoke profile to verify model load, token-score alignment,
   sample decoding, checkpointing, OOM recovery, and report generation:

   ```powershell
   $RunDir = Join-Path $env:LOCALAPPDATA 'TeacherReliability\runs\smoke-reviewed'
   python -m teacher_reliability.run --backend transformers-local --device-preset rtx4060-8gb --profile smoke --run-dir $RunDir
   python -m teacher_reliability.report --run-dir $RunDir
   ```

The first model load downloads several gigabytes and may take time. Model and
dataset caches remain outside Git. A first-time local run should be done while
plugged in with other GPU-heavy applications closed.

## Run profiles

- `smoke`: one GSM8K row, one row per MATH subject, and one GSM-Plus row; two
  samples per example and up to 1,024 generated tokens so the model can finish
  its boxed answer. This profile is not evidence.
- `pilot`: 50 GSM8K rows, 8 per MATH subject, and 100 GSM-Plus rows chosen from
  the current `test` split with at most one variant per seed question; eight
  samples per example. Use it to measure runtime and inspect parse coverage.
- `full`: all GSM8K test rows, 50 deterministic rows per MATH subject, and all
  current GSM-Plus test rows; eight samples per example. The MATH subset is
  defined as 50 examples from each of its seven subject configs.
- `a100_12h` and `a100_24h`: the deterministic time-targeted subsets above;
  both keep all 350 MATH rows and eight samples per example. They are
  exploratory, not full-benchmark evidence.

New runs first resolve exact Hub commit SHAs; resume uses the recorded commits.
The GSM-Plus Hub loader uses
the current `test` split; it does not depend on the older `testmini` layout.
Only a complete `full` profile can produce a candidate KD-gating signal.

Exact command flags are printed by `python -m teacher_reliability.run --help`.
The run prints its output directory. Resume by passing that same directory
with `--resume`; changing profile, seed, source revisions, package versions,
code, or recorded Python/platform/GPU identity creates a different run.
Start a new run when moving from the Windows laptop to a Linux cloud instance;
this prevents silently mixing platform-specific grading or inference behavior
under the first machine's hardware record. Generate reports with
`python -m teacher_reliability.report --run-dir <run-directory>`.
The `--local-files-only` flag applies to model/tokenizer loading. A new run
still needs Hub revision lookup and dataset acquisition; it is not an offline
mode. Cached, revision-pinned dataset loading can use Hugging Face's documented
offline environment settings when every required artifact is already cached.

On Windows, Math-Verify's positive timeout attempts to start a worker process
that is not spawn-picklable. The project caps answer spans at 2,048 characters
and disables that package timeout on Windows. Linux keeps the five-second
timeout. Review parse-coverage rates before interpreting any result.

## Vast.ai account preparation and A100 use

The recorded setup session verified the local Vast.ai CLI's existing credential
with `vastai show user --raw` and registered the local Ed25519 public key. The
Codex plugin `vastai@vast-ai` version 0.1.0 was installed from
`vast-ai/vast-codex-plugin`. These account checks establish controller access;
Linux installation and A100 inference still require an instance.

For a fresh controller, install the Vast.ai CLI separately from this research
environment and configure its key in your own local terminal:

```powershell
vastai set api-key <YOUR_API_KEY>
vastai show user --raw
```

The CLI stores its credential in the local user configuration. Register the
contents of an SSH `.pub` file before launching; keep the private key on the
controller. The research process on the GPU host does not need the Vast API key.
The plugin's `/prompts:vast-setup` offers the same onboarding steps; a literal
key entered into a shell command can remain in shell history.

1. Select one verified, rentable A100 with at least 40 GB visible VRAM and
   enough disk for the model cache and outputs (100 GiB is the planning size).
   Record the rental limit, instance image, GPU, driver, CUDA runtime, and region.
2. Clone this repository and run `scripts/setup_linux.sh` to install the
   pinned CUDA wheel, requirements, and project-local package.
3. Transfer only code/config and authorized input manifests. Download public
   datasets/model weights on the instance; do not upload credentials or commit
   them.
4. Run a smoke or pilot first and inspect OOM/rate/parse behavior. Select
   `a100_12h` or `a100_24h` using observed throughput and the rental limit.
   Start each profile in a new directory; resume only the same run identity.
   Persist the run directory off-instance before termination. The profile does
   not enforce the rental deadline; manage that deadline separately.
5. Compare manifest IDs and data/model SHAs before merging any results with
   local outputs.

On the later Linux instance, use Python 3.10 for the exercised dependency
versions and a fresh directory for each profile:

```bash
PYTHON_BIN=python3.10 bash scripts/setup_linux.sh
source .venv/bin/activate
python -m teacher_reliability.env_check
python -m teacher_reliability.run --profile pilot --run-dir outputs/pilot-vast
python -m teacher_reliability.report --run-dir outputs/pilot-vast
# Or run a bounded A100 profile:
python -m teacher_reliability.run --profile a100_12h --run-dir outputs/a100-12h-vast
python -m teacher_reliability.report --run-dir outputs/a100-12h-vast
# The complete protocol remains available in a separate directory.
python -m teacher_reliability.run --profile full --run-dir outputs/full-vast
python -m teacher_reliability.report --run-dir outputs/full-vast
# If that full run is interrupted, continue with the same identity:
python -m teacher_reliability.run --profile full --run-dir outputs/full-vast --resume
```

## Optional A100 vLLM hybrid run

The optional A100/vLLM preparation path is separate from the current local
study. Its CLI and coordinator remain experimental and unqualified. A source
audit also found that hybrid rows/manifests do not yet include the origin/scope
fields needed for a complete full-study report, and a worker stop-reason field
needs an API-shaped regression. The current hybrid path therefore cannot
produce an accepted complete full-study report. Do not use it to generate the
final study data until those issues are fixed and qualified. No A100 GPU
qualification or performance benchmark is claimed here.

The baseline runner and artifact exporter use the existing Python 3.10
`.venv`, created by `scripts/setup_linux.sh`. The vLLM worker has its own Python
3.12 `.venv-vllm`, created by `scripts/setup_vllm.sh`; keep the environments
separate. Use the immutable model commit SHA recorded in the baseline run
manifest when exporting the model, so the optimized run uses the same teacher
revision. Export into a new artifact directory and give the run a new output
directory. Replace `<SHA>` below with that 40-character commit SHA; do not type
the brackets literally.

```bash
PYTHON_BIN=python3.10 bash scripts/setup_linux.sh
PYTHON_BIN=python3.12 bash scripts/setup_vllm.sh
source .venv/bin/activate
python -m teacher_reliability.env_check
python -m teacher_reliability.model_artifact --revision <SHA> --output-dir models/qwen-nf4
python -m teacher_reliability.run --backend vllm-hybrid --profile full --run-dir outputs/full-hybrid --model-dir models/qwen-nf4 --vllm-python .venv-vllm/bin/python --max-runtime-hours 12
```

The experimental hybrid backend uses vLLM on one GPU without Ray. It keeps the
teacher's existing 4-bit NF4 configuration, including double quantization. For
greedy generations, it streams the full-vocabulary scores needed for token
entropy and selected-token log probabilities instead of retaining full
vocabulary score tensors; vLLM handles the stochastic sample generations.
The intended design preserves the requested scores and sampling protocol, but
the current reporting and worker-contract gaps above prevent final-study use.
vLLM runs on Linux; Windows use is through a separate WSL environment. A
single-GPU run does not need a Ray cluster.

The hybrid runner checkpoints each completed greedy result and each sample
separately under its run directory. Continue an interrupted run with the same
command and identity plus `--resume`; completed stages are reused, and only
unfinished stages are retried. Then generate the report from that same
directory:

```bash
python -m teacher_reliability.run --backend vllm-hybrid --profile full --run-dir outputs/full-hybrid --model-dir models/qwen-nf4 --vllm-python .venv-vllm/bin/python --max-runtime-hours 12 --resume
python -m teacher_reliability.report --run-dir outputs/full-hybrid
```

On a recognized CUDA OOM, bounded backoff reduces the batch or memory profile
and retries only unfinished requests with their original IDs and seeds. If a
single request still OOMs at the smallest supported profile, the runner records
the failure and continues unaffected requests. Other worker or protocol errors
stop the run while preserving completed checkpoints; they are not treated as
successful or silently skipped examples. vLLM streams each finished sample to
the coordinator, which checkpoints it before waiting for the rest of the batch.
Global free GPU memory is checked before a worker starts, and the chosen memory
settings are recorded in the manifest.

The first start persists a 12-hour UTC deadline in the run manifest. Resume
uses that same deadline and never grants another 12 hours. If it expires, keep
the partial checkpoints and report; any attempt with additional runtime must
use a new run identity and directory rather than editing the saved deadline.

Before ending the separate hybrid rental, copy the run directory and required
artifact to durable storage and verify the manifest and report are present.
Then stop only that hybrid rental under its own rental record. This workflow
does not authorize stopping or changing the existing baseline rental on
instance 53530215.

## Failure recovery

- OOM: lower the configured prompt/output limits or use a larger GPU, then
  start a new run identity so the changed protocol is visible.
- Interrupted run: use the same run directory and manifest; only validated
  complete per-example checkpoints are skipped.
- Missing answer score: inspect the recorded answer span and token-alignment
  reason; never fill from sequence confidence.
- Dataset/model drift: resume continues with the recorded Hub commits. To
  evaluate newer upstream data or weights, start a new run directory; recorded
  inputs cannot be relabeled as a different revision.
