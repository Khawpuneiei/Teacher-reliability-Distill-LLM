# Local and Vast.ai runbook

This runbook separates a quick pipeline smoke test, a bounded pilot, and the
full benchmark. The attached estimate is 4–8 GPU-hours on a T4/L4; it has not
been validated. With the current Hub sizes, the full profile evaluates about
12,221 rows (1,319 GSM8K, 350 MATH, and 10,552 GSM-Plus) and makes 109,989
greedy/sample generations at `k=8`, before accounting for the output-token
length. Plan for the possibility that it takes materially longer than the
estimate. The 8-GiB RTX 4060 Laptop GPU passed the environment check and completed
the nine-row smoke with 4-bit model loading, peaking near 7.8 GiB GPU memory.
Full-run throughput and duration still need a pilot measurement. Do not interpret an
OOM or smoke run as a benchmark result.

## A100 12–24 hour profiles

The local smoke manifest records nine examples, 27 generations (one greedy
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
5. Run the smoke profile to verify model load, token-score alignment, sample
   decoding, checkpointing, and report generation:

   ```powershell
   python -m teacher_reliability.run --profile smoke --run-dir outputs\smoke-reviewed
   python -m teacher_reliability.report --run-dir outputs\smoke-reviewed
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
