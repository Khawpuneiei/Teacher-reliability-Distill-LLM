# Local RTX 4060 project plan and remaining tasks

Status snapshot: **2026-10-01, 20:20 Asia/Singapore (UTC+08:00)**.
Checkboxes record the current evidence; update this snapshot as work advances.

Repository: [Teacher-reliability-Distill-LLM](https://github.com/Khawpuneiei/Teacher-reliability-Distill-LLM).
Working branch: `codex/local-rtx4060`.
Base commit: `126621bfbe1c6228ef27debf7bdc69db8945dfba`.

This document tracks the approved local experiment through qualification,
review, execution, reporting, and repository delivery. The
[remaining-tasks checklist](remaining-tasks-to-complete-study.md) summarizes the
ordered work from the current checkpoint to delivery. The
[runbook](runbook.md#full-study-on-the-local-rtx-4060) contains operational
commands; the [experiment design](experiment-design.md) defines the scientific
protocol; [DEVLOG.md](../DEVLOG.md) records the implementation process.

## 1. Project outcome

Complete the teacher-reliability and calibration study on all **12,221 selected
questions**. Reuse **110 validated A100 rows** with their original provenance,
then generate the remaining **12,111 rows** on the local RTX 4060 Laptop GPU.

| Dataset | Required rows | A100 rows available for import | Remaining local rows |
| --- | ---: | ---: | ---: |
| GSM8K test | 1,319 | 110 | 1,209 |
| MATH test: 50 per each of seven subjects | 350 | 0 | 350 |
| GSM-Plus test | 10,552 | 0 | 10,552 |
| **Total** | **12,221** | **110** | **12,111** |

Each local row needs one greedy answer and eight sampled answers: **108,999
remaining generations**, before retries. The final deliverable includes
confidence/correctness metrics, reliability plots, an evidence-based gate
recommendation when the metric criteria permit one, and recoverable execution
history. Testing whether the gate improves student distillation is a separate
follow-up experiment.

## 2. Fixed experiment settings

| Setting | Required value |
| --- | --- |
| Teacher | `Qwen/Qwen2.5-Math-7B-Instruct` |
| Model revision | `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d` |
| Quantization | NF4 with double quantization and FP16 compute |
| Profile / seed | `full` / `42` |
| Samples per question | Eight, with independent request seeds |
| Sampling | Temperature `0.7`, `top_k=50`, `top_p=1.0` |
| Output cap | 1,024 new tokens per generation |
| Context limit | 4,096 tokens, including prompt and generated tokens |
| Confidence normalization | Model output vocabulary: 152,064 tokens |
| Default deadline | None; run across resumable sessions |
| Optional deadline | Positive `--max-runtime-hours`, persisted across restarts |
| Backend / device preset | `transformers-local` / `rtx4060-8gb` |

Preserve full question/reference content, sample counts, precision, seeds,
and EOS handling during memory recovery. Record any over-limit input explicitly.

### Owner-directed fast path

The owner prioritizes starting the complete study quickly and has waived the
12-profile comparison and 24-question throughput benchmark. The nine-row smoke
has completed successfully. Next are candidate adversarial verification and
the required final-strict review, then branch publication, one-time import of
the validated 110-row A100 archive, and the full resumable local run using the
explicit fast-start-unqualified mode.

Preserve the model, precision, datasets, full prompts/references, seeds, eight
samples, and output cap. Keep checkpointing and memory recovery enabled. Report
a dataset-weighted ETA only after measured local throughput is available.
Archive validation, final-strict review, full row accounting, complete reports,
and a verified backup remain required.

Pinned dataset revisions:

| Repository | Revision |
| --- | --- |
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` |
| `EleutherAI/hendrycks_math` | `21a5633873b6a120296cce3e2df9d5550074f4a3` |
| `qintongli/GSM-Plus` | `3b708db57b96a16e8e3368ed2956990c0809440e` |

## 3. Local execution and memory plan

Use the existing pinned Windows environment and cached weights. One CPU
coordinator owns checkpoints; one persistent GPU worker runs both greedy and
sampled generation. Greedy scoring calculates full-vocabulary entropy and
selected-token log probability during decoding, retaining scalars and token IDs.
Sample batches give each request its own `torch.Generator`.

| Control | Initial local setting | Qualification candidates |
| --- | --- | --- |
| Resident GPU models | One | One |
| Greedy batch size | 1 | 1, 2, 4 |
| Sampling batch size | 2 | 1, 2, 4, 8 |
| PyTorch allocator ceiling | 85% of VRAM | Monitor global memory as well |
| Global GPU free-memory target | At least 1 GiB | Measure throughout qualification |
| Offloaded KV cache | Recovery after batch one fails | Available RAM must cover estimated cache plus 2 GiB |

The implementation also has a **512 MiB emergency free-VRAM floor for offloaded
KV cache**. This is a recovery setting that needs explicit qualification; the
normal target remains 1 GiB. The 1,024-token stress check now uses the
recommended profile's cache mode: when both greedy and sampling use offloaded
KV cache, stress uses offloaded cache and rechecks the 2 GiB host-RAM reserve;
otherwise it stresses the standard cache, which puts more pressure on VRAM. The
check must produce all 1,024 tokens and retain at least 1 GiB global free VRAM,
matching the full-run admission gate.

On CUDA OOM, retain completed stages, release failed allocations, restart the
affected worker when necessary, and halve the affected batch down to one.
Persist smaller settings. At batch one, attempt KV-cache offload only when host
RAM satisfies the reserve. Exhausted example recovery records a failure and
continues; repeated model-loading failures stop safely. Preserve actual reasons
for headroom refusals, host-memory shortages, worker crashes, and other errors.

The performance objective is completed rows per hour with safe headroom.
The RTX 4060 plan does **not** use vLLM or Ray. It uses one native Windows
Transformers model with small, measured batches. Batching lets the GPU process
multiple requests together where memory allows; a vLLM server or Ray cluster
does not guarantee higher utilization on an 8 GB laptop GPU. The prepared
A100/vLLM backend remains experimental. vLLM's current GPU guide requires
Linux, says Windows is not natively supported, and identifies WSL as a separate
option. Its scaling guide recommends a single GPU without distributed
inference when the model fits on one GPU. A local vLLM/WSL route would therefore
be a separate project and qualification, not a drop-in acceleration for this
approved native-Windows run. See [vLLM GPU requirements](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/)
and [single-GPU parallelism guidance](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/).

## 4. Current evidence and status

| Area | Status at the snapshot |
| --- | --- |
| Local backend, batching, streaming scores, recovery, checkpointing | Implemented in the working branch; final acceptance pending |
| ZIP import validation and execution provenance | The exact supplied archive passed the current importer from a file-backed Windows script: 110 rows, eight samples per row, expected source content/revisions/seeds, and A100-reported runtime metadata. Archive SHA-256: `7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82`; prediction JSONL SHA-256: `085d4e38213744311e509b064c3a3c2dd1f0fd7edf5883ebc5870309254e8298`. An earlier stdin diagnostic emitted Windows multiprocessing spawn errors and then reported a derived-counter mismatch; this was not reproduced by the file-backed importer and is treated as an invalid diagnostic. The ZIP remains unchanged and no rows have been imported. Preserve the three legacy counters absent from the historical schema as unavailable (`null`), never inferred as zero. The archive's `run_signature` is an unkeyed SHA-256 self-hash: it detects changes but does not independently authenticate the physical GPU. |
| Selected inputs and seeds | Archive matches cached selection; all 880 imported sample seeds checked |
| Context admission | Recorded longest full-selection prompt: 659 tokens |
| Automated verification | The fixed-input source passed 262 tests in 192.218 seconds; bytecode compilation, pip check, CLI help, git diff checks, and the untracked-aware text whitespace scan passed. The final documentation bytes are rebound for the next frozen candidate. The earlier 256-test receipt is historical. |
| Historical diagnostic smoke | An earlier exploratory smoke completed 8/9 rows; one failed the offloaded-cache RAM reserve check. It predates this current run and remains historical evidence. |
| Current smoke | Complete: 9/9 rows, zero failures and zero pending, across all three datasets and all seven MATH subjects. The 22.60-minute smoke generated 18 samples; three reached the 1,024-token cap. After model load it recorded 4,816 MiB available host RAM, 5,311 MiB allocated / 6,082 MiB reserved VRAM, and 1,314 MiB free VRAM. At its final progress heartbeat (17:52:56 SGT), the snapshot was 4,143 MiB available host RAM, 5,324 MiB allocated / 6,094 MiB reserved VRAM, and 896 MiB free VRAM. That is below the normal 1,024 MiB free-VRAM target but above the 512 MiB emergency floor used with offloaded KV cache; it is smoke evidence, not long-duration qualification. Five adaptations ended at greedy batch 1 and sample batch 1 with offloaded KV cache. Run path: %LOCALAPPDATA%/TeacherReliability/runs/smoke-rtx4060-20261001-173011. The smoke rate is not a full-run ETA. |
| GPU qualification | Qualification rtx4060-20261001T073411Z-5e1c5132 ended incomplete on source snapshot 5e1c513218c8079af04905fa5e43259b85f4c37ef82ee6cf26c95eefa295432e: all 12 profiles had 283 InsufficientHostMemory failures, no recommended profile, pending 1,024-token stress, and no ETA. The memory audit found profile minima of 2,029-2,246 MiB versus a 2,272 MiB admission floor. At 17:18 SGT, 5,997 MiB was free while idle; this is historical and does not establish loaded headroom. The current smoke's loaded and live memory are recorded in the preceding row. Qualification code releases prompt tokens and closes the validated archive snapshot before teacher loading. Current inference source snapshot: ae37dee3b84ecd691e46815c80bda48b34197c9bbd53dc83f3d1385a50e96d91. Preserve the old qualification as historical and use the owner-directed smoke plus unqualified fast-start instead of the exhaustive matrix. Keep experiment settings unchanged. The historical 11.2949 rows/hour and 1,002.7-hour estimate remain provisional. |
| Archived A100 comparison | Five archived GSM8K rows overlap the 24-question qualification set. All five greedy token sequences/texts, extracted answers, and correctness labels match. Greedy score arrays align; maximum absolute entropy difference is 0.02679 nats and maximum selected-token log-probability difference is 0.01228. All 40 derived sample seeds match; sampled token/text sequences match 0/40, while extracted answers are equivalent for 23/40. This small cross-run comparison is descriptive, not a hardware-only causal comparison or throughput benchmark. |
| Qualified profile / local ETA | No profile matrix is being treated as complete. Per owner direction, the fast path skips the 12-profile matrix and uses the recorded default greedy-1/sample-2 configuration only after the nine-row smoke and memory gate. The full-run launch has no ETA; calculate one from actual per-dataset rates. |
| New full local run | Not created; the validated 110 source rows have not yet been imported into a full local run |
| Assurance / branch delivery | The existing inference-optimization/vllm-hybrid unit is open and not review-ready, with 1/3 calls used and no reservation. Call 1 returned fix-first; its fixed-input blocker is now closed under TDD, 262 tests pass on the changed source, and the ledger header is reconciled. The current continuation is verified as Sol/max. Candidate rebinding and re-review remain before commit, push, and sustained execution. |

### Audit findings and current closure state

The earlier parent-focused result of 76 tests across eight suites predates the
audit findings. Follow-up TDD and focused verification now close the three
data-integrity findings; the recommendation-coverage issue is closed as well:

1. **Sampling semantics — closed:** batched sampling now applies Transformers'
   temperature and top-k warpers before each independently seeded draw. The
   64-token regression produced a behavioral RED (raw choice 56 versus
   transformed scalar choice 19) before the fix. The focused sampling suite
   passes all four tests.
2. **Resume integrity — closed:** local records are checked against selected
   source content, raw generation stages, derived answer spans, and an atomic
   checksum receipt bound to the run signature. Imported records must equal
   the validated archive row plus expected provenance. Valid-shape mutations
   to text, score vectors, sample/index/seed fields, confidence, labels, and
   consistency shares are rejected before workers run.
3. **Report lineage — closed:** reports now require the exact imported ID set
   and validate each imported row's archive hash, source run ID/signature, and
   GPU metadata. Focused report tests pass all 19 cases, including tampering
   tests that preserve aggregate counts.

4. **Report row content — implemented, focused checks passed:** the report
   validates each imported row against the exact prediction content in the
   preserved, hash-verified A100 ZIP and checks each local row against its
   run-signature-bound integrity receipt. Full-run behavioral REDs showed that
   changing an imported correctness label or local entropy confidence still
   allowed `run_complete`; compact helper regressions now pass, and the valid
   12,221-row report integration test still passes. The complete report module
   now passes **27 tests**, including source-revision and legacy-null cases.
5. **Empty-source and tokenizer identity — closed:** requested empty benchmark
   splits fail with their source details, full reports require all three
   benchmarks, and the pinned `tokenizers` version contributes to run identity.
   The documented TDD RED/GREEN evidence is in `DEVLOG.md`.
6. **Imported generation termination — closed:** imported greedy and sampled
   outputs must include typed EOS/truncation/reason fields with consistent
   values. TDD observed RED for missing greedy fields, conflicting flags, and a
   malformed reason type; all ten archive-import tests now pass.
7. **Qualification stress/cache alignment — closed:** the 1,024-token test now
   uses the selected cache implementation, rechecks host RAM for offloaded
   cache, requires exactly 1,024 generated tokens, and records its profile.
   TDD regressions for cache selection and forwarding pass in focused checks.
8. **Qualification stress fallback — closed:** candidates are tested in
   measured throughput order, and selection continues to the next-fastest
   profile when a faster profile fails the 1,024-token stress/headroom gate.
   Completed stress results are reused on resume and every new result is
   checkpointed. `tests.test_local_benchmark` passes all 19 tests, including
   fallback and checkpoint-reuse regressions. The source change requires a
   fresh qualification snapshot; these unit tests do not qualify a profile.

The report recommendation-coverage risk is also closed. A candidate now
requires at least 95% source-scorable references in every dataset, a known
binary outcome for every source-scorable row, and full score coverage for each
candidate signal. The report publishes those coverage fractions and labels its
metric ranking descriptive; it makes no confidence-interval or statistical
superiority claim. Two assertion-based RED cases showed recommendations were
previously allowed at 95% outcome-label coverage and 90% source-reference
coverage. The new report tests reject both cases.

Additional audit closure at the 13:27 SGT snapshot:

9. **Legacy A100 schema — closed:** the exact supplied archive passed
   file-backed import validation: 110/110 rows match the full cached selection,
   exact source inputs/revisions, request seeds, archived derived values, and
   metadata that reports an A100 runtime. The metadata is internally
   consistent, but does not authenticate the physical GPU. The original
   prediction JSONL hash is recorded
   above. Three fields introduced after the archive run stay `null`.
10. **Report scope and revision identity — closed:** a 12,221-row report cannot
    become synthetic by editing only `study_scope`; full reports bind model,
    tokenizer, dataset, and per-row source revisions. TDD regressions observed
    assertion RED. Legacy `null` counters survive preserved-archive comparison,
    appear as unavailable in outputs, and block recommendation signals without
    making complete source-schema-valid rows structurally incomplete.
11. **Local recovery classification — closed in focused tests:** insufficient
    host RAM remains distinct from CUDA OOM; an exhausted offloaded-cache row
    is recorded as failed and later rows continue at a regular-cache profile.
    Deadline expiry interrupts streamed generation and preserves checkpoints;
    qualification records and gates on transient global-VRAM minima. A real
    worker-process crash/restart test confirms only unfinished requests rerun.
12. **Parent focused integration — passed:** the combined command covering 13
    changed/affected test modules passed **167 tests** in 135.381 seconds.
    Repository-wide discovery and the current GPU qualification have not run.

13. **Nullable majority-vote resume validation — closed in focused tests:** a
    parent rerun of `tests.test_local_run` exposed six resume errors when an
    unknown answer-equivalence result correctly made `majority_vote_count`
    null. The validator now permits that documented provisional shape only
    when the unknown-comparison count and related majority fields agree; other
    count bounds and state combinations remain strict. The mutation test now
    begins from a fixed valid majority fixture. RED/GREEN evidence and three
    consecutive 22-test worker runs are recorded in `DEVLOG.md`; the parent
    independently reran the module and observed **22 passed**. The parent then
    reran the 13 affected modules after the report and grading fixes and
    observed **175 passed** in 150.437 seconds. Candidate-wide checks remain
    pending.

14. **Full-report nullable majority state — closed:** a valid local row with
    unresolved pairwise answer comparisons and null majority metrics previously
    caused a full-profile report to become incomplete. The full-report
    regression failed before the fix; additional assertion REDs showed that
    contradictory majority metrics and out-of-range counts were accepted.
    `_full_row_issues` now accepts only the consistent unknown-comparison state,
    rejects contradictory majority fields, and enforces the eight-sample and
    28-pair bounds while retaining legacy-null counter handling. The report
    module passed **32 tests** on 2026-10-01, including a 12,221-row
    mixed-origin case and GPU-metadata tampering cases.

15. **Verifier timeout classification — closed:** on a platform using
    Math-Verify's inner timeout, a timeout during verification carried the
    broader operation label `grade` despite progress identifying stage `verify`.
    A behavioral RED returned `parse_timeout` instead of `verifier_timeout`.
    `grade_prediction` now uses the latest nonempty progress stage when
    classifying the timeout. The grading and consistency modules pass **21
  tests**, including the new classification regression and worker restart
  checks.

16. **Qualification prompt-token memory — fixed under TDD:** the qualification
   previously retained token-ID sequences for the full selected dataset after
   using them only to calculate prompt lengths (about 52 MiB by saved counts).
   The RED regression observed those weak-referenced sequences still alive at
   cohort selection. Tokenization now counts each prompt immediately, and the
   exact token counts remain unchanged while earlier sequences can be released.
   The focused test passed, and `tests.test_local_benchmark` passed **23 tests**
   in 36.957 seconds. The actual RAM savings are not yet measured with the model
   loaded; this does not satisfy the 2.8 GiB operating target by itself.
   Inference source snapshot is now
   `530296039533dce7142c7b9327c3a7c3044706c404aaed751f0b8944e83c5ec2`; the
   incomplete qualification used the old snapshot and must remain historical.

The parent reran the sampling, local-run, and report suites together after the
coverage change: **33 tests passed** (4 sampling, 10 local-run, 19 report).
Later report integrity checks pass their two compact helper tests, and the
valid full mixed-origin report test passes with all 12,221 rows. A separate
output-cap fix now rejects over-limit greedy/sample stages and completed local
rows; the focused local-worker/hybrid/local-run suite passed 50 tests.
The archive importer now also rejects missing or conflicting generation
termination metadata; its nine focused tests pass. The RTX 4060 was idle at the
12:28 SGT checkpoint, and no new qualification workload has started.
The owner later waived the long profile matrix and benchmark for the
unqualified fast path. That path remains gated on smoke evidence, a fresh
loaded-memory preflight, candidate verification, parent adversarial review, and
final-strict acceptance; claiming a qualified profile still requires the full
matrix.
The experimental vLLM/hybrid audit is complete. That path has a separate report
integration blocker: generated rows omit execution origin, and the manifest
omits study scope and origin counts, so its documented full run cannot produce
`run_complete: true`. The worker also reads a stop-reason field from the wrong
vLLM output object; add an API-shaped regression if that experimental backend
is ever promoted. These findings do not block the selected local Transformers
run; do not use or recommend the vLLM full-study path until it is fixed and
qualified.

Archive metadata is checked for internal consistency; it is not cryptographic
attestation of the physical GPU. Recommendation coverage is governed by the
95% per-dataset source-reference threshold, complete outcomes on source-scorable
rows, and complete score coverage described above; ranking remains descriptive.

### Qualification completion-gate fix (qualified profile path)

Read-only review found that `run_qualification` defined `matrix_complete` by
checking only that all twelve configuration names existed in `config_results`;
it did not require each profile result to have `status == "complete"`. The
per-request recovery path can record a minimum-profile failure and still let
the matrix continue, so this state is reachable. Under TDD, a persisted-manifest
assertion RED reproduced the defect. The completion gate now requires all
twelve profiles to be complete; its focused regression passes. A run claiming
a qualified profile must use a fresh source snapshot and pass the full matrix
and stress check. The current owner-approved fast-start path makes no such
qualification claim and explicitly waives those time-consuming comparisons.

`TDD_REQUIRED: yes`. The observable seam is the persisted qualification
manifest when all twelve profile files exist but one has `status: incomplete`:
the overall manifest must remain incomplete and must not recommend a profile.
RED command used:
`.venv\Scripts\python.exe -m unittest tests.test_local_benchmark.LocalBenchmarkQualificationMatrixTests.test_matrix_manifest_stays_incomplete_when_one_profile_is_incomplete -v`.
The assertion failed before the gate change and passed after it. The matrix has
not yet been rerun on a fresh final source snapshot.

### Repository verification profile

The repository verification entry point is Python unittest discovery, documented
in README.md and the runbook. The candidate gate also runs compileall, pip check,
and git diff --check. This checkout has no repository AGENTS.md, CI workflow,
Make/Just/Tox/Nox task runner, or Compose entrypoint. The current code candidate
passed unittest discovery: **256 tests in 175.167 seconds**. Compileall,
dependency consistency, CLI help, tracked diff checks, and the untracked-aware
text whitespace scan also passed. The
verified inference source snapshot is
**ae37dee3b84ecd691e46815c80bda48b34197c9bbd53dc83f3d1385a50e96d91**; later
changes in this turn are documentation-only. Parent adversarial review and
frozen-candidate binding remain pending. The long profile matrix and separate
24-question / 1,024-token stress comparison are waived by owner direction; the
completed smoke exercises the output cap but does not qualify sustained
throughput.

### Windows grading-timeout gate

The original implementation passed a zero timeout to Math-Verify on Windows,
leaving coordinator-side grading unbounded. The launch blocker is now addressed
with an isolated persistent worker and a 30-second request deadline. Injected
parse and verifier stalls produce explicit unscorable statuses, terminate the
worker, and allow a later request to start a replacement. Self-consistency uses
tri-state parse/equivalence checks: timeout or worker failure is counted as
unknown, not as a false answer or non-equivalence. Run records and report
summaries expose those counts. The focused grading/consistency/run/report suite
passed 40 tests; an earlier repository-wide verification passed 251 tests.
Those repository-wide results predate the latest candidate changes.
The separate long-answer memory-stress qualification was waived for the fast path; the completed smoke reached the 1,024-token cap, but the full run remains labeled unqualified.

The diagnostic smoke is not a throughput benchmark. The first qualification
attempt is also unusable for selecting settings: its 13 recorded offloaded-cache
admission failures each needed 2,382,364,672 bytes (2.219 GiB) including the
2-GiB host-RAM reserve, while available RAM ranged from 1,893,404,672 to
1,990,770,688 bytes (1.763–1.854 GiB). The first config lacks minimum-RAM
telemetry, and no worker from that attempt remains running, so these failures
show the reserve shortage but do not identify its system-level cause. That first qualification process stopped
incomplete; preserve its directory as failed-run evidence. The corrected retry
completed `g1-s1` at 07:46 SGT: 24/24 greedy outputs and 192/192 samples, zero
active or historical failures, and exact token agreement with the serial
reference for all 24 greedy answers and 192 samples. Maximum entropy and
log-probability differences were both zero. The actual greedy and sample
profiles both used batch size one with offloaded KV cache. Peak allocated and
reserved GPU memory were 5,517 and 6,114 MiB; minimum available host RAM was
2,279 MiB (7 MiB above its 2,272-MiB reserve requirement), and minimum free
VRAM was 876 MiB. Dataset rates were 14.596 GSM8K, 11.990 GSM-Plus, and 8.796
MATH rows/hour (11.295 overall). Applying those rates to 1,209, 10,552, and 350
remaining rows yields a **provisional 1,002.7 active-hour estimate**. It is only
the batch-one baseline; the matrix must determine whether safe batching improves
it.

The full matrix resumed in the same identity at 07:49 SGT after the first
worker exited and the GPU cooled to 56 C with thermal slowdown flags inactive.
The launcher PID 30504 and Python child 30360 were present at 09:28, but both
had exited by 09:29. `g1-s2` saved 24/24 greedy outputs and 154/192 samples,
with zero per-example failures. The requested configuration was greedy batch
one / sampling batch two; both stages adapted to batch one with offloaded cache
after global free VRAM fell below the 1-GiB target. The profile then failed
while atomically replacing `config.json` from a temporary file:
`PermissionError [WinError 5] Access is denied`. Its `last_error` and partial
stage data are in the preserved qualification directory. A read-only audit
found 178 valid stage JSON files: all 24 greedy outputs and 154 samples, with
matching run signatures and unique request IDs; no temporary config file
remains. Source inspection indicates that a same-code rerun can load those
stages and submit only missing requests using the same derived seeds; that
recovery has not yet been run. The atomic JSON writer has no retry for a
transient replace failure, so the current invocation stops. The overall
manifest is failed and no production profile is selected. Before using a
changed source snapshot, exercise this resume path under TDD and requalify the
full matrix. Earlier readings for this profile
were 3,224 MiB minimum available RAM and 876 MiB minimum free VRAM; at 09:15,
Windows showed 3,175 MiB available RAM while NVIDIA reported 876 MiB free, 85 C,
79% utilization, and software thermal slowdown. Treat those as historical.
A resume defect was found in qualification failure handling: persisted failures
blocked unfinished requests on restart. A regression test and recovery fix are
now in place. The owner's
old A100 observation of 110 rows in about three hours does not establish an
RTX 4060 ETA. Qualification generations do not count toward full-study rows.

At 08:27, a read-only comparison of `g1-s1` against the original A100 ZIP found
five overlapping GSM8K examples. Greedy token IDs and decoded text matched for
all five, as did extracted-answer equivalence and correctness labels. Score
arrays were aligned; maximum absolute entropy and selected-token
log-probability differences were 0.02679 nats and 0.01228 respectively. All
40 sample seeds matched the base-seed derivation, but none of the sampled token
sequences/texts matched exactly; extracted sample answers were equivalent for
23/40. These rows came from different executions and software paths, so the
observations describe reproducibility across runs, not an isolated GPU effect.

The qualification coordinator archives prior invocation failures and retries
unfinished stages with their original IDs and seeds. The staged pass used source
snapshot `f5bfcc911fe01ecdb348665d756ecaffde211a285a0eaa1c7eb6180c1e04a572`.
The unfiltered matrix reuses that qualification identity and source snapshot.
Its exact command is recorded in `qualification-matrix.command.txt`; the first
profile logs remain `qualification.log` and `qualification-error.log`, while
the matrix writes to distinct files so prior evidence is preserved.

### Files and directories to preserve

| Item | Location |
| --- | --- |
| Original archive | `%USERPROFILE%\Downloads\full-a10080-20260930T131901Z.zip` |
| First qualification attempt (preserve) | `%LOCALAPPDATA%\TeacherReliability\qualifications\rtx4060-20261001-033104` |
| Qualification attempt to preserve (g1-s2 checkpoint failure) | `%LOCALAPPDATA%\TeacherReliability\qualifications\rtx4060-20261001-053702-retry-fix` |
| Qualification identity/results | `qualification_manifest.json` and each profile's `config.json` / `stages` |
| Future sustained run | `%LOCALAPPDATA%\TeacherReliability\runs\full-rtx4060-<unique-name>` |
| Full-run path pointer | `%LOCALAPPDATA%\TeacherReliability\runs\current_run_dir.txt` |
| Process/concept log | Repository `DEVLOG.md` |

Archive SHA-256:
`7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82`.
The source manifest's completed counter says 100; validated complete prediction
records establish the import count of 110. Preserve both original files and
record this discrepancy in provenance. Read-only validation of the supplied
ZIP found 110/110 complete rows, all GSM8K. Its `identity.runtime` metadata and
top-level `hardware` both report an NVIDIA A100 80GB PCIe on Linux. The
`run_signature` is an unkeyed SHA-256 self-hash, so those fields provide archive
consistency checks, not cryptographic authentication of the physical GPU. The final
origin-by-dataset counts must therefore be A100 import: GSM8K 110, MATH 0,
GSM-Plus 0; RTX 4060 local: GSM8K 1,209, MATH 350, GSM-Plus 10,552.

First failed qualification source snapshot:
`674fe224a4418b2162e24807c7c86523e8eeb28170ca558c70683638583080be`.
It binds package Python source, `requirements.txt`, and `pyproject.toml`. The
qualification code has since changed; keep the old directory as evidence and
use a new output directory/source snapshot for the next attempt. Documentation
edits do not change the inference snapshot, but remain part of the final review
candidate.

## 5. Ordered tasks from now to completion

### Closed TDD regression work before the fast-path run

- [x] Add a batched-sampling regression with a vocabulary larger than 50 that
      shows temperature and top-k are applied before the sample draw; fix under
      TDD while preserving independent seeds, EOS behavior, and token limits.
- [x] Add valid-shape tampering regressions for local and imported checkpoint
      rows. Coverage includes changed question, text, score vector, sample,
      index/seed, derived confidence/label, and imported provenance. Seed and
      import cases already rejected before the latest implementation are
      regression coverage, not claimed behavioral REDs.
- [x] Add report regressions that preserve A100/local aggregate counts while
      swapping imported IDs or changing archive provenance. Require exact
      imported ID-set and row-level manifest validation.
- [x] Record RED/GREEN evidence and rerun focused checks for sampling, resume,
      lineage, and recommendation coverage. Parent integration passes 33 tests.
      Update the benchmark source identity if inference changes before qualification.
- [x] Bind imported report rows to their preserved ZIP content and local report
      rows to per-row integrity receipts. Historical behavioral REDs showed
      same-shape content changes could pass; focused helper checks and one
      valid full mixed-origin report integration test now pass.
- [x] Finish the TDD output-cap audit: reject any saved greedy/sample stage or
      completed local record whose token count exceeds that run profile's
      `max_new_tokens` setting. **Closed under TDD:** assertion RED showed that
      over-limit greedy/sample stages and completed rows were accepted; the
      focused local-worker/hybrid/local-run suite passed 50 tests after the cap
      was enforced on both output types.

**Gate:** the documented local-path regressions pass the current 262-test
candidate suite. The next steps are the parent whole-candidate audit, adversarial
readiness, and final-strict review. The qualification matrix is waived; do not
resume it for this run.

### Phase A — Fast-path smoke and memory admission

- [x] Preserve the failed 12-profile qualification as historical evidence. The
      owner waived the profile matrix and the 24-question throughput comparison;
      do not spend time resuming or repeating them for this run.
- [x] Run the current-code three-dataset smoke. It completed 9/9 rows with
      zero failures and zero pending, covered all seven MATH subjects, and
      exercised the 1,024-token cap (three samples reached the limit).
- [x] Verify checkpoint and resume behavior. The smoke resumed after two clean
      stops, retained original request seeds, and ended with no failed rows.
- [x] Verify local device and memory telemetry. The manifest identifies the
      NVIDIA GeForce RTX 4060 Laptop GPU. After load it recorded 4,816 MiB
      available host RAM, 5,311 MiB allocated / 6,082 MiB reserved VRAM, and
      1,314 MiB free VRAM. Final settings were greedy batch 1 and sample batch
      1 with offloaded KV cache; five adaptations are persisted.
- [x] Record the smoke as unqualified fast-path evidence, not as a full
      benchmark or a basis for a 12,111-row ETA.
- [ ] Before sustained inference, recheck host RAM after model load and keep
      the approved 2.8 GiB operating target, 2 GiB offloaded-cache reserve,
      and GPU free-memory admission rules.

The previous matrix attempt remains incomplete and historical. Its profile
failures establish that host memory fell below the configured reserve; no
system-level cause is inferred. The new smoke's successful recovery does not
qualify a throughput profile.

**Fast-path exit condition:** the smoke, archive validation, candidate software
checks, and parent adversarial pass are complete. Remaining launch gates are
final-strict readiness and review, then publication of the reviewed branch.
The 12-profile matrix and 24-question comparison remain waived by owner
direction. Do not calculate a full-run ETA until local full-run rates are
measured.

### Phase B — Close correctness and provenance checks

- [x] Verify the current parent runtime for this turn. The exact rollout
      turn_context at ordinal 37954 records gpt-6-luna/max, turn
      01a0f6ee-0d35-7350-83be-6f07de7eec38; SHA-256 of the record line is
      f9e6863757b1328ad7f8018e02669b12f35acd73faa78a66d9e4338f48cdc5fd.
      The fresh final-strict reviewer must still be independently verified as
      gpt-6-sol/max.

- [x] Audit the full candidate, including the experimental A100 code and all
      tracked, staged, unstaged, and untracked in-scope files.
- [x] Confirm independent RNG streams, left padding, early EOS, token/score
      alignment, and streaming-score agreement with retained-score references.
- [x] Verify archive path/record validation, exact revisions and content,
      interrupted import recovery, and importing the 110 rows exactly once.
- [x] Enforce the fixed execution split in code and reports: exactly 110
      validated A100 imports and 12,111 local rows for 12,221 required IDs.
      Do not calculate the local remainder from whatever row count the archive
      happens to contain. Test that short, oversized, duplicate, and
      wrong-origin archives cannot yield `run_complete: true`.
- [x] Bind archive validation and its SHA-256 to the same opened file bytes,
      then preserve those exact bytes. Test replacing the ZIP path between
      validation and hashing; imported rows and preserved provenance must not
      refer to different archives.
- [x] Verify A100 origin from the archived execution identity/runtime fields,
      not only a separate top-level GPU-name claim. Reject contradictory or
      missing hardware metadata, and test that reports preserve the required
      A100/local origin totals.
- [x] Verify lock exclusion, atomic writes, torn-tail repair, duplicate
      rejection, out-of-order completion, worker crashes, OOM backoff,
      persisted recovery settings, `--retry-failed`, and optional deadlines.
- [x] Verify resume validates imported and local records, source/environment
      identity, archive hash, qualification identity, original IDs, and seeds.
- [x] Close the resume-integrity finding with valid-shape local content
      mutations. Compare each local row to its durable generation stages and
      run-signature-bound checksum; compare imported rows with the validated
      archive before reuse.
- [x] Close the newly reported report-lineage finding: prove a swapped A100 ID
      or archive hash fails even when all origin-by-dataset aggregate counts are
      unchanged; validate per-row provenance against the manifest.
- [x] Close the newly reported sampling finding under TDD with a vocabulary
      larger than 50; apply the required temperature/top-k transforms before
      sampling and retain independent request RNG behavior.
- [x] Set and test a recommendation-coverage gate: each dataset needs parsed
      source references for at least 95% of selected rows; every source-scorable
      row needs a known binary outcome; every candidate signal needs complete
      scores for those rows. Report each coverage fraction and state that metric
      rankings are descriptive because confidence intervals are not calculated.
- [x] Verify progress counts, active request, token count, memory, throughput,
      and last-progress time are emitted at the required 30-second cadence.
- [x] Add bounded Windows Math-Verify parse/verify deadlines and worker restart
      under TDD. A timeout is explicitly unscorable. Tri-state consistency checks
      preserve unknown parse/equivalence counts through run records and reports;
      parent-focused verification passes 76 tests across eight suites.
      Representative long-answer behavior is part of a qualified hardware
      profile only; the owner-approved fast path waives that full qualification.
- [x] Require complete, typed, mutually consistent EOS/truncation/reason fields
      for imported greedy and sampled generations. RED/GREEN evidence covers
      missing fields, conflicting flags, and malformed reason types; the
      archive-import suite passes nine tests.
- [x] Verify incomplete/failed rows suppress study completion and reports
      expose combined and execution-specific counts and metrics.
- [x] Close the three provenance findings from the read-only integrity audit:
      exact 110/12,111 origin split, ZIP-content/hash race, and unverified A100
      runtime identity. All three had assertion RED evidence followed by TDD
      fixes; importer/local-run/report suites pass 25 tests.
- [x] Close the reviewer-identified fixed-study-input behavior gap at local
      launch and full report under TDD. Preserve the documented experimental
      vLLM limitations as out-of-route residuals.
- [x] Finish the parent adversarial pass, including negative paths, interactions
      with unchanged code, and evidence that tests detect the relevant defects.
- [x] Freeze the call 1 candidate and inspect its complete scope.
- [x] Refreeze after call 1 fixes and bind a current full-gate receipt to the
      exact new candidate. The earlier 256-test receipt is historical; the
      fixed-input source passed 262 tests before final documentation rebinding.

**Exit condition:** current candidate verification and adversarial checks pass;
all launch-blocking findings have documented closure.

### Phase C — Complete the existing optimization review

Use the existing unit
`Teacher-reliability-Distill-LLM/inference-optimization/vllm-hybrid`, generation
0. Its ledger is under the workspace's
`.assurance\Teacher-reliability-Distill-LLM\vllm-hybrid\ledger.md`.
At this snapshot it records **open, 1 call used, target 1, maximum 3**, with
no active reservation and review readiness set to no. Call 1 returned
**fix-first**; two calls remain under the same unit and generation.

The original setup unit
`Khawpuneiei/Teacher-reliability-Distill-LLM/apply-1-teacher-reliability`
has its own ledger, **ship** status, and three consumed calls. Its earlier
`apply1_review_*` children belong to that completed setup unit. The previous
chat concern that those calls were missing from the optimization ledger was
resolved by reading both durable units. Preserve both histories; the local
scope revision continues the existing optimization unit and its budget.

- [x] Account for call 1 and its persisted GPT-6-Sol/max reviewer runtime.
      Its verdict is fix-first, its one consumed call is recorded under the
      exclusive sidecar lock, and its active reservation is cleared.
- [x] Enforce the exact archive SHA-256, model/dataset revisions, and seed at
      full local launch and report; add behavioral RED/GREEN evidence first.
- [ ] Refreeze the entire changed candidate, rerun the candidate-bound suite,
      repeat the parent adversarial pass, and prepare a neutral closure matrix
      for call 1's four findings. Preserve call 1's Luna parent history and
      record the verified Sol runtime for this continuation.
- [ ] Complete the acceptance map, cumulative diff, candidate manifest,
      evidence packet, and fresh readiness record. Validate the SHA-256-bound
      packet with the Solweaver machine validator; bind any applicable delivery
      copies with a deterministic manifest.
- [ ] Under exclusive coordination, reserve the next call only after all
      readiness gates pass; obtain a fresh read-only reviewer verdict and verify
      its actual persisted `turn_context`. Resolve the outcome within the
      immutable three-call budget.

**Exit condition:** the required final-strict gate is satisfied before sustained
optimized execution and optimization-branch publication. Prior setup acceptance
does not accept these new changes.

### Phase D — Publish the reviewed implementation

- [ ] Finalize Windows start/status/stop/resume/retry commands, benchmark
      measurements, archive provenance, the process/concept log, and this checklist.
- [ ] Commit the reviewed code, tests, dependency files, and documentation on
      `codex/local-rtx4060`. Keep credentials, environment/cache directories,
      and frequent runtime checkpoints outside the source commit.
- [ ] Push the branch to the existing `origin` and verify that its remote commit
      SHA matches the reviewed local commit.
- [ ] Record the commit and branch URL as the code-delivery receipt.

**Exit condition:** the reviewed implementation and operating documentation are
recoverable from the repository.

### Phase E — Import the archive and start the full local run

- [ ] Confirm the reviewed branch has been pushed and its remote SHA verified;
      confirm no study worker is active and recheck RAM/GPU state.
- [ ] Create a new run directory under %LOCALAPPDATA%/TeacherReliability/runs;
      do not reuse the smoke or an old qualification directory.
- [ ] Import the exact validated ZIP once. Before generation, verify 110
      imported, 12,111 pending, zero failed, and 12,221 unique selected IDs.
      Confirm the local manifest preserves the original archive, source
      manifest, hashes, per-row provenance, and original sample seeds.
- [ ] Launch the full profile with backend transformers-local, preset
      rtx4060-8gb, seed 42, the archive, and --fast-start-unqualified.
      Do not pass a qualification manifest. Leave the deadline unset.
- [ ] Confirm the manifest records the unqualified fast-start, a distinct
      resumable run identity, and initial accounting before substantial
      generation begins.
- [ ] Record the exact command, run path, PID/log locations, source commit,
      manifest identity, and initial memory telemetry in DEVLOG.md.

**Exit condition:** one recoverable coordinator is processing the 12,111 local
rows with imported A100 provenance and no competing worker.

### Phase F — Complete the remaining 12,111 rows

- [ ] Track completed/imported/failed/pending counts and local-only throughput.
      Refresh ETA from observed dataset rates as the run progresses.
- [ ] Preserve completed greedy answers and each sample immediately. Append a
      complete prediction only when all nine required generations are present.
- [ ] Inspect genuine stalls using worker logs, active request, last token
      progress, memory telemetry, and grading activity before restarting.
- [ ] Pause cleanly when needed; resume the same identity and seeds with
      `--resume`. Keep recorded deadline/recovery settings across restarts.
- [ ] Retry failed rows explicitly with `--retry-failed` after resolving their
      cause. Verify completed rows and samples are not duplicated or regenerated.
- [ ] Make recoverable backups of checkpoints, manifests, provenance, and logs;
      record backup hashes and check a restore from a stable checkpoint copy.
- [ ] Investigate every remaining failed or pending ID. Preserve failure history
      and original settings while completing recovery.
- [ ] Reach **110 imported + 12,111 local = 12,221 complete rows**, with
      **zero failed and zero pending**.

**Exit condition:** all required IDs have valid complete records, and the run
can reconstruct its history from saved artifacts.

### Phase G — Produce and deliver final study results

- [ ] Revalidate all records: exact selection, dataset counts, origins, eight
      samples per row, seeds, score alignment, revisions, and duplicate absence.
- [ ] Generate `report.md`, `summary.json`, `metrics.csv`,
      `metrics_by_execution.csv`, `calibration_bins.csv`, and reliability plots.
- [ ] Verify `run_complete: true`, counts reconcile to 12,221, and every required
      dataset is represented. Account for parse failures, truncation, and missing
      confidence coverage explicitly in the analysis.
- [ ] Label the study **mixed A100/RTX 4060 execution**. Include combined metrics
      and metrics/counts by execution origin. The A100 import contains GSM8K
      rows only, so use paired archived-question comparisons to discuss hardware
      differences rather than attributing differing cohort metrics to the GPU.
- [ ] Compare entropy and self-consistency against the shared greedy-correctness
      outcome; report majority-vote correctness separately. Use the documented
      ECE/AUROC/coverage rules for a recommendation or an insufficient-evidence result.
- [ ] Document measured total runtime, throughput, memory peaks, recovery events,
      numerical differences, remaining scientific limitations, and reproducibility.
- [ ] Publish the final Markdown report, suitable report artifacts, and
      reproducibility information to the repository. Preserve the full raw run
      archive in durable storage and record its location/hash; check artifact
      sizes before deciding how large raw outputs are published.
- [ ] Verify the final remote commit and the recoverable output archive. Close
      the checklist and record the required workflow retrospective.

**Exit condition:** complete, auditable reports and execution history are
delivered with repository references and verified output backups.

## 6. Quick status commands

Use PowerShell from the repository root. Before model loading, inspect RAM and
GPU availability; recheck the worker's loaded-memory telemetry once it starts.

- Available RAM: query Win32_OperatingSystem and read FreePhysicalMemory.
- GPU state: nvidia-smi.

After the full run is created, read current_run_dir.txt under
%LOCALAPPDATA%/TeacherReliability/runs and run:

    python -m teacher_reliability.run --status --run-dir <run-directory>

Use the runbook for detached launch, clean pause, resume, retry, and report
commands. Verify run identity and process ownership before resuming.

## 7. Definition of done

- [ ] Reviewed implementation and documentation are pushed; remote SHA is verified.
- [ ] The waived matrix and 24-question comparison are disclosed. The smoke,
      loaded-memory reading, full-run recovery events, and measured ETA are
      documented; no ETA is inferred from the smoke.
- [ ] Exactly 12,221 rows are valid: 110 A100 imports plus 12,111 local rows.
      Failed and pending required-row counts are both zero.
- [ ] Eight sample seeds, greedy/sample score alignment, source content,
      execution provenance, and no-duplicate checks pass.
- [ ] Complete reports show combined and per-origin metrics, eligibility
      rules, and scientific limitations.
- [ ] Raw archive, manifests, stages, predictions, logs, and reports have
      hash-verified recoverable copies, and a restore check passes.
- [ ] Final results, reproducibility instructions, and the process/concept log
      are pushed to the repository.

Finishing a smoke or launching a background process is an intermediate
milestone. Completion requires all 12,221 required rows, complete reports, and
a verified restore.
