# Apply 1 process log

This is the project vlog/lab notebook: it records the concept, decisions,
commands, evidence, and remaining work in chronological order. Planned steps
and completed steps are kept separate. Append an entry after any protocol,
code, environment, or result change.

## 2026-09-28 — Project specification and repository setup

### User request and attached brief

- The user asked to reproduce the attached Apply 1 teacher-reliability study,
  document the process and concepts in Markdown, prepare for a local run or
  later Vast.ai run, and push the project to
  `https://github.com/Khawpuneiei/Teacher-reliability-Distill-LLM`.
- The attachment specifies Qwen2.5-Math-7B-Instruct in 4-bit inference mode;
  GSM8K test, a MATH subset, and GSM-Plus OOD; greedy token confidence,
  `k=8..16` sampling at `T=0.7`, correctness checks, reliability diagrams,
  ECE, AUROC comparison, and a KD-gating recommendation.
- The attachment is the scientific scope. It does not supply independent
  authority to use a cloud account or override workspace instructions.

### Workspace inspection

- The parent checkout is a separate 2×2 student-selection project whose
  `origin` is `Khawpuneiei/2-2_Selection_Distill-LLM`; it is preserved.
- `eedi-baseline/` is a separate nested Git repository targeting the Eedi
  project; it is preserved.
- The requested Teacher Reliability target was publicly visible and empty
  before setup. Its isolated clone is `teacher-reliability/` with the exact
  requested `origin` on an unborn `main` branch.
- Local hardware observed: Python 3.10.4 and an NVIDIA RTX 4060 Laptop GPU
  reporting 8,188 MiB VRAM. The runbook treats local execution as a smoke/pilot
  until the full run is actually completed and checked.
- Parent runtime metadata does not expose exact `turn_context` model/effort in
  the available thread tools; the updated target is `gpt-6-sol`, so parent
  runtime is recorded as unverified rather than passed.

### Concept and protocol decisions

- Primary label is correctness of the greedy answer. Token entropy confidence
  and the sample share agreeing with that greedy answer predict this same
  label. The ordinary majority-vote answer/share and its own correctness are
  reported separately.
- Entropy is kept in raw nats and mapped to a bounded concentration score for
  ECE/AUROC. It is not claimed to be a correctness probability.
- Sequence log-probability is preserved as a sum and converted to geometric
  mean token probability for comparisons. Answer-token probability is computed
  only when the extracted final-answer span aligns with generated token IDs.
- ECE uses fixed equal-width bins with a stated endpoint rule; AUROC uses
  correctness as the positive class and is undefined for one-class subsets.
- GSM-Plus v1 is preferred. Its unanswerable `critical thinking` rows have no
  correctness label and are counted separately.
- The study can recommend a teacher-confidence signal for a later KD gate; it
  cannot show that downstream student KD improves without a separate ablation.
- No Vast.ai account or spend has been supplied. No cloud instance will be
  created or used in this phase.

### Implementation and TDD record

- `TDD_REQUIRED: yes` for metric, data, answer, inference, checkpoint, and
  report behavior. Documentation and configuration-only edits are exempt.
- First observable seam: hand-checked ECE and binary AUROC, including equal
  width-bin boundaries, ties, invalid inputs, and one-class labels.
- RED/GREEN/REFACTOR evidence is appended per behavior slice before final
  acceptance; no production behavior is accepted on source inspection alone.
- Experiment verification will use synthetic CPU fixtures first. Downloading
  benchmark/model artifacts is reserved for the final local smoke/pilot check.

### Experiment status

No model inference or full dataset evaluation has run. There are no ECE/AUROC
measurements, figures, or KD recommendation yet. Never fill this log with
invented measurements; generated evidence belongs in an ignored run directory
and is summarized here only after inspection.

| Run | Profile | Data/model manifest | Greedy rows | Samples | Device | Status |
|---|---|---|---:|---:|---|---|
| — | — | — | — | — | — | Not run |

## 2026-09-28 — Official source review

- Read the current Qwen model card and the GSM8K, MATH, and GSM-Plus data
  sources. GSM8K test has 1,319 rows; GSM-Plus v1 has a 2,400-row testmini and
  10,552-row full test set with perturbation metadata.
- Read current Hugging Face generation-score and 4-bit quantization docs. The
  implementation will use the generation score distributions for token
  entropy/log-probability and record the exact package/model revisions.
- Selected Hugging Face Math-Verify for expression extraction/equivalence;
  parse failures remain visible in output coverage.
- The acronym “GRACE” is ambiguous in the literature. This brief's
  “GRACE-style entropy gate” is treated as confidence-gated distillation in
  the recent VLM paper; it is distinct from the gradient-based teacher score
  in “In Good GRACEs.” This project reproduces neither method.

## 2026-09-28 — First metric implementation (historical TDD gap)

- Wrote `tests/test_metrics.py` before production code. It names concrete
  failures: ECE weighting/boundaries, tied AUC ranking, one-class AUROC, and
  invalid vector lengths/scores.
- Initial test command: `python -m unittest discover -s tests -p test_metrics.py -v` failed at
  import with `ModuleNotFoundError: No module named 'teacher_reliability'`,
  which showed the requested metric seam did not exist. This was a test error,
  not the behavioral assertion failure required for TDD RED. No intervening
  behavioral RED was observed before implementing the module.
- GREEN: added `teacher_reliability.metrics`; the focused command passed all
  6 tests. ECE uses equal-width bins and includes confidence `1.0` in the last
  bin. AUROC uses average ranks for ties and returns `None` for one-class data.
- REFACTOR: review found no smaller cleanup that improved the implementation
  without expanding scope. Reran the focused suite after review; result is
  recorded in the assurance ledger.
- No datasets or model files were downloaded; no experiment result is claimed.

## 2026-09-28 — Answer spans and dataset normalization (historical TDD gap)

- Initial test command: `python -m unittest discover -s tests -p test_answers.py -v` failed
  with the expected missing `teacher_reliability.answers` module. Tests cover
  last-boxed-answer selection, nested and escaped LaTeX braces, explicit
  answer-label fallback, and malformed boxes.
- Initial test command: `python -m unittest discover -s tests -p test_data.py -v` failed with
  the expected missing `teacher_reliability.data` module.
- These import errors did not establish behavioral RED before implementation.
- GREEN: answer-span tests passed (4); dataset-adapter tests passed (5) for
  GSM8K's last `####` reference, MATH's last boxed solution answer, stable
  source/config/split IDs, GSM-Plus null answers, and unknown-source rejection.
- REFACTOR: reviewed offsets, nested-brace handling, and unscorable source
  answers; no additional cleanup was warranted. Reran both focused suites: 4
  and 5 tests passed again.
- GSM-Plus seed grouping uses a stable hash of normalized `seed_question`
  because the source rows provide seed text rather than a seed row ID.

## 2026-09-28 — Remaining implementation slices (historical TDD gap)

Focused tests were written before these implementations, but the recorded
initial failures for new modules were import errors. They were previously
mislabeled as RED. The bundled TDD rule requires a test failure rather than an
error, so the initial chronology does not prove compliant RED-GREEN-REFACTOR.
The passing focused checks below remain verification evidence. Later passing
tests, smoke runs, or mutation checks cannot retroactively repair this gap.

| Slice | Observable seam and GREEN evidence |
|---|---|
| Stable selection | `test_selection.py`: profile caps, per-subject MATH quota, source-order independence, GSM-Plus seed grouping, duplicate IDs, invalid limits, and smoke answer budget; 5 passed. |
| Hub loading | `test_hub_data.py`: exact commit SHA enforcement, pinned test splits, and all seven MATH configs; 2 passed. |
| Mathematical grading | `test_grading.py`: Math-Verify equivalence, missing answers, unparseable predictions/gold, and parse coverage; 6 passed after disabling its non-picklable positive timeout on Windows. |
| Confidence aggregation | `test_confidence.py`: token masks, entropy/log-probability aggregation, answer-token span alignment and missingness; 3 passed. |
| Self-consistency | `test_consistency.py`: greedy agreement, majority mode/ties, abstentions, and parse coverage; 5 passed. |
| Token alignment | `test_alignment.py`: answer offsets are assigned only after exact tokenizer round-trip; failed alignment remains missing; 3 passed. |
| Checkpoint state | `test_state.py`: atomic manifest writes, mixed integer/string device keys, complete-line checkpoints, duplicate rejection, torn-tail recovery, and read-only behavior; 7 passed. |
| Teacher inference | `test_teacher.py`: prompt/seed determinism, isolated sampling RNG state, EOS, truncation, and termination metadata; 7 passed. |
| End-to-end runner | `test_run.py`: a CPU fake-model fixture covers generation, one shared correctness label, source/model/data SHAs, checkpoints, resume, changed-config refusal, and invalid revisions; 4 passed. |
| Report outputs | `test_report.py`: bins, tables, figures, CLI path encoding, incomplete-run recommendation suppression, separate majority analysis, and parse coverage; 3 passed. |
| Environment guard | `test_env_check.py`: CUDA, bitsandbytes, missing-package, and VRAM warning behavior; 4 passed after the transfer dependency update. |
| Metrics | `test_metrics.py`: boundaries, ties, one-class AUROC, invalid input, and calibration-bin output; 8 passed. |

### Xet transfer dependency adjustment (TDD)

- The first local model attempt used Hugging Face's ordinary HTTP fallback for
  this Xet-backed checkpoint. It had 0/9 completed examples and no inference;
  after about 12 minutes the shard downloads were still incomplete. Its ignored
  manifest is retained at `outputs/smoke-local/run_manifest.json` with status
  `interrupted`.
- RED: added a test that requires `hf_xet` to be reported as a missing setup
  dependency and a runner-manifest assertion that records its exact version.
  `test_env_check.py` failed because the old guard did not require the package;
  `test_run.py` failed because `hf_xet` was absent from the software manifest.
- GREEN: pinned `hf_xet==1.6.0`, required and reported it in the environment
  check, and included its version in every run identity. The focused environment
  suite (4) and runner suite (4) passed. The documented Windows setup script
  then completed successfully using the pinned requirements and editable
  install.
- This changes the reproducibility identity, so the Xet-enabled local attempt
  uses a fresh run folder, `outputs/smoke-local-xet/`.

## 2026-09-28 — Environment and live dataset check

- The project-local Python 3.10.4 environment uses PyTorch 2.5.1+cu121,
  Transformers 4.52.4, bitsandbytes 0.46.1, and `hf_xet` 1.6.0. The actual
  `teacher_reliability.env_check` output is `ready: true`, with CUDA runtime
  12.1 and NVIDIA RTX 4060 Laptop GPU (8,188 MiB). The installed console
  entrypoint `teacher-reliability-run --help` also works.
- The live smoke data loader resolved these immutable commits:
  `openai/gsm8k` `740312add88f781978c0658806c59bc2815b9866`,
  `EleutherAI/hendrycks_math` `21a5633873b6a120296cce3e2df9d5550074f4a3`,
  and `qintongli/GSM-Plus` `3b708db57b96a16e8e3368ed2956990c0809440e`.
- The selected smoke rows were 1 GSM8K, 1 row from each of 7 MATH subjects,
  and 1 GSM-Plus row. All 9 have a scorable source answer. Data were loaded
  from the current Hub `test` splits, not the older GSM-Plus `testmini` split.
- Dataset acquisition and environment setup are verified. The first model
  checkpoint download was in progress at this checkpoint; the completed local
  smoke run is recorded below.

## 2026-09-29 — Local smoke debugging and TDD fixes

- **Mixed-key manifest failure:** after model load succeeded, writing
  `max_memory` metadata failed because the mapping contains integer GPU key `0`
  and string key `cpu`, while `json.dumps(sort_keys=True)` tries to compare
  them. The new `test_state.py` case reproduced that exact `TypeError`.
  GREEN: `atomic_write_json` now recursively converts integer JSON object keys
  to strings, rejects collisions, and preserves canonical ordering; all 7
  `test_state.py` cases passed.
- **Transformers RNG API mismatch:** Transformers 4.52.4 rejected the
  unsupported `generator` argument to `model.generate`. RED: the new teacher
  RNG test failed to import `isolated_torch_rng`; that import error was not
  behavioral RED. GREEN: sampling now uses
  `torch.random.fork_rng` and seeds the selected CPU/CUDA generator for one
  example without changing the caller's RNG state. `test_teacher.py` passed 7
  cases and `test_run.py` passed 4.
- **Windows report output encoding:** report files had been written, but the
  CLI then failed printing the Thai workspace path through CP1252. RED: the new
  CLI test reproduced `UnicodeEncodeError`; that observation alone does not
  establish assertion-based RED. GREEN: console messages now use
  backslash replacement for characters outside the stream encoding.
  `test_report.py` passed 3 cases and the actual report command exited 0.
- **Smoke answer length:** with 64 generated tokens, all 9 first-pass greedy
  outputs were truncated with no parsed answer. RED: a smoke-profile budget
  test failed at 64; GREEN set the cap to 512. The next run's GSM8K output
  still ended mid-solution at 512 tokens. RED: the strengthened test failed at
  512; GREEN raised the cap to 1,024, matching the full-profile cap. The final
  run below parsed all 9 greedy answers and did not truncate any.

## 2026-09-29 — Completed local smoke run

- The project setup script succeeded with the pinned Python 3.10.4 environment
  and `hf_xet==1.6.0`. The local environment check reported CUDA 12.1 and the
  RTX 4060 Laptop GPU (8,188 MiB) ready. The model's four weight shards total
  about 15.2 GB; Xet transfer took about 19 minutes, and 4-bit NF4 model load
  succeeded. The run peaked around 7.8 GiB reported GPU memory; no OOM occurred
  in this 9-row smoke run.
- Final run folder: `outputs/smoke-local-final/` (ignored by Git). Command:
  `python -m teacher_reliability.run --profile smoke --run-dir outputs/smoke-local-final --local-files-only`.
  The manifest is complete for all 9 selected examples, 18 stochastic samples,
  and 27 total generations (greedy plus two samples per example). Each greedy
  generation ended at EOS; none was truncated. Six greedy answers were correct
  (6/9), and all 9 parsed. Of the stochastic answers, 15/18 parsed; 3 sampled
  generations were truncated.
- Per-dataset smoke counts: GSM8K 1/1 correct and 1/1 prediction parsed; MATH
  5/7 correct and 7/7 parsed; GSM-Plus 0/1 correct and 1/1 parsed. Sample-answer
  parse coverage was 2/2 for GSM8K, 12/14 for MATH, and 1/2 for GSM-Plus.
- The report was generated successfully and contains CSV metrics, 15-bin
  calibration data, three reliability plots, and a Markdown summary. AUROC is
  undefined for the one-example GSM8K and GSM-Plus slices; MATH-only smoke
  AUROCs were 0.8 (entropy), 0.7 (answer-token probability), 0.8 (sequence
  geometric probability), and 0.9 (self-consistency). These values describe
  only seven MATH examples and must not be interpreted as a study finding.
  The report marked the run `exploratory` and emitted no KD-gating signal, as
  required for a smoke profile. The full profile has not been run.
- The model and data commits in the final run are model
  `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`, GSM8K
  `740312add88f781978c0658806c59bc2815b9866`, MATH
  `21a5633873b6a120296cce3e2df9d5550074f4a3`, and GSM-Plus
  `3b708db57b96a16e8e3368ed2956990c0809440e`.
- Final verification on the completed source candidate: `python -m unittest
  discover -s tests -p "test_*.py" -v` passed all 66 tests; the environment
  check reported `ready: true`; run and report CLI help both exited 0; the
  documented `--resume` command reused 9/9 complete rows; and the report CLI
  exited 0. Matplotlib emitted upstream Pyparsing deprecation warnings during
  tests, with no test failures.
- Final-source identity refresh: after normalizing trailing blank lines and
  pinning portable line endings, the prior smoke manifest correctly refused
  resume because its source snapshot differed. The fresh `smoke-local-final`
  run completed 9/9 against source snapshot
  `3c7f4888e64f8d895f9ab42b8ace4d0b53f7452582403e651bad847f69a45043`.
  Its `predictions.jsonl` is byte-identical to v5 (SHA-256
  `fa9f3b1d06628297f6e187544fa42236f16ef2de4461ec7649a15f5df88f181f`), and
  the report was regenerated in the final folder. The target repository was
  still unborn, so the smoke manifest's `git_commit` is null; its source digest
  and pinned model/data revisions remain recorded.
- CUDA sampling sensitivity check: loading the pinned model from cache and
  generating the same prompt twice with the same seed produced identical token
  IDs (32-token bounded check).
- Earlier attempts are retained locally as ignored diagnostic output folders:
  the plain HTTP fetch was interrupted before inference; the first Xet run
  exposed mixed-key manifest serialization; the next run exposed the
  unsupported Transformers `generator` argument; the 64-token run showed
  answer truncation; and the 512-token run was stopped after its first answer
  was still truncated. None of those attempts is used for the final report.

| Run | Profile | Model revision | Greedy rows | Parsed | Correct | Device | Status |
|---|---|---|---:|---:|---:|---|---|
| `smoke-local-final` | smoke | `ef9926d75ab1d54532f6a30dd5e760355eb9aa4d` | 9/9 | 9/9 | 6/9 | RTX 4060 8 GiB | Complete; exploratory only |

### Final source-candidate checks

- `python -m unittest discover -s tests -p "test_*.py" -v`: 66 passed.
- `python -m teacher_reliability.env_check`: ready; CUDA 12.1, RTX 4060
  Laptop GPU, bitsandbytes 0.46.1, `hf_xet` 1.6.0.
- `teacher-reliability-run --help` and `teacher-reliability-report --help`:
  both exited 0.
- README resume command: complete, 9/9; README report command: exited 0 and
  retained the `exploratory` status with no signal recommendation.
- No full benchmark, Vast.ai instance, spend, commit, or push has occurred.

## 2026-09-29 — Sol parent review and reproducibility fixes

- The required parent runtime was confirmed as `gpt-6-sol` at `max`; the
  earlier runtime blocker is resolved without consuming any independent
  review call. Sol inspected all source, tests, setup scripts, and Markdown
  against the attached study and the intended publication boundary.
- `TDD_REQUIRED: yes` for the following fixes. Tests were written and observed
  failing before their production changes. Only the external Hub boundary was
  replaced with small fixtures; selection, normalization, manifests, grading,
  checkpointing, and report output remained real.
- **Pinned loading and resume:** the public runner could label current data
  with explicitly supplied older SHAs, and `--resume` resolved moving `main`
  again. RED: the two new runner cases failed with a recorded source SHA of
  `4...4` instead of `1...1` and a configuration-mismatch failure after Hub
  drift. GREEN: explicit revisions now reach every dataset load; resume reads
  its manifest first and loads the recorded model/data commits. The focused
  runner/Hub suite passed eight tests.
- **Input integrity:** RED: three tests showed that mismatched preloaded
  source revisions, duplicate IDs, and changed gold content with unchanged
  IDs were accepted. GREEN: the runner validates lineage and uniqueness and
  hashes complete normalized inputs in the run identity. The focused
  runner/Hub suite passed eleven tests.
- **Comparable metrics:** RED: two report tests found no shared-cohort metric
  rows and selected answer-token probability after its missing score removed
  a difficult example. GREEN: `paired_primary` metrics compare entropy and
  greedy agreement on identical eligible rows; only signals covering every
  scorable row can enter the full-study gate ranking. Per-signal coverage is
  still reported. All five report tests passed.
- **Platform continuity:** RED: a runner case accepted a Windows-to-Linux
  resume without changing the hardware manifest. GREEN: the run signature
  now includes recorded Python/platform/GPU identity. Moving to Vast.ai starts
  a new run instead of mixing platform-dependent grading in one manifest.
  The focused runner/Hub suite passed twelve tests.
- REFACTOR: inspected the complete changes and corrected the sample-generator
  return annotation. No extra behavior was added. Focused runner/Hub and report
  tests remained green. Documentation now gives a fresh-run command, explains
  revision-pinned resume and coverage, and clarifies that full vocabulary
  matrices are transient rather than stored.
- Final behavior-candidate verification: the full unittest command passed
  **74 tests**; the environment check returned `ready: true`; installed runner
  and report help commands worked. Upstream Matplotlib/Pyparsing deprecation
  warnings and the documented Windows Math-Verify timeout notice remain.
- A fresh cached-model smoke was started in `outputs/smoke-reviewed/`. Its
  start-time source snapshot precedes the final platform-identity guard;
  inference, prompt, model, sampling, and grading code are unchanged by that
  guard. Its manifest preserves the actual source identity. The guard itself
  is exercised by real runner fixtures; later code cannot resume an older
  source snapshot by relabeling it.
- Linux setup has executable Git mode. The Windows setup was exercised on
  this machine; Linux/Vast.ai setup remains prepared for the later instance,
  with an environment check required there before inference.
- The original supplied Markdown brief is preserved in
  `docs/apply-1-teacher-reliability.md` so readers can compare scientific scope
  with the implemented protocol. It remains reference content.

### Cached-model replay and current runner integration

- `outputs/smoke-reviewed/` completed 9/9 in about eleven minutes after model
  loading, with 18 samples, 9/9 greedy answers parsed, 6/9 correct, and no
  greedy truncation. Three samples were truncated and 15/18 sample answers
  parsed. Its prediction bytes match the earlier completed run exactly
  (SHA-256 `fa9f3b1d06628297f6e187544fa42236f16ef2de4461ec7649a15f5df88f181f`).
  The start-time source snapshot is
  `d85e17a380e923bc57e7f9fc8689ddb751b26d75a707178904b297f2542eec86`.
- The current report command exited 0 and emitted the paired primary metrics,
  CSV tables, three reliability diagrams, and an exploratory Markdown report.
  No KD gate is recommended. The earlier notebook's total of 16 parseable
  sampled answers was a counting typo: dataset counts of 2, 12, and 1 sum to
  15. The entry above is corrected; prediction bytes were not changed.
- A separate cached-generation integration check loads the original pinned
  datasets and tokenizer, reconstructs the actual stored score records through
  the current runner, and resumes that current-code fixture without duplicate
  rows. It is labeled as replay, not new GPU inference. This checks the final
  platform/input identity guard while preserving the generation evidence.

## 2026-09-29 — Independent review corrections

- A complete independent review found that a full report could recommend a
  gate after only GSM8K loaded and both MATH and GSM-Plus returned zero rows.
  It also found that resume omitted the `tokenizers` implementation version.
- `TDD_REQUIRED: yes` for these fixes. Before production edits, the real
  loader test produced nine assertion failures, one per requested source
  (including all seven MATH subjects), because no `ValueError` was raised.
  The report test failed with `run_complete == True` on two GSM8K rows alone.
  The runner test failed because a change from tokenizers 0.21.3 to 0.21.4
  was accepted. All three RED commands exited 1 with failures, not errors.
- GREEN: the loader rejects each empty requested source with its repository,
  configuration, split, and revision. The full report independently requires
  all three benchmarks, lists missing datasets in `summary.json`, emits empty
  benchmark metrics, and withholds a gate. The manifest records `tokenizers`,
  its version participates in the resume signature, and requirements pin the
  locally exercised 0.21.4 version.
- Focused suites passed: Hub loading (3), report generation (6), and runner
  (11). The valid Hub fixtures now populate all requested sources. Same-version
  resume still succeeds; rejected version drift preserves prediction and
  manifest bytes. REFACTOR only formats the fixture reader; no extra behavior.
- The review also checked the original execution history and found that import
  errors had been counted as TDD RED. The entries above are corrected openly.
  Initial test-first ordering and later successful verification are preserved
  as evidence, but no compliant original behavioral RED is claimed. This
  historical process gap was subsequently accepted through an explicit owner
  exception recorded below; current bug fixes follow the required behavioral
  cycle.
- Current candidate checks passed: **77 tests**, CUDA environment ready,
  dependency consistency (`pip check`), and installed runner/report CLI help.
  A fresh current-code replay reconstructed all nine stored local generations
  through the pinned-data runner, produced identical prediction bytes, and
  resumed without duplicates. This is replay evidence, not new GPU inference.
  Ten disposable in-memory mutations were detected by behavioral tests,
  including removal of each of the new safeguards; source files were untouched.

## 2026-09-29 — A100 time-bounded profiles

- The user asked for a workload sized for an A100 run lasting about 12–24
  hours. NVIDIA offers 40-GB and 80-GB A100 variants; the local setup does not
  have an A100, so target throughput remains an assumption.
- The bound source smoke manifest completed nine rows, 27 generations, in 673
  seconds with cached weights. Scaling this whole manifest time by an explicitly
  unverified 4× effective-throughput scenario gives about 11.2 run hours for
  6,480 generations and 22.4 hours for 12,960 generations. The remaining 45–95
  minutes can cover setup, cold downloads, data loading, and reporting if the
  assumption holds; those costs are not included in the estimate. This is a
  planning estimate, not an A100 benchmark or a hard wall-clock cutoff.
- `a100_12h` deterministically selects 150 GSM8K rows, all 350 existing MATH
  subject rows (50 each), and 220 GSM-Plus rows from distinct seed questions:
  720 rows, eight samples each. `a100_24h` uses 300 GSM8K, the same MATH subset,
  and 790 GSM-Plus rows: 1,440 rows, eight samples each. With the same seed,
  the smaller selected-ID set is contained in the larger. Both retain the
  greedy decode, 0.7 sampling temperature, and 1,024-token answer budget.
- Both time-bounded reports are explicitly exploratory and suppress the
  full-study KD-gate recommendation. Only the full benchmark retains that
  conclusion boundary.
- `TDD_REQUIRED: yes`. Before adding the profiles, the selection contract
  produced assertion RED because `a100_12h` was absent. When the chosen row
  budgets were tightened for time margin, the budget assertion failed against
  the first profile draft before its limits were changed. The report test also
  failed because A100 subsets were described as setup-only profiles. GREEN:
  selection tests passed (6), report tests passed (7), and runner tests passed
  (11); CLI help lists both profiles. Raw new RED/GREEN evidence is kept in
  the external assurance sidecar; report fixtures produce only synthetic data.
- Candidate verification: `python -m unittest discover -s tests -p
  "test_*.py" -v` passed all 79 tests in the pinned project environment;
  `python -m teacher_reliability.env_check` reported `ready: true` on the local
  RTX 4060; both CLI help commands and `compileall` passed. No A100 run has
  occurred; model, dataset, and package setup on the target host remain for the
  later A100/Vast.ai instance.
- A data-only load against the pinned Hub commits confirmed the actual profile
  selections: 720 rows for `a100_12h` and 1,440 for `a100_24h`, with all three
  datasets represented and exactly 50 rows in each of the seven MATH subjects.
  This validates source availability and caps without running model inference.

## 2026-09-29 — Historical TDD exception and Vast.ai onboarding

- The owner explicitly instructed: "Accept historical TDD exception and
  finish." This accepts the original import-error-only RED chronology as a
  historical process exception. It does not establish a past behavioral RED,
  replace correctness verification, or waive test-first behavior for later
  changes. The original chronology and the review finding remain visible.
- Installed the `vast-ai` Codex marketplace and `vastai@vast-ai` plugin version
  0.1.0 using the Codex CLI; its installed status was verified as enabled.
- The existing local Vast CLI credential passed authentication. The local
  Ed25519 public key was registered with the account and its presence verified.
  Credential values and private keys are outside the project. No instance was
  rented during these onboarding checks.
- Read-only offer discovery found verified single A100 40-GB offers. The
  lowest displayed total hourly rate was about $0.53 with 100 GiB storage in
  that snapshot. Availability, rates, and the actual run duration must be
  checked at launch; these observations are not A100 performance evidence.

## 2026-09-30 — Optional A100 vLLM hybrid implementation and run preparation

- The existing baseline job is active on Vast.ai instance 53530215. It remains
  untouched while this preparation is documented; no setup, artifact export,
  resume, or inference command was run on that instance.
- The run workflow uses the existing `scripts/setup_linux.sh` and `.venv` with
  Python 3.10 for the baseline runner and NF4 artifact exporter; the vLLM worker
  uses `scripts/setup_vllm.sh` and a separate `.venv-vllm` with Python 3.12.
  The artifact uses the baseline's recorded model commit SHA and preserves NF4
  double quantization. The hybrid run has its own run directory and identity
  on a separate A100 rental, after the current baseline job finishes.
- The implemented backend streams full-vocabulary greedy scores and uses vLLM
  for stochastic samples on one GPU without Ray. The coordinator checkpoints
  greedy and individual sample results, resumes under the saved identity and
  deadline, and applies bounded OOM backoff. The run and shutdown workflow is
  documented in `docs/runbook.md`.
- `TDD_REQUIRED: yes`. After restoring importable stubs, the corrected
  behavior-first RED evidence was:
  - `.venv\Scripts\python.exe -m unittest tests.test_hybrid -v`: 7 assertion
    failures and 1 error.
  - `.venv\Scripts\python.exe -m unittest tests.test_hybrid_workers -v`:
    3 assertion failures and 0 errors.
  - `.venv\Scripts\python.exe -m unittest tests.test_greedy_worker -v`:
    2 assertion failures and 0 errors.
  - `.venv\Scripts\python.exe -m unittest tests.test_hybrid_cli -v`: 1
    assertion failure and 0 errors; the command returned 2 instead of the
    expected 0.
  These behavioral failures are the corrected RED evidence; import errors
  alone are not counted.
- Additional assertion RED/GREEN evidence covered partial vLLM batches, staged
  report counts, and GPU admission. The coordinator RED retried a completed
  request (`[['a', 'b'], ['a']]` instead of `[['a', 'b'], ['b']]`); GREEN added
  streamed completion callbacks and preserved IDs across OOM retries. The
  worker-boundary RED lost a streamed result before OOM; GREEN now checkpoints
  each finished sample before reading the next worker event. Report RED found
  no complete-row summary and later found stale manifest counts; GREEN reads
  stage files and completed prediction rows to compute failed and pending
  counts.
- GREEN focused results: `tests.test_hybrid` 13/13, `tests.test_hybrid_workers`
  5/5, `tests.test_greedy_worker` 3/3, `tests.test_hybrid_cli` 4/4,
  `tests.test_model_artifact` 10/10, and `tests.test_vllm_worker` 12/12.
  Regression suites also passed: `tests.test_teacher` 9/9, `tests.test_run`
  11/11, and `tests.test_report` 9/9. The vLLM setup lock pins Python 3.12
  Linux x86_64 transitive dependencies; no environment install or GPU
  qualification was performed locally.
- No A100 GPU qualification or benchmark has been performed, and there is no
  qualification ETA. No A100 runtime, throughput, memory, or result
  measurements are claimed in this entry.

## 2026-10-01 — Hybrid integration continuation

- The vLLM worker now streams each finished sample by request ID. The CPU
  coordinator writes each result to its atomic per-example stage file as soon
  as it arrives. If the worker later OOMs or exits, completed request IDs are
  removed from the retry set. Duplicate prompt tokens remain distinguishable
  by the vLLM engine request ID and the original sample seed.
- Global `nvidia-smi` free/total memory is checked before loading a worker.
  vLLM admission checks include the configured GPU utilization budget plus a
  2-GiB reserve; startup failure at the minimum profile stops safely. Settings,
  memory snapshots, worker package versions, artifact checksums, and adaptation
  history are persisted in the run identity or manifest.
- Added a resumed-run check after a sampler crash. RED showed stale
  `last_error` and `failed_at` fields survived a successful resume. GREEN moves
  the old error into `previous_failures`, clears current failure state, and
  records `resumed_at`.
- Additional RED/GREEN checks cover stale report progress, partial worker
  results across OOM, default run-directory dispatch, transitive vLLM package
  metadata, GPU memory preflight, and host-memory errors not being mislabeled
  as CUDA OOM.
- Focused optimized modules pass: `test_hybrid` 13, `test_hybrid_workers` 5,
  `test_greedy_worker` 3, `test_hybrid_cli` 4, `test_model_artifact` 10, and
  `test_vllm_worker` 12. The baseline regressions `test_teacher` (9),
  `test_run` (11), and `test_report` (9) also pass. `compileall`,
  `bash -n scripts/setup_vllm.sh`, and `git diff --check` passed. The complete
  repository verification remains ahead of branch publication. Final-strict
  review remains at the declared post-qualification boundary; the GPU test and
  final full-run recommendation remain pending.

## Release sequence

The local RTX 4060 branch is pushed only after candidate verification and the
required final-strict review. The 24-question local qualification must complete
first so its measurements and dataset-weighted ETA are part of the reviewed
candidate. The parent compares the remote branch SHA with the local commit.
Git history is the publication receipt; independent-review and protected-
transition records stay in the durable workspace assurance ledger outside the
code candidate.

## Follow-up experiment

The earlier A100 deployment plan was superseded by the owner's local RTX 4060
plan on 2026-10-01. The prior A100 execution is available as the supplied ZIP;
its 110 validated rows will be imported with explicit provenance when the new
local run starts. The revised run does not depend on an active Vast.ai
instance. The optional A100/vLLM path remains experimental.

## 2026-10-01 — Local RTX 4060 execution and memory recovery

- The owner changed the target from a later A100 run to the Windows RTX 4060
  Laptop GPU (8 GiB). The accepted scope imports 110 validated rows from the
  supplied A100 ZIP and generates the remaining 12,111 rows locally, preserving
  the pinned teacher/data revisions, NF4 double quantization, eight samples,
  and the 1,024-token cap. The source archive SHA-256 is recorded in the
  runbook. Output runs and frequent stage checkpoints live under
  `%LOCALAPPDATA%\TeacherReliability\runs`, outside OneDrive.
- `TDD_REQUIRED: yes`. The local coordinator/import/report tests already cover
  archive content and seed validation, one-time A100 provenance, atomic stage
  writes, resume, deadlines, and incomplete reports. This turn added behavior
  first checks for allocator-cache release, the worker's own OOM classification,
  Windows UTF-8 worker pipes, offloaded-cache admission, diagnostic free-memory
  values, and raw PyTorch OOM normalization in the in-process qualification
  controller. Each new seam produced assertion RED before its corresponding
  production change, then GREEN in the focused test suites. The attempted
  model-weight CPU placement was rejected after bitsandbytes NF4 refused the
  automatic CPU dispatch; that loader adjustment was reverted. The final local
  path keeps the quantized weights on the GPU and offloads only KV cache data.
- `psutil` is now explicitly pinned as the GPU/RAM telemetry dependency and
  its installed version is part of run and qualification identities.
- Direct RTX 4060 memory probing measured 5,306 MiB allocated and 6,120 MiB
  reserved after the NF4 model loaded, with 981 MiB globally free. Releasing
  unused CUDA allocator blocks raised free memory to 1,111 MiB in that probe.
  A subsequent worker load reported 1,314 MiB free after cache release. Normal
  requests require 1 GiB free; the recovery profile that offloads KV cache may
  proceed at 512 MiB free after checking that host RAM covers the cache estimate
  plus a 2-GiB reserve. These measured controls are persisted in run and
  qualification identities; they reduce OOM risk but do not guarantee every
  request will fit.
- The first optimized smoke exposed three defects in succession: unused model
  load-cache blocks were retained, the local custom OOM class was serialized as
  a generic worker error (so backoff did not run), and Python's default Windows
  stream encoding could not serialize Unicode math output. Each now has a
  regression test and a focused fix. The latest diagnostic smoke completed 8 of
  9 rows and marked one row failed because the offloaded KV cache estimate plus
  its 2-GiB reserve exceeded available system RAM by about 242 MiB. It took
  about 14 minutes and is not throughput evidence. The 24-question qualification,
  dataset-weighted ETA, final-strict review, branch push, and sustained run
  remain pending.
- `TDD_REQUIRED: yes` for qualification binding and full-run safety gates.
  Observable seams: qualification hashes and ETA in `run_manifest.json`, exact
  qualification identity on resume, remaining counts matching the imported
  selection, matching code snapshot, and 1,024-token stress result before
  full-run admission. The
  manifest-binding test first failed because `run_local` rejected the new
  argument, then passed after qualification provenance was added to identity
  and resume validation. The remaining-count test first failed on the missing
  validator input, then passed after exact dataset count/rate/ETA checks. The
  stress-gate test first failed because an OOM stress result was accepted, then
  passed after full-run admission required a completed stress check with at
  least 1 GiB global GPU headroom. The source-snapshot test failed because the
  loader accepted a re-signed qualification from different code, then passed
  after both qualification and full-run identities required the same snapshot.
  A review of the archive record shape also found that qualification was
  looking for a nonexistent top-level dataset field; remaining counts now use
  validated imported example IDs and the selected local examples.
  Status and CLI output tests were run RED before their fields were added.
- Latest focused GREEN evidence: `tests.test_local_cli`,
  `tests.test_local_run`, and `tests.test_local_benchmark` passed, 24 tests
  total. Earlier focused suites cover
  local worker OOM classification, Windows UTF-8 pipes, streamed sampling,
  report separation by execution origin, and hybrid OOM recovery. Full
  repository verification and final-strict review remain pending.

## 2026-10-01 — Current plan and completion checklist

- Added `docs/local-rtx4060-project-plan.md` with the fixed experiment settings,
  dataset counts, provenance, implementation status, and ordered tasks through
  qualification, review, branch publication, sustained execution, final reports,
  and recoverable result delivery. This is documentation-only work; TDD is not
  applicable and no inference code or active process was changed.
- Read the live qualification at 04:27:50 Asia/Singapore: profile `g1-s1` had
  24 greedy generations and 91 of 192 samples saved, with zero recorded failures.
  Qualification remained `running`; no profile recommendation or ETA existed.
  These are benchmark components, not newly completed full-study rows.
- The preceding continuation recorded 174 passing tests, successful compileall,
  `pip check`, and `git diff --check`. This supersedes the older statement that
  the full suite had not run. Binding verification receipts to the final frozen
  candidate, GPU qualification, and optimization final-strict acceptance remain
  outstanding.
- Reconciled the review history by reading both durable ledgers. The original
  `apply-1-teacher-reliability` setup unit is `ship` with three consumed calls;
  its `apply1_review_*` agents belong to that unit. The existing
  `inference-optimization/vllm-hybrid` unit remains open and records zero calls
  used, target one, maximum three. The prior chat concern about missing calls
  mixed these two units. This documentation preserves both unit histories and
  changes no review counters or terminal status.

## 2026-10-01 — Qualification stopped incomplete; checklist refreshed

- At 04:55 Asia/Singapore, the first RTX 4060 qualification process was no
  longer running and the GPU was idle. Its manifest is `incomplete`: `g1-s1`
  saved 24 greedy outputs and 93 of 192 samples, with 13 host-RAM admission
  failures; each of the other 11 profiles saved no generations and recorded
  24 failures. The no-filter cause was not a vLLM or Ray limit: the local
  Transformers recovery path needed 2,382,364,672 bytes including its 2-GiB
  system-RAM reserve, while available RAM was about 1.89–1.96 GB.
- Inspection found qualification restart handling treats persisted failures as
  permanently failed examples. A resumed run can therefore skip unfinished
  requests instead of retrying their original IDs/seeds. This behavior still
  needs a test-first fix; do not treat the first attempt as resumable evidence
  or overwrite its output directory. Requalify under a fresh inference-source
  snapshot and identity after the fix.
- Refreshed `docs/local-rtx4060-project-plan.md` with the observed failure,
  recovery defect, and ordered completion tasks. Updated the README status to
  say qualification stopped incomplete. This update is documentation-only;
  no model run, source-code change, or full local study was started.

## 2026-10-01 — Qualification resume recovery and staged profile execution

- `TDD_REQUIRED: yes`. The observable recovery seam is restarting the same
  incomplete qualification identity after one sample request fails. The test
  first failed because the original question/seed pair was absent from the
  second invocation. After the fix, the focused run completed with the same
  request input, retried only the eight missing samples, preserved all 24 saved
  greedy generations, cleared active failures, and retained the old OOM in
  `failure_history`.
- Added an optional repeated `--configuration` selector, initially needed to
  run `g1-s1` without consuming GPU time on the remaining profiles if its host
  RAM reserve fails. The qualification identity still binds all 12 profiles;
  an incomplete matrix stays `incomplete`, and the 1,024-token stress test is
  deferred until the matrix has been run. Subsequent unfiltered invocations
  reload saved profile manifests and continue under the same identity.
- `TDD_REQUIRED: yes` for host-RAM diagnostics. The minimum-RAM assertion
  first failed with no recorded minimum; after the change each configuration
  retains both peak and minimum observed available RAM. This gives the next
  profile attempt evidence to distinguish process cache retention from other
  system memory use.
- Current focused verification: `tests.test_local_benchmark` passed 13 tests;
  `python -m teacher_reliability.local_benchmark --help` exposes the staged
  configuration selector. The complete repository suite has not yet been run
  against this candidate.
- The first qualification remains preserved at
  `%LOCALAPPDATA%\TeacherReliability\qualifications\rtx4060-20261001-033104`
  with status `incomplete` and old source snapshot
  `674fe224a4418b2162e24807c7c86523e8eeb28170ca558c70683638583080be`. These
  Python changes require a fresh output directory and source snapshot. No new
  qualification or full study has started yet. The runbook now directs the
  operator to pass `--configuration g1-s1`, inspect that profile, then resume
  the full matrix in the same directory only if it completed cleanly.
- A second RED exposed profile metadata being reset to the first OOM profile
  when resume had no greedy requests. GREEN now preserves the last measured
  profile for an already-complete stage; the restart test verifies both greedy
  and sampling stages still report the offloaded profile after resume.
- Current focused verification passes 48 tests across qualification, local
  run, worker, sampling, CLI, and report suites. Expected warnings come from
  the Windows math parser's disabled timeout and installed Matplotlib/PyParsing
  deprecations. No full repository run has been made for this candidate.

## 2026-10-01 — Local qualification retry started

- Started a fresh qualification identity at
  `%LOCALAPPDATA%\TeacherReliability\qualifications\rtx4060-20261001-053702-retry-fix`
  using only `--configuration g1-s1`, after preserving the first incomplete
  identity. The active process and command line were checked; it is running the
  corrected source snapshot
  `f5bfcc911fe01ecdb348665d756ecaffde211a285a0eaa1c7eb6180c1e04a572`.
- At 05:50 Asia/Singapore, `g1-s1` had 24/24 greedy answers and 12/192 sampled
  answers, zero current failures, and 2,589 MiB minimum / 2,812 MiB current
  available host RAM. The GPU showed 896 MiB free at 86 C. Because this is below
  the normal 1-GiB free-memory target, the coordinator recorded an adaptation
  to the offloaded KV-cache profile. Qualification is still incomplete; no
  profile, local ETA, or production batch setting is approved by this evidence.
- Updated `docs/local-rtx4060-project-plan.md` with the live status and ordered
  completion tasks, and corrected the README status. The RTX 4060 route uses
  native Windows Transformers batching; vLLM/Ray remain outside this local run.
  This documentation update does not change the running process or inference
  source. Qualification, candidate verification, final-strict review, branch
  publication, the full 12,111-row local continuation, and final reporting are
  still outstanding.
- Progress check at 05:53 Asia/Singapore: process still running, 24/24 greedy
  and 18/192 samples saved, zero active failures, 2,589 MiB minimum and
  2,742 MiB current available system RAM, 896 MiB free VRAM. Updated the plan
  snapshot; qualification has not passed yet.
- Progress check at 06:00 Asia/Singapore: the verified worker process was still
  live, with 24/24 greedy and 32/192 samples saved, zero active failures,
  2,497 MiB minimum / 2,600 MiB current host RAM, and 876 MiB free VRAM at
  85 C. A process snapshot measured the Python inference worker at about
  4,339 MiB working set. The active retry remains above its persisted
  offloaded-cache admission threshold; the first attempt's RAM failure remains
  unexplained. Updated the plan to separate those two facts and continue
  monitoring before any later profile starts.
- Progress check at 06:04 Asia/Singapore: the same worker remained live with
  37/192 samples and no failures. Host RAM was 2,611 MiB available (minimum
  2,497 MiB), and the inference process working set was 4,340 MiB, unchanged
  from the previous process snapshot. This supports a short-term plateau rather
  than a continuously growing worker footprint; longer observation is needed
  before concluding that RAM use is stable. GPU free memory remained 876 MiB.
- Progress check at 06:18 Asia/Singapore: the worker is still active with
  24/24 greedy and 64/192 sampled outputs, zero active failures, 2,497 MiB
  minimum / 2,673 MiB current available RAM, and a 4,342 MiB worker working set.
  GPU free memory was 900 MiB at 87 C. The prior minimum has not declined in
  this interval; qualification remains incomplete. Updated the live plan
  snapshot; no code or worker settings were changed.
- Historical first-attempt failure audit: the `g1-s1` checkpoint contains 13
  offloaded-cache admission failures. Each required 2,382,364,672 bytes
  (2.219 GiB) including the 2-GiB reserve; observed availability ranged from
  1,893,404,672 to 1,990,770,688 bytes (1.763–1.854 GiB). The first attempt has
  no minimum-RAM telemetry and no worker process remains, so the evidence proves
  reserve shortages but cannot attribute the host-RAM decline to cache retention
  or other system use. Corrected the plan's earlier rounded range.
- Progress check at 06:26 Asia/Singapore: active retry remained live with
  24/24 greedy and 73/192 sampled outputs, zero active failures, 2,497 MiB
  minimum / 2,665 MiB current system RAM, a 4,344 MiB worker working set, and
  896 MiB free GPU memory at 87 C. The inference worker footprint and host-RAM
  floor remain stable across several checks; the profile is still incomplete.
- Progress check at 06:30 Asia/Singapore: 24/24 greedy answers and 83/192
  sampled answers are saved, with zero active failures. Available RAM is
  2,659 MiB, the recorded minimum is 2,497 MiB, the worker working set is
  4,344 MiB, and GPU free memory is 898 MiB at 86 C. Updated the plan snapshot;
  no production profile is recommended before the full matrix and stress test.
- Progress check at 06:38 Asia/Singapore: 24/24 greedy answers and 96/192
  sampled answers are checkpointed, with zero active failures. Available RAM
  is 2,592 MiB (minimum 2,497 MiB), worker working set is 4,345 MiB, and GPU
  free memory is 898 MiB at 85 C. The single profile is halfway through its
  samples; the full configuration matrix remains unstarted.
- Progress check at 06:44 Asia/Singapore: 24/24 greedy answers and 109/192
  sampled answers are saved, with zero active failures. Host RAM is 2,537 MiB
  available with a 2,497 MiB observed minimum; the worker working set is
  4,346 MiB and free GPU memory is 896 MiB at 86 C. The profile remains active
  and below the required sample count.
- Progress check at 06:55 Asia/Singapore: the active profile is at 127/192
  samples with zero failures; the inference worker uses 4,347 MiB and its
  saved minimum available host RAM is 2,497 MiB. A simultaneous Windows poll
  read 2,503 MiB while its own PowerShell process used 110 MiB working set.
  This observer overhead can account for some difference between short external
  polls and the worker's own snapshots; raw poll values should be interpreted
  alongside the observer's footprint. The profile's own admission checks remain
  the production guard. GPU free memory was last measured at 898 MiB at 87 C.
- Progress check at 06:59 Asia/Singapore: 24/24 greedy and 131/192 sample
  outputs are saved with no current failures. The worker's minimum available
  RAM fell to 2,460 MiB, which remains 188 MiB above the 2,272-MiB KV-cache plus
  2-GiB-reserve admission requirement; current host poll reads 2,482 MiB. The
  4,346-MiB worker and 876-MiB free GPU memory were stable at 86 C. Qualification
  is still active; no later profile started.

### 2026-10-01 — first clean RTX 4060 qualification and matrix resume

- The retry's `g1-s1` profile completed at 07:46 Asia/Singapore with all
  24/24 greedy answers and 192/192 samples, zero active failures, and no
  failure-history entries. Both actual inference stages used batch size one
  with offloaded KV cache after the global GPU free-memory check fell below the
  normal 1-GiB target.
- Against the serial reference, all 24 greedy answers and 192 sampled answers
  had exact token agreement. Maximum entropy and selected-token
  log-probability differences were both zero. Peak allocated and reserved GPU
  memory were 5,517 and 6,114 MiB; minimum available host RAM was 2,279 MiB,
  only 7 MiB above the 2,272-MiB cache-plus-2-GiB-reserve admission threshold;
  minimum global free VRAM was 876 MiB.
- Measured dataset rates for this baseline were 14.596 GSM8K, 11.990 GSM-Plus,
  and 8.796 MATH rows/hour (11.295 overall). Applying those rates to the 1,209
  remaining GSM8K rows, 10,552 GSM-Plus rows, and 350 MATH rows gives a
  provisional 1,002.7 active-hour estimate. This is not the final ETA; the
  other batch profiles and stress test must establish the production setting.
- The GPU cooled to 56 C and NVIDIA thermal slowdown flags were inactive before
  the unfiltered matrix was restarted in the same qualification identity at
  07:49. The original first-profile logs were preserved; the matrix uses its own
  command, PID, stdout, and stderr files. At 07:52, `g1-s2` had 8/24 greedy
  outputs, no samples or failures, and 3,436 MiB minimum available host RAM.
  Greedy had adapted to batch one with offloaded cache. NVIDIA software thermal
  slowdown was active at 85 C under load, while hardware thermal and power
  slowdown flags were inactive. The full matrix remains incomplete.
- Progress check at 08:00 Asia/Singapore: `g1-s2` completed 24/24 greedy
  outputs and started sampling, with 3/192 samples saved and no failures. Its
  greedy stage used batch one with offloaded cache; the sample batch-two stage
  is underway. Minimum available host RAM remains 3,436 MiB; the Windows
  `Available MBytes` counter read 3,605 MiB. Minimum global free VRAM was
  878 MiB, current free VRAM 898 MiB, and GPU temperature 86 C. NVIDIA's
  software thermal slowdown flag was active; hardware thermal and power
  slowdown flags were inactive. The overall matrix is still incomplete.
- Progress check at 08:19 Asia/Singapore: `g1-s2` has 24/24 greedy answers
  and 41/192 samples saved, zero failures, and no failure-history entries.
  Minimum available RAM is 3,436 MiB; the Windows `Available MBytes` counter
  reads 3,622 MiB. Minimum free VRAM is 876 MiB and the current reading is
  898 MiB at 86 C. Sample checkpoints continue to be written; the batch-two
  profile has not yet finished or recorded its final actual sample settings.

- Progress check at 08:28 Asia/Singapore: `g1-s2` has 24/24 greedy answers and
  60/192 sampled answers, zero failures. Windows reports 3,627 MiB available
  RAM; the profile's minimum remains 3,436 MiB. GPU memory is 7,074/8,188 MiB
  used with 896 MiB free at 86 C and 68% utilization.
- Read-only comparison of the five qualification examples also present in the
  archived A100 output: greedy tokens/text, extracted answers, and correctness
  match 5/5. Score vectors align; maximum absolute entropy and log-probability
  differences are 0.02679 nats and 0.01228. All 40 deterministic sample seeds
  match, while sampled token/text sequences match 0/40 and extracted answers
  are equivalent 23/40. This is cross-run evidence, not a hardware-only causal
  comparison. Full matrix throughput, memory stress, and restart qualification
  remain open.
- Progress check at 08:30 Asia/Singapore: qualification launcher PID 30504 and
  worker PID 30360 are still live. `g1-s2` advanced to 64/192 samples with no
  failures; Windows available RAM is 3,607 MiB and GPU free memory is 898 MiB
  at 85 C. No second GPU workload was started.
- Progress check at 08:36 Asia/Singapore: the same launcher and worker remain
  live; `g1-s2` advanced to 70/192 samples with no failures. Available RAM is
  3,501 MiB and free GPU memory is 898 MiB at 87 C. The minimum recorded host
  RAM remains 3,436 MiB; the overall qualification is incomplete.
- Derived the repository verification profile from the current checkout:
  unittest discovery is documented in `README.md` and `docs/runbook.md`; the
  plan's full gate also calls for compileall, pip check, and diff check. No
  repository CI workflow, task runner, Compose file, or AGENTS.md was found.
  Full verification remains deferred until qualification finishes and the
  candidate is frozen. A read-only pass over the local coordinator, worker,
  stage store, and archive importer was started; no source edits were made.
- Progress check at 08:39 Asia/Singapore: `g1-s2` advanced to 76/192 samples
  with zero active failures. The worker PIDs remain live; Windows available RAM
  is 3,263 MiB and global GPU free memory is 898 MiB at 87 C.
- Read-only behavior audit confirmed the local coordinator checkpoints each
  greedy result and sample through `StageStore`, with atomic JSON replacement;
  complete rows append with `fsync`. The worker protocol streams per-request
  outputs, classifies CUDA OOM separately, and closes a failed worker so the
  next backoff attempt starts a fresh process. Sampling uses a seeded
  `torch.Generator` per request. These are source observations; runtime tests,
  the full candidate adversarial pass, and final-strict review remain pending.
- Progress check at 08:42 Asia/Singapore: `g1-s2` advanced to 82/192 samples
  with no failures; both worker processes are live. Available RAM is 3,476 MiB,
  free VRAM 898 MiB, and GPU temperature 86 C. The error log contains the
  Math-Verify timeout warning but no fatal error.
- The source audit traced the warning to `grading.py`: Math-Verify's timeout
  uses a multiprocessing path that is not working in this Windows runtime, so
  the implementation currently passes a zero timeout. This is not a benchmark
  failure and no grading stall has been observed, but it leaves a possible
  coordinator hang on a pathological generated answer. Added a launch gate to
  qualify bounded/recoverable grading behavior before the sustained run.
- Read-only inspection of the completed `g1-s1` record found 548.214 seconds
  in greedy score generation, 7,100.777 seconds in sampling, and 0.219 seconds
  in grading across 24 rows. No grading hang occurred in that cohort; these
  measurements do not bound a pathological parse.
- Progress check at 08:46 Asia/Singapore: qualification launcher PID 30504 and
  Python worker PID 30360 remain live. `g1-s2` has 24/24 greedy answers and
  90/192 samples, zero failures. Available RAM is 3,626 MiB; global free VRAM
  is 876 MiB at 86 C. The full matrix remains incomplete.
- Progress check at 08:48 Asia/Singapore: both qualification processes remain
  live. `g1-s2` advanced to 93/192 samples with zero failures; Windows has
  3,421 MiB available RAM and GPU free memory is 900 MiB at 85 C.
- Qualification-gate review found that the aggregator currently defines the
  12-profile matrix as complete when each configuration result is present,
  even if one has `status: incomplete`. Since minimum-profile OOMs can be
  recorded per request while the matrix proceeds, this state is reachable.
  The full-run CLI checks the selected recommendation but not all twelve
  statuses. Added a TDD fix-and-requalify gate; no Python source changed, so
  the live qualification remains bound to its recorded source snapshot.
- Progress and roadmap refresh at 08:57 Asia/Singapore: launcher PID 30504 and
  Python worker PID 30360 remain active; `g1-s2` now has 24/24 greedy answers
  and 110/192 samples with zero failures. Both stages recorded adaptations to
  batch one/offloaded cache. Its minimum available RAM and free VRAM are
  3,286 MiB and 876 MiB; the live system poll reports 3,716 MiB available RAM,
  896 MiB free VRAM, 86 C, and 85% GPU utilization with software thermal
  slowdown active. The matrix remains incomplete.
- Updated `docs/remaining-tasks-to-complete-study.md` as the ordered entry-point
  checklist from the active qualification through implementation fixes,
  final-strict review, branch push, the 12,111-row run, reports, and verified
  backups. Refreshed both roadmap snapshots and clarified that local Windows
  RTX 4060 execution uses native Transformers batching; vLLM/Ray are outside
  this local run. Documentation-only update; no inference source changed.
- Progress check at 09:00 Asia/Singapore: launcher PID 30504 and Python worker
  PID 30360 remain active. `g1-s2` advanced to 117/192 samples, with 24/24
  greedy answers and zero failures. Available RAM is 3,619 MiB; GPU free memory
  is 896 MiB at 85 C and 74% utilization. The qualification and checklist
  snapshots now reflect this checkpoint; no source code changed.
- Progress check at 09:11 Asia/Singapore: launcher PID 30504 and Python worker
  PID 30360 remain active. `g1-s2` advanced to 131/192 samples, with 24/24
  greedy answers and zero failures. Available RAM is 3,462 MiB; GPU free memory
  is 876 MiB at 86 C and 81% utilization. Both requested batch settings have
  adapted to batch one/offloaded cache; the full matrix remains incomplete.
- Progress check at 09:15 Asia/Singapore: launcher PID 30504 and Python worker
  PID 30360 remain active. `g1-s2` advanced to 135/192 samples with zero
  failures. The profile's minimum available RAM is 3,224 MiB; system RAM is
  3,175 MiB available and GPU free memory is 876 MiB at 85 C and 79%
  utilization. Matrix and memory-stress gates remain incomplete.
- Checkpoint failure at 09:29 Asia/Singapore: the qualification launcher and
  worker exited while saving `g1-s2/config.json`. The saved profile contains
  24/24 greedy outputs and 154/192 samples, with zero per-example failures, but
  records `PermissionError [WinError 5] Access is denied` while replacing the
  temporary config file. Both the profile and overall qualification manifest
  are marked `failed`; this attempt is not qualification evidence. Preserve
  this directory and diagnose atomic-write recovery before starting another GPU
  workload. The next implementation checks are a TDD regression for replace
  failure/restart recovery and a separate TDD fix requiring all twelve matrix
  profiles to be complete. Then rerun qualification with a fresh source
  snapshot and identity.
- Updated `docs/local-rtx4060-project-plan.md` and
  `docs/remaining-tasks-to-complete-study.md` with the stopped-run state,
  failure, partial counts, and ordered steps from checkpoint repair through the
  full study and delivery. No inference source changed in this documentation
  update; the previous thermal and RAM readings are labeled historical.
- Read-only checkpoint audit at 09:36 Asia/Singapore found 178 valid `g1-s2`
  stage files (24 greedy + 154 samples), each with the expected run signature
  and a unique request ID; no temporary config file remains. Source inspection
  indicates a same-code rerun will load those stages and schedule only missing
  requests with the same derived seeds. This resume behavior has not been
  executed. The atomic writer has no replace retry, so a transient Windows
  `PermissionError` terminates the invocation. Updated both roadmap documents
  to distinguish observed stage integrity from untested resume behavior.
- A second read-only integrity audit found three import/provenance blockers:
  the full-run path derives its local remainder from the archive's actual row
  count instead of requiring the specified 110 A100 + 12,111 local split; the
  importer validates an opened ZIP but reopens its path to compute the recorded
  hash, so replacement between those steps can mismatch imported bytes and
  preserved provenance; and A100 source labeling checks a top-level GPU name
  without verifying archived `identity.runtime` hardware. Current report
  completeness checks total selected IDs and counts but not the required
  execution-origin totals. Added TDD regression and implementation tasks for
  these findings to the remaining-work documents. No production files or tests
  were changed and no tests were run.
- Read-only inspection of the supplied ZIP confirmed 110/110 complete imported
  rows, all from GSM8K. The signed `identity.runtime` and top-level hardware
  metadata both identify an NVIDIA A100 80GB PCIe on Linux. The study's exact
  origin-by-dataset totals are A100: 110 GSM8K; RTX 4060: 1,209 GSM8K, 350 MATH,
  and 10,552 GSM-Plus. The import code still needs to enforce and report those
  totals before a full run can start.

## 2026-10-01 — Project completion roadmap and grading uncertainty

- Refreshed `docs/local-rtx4060-project-plan.md` and
  `docs/remaining-tasks-to-complete-study.md` with the ordered checklist from
  the current checkpoint through full-run recovery, reports, backups, final
  review, push, and delivery. The detailed plan remains the acceptance record;
  the remaining-tasks document is the short entry point.
- Clarified engine selection against current official vLLM documentation: its
  GPU guide requires Linux and says Windows is not natively supported (WSL is
  a separate option); the parallelism guide recommends a single GPU without
  distributed inference when the model fits. This approved local Windows run
  remains Transformers-based; vLLM/Ray are not introduced.
- Added TDD coverage for Math-Verify parse and equivalence timeouts flowing into
  self-consistency. RED assertions showed that unknown checks were not exposed
  and that an equivalence timeout could still produce a graded majority. Added
  tri-state grading checks, unknown counters, an incomplete-majority status,
  per-row serialization, and dataset/report Markdown summaries. Existing
  boolean helpers retain their compatibility behavior.
- Focused parent command
  `.venv\Scripts\python.exe -m unittest tests.test_grading tests.test_consistency tests.test_run tests.test_report -v`
  passed **40 tests**. The qualification-recovery lane passed 23 focused state
  and benchmark tests; archive/import/report fixes passed 25 focused tests.
- Parent then reran the integrated changed-code surface with
  `.venv\Scripts\python.exe -m unittest tests.test_state tests.test_local_benchmark tests.test_archive_import tests.test_local_run tests.test_grading tests.test_consistency tests.test_run tests.test_report -v`;
  **76 tests across eight suites passed**. This is focused integration
  verification, not repository-wide verification.
- No local GPU qualification was run in this step. The stopped `g1-s2` profile
  remains partial, the complete qualification matrix and 1,024-token stress
  check remain pending, and full candidate verification, final-strict review,
  branch push, and the sustained study are not complete.

## 2026-10-01 — Audit blockers added to completion roadmap

- Follow-up read-only audits found three local-path blockers: batched sampling
  draws before temperature/top-k transforms; resume does not compare all saved
  local/imported row contents with the validated source records; and reports do
  not bind the exact imported ID set and row-level archive provenance to the
  manifest. The prior 76 focused tests predate these findings and do not close
  them.
- The experimental full vLLM path has a separate report integration blocker:
  its rows/manifest omit origin and scope metadata required for complete full
  reports. A stop-reason field lookup also needs an API-shaped test before that
  backend is promoted. vLLM/Ray remain outside the approved local Transformers
  workflow.
- The report audit also identified recommendation eligibility without a
  minimum scorable-label coverage or uncertainty gate. The roadmap now requires
  this rule to be justified before a recommendation is issued, with per-dataset
  coverage reported.
- Updated both project roadmap documents with these findings and the ordered
  next actions. No source code or tests changed, no tests were run, and GPU
  qualification remains stopped pending the local-path TDD fixes.

## 2026-10-01 — Sampling and report-lineage fixes; completion checklist refreshed

- Updated `docs/remaining-tasks-to-complete-study.md` as the ordered checklist
  from the active source-integrity work through qualification, final-strict
  review, publication, the 12,111-row local run, report validation, and durable
  backup verification. Updated `docs/local-rtx4060-project-plan.md` to match
  the same live closure state.
- Sampling finding: the controlled 64-token test gave an assertion RED before
  implementation (raw batched draw selected token 56; transformed scalar
  reference selected token 19). After applying temperature/top-k before each
  seeded draw, the sampling suite passed all four tests.
- Report-lineage finding: added exact imported-ID and per-row provenance
  validation. Parent focused report verification passed all 17 tests, including
  cases that alter imported IDs or archive/signature fields while preserving
  aggregate origin counts.
- Parent combined command
  `.venv\Scripts\python.exe -m unittest tests.test_local_sampling tests.test_local_run tests.test_report -v`
- `TDD_REQUIRED: yes` for resume-record behavior. Worker REDs were assertion
  failures, not import/setup errors: the first run showed two changed-record
  cases were accepted; the expanded run showed four valid-shape local cases
  (greedy text, same-length score, sample text, and swapped indices with
  matching seeds) were accepted. Changed seed and imported content/provenance
  cases were already rejected by existing derivation and archive comparison,
  so their new checks are regression coverage, not claimed behavioral REDs.
- The parent added whole-record integrity receipts after a further test-first
  check showed valid edits to derived entropy confidence, correctness, and
  majority share were still accepted. RED: three `AssertionError: ValueError
  not raised` cases. GREEN: the focused resume mutation test passed after each
  completed local prediction was atomically bound to a checksum receipt carrying
  its run signature. Raw text, token-score vectors, sample order/content,
  derived answer spans, source identity, and imported archive rows are also
  checked against their source evidence.
- Final parent command
  `.venv\Scripts\python.exe -m unittest tests.test_local_sampling tests.test_local_run tests.test_report -v`
  passed **31 tests** (4 sampling, 10 local-run, 17 report). Only existing
  Matplotlib/Pyparsing deprecation warnings were emitted.
- Extracted child-runtime proofs from each worker's persisted rollout using
  `scripts/extract_child_runtime.py`; all three actual `turn_context` records
  verified as `gpt-6-luna/max`. Rollout SHA-256 values: resume
  `0f6346a2c9bfbd205b9f7e58abd5db6f8238ba38c031c716f3eb76847d2ddec9`,
  sampling `70b68c13ffcf7afa1ea02b2ebee76cc8ab5d96d405d98ee6c05742908e09ac35`,
  report lineage `781608855c5d7da10d98bb26170c82372fd70cc377ac1ad952e1cfdba8065d21`.
- No GPU workload was run. Qualification and the sustained local study remain
  stopped at this checkpoint; the fresh qualification matrix, complete
  candidate verification, recommendation-eligibility policy, final-strict
  review, branch push, and full run remain outstanding.

## 2026-10-01 — Recommendation coverage gate and completion roadmap refresh

- Updated `docs/local-rtx4060-project-plan.md` and
  `docs/remaining-tasks-to-complete-study.md` to the 11:08 SGT checkpoint and
  aligned their completion checklist with the latest report behavior.
- `TDD_REQUIRED: yes` for report eligibility behavior. Observable seam: a
  full-profile report must withhold a recommendation when source-reference
  coverage is below 95% in any dataset or when any source-scorable row lacks a
  known outcome; candidate scores must cover all eligible rows.
- RED evidence: two assertion-based tests showed that reports still issued a
  candidate when label coverage was 95% and when source-reference coverage was
  90%. GREEN evidence: `tests.test_report` passes 19 tests, including both new
  insufficient-coverage cases. The combined parent command
  `.venv\Scripts\python.exe -m unittest tests.test_local_sampling tests.test_local_run tests.test_report -v`
  passed 33 tests (4 sampling, 10 local-run, 19 report).
- The policy requires parsed source references for at least 95% of selected
  rows in every dataset, known binary outcomes for all source-scorable rows,
  and complete signal scores for those rows. The report exposes coverage and
  describes metric ranking without confidence intervals or a statistical
  superiority claim.
- Next work remains the whole-candidate audit and final source verification,
  followed by a fresh 12-profile qualification, 1,024-token memory stress and
  recovery check, final-strict review, branch push, archive import, and the
  resumable 12,111-row local run. No GPU job was started in this documentation
  update; measured qualification and final-study ETA remain pending.

## 2026-10-01 — Report row integrity and output-cap recovery

- `TDD_REQUIRED: yes` for report integrity. Observable seams: a changed imported
  prediction must fail comparison with the preserved archive row, and a changed
  local prediction must fail its run-signature-bound integrity receipt. The
  full-run mutation tests first produced assertion RED (`run_complete` remained
  true after changing imported correctness or local entropy confidence).
- GREEN: two compact tests call the production archive-binding and local-receipt
  helpers directly and pass. Parent also ran the valid full mixed-origin report
  integration case with 12,221 rows; it passed in 15.953 seconds. The report
  module has 23 tests; its complete focused suite remains pending for the
  frozen-candidate verification gate.
- A separate recovery audit found that saved generations could exceed the
  selected profile's `max_new_tokens` limit. `TDD_REQUIRED: yes`. Observable
  seam: resuming saved greedy/sample stages and completed local rows with more
  token IDs than the run cap. Assertion RED showed the stage validator and the
  completed-record validator accepted the over-limit data. The fix now checks
  greedy and sample stages in both local and hybrid runners and checks completed
  local records before reuse.
- Parent verification command
  `.venv\Scripts\python.exe -m unittest tests.test_local_worker tests.test_hybrid tests.test_hybrid_workers tests.test_local_run -v`
  passed **50 tests**. Parent also reran the two report-helper mutation tests;
  both passed. The valid full-report integration test passed separately.
- Updated the plan and ordered checklist to record these closures. No GPU job
  was started. Full report/repository verification, fresh GPU qualification,
  stress/restart checks, final-strict review, branch publication, A100 row
  import, and the 12,111-row local run remain outstanding.

## 2026-10-01 — Archive termination validation and project roadmap

- Refreshed `docs/local-rtx4060-project-plan.md` and
  `docs/remaining-tasks-to-complete-study.md` to the 12:28 SGT checkpoint. The
  checklist orders the remaining work through final-strict review, branch push,
  archive import, resumable local execution, final reporting, and verified
  backup. The RTX 4060 is idle; no qualification workload is running.
- `TDD_REQUIRED: yes` for imported-row termination integrity. Observable seam:
  the archive importer rejects a greedy generation missing `eos_generated`,
  one with contradictory EOS/truncation flags, or a non-text termination reason.
  RED commands and evidence:
  `.venv\Scripts\python.exe -m unittest tests.test_archive_import.ArchiveImportTests.test_import_rejects_greedy_output_missing_termination_metadata -v`
  failed because no `ValueError` was raised;
  `.venv\Scripts\python.exe -m unittest tests.test_archive_import.ArchiveImportTests.test_import_rejects_conflicting_greedy_termination_flags -v`
  failed for the same reason; and
  `.venv\Scripts\python.exe -m unittest tests.test_archive_import.ArchiveImportTests.test_import_rejects_non_text_greedy_termination_reason -v`
  raised an uncaught `TypeError` for an unhashable reason value.
- GREEN: the importer now requires all three termination fields, checks the
  EOS/truncation/reason relationship for greedy and sampled outputs, and
  classifies malformed reason types as validation errors. The focused command
  `.venv\Scripts\python.exe -m unittest tests.test_archive_import -v` passed
  all **9 tests**.
- This closes an import-time validation gap; full candidate verification,
  complete adversarial review, fresh 12-profile GPU qualification, 1,024-token
  stress/restart checks, final-strict review, branch publication, and the full
  local study remain outstanding.

## 2026-10-01 — Qualification stress/cache alignment

- The parent audit found that the 1,024-token stress check always used the
  standard KV cache, even when the fastest qualified profile required offloaded
  cache. This left the stress evidence disconnected from the production cache
  mode and could reject a viable measured profile or fail to describe what the
  full runner would use.
- `TDD_REQUIRED: yes`. Observable seams: the stress model receives the selected
  cache implementation; an all-offloaded qualified configuration selects
  offloaded stress, while a normal or mixed configuration uses standard cache.
  RED: `.venv\Scripts\python.exe -m unittest tests.test_local_benchmark.LocalBenchmarkMemoryTests.test_1024_token_stress_uses_the_selected_cache_implementation -v`
  raised `TypeError` because `run_1024_token_stress` did not accept the cache
  mode. The offloaded-profile policy assertion also failed because no stress
  cache selector existed.
- GREEN: the stress runner now uses the selected mode, checks the host-RAM
  reserve before offloaded stress, records the cache and tested configuration,
  and requires exactly 1,024 tokens. Qualification forwards the recommended
  profile's cache mode. The focused command
  `.venv\Scripts\python.exe -m unittest tests.test_local_benchmark.LocalBenchmarkMemoryTests.test_1024_token_stress_uses_the_selected_cache_implementation tests.test_local_benchmark.LocalBenchmarkMemoryTests.test_complete_offloaded_configuration_can_qualify_at_its_recorded_floor tests.test_local_benchmark.LocalBenchmarkQualificationResumeTests.test_matrix_with_persisted_incomplete_profile_is_not_recommended_or_stress_tested -v`
  passed **3 tests**.
- No GPU job was started. Full candidate verification, parent adversarial
  review, fresh qualification, final-strict review, branch publication, and the
  sustained local experiment remain pending.

## 2026-10-01 — Qualification fallback and completion roadmap

- The qualification audit found that the coordinator selected the fastest
  measured candidate before stress testing, but stopped if that candidate
  failed the 1,024-token memory gate. That could miss a slower candidate that
  passed the required headroom checks. It also needed to reuse persisted stress
  results when resuming qualification.
- `TDD_REQUIRED: yes`. Observable seam: qualification must test candidates in
  throughput order until the first stress-complete profile with at least 1 GiB
  free VRAM, checkpoint each new stress result, and reuse an existing result
  without rerunning it. The initial regressions failed because the
  `select_stress_candidate` behavior did not exist.
- GREEN: added the selector and connected it to the qualification coordinator.
  The coordinator now checkpoints results as each profile is stressed and only
  recommends a profile that passes the stress and memory gate. Verification:
  `.venv\Scripts\python.exe -m unittest tests.test_local_benchmark -v` passed
  **19 tests**, including fallback and resume reuse.
- Refreshed `docs/local-rtx4060-project-plan.md` and
  `docs/remaining-tasks-to-complete-study.md`, and added
  `docs/project-roadmap.md` with the current checkpoint and ordered path to
  completion. At 12:37 SGT `nvidia-smi` showed 52 C and 0 MiB in use; no
  qualification worker was active. This qualification-source change means the
  next GPU matrix must use a fresh source snapshot and qualification identity.
- The next work remains the full candidate audit and repository verification,
  fresh 12-profile qualification with long-output stress and restart/resume,
  final-strict review and push, 110-row archive import, 12,111-row resumable
  local run, final mixed-origin report, and verified durable backup. No
  production qualification or full-study run was started.

## 2026-10-01 — Legacy archive, local recovery, report integrity

- Validated the exact user-supplied A100 archive from a file-backed unittest
  process against the cached 12,221-row selection. The importer accepted all
  110 complete archived records, their 880 derived sample seeds, exact question
  and reference content, pinned revisions, and signed A100 runtime provenance.
  The raw prediction JSONL SHA-256 is
  `085d4e38213744311e509b064c3a3c2dd1f0fd7edf5883ebc5870309254e8298`. The
  historical schema lacks `sample_answer_parse_unknown_count`,
  `greedy_agreement_unknown_count`, and
  `majority_comparison_unknown_count`; these remain `null`, not zero. The
  original archive has not yet been imported into a local run directory.
- `TDD_REQUIRED: yes`. Grading/consistency observables: parser and verifier
  exceptions remain unknown, unresolved equivalence suppresses provisional
  majority metrics, and legacy counters are accepted as unavailable without
  altering archived fields or the raw checksum. The grading, consistency, and
  archive-import regressions produced assertion RED before implementation.
  GREEN: `.venv\Scripts\python.exe -m unittest tests.test_grading
  tests.test_consistency tests.test_archive_import -v` passed **30 tests**.
  The actual supplied archive additionally passed file-backed validation with
  **110 rows**.
- Report TDD observables: reject changing a full run's scope label to
  `synthetic`, reject model/tokenizer/dataset/row revision drift, keep legacy
  null counters unavailable, and allow only those three null placeholders when
  comparing an imported row with the preserved old-schema archive. Each
  regression showed an assertion RED. The report suite passed **27 tests**.
  Structurally complete 12,221-row runs may remain complete, but recommendation
  signals are withheld while required unknown metrics are unavailable.
- Local runtime TDD observables: classify insufficient host RAM separately
  from CUDA OOM; stop active generation when a persisted deadline expires;
  fail an exhausted host-RAM request while continuing later examples; and fail
  qualification if global free VRAM dips below the measured floor during
  generation. Assertion REDs were observed before the changes. A real
  subprocess crash/restart test confirms completed samples are checkpointed
  once and retries exclude those IDs. The focused worker/local-run/benchmark
  checks passed; the combined parent pass below is the integrated receipt.
- Parent integration command:
  `.venv\Scripts\python.exe -m unittest tests.test_archive_import
  tests.test_report tests.test_local_run tests.test_local_worker
  tests.test_local_sampling tests.test_hybrid tests.test_hybrid_workers
  tests.test_local_benchmark tests.test_grading tests.test_consistency
  tests.test_run tests.test_state tests.test_teacher -v`
  completed with **167 tests passed** in 135.381 seconds. Pyparsing/Matplotlib
  dependency deprecation warnings appeared; there were no test failures.
- At 13:27 SGT, `nvidia-smi` reported 57 C and 0 MiB used on the RTX 4060; no
  Python worker remained. No qualification or sustained study was launched.
  Prior qualification outputs are historical because inference-source changes
  require a fresh source snapshot and qualification identity.
- Assurance state was rechecked in the durable sidecar. The active
  `Teacher-reliability-Distill-LLM/inference-optimization/vllm-hybrid` unit
  remains open with 0/3 calls used and includes this local scope revision. The
  separate `apply-1-teacher-reliability` unit remains terminal `ship` for an
  earlier candidate. Neither history or budget was changed.
- Remaining: parent audit of every tracked and untracked file; full repository
  verification; current-candidate adversarial pass; fresh 12-profile RTX 4060
  qualification, 1,024-token stress, and resume measurement; final-strict
  review through the existing optimization unit; authorized branch push;
  one-time import; 12,111-row local execution; final combined/per-origin
  report; verified backup; and project retrospective.

## 2026-10-01 — Project roadmap refresh and cache provenance audit

- Refreshed `docs/local-rtx4060-project-plan.md`,
  `docs/remaining-tasks-to-complete-study.md`, and
  `docs/project-roadmap.md` with the ordered remaining tasks through candidate
  review, GPU qualification, archive import, the resumable full study, reports,
  and verified artifact backup. Corrected the runtime note: the owner reported
  switching this chat to GPT-6 Sol, but exact persisted current-turn evidence
  must be checked before recording a parent-runtime pass or preparing the
  final-strict packet.
- At 13:38 SGT, `nvidia-smi` showed the RTX 4060 at 50 C with 0 MiB used; no
  Python qualification or study worker was active. No qualification or full
  run was started.
- Parent audit identified a manifest-telemetry risk: a single
  `cache_implementation` field can misstate which inference stage used
  offloaded cache when greedy and sampling adapt independently. A TDD follow-up
  observed RED on stage-specific manifest assertions, then GREEN on the focused
  local-run and host-memory recovery checks. Parent review also found a separate
  resume-validation edge case for a null majority-vote count after unknown
  answer comparisons; its regression and fix are still in progress.
- A separate read-only inference audit checked the pinned sampler configuration:
  the selected model commit's generation config leaves `top_k` and `top_p` at
  Transformers defaults, which the pinned Transformers 4.52.4 environment
  resolves to 50 and 1.0; `generate_sample` explicitly supplies temperature
  0.7. The optimized local sampler pins all three values directly. No
  production sampling mismatch was found. The audit also identified a
  conditional device-mapping risk if `CUDA_VISIBLE_DEVICES` remaps multiple
  NVIDIA GPUs, because `nvidia-smi --id=0` and Torch logical device 0 may then
  differ. This host has one NVIDIA adapter and the variable is unset; verify
  that precondition before qualification.
- Next: finish the tracked/untracked candidate audit, run one candidate-wide
  verification pass, then do fresh GPU qualification and final-strict
  readiness. The 12,221-row study has not started.

## 2026-10-01 — Resume validation closure and integrated rerun

- `TDD_REQUIRED: yes`. Observable seam: a locally generated, complete record
  may resume when answer-equivalence checks are unknown and its majority-vote
  count and provisional metrics are intentionally unavailable; invalid count
  bounds and inconsistent unknown-result states must still be rejected.
- RED: before the validator change, the resume regression failed because
  `majority_vote_count: null` was treated as an invalid count even when the
  unknown-comparison count was positive. Invalid-count mutations also showed
  that the earlier validator accepted an over-limit count and inconsistent
  provisional state. These were behavioral assertion failures.
- GREEN: validation now accepts only the internally consistent unknown
  comparison shape, retains strict type/bound checks for all other counts, and
  rejects contradictory provisional majority fields. A test follow-up fixes
  the baseline fixture to a literal valid complete-majority result, removing
  dependence on Math-Verify subprocess outcomes. Parent verification:
  `.venv\Scripts\python.exe -m unittest tests.test_local_run -v` passed **22
  tests**. The worker repeated this module three times, each with 22 passes.
- Integrated focused verification:
  `.venv\Scripts\python.exe -m unittest tests.test_archive_import
  tests.test_report tests.test_local_run tests.test_local_worker
  tests.test_local_sampling tests.test_hybrid tests.test_hybrid_workers
  tests.test_local_benchmark tests.test_grading tests.test_consistency
  tests.test_run tests.test_state tests.test_teacher -v` passed **171 tests**
  in 135.005 seconds. Dependency deprecation warnings were emitted; no tests
  failed. This remains a focused affected-module pass, not repository-wide
  discovery, and the complete source audit remains open.
- At 13:59 Asia/Singapore, `nvidia-smi` reported RTX 4060 at 50 C and 0 MiB
  used; `CUDA_VISIBLE_DEVICES` was unset and no Python qualification or study
  process was active. No GPU work was started.
- Current parent runtime proof was read from the persisted root rollout:
  `gpt-6-luna/max`, turn `01a0f60b-653a-7a92-babe-f5979b60f698`, ordinal
  `30016`, record-line SHA-256
  `6640bd7712f5f936c0c1d71f419bc6e608c4166afa75aa9a0d0f138ff74a4a1c`.
  This satisfies the user's requested Luna parent for this continuation; the
  required fresh final-strict reviewer remains GPT-6-Sol/max and needs its own
  exact runtime proof before any reservation.
- Next: complete the tracked/untracked candidate audit, run repository-wide
  discovery plus `compileall`, `pip check`, and `git diff --check` when the
  behavior candidate is ready; then produce a fresh RTX 4060 qualification
  source snapshot and measured matrix/stress/restart evidence. The local full
  study has not started.

## 2026-10-01 14:34 SGT — Project plan and remaining-work checklist refreshed

- `TDD_REQUIRED: no` — documentation-only status and plan update; no production
  code or test behavior changed.
- Refreshed `docs/local-rtx4060-project-plan.md`,
  `docs/project-roadmap.md`, and
  `docs/remaining-tasks-to-complete-study.md` with the current ordered path:
  complete candidate audit and repository-wide checks, run a fresh local GPU
  qualification, pass final-strict review, publish the implementation, import
  the archive, complete the remaining 12,111 local rows, validate reports, and
  verify recoverable backups.
- Current checkpoint: the RTX 4060 reported 8,188 MiB total, 0 MiB used, 50 C,
  and 0% utilization at 14:31 SGT. No Python qualification/full-run worker was
  present when checked at 14:33. The prior qualification is incomplete; the
  full study has not started. The latest affected-module integration result is
  175 passed; repository-wide checks and a fresh qualification remain pending.
- Provenance terminology correction: earlier log entries called the archive's
  `run_signature` "signed." It is an unkeyed SHA-256 self-hash. It checks the
  archive's internal integrity but does not independently authenticate the
  physical GPU. Current plan documents describe the archived A100 details as
  runtime metadata reported by the archive, not hardware attestation.

## 2026-10-01 — Full-report unknown-majority and timeout-stage fixes

- `TDD_REQUIRED: yes`. Observable report seam: a full 12,221-row report remains
  complete when a source row has unresolved pairwise answer comparisons and
  correctly stores null majority metrics; contradictory majority states and
  counts outside the eight-sample limits remain malformed.
- RED: the full mixed-origin report regression failed with
  `run_complete == false` and issue `self_consistency_counts_invalid` for an
  otherwise-valid local row with `majority_comparison_unknown_count == 1` and
  null majority answer/count/share/correct fields. Separate assertion REDs
  showed the old row validator accepted non-null majority metrics despite an
  unknown comparison and accepted a parseable-answer count of nine.
- GREEN: `_full_row_issues` now recognizes the consistent unknown-comparison
  state, enforces related null fields and status, applies sample and pairwise
  count bounds, and retains legacy-null counter availability behavior. The
  full report module passed **30 tests**; focused report/local-resume behavior
  passed **4 tests**, including the new 12,221-row integration case.
- A second TDD issue was found in the Linux grader path. Observable seam:
  timeout classification follows the last reported grading stage when the
  external timeout exception carries only the broader `grade` operation.
  RED: an injected verifier timeout with latest progress `stage=verify`
  returned `parse_timeout`, not `verifier_timeout`. GREEN: `grade_prediction`
  now classifies from the latest nonempty progress stage. The new regression
  and existing real timeout/restart tests passed; `tests.test_grading
  tests.test_consistency` passed **21 tests**.
- Integrated verification after both source changes:
  `.venv\Scripts\python.exe -m unittest tests.test_archive_import
  tests.test_report tests.test_local_run tests.test_local_worker
  tests.test_local_sampling tests.test_hybrid tests.test_hybrid_workers
  tests.test_local_benchmark tests.test_grading tests.test_consistency
  tests.test_run tests.test_state tests.test_teacher -v` passed **175 tests**
  in 150.437 seconds. Repository-wide discovery has not yet run.
- Read-only audits of report/import, inference/worker, grader, and CLI/setup
  paths are complete. They found no other current Windows local-run blocker.
  Remaining conditions: verify physical/logical GPU correspondence before
  qualification; vLLM is experimental/unqualified on this host, and direct
  vLLM worker invocation has higher defaults than the supported coordinator
  invocation. These do not enter the approved native Windows full run.
- At 14:25 Asia/Singapore, the RTX 4060 was 53 C and 0 MiB used; no GPU worker
  or full-study process was active.
- Next: finish the parent adversarial pass and exact candidate diff audit,
  then run repository-wide unittest discovery, compileall, pip check, and
  whitespace checks. After those gates, run the fresh 24-question local matrix,
  1,024-token stress, and actual worker restart/resume qualification before
  preparing the existing final-strict unit.

## 2026-10-01 — Resume validation closure and integrated rerun

- `TDD_REQUIRED: yes`. Observable seam: a locally generated, complete record
  may resume when answer-equivalence checks are unknown and its majority-vote
  count and provisional metrics are intentionally unavailable; invalid count
  bounds and inconsistent unknown-result states must still be rejected.
- RED: before the validator change, the resume regression failed because
  `majority_vote_count: null` was treated as an invalid count even when the
  unknown-comparison count was positive. Invalid-count mutations also showed
  that the earlier validator accepted an over-limit count and inconsistent
  provisional state. These were behavioral assertion failures.
- GREEN: validation now accepts only the internally consistent unknown
  comparison shape, retains strict type/bound checks for all other counts, and
  rejects contradictory provisional majority fields. A test follow-up fixes
  the baseline fixture to a literal valid complete-majority result, removing
  dependence on Math-Verify subprocess outcomes. Parent verification:
  `.venv\Scripts\python.exe -m unittest tests.test_local_run -v` passed **22
  tests**. The worker repeated this module three times, each with 22 passes.
- Integrated focused verification:
  `.venv\Scripts\python.exe -m unittest tests.test_archive_import
  tests.test_report tests.test_local_run tests.test_local_worker
  tests.test_local_sampling tests.test_hybrid tests.test_hybrid_workers
  tests.test_local_benchmark tests.test_grading tests.test_consistency
  tests.test_run tests.test_state tests.test_teacher -v` passed **171 tests**
  in 135.005 seconds. Dependency deprecation warnings were emitted; no tests
  failed. This is still a focused affected-module pass, not repository-wide
  discovery, and the complete source audit remains open.
- At 13:59 Asia/Singapore, `nvidia-smi` reported RTX 4060 at 50 C and 0 MiB
  used; `CUDA_VISIBLE_DEVICES` was unset and no Python qualification or study
  process was active. No GPU work was started.
- Current parent runtime proof was read from the persisted root rollout:
  `gpt-6-luna/max`, turn `01a0f60b-653a-7a92-babe-f5979b60f698`, ordinal
  `30016`, record-line SHA-256
  `6640bd7712f5f936c0c1d71f419bc6e608c4166afa75aa9a0d0f138ff74a4a1c`.
  This satisfies the user's requested Luna parent for this continuation; the
  required fresh final-strict reviewer remains GPT-6-Sol/max and must have its
  own exact runtime proof before any reservation.
- Next: complete the tracked/untracked candidate audit, run repository-wide
  discovery plus `compileall`, `pip check`, and `git diff --check` when the
  behavior candidate is ready; then produce a fresh RTX 4060 qualification
  source snapshot and measured matrix/stress/restart evidence. The local full
  study has not started.

## 2026-10-01 15:08 SGT — Report lineage and qualification lock audit

- `TDD_REQUIRED: yes` for both report-lineage and concurrent-qualification
  behavior. Report seams: a mixed full-study report rejects source-A100 GPU
  metadata changed across all local claims when the preserved archive remains
  unchanged; it also rejects local RTX metadata that conflicts with the
  run-signature-bound local hardware identity.
- Report RED: the A100-metadata mutation and local-GPU-metadata mutation each
  left `run_complete: true` before the production checks. GREEN: preserved ZIP
  identity/runtime/GPU metadata is now checked against its self-hash and row
  lineage; local runtime, hardware, run signature, and execution origin must
  agree. `.venv\Scripts\python.exe -m unittest tests.test_report -v` passed
  **32 tests** in 150.347 seconds.
- Qualification-lock seam: while one process holds the qualification output
  directory lock, a second call must raise `RunAlreadyActive` before touching
  qualification state. The test was written first. RED with the decorator
  temporarily disabled reached the controlled `QualificationReached` sentinel
  instead of refusing the write. GREEN after the decorator acquired the shared
  `run_lock`; its focused test passed. The final
  `.venv\Scripts\python.exe -m unittest tests.test_local_benchmark -v` run
  passed **21 tests** in 10.369 seconds.
- `run_qualification` now uses the same exclusive lock file as full runs, so
  same-directory qualification processes cannot race on profile, stress, and
  top-level manifests. The check runs before CUDA or model work begins.
- Current vLLM decision was checked against the official installation and
  scaling guides: vLLM requires Linux and is not native to Windows (WSL is a
  separate route); its single-GPU guidance does not recommend distributed
  inference when the model fits one card. The approved local path therefore
  remains one native-Windows Transformers worker with measured batches. The
  experimental vLLM route remains separate and unqualified; Ray does not add
  GPU capacity to this one-card setup.
- At 15:02 SGT, `nvidia-smi` reported RTX 4060 8,188 MiB total, 0 MiB used,
  55 C, and 0% utilization. No qualification or full-study worker was active.
  The archived partial qualification is unchanged; no A100 ZIP import or
  sustained local study has started.
- Candidate-wide unittest discovery, `compileall`, `pip check`, full diff
  inspection, fresh GPU qualification, final-strict readiness, review, and
  push remain pending. The existing vLLM-hybrid assurance unit remains open
  with 0/3 calls used and no active reviewer reservation.

## 2026-10-01 15:18 SGT — Lock qualification before GPU model loading

- `TDD_REQUIRED: yes`. Observable seam: when another process owns the target
  qualification directory, the CLI exits with the lock error before setting
  the CUDA allocator or calling `load_teacher`.
- RED: with a valid patched archive/cache path and a held run lock, the new CLI
  test failed because `load_teacher` was called once before `run_qualification`
  attempted its lock.
- GREEN: `main` now takes the output-directory lock before loading archive/cache
  dependencies or touching CUDA. A private CLI helper runs the already-locked
  qualification body; the public `run_qualification` entry point still locks
  direct callers. No nested lock is acquired.
- Verification: `tests.test_local_benchmark` passed **22 tests** in 18.832
  seconds, including both direct-call and pre-model-load lock cases;
  `tests.test_local_cli` passed **7 tests**. The CLI concurrency case fails by
  assertion if lock acquisition is removed and passes with the lock present.
- `nvidia-smi` at 15:18 SGT reported RTX 4060 8,188 MiB total, 0 MiB used,
  51 C, 0% utilization. No qualification or full-run process is active.
- Full candidate verification remains pending. The source audit must inspect
  the complete tracked/untracked set before one repository-wide test pass,
  compile/dependency checks, frozen-candidate manifest, adversarial review,
  and final-strict reviewer reservation.

## 2026-10-01 15:26 SGT — Repository-wide verification pass

- Parent candidate audit covered every tracked and untracked source, test,
  dependency, and documentation path. The selected native-Windows local route
  has no additional implementation blocker in that audit. The optional vLLM
  full-study route remains explicitly experimental because report-origin/scope
  fields and one worker stop-reason contract are incomplete.
- Full verification command:
  `.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v`
  passed **251 tests** in 182.483 seconds. This includes report provenance,
  archive import, local sampling, OOM/restart/resume, CLI locking, experimental
  worker protocol tests, and the latest full-report integration.
- `.venv\Scripts\python.exe -m compileall -q teacher_reliability scripts`
  passed. `.venv\Scripts\python.exe -m pip check` reported no broken
  requirements. `git diff --check` reported no whitespace errors; Git emitted
  its existing warning that `.gitignore` has LF and will be normalized to
  CRLF when Git next writes it.
- The plan, roadmap, and remaining-task checklist now show the 251-test result,
  compile/dependency status, and the exact next phase. Subsequent changes in
  this checkpoint are documentation-only; repeat final diff check and rerun
  the full gate if behavior code changes.
- No GPU qualification was launched during this pass. At 15:18 SGT,
  `nvidia-smi` showed the RTX 4060 at 8,188 MiB total, 0 MiB used, 51 C, and
  0% utilization; no qualification/full-run process was present. The prior
  GPU qualification remains incomplete, the supplied ZIP remains unimported,
  and the sustained local study has not started.
- Next: rehash/revalidate the supplied ZIP, confirm the current source snapshot,
  then run the fresh 12-profile RTX 4060 qualification, 1,024-token stress,
  and actual worker restart/resume check. Final adversarial readiness, the
  active review unit's packet validation, final-strict review, push, archive
  import, full run, reports, and backups remain pending.

## 2026-10-01 15:39 SGT — Fresh RTX 4060 qualification started

- Started the fresh qualification from source snapshot
  `5e1c513218c8079af04905fa5e43259b85f4c37ef82ee6cf26c95eefa295432e` with
  identity `rtx4060-20261001T073411Z-5e1c5132`. Output is under
  `%LOCALAPPDATA%\TeacherReliability\qualifications\` and is protected by the
  run lock. The supplied A100 archive hash is
  `7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82`.
- At 15:39:35 SGT the qualification process was still running and the manifest
  status was `running`. The `g1-s1` output directory existed, but the manifest
  had not recorded a completed profile result. This is startup/progress
  evidence only; no new qualification result is accepted yet.
- The last observed GPU sample was 7,072/8,188 MiB used, 86 C, and 56%
  utilization. Continue tracking temperature, headroom, and qualification
  progress. Preserve this output and inspect its manifest/stages before any
  recovery; do not combine older partial qualification stages with this source
  snapshot.
- The old batch-one measurement and 1,002.7-hour ETA remain provisional. Still
  pending: all 12 profile combinations, the 1,024-token stress check, actual
  worker restart/resume evidence, final candidate adversarial/readiness gates,
  the fresh final-strict review, branch push, A100 archive import, sustained
  12,111-row run, final reports, verified backup, and retrospective.

## 2026-10-01 15:48 SGT — Qualification checkpoint and memory adaptation

- Rechecked the live qualification process and its persisted `g1-s1` checkpoint.
  At 15:47:55, the profile and top-level manifest were still `running`; it had
  saved **24/24 greedy outputs and 13/192 sample outputs**, with zero recorded
  failures. The process remained alive and the profile stage directory was
  changing.
- The profile adapted to greedy batch 1 and sampling batch 1, both with
  offloaded KV cache. Two adaptations were recorded after global free VRAM fell
  below the normal 1-GiB target. Persisted minima/peaks were 878 MiB free VRAM,
  2,882 MiB available system RAM, and 5,341/6,112 MiB allocated/reserved VRAM.
- The last `nvidia-smi` sample was 7,072/8,188 MiB GPU memory used, 86 C, and
  51% utilization. `nvidia-smi -q` reported software thermal slowdown active;
  hardware thermal slowdown was not active. Preserve this as a time-stamped
  measurement and continue monitoring. No profile is complete yet, so no
  throughput recommendation or ETA is accepted.
- This is monitoring/documentation only (`TDD_REQUIRED: no`): no inference
  code, test behavior, qualification process, or checkpoint was changed. The
  previous 251-test candidate-wide result still predates these documentation
  updates; no tests were run in this checkpoint.

## 2026-10-01 15:53 SGT — Verified qualification progress

- A verified wait on the same qualification found the worker alive and the
  `g1-s1` checkpoint advancing: 24/24 greedy outputs, 26/192 sample outputs,
  zero persisted failures. Profile and matrix manifest remain `running`.
- At 15:53:07, the latest GPU reading was 7,070 MiB used, 86 C, and 62%
  utilization. Persisted minimum free VRAM was 878 MiB and available host RAM
  reached a 2,426 MiB minimum. Software thermal slowdown remained active;
  hardware thermal slowdown was inactive. These measurements are specific to
  the running qualification and do not establish a final safe production rate.
- Recomputed the exact inference source snapshot; it still matches the
  qualification identity at
  `5e1c513218c8079af04905fa5e43259b85f4c37ef82ee6cf26c95eefa295432e`.
  Source files, runtime, and checkpoint were left untouched. This is a
  documentation/monitoring checkpoint only; no tests were run.

## 2026-10-01 16:02 SGT — Qualification halted incomplete on host RAM

- The fresh qualification process reached terminal `incomplete` at 16:01:19.
  Its persisted matrix has all 12 configurations marked incomplete, all with
  `InsufficientHostMemory` failures. Summary: no recommended configuration,
  stress test pending, and no dataset-weighted ETA.
- The first profiles retained useful stages: `g1-s1` has 24/24 greedy and
  40/192 samples with 19 sample failures; `g1-s2` has 3 greedy, 1 sample, and
  24 failures; `g4-s2` has 2 greedy and 24 failures. Other profiles retained
  no generated outputs and each recorded 24 failures. The same qualification
  identity and source snapshot remain available for resuming.
- Failure messages show the offloaded-cache check required
  2,382,364,672 bytes including the 2-GiB reserve, while available system RAM
  during inference fell to about 2.1–2.3 GiB. After the Python worker exited,
  Windows reported 4,624 MiB free; the GPU was idle. This confirms memory
  pressure specifically while the inference process is loaded. Keep the
  approved reserve and experiment settings; resume the same identity only
  after available RAM stays above the requirement during model loading and
  generation.
- `TDD_REQUIRED: no` — no production behavior changed. The run and partial
  artifacts were preserved; no tests were run. The full 251-test result remains
  candidate evidence from 15:26, not qualification or final-review evidence.

## 2026-10-01 16:10 SGT — Completion checklist and archive validation gate

- Updated the local RTX 4060 completion checklist to put the immediate RAM
  recovery and A100 archive-validation tasks first. The current qualification
  remains incomplete and must resume under its existing identity only after
  the available-RAM requirement can be maintained with the model loaded.
- A read-only diagnostic script was run from standard input to inspect cached
  dataset/import memory. Windows multiprocessing emitted spawn errors because
  child processes cannot reload `<stdin>`. The importer call nevertheless
  reached the supplied archive and rejected
  `gsm8k:main:test:0` because the archived
  `self_consistency.sample_answer_parseable_count` did not match (or was absent
  from) the value derived from its samples. This conflicts with earlier
  file-backed checks, so archive validation is not accepted until the field is
  reconciled and the current importer succeeds. No ZIP, run output, or imported
  row was written.
- Preserve the three absent legacy counters as unknown (`null`), and preserve
  the original ZIP. If evidence shows an importer behavior defect, add a
  behavior-focused TDD regression and fix it before import.
- `TDD_REQUIRED: no` for this documentation-only checkpoint; no production
  code changed and no tests were run. The prior 251-test result remains
  candidate evidence only.

## 2026-10-01 16:16 SGT — File-backed archive validation passed

- Repeated the exact A100 archive import validation with a temporary,
  file-backed Windows Python script guarded by `if __name__ == "__main__"` so
  Math-Verify child workers could spawn correctly. The current importer
  accepted **110/110** rows, with eight samples per row, expected selected
  content/revisions/seeds, and the archived runtime provenance.
- Archive SHA-256:
  `7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82`.
  Prediction JSONL SHA-256:
  `085d4e38213744311e509b064c3a3c2dd1f0fd7edf5883ebc5870309254e8298`.
- The earlier standard-input diagnostic emitted multiprocessing spawn errors;
  its derived-counter mismatch did not reproduce and is treated as an invalid
  diagnostic. The ZIP was read-only, no rows were imported, and the temporary
  script was removed. The three absent legacy counters remain `null`.
- `TDD_REQUIRED: no` — no production code changed and no tests were run. This
  closes archive validation only; creating the mixed-origin local run remains
  pending the accepted candidate and qualification gates.

## 2026-10-01 16:19 SGT — Qualification remains stopped pending loaded-RAM headroom

- Rechecked the actual qualification process and local device after the
  file-backed archive validation. No qualification process is running. At
  16:18:57, Windows reported **4,070 MiB** free RAM; `nvidia-smi` showed the
  RTX 4060 at **0 MiB / 8,188 MiB** and **0% utilization**.
- This idle reading does not establish the approved offloaded-cache admission
  requirement while the model is resident: at least 2,382,364,672 bytes free,
  including the 2 GiB reserve, with a 2.8 GiB operating target. Do not resume
  the saved qualification yet. Preserve its identity and partial outputs.
- `TDD_REQUIRED: no` — status/documentation only; no process was started, no
  experiment artifacts changed, and no tests were run.

## 2026-10-01 16:29 SGT — Refreshed completion roadmap from current checkpoint

- Updated `docs/project-roadmap.md` with the approved 12,221-row scope, current
  qualification and archive status, ordered tasks through reporting/backup,
  and the native-Windows inference-engine decision. For this RTX 4060 run,
  Transformers batching is the selected local route; vLLM/WSL would need a
  separate setup and qualification, and Ray is not required for one GPU.
- Added a README link to the roadmap so the ordered to-do list is discoverable
  from the repository front page.
- Refreshed the status snapshots in the detailed plan and checklist. The
  latest Windows reading was 3,284 MiB free RAM while idle; GPU use was 0 / 8,188
  MiB. This is not evidence of adequate headroom with the model loaded. The
  qualification remains incomplete and must resume under the same preserved
  identity after its memory gate is met.
- Recorded the memory audit's estimate that prompt-token lists retain about
  52 MiB after length calculation; this can help borderline admission but
  cannot close the 621–838 MiB gap to the 2.8 GiB loaded operating target.
- Recorded the latest focused report/run check (44 tests passed) separately
  from the earlier 251-test repository run. Candidate-wide checks remain
  pending because the earlier full run predates current candidate changes.
- `TDD_REQUIRED: no` — documentation-only update. No source code, tests,
  qualification state, archive, or run checkpoint was changed; no tests were
  run in this documentation checkpoint.

## 2026-10-01 16:33–16:40 SGT — Qualification token-memory cleanup (TDD)

- `TDD_REQUIRED: yes`. Observable seam: by the time qualification passes prompt
  lengths to cohort selection, it retains the exact per-prompt counts and does
  not retain any full token-ID sequences. Expected production change: count
  each prompt immediately instead of holding token IDs for the entire selected
  dataset. This preserves the prompt counts, cohort, selection signature, and
  experiment settings while reducing peak host memory.
- RED command:
  `.venv\Scripts\python.exe -m unittest tests.test_local_benchmark.LocalBenchmarkMemoryTests.test_qualification_releases_prompt_token_ids_before_selection -v`.
  The test failed at the intended assertion: both weak-referenced token-ID
  objects remained alive at cohort selection. GREEN changed qualification to
  compute each prompt length immediately instead of storing all token-ID lists.
  The targeted regression passed, then
  `.venv\Scripts\python.exe -m unittest tests.test_local_benchmark -v`
  passed **23 tests** in 36.957 seconds. REFACTOR review found no extraction or
  helper warranted; the direct comprehension is the minimal implementation.
- The current inference source snapshot is
  `530296039533dce7142c7b9327c3a7c3044706c404aaed751f0b8944e83c5ec2`.
  The earlier qualification used source snapshot
  `5e1c513218c8079af04905fa5e43259b85f4c37ef82ee6cf26c95eefa295432e`; preserve
  that incomplete run as historical evidence and use a fresh identity after
  the candidate is frozen. No GPU qualification or loaded-memory measurement
  was run; the estimated 52 MiB improvement does not prove the 2.8 GiB loaded
  RAM target.
- Current parent `turn_context` is `gpt-6-luna/max`, turn
  `01a0f68f-b00a-7282-8b3b-98affb5e62a9`, ordinal 20, in
  `C:\Users\user\.codex\sessions\2026\10\01\rollout-2026-10-01T16-23-26-01a0f68f-af37-70b1-bb70-a871778f9e9c.jsonl`.
  SHA-256 of the exact JSON record line (without the line delimiter):
  `caaa5c2379bae624a0b316a44415f13da5ea07bd22e418be161d7ab4c27364c8`.
  The fresh final-strict reviewer still must be `gpt-6-sol/max` and verified
  from its own runtime context.

## 2026-10-01 16:46 SGT — Verified no safe qualification headroom yet

- A fresh Windows check at 16:46 found **2,589 MiB free RAM while idle**;
  `nvidia-smi` showed the RTX 4060 at **0 / 8,188 MiB** and 0% utilization.
  No qualification process was running. Even idle RAM is below the 2.8 GiB
  operating target; no model load or inference was started.
- The qualification prompt-token cleanup remains green:
  `tests.test_local_benchmark` passed 23 tests in 36.957 seconds, and the
  modified source and test compiled. `git diff --check` exited 0 with the
  existing `.gitignore` line-ending normalization warning. The estimated
  52 MiB cleanup does not itself meet the memory gate.
- `TDD_REQUIRED: yes` for the completed source change; RED observed token-ID
  sequences retained at cohort selection, GREEN released them immediately
  after counting, and REFACTOR review found no further extraction needed.
  The prior incomplete qualification has a different source snapshot and is
  preserved as historical evidence; a fresh identity is required from the
  frozen source.

## 2026-10-01 16:48 SGT — Host RAM below qualification admission floor

- Rechecked live Windows state: **2,062 MiB RAM free while idle**, below the
  **2,272 MiB** offloaded-cache admission floor. `nvidia-smi` showed the RTX
  4060 at 0 / 8,188 MiB and 0% utilization; no qualification process was
  running. No model load or inference was started.
- Continue candidate review and software checks offline. Wait to start the
  qualification preflight until available RAM clears the admission floor; the
  operational target remains 2.8 GiB while the model is loaded.
- `TDD_REQUIRED: no` — this is read-only monitoring and documentation only.
  No source, tests, qualification artifacts, or assurance counters changed.

## 2026-10-01 16:49 SGT — TDD plan: release validated archive from qualification memory

- `TDD_REQUIRED: yes`. Observable seam: the validated archive result and its
  temporary snapshot are closed and collectible before `load_teacher` begins,
  while the archive hash, imported ID set, and remaining dataset counts are
  retained for the qualification identity. This avoids holding 110 imported
  rows plus an open snapshot through GPU model loading and the full matrix.
- Planned regression: call the real qualification CLI with a controlled
  importer result whose `close()` records its effect and whose instance is
  observed via weak reference. At the `load_teacher` boundary, assert it was
  closed and collected. The current implementation is expected to fail those
  assertions. RED command:
  `.venv\Scripts\python.exe -m unittest tests.test_local_benchmark.LocalBenchmarkQualificationResumeTests.test_cli_closes_imported_archive_before_loading_teacher -v`.
  No production code has changed for this second TDD checkpoint.

## 2026-10-01 17:00 SGT — Archive lifecycle regression and fast-run direction

- `TDD_REQUIRED: yes`. Observable seam: the validated archive result and its
  temporary snapshot are closed and collectible before `load_teacher`, while
  the archive hash, imported IDs, and per-dataset remaining counts stay
  available to qualification. RED first observed `snapshot_closed: false` and
  `imported_result_alive: true`. GREEN copied only those required scalar/set
  values, closed the archive in `finally`, and deleted the result before model
  loading. The first suite-order rerun exposed a test-fixture module-cache
  mismatch, not a production failure; the fixture now binds the patched
  importer module in `sys.modules`. The paired CLI tests then passed, and
  `tests.test_local_benchmark` passed **24 tests** in 13.871 seconds.
  REFACTOR review found no further extraction needed.
- Current inference source snapshot:
  `216f4bda7e44cc0161978f83c8c76b401b447515f16f5bc97dc8ec6b8fb99813`.
  This is still an unfrozen candidate; the historical incomplete qualification
  must not be resumed under this source.
- Owner direction is to prioritize starting the complete 12,221-row study and
  skip long checklist items that only compare performance profiles. The
  roadmap now replaces the 12-profile/24-row throughput matrix with one short
  three-dataset smoke on the conservative `greedy=1`, `sampling=2` profile,
  including a long prompt and 1,024-token limit. Preserve integrity checks,
  loaded-memory admission, final-strict review, resumability, result
  validation, reports, and backup.
- Live check at 16:58 SGT: only **1,437 MiB host RAM free**; RTX 4060 at
  **0 / 8,188 MiB**, 0% utilization, with no qualification/full-run process.
  This is below the 2,272 MiB cache floor and 2.8 GiB loaded target. No model
  was loaded. Free host RAM before GPU work.
- Assurance unit remains open, not review-ready, with 0/3 calls used and no
  reservation. No full candidate verification, freeze, reviewer call, branch
  push, archive import, or sustained run has occurred.

## 2026-10-01 17:20 SGT — Owner-authorized unqualified fast start (TDD)

- `TDD_REQUIRED: yes`. Observable seam: a full `transformers-local` run can
  skip the long profile matrix only with an explicit CLI flag; the run identity
  records `qualified: false`, the conservative default profile pair
  (greedy 1 / sampling 2), the reason for skipping, and no launch ETA. The
  final report and `summary.json` must disclose this status.
- RED: the CLI test failed because `--fast-start-unqualified` was rejected;
  the metadata test could not import the missing fast-start record helper; and
  the report regression failed because no qualification status appeared in
  `report.md`. GREEN added the gated CLI option, a signed run-identity record
  bound through the manifest, and report/summary disclosure. The flag is valid
  only for a full local run and cannot be combined with a qualification
  manifest. The existing OOM fallback sequence remains active.
- Focused GREEN evidence: the three fast-start regressions passed, then
  `.venv\Scripts\python.exe -m unittest tests.test_local_cli tests.test_local_run tests.test_report -v`
  passed **64 tests** in 143.704 seconds. The full suite passed **253 tests**
  before this code change and must be rerun on the frozen candidate.
- Current inference source snapshot:
  `ae37dee3b84ecd691e46815c80bda48b34197c9bbd53dc83f3d1385a50e96d91`.
  The candidate is not yet frozen or independently reviewed.
- Owner-authorized execution sequence: nine-row `smoke` profile; confirm at
  least 2.8 GiB available host RAM after model load; final-strict candidate
  review; import the validated 110 A100 rows into a new full run; start with
  `--fast-start-unqualified`; calculate ETA only from observed rates. The
  exhaustive matrix and 24-question comparison are waived.
- Live check at 17:18 SGT: **5,997 MiB host RAM free while idle**, GPU
  **0 / 8,188 MiB**, 0% utilization; no smoke or full-run process exists.
  Loaded memory is not yet measured.
- Assurance remains `UNIT_STATUS: open`, `REVIEW_READY: no`, 0/3 calls used,
  with no reservation. Full candidate checks, parent adversarial pass,
  final-strict review, branch push, import, full run, reports, and backup remain
  outstanding.

## 2026-10-01 17:33 SGT — Candidate checks and RTX 4060 smoke

- The current-candidate repository suite passed **256 tests in 165.347 seconds**. `compileall`, `pip check`, `python -m teacher_reliability.run --help`, and `git diff --check` passed. Git reported only the existing `.gitignore` LF-to-CRLF warning.
- Source snapshot: `ae37dee3b84ecd691e46815c80bda48b34197c9bbd53dc83f3d1385a50e96d91`; HEAD is still base `126621bfbe1c6228ef27debf7bdc69db8945dfba` on `codex/local-rtx4060`.
- Started the separate nine-row smoke in `%LOCALAPPDATA%\TeacherReliability\runs\smoke-rtx4060-20261001-173011` at 17:30:11 SGT. Idle host RAM was **5,790 MiB**, and GPU memory was 0 / 8,188 MiB before model load. The manifest records **5,130 MiB host RAM available after loading Qwen2.5-Math-7B-Instruct with NF4 double quantization**.
- At 17:33:02 the smoke was running with 0/9 complete rows, no failures, four durable stage files, and 294 tokens on the active request. Live host RAM was **2,939 MiB**, 71 MiB above the 2,868 MiB operating target; worker GPU free memory was 896 MiB.
- The memory controller observed 920 MiB normal-cache GPU free, below its 1,024 MiB threshold, and restarted the worker using the offloaded KV cache for greedy generation. Current settings are greedy batch 1 / offloaded cache and sampling batch 2. Do not lower the 2 GiB host reserve, memory floors, or change cache precision. Continue monitoring host RAM and let OOM backoff preserve completed work.
- Native Windows execution uses the single resident Transformers model with small batches and backoff; vLLM/Ray are not being used for this run. The 4060 reached 93% utilization during the smoke, with about 7 GiB allocated.
- Assurance remains open, not ready, 0/3 calls used, and no active reviewer reservation. No archive rows have been imported, and the sustained full run has not started.
## 2026-10-01 17:58 SGT — Smoke completion and remaining-work checklist

- Refreshed the project roadmap, runbook fast-path status, and completion
  checklist from the persisted RTX 4060 smoke manifest.
- Smoke completed 9/9 rows with zero failures and zero pending in 22.60
  minutes. It covered all three datasets and all seven MATH subjects; three
  generated samples reached the 1,024-token cap. The final persisted batches
  were greedy 1 and sampling 1 with offloaded KV cache after five memory
  adaptations. This smoke rate is not a full-run ETA.
- At 17:58 SGT, the study worker was stopped, GPU use was 0 MiB, and host RAM
  availability was 7,627 MiB. The validated 110-row archive remains unimported.
- Created docs/remaining-tasks-to-complete-study.md as the authoritative
  ordered checklist through review, push, archive import, sustained run,
  reporting, backup/restore, and final delivery. The exhaustive profile and
  24-question comparisons remain waived by owner direction.
- No source behavior changed. Candidate review, branch publication, full run,
  reports, and verified backup remain outstanding.

## 2026-10-01 18:51 SGT — Full-candidate verification checkpoint

- Re-read the persisted smoke manifest and progress file. The nine-row smoke
  completed 9/9 with zero failures or pending rows. Its final heartbeat at
  17:52:56 SGT reported 4,143 MiB available host RAM, 5,324 MiB allocated /
  6,094 MiB reserved GPU memory, and 896 MiB free VRAM. The latter is below the
  normal 1,024 MiB headroom target but above the 512 MiB emergency floor for
  offloaded KV cache. This is limited smoke evidence; the full run remains
  unqualified and needs live headroom monitoring.
- The working source fingerprint remains
  `ae37dee3b84ecd691e46815c80bda48b34197c9bbd53dc83f3d1385a50e96d91`, the
  same fingerprint saved by the smoke manifest. No behavior code changed after
  that run.
- Final-candidate checks passed: `.venv\Scripts\python.exe -m unittest
  discover -s tests -p 'test_*.py' -v` ran **256 tests in 175.167 seconds**;
  `compileall`, `pip check`, runner/report CLI help, `git diff --check`, and the
  untracked-aware text whitespace scan passed. The test output included only
  third-party Matplotlib/PyParsing deprecation warnings.
- Current parent `turn_context` records `gpt-6-luna/max` at ordinal 38371 in
  the persisted rollout; exact record-line SHA-256 is
  `a48d68e28c07656d58b1130c91b06f130edb5d0c9983c12e1e8f67824431d78f`.
  The fresh independent reviewer must still be `gpt-6-sol/max`.
- At 18:51 SGT, GPU use was 0 / 8,188 MiB, host RAM available was 6,626 MiB,
  and no local study worker was running. The archive remains unchanged at
  SHA-256 `7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82`
  and has not been imported.
- The existing assurance unit remains open and not review-ready, with 0/3
  calls used and no reservation. Branch `codex/local-rtx4060` remains at base
  `126621bfbe1c6228ef27debf7bdc69db8945dfba`, uncommitted and unpushed. The
  owner-waived profile matrix and 24-question benchmark are skipped; the next
  gates are parent adversarial evidence, exact candidate freezing, and the
  required final-strict review. No full run has started.

## 2026-10-01 19:00 SGT — Parent adversarial pass and candidate preparation

- Completed and recorded the parent adversarial pass at
  `.assurance/Teacher-reliability-Distill-LLM/vllm-hybrid/parent-adversarial.md`.
  It maps inference/scoring, seeded sampling, GPU/host-memory recovery, worker
  lifecycle, archive import, checkpoint integrity, Windows grading, report
  completeness, compatibility, counterexamples, changed-to-unchanged paths,
  and test sensitivity.
- The selected Windows Transformers path has no remaining blocking gap against
  the pre-run software acceptance. The separately experimental vLLM path's
  report-origin and stop-reason limitations remain explicit residuals; it is
  not used or recommended for this run. A100 hardware fields remain reported
  provenance, not independent physical-GPU attestation.
- Rebound all **46 changed or untracked repository files** to a complete
  cumulative diff from base `126621bfbe1c6228ef27debf7bdc69db8945dfba` on
  `codex/local-rtx4060`. The candidate includes the source, tests, dependency
  files, runbook, roadmap, completion checklist, and process log. The machine
  manifest and diff are outside the candidate in the existing assurance unit.
- The current source snapshot remains
  `ae37dee3b84ecd691e46815c80bda48b34197c9bbd53dc83f3d1385a50e96d91`. The
  candidate-wide suite passed 256 tests in 175.167 seconds, and compileall,
  pip check, runner/report CLI help, diff checks, and the untracked-aware text
  whitespace scan passed. Only documentation and evidence checkboxes changed
  after that suite; no production or test code changed.
- The current parent runtime is observed as `gpt-6-luna/max` in the persisted
  `turn_context` evidence recorded above. The required fresh reviewer remains
  `gpt-6-sol/max`.
- The existing assurance unit remains open at 0/3 calls with no reservation.
  The machine readiness record and exclusive attempt reservation remain
  pending. The branch is not committed or pushed, the 110-row archive has not
  been imported, and the sustained full run has not started.

## 2026-10-01 20:21 SGT — First local-review result and fixed-input RED

- The fresh reviewer completed call 1 with **fix-first** and a complete audit.
  Its observed runtime was GPT-6-Sol/max. F1 found that full local launch did
  not enforce the declared model/dataset revisions, seed, and exact validated
  A100 ZIP digest; the report only checked internally consistent revisions.
  F2 found the 256-test receipt predates later in-scope documentation. F3
  identified the historical call-1 Luna parent runtime. F4 found an internally
  contradictory ledger header. The candidate was not accepted.
- Under the existing `.NET FileStream CreateNew/FileShare.None` coordination
  lock, operation `30eb06b9-da01-4e90-9f72-aa6f888c2996` recorded call 1 as
  consumed, cleared its reservation, and set the unit to open and not ready.
  The durable sidecar now has **1/3 calls used** and two remaining; no study
  worker was launched. This continuation's parent `turn_context` records
  GPT-6-Sol/max, turn `01a0f763-2e61-7772-9af7-db396fe3036f`, ordinal
  40121, exact line SHA-256
  `ac04ac3850a8f00c3ad9edb9e7e949956143c47f35b0bf5ac2063a23342846d0`.
  The Sol continuation does not change the call-1 runtime history.
- `TDD_REQUIRED: yes` for F1. Observable seams: a self-consistent alternate
  A100 ZIP is rejected by the expected snapshot SHA-256; full local launch
  rejects seed 43 and an alternate ZIP; the full-report validator rejects
  self-consistent but unpinned model/dataset revisions and seed. The five
  focused tests were written before implementation. The RED command was
  `.venv\Scripts\python.exe -m unittest` for the five named tests in
  `tests.test_archive_import`, `tests.test_local_cli`, and `tests.test_report`.
  It ran five tests in **5.293 seconds** with five assertion failures: the
  importer lacked an expected-digest parameter; seed 43 reached dataset
  loading; an alternate ZIP reached manifest validation; and two self-
  consistent unpinned report inputs were marked valid. Production code had not
  changed at the RED checkpoint.
- GREEN: the five focused F1 tests passed in **3.079 seconds** after the
  importer began checking the SHA-256 of its immutable byte snapshot, and full
  launch/report began enforcing seed 42, the pinned model commit, and the three
  pinned dataset commits. A separate report test then observed RED: a
  self-consistent alternate archive yielded `run_complete: true` before the
  report pinned its source archive hash (one assertion failure in 14.188 s).
  After pinning that hash, the positive exact-archive, alternate-archive, and
  legacy-counter report tests passed (three tests in 47.056 s). The supplied
  110-row ZIP was copied into the test fixtures at the same SHA-256 so the
  complete-report success path exercises real imported rows. The final focused
  report module passed **36 tests in 152.034 seconds**. The historical
  256-test candidate-wide receipt remains stale; a fresh full pass is next.
- The runbook now checks the approved ZIP hash before start and resume, and
  its status command resolves the repository Python interpreter. The full
  entry point independently checks the immutable imported-byte snapshot, so
  changing the path after the shell check cannot substitute another archive.

## 2026-10-01 20:47 SGT — Fixed-input candidate verification

- The fixed-input source passed the full unittest discovery: **262 tests in
  192.218 seconds**. Bytecode compilation, `pip check`, runner and report CLI
  help, `git diff --check`, and the untracked-aware whitespace scan passed.
  The source snapshot is
  `86294937b66f00a0dc729435e1e243292d257ec840fa5a46c02f7a37c513d1b8`.
- The candidate now includes a byte-exact copy of the 110-row archive as a
  regression fixture. Both the fixture and original ZIP hash to
  `7fa37f4a6e083e293e4f29606f7643f7ede940569d531410c47e8162130a4d82`.
  The current scope contains 47 changed or added files, including that binary
  fixture in the Git binary patch.
- This status documentation was updated after the first 262-test pass. The
  final frozen candidate will receive a fresh byte-bound suite receipt in the
  existing assurance unit before reviewer call 2. No sustained run, commit,
  or push has started.

## 2026-10-02 — 12-hour mini study (`mini_12h`) complete

- Owner downscaled the study to a 12-hour mini run. Profile `mini_12h`: 40 GSM8K, 21 MATH (3 per subject), 50 GSM-Plus (one per seed), full protocol (8 samples, 1,024-token cap, seed 42, NF4/FP16). No A100 import (importer is full-profile only).
- Run `mini12h-rtx4060-20261001-230236`, started 2026-10-01 23:02 SGT, first pass ended 04:40 SGT with 109/111 (2 rows failed `InsufficientHostMemory` for the offloaded-KV-cache 2 GiB reserve). `--resume --retry-failed` completed both: **111/111, 0 failed, 0 pending**.
- Measured throughput: about **19.4 rows/hour** (greedy 1 / sample 1, offloaded KV cache, GPU ~85 °C with thermal slowdown).
- Report: `results/mini12h-rtx4060-20261001/` (recommendation status exploratory). Raw-run backup `mini12h-rtx4060-20261001-230236.tar.gz`, SHA-256 `06ed0976394cc485ee5425160fb08ea0ab59e28ce14e41a42d7bc599a4cd1014`; restore check reproduced the 111-line `predictions.jsonl` with identical SHA-256.

## 2026-10-02 — Mini distillation follow-up

- `scripts/mini_distill.py`: LoRA SFT of Qwen2.5-0.5B on the mini_12h teacher outputs; conditions base / kd_all / kd_gated (self-consistency, majority share ≥ 0.75) / kd_oracle; 640 held-out questions; paired bootstrap.
- `peft==0.15.2` installed with `--no-deps --target %LOCALAPPDATA%\TeacherReliability\pylib` (project venv unchanged; run with that folder on `PYTHONPATH`).
- Fixes found during the smoke: loader returns only the profile selection (held-out pool now loads the `full` profile); base student never learned `<|im_end|>`, so training ends responses with `<|endoftext|>`, and every condition is graded on text through its first `\boxed{}`.
- `kd_oracle` hit CUDA OOM in the first process after three conditions; a fresh-process resume with identical settings completed it.
- Overall held-out accuracy: base 25.5%, kd_all 37.8%, kd_gated 38.8%, kd_oracle 37.0%. Gate − all = +0.9 pts, 95% CI [−2.0, +4.1]. See `results/mini-distill-20261002/report.md`.
