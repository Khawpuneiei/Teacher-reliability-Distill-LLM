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

## Release sequence

The frozen project is reviewed before publication. The parent then commits and
pushes the accepted source tree to the exact requested `origin` and compares
the remote branch SHA with the local commit. Git history is the publication
receipt; the independent-review and protected-transition records are kept in
the durable workspace assurance ledger outside the code candidate.

## Follow-up experiment

1. Run the full pinned evaluation only after deciding whether local multi-hour
   execution or A100/Vast.ai execution is appropriate. Account onboarding is
   complete for the recorded controller; the full study has not been run.
