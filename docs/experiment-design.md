# Experiment design: when to trust the teacher

## Research questions

- **When:** Does confidence from a math teacher predict whether its answer is
  correct?
- **What:** Which measured signal is the strongest, most stable candidate for
  filtering or weighting teacher examples before knowledge distillation?

The experiment evaluates a teacher-side gate. It does not train a student or
prove that a downstream KD run improves; that requires a later controlled KD
ablation.

## Model and evaluation sets

- Teacher: [`Qwen/Qwen2.5-Math-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct), inference only, loaded with 4-bit
  quantization. Resolve the Hub `main` revision at run start and record its
  commit SHA and quantization settings.
  Resume reuses the manifest's commit rather than resolving `main` again.
- GSM8K: `openai/gsm8k`, `main` configuration, `test` split (1,319 rows in the
  dataset card).
- MATH: `EleutherAI/hendrycks_math`, `test` split across its seven subject
  configurations. Select a deterministic, subject-stratified subset; record
  every included dataset row ID. The source benchmark contains multiple
  competition-math subjects and full worked solutions.
- GSM-Plus: `qintongli/GSM-Plus`, current Hub `test` split. The current dataset
  viewer exposes about 10,552 variations in that split. Smoke and pilot use
  deterministic hash-selected rows from the same pinned split; the pilot takes
  at most one variation per seed question. Preserve perturbation categories
  and seed-question identity so related rows can be grouped. The older project
  `testmini` layout is not used by the current Hub loader.

The `critical thinking` GSM-Plus category can have no gold answer. Such rows
are retained in coverage diagnostics but marked unscorable for answer
correctness; they are never silently labeled wrong or dropped without a
count.

### Time-bounded A100 subsets

`a100_12h` selects 150 GSM8K rows, all 350 MATH rows (50 per subject), and 220
GSM-Plus rows from distinct seed questions. `a100_24h` uses 300, 350, and 790
rows respectively. With a fixed seed, both selections are deterministic, and
the smaller is contained in the larger. Both retain the greedy completion,
eight samples at temperature 0.7, and the 1,024-token cap. All sources load at
immutable revisions, while their selected row counts remain below `full`.

These names describe approximate runtime targets, not enforced deadlines. The
planning baseline is the local nine-row smoke: 27 generations completed in
673 seconds. Assuming fourfold A100 generation throughput estimates about
11.2 and 22.4 hours of inference. No A100 timing has been measured yet; host
load, downloads, A100 variant, token lengths, and kernel performance affect
actual elapsed time. The run manifest records target hours, GPU model/memory,
selected IDs, and actual times. These smaller profiles report exploratory
metrics only; a complete `full` profile remains necessary for a candidate
KD-gating recommendation.

## Per-example protocol

1. Render one fixed chat prompt with the problem and the instruction to show
   reasoning and put the final answer in `\\boxed{...}`. No gold solution,
   reference answer, or test-time demonstration is sent to the teacher.
2. Generate one greedy completion (`do_sample=False`, one beam). Save the raw
   text and generated token IDs.
3. From the greedy step distributions, save per-token entropy and token log
   probabilities. Extract the final-answer character span and align it to the
   generated token positions only when token IDs round-trip exactly through
   the fast tokenizer. If alignment or parsing fails, save a failure reason
   and leave answer-token confidence missing; do not impute it. Record EOS and
   max-token termination separately.
4. Generate `k=8` independent samples at temperature `T=0.7`. Normalize
   extracted answers with the same mathematical answer checker. Record the
   fraction agreeing with the greedy answer, the modal answer, its vote share,
   and its ground-truth correctness.
5. Compare the greedy answer with the reference via Math-Verify. Preserve
   unparseable predictions as incorrect for the primary exact-correct label,
   while recording parse coverage separately. A missing gold answer is
   unscorable instead.

Sampling is performed per example in a stable order after deriving each sample
seed from the configured base seed, example ID, and sample index. The run
manifest records model/dataset revisions, IDs, prompt, seed, decoding and
quantization settings, package versions, hardware, start/end time, and status.
Per-example outputs are checkpointed so interrupted inference can resume
without duplicating completed examples.

Sampling overrides `do_sample=True`, one beam, and temperature `0.7`. Other
generation options follow the pinned model configuration and Transformers
version. Inspecting the local pinned configuration resolved `top_k=50`,
`top_p=1.0`, and `repetition_penalty=1.0`; its EOS IDs are 151645 and 151643.
These inherited options are part of the recorded model/package provenance,
so a new model configuration or package version is a new run identity.

## Confidence signals

Every ECE and AUROC signal is oriented so a larger score means “more likely
correct.” Raw values are retained alongside the transformed values.

| Signal | Stored raw value | Score used for ECE/AUROC | Interpretation |
|---|---|---|---|
| Token entropy | Mean Shannon entropy across greedy generated-token distributions, in nats | `1 - mean_entropy / ln(vocabulary_size)`, clipped only for floating-point roundoff | Bounded concentration proxy; it is **not** a probability of correctness |
| Answer-token probability | Geometric mean probability of tokens overlapping the extracted final-answer span | Same value in `[0,1]` | Average conditional probability of the answer tokens |
| Sequence log-probability | Sum of greedy generated-token log probabilities | `exp(sequence_logprob / generated_token_count)` | Length-normalized geometric mean token probability |
| Self-consistency agreement | Count of the `k` sampled answers matching the greedy answer | `matching_samples / k` | Sample support for the exact greedy answer; directly comparable against greedy correctness |
| Majority vote (secondary) | Largest normalized-answer count among the `k` samples | `majority_count / k` | Confidence in a different prediction; report with majority-answer correctness, separately from the primary AUROC comparison |

Special tokens excluded from the visible completion are excluded from
confidence aggregation. The implementation records whether EOS ended
generation or the token cap truncated it. If there are no scored answer
tokens, answer-token confidence is missing rather than substituted by
sequence confidence.

On Windows, Math-Verify's built-in positive timeout uses a worker that is not
spawn-picklable in this environment. The project therefore caps parsed answer
text at 2,048 characters and disables that package timeout on Windows; the
Linux setup retains Math-Verify's five-second timeout. This platform difference
is recorded because it affects pathological-expression handling, not the
math-equivalence rule.

## Outcomes and metrics

- Primary binary outcome: `greedy_correct ∈ {0,1}` by Math-Verify against the
  dataset reference.
- For each dataset and each eligible confidence signal, report `n`, signal
  coverage, accuracy, and equal-width expected calibration error (ECE).
- ECE uses 15 bins by default. Intervals are left-closed/right-open except
  that the final interval includes confidence `1.0`; empty bins contribute
  zero. Report the per-bin count, mean confidence, and empirical accuracy so
  the aggregate can be audited.
- AUROC treats correctness as the positive class. Compare normalized token
  entropy confidence with greedy-answer agreement share on the same examples:
  the `paired_primary` rows use the intersection where both scores and the
  correctness label are available. The shared count and coverage are reported.
  Also show answer-token and sequence confidence as secondary predictors.
  AUROC is unavailable (not zero) when the evaluated rows contain only one
  class.
- The majority-vote answer and vote share are evaluated against a separate
  `majority_correct` outcome. Do not mix these rows into the primary greedy
  correctness comparison.
- Produce reliability plots by dataset and a CSV with per-dataset ECE/AUROC.
  An optional macro summary gives each dataset equal weight; it must not pool
  all examples and hide the GSM-Plus distribution shift.

## Recommendation rule and limitations

Only a complete `full` profile can receive a candidate-gate recommendation.
Smoke, pilot, and time-bounded A100 reports are explicitly exploratory. For the full profile, the
report ranks confidence signals that cover every scorable row in every dataset
by macro AUROC against greedy correctness, then uses macro ECE as a tie-breaker.
Per-signal metrics and reliability diagrams retain each signal's eligible
rows and report coverage; a signal with missing scores remains descriptive and
cannot win a gate recommendation by omitting difficult rows. It names a candidate KD gate
only when all requested datasets have defined AUROC and the leading signal is
not below chance on any dataset. Otherwise it says the evidence is mixed or
insufficient. A low-confidence cutoff for a later KD experiment must be chosen
on a separate calibration/validation split, not on the final evaluation sets.

The report also applies a prespecified coverage gate before it names any
candidate signal. In every dataset, at least **95% of selected rows** must have
a parsed source reference, every source-scorable row must have a known binary
greedy outcome, and the signal must have a score for every such row. The 95%
floor limits source-reference exclusions to 5% per dataset; unknown grader
outcomes have zero tolerance because selectively dropping timeout or worker
failure rows could change the evaluated cohort. The report exposes selected,
source-scorable, labeled, and unknown counts and their fractions by dataset.
Failing this gate withholds the recommendation while retaining descriptive
metrics for the complete run.

The report does not calculate confidence intervals for AUROC or ECE. Its
ranking is a descriptive candidate for a later KD ablation, not a claim of
statistical superiority or expected student improvement. Any threshold must
be selected on a separate calibration/validation split and tested in a
downstream distillation experiment.

ECE depends on bin count and confidence transformations; normalized entropy
is a heuristic score, not calibrated correctness probability. AUROC does not
measure calibration. Quantized weights, prompt choice, answer extraction,
sampling variance, finite subset size, and GSM-Plus variants can change the
ranking. The study can motivate a GRACE-style entropy gate; it cannot establish
that the gate improves student learning without a downstream KD experiment.
Here “GRACE-style” means the confidence-gated distillation framing in
[Gated Relational Alignment via Confidence-based Distillation for Efficient
VLMs](https://arxiv.org/abs/2601.22709). That paper concerns quantization-aware
vision-language-model distillation, so this math-only study is motivation and
signal validation, not a reproduction of GRACE. It is also distinct from
[In Good GRACEs](https://arxiv.org/abs/2511.02833), which proposes a separate
gradient-based score for teacher selection rather than an entropy gate.

## Reproduction references

- [Qwen2.5-Math model card](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
- [GSM8K card](https://huggingface.co/datasets/openai/gsm8k)
- [EleutherAI MATH card](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
- [GSM-Plus dataset card and current test split](https://huggingface.co/datasets/qintongli/GSM-Plus) and [official project](https://github.com/qtli/GSM-Plus)
- [Math-Verify parser and comparison](https://github.com/huggingface/Math-Verify)
- [Transformers score outputs](https://huggingface.co/docs/transformers/main_classes/text_generation)
