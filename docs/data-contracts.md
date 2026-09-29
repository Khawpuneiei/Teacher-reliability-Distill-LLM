# Data and artifact contracts

## Input example schema

Every normalized benchmark row has:

| Field | Type | Meaning |
|---|---|---|
| `example_id` | string | Stable ID derived from source, configuration, split, and source row ID |
| `dataset` | string | `gsm8k`, `math`, or `gsm_plus` |
| `source_revision` | string | Resolved dataset repository SHA |
| `source_config` | string | Source dataset configuration (`main`, a MATH subject, or `default`) |
| `split` | string | Evaluation split; currently `test` for all sources |
| `source_row_id` | string | Original row/config identifier |
| `question` | string | Problem sent to the teacher |
| `gold_answer` | string or null | Reference answer only; never sent in the prompt |
| `gold_solution` | string or null | Dataset solution, retained locally for provenance/debugging but never sent |
| `gold_status` | string | `parsed`, `unanswerable`, or a specific missing-answer reason |
| `subject` | string or null | MATH subject when present |
| `perturbation_type` | string or null | GSM-Plus variation category when present |
| `seed_question_id` | string or null | Stable SHA-256 prefix of normalized GSM-Plus seed-question text, shared by its variants |

Dataset IDs are source/config/split/row based, not hashes of normalized text.
The loaders resolve one immutable commit SHA per dataset repository, then pass
that SHA to every `load_dataset` call. MATH quotas are independently applied
within each subject. The smoke/pilot GSM-Plus cap chooses at most one
hash-ranked row per seed question; the full profile retains the current Hub
`test` split in full.
Before computing correctness, the dataset adapter must confirm that the
problem, reference answer, source ID, and split are present and non-empty,
except that GSM-Plus's documented unanswerable cases may have a null gold
answer. These are tagged `unanswerable` and counted separately. Missing GSM8K
final markers and missing MATH boxed answers are retained with a distinct
`gold_status` and excluded from correctness metrics rather than silently
discarded.

## Per-example result schema

Each checkpoint record contains the normalized source identity, question,
gold answer, and:

- raw greedy response, generated token IDs, per-token entropy and log
  probability, and a visible-token mask;
- greedy answer, Math-Verify parse status, correctness, entropy confidence,
  sequence log-probability and geometric token probability;
- answer-token geometric probability, token count, offset-alignment status,
  and an explicit null when token offsets could not be verified;
- EOS versus max-token termination metadata for greedy and sampled outputs;
- each sampled response, derived seed, extracted answer, and parse metadata;
- greedy-agreement count/share and the separately graded majority answer,
  vote share, correctness, and sample parse count.

Records are UTF-8 JSON Lines, appended one complete example at a time and
flushed before inference continues. A completed record is never edited in
place. Resume checks the complete run signature before skipping saved IDs; a
torn final JSONL line is discarded only when resuming.

## Run manifest

`run_manifest.json` binds each output directory to:

- run ID, profile, package/source digest, base seed, and prompt-template hash;
- model and tokenizer repository/revision, 4-bit NF4 quantization, decode and
  sample parameters;
- each dataset repository SHA, selected IDs, ordered-selection hash, and hash
  of the complete normalized input records;
- Python/platform, pinned package versions, CUDA runtime, GPU name/VRAM,
  timestamps, progress count, and terminal run status. Python/platform/GPU
  identity is included in the resume signature.

The dataset/model repo revisions default to `main` only as a lookup request;
the resolved commit SHAs are written to the manifest and used for loading.
The tokenizer is loaded from the same model repository commit and records that
revision explicitly.
Resume loads those same commits and checks the signature before skipping any
checkpoint. Preloaded examples must agree with the dataset revision manifest,
and duplicate IDs or altered input content are rejected.

## Generated outputs

Reports include `predictions.jsonl`, `metrics.csv`, `calibration_bins.csv`,
`reliability_<dataset>.png`, `summary.json`, `report.md`, and the run manifest.
All generated files live under ignored `outputs/`; Hugging Face caches live in
the configured cache directory outside the repository. Never commit raw
benchmark rows, cached model files, generated solutions, or report artifacts.
Keep licenses and citations in the documentation; follow each dataset's own
terms.

## Sources and licensing notes

- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) provides `question`
  and `answer`, including a final answer after `####`; the canonical test split
  has 1,319 examples.
- [MATH](https://huggingface.co/datasets/EleutherAI/hendrycks_math) provides
  subject-specific problem/solution records. The original benchmark is
  [hendrycks/math](https://github.com/hendrycks/math); preserve its license and
  citation requirements.
- [GSM-Plus](https://huggingface.co/datasets/qintongli/GSM-Plus) provides
  `question`, `solution`, `answer`, perturbation type, and seed-example fields.
  The dataset card lists CC BY-SA 4.0. The official project recommends its v1
  data over v0. The current Hub view exposes one `test` split; this project
  selects deterministic pilot rows from that split rather than relying on the
  older `testmini` layout.
- [Qwen2.5-Math-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)
  is an Apache-2.0 model card. Review the model license and dataset-specific
  terms before redistribution; this repository includes no weights or rows.
