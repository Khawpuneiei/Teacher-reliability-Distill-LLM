# Teacher reliability report

This completed 111-row run is the teacher portion of the mini study. It
remains exploratory for selecting a production confidence gate.

- Run: `mini12h-rtx4060-20261001-230236`
- Profile: `mini_12h`
- Run status: `complete`
- Completed rows: 111 / 111
- Failed rows: 0
- Pending rows: 0
- Recommendation status: **exploratory**
- Candidate signal: `none`
- Reason: The mini_12h subset (111 questions) is too small to justify a KD gate; metrics are descriptive only.

The primary greedy analysis predicts greedy-answer correctness. Majority-vote correctness is a separate analysis.
The paired_primary rows compare entropy and self-consistency on their shared eligible cohort; greedy rows also show per-signal coverage.
Math-Verify timeout and worker-failure checks are recorded as unknown, not as false.
The candidate ranking is descriptive; no confidence intervals or statistical-significance claims are produced.
Smoke and mini_12h runs are exploratory and cannot justify a KD gate.

## Grading-check uncertainty

| Dataset | Sample parse unknowns | Greedy-comparison unknowns | Majority-comparison unknowns |
|---|---:|---:|---:|
| gsm8k | 0 | 0 | 0 |
| gsm_plus | 0 | 0 | 0 |
| math | 0 | 0 | 0 |

## Outcome label coverage

Candidate recommendations require at least 95% source-scorable rows and a known binary outcome for every source-scorable row in every dataset.

| Dataset | Selected rows | Source-scorable rows | Source coverage | Known outcomes | Outcome coverage | Unknown outcomes |
|---|---:|---:|---:|---:|---:|---:|
| gsm8k | 40 | 40 | 100.0% | 40 | 100.0% | 0 |
| gsm_plus | 50 | 48 | 96.0% | 48 | 100.0% | 0 |
| math | 21 | 21 | 100.0% | 21 | 100.0% | 0 |

## Execution provenance

Execution label: **RTX 4060 local**

| Execution | Hardware | Rows | GSM8K | MATH | GSM-Plus |
|---|---|---:|---:|---:|---:|
| rtx4060-local | NVIDIA GeForce RTX 4060 Laptop GPU | 111 | 40 | 21 | 50 |

Metrics by source execution are also in `metrics_by_execution.csv`.

## Metrics

| Analysis | Dataset | Signal | n | Coverage | Accuracy | ECE | AUROC |
|---|---|---|---:|---:|---:|---:|---:|
| greedy | gsm8k | entropy_confidence | 40 | 1.000 | 0.850 | 0.143 | 0.765 |
| greedy | gsm8k | answer_token_probability | 40 | 1.000 | 0.850 | 0.148 | 0.716 |
| greedy | gsm8k | sequence_geometric_probability | 40 | 1.000 | 0.850 | 0.125 | 0.789 |
| greedy | gsm8k | self_consistency_agreement | 40 | 1.000 | 0.850 | 0.103 | 0.895 |
| paired_primary | gsm8k | entropy_confidence | 40 | 1.000 | 0.850 | 0.143 | 0.765 |
| paired_primary | gsm8k | self_consistency_agreement | 40 | 1.000 | 0.850 | 0.103 | 0.895 |
| majority | gsm8k | majority_vote_share | 40 | 1.000 | 0.900 | 0.072 | 0.809 |
| greedy | gsm_plus | entropy_confidence | 48 | 1.000 | 0.812 | 0.162 | 0.846 |
| greedy | gsm_plus | answer_token_probability | 47 | 0.979 | 0.830 | 0.167 | 0.760 |
| greedy | gsm_plus | sequence_geometric_probability | 48 | 1.000 | 0.812 | 0.130 | 0.858 |
| greedy | gsm_plus | self_consistency_agreement | 48 | 1.000 | 0.812 | 0.102 | 0.736 |
| paired_primary | gsm_plus | entropy_confidence | 48 | 1.000 | 0.812 | 0.162 | 0.846 |
| paired_primary | gsm_plus | self_consistency_agreement | 48 | 1.000 | 0.812 | 0.102 | 0.736 |
| majority | gsm_plus | majority_vote_share | 48 | 1.000 | 0.833 | 0.120 | 0.698 |
| greedy | math | entropy_confidence | 21 | 1.000 | 0.714 | 0.244 | 0.900 |
| greedy | math | answer_token_probability | 19 | 0.905 | 0.789 | 0.207 | 0.867 |
| greedy | math | sequence_geometric_probability | 21 | 1.000 | 0.714 | 0.202 | 0.867 |
| greedy | math | self_consistency_agreement | 21 | 1.000 | 0.714 | 0.131 | 0.950 |
| paired_primary | math | entropy_confidence | 21 | 1.000 | 0.714 | 0.244 | 0.900 |
| paired_primary | math | self_consistency_agreement | 21 | 1.000 | 0.714 | 0.131 | 0.950 |
| majority | math | majority_vote_share | 21 | 1.000 | 0.810 | 0.101 | 0.868 |

## Metrics by execution

| Execution | Analysis | Dataset | Signal | n | Coverage | Accuracy | ECE | AUROC |
|---|---|---|---|---:|---:|---:|---:|---:|
| rtx4060-local | greedy | gsm8k | entropy_confidence | 40 | 1.000 | 0.850 | 0.143 | 0.765 |
| rtx4060-local | greedy | gsm8k | answer_token_probability | 40 | 1.000 | 0.850 | 0.148 | 0.716 |
| rtx4060-local | greedy | gsm8k | sequence_geometric_probability | 40 | 1.000 | 0.850 | 0.125 | 0.789 |
| rtx4060-local | greedy | gsm8k | self_consistency_agreement | 40 | 1.000 | 0.850 | 0.103 | 0.895 |
| rtx4060-local | paired_primary | gsm8k | entropy_confidence | 40 | 1.000 | 0.850 | 0.143 | 0.765 |
| rtx4060-local | paired_primary | gsm8k | self_consistency_agreement | 40 | 1.000 | 0.850 | 0.103 | 0.895 |
| rtx4060-local | majority | gsm8k | majority_vote_share | 40 | 1.000 | 0.900 | 0.072 | 0.809 |
| rtx4060-local | greedy | gsm_plus | entropy_confidence | 48 | 1.000 | 0.812 | 0.162 | 0.846 |
| rtx4060-local | greedy | gsm_plus | answer_token_probability | 47 | 0.979 | 0.830 | 0.167 | 0.760 |
| rtx4060-local | greedy | gsm_plus | sequence_geometric_probability | 48 | 1.000 | 0.812 | 0.130 | 0.858 |
| rtx4060-local | greedy | gsm_plus | self_consistency_agreement | 48 | 1.000 | 0.812 | 0.102 | 0.736 |
| rtx4060-local | paired_primary | gsm_plus | entropy_confidence | 48 | 1.000 | 0.812 | 0.162 | 0.846 |
| rtx4060-local | paired_primary | gsm_plus | self_consistency_agreement | 48 | 1.000 | 0.812 | 0.102 | 0.736 |
| rtx4060-local | majority | gsm_plus | majority_vote_share | 48 | 1.000 | 0.833 | 0.120 | 0.698 |
| rtx4060-local | greedy | math | entropy_confidence | 21 | 1.000 | 0.714 | 0.244 | 0.900 |
| rtx4060-local | greedy | math | answer_token_probability | 19 | 0.905 | 0.789 | 0.207 | 0.867 |
| rtx4060-local | greedy | math | sequence_geometric_probability | 21 | 1.000 | 0.714 | 0.202 | 0.867 |
| rtx4060-local | greedy | math | self_consistency_agreement | 21 | 1.000 | 0.714 | 0.131 | 0.950 |
| rtx4060-local | paired_primary | math | entropy_confidence | 21 | 1.000 | 0.714 | 0.244 | 0.900 |
| rtx4060-local | paired_primary | math | self_consistency_agreement | 21 | 1.000 | 0.714 | 0.131 | 0.950 |
| rtx4060-local | majority | math | majority_vote_share | 21 | 1.000 | 0.810 | 0.101 | 0.868 |

## Interpretation limits

Entropy confidence is a normalized concentration proxy, not a probability of correctness. This report does not measure whether a student improves under knowledge distillation.

Raw per-example generations and score vectors are in `predictions.jsonl`; this run directory is ignored by Git.
