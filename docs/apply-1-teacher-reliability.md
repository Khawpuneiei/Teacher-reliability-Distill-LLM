# Apply 1 — Teacher Reliability & Calibration ("When to trust the teacher")

## Goal
Test if teacher confidence predicts correctness on math.

## Models / Data
- Teacher: Qwen2.5-Math-7B-Instruct (4-bit, inference only)
- Data: GSM8K test, MATH subset, GSM-Plus (OOD)

## Steps
1. Greedy decode; log per-token entropy, answer-token prob, sequence log-prob.
2. Sample k=8–16 @ T=0.7 → self-consistency vote share.
3. Check agreement with ground truth.
4. Reliability diagrams + ECE per confidence signal.
5. AUROC: token entropy vs self-consistency as correctness predictor.

## Compute
4–8 GPU-hrs, T4/L4 fine.

## Outputs
- Reliability diagrams, ECE table, AUROC comparison
- Recommendation: which signal to gate KD on (justifies GRACE-style entropy gate)

## RQ link
When (teacher reliability); secondary What (gating signal choice).
