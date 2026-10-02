"""Mini knowledge-distillation follow-up on the mini_12h teacher run.

Fine-tunes a small student (Qwen2.5-0.5B, LoRA) on teacher responses from a
completed teacher-reliability run and compares, on held-out questions:

- ``base``: the untrained student;
- ``kd_all``: SFT on every complete teacher response (greedy + 8 samples);
- ``kd_gated``: SFT only on responses that pass a self-consistency gate
  (the response answer equals the question's majority answer and the majority
  vote share is at least ``--gate-threshold``); no gold labels are used;
- ``kd_oracle``: SFT only on responses graded correct against the gold
  answer (a label-using reference, not a deployable gate).

Held-out evaluation questions are disjoint from the training questions, and
GSM8K / GSM-Plus items that share a seed question with any training item are
excluded to avoid perturbation leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from teacher_reliability.answers import _matching_brace  # noqa: E402
from teacher_reliability.data import _seed_question_id  # noqa: E402
from teacher_reliability.grading import MathVerifyWorker, grade_prediction  # noqa: E402
from teacher_reliability.run import LOCAL_MODEL_REVISION, _load_cached_examples  # noqa: E402
from teacher_reliability.selection import RunProfile  # noqa: E402
from teacher_reliability.teacher import build_chat_messages, derive_sample_seed  # noqa: E402

STUDENT = "Qwen/Qwen2.5-0.5B"
STUDENT_REVISION = "060db6499f32faf8b98477b0a26969ef7d8b9987"
CONDITIONS = ("base", "kd_all", "kd_gated", "kd_oracle")
HELDOUT_POOL = RunProfile("heldout_pool", None, None, None, False, "test", 8, 1024)
MINI_TEACHER_ROW_COUNT = 111
MINI_TEACHER_SAMPLE_COUNT = 8
MINI_TEACHER_SEED = 42
MINI_TEACHER_MODEL_REPOSITORY = "Qwen/Qwen2.5-Math-7B-Instruct"
MINI_TEACHER_DATASET_REVISIONS = {
    "EleutherAI/hendrycks_math": "21a5633873b6a120296cce3e2df9d5550074f4a3",
    "openai/gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "qintongli/GSM-Plus": "3b708db57b96a16e8e3368ed2956990c0809440e",
}
MINI_TEACHER_ROW_REVISIONS = {
    "gsm8k": MINI_TEACHER_DATASET_REVISIONS["openai/gsm8k"],
    "math": MINI_TEACHER_DATASET_REVISIONS["EleutherAI/hendrycks_math"],
    "gsm_plus": MINI_TEACHER_DATASET_REVISIONS["qintongli/GSM-Plus"],
}
MINI_TEACHER_PROFILE = {
    "name": "mini_12h",
    "gsm8k_limit": 40,
    "math_per_subject": 3,
    "gsm_plus_limit": 50,
    "gsm_plus_one_per_seed": True,
    "gsm_plus_split": "test",
    "sample_count": MINI_TEACHER_SAMPLE_COUNT,
    "max_new_tokens": 1024,
    "target_hours": 12,
}
MINI_TEACHER_DATASET_COUNTS = {"gsm8k": 40, "math": 21, "gsm_plus": 50}
MINI_MATH_SUBJECTS = {
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
}


def _rank(example_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{example_id}".encode("utf-8")).hexdigest()


def _seed_group(example) -> str | None:
    if example.dataset == "gsm8k":
        return _seed_question_id(example.question)
    if example.dataset == "gsm_plus":
        return example.seed_question_id
    return None


def load_training_rows(predictions_path: Path) -> list[dict]:
    manifest_path = predictions_path.with_name("run_manifest.json")
    if not manifest_path.is_file():
        raise ValueError(f"mini teacher run is missing {manifest_path.name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = []
    with predictions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                raise ValueError("mini teacher predictions contain an empty line")
            rows.append(json.loads(line))
    _validate_mini_teacher_run(manifest, rows)
    return rows


def _validate_mini_teacher_run(manifest: dict, rows: list[dict]) -> None:
    """Reject teacher inputs that do not match the completed 111-row study."""
    if manifest.get("status") != "complete":
        raise ValueError("mini teacher run must have status 'complete'")
    profile = manifest.get("profile")
    if not isinstance(profile, dict) or any(
        profile.get(key) != expected for key, expected in MINI_TEACHER_PROFILE.items()
    ):
        raise ValueError("mini distillation requires the exact mini_12h teacher profile")
    if manifest.get("seed") != MINI_TEACHER_SEED:
        raise ValueError("mini distillation requires teacher seed 42")
    if manifest.get("model_repository") != MINI_TEACHER_MODEL_REPOSITORY:
        raise ValueError("mini distillation requires Qwen2.5-Math-7B-Instruct teacher outputs")
    if manifest.get("model_revision") != LOCAL_MODEL_REVISION:
        raise ValueError("mini teacher model revision differs from the pinned run")
    identity = manifest.get("identity")
    if (
        manifest.get("dataset_revisions") != MINI_TEACHER_DATASET_REVISIONS
        or not isinstance(identity, dict)
        or identity.get("datasets") != MINI_TEACHER_DATASET_REVISIONS
    ):
        raise ValueError("mini teacher dataset revisions do not match the pinned commits")

    selected_ids = manifest.get("selected_example_ids")
    if (
        manifest.get("selected_example_count") != MINI_TEACHER_ROW_COUNT
        or not isinstance(selected_ids, list)
        or len(selected_ids) != MINI_TEACHER_ROW_COUNT
        or len(set(selected_ids)) != MINI_TEACHER_ROW_COUNT
        or manifest.get("completed_example_count") != MINI_TEACHER_ROW_COUNT
        or manifest.get("failed_example_count") != 0
        or manifest.get("pending_example_count") != 0
        or manifest.get("imported_example_count") != 0
    ):
        raise ValueError("mini teacher manifest must account for 111 complete local rows")
    if len(rows) != MINI_TEACHER_ROW_COUNT:
        raise ValueError("mini teacher predictions must contain exactly 111 rows")
    row_ids = [row.get("example_id") for row in rows]
    if (
        any(not isinstance(example_id, str) or not example_id for example_id in row_ids)
        or len(set(row_ids)) != MINI_TEACHER_ROW_COUNT
        or set(row_ids) != set(selected_ids)
    ):
        raise ValueError("mini teacher prediction IDs must match the unique selected IDs")

    counts: dict[str, int] = defaultdict(int)
    math_subject_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        if row.get("status") != "complete":
            raise ValueError("every mini teacher prediction must be complete")
        source = row.get("source")
        if not isinstance(source, dict) or source.get("dataset") not in MINI_TEACHER_DATASET_COUNTS:
            raise ValueError("mini teacher prediction has an unsupported dataset")
        dataset = source["dataset"]
        if source.get("revision") != MINI_TEACHER_ROW_REVISIONS[dataset]:
            raise ValueError("mini teacher row dataset revision differs from the pinned run")
        counts[dataset] += 1
        if dataset == "math":
            math_subject_counts[source.get("config")] += 1
        samples = row.get("samples")
        if (
            row.get("requested_sample_count") != MINI_TEACHER_SAMPLE_COUNT
            or not isinstance(samples, list)
            or len(samples) != MINI_TEACHER_SAMPLE_COUNT
            or any(not isinstance(sample, dict) for sample in samples)
            or [sample.get("index") for sample in samples] != list(range(MINI_TEACHER_SAMPLE_COUNT))
        ):
            raise ValueError("each mini teacher prediction must contain eight indexed seeded samples")
        if any(
            sample.get("seed") != derive_sample_seed(MINI_TEACHER_SEED, row["example_id"], index)
            for index, sample in enumerate(samples)
        ):
            raise ValueError("mini teacher sample seeds do not match the run seed and sample indices")

    if dict(counts) != MINI_TEACHER_DATASET_COUNTS:
        raise ValueError("mini teacher dataset counts must be 40 GSM8K, 21 MATH, and 50 GSM-Plus")
    if dict(math_subject_counts) != {subject: 3 for subject in MINI_MATH_SUBJECTS}:
        raise ValueError("mini teacher run must contain three MATH rows for each of seven subjects")


def build_training_sets(rows: list[dict], gate_threshold: float) -> dict[str, list[dict]]:
    sets: dict[str, list[dict]] = {"kd_all": [], "kd_gated": [], "kd_oracle": []}
    for row in rows:
        consistency = row["self_consistency"]
        majority = consistency.get("majority_answer")
        share = consistency.get("majority_vote_share") or 0.0
        greedy = row["greedy"]
        responses = [
            {
                "text": greedy["text"],
                "answer": greedy.get("predicted_answer"),
                "truncated": greedy.get("was_truncated", False),
                "correct": greedy.get("correct"),
            }
        ]
        for sample in row["samples"]:
            responses.append(
                {
                    "text": sample["text"],
                    "answer": sample.get("answer"),
                    "truncated": sample.get("termination_reason") != "eos",
                    "correct": None,
                }
            )
        for response in responses:
            if response["truncated"] or response["answer"] is None:
                continue
            item = {"example_id": row["example_id"], "question": row["question"],
                    "response": response["text"], "answer": response["answer"],
                    "gold": row.get("gold_answer"), "correct": response["correct"]}
            sets["kd_all"].append(item)
            if majority is not None and share >= gate_threshold and response["answer"] == majority:
                sets["kd_gated"].append(item)
    return sets


def grade_oracle(items: list[dict], worker: MathVerifyWorker) -> list[dict]:
    kept = []
    for item in items:
        correct = item["correct"]
        if correct is None:
            correct = grade_prediction(item["response"], item["gold"], worker=worker).correct
        if correct:
            kept.append(item)
    return kept


def select_eval(examples, training_ids: set[str], training_groups: set[str], args) -> list:
    limits = {"gsm8k": args.eval_gsm8k, "gsm_plus": args.eval_gsm_plus}
    by_bucket: dict[tuple[str, str | None], list] = defaultdict(list)
    for example in examples:
        if example.example_id in training_ids or not example.scorable:
            continue
        group = _seed_group(example)
        if group is not None and group in training_groups:
            continue
        bucket = (example.dataset, example.source_config if example.dataset == "math" else None)
        by_bucket[bucket].append(example)
    selected = []
    for (dataset, _config), items in sorted(by_bucket.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")):
        items.sort(key=lambda e: _rank(e.example_id, args.eval_seed))
        if dataset == "gsm_plus":
            seen, unique = set(), []
            for item in items:
                if item.seed_question_id not in seen:
                    seen.add(item.seed_question_id)
                    unique.append(item)
            items = unique
        limit = args.eval_math_per_subject if dataset == "math" else limits[dataset]
        selected.extend(items[:limit])
    return selected


def cut_after_first_answer(text: str) -> str:
    """Keep text through the first closed boxed answer; applied to every condition."""
    marker = r"\boxed{"
    start = text.find(marker)
    if start < 0:
        return text
    close = _matching_brace(text, start + len(marker) - 1)
    return text if close is None else text[: close + 1]


def prompt_ids(tokenizer, question: str) -> list[int]:
    return tokenizer.apply_chat_template(
        build_chat_messages(question), add_generation_prompt=True, tokenize=True
    )


def train_lora(model_dir: str, items: list[dict], tokenizer, args, log) -> torch.nn.Module:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.train_seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, revision=STUDENT_REVISION, torch_dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    # The base student never trained <|im_end|>, and LoRA cannot move its
    # embedding, so responses end with the base model's own EOS token.
    end_id = tokenizer.eos_token_id
    encoded = []
    for item in items:
        prompt = prompt_ids(tokenizer, item["question"])
        response = tokenizer(item["response"], add_special_tokens=False)["input_ids"] + [end_id]
        ids = (prompt + response)[: args.max_train_tokens]
        labels = ([-100] * len(prompt) + response)[: args.max_train_tokens]
        encoded.append((ids, labels))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = math.ceil(len(encoded) / (args.micro_batch * args.grad_accum))
    total_steps = steps_per_epoch * args.epochs
    warmup = max(1, int(0.05 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / warmup)
        * max(0.0, 0.5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps))),
    )
    rng = random.Random(args.train_seed)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_id
    model.train()
    step = 0
    for epoch in range(args.epochs):
        order = list(range(len(encoded)))
        rng.shuffle(order)
        batches = [order[i:i + args.micro_batch] for i in range(0, len(order), args.micro_batch)]
        running = []
        for index, batch in enumerate(batches):
            width = max(len(encoded[i][0]) for i in batch)
            input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, i in enumerate(batch):
                ids, labs = encoded[i]
                input_ids[row, : len(ids)] = torch.tensor(ids)
                labels[row, : len(labs)] = torch.tensor(labs)
                mask[row, : len(ids)] = 1
            out = model(input_ids=input_ids.cuda(), attention_mask=mask.cuda(), labels=labels.cuda())
            (out.loss / args.grad_accum).backward()
            running.append(out.loss.item())
            if (index + 1) % args.grad_accum == 0 or index + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == total_steps:
                    log(f"  epoch {epoch + 1} step {step}/{total_steps} loss {sum(running) / len(running):.4f}")
                    running = []
    model = model.merge_and_unload()
    model.config.use_cache = True
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, tokenizer, eval_examples, worker, args, log) -> list[dict]:
    tokenizer.padding_side = "left"
    end_ids = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_ids[1]
    order = sorted(range(len(eval_examples)), key=lambda i: len(eval_examples[i].question))
    texts: dict[int, tuple[str, bool]] = {}
    for start in range(0, len(order), args.eval_batch):
        chunk = order[start:start + args.eval_batch]
        prompts = [prompt_ids(tokenizer, eval_examples[i].question) for i in chunk]
        width = max(map(len, prompts))
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, ids in enumerate(prompts):
            input_ids[row, width - len(ids):] = torch.tensor(ids)
            mask[row, width - len(ids):] = 1
        output = model.generate(
            input_ids=input_ids.cuda(), attention_mask=mask.cuda(), do_sample=False,
            max_new_tokens=args.eval_max_new_tokens, eos_token_id=end_ids, pad_token_id=pad_id,
        )
        for row, i in enumerate(chunk):
            generated = output[row, width:].tolist()
            finished = any(t in end_ids for t in generated)
            if finished:
                generated = generated[: min(generated.index(t) for t in end_ids if t in generated)]
            texts[i] = (tokenizer.decode(generated, skip_special_tokens=True), not finished)
        log(f"  generated {min(start + args.eval_batch, len(order))}/{len(order)}")
    results = []
    for i, example in enumerate(eval_examples):
        text, truncated = texts[i]
        outcome = grade_prediction(cut_after_first_answer(text), example.gold_answer, worker=worker)
        results.append({
            "example_id": example.example_id, "dataset": example.dataset,
            "correct": bool(outcome.correct), "grading_status": outcome.status,
            "truncated": truncated, "response": text,
        })
    return results


def summarize(results: list[dict]) -> dict:
    by_dataset: dict[str, list[bool]] = defaultdict(list)
    for item in results:
        by_dataset[item["dataset"]].append(item["correct"])
    summary = {d: {"n": len(v), "accuracy": sum(v) / len(v)} for d, v in sorted(by_dataset.items())}
    macro = sum(s["accuracy"] for s in summary.values()) / len(summary)
    summary["overall"] = {"n": len(results), "accuracy": sum(r["correct"] for r in results) / len(results),
                          "macro_accuracy": macro,
                          "truncated_fraction": sum(r["truncated"] for r in results) / len(results)}
    return summary


def paired_bootstrap(a: list[dict], b: list[dict], seed: int, rounds: int = 5000) -> dict:
    left = {r["example_id"]: r["correct"] for r in a}
    right = {r["example_id"]: r["correct"] for r in b}
    ids = sorted(left)
    diffs = [int(right[i]) - int(left[i]) for i in ids]
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(diffs) for _ in ids) / len(ids) for _ in range(rounds))
    return {"delta": sum(diffs) / len(ids), "ci95": [draws[int(0.025 * rounds)], draws[int(0.975 * rounds) - 1]]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--gate-threshold", type=float, default=0.75)
    parser.add_argument("--eval-gsm8k", type=int, default=300)
    parser.add_argument("--eval-math-per-subject", type=int, default=20)
    parser.add_argument("--eval-gsm-plus", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=43)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--eval-max-new-tokens", type=int, default=512)
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--micro-batch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-train-tokens", type=int, default=1400)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "distill.log"

    def log(message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    manifest = json.loads((args.teacher_run / "run_manifest.json").read_text(encoding="utf-8"))
    rows = load_training_rows(args.teacher_run / "predictions.jsonl")
    log(f"teacher run {manifest['run_id']}: {len(rows)} complete rows")
    worker = MathVerifyWorker()
    try:
        sets = build_training_sets(rows, args.gate_threshold)
        sets["kd_oracle"] = grade_oracle(sets["kd_all"], worker)
        sizes = {name: {"responses": len(items), "questions": len({i["example_id"] for i in items})}
                 for name, items in sets.items()}
        log(f"training sets: {json.dumps(sizes)}")

        # An uncapped pool profile keeps every source test row as held-out candidates.
        examples, _ = _load_cached_examples(HELDOUT_POOL, 42, manifest["dataset_revisions"])
        training_ids = {row["example_id"] for row in rows}
        by_id = {e.example_id: e for e in examples}
        training_groups = {g for g in (_seed_group(by_id[i]) for i in training_ids if i in by_id) if g}
        eval_examples = select_eval(examples, training_ids, training_groups, args)
        counts = defaultdict(int)
        for example in eval_examples:
            counts[example.dataset] += 1
        log(f"held-out eval: {len(eval_examples)} questions {dict(counts)}")

        tokenizer = AutoTokenizer.from_pretrained(STUDENT, revision=STUDENT_REVISION, local_files_only=True)
        results_path = args.out_dir / "results.json"
        all_results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
        for condition in args.conditions:
            if condition in all_results:
                log(f"{condition}: already evaluated, skipping")
                continue
            started = time.time()
            log(f"{condition}: start")
            if condition == "base":
                from transformers import AutoModelForCausalLM
                model = AutoModelForCausalLM.from_pretrained(
                    STUDENT, revision=STUDENT_REVISION, torch_dtype=torch.bfloat16, local_files_only=True
                ).to("cuda").eval()
            else:
                tokenizer.padding_side = "right"
                model = train_lora(STUDENT, sets[condition], tokenizer, args, log)
            train_seconds = time.time() - started
            results = evaluate(model, tokenizer, eval_examples, worker, args, log)
            del model
            torch.cuda.empty_cache()
            summary = summarize(results)
            all_results[condition] = {
                "summary": summary, "train_seconds": train_seconds,
                "total_seconds": time.time() - started,
                "training_set": sizes.get(condition),
            }
            with (args.out_dir / f"eval_{condition}.jsonl").open("w", encoding="utf-8") as handle:
                for item in results:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
            log(f"{condition}: {json.dumps(summary)}")
    finally:
        worker.close()

    loaded = {c: [json.loads(l) for l in (args.out_dir / f"eval_{c}.jsonl").open(encoding="utf-8")]
              for c in CONDITIONS if (args.out_dir / f"eval_{c}.jsonl").exists()}
    comparisons = {}
    for left, right in [("base", "kd_all"), ("base", "kd_gated"), ("kd_all", "kd_gated"), ("kd_all", "kd_oracle"),
                        ("kd_gated", "kd_oracle")]:
        if left in loaded and right in loaded:
            comparisons[f"{right}_minus_{left}"] = paired_bootstrap(loaded[left], loaded[right], args.eval_seed)
    all_results["_comparisons"] = comparisons
    all_results["_config"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    all_results["_config"].update({"student": STUDENT, "student_revision": STUDENT_REVISION,
                                   "teacher_run": manifest["run_id"], "training_sets": sizes})
    results_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    log(f"comparisons: {json.dumps(comparisons)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
