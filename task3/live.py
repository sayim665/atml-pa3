"""Step 4: single long response vs. parallel sampling vs. sequential refinement on 50 MATH questions.

    python -m task3.live --shard 0 --num-shards 2      (one shard per GPU; writes outputs/task3/live/shard<k>.jsonl)
    python -m task3.live --summarize                   (grades all shards; writes results/task3/live/)

Qwen2.5-3B-Instruct (pinned revision), FP16, no quantization, batch size 1, temperature 0.7, top-p 0.8, top-k 20,
repetition penalty 1.0, native chat template. Every call is seeded independently; timings exclude model loading.
"""

import argparse
import json
import time

import numpy as np

from task1.common import append_jsonl, environment_info, load_json, read_jsonl, save_json
from task3.answers import MathEquivalence, grade_math, group_math_answers, last_boxed, scorer_versions
from task3.config import (LIVE_MODEL, LIVE_PARALLEL_SAMPLES, LIVE_PARALLEL_TOKENS, LIVE_REVIEW, LIVE_REVISION,
                          LIVE_REVISION_TOKENS, LIVE_SAMPLING, LIVE_SINGLE_TOKENS, LIVE_SUFFIX, LIVE_SYSTEM, SEED,
                          SMOKE_TEST, output_root, results_root)
from task3.selection import self_consistency

STRATEGY_CALLS = {"single": 0, "parallel": 1, "revision": 9}


def call_seed(question_index, call_index):
    return SEED * 1000 + question_index * 10 + call_index


def user_message(question):
    return question + "\n\n" + LIVE_SUFFIX


class Generator:
    def __init__(self, device):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = LIVE_MODEL if not SMOKE_TEST else "Qwen/Qwen3-0.6B"
        revision = LIVE_REVISION if not SMOKE_TEST else None
        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, dtype=dtype).to(device)
        self.model.eval()
        self.end_ids = [self.tokenizer.convert_tokens_to_ids("<|im_end|>"), self.tokenizer.eos_token_id]

    def generate(self, messages, max_new_tokens, seed):
        torch = self.torch
        from transformers import set_seed

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        set_seed(seed)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                         eos_token_id=self.end_ids, pad_token_id=self.tokenizer.pad_token_id,
                                         **LIVE_SAMPLING)
        new_tokens = output[0, inputs["input_ids"].shape[1]:].tolist()
        finished = bool(new_tokens) and new_tokens[-1] in self.end_ids
        retained = new_tokens[:-1] if finished else new_tokens
        text = self.tokenizer.decode(retained, skip_special_tokens=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return {"prompt": prompt, "response": text, "seed": seed, "input_tokens": int(inputs["input_ids"].shape[1]),
                "output_tokens": len(retained), "truncated": not finished and len(new_tokens) >= max_new_tokens,
                "seconds": time.time() - started, "max_new_tokens": max_new_tokens,
                "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None,
                "extracted": last_boxed(text)}


def live_questions():
    """First 50 held-out MATH questions of the seeded split (built from generations.parquet if not saved yet)."""
    path = results_root() / "splits" / "math_live_questions.json"
    if not path.exists():
        import pyarrow.parquet as pq

        from task3.config import DATASETS, LIVE_QUESTIONS, data_root
        from task3.data import splits

        table = pq.read_table(data_root() / DATASETS["math"]["folder"] / "generations.parquet",
                              columns=["sample_id", "rollout_id", "question_text", "gold_answer"])
        first = {}
        for sample, rollout, question, gold in zip(*(table.column(name).to_pylist() for name in table.column_names)):
            if rollout == 0:
                first[sample] = {"id": sample, "question": question, "gold": gold}
        chosen = splits(list(first))["eval"][:LIVE_QUESTIONS]
        save_json(path, [first[question] for question in chosen])
    return load_json(path)


def run_shard(shard, num_shards):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = output_root() / "live" / f"shard{shard}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {(row["question_index"], row["strategy"], row["call"]) for row in read_jsonl(path)}
    questions = live_questions()
    if SMOKE_TEST:
        questions = questions[:2]
    generator = Generator(device)
    scale = 1 if not SMOKE_TEST else 64
    for question_index, item in enumerate(questions):
        if question_index % num_shards != shard:
            continue
        base = [{"role": "system", "content": LIVE_SYSTEM}, {"role": "user", "content": user_message(item["question"])}]
        plan = [("single", 0, base, LIVE_SINGLE_TOKENS // scale)]
        plan += [("parallel", index, base, LIVE_PARALLEL_TOKENS // scale) for index in range(LIVE_PARALLEL_SAMPLES)]
        for strategy, call, messages, cap in plan:
            if (question_index, strategy, call) in done:
                continue
            result = generator.generate(messages, cap, call_seed(question_index, STRATEGY_CALLS[strategy] + call))
            append_jsonl(path, {"question_index": question_index, "id": item["id"], "strategy": strategy,
                                "call": call, **result})
        if (question_index, "revision", 0) not in done:
            initial = next(row for row in read_jsonl(path) if row["question_index"] == question_index
                           and row["strategy"] == "parallel" and row["call"] == 0)
            messages = base + [{"role": "assistant", "content": initial["response"]},
                               {"role": "user", "content": LIVE_REVIEW}]
            result = generator.generate(messages, LIVE_REVISION_TOKENS // scale,
                                        call_seed(question_index, STRATEGY_CALLS["revision"]))
            append_jsonl(path, {"question_index": question_index, "id": item["id"], "strategy": "revision",
                                "call": 0, **result})
        print(f"[shard {shard}] question {question_index} done", flush=True)
    save_json(output_root() / "live" / f"shard{shard}_environment.json",
              {"environment": environment_info(), "model": LIVE_MODEL, "revision": LIVE_REVISION,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})


def summarize():
    rows = []
    for path in sorted((output_root() / "live").glob("shard*.jsonl")):
        rows.extend(read_jsonl(path))
    questions = live_questions()
    equivalence = MathEquivalence()
    by_question = {}
    for row in rows:
        by_question.setdefault(row["question_index"], {}).setdefault(row["strategy"], {})[row["call"]] = row
    per_question = []
    for question_index, strategies in sorted(by_question.items()):
        gold = questions[question_index]["gold"]
        single = strategies["single"][0]
        parallel = [strategies["parallel"][index] for index in range(LIVE_PARALLEL_SAMPLES)]
        revision = strategies["revision"][0]
        answers = [single["extracted"]] + [call["extracted"] for call in parallel] + [revision["extracted"]]
        groups = group_math_answers(answers, equivalence)
        correct = grade_math(groups, gold, equivalence)
        parallel_groups = groups[1:1 + LIVE_PARALLEL_SAMPLES]
        parallel_answer = self_consistency(parallel_groups)[0]
        initial_correct = correct[parallel_groups[0]]
        revised_correct = correct[groups[-1]]
        per_question.append({
            "question_index": question_index, "id": questions[question_index]["id"], "gold": gold,
            "single": {"answer": groups[0], "correct": correct[groups[0]], "output_tokens": single["output_tokens"],
                       "input_tokens": single["input_tokens"], "seconds": single["seconds"],
                       "truncated": single["truncated"], "extracted": single["extracted"] is not None},
            "parallel": {"answers": parallel_groups, "answer": parallel_answer, "correct": correct[parallel_answer],
                         "output_tokens": sum(call["output_tokens"] for call in parallel),
                         "input_tokens": sum(call["input_tokens"] for call in parallel),
                         "seconds": sum(call["seconds"] for call in parallel),
                         "truncated": sum(call["truncated"] for call in parallel),
                         "extracted": sum(call["extracted"] is not None for call in parallel)},
            "refinement": {"initial_answer": parallel_groups[0], "answer": groups[-1], "correct": revised_correct,
                           "initial_correct": initial_correct,
                           "transition": f"{'correct' if initial_correct else 'wrong'}_to_"
                                         f"{'correct' if revised_correct else 'wrong'}",
                           "answer_changed": groups[-1] != parallel_groups[0],
                           "output_tokens": parallel[0]["output_tokens"] + revision["output_tokens"],
                           "input_tokens": parallel[0]["input_tokens"] + revision["input_tokens"],
                           "seconds": parallel[0]["seconds"] + revision["seconds"],
                           "initial_truncated": parallel[0]["truncated"], "revision_truncated": revision["truncated"],
                           "revision_extracted": revision["extracted"] is not None,
                           "revision_input_tokens": revision["input_tokens"],
                           "revision_output_tokens": revision["output_tokens"]},
        })

    def mean(strategy, field):
        return float(np.mean([float(row[strategy][field]) for row in per_question]))

    count = len(per_question)
    summary = {
        "questions": count,
        "single": {"accuracy": mean("single", "correct"), "mean_output_tokens": mean("single", "output_tokens"),
                   "mean_input_tokens": mean("single", "input_tokens"), "mean_seconds": mean("single", "seconds"),
                   "truncation_rate": mean("single", "truncated"), "extraction_rate": mean("single", "extracted")},
        "parallel": {"accuracy": mean("parallel", "correct"), "mean_output_tokens": mean("parallel", "output_tokens"),
                     "mean_input_tokens": mean("parallel", "input_tokens"), "mean_seconds": mean("parallel", "seconds"),
                     "truncation_rate_per_response": mean("parallel", "truncated") / LIVE_PARALLEL_SAMPLES,
                     "extraction_rate_per_response": mean("parallel", "extracted") / LIVE_PARALLEL_SAMPLES,
                     "first_response_accuracy": float(np.mean([row["refinement"]["initial_correct"]
                                                               for row in per_question]))},
        "refinement": {"accuracy": mean("refinement", "correct"),
                       "initial_accuracy": mean("refinement", "initial_correct"),
                       "mean_output_tokens": mean("refinement", "output_tokens"),
                       "mean_input_tokens": mean("refinement", "input_tokens"),
                       "mean_seconds": mean("refinement", "seconds"),
                       "mean_revision_input_tokens": mean("refinement", "revision_input_tokens"),
                       "mean_revision_output_tokens": mean("refinement", "revision_output_tokens"),
                       "initial_truncation_rate": mean("refinement", "initial_truncated"),
                       "revision_truncation_rate": mean("refinement", "revision_truncated"),
                       "revision_extraction_rate": mean("refinement", "revision_extracted"),
                       "answer_changed_rate": mean("refinement", "answer_changed"),
                       "transitions": {name: sum(row["refinement"]["transition"] == name for row in per_question)
                                       for name in ("wrong_to_correct", "correct_to_wrong", "correct_to_correct",
                                                    "wrong_to_wrong")}},
        "peak_gpu_memory_gb": max((row["peak_gpu_memory_gb"] or 0) for row in rows),
        "scorer": scorer_versions(),
    }
    output = results_root() / "live"
    save_json(output / "summary.json", summary)
    save_json(output / "per_question.json", per_question)
    with open(output / "generations.jsonl", "w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda row: (row["question_index"], row["strategy"], row["call"])):
            handle.write(json.dumps(row) + "\n")
    for path in sorted((output_root() / "live").glob("shard*_environment.json")):
        save_json(output / path.name, load_json(path))
    print(json.dumps({key: value for key, value in summary.items() if key != "scorer"}, indent=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    if args.summarize:
        summarize()
    else:
        run_shard(args.shard, args.num_shards)


if __name__ == "__main__":
    main()
