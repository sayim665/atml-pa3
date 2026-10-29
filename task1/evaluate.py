"""Benchmark + qualitative evaluation for one of the seven Task 1 conditions.

    python -m task1.evaluate --condition baseline   (baseline | full_ft_mid | full_ft | l2sp | lora4 | lora16 | oplora16)
"""

import argparse
import gzip
import json
import time
from pathlib import Path

import torch
import yaml

from task1.common import (REPO_ROOT, chat_prompt_text, environment_info, force_non_thinking_template, load_causal_lm,
                          load_json, load_tokenizer, output_root, results_root, run_dir, save_json, seed_everything)
from task1.config import (EVAL_CONDITIONS, GSM8K_MAX_NEW_TOKENS, MODEL_NAME, QUALITATIVE_MAX_NEW_TOKENS, RUNS, SEED)
from task1.data import build_probe_set
from task1.lora import attach_frozen_adapters, load_adapter_factors, merge_adapter_factors

QUALITATIVE_FILE = REPO_ROOT / "task1" / "data" / "first_plot_questions.yaml"
GSM8K_TASK = "gsm8k_cot"
LIKELIHOOD_TASKS = ("hellaswag", "boolq", "arc_easy")
MERGE_CHECK_PROMPTS = (
    "What is 17 times 23?",
    "hey I feel bored",
    "Name three primary colors.",
    "A train travels 60 miles in 1.5 hours. What is its average speed?",
)


def condition_source(condition):
    if condition == "baseline":
        return {"kind": "base", "path": MODEL_NAME}
    if condition == "full_ft_mid":
        return {"kind": "full", "path": str(run_dir("full_ft") / "midpoint")}
    run = RUNS[condition]
    if run.is_full_ft():
        return {"kind": "full", "path": str(run_dir(condition) / "final")}
    return {"kind": "adapter", "path": str(run_dir(condition) / "final" / "adapter_factors.safetensors")}


@torch.no_grad()
def prompt_logits(model, tokenizer, device):
    texts = [chat_prompt_text(tokenizer, prompt) for prompt in MERGE_CHECK_PROMPTS]
    tokenizer.padding_side = "right"
    batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    logits = model(**batch).logits.float()
    mask = batch["attention_mask"].bool()
    return logits[mask]


@torch.no_grad()
def merged_versus_unmerged(factors, scaling, tokenizer, device):
    unmerged = load_causal_lm(MODEL_NAME, torch.float16, device).eval()
    attach_frozen_adapters(unmerged, factors, scaling)
    unmerged_logits = prompt_logits(unmerged, tokenizer, device)
    del unmerged

    merged = load_causal_lm(MODEL_NAME, torch.float16, device).eval()
    merge_adapter_factors(merged, factors, scaling)
    merged_logits = prompt_logits(merged, tokenizer, device)

    difference = (merged_logits - unmerged_logits).abs()
    check = {
        "prompts": list(MERGE_CHECK_PROMPTS),
        "max_abs_logit_diff": difference.max().item(),
        "mean_abs_logit_diff": difference.mean().item(),
        "logit_scale_mean_abs": unmerged_logits.abs().mean().item(),
        "argmax_agreement": (merged_logits.argmax(-1) == unmerged_logits.argmax(-1)).float().mean().item(),
    }
    return merged, check


def load_condition_model(condition, tokenizer, device):
    source = condition_source(condition)
    info = {"source": source}
    if source["kind"] == "adapter":
        factors, scaling = load_adapter_factors(source["path"])
        model, check = merged_versus_unmerged(factors, scaling, tokenizer, device)
        info["merge_check"] = check
    else:
        model = load_causal_lm(source["path"], torch.float16, device)
    model.eval()
    model.config.use_cache = True
    return model, info


def run_lm_eval(model, tokenizer, tasks, apply_chat_template, batch_size, gen_kwargs=None, limit=None):
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    tokenizer.padding_side = "left"
    language_model = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    return simple_evaluate(
        model=language_model,
        tasks=list(tasks),
        num_fewshot=0,
        apply_chat_template=apply_chat_template,
        gen_kwargs=gen_kwargs,
        log_samples=True,
        limit=limit,
        random_seed=SEED,
        numpy_random_seed=SEED,
        torch_random_seed=SEED,
        fewshot_random_seed=SEED,
    )


def first_response(sample):
    response = sample["resps"][0]
    while isinstance(response, (list, tuple)):
        response = response[0]
    return response


def compact_gsm8k_samples(samples, tokenizer):
    by_doc = {}
    for sample in samples:
        doc_id = sample["doc_id"]
        entry = by_doc.setdefault(doc_id, {"doc_id": doc_id, "target": sample.get("target")})
        response = first_response(sample)
        if "response" not in entry:
            token_count = len(tokenizer(response, add_special_tokens=False)["input_ids"])
            entry["response"] = response
            entry["response_tokens"] = token_count
            entry["truncated"] = token_count >= GSM8K_MAX_NEW_TOKENS - 1
        filter_name = sample.get("filter", "none")
        extracted = sample.get("filtered_resps", [None])[0]
        entry[f"extracted[{filter_name}]"] = extracted
        entry[f"exact_match[{filter_name}]"] = sample.get("exact_match")
        entry[f"unextractable[{filter_name}]"] = extracted in (None, "", "[invalid]")
    return [by_doc[key] for key in sorted(by_doc)]


def compact_likelihood_samples(samples):
    rows = []
    for sample in samples:
        row = {"doc_id": sample["doc_id"], "target": sample.get("target")}
        for metric in ("acc", "acc_norm"):
            if metric in sample:
                row[metric] = sample[metric]
        row["loglikelihoods"] = [float(item[0]) for item in sample.get("filtered_resps", [])]
        rows.append(row)
    return sorted(rows, key=lambda row: row["doc_id"])


def prompt_of(sample):
    arguments = sample["arguments"]
    if isinstance(arguments, dict):
        first = arguments.get("gen_args_0", next(iter(arguments.values())))
        return first.get("arg_0")
    return arguments[0][0]


def save_jsonl_gz(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def save_raw_samples(path, samples):
    save_jsonl_gz(path, samples)


def strip_samples(results):
    return {key: value for key, value in results.items() if key != "samples"}


def score_summary(gsm_results, likelihood_results):
    summary = {}
    gsm = gsm_results["results"][GSM8K_TASK]
    summary["gsm8k"] = {key: value for key, value in gsm.items() if isinstance(value, (int, float))}
    for task in LIKELIHOOD_TASKS:
        task_result = likelihood_results["results"][task]
        summary[task] = {key: value for key, value in task_result.items() if isinstance(value, (int, float))}
    return summary


def qualitative_prompts():
    with open(QUALITATIVE_FILE, "r", encoding="utf-8") as handle:
        entries = yaml.safe_load(handle)
    prompts = []
    for entry in entries:
        identifier = entry["id"]
        if identifier.endswith("_json"):
            question_id, prompt_format = identifier[: -len("_json")], "json"
        elif identifier.endswith("_template"):
            question_id, prompt_format = identifier[: -len("_template")], "template"
        else:
            question_id, prompt_format = identifier, "plain"
        prompts.append({
            "id": identifier,
            "question_id": question_id,
            "format": prompt_format,
            "system": entry.get("system"),
            "user": entry["paraphrases"][0],
        })
    return prompts


@torch.no_grad()
def generate_qualitative(model, tokenizer, device, batch_size=8):
    prompts = qualitative_prompts()
    tokenizer.padding_side = "left"
    outputs = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        texts = [chat_prompt_text(tokenizer, item["user"], item["system"]) for item in chunk]
        batch = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        generated = model.generate(**batch, max_new_tokens=QUALITATIVE_MAX_NEW_TOKENS, do_sample=False,
                                   temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tokenizer.pad_token_id)
        new_tokens = generated[:, batch["input_ids"].shape[1]:]
        for item, tokens in zip(chunk, new_tokens):
            token_list = tokens.tolist()
            finished = tokenizer.convert_tokens_to_ids("<|im_end|>") in token_list or \
                tokenizer.eos_token_id in token_list
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            content_tokens = sum(1 for token in token_list if token != tokenizer.pad_token_id)
            outputs.append({**item, "response": text, "new_tokens": content_tokens, "truncated": not finished})
    return outputs


def evaluate(condition, device=None, gen_batch_size=64, likelihood_batch_size=32, skip_benchmarks=False, limit=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(SEED)
    raw_dir = output_root() / "eval" / condition
    result_dir = results_root() / "eval" / condition
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = force_non_thinking_template(load_tokenizer())
    load_start = time.time()
    model, info = load_condition_model(condition, tokenizer, device)
    info["load_seconds"] = time.time() - load_start
    info["environment"] = environment_info()
    if "merge_check" in info:
        save_json(result_dir / "merge_check.json", info["merge_check"])

    timings = {}
    if not skip_benchmarks:
        start = time.time()
        gsm_results = run_lm_eval(model, tokenizer, [GSM8K_TASK], True, gen_batch_size,
                                  gen_kwargs=f"max_gen_toks={GSM8K_MAX_NEW_TOKENS},do_sample=False", limit=limit)
        timings["gsm8k_seconds"] = time.time() - start
        gsm_samples = gsm_results["samples"][GSM8K_TASK]
        first_prompt = prompt_of(gsm_samples[0])
        info["gsm8k_prompt_example"] = first_prompt
        info["gsm8k_non_thinking_prompt"] = "<think>\n\n</think>" in first_prompt

        start = time.time()
        likelihood_results = run_lm_eval(model, tokenizer, LIKELIHOOD_TASKS, False, likelihood_batch_size,
                                         limit=limit)
        timings["likelihood_seconds"] = time.time() - start

        save_raw_samples(raw_dir / "samples_gsm8k.jsonl.gz", gsm_samples)
        for task in LIKELIHOOD_TASKS:
            save_raw_samples(raw_dir / f"samples_{task}.jsonl.gz", likelihood_results["samples"][task])
        save_json(raw_dir / "lm_eval_gsm8k.json", strip_samples(gsm_results))
        save_json(raw_dir / "lm_eval_likelihood.json", strip_samples(likelihood_results))

        gsm_rows = compact_gsm8k_samples(gsm_samples, tokenizer)
        save_jsonl_gz(result_dir / "per_example_gsm8k.jsonl.gz", gsm_rows)
        for task in LIKELIHOOD_TASKS:
            save_jsonl_gz(result_dir / f"per_example_{task}.jsonl.gz",
                          compact_likelihood_samples(likelihood_results["samples"][task]))

        scores = score_summary(gsm_results, likelihood_results)
        scores["gsm8k"]["truncation_rate"] = sum(row["truncated"] for row in gsm_rows) / len(gsm_rows)
        scores["gsm8k"]["mean_response_tokens"] = sum(row["response_tokens"] for row in gsm_rows) / len(gsm_rows)
        for filter_name in ("flexible-extract", "strict-match"):
            key = f"unextractable[{filter_name}]"
            if key in gsm_rows[0]:
                scores["gsm8k"][f"unextractable_rate[{filter_name}]"] = \
                    sum(row[key] for row in gsm_rows) / len(gsm_rows)
        save_json(result_dir / "scores.json", scores)

        if condition == "baseline" and limit is None:
            save_probe_prompts(gsm_samples, likelihood_results["samples"]["hellaswag"])

    start = time.time()
    qualitative = generate_qualitative(model, tokenizer, device)
    timings["qualitative_seconds"] = time.time() - start
    save_json(result_dir / "qualitative.json", qualitative)

    info["timings"] = timings
    if device.startswith("cuda"):
        info["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
    save_json(result_dir / "eval_info.json", info)
    (raw_dir / "DONE").write_text("ok")
    return info


def save_probe_prompts(gsm_samples, hellaswag_samples):
    """Store the exact benchmark-format prompts for the fixed probe IDs (used by diagnostics)."""
    probe = build_probe_set()
    gsm_by_doc = {sample["doc_id"]: prompt_of(sample) for sample in gsm_samples}
    hellaswag_by_doc = {sample["doc_id"]: prompt_of(sample) for sample in hellaswag_samples}
    prompts = {
        "gsm8k": [{"doc_id": doc_id, "prompt": gsm_by_doc[doc_id]} for doc_id in probe["gsm8k_test_doc_ids"]],
        "hellaswag": [{"doc_id": doc_id, "prompt": hellaswag_by_doc[doc_id]}
                      for doc_id in probe["hellaswag_validation_doc_ids"]],
    }
    save_json(results_root() / "splits" / "probe_prompts.json", prompts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=EVAL_CONDITIONS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gen-batch-size", type=int, default=64)
    parser.add_argument("--likelihood-batch-size", type=int, default=32)
    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="documents per task, for smoke tests only")
    args = parser.parse_args()
    evaluate(args.condition, args.device, args.gen_batch_size, args.likelihood_batch_size, args.skip_benchmarks,
             args.limit)


if __name__ == "__main__":
    main()
