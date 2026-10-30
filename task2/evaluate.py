"""Development metrics (teacher-forced) and final-test evaluation under four image conditions.

    python -m task2.evaluate --run vlm196

Final evaluation: greedy decoding, thinking disabled, 16 new tokens, batch 8, on both final-test suites with the
correct image, the complementary partner image, a fixed mismatched image from another pair, and no image.
"""

import argparse
import gzip
import json
import time

import numpy as np
import torch

from task1.common import environment_info, save_json
from task2.config import (BOOTSTRAP_RESAMPLES, EVAL_BATCH, GENERATION_BATCH, GENERATION_MAX_NEW_TOKENS, RUNS, SEED,
                          results_root, run_dir)
from task2.model import collate_generation, collate_training
from task2.vqa_eval import exact_match, normalize_answer, soft_accuracy

CONDITIONS = ("correct", "complementary", "mismatched", "none")
DEV_CONDITIONS = ("correct", "complementary", "none")
QUALITATIVE_CAPTIONS = 8
QUALITATIVE_PAIRS_PER_TYPE = 3


def question_type(example):
    return example["answer_type"] if example["kind"] == "vqa" else example["kind"]


def mismatch_mapping(pairs, seed=SEED):
    """Pair p borrows the images of pair sigma(p) != p; fixed before generation and reused everywhere."""
    order = np.random.default_rng(seed).permutation(len(pairs))
    mapping = {}
    for position, pair_index in enumerate(order):
        mapping[int(pair_index)] = int(order[(position + 1) % len(order)])
    return mapping


def condition_items(pairs, condition, mapping=None):
    items = []
    for pair_index, pair in enumerate(pairs):
        for side, example in enumerate(pair["examples"]):
            if condition == "correct":
                image = example["image"]
            elif condition == "complementary":
                image = pair["examples"][1 - side]["image"]
            elif condition == "mismatched":
                image = pairs[mapping[pair_index]]["examples"][side]["image"]
            else:
                image = None
            items.append({"example": example, "image": image, "pair_index": pair_index, "side": side})
    return items


def batches(items, size):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def batch_features(features, batch):
    images = [item["image"] for item in batch]
    if images[0] is None:
        return None
    return features.get(images)


@torch.no_grad()
def reference_nll(model, items, features, include_end=False):
    """Mean token NLL of each original reference (termination token excluded unless include_end)."""
    values = []
    use_cuda = torch.cuda.is_available()
    for batch in batches(items, EVAL_BATCH):
        with_image = batch[0]["image"] is not None
        input_ids, labels, attention_mask = collate_training(model.encoder, [item["example"] for item in batch],
                                                             with_image=with_image, include_end=include_end)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            losses = model.example_losses(input_ids, labels, attention_mask, batch_features(features, batch))
        values.extend(losses.float().cpu().tolist())
    return values


@torch.no_grad()
def generate_answers(model, items, features, max_new_tokens=GENERATION_MAX_NEW_TOKENS):
    answers, seconds = [], []
    use_cuda = torch.cuda.is_available()
    for batch in batches(items, GENERATION_BATCH):
        with_image = batch[0]["image"] is not None
        input_ids, attention_mask = collate_generation(model.encoder, [item["example"] for item in batch],
                                                       with_image=with_image)
        if use_cuda:
            torch.cuda.synchronize()
        started = time.time()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
            texts = model.generate(input_ids, attention_mask, batch_features(features, batch), max_new_tokens)
        if use_cuda:
            torch.cuda.synchronize()
        elapsed = time.time() - started
        answers.extend(texts)
        seconds.extend([elapsed / len(batch)] * len(batch))
    return answers, seconds


def development_metrics(model, manifest, features):
    was_training = model.training
    model.eval()
    started = time.time()
    record = {}
    captions = [{"example": example, "image": example["image"]} for example in manifest["dev"]["caption"]]
    record["caption_loss"] = float(np.mean(reference_nll(model, captions, features, include_end=True)))
    summary = [f"cap={record['caption_loss']:.3f}"]
    for suite in ("grounding", "vqa"):
        pairs = manifest["dev"][suite]
        suite_record = {}
        per_condition = {}
        for condition in DEV_CONDITIONS:
            items = condition_items(pairs, condition)
            values = reference_nll(model, items, features)
            per_condition[condition] = values
            suite_record[f"nll_{condition}"] = float(np.mean(values))
        types = [question_type(item["example"]) for item in condition_items(pairs, "correct")]
        delta = np.array(per_condition["complementary"]) - np.array(per_condition["correct"])
        suite_record["delta_nll"] = float(delta.mean())
        suite_record["by_type"] = {
            kind: {"nll_correct": float(np.mean([v for v, t in zip(per_condition["correct"], types) if t == kind])),
                   "delta_nll": float(np.mean([v for v, t in zip(delta, types) if t == kind]))}
            for kind in sorted(set(types))}
        record[suite] = suite_record
        summary.append(f"{suite}: nll={suite_record['nll_correct']:.3f} dNLL={suite_record['delta_nll']:+.3f}")
    record["seconds"] = time.time() - started
    record["summary"] = " | ".join(summary)
    if was_training:
        model.train()
    return record


def score(example, prediction):
    if example["kind"] == "vqa":
        return soft_accuracy(prediction, example["answers"])
    return exact_match(prediction, example["target"])


def bootstrap_interval(values, indices):
    estimates = values[indices].mean(axis=1)
    return [float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))]


def suite_metrics(pairs, rows, rng_indices):
    """Accuracy, pair diagnostics and NLL per condition, with pair-bootstrap intervals and correct-image advantages."""
    pair_count = len(pairs)
    metrics = {}
    pair_arrays = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        by_pair = {}
        for row in condition_rows:
            by_pair.setdefault(row["pair_index"], {})[row["side"]] = row
        accuracy = np.zeros(pair_count)
        nll = np.zeros(pair_count)
        both_canonical = np.zeros(pair_count)
        identical = np.zeros(pair_count)
        for pair_index in range(pair_count):
            left, right = by_pair[pair_index][0], by_pair[pair_index][1]
            accuracy[pair_index] = (left["score"] + right["score"]) / 2
            nll[pair_index] = (left["reference_nll"] + right["reference_nll"]) / 2
            both_canonical[pair_index] = float(left["matches_canonical"] and right["matches_canonical"])
            identical[pair_index] = float(left["normalized_prediction"] == right["normalized_prediction"])
        pair_arrays[condition] = {"accuracy": accuracy, "nll": nll, "pair_accuracy": both_canonical}
        by_type = {}
        for kind in sorted({row["question_type"] for row in condition_rows}):
            typed = [row for row in condition_rows if row["question_type"] == kind]
            typed_pairs = sorted({row["pair_index"] for row in typed})
            by_type[kind] = {"accuracy": float(np.mean([row["score"] for row in typed])),
                             "pair_accuracy": float(np.mean(both_canonical[typed_pairs])),
                             "identical_answer_rate": float(np.mean(identical[typed_pairs])),
                             "reference_nll": float(np.mean([row["reference_nll"] for row in typed])),
                             "questions": len(typed)}
        metrics[condition] = {
            "accuracy": float(accuracy.mean()),
            "accuracy_ci95": bootstrap_interval(accuracy, rng_indices),
            "pair_accuracy": float(both_canonical.mean()),
            "pair_accuracy_ci95": bootstrap_interval(both_canonical, rng_indices),
            "identical_answer_rate": float(identical.mean()),
            "empty_answer_rate": float(np.mean([row["prediction"].strip() == "" for row in condition_rows])),
            "reference_nll": float(nll.mean()),
            "reference_nll_ci95": bootstrap_interval(nll, rng_indices),
            "generation_seconds_per_answer": float(np.mean([row["generation_seconds"] for row in condition_rows])),
            "by_type": by_type,
        }
        if condition == "complementary":
            follows = [row["follows_partner"] for row in condition_rows]
            metrics[condition]["follows_partner_answer_rate"] = float(np.mean(follows))
            metrics[condition]["changed_from_correct_image_rate"] = float(np.mean([row["changed_from_correct"]
                                                                                   for row in condition_rows]))
    advantages = {}
    correct = pair_arrays["correct"]
    for condition in CONDITIONS[1:]:
        other = pair_arrays[condition]
        accuracy_gain = correct["accuracy"] - other["accuracy"]
        pair_gain = correct["pair_accuracy"] - other["pair_accuracy"]
        nll_gain = other["nll"] - correct["nll"]
        advantages[f"correct_vs_{condition}"] = {
            "accuracy": float(accuracy_gain.mean()), "accuracy_ci95": bootstrap_interval(accuracy_gain, rng_indices),
            "pair_accuracy": float(pair_gain.mean()), "pair_accuracy_ci95": bootstrap_interval(pair_gain, rng_indices),
            "delta_nll": float(nll_gain.mean()), "delta_nll_ci95": bootstrap_interval(nll_gain, rng_indices),
        }
    metrics["advantages"] = advantages
    return metrics


def qualitative_pairs(pairs, rows, rng):
    """Selection rule: a seeded random sample (seed 6304) of three pairs per question type, independent of outcomes."""
    by_type = {}
    for pair_index, pair in enumerate(pairs):
        by_type.setdefault(question_type(pair["examples"][0]), []).append(pair_index)
    chosen = []
    for kind in sorted(by_type):
        candidates = by_type[kind]
        for pair_index in sorted(rng.choice(candidates, size=min(QUALITATIVE_PAIRS_PER_TYPE, len(candidates)),
                                            replace=False).tolist()):
            entry = {"pair_id": pairs[pair_index]["pair_id"], "question_type": kind,
                     "question": pairs[pair_index]["examples"][0]["question"], "sides": []}
            for side, example in enumerate(pairs[pair_index]["examples"]):
                predictions = {row["condition"]: row["prediction"] for row in rows
                               if row["pair_index"] == pair_index and row["side"] == side}
                entry["sides"].append({"image": example["image"], "reference": example["target"],
                                       "all_references": example.get("answers"), "predictions": predictions})
            chosen.append(entry)
    return chosen


def load_trained_model(run_name, device):
    from task2.model import VLM

    model = VLM(RUNS[run_name]["visual_tokens"], device)
    weights = torch.load(run_dir(run_name) / "stage_vqa" / "weights.pt", map_location="cpu", weights_only=False)
    model.connector.load_state_dict(weights["connector"])
    model.load_adapter_state(weights["adapters"])
    model.eval()
    return model


def evaluate(run_name):
    from task2.data import load_manifest
    from task2.features import FeatureStore

    output = results_root() / "eval" / run_name
    if (output / "DONE").exists():
        print(f"{run_name} already evaluated", flush=True)
        return
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    manifest = load_manifest()
    features = FeatureStore()
    model = load_trained_model(run_name, device)
    started = time.time()

    results = {"run": run_name, "environment": environment_info()}
    all_rows = []
    for suite in ("grounding", "vqa"):
        pairs = manifest["test"][suite]
        mapping = mismatch_mapping(pairs)
        rows = []
        correct_predictions = {}
        for condition in CONDITIONS:
            items = condition_items(pairs, condition, mapping)
            predictions, seconds = generate_answers(model, items, features)
            nll_values = reference_nll(model, items, features)
            for item, prediction, elapsed, nll_value in zip(items, predictions, seconds, nll_values):
                example = item["example"]
                partner = pairs[item["pair_index"]]["examples"][1 - item["side"]]
                normalized = normalize_answer(prediction)
                key = (item["pair_index"], item["side"])
                if condition == "correct":
                    correct_predictions[key] = normalized
                rows.append({
                    "suite": suite, "condition": condition, "pair_id": example["pair_id"],
                    "pair_index": item["pair_index"], "side": item["side"], "uid": example["uid"],
                    "question_type": question_type(example), "question": example["question"],
                    "image_used": item["image"], "reference": example["target"], "canonical": example["canonical"],
                    "prediction": prediction, "normalized_prediction": normalized,
                    "score": score(example, prediction), "matches_canonical": normalized == example["canonical"],
                    "follows_partner": normalized == partner["canonical"],
                    "changed_from_correct": normalized != correct_predictions.get(key, normalized),
                    "reference_nll": nll_value, "generation_seconds": elapsed,
                })
            print(f"[{run_name}] {suite}/{condition}: "
                  f"acc={np.mean([row['score'] for row in rows if row['condition'] == condition]):.3f}", flush=True)
        indices = np.random.default_rng(SEED).integers(len(pairs), size=(BOOTSTRAP_RESAMPLES, len(pairs)))
        results[suite] = suite_metrics(pairs, rows, indices)
        results[suite]["pairs"] = len(pairs)
        results[suite]["mismatch_mapping"] = {pairs[source]["pair_id"]: pairs[target]["pair_id"]
                                              for source, target in mapping.items()}
        save_json(output / f"qualitative_{suite}.json",
                  qualitative_pairs(pairs, rows, np.random.default_rng(SEED)))
        all_rows.extend(rows)

    caption_items = [{"example": example, "image": example["image"]}
                     for example in manifest["dev"]["caption"][:QUALITATIVE_CAPTIONS]]
    captions, _ = generate_answers(model, caption_items, features, max_new_tokens=48)
    save_json(output / "qualitative_captions.json",
              [{"image": item["image"], "reference": item["example"]["target"], "generated": caption}
               for item, caption in zip(caption_items, captions)])

    results["evaluation_seconds"] = time.time() - started
    results["peak_gpu_memory_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30 if use_cuda else None
    save_json(output / "metrics.json", results)
    with gzip.open(output / "per_example.jsonl.gz", "wt", encoding="utf-8") as handle:
        for row in all_rows:
            handle.write(json.dumps(row) + "\n")
    (output / "DONE").write_text("done\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=sorted(RUNS))
    args = parser.parse_args()
    evaluate(args.run)


if __name__ == "__main__":
    main()
