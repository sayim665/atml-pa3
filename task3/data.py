"""Load and join the supplied rollout tables, check token alignment, build answer groups, splits and orders.

    python -m task3.data            (writes outputs/task3/prepared_<dataset>.pkl and results/task3/splits/)

Per-token lists are read as flat Arrow buffers (no Python lists) and summarized per rollout:
T (retained tokens), L (mean logp), K (mean kl_u_p); for MathVista also the per-token kl and visual-engagement
arrays needed for candidate-relative VisConf calibration.
"""

import json
import pickle
import time
import urllib.request

import numpy as np

from task1.common import save_json
from task3.answers import (MathEquivalence, grade_math, grade_mathvista, group_math_answers,
                           group_mathvista_answers, mathvista_gold, scorer_versions)
from task3.config import (DATASETS, DEV_QUESTIONS, LIVE_QUESTIONS, MATHVISTA_METADATA_URL, NUM_ORDERS, ROLLOUTS,
                          SEED, data_root, output_root, results_root)

KEYS = ("dataset", "sample_id", "rollout_id")


def read_table(path, columns):
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=list(columns)).combine_chunks()


def list_column(table, name):
    """(flat float64 values, offsets) for a list column."""
    column = table.column(name).combine_chunks()
    offsets = column.offsets.to_numpy()
    values = column.values.to_numpy(zero_copy_only=False).astype(np.float64)
    return values, offsets


def key_order(table):
    sample_ids = table.column("sample_id").to_pylist()
    rollout_ids = table.column("rollout_id").to_pylist()
    return {(sample, int(rollout)): row for row, (sample, rollout) in enumerate(zip(sample_ids, rollout_ids))}


def segment_means(values, offsets):
    lengths = np.diff(offsets)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    sums = cumulative[offsets[1:]] - cumulative[offsets[:-1]]
    return sums / np.maximum(lengths, 1), lengths


def mathvista_metadata():
    path = data_root() / "testmini.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MATHVISTA_METADATA_URL, path)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def splits(question_ids):
    ordered = np.array(sorted(question_ids))
    permuted = ordered[np.random.default_rng(SEED).permutation(len(ordered))].tolist()
    return {"dev": permuted[:DEV_QUESTIONS], "eval": permuted[DEV_QUESTIONS:]}


def rollout_orders(question_ids):
    """Ten seeded permutations of the 32 rollout IDs per question, shared by every method and stopping policy."""
    rng = np.random.default_rng([SEED, 1])
    return {question: np.stack([rng.permutation(ROLLOUTS) for _ in range(NUM_ORDERS)])
            for question in sorted(question_ids)}


def prepare(name):
    started = time.time()
    folder = data_root() / DATASETS[name]["folder"]
    vision = DATASETS[name]["vision"]
    generations = read_table(folder / "generations.parquet",
                             KEYS + ("question_text", "generated_text", "generated_token_ids",
                                     "n_generated_text_tokens", "hit_max_new_tokens", "gold_answer",
                                     "extracted_answer"))
    probabilities = read_table(folder / "probability_metrics.parquet", KEYS + ("logp", "kl_u_p"))
    generation_rows = key_order(generations)
    probability_rows = key_order(probabilities)
    assert set(generation_rows) == set(probability_rows), "generation/probability keys differ"

    token_ids = generations.column("generated_token_ids").combine_chunks()
    id_lengths = np.diff(token_ids.offsets.to_numpy())
    scalar_lengths = generations.column("n_generated_text_tokens").to_numpy()
    logp, logp_offsets = list_column(probabilities, "logp")
    kl, kl_offsets = list_column(probabilities, "kl_u_p")
    mean_logp, logp_lengths = segment_means(logp, logp_offsets)
    mean_kl, kl_lengths = segment_means(kl, kl_offsets)

    alignment = {"rows": len(generation_rows), "token_id_vs_scalar_mismatches": int((id_lengths != scalar_lengths).sum())}
    probability_index = np.array([probability_rows[key] for key in sorted(generation_rows, key=generation_rows.get)])
    alignment["logp_length_mismatches"] = int((logp_lengths[probability_index] != id_lengths).sum())
    alignment["kl_length_mismatches"] = int((kl_lengths[probability_index] != id_lengths).sum())

    if vision:
        attention = read_table(folder / "attention_metrics.parquet",
                               KEYS + ("img_attn_total", "prompt_text_attn_total", "generated_text_attn_total"))
        attention_rows = key_order(attention)
        assert set(attention_rows) == set(generation_rows), "attention keys differ"
        image, image_offsets = list_column(attention, "img_attn_total")
        prompt, _ = list_column(attention, "prompt_text_attn_total")
        generated, _ = list_column(attention, "generated_text_attn_total")
        denominator = image + prompt + generated
        engagement = np.where(denominator > 0, image / np.where(denominator > 0, denominator, 1), 0.0)
        attention_index = np.array([attention_rows[key] for key in sorted(generation_rows, key=generation_rows.get)])
        alignment["attention_length_mismatches"] = int(
            (np.diff(image_offsets)[attention_index] != id_lengths).sum())
        metadata = mathvista_metadata()
    else:
        equivalence = MathEquivalence()

    sample_ids = generations.column("sample_id").to_pylist()
    rollout_ids = generations.column("rollout_id").to_pylist()
    extracted = generations.column("extracted_answer").to_pylist()
    gold = generations.column("gold_answer").to_pylist()
    capped = generations.column("hit_max_new_tokens").to_numpy(zero_copy_only=False)
    texts = generations.column("question_text").to_pylist()

    by_question = {}
    for row, (sample, rollout) in enumerate(zip(sample_ids, rollout_ids)):
        by_question.setdefault(sample, [None] * ROLLOUTS)[int(rollout)] = row

    questions = {}
    for question, rows in sorted(by_question.items()):
        assert all(row is not None for row in rows), f"{question} is missing rollouts"
        probability_rows_q = [probability_rows[(question, rollout)] for rollout in range(ROLLOUTS)]
        answers = [extracted[row] for row in rows]
        if vision:
            meta = metadata[str(question)]
            groups = group_mathvista_answers(answers, meta)
            correct_by_group = grade_mathvista(groups, meta)
            gold_label = mathvista_gold(meta)
        else:
            groups = group_math_answers(answers, equivalence)
            correct_by_group = grade_math(groups, gold[rows[0]], equivalence)
            gold_label = gold[rows[0]]
        record = {
            "question_text": texts[rows[0]], "gold": gold[rows[0]], "gold_label": gold_label,
            "extracted": answers, "groups": groups,
            "correct": np.array([correct_by_group[group] for group in groups], dtype=bool),
            "tokens": id_lengths[rows].astype(np.int64), "capped": capped[rows].astype(bool),
            "L": mean_logp[probability_rows_q], "K": mean_kl[probability_rows_q],
        }
        if vision:
            attention_rows_q = [attention_rows[(question, rollout)] for rollout in range(ROLLOUTS)]
            record["token_kl"] = [kl[kl_offsets[row]:kl_offsets[row + 1]].astype(np.float32)
                                  for row in probability_rows_q]
            record["token_vis"] = [engagement[image_offsets[row]:image_offsets[row + 1]].astype(np.float32)
                                   for row in attention_rows_q]
            record["V"] = np.array([values.mean() if len(values) else 0.0 for values in record["token_vis"]])
            if meta["question_type"] == "multi_choice":
                record["choices"] = meta["choices"]
        questions[question] = record

    question_ids = sorted(questions)
    prepared = {"dataset": name, "questions": questions, "splits": splits(question_ids),
                "orders": rollout_orders(question_ids), "alignment": alignment,
                "scorer": scorer_versions(), "seconds": time.time() - started,
                "live_questions": [{"id": question, "question": questions[question]["question_text"],
                                    "gold": questions[question]["gold"]}
                                   for question in splits(question_ids)["eval"][:LIVE_QUESTIONS]]
                if name == "math" else None}
    return prepared


def prepared_path(name):
    return output_root() / f"prepared_{name}.pkl"


def load_prepared(name):
    with open(prepared_path(name), "rb") as handle:
        return pickle.load(handle)


def summary(prepared):
    questions = prepared["questions"]
    correct = np.stack([record["correct"] for record in questions.values()])
    no_answer = np.mean([[group == "NO_ANSWER" for group in record["groups"]] for record in questions.values()])
    return {"questions": len(questions), "rollouts": int(correct.size), "rollout_accuracy": float(correct.mean()),
            "no_answer_rate": float(no_answer), "alignment": prepared["alignment"], "scorer": prepared["scorer"],
            "mean_tokens": float(np.mean([record["tokens"].mean() for record in questions.values()])),
            "capped_rate": float(np.mean([record["capped"].mean() for record in questions.values()])),
            "seconds": prepared["seconds"]}


def main():
    output_root().mkdir(parents=True, exist_ok=True)
    splits_dir = results_root() / "splits"
    for name in DATASETS:
        if prepared_path(name).exists():
            print(f"{name}: prepared data exists", flush=True)
            continue
        prepared = prepare(name)
        with open(prepared_path(name), "wb") as handle:
            pickle.dump(prepared, handle, protocol=pickle.HIGHEST_PROTOCOL)
        save_json(splits_dir / f"{name}_splits.json", prepared["splits"])
        save_json(splits_dir / f"{name}_orders.json", {question: order.tolist()
                                                       for question, order in prepared["orders"].items()})
        if prepared["live_questions"]:
            save_json(splits_dir / "math_live_questions.json", prepared["live_questions"])
        save_json(results_root() / "data_summary" / f"{name}.json", summary(prepared))
        print(name, json.dumps(summary(prepared), indent=1), flush=True)


if __name__ == "__main__":
    main()
