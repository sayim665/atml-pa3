"""Synthetic rollout tables with the supplied schema, for a CPU smoke test of the offline pipeline.

    ATML_SMOKE_TEST=1 python -m task3.smoke  (then python -m task3.data and python -m task3.analysis)
"""

import json
import shutil
import sys

import numpy as np

from task3.config import DATASETS, ROLLOUTS, SMOKE_TEST, data_root

QUESTIONS = 12


def write_tables(name, rng, metadata):
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = data_root() / DATASETS[name]["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    generations, probabilities, attention = [], [], []
    for index in range(QUESTIONS):
        question = f"test:{index}" if name == "math" else str(index + 1)
        if name == "math":
            gold = ["\\frac{1}{2}", "12", "x^2+1", "\\sqrt{2}"][index % 4]
            pool = [gold, "0.5" if index % 4 == 0 else gold, "7", None, "\\dfrac{3}{4}"]
        else:
            meta = metadata[question]
            gold = meta["answer"]
            pool = ([gold, "(A)", meta["choices"][-1], None] if meta["question_type"] == "multi_choice"
                    else [gold, "3", "4.0", None])
        for rollout in range(ROLLOUTS):
            length = int(rng.integers(0, 60)) if rollout else 0
            answer = pool[int(rng.integers(len(pool)))]
            generations.append({"dataset": name, "sample_id": question, "rollout_id": rollout,
                                "model": "synthetic", "question_text": f"Question {index}",
                                "generated_text": f"... \\boxed{{{answer}}}", "generated_token_ids": list(range(length)),
                                "n_generated_text_tokens": length, "hit_max_new_tokens": bool(rollout == 5),
                                "gold_answer": gold, "extracted_answer": answer})
            probabilities.append({"dataset": name, "sample_id": question, "rollout_id": rollout,
                                  "logp": (-rng.random(length)).astype(np.float32).tolist(),
                                  "kl_u_p": (8 + 3 * rng.random(length)).astype(np.float32).tolist()})
            attention.append({"dataset": name, "sample_id": question, "rollout_id": rollout,
                              "n_image_tokens": 100, "n_prompt_text_tokens": 30,
                              "img_attn_total": rng.random(length).astype(np.float32).tolist(),
                              "prompt_text_attn_total": rng.random(length).astype(np.float32).tolist(),
                              "generated_text_attn_total": rng.random(length).astype(np.float32).tolist(),
                              "n_generated_text_tokens": list(range(length))})
    order = rng.permutation(len(generations))
    pq.write_table(pa.Table.from_pylist([generations[i] for i in order]), folder / "generations.parquet")
    pq.write_table(pa.Table.from_pylist(probabilities), folder / "probability_metrics.parquet")
    if DATASETS[name]["vision"]:
        pq.write_table(pa.Table.from_pylist(attention), folder / "attention_metrics.parquet")


def main():
    assert SMOKE_TEST, "set ATML_SMOKE_TEST=1"
    metadata_source = sys.argv[1]
    data_root().mkdir(parents=True, exist_ok=True)
    shutil.copy(metadata_source, data_root() / "testmini.json")
    with open(metadata_source, encoding="utf-8") as handle:
        metadata = json.load(handle)
    rng = np.random.default_rng(0)
    for name in DATASETS:
        write_tables(name, rng, metadata)
    print(f"synthetic tables in {data_root()}")


if __name__ == "__main__":
    main()
