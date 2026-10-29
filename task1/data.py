"""Training-set construction (MetaMathQA) and the fixed activation-probe set.

Run once before training:  python -m task1.data
"""

import argparse
import re
import unicodedata

import numpy as np
import torch

from task1.common import (chat_prompt_text, load_json, load_tokenizer, output_root, results_root, save_json)
from task1.config import (GSM8K_DATASET, MAX_SEQ_LEN, NUM_TRAIN_EXAMPLES, PROBE_GSM8K, PROBE_HELLASWAG, SEED,
                          TRAIN_DATASET)

GSM8K_TEST_SIZE = 1319
HELLASWAG_VALIDATION_SIZE = 10042
CHUNK_SIZE = 2000


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def train_cache_path():
    return output_root() / "cache" / "train_data.pt"


def gsm8k_test_question_set():
    from datasets import load_dataset

    test_split = load_dataset(GSM8K_DATASET, "main", split="test")
    return {normalize_text(row["question"]) for row in test_split}


def exclusion_reason(row, blocked_questions):
    query = row.get("query") or ""
    response = row.get("response") or ""
    if not query.strip() or not response.strip():
        return "empty"
    if normalize_text(query) in blocked_questions:
        return "gsm8k_test_overlap"
    original_question = row.get("original_question") or ""
    if original_question and normalize_text(original_question) in blocked_questions:
        return "gsm8k_test_overlap"
    return None


def streamed_chunks(rows, size):
    chunk = []
    for source_index, row in enumerate(rows):
        chunk.append((source_index, row))
        if len(chunk) == size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def build_training_set(tokenizer, limit=NUM_TRAIN_EXAMPLES):
    from datasets import load_dataset

    source = load_dataset(TRAIN_DATASET, split="train", streaming=True)
    blocked_questions = gsm8k_test_question_set()
    end_of_turn = "<|im_end|>"

    kept_ids = []
    kept_prompt_lengths = []
    kept_source_indices = []
    excluded = []

    for chunk in streamed_chunks(source, CHUNK_SIZE if limit >= CHUNK_SIZE else 4 * limit):
        if len(kept_ids) >= limit:
            break
        candidates = []
        for source_index, row in chunk:
            reason = exclusion_reason(row, blocked_questions)
            if reason is not None:
                excluded.append((source_index, reason))
                continue
            candidates.append((source_index, row))

        if not candidates:
            continue
        prompts = [chat_prompt_text(tokenizer, row["query"]) for _, row in candidates]
        targets = [row["response"] + end_of_turn for _, row in candidates]
        prompt_tokens = tokenizer(prompts, add_special_tokens=False)["input_ids"]
        target_tokens = tokenizer(targets, add_special_tokens=False)["input_ids"]

        for (source_index, _), prompt_ids, target_ids in zip(candidates, prompt_tokens, target_tokens):
            if len(kept_ids) >= limit:
                break
            if len(prompt_ids) + len(target_ids) > MAX_SEQ_LEN:
                excluded.append((source_index, "too_long"))
                continue
            kept_ids.append(prompt_ids + target_ids)
            kept_prompt_lengths.append(len(prompt_ids))
            kept_source_indices.append(source_index)

    if len(kept_ids) < limit:
        raise RuntimeError(f"Only {len(kept_ids)} eligible examples found, expected {limit}")

    lengths = [len(ids) for ids in kept_ids]
    offsets = np.zeros(len(kept_ids) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    flat_ids = np.fromiter((token for ids in kept_ids for token in ids), dtype=np.int32, count=int(offsets[-1]))

    last_source_index = kept_source_indices[-1]
    exclusions = {"empty": 0, "gsm8k_test_overlap": 0, "too_long": 0}
    for source_index, reason in excluded:
        if source_index < last_source_index:
            exclusions[reason] += 1
    summary = {
        "dataset": TRAIN_DATASET,
        "num_examples": len(kept_ids),
        "last_source_index": last_source_index,
        "source_rows_scanned": last_source_index + 1,
        "exclusions_before_last_kept": exclusions,
        "total_tokens": int(offsets[-1]),
        "response_tokens": int(sum(lengths) - sum(kept_prompt_lengths)),
        "mean_length": float(np.mean(lengths)),
        "max_length": int(np.max(lengths)),
        "normalization": "NFKC, lowercase, non-alphanumerics to spaces, collapse whitespace",
    }
    data = {
        "input_ids": torch.from_numpy(flat_ids),
        "offsets": torch.from_numpy(offsets),
        "prompt_lengths": torch.tensor(kept_prompt_lengths, dtype=torch.int32),
        "source_indices": kept_source_indices,
        "summary": summary,
    }
    return data


def prepare_training_set(force=False):
    path = train_cache_path()
    if path.exists() and not force:
        return torch.load(path)
    tokenizer = load_tokenizer()
    data = build_training_set(tokenizer)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    save_json(results_root() / "splits" / "train_source_indices.json", data["source_indices"])
    save_json(results_root() / "splits" / "train_summary.json", data["summary"])
    return data


def example_tensors(data, index):
    start = int(data["offsets"][index])
    end = int(data["offsets"][index + 1])
    input_ids = data["input_ids"][start:end].long()
    labels = input_ids.clone()
    labels[: int(data["prompt_lengths"][index])] = -100
    return input_ids, labels


def collate(examples, pad_token_id):
    max_length = max(len(ids) for ids, _ in examples)
    batch_size = len(examples)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    for row, (ids, row_labels) in enumerate(examples):
        input_ids[row, : len(ids)] = ids
        labels[row, : len(ids)] = row_labels
        attention_mask[row, : len(ids)] = 1
    return input_ids, labels, attention_mask


def probe_set_path():
    return results_root() / "splits" / "probe_set.json"


def build_probe_set():
    """Sample probe IDs with seed 6304 before any model results exist; indices follow lm-eval doc_id order."""
    path = probe_set_path()
    if path.exists():
        return load_json(path)
    rng = np.random.default_rng(SEED)
    gsm8k_ids = sorted(int(i) for i in rng.choice(GSM8K_TEST_SIZE, size=PROBE_GSM8K, replace=False))
    hellaswag_ids = sorted(int(i) for i in rng.choice(HELLASWAG_VALIDATION_SIZE, size=PROBE_HELLASWAG, replace=False))
    probe = {
        "seed": SEED,
        "gsm8k_test_doc_ids": gsm8k_ids,
        "hellaswag_validation_doc_ids": hellaswag_ids,
    }
    save_json(path, probe)
    return probe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_probe_set()
    data = prepare_training_set(force=args.force)
    print(data["summary"])


if __name__ == "__main__":
    main()
