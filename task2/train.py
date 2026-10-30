"""Three-stage curriculum for the SigLIP-Qwen VLM (caption alignment, visual instruction tuning, natural VQA).

    python -m task2.train --run vlm196        (vlm196 | vlm49)

Resumes from <ATML2_OUT>/runs/<run>/recovery/state.pt if present (saved every 250 updates).
"""

import argparse
import json
import math
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from task1.common import (append_jsonl, environment_info, read_jsonl, rng_state, save_json, seed_everything,
                          set_rng_state)
from task2.config import (ADAM_BETAS, ADAM_EPS, CALIBRATION_IMAGES, CHECKPOINT_EVERY, DEV_EVAL_EVERY,
                          DRAWS_PER_MICRO_BATCH, EFFECTIVE_BATCH, FINAL_LR_FRACTION, GRAD_ACCUM, MAX_GRAD_NORM,
                          MICRO_BATCH, RUNS, SEED, STAGE_END_UPDATES, STAGES, TOTAL_UPDATES, WARMUP_FRACTION,
                          WEIGHT_DECAY, run_dir)
from task2.data import check_isolation, load_manifest
from task2.evaluate import development_metrics
from task2.features import FeatureStore
from task2.model import VLM, collate_training


class CurriculumSampler:
    """Stage 1 walks seeded permutations of the caption set; later stages draw components at random.

    Each question-stage draw contributes two examples: a complete pair, or two caption examples for replay.
    """

    def __init__(self, manifest, seed):
        self.captions = manifest["train"]["caption"]
        self.pools = {kind: manifest["train"][kind] for kind in ("presence", "counting", "vqa")}
        self.rng = np.random.default_rng(seed)
        self.permutation = None
        self.cursor = 0
        self.passes = 0
        self.counts = {}

    def caption_batch(self, size):
        batch = []
        while len(batch) < size:
            if self.permutation is None or self.cursor >= len(self.permutation):
                self.permutation = self.rng.permutation(len(self.captions))
                self.cursor = 0
                self.passes += 1
            batch.append(self.captions[self.permutation[self.cursor]])
            self.cursor += 1
        self.counts["caption"] = self.counts.get("caption", 0) + size
        return batch

    def mixture_batch(self, mixture, draws):
        names = list(mixture)
        probabilities = np.array([mixture[name] for name in names])
        batch = []
        for _ in range(draws):
            name = names[self.rng.choice(len(names), p=probabilities)]
            if name == "caption":
                batch.extend(self.captions[index] for index in self.rng.integers(len(self.captions), size=2))
            else:
                pool = self.pools[name]
                batch.extend(pool[self.rng.integers(len(pool))]["examples"])
            self.counts[name] = self.counts.get(name, 0) + 2
        return batch

    def next_micro_batch(self, stage):
        if list(stage["mixture"]) == ["caption"]:
            return self.caption_batch(MICRO_BATCH)
        return self.mixture_batch(stage["mixture"], DRAWS_PER_MICRO_BATCH)

    def state_dict(self):
        return {"rng": self.rng.bit_generator.state, "permutation": self.permutation, "cursor": self.cursor,
                "passes": self.passes, "counts": dict(self.counts)}

    def load_state_dict(self, state):
        self.rng.bit_generator.state = state["rng"]
        self.permutation = state["permutation"]
        self.cursor = state["cursor"]
        self.passes = state["passes"]
        self.counts = dict(state["counts"])


def stage_index_for(update):
    for index, end in enumerate(STAGE_END_UPDATES):
        if update < end:
            return index
    return len(STAGES) - 1


def lr_factor(step, total):
    """5% linear warmup, then cosine decay from the peak to 10% of the peak."""
    warmup = max(1, math.ceil(WARMUP_FRACTION * total))
    if step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return FINAL_LR_FRACTION + (1 - FINAL_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress))


def build_optimizer(model, stage, use_cuda):
    groups = []
    peaks = {"connector": stage["connector_lr"], "lora": stage["lora_lr"]}
    for name, parameters in model.trainable_parameters().items():
        decay = [parameter for parameter in parameters if parameter.ndim >= 2]
        no_decay = [parameter for parameter in parameters if parameter.ndim < 2]
        groups.append({"params": decay, "lr": peaks[name], "weight_decay": WEIGHT_DECAY, "name": name})
        if no_decay:
            groups.append({"params": no_decay, "lr": peaks[name], "weight_decay": 0.0, "name": name})
    optimizer = torch.optim.AdamW(groups, betas=ADAM_BETAS, eps=ADAM_EPS, fused=use_cuda)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_factor(step, stage["updates"]))
    return optimizer, scheduler


def count_parameters(parameters):
    return int(sum(parameter.numel() for parameter in parameters))


@torch.no_grad()
def calibrate_connector(model, manifest, features):
    """Fix the connector output scale so initial visual tokens match the RMS norm of text embeddings.

    Rule (training data only): scale = mean L2 norm of Qwen input embeddings over the prompt and target tokens of
    256 training captions / mean L2 norm of the freshly initialized connector outputs on those 256 images.
    """
    captions = manifest["train"]["caption"]
    rng = np.random.default_rng(SEED)
    chosen = [captions[index] for index in rng.choice(len(captions), size=CALIBRATION_IMAGES, replace=False)]
    embedding = model.language_model.get_input_embeddings()
    text_norms = []
    for example in chosen:
        ids = model.encoder.prompt_ids(example, with_image=False) + model.encoder.target_ids(example)
        vectors = embedding(torch.tensor(ids, device=model.device)).float()
        text_norms.append(vectors.norm(dim=-1))
    text_norm = torch.cat(text_norms).mean().item()
    model.connector.output_scale.fill_(1.0)
    visual_norms = []
    for start in range(0, len(chosen), 32):
        batch = features.get([example["image"] for example in chosen[start:start + 32]]).to(model.device)
        visual_norms.append(model.connector(batch).norm(dim=-1).flatten())
    visual_norm = torch.cat(visual_norms).mean().item()
    scale = text_norm / visual_norm
    model.connector.output_scale.fill_(scale)
    return {"rule": "scale = mean ||text embedding|| / mean ||initial connector output||, 256 training captions",
            "text_embedding_norm": text_norm, "initial_visual_norm": visual_norm, "scale": scale}


def interface_checks(model, manifest, features):
    example = manifest["train"]["caption"][0]
    question = manifest["train"]["vqa"][0]["examples"][0]
    batch_features = features.get([example["image"]])
    frozen = [parameter for parameter in model.language_model.parameters() if not parameter.requires_grad]
    trainable_lm = [name for name, parameter in model.language_model.named_parameters() if parameter.requires_grad
                    and "lora_" not in name]
    return {
        "feature_shape": list(batch_features.shape),
        "feature_dtype": str(batch_features.dtype),
        "connector_output_shape": list(model.connector(batch_features.to(model.device)).shape),
        "frozen_language_model_parameters": count_parameters(frozen),
        "trainable_original_language_model_parameters": trainable_lm,
        "prompt_matches_chat_template": {
            "caption": model.encoder.prompt_ids(example) == model.encoder.chat_template_ids(example),
            "question": model.encoder.prompt_ids(question) == model.encoder.chat_template_ids(question),
        },
        "split_isolation": "passed",
    }


def save_recovery(directory, payload):
    recovery = directory / "recovery"
    recovery.mkdir(parents=True, exist_ok=True)
    temporary = recovery / "state.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, recovery / "state.pt")


def save_stage_weights(model, directory, stage):
    path = directory / f"stage_{stage['name']}"
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"connector": model.connector.state_dict(), "adapters": model.adapter_state()}, path / "weights.pt")
    return path


def is_dev_eval_update(update):
    return update % DEV_EVAL_EVERY == 0 or update in STAGE_END_UPDATES


def truncate_log(path, last_update):
    rows = [row for row in read_jsonl(path) if row.get("update", 0) <= last_update]
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")


def train(run_name):
    visual_tokens = RUNS[run_name]["visual_tokens"]
    directory = run_dir(run_name)
    if (directory / "DONE").exists():
        print(f"{run_name} already trained", flush=True)
        return
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    seed_everything(SEED)
    manifest = load_manifest()
    check_isolation(manifest)
    features = FeatureStore()

    model = VLM(visual_tokens, device)
    calibration = calibrate_connector(model, manifest, features)
    sampler = CurriculumSampler(manifest, SEED)
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    train_log = directory / "train_log.jsonl"
    dev_log = directory / "dev_log.jsonl"

    start_update = 0
    elapsed_before = 0.0
    stage_parameter_counts = {}
    recovery_path = directory / "recovery" / "state.pt"
    state = None
    if recovery_path.exists():
        state = torch.load(recovery_path, map_location="cpu", weights_only=False)
        start_update = state["update"]
        elapsed_before = state["elapsed_seconds"]
        stage_parameter_counts = state["stage_parameter_counts"]
        model.connector.load_state_dict(state["connector"])
        model.load_adapter_state(state["adapters"])
        sampler.load_state_dict(state["sampler"])
        set_rng_state(state["rng"])
        truncate_log(train_log, start_update)
        truncate_log(dev_log, start_update)
        print(f"resumed {run_name} at update {start_update}", flush=True)
    else:
        save_json(directory / "config.json", {
            "run": run_name, "visual_tokens": visual_tokens, "stages": STAGES, "micro_batch": MICRO_BATCH,
            "grad_accum": GRAD_ACCUM, "effective_batch": EFFECTIVE_BATCH, "seed": SEED,
            "calibration": calibration, "environment": environment_info(),
        })
        save_json(directory / "interface_checks.json", interface_checks(model, manifest, features))

    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    started = time.time()
    current_stage = None
    optimizer = scheduler = None
    connector_grad_checked = (directory / "connector_grad_check.json").exists()

    if start_update == 0 and not read_jsonl(dev_log):
        record = development_metrics(model, manifest, features)
        append_jsonl(dev_log, dict(record, update=0, stage="initial"))

    update = start_update
    while update < TOTAL_UPDATES:
        stage_index = stage_index_for(update)
        stage = STAGES[stage_index]
        if current_stage != stage_index:
            if stage["lora_lr"] is not None and not model.adapters:
                model.enable_lora()
            optimizer, scheduler = build_optimizer(model, stage, use_cuda)
            stage_start = STAGE_END_UPDATES[stage_index] - stage["updates"]
            if state is not None and state["stage_index"] == stage_index and state["optimizer"] is not None:
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                scaler.load_state_dict(state["scaler"])
            else:
                for _ in range(update - stage_start):
                    scheduler.step()
            groups = model.trainable_parameters()
            stage_parameter_counts[stage["name"]] = {name: count_parameters(parameters)
                                                     for name, parameters in groups.items()}
            current_stage = stage_index
            state = None

        model.train()
        optimizer.zero_grad(set_to_none=True)
        update_started = time.time()
        losses = []
        for _ in range(GRAD_ACCUM):
            examples = sampler.next_micro_batch(stage)
            input_ids, labels, attention_mask = collate_training(model.encoder, examples)
            batch_features = features.get([example["image"] for example in examples])
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_cuda):
                example_losses = model.example_losses(input_ids, labels, attention_mask, batch_features)
            scaler.scale(example_losses.sum() / EFFECTIVE_BATCH).backward()
            losses.extend(example_losses.detach().float().cpu().tolist())
        scaler.unscale_(optimizer)
        parameters = [parameter for group in model.trainable_parameters().values() for parameter in group]
        if not connector_grad_checked:
            grad_norms = {name: float(torch.norm(torch.stack([parameter.grad.norm() for parameter in group
                                                              if parameter.grad is not None])))
                          for name, group in model.trainable_parameters().items()}
            missing = [name for name, parameter in model.connector.named_parameters() if parameter.grad is None]
            assert not missing, f"no gradient reached the connector: {missing}"
            # FP16 loss scaling overflows on the first updates (the scaler skips them); check a finite one.
            if math.isfinite(grad_norms["connector"]):
                save_json(directory / "connector_grad_check.json", {"update": update, "grad_norms": grad_norms})
                assert grad_norms["connector"] > 0, "connector gradient is zero"
                connector_grad_checked = True
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, MAX_GRAD_NORM).item()
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped = use_cuda and scaler.get_scale() < scale_before
        learning_rates = {group["name"]: group["lr"] for group in optimizer.param_groups}
        scheduler.step()
        update += 1

        append_jsonl(train_log, {
            "update": update, "stage": stage["name"], "loss": float(np.mean(losses)), "grad_norm": grad_norm,
            "lr": learning_rates, "skipped": skipped, "seconds": time.time() - update_started,
            "mixture_counts": dict(sampler.counts),
        })
        if update % 50 == 0:
            print(f"[{run_name}] update {update}/{TOTAL_UPDATES} stage={stage['name']} loss={np.mean(losses):.4f}",
                  flush=True)

        if update in STAGE_END_UPDATES:
            save_stage_weights(model, directory, stage)
        if is_dev_eval_update(update):
            record = development_metrics(model, manifest, features)
            append_jsonl(dev_log, dict(record, update=update, stage=stage["name"]))
            print(f"[{run_name}] dev @ {update}: {record['summary']}", flush=True)
        if update % CHECKPOINT_EVERY == 0 and update < TOTAL_UPDATES:
            save_recovery(directory, {
                "update": update, "stage_index": stage_index_for(update),
                "connector": model.connector.state_dict(), "adapters": model.adapter_state(),
                "optimizer": optimizer.state_dict() if stage_index_for(update) == stage_index else None,
                "scheduler": scheduler.state_dict() if stage_index_for(update) == stage_index else None,
                "scaler": scaler.state_dict(), "sampler": sampler.state_dict(), "rng": rng_state(),
                "elapsed_seconds": elapsed_before + time.time() - started,
                "stage_parameter_counts": stage_parameter_counts,
            })

    save_json(directory / "train_summary.json", {
        "run": run_name, "updates": TOTAL_UPDATES,
        "example_presentations": TOTAL_UPDATES * EFFECTIVE_BATCH,
        "mixture_counts": sampler.counts, "caption_passes": sampler.passes,
        "trainable_parameters_by_stage": stage_parameter_counts,
        "training_seconds": elapsed_before + time.time() - started,
        "peak_gpu_memory_gb_this_session": torch.cuda.max_memory_allocated() / 2 ** 30 if use_cuda else None,
        "resumed_from_update": start_update, "calibration": calibration,
    })
    (directory / "DONE").write_text("done\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=sorted(RUNS))
    args = parser.parse_args()
    train(args.run)


if __name__ == "__main__":
    main()
