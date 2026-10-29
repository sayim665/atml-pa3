"""One-epoch SFT of Qwen3-0.6B on MetaMathQA for a single Task 1 variant.

    python -m task1.train --run full_ft        (full_ft | l2sp | lora4 | lora16 | oplora16)

Resumes automatically from <ATML_OUT>/runs/<run>/recovery if present.
"""

import argparse
import os
import shutil
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import get_cosine_schedule_with_warmup

from task1.common import (append_jsonl, directory_size_bytes, environment_info, load_causal_lm, load_json,
                          load_tokenizer, read_jsonl, rng_state, run_dir, save_json, seed_everything, set_rng_state)
from task1.config import (ADAM_BETAS, ADAM_EPS, CHECKPOINT_UPDATES, DIAGNOSTIC_LAYERS, EFFECTIVE_BATCH,
                          FINAL_UPDATE, GRAD_ACCUM, MAX_GRAD_NORM, MICRO_BATCH, MIDPOINT_UPDATE, MODEL_NAME, RUNS,
                          SEED, TOTAL_UPDATES, WARMUP_UPDATES, WEIGHT_DECAY)
from task1.data import collate, example_tensors, prepare_training_set
from task1.lora import export_adapter_factors, find_adapters, inject_adapters

LOSS_CHUNK_TOKENS = 256
PARAMETER_CHUNK_ELEMENTS = 1 << 24


def build_model(run, device):
    if run.is_full_ft():
        model = load_causal_lm(MODEL_NAME, torch.float32, device)
        adapters = {}
        overhead = {}
    else:
        model = load_causal_lm(MODEL_NAME, torch.float16, device)
        adapters, overhead = inject_adapters(model, run.lora_rank, run.lora_alpha, run.protection_rank)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, adapters, overhead


def unique_trainable_parameters(model):
    seen = set()
    named = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        named.append((name, parameter))
    return named


def build_optimizer(named_parameters, peak_lr, use_fused):
    decay = [parameter for _, parameter in named_parameters if parameter.ndim >= 2]
    no_decay = [parameter for _, parameter in named_parameters if parameter.ndim < 2]
    groups = [{"params": decay, "weight_decay": WEIGHT_DECAY}]
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=peak_lr, betas=ADAM_BETAS, eps=ADAM_EPS, fused=use_fused)


def reference_dtype(model):
    configured = getattr(model.config, "torch_dtype", None) or getattr(model.config, "dtype", None)
    if configured in (torch.bfloat16, "bfloat16"):
        return torch.bfloat16
    return torch.float32


def snapshot_reference(named_parameters, dtype, device):
    """theta_0, stored losslessly in the checkpoint's own dtype (bf16 for Qwen3-0.6B)."""
    return [parameter.detach().to(device=device, dtype=dtype).clone() for _, parameter in named_parameters]


@torch.no_grad()
def reference_check(named_parameters, reference):
    worst = 0.0
    for (_, parameter), anchor in zip(named_parameters, reference):
        difference = parameter.detach() - anchor.to(parameter.device)
        worst = max(worst, difference.abs().max().item())
    return worst


def flat_chunks(*tensors):
    """Matching slices of flattened tensors, so large parameters never need a full-size temporary."""
    flats = [tensor.reshape(-1) for tensor in tensors]
    return zip(*(flat.split(PARAMETER_CHUNK_ELEMENTS) for flat in flats))


@torch.no_grad()
def squared_displacement(named_parameters, reference):
    total = torch.zeros((), dtype=torch.float64, device=named_parameters[0][1].device)
    for (_, parameter), anchor in zip(named_parameters, reference):
        anchor = anchor.to(parameter.device)
        for parameter_chunk, anchor_chunk in flat_chunks(parameter.detach(), anchor):
            total += (parameter_chunk.float() - anchor_chunk.float()).square_().sum(dtype=torch.float64)
    return total.item()


@torch.no_grad()
def add_l2sp_gradient(named_parameters, reference, strength):
    for (_, parameter), anchor in zip(named_parameters, reference):
        if parameter.grad is None:
            continue
        for grad_chunk, parameter_chunk, anchor_chunk in flat_chunks(parameter.grad, parameter.detach(), anchor):
            grad_chunk.add_(parameter_chunk - anchor_chunk, alpha=strength)


def chunk_loss_sum(hidden, targets, lm_head):
    logits = lm_head(hidden)
    return F.cross_entropy(logits.float(), targets, reduction="sum")


def response_loss_sum(model, input_ids, labels, attention_mask):
    """Summed next-token CE over response tokens; logits are only formed at supervised positions."""
    hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    shifted_hidden = hidden[:, :-1]
    shifted_targets = labels[:, 1:]
    supervised = shifted_targets != -100
    selected_hidden = shifted_hidden[supervised]
    selected_targets = shifted_targets[supervised]
    total = hidden.new_zeros((), dtype=torch.float32)
    for start in range(0, selected_targets.numel(), LOSS_CHUNK_TOKENS):
        end = start + LOSS_CHUNK_TOKENS
        total = total + checkpoint(chunk_loss_sum, selected_hidden[start:end], selected_targets[start:end],
                                   model.lm_head, use_reentrant=False)
    return total


def save_recovery(directory, update, trainable, optimizer, scheduler, scaler, log_path):
    recovery = directory / "recovery"
    staging = directory / "recovery_staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    state = {
        "update": update,
        "examples_seen": update * EFFECTIVE_BATCH,
        "trainable": {name: parameter.detach().cpu() for name, parameter in trainable},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": rng_state(),
        "data_position": update * EFFECTIVE_BATCH,
    }
    torch.save(state, staging / "state.pt")
    shutil.copy(log_path, staging / "train_log.jsonl")
    if recovery.exists():
        shutil.rmtree(recovery)
    staging.rename(recovery)
    return directory_size_bytes(recovery / "state.pt")


def load_recovery(directory, trainable, optimizer, scheduler, scaler):
    state_path = directory / "recovery" / "state.pt"
    if not state_path.exists():
        return 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    with torch.no_grad():
        for name, parameter in trainable:
            parameter.copy_(state["trainable"][name].to(parameter.device))
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    set_rng_state(state["rng"])
    return state["update"]


@torch.no_grad()
def export_full_model(model, named_parameters, reference, export_dir, tokenizer):
    start = time.time()
    export_dir.mkdir(parents=True, exist_ok=True)
    half_state = {name: tensor.detach().half().cpu() for name, tensor in model.state_dict().items()}
    model.save_pretrained(export_dir, state_dict=half_state, safe_serialization=True)
    tokenizer.save_pretrained(export_dir)
    diagnostic = {}
    for layer_index in DIAGNOSTIC_LAYERS:
        weight = model.model.layers[layer_index].mlp.down_proj.weight
        diagnostic[f"model.layers.{layer_index}.mlp.down_proj"] = weight.detach().float().cpu()
    torch.save(diagnostic, export_dir / "diagnostic_down_proj_fp32.pt")
    displacement = squared_displacement(named_parameters, reference) ** 0.5
    return {
        "export_seconds": time.time() - start,
        "export_bytes": directory_size_bytes(export_dir),
        "displacement_l2": displacement,
    }


def export_adapters(adapters, export_dir):
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / "adapter_factors.safetensors"
    seconds = export_adapter_factors(adapters, path)
    return {"export_seconds": seconds, "export_bytes": directory_size_bytes(path)}


def train(run_name, max_updates=None, device=None, keep_recovery=False):
    run = RUNS[run_name]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.startswith("cuda")
    directory = run_dir(run_name)
    log_path = directory / "train_log.jsonl"
    summary_path = directory / "train_summary.json"
    if (directory / "DONE").exists():
        print(f"[{run_name}] already complete, skipping")
        return load_json(summary_path)

    seed_everything(SEED)
    load_start = time.time()
    data = prepare_training_set()
    tokenizer = load_tokenizer()
    model, adapters, adapter_overhead = build_model(run, device)
    named_trainable = unique_trainable_parameters(model)
    trainable_count = sum(parameter.numel() for _, parameter in named_trainable)

    reference = None
    reference_info = {}
    if run.is_full_ft():
        reference_dtype_used = reference_dtype(model)
        reference_device = device if run.l2sp_lambda > 0 else "cpu"
        reference = snapshot_reference(named_trainable, reference_dtype_used, reference_device)
        reference_info = {
            "reference_dtype": str(reference_dtype_used),
            "reference_device": reference_device,
            "reference_bytes": sum(anchor.numel() * anchor.element_size() for anchor in reference),
            "reference_max_abs_error": reference_check(named_trainable, reference),
        }

    optimizer = build_optimizer(named_trainable, run.peak_lr, use_fused=use_amp)
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_UPDATES, TOTAL_UPDATES)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_update = load_recovery(directory, named_trainable, optimizer, scheduler, scaler)
    if start_update > 0:
        kept = [record for record in read_jsonl(log_path) if record["update"] <= start_update]
        log_path.unlink(missing_ok=True)
        for record in kept:
            append_jsonl(log_path, record)
    elif log_path.exists():
        log_path.unlink()

    previous = {}
    if summary_path.exists():
        previous = load_json(summary_path)
    summary = {
        "run": run.to_dict(),
        "environment": environment_info(),
        "trainable_parameters": trainable_count,
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "adapter_overhead": adapter_overhead,
        "reference": reference_info,
        "model_load_seconds": time.time() - load_start,
        "segments": previous.get("segments", []),
        "checkpoints": previous.get("checkpoints", {}),
        "exports": previous.get("exports", {}),
        "skipped_updates": previous.get("skipped_updates", []),
    }

    last_update = TOTAL_UPDATES if max_updates is None else min(TOTAL_UPDATES, start_update + max_updates)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    segment_start = time.time()
    trainable_parameters = [parameter for _, parameter in named_trainable]

    for update in range(start_update, last_update):
        update_start = time.time()
        first_example = update * EFFECTIVE_BATCH
        examples = [example_tensors(data, index) for index in range(first_example, first_example + EFFECTIVE_BATCH)]
        supervised_tokens = sum(int((labels[1:] != -100).sum()) for _, labels in examples)

        loss_sum_total = 0.0
        for micro_start in range(0, EFFECTIVE_BATCH, MICRO_BATCH):
            input_ids, labels, attention_mask = collate(examples[micro_start:micro_start + MICRO_BATCH],
                                                        tokenizer.pad_token_id)
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss_sum = response_loss_sum(model, input_ids, labels, attention_mask)
            scaler.scale(loss_sum / supervised_tokens).backward()
            loss_sum_total += loss_sum.item()

        scaler.unscale_(optimizer)
        omega = None
        if run.l2sp_lambda > 0:
            omega = 0.5 * squared_displacement(named_trainable, reference)
            add_l2sp_gradient(named_trainable, reference, run.l2sp_lambda)
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, MAX_GRAD_NORM).item()

        scale_before = scaler.get_scale()
        learning_rate = scheduler.get_last_lr()[0]
        scaler.step(optimizer)
        scaler.update()
        skipped = use_amp and scaler.get_scale() < scale_before
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        completed = update + 1
        sft_loss = loss_sum_total / supervised_tokens
        record = {
            "update": completed,
            "examples_seen": completed * EFFECTIVE_BATCH,
            "sft_loss": sft_loss,
            "lr": learning_rate,
            "grad_norm": grad_norm,
            "loss_scale": scale_before,
            "skipped": skipped,
            "supervised_tokens": supervised_tokens,
            "seconds": time.time() - update_start,
        }
        if omega is not None:
            record["omega"] = omega
            record["lambda_omega"] = run.l2sp_lambda * omega
            record["total_loss"] = sft_loss + run.l2sp_lambda * omega
        if skipped:
            summary["skipped_updates"].append(completed)
        append_jsonl(log_path, record)
        if completed % 10 == 0 or completed == 1:
            print(f"[{run_name}] update {completed}/{TOTAL_UPDATES} loss {sft_loss:.4f} lr {learning_rate:.2e} "
                  f"gnorm {grad_norm:.3f} {record['seconds']:.1f}s", flush=True)

        if completed in (MIDPOINT_UPDATE, FINAL_UPDATE):
            stage = "midpoint" if completed == MIDPOINT_UPDATE else "final"
            export_dir = directory / stage
            if run.is_full_ft():
                summary["exports"][stage] = export_full_model(model, named_trainable, reference, export_dir,
                                                              tokenizer)
            else:
                summary["exports"][stage] = export_adapters(adapters or find_adapters(model), export_dir)
        if completed in CHECKPOINT_UPDATES:
            size = save_recovery(directory, completed, named_trainable, optimizer, scheduler, scaler, log_path)
            summary["checkpoints"][str(completed)] = {"recovery_state_bytes": size}
            segment = {
                "start_update": start_update,
                "end_update": completed,
                "seconds": time.time() - segment_start,
                "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated() if use_amp else None,
                "peak_memory_reserved_bytes": torch.cuda.max_memory_reserved() if use_amp else None,
                "gpu": environment_info().get("gpu"),
            }
            summary["segments"] = [item for item in summary["segments"]
                                   if item["start_update"] != start_update] + [segment]
            save_json(summary_path, summary)

    summary["completed_updates"] = last_update
    summary["training_seconds"] = sum(segment["seconds"] for segment in summary["segments"])
    save_json(summary_path, summary)
    if last_update == TOTAL_UPDATES:
        (directory / "DONE").write_text("ok")
        if not keep_recovery and (directory / "recovery").exists():
            shutil.rmtree(directory / "recovery")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, choices=sorted(RUNS))
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--keep-recovery", action="store_true",
                        help="keep the last optimizer-state checkpoint after the run completes (~7 GB for full FT)")
    args = parser.parse_args()
    train(args.run, args.max_updates, args.device, args.keep_recovery)


if __name__ == "__main__":
    main()
