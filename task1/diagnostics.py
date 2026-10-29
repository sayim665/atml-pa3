"""Weight-space and activation diagnostics on down_proj in layers 0, 13, 27 (FP32, CPU is fine).

    python -m task1.diagnostics

Needs: trained exports under <ATML_OUT>/runs and results/task1/splits/probe_prompts.json (written by
the baseline evaluation).
"""

import argparse

import torch

from task1.common import load_causal_lm, load_json, load_tokenizer, output_root, results_root, run_dir, save_json
from task1.config import DIAGNOSTIC_LAYERS, MODEL_NAME, PROTECTION_RANK, RUNS
from task1.lora import adapter_delta, leading_singular_bases, load_adapter_factors

DIAGNOSED_CONDITIONS = ("full_ft_mid", "full_ft", "l2sp", "lora4", "lora16", "oplora16")


def module_name(layer_index):
    return f"model.layers.{layer_index}.mlp.down_proj"


def starting_weights(model):
    return {layer_index: model.model.layers[layer_index].mlp.down_proj.weight.detach().float().clone()
            for layer_index in DIAGNOSTIC_LAYERS}


def condition_deltas(condition, starting):
    if condition == "full_ft_mid":
        export = torch.load(run_dir("full_ft") / "midpoint" / "diagnostic_down_proj_fp32.pt")
        return {layer_index: export[module_name(layer_index)] - starting[layer_index]
                for layer_index in DIAGNOSTIC_LAYERS}
    run = RUNS[condition]
    if run.is_full_ft():
        export = torch.load(run_dir(condition) / "final" / "diagnostic_down_proj_fp32.pt")
        return {layer_index: export[module_name(layer_index)] - starting[layer_index]
                for layer_index in DIAGNOSTIC_LAYERS}
    factors, scaling = load_adapter_factors(run_dir(condition) / "final" / "adapter_factors.safetensors")
    return {layer_index: adapter_delta(factors, module_name(layer_index), scaling)
            for layer_index in DIAGNOSTIC_LAYERS}


def weight_space_metrics(delta, weight, output_basis, input_basis):
    delta_norm_sq = delta.pow(2).sum().item()
    projected_out_sq = (output_basis.T @ delta).pow(2).sum().item()
    projected_in_sq = (delta @ input_basis).pow(2).sum().item()
    return {
        "rho_k": projected_out_sq / delta_norm_sq if delta_norm_sq > 0 else float("nan"),
        "input_side_fraction": projected_in_sq / delta_norm_sq if delta_norm_sq > 0 else float("nan"),
        "eta": delta_norm_sq ** 0.5 / weight.norm().item(),
        "delta_frobenius": delta_norm_sq ** 0.5,
        "max_abs_U_k_T_delta": (output_basis.T @ delta).abs().max().item(),
        "max_abs_delta_V_k": (delta @ input_basis).abs().max().item(),
    }


@torch.no_grad()
def cache_probe_inputs(model, tokenizer):
    prompts = load_json(results_root() / "splits" / "probe_prompts.json")
    captured = {}
    hooks = []
    for layer_index in DIAGNOSTIC_LAYERS:
        module = model.model.layers[layer_index].mlp.down_proj

        def hook(_, inputs, layer_index=layer_index):
            captured[layer_index] = inputs[0][0, -1].float().clone()

        hooks.append(module.register_forward_hook(hook))

    cache = {"gsm8k": {layer: [] for layer in DIAGNOSTIC_LAYERS},
             "hellaswag": {layer: [] for layer in DIAGNOSTIC_LAYERS}}
    for group in ("gsm8k", "hellaswag"):
        for item in prompts[group]:
            input_ids = tokenizer(item["prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"]
            model(input_ids=input_ids)
            for layer_index in DIAGNOSTIC_LAYERS:
                cache[group][layer_index].append(captured[layer_index])
    for hook in hooks:
        hook.remove()
    return {group: {layer: torch.stack(vectors) for layer, vectors in layers.items()}
            for group, layers in cache.items()}


def activation_ratios(delta, weight, inputs):
    numerator = (inputs @ delta.T).norm(dim=-1)
    denominator = (inputs @ weight.T).norm(dim=-1) + 1e-8
    return numerator / denominator


def summarize(values):
    return {
        "mean": values.mean().item(),
        "median": values.median().item(),
        "std": values.std().item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conditions", nargs="+", default=list(DIAGNOSED_CONDITIONS))
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    model = load_causal_lm(MODEL_NAME, torch.float32, "cpu").eval()
    starting = starting_weights(model)
    bases = {layer_index: leading_singular_bases(starting[layer_index], PROTECTION_RANK)
             for layer_index in DIAGNOSTIC_LAYERS}

    cache_path = output_root() / "diagnostics" / "probe_down_proj_inputs.pt"
    if cache_path.exists():
        probe_inputs = torch.load(cache_path)
    else:
        probe_inputs = cache_probe_inputs(model, tokenizer)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(probe_inputs, cache_path)

    weight_results = {}
    activation_results = {}
    per_prompt = {}
    for condition in args.conditions:
        deltas = condition_deltas(condition, starting)
        weight_results[condition] = {}
        activation_results[condition] = {}
        per_prompt[condition] = {}
        for layer_index in DIAGNOSTIC_LAYERS:
            output_basis, input_basis = bases[layer_index]
            weight = starting[layer_index]
            delta = deltas[layer_index].float()
            weight_results[condition][str(layer_index)] = weight_space_metrics(delta, weight, output_basis,
                                                                               input_basis)
            math_ratios = activation_ratios(delta, weight, probe_inputs["gsm8k"][layer_index])
            other_ratios = activation_ratios(delta, weight, probe_inputs["hellaswag"][layer_index])
            activation_results[condition][str(layer_index)] = {
                "math": summarize(math_ratios),
                "non_math": summarize(other_ratios),
                "math_over_non_math_mean": (math_ratios.mean() / other_ratios.mean()).item(),
            }
            per_prompt[condition][str(layer_index)] = {"math": math_ratios.tolist(),
                                                       "non_math": other_ratios.tolist()}
        print(condition, weight_results[condition], flush=True)

    out_dir = results_root() / "diagnostics"
    save_json(out_dir / "weight_space.json", weight_results)
    save_json(out_dir / "activation_ratios.json", activation_results)
    save_json(out_dir / "activation_ratios_per_prompt.json", per_prompt)


if __name__ == "__main__":
    main()
