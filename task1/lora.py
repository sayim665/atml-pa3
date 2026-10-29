"""LoRA and OPLoRA adapters written from scratch so both share identical initialization and code paths.

OPLoRA (Xiong and Xie): with W0 = U S V^T and frozen leading bases U_k, V_k,
    B_eff = B - U_k (U_k^T B),   A_eff = A - (A V_k) V_k^T,   dW = (alpha / r) B_eff A_eff.
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from task1.config import LORA_DROPOUT, LORA_TARGETS


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout=LORA_DROPOUT, protection_rank=0):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.protection_rank = protection_rank
        self.dropout = nn.Dropout(dropout)

        device = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        if protection_rank > 0:
            output_basis, input_basis = leading_singular_bases(base.weight, protection_rank)
            self.register_buffer("U_k", output_basis, persistent=False)
            self.register_buffer("V_k", input_basis, persistent=False)
        else:
            self.U_k = None
            self.V_k = None

    def effective_factors(self):
        lora_A = self.lora_A
        lora_B = self.lora_B
        if self.protection_rank == 0:
            return lora_A, lora_B
        with torch.autocast(device_type=lora_A.device.type, enabled=False):
            projected_B = lora_B - self.U_k @ (self.U_k.T @ lora_B)
            projected_A = lora_A - (lora_A @ self.V_k) @ self.V_k.T
        return projected_A, projected_B

    def delta_weight(self):
        lora_A, lora_B = self.effective_factors()
        with torch.autocast(device_type=lora_A.device.type, enabled=False):
            return self.scaling * (lora_B.float() @ lora_A.float())

    def forward(self, inputs):
        base_output = self.base(inputs)
        lora_A, lora_B = self.effective_factors()
        hidden = F.linear(self.dropout(inputs), lora_A.to(inputs.dtype))
        update = F.linear(hidden, lora_B.to(inputs.dtype))
        return base_output + self.scaling * update


def leading_singular_bases(weight, protection_rank):
    matrix = weight.detach().float()
    left, _, right_transposed = torch.linalg.svd(matrix, full_matrices=False)
    output_basis = left[:, :protection_rank].contiguous()
    input_basis = right_transposed[:protection_rank, :].T.contiguous()
    return output_basis, input_basis


def target_linears(model):
    for layer_index, layer in enumerate(model.model.layers):
        for parent_name, parent in (("self_attn", layer.self_attn), ("mlp", layer.mlp)):
            for target in LORA_TARGETS:
                module = getattr(parent, target, None)
                if module is None:
                    continue
                yield f"model.layers.{layer_index}.{parent_name}.{target}", parent, target, module


def inject_adapters(model, rank, alpha, protection_rank=0):
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    start = time.time()
    adapters = {}
    for name, parent, target, module in list(target_linears(model)):
        adapter = LoRALinear(module, rank, alpha, protection_rank=protection_rank)
        setattr(parent, target, adapter)
        adapters[name] = adapter
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    basis_bytes = 0
    if protection_rank > 0:
        for adapter in adapters.values():
            basis_bytes += adapter.U_k.numel() * adapter.U_k.element_size()
            basis_bytes += adapter.V_k.numel() * adapter.V_k.element_size()
    overhead = {
        "num_adapted_modules": len(adapters),
        "injection_seconds": time.time() - start,
        "basis_bytes": basis_bytes,
    }
    return adapters, overhead


def find_adapters(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}


def export_adapter_factors(adapters, path):
    """Save the effective (projected, for OPLoRA) factors used for inference and diagnostics."""
    start = time.time()
    tensors = {}
    scaling = None
    with torch.no_grad():
        for name, adapter in adapters.items():
            lora_A, lora_B = adapter.effective_factors()
            tensors[f"{name}.lora_A"] = lora_A.detach().float().cpu().contiguous()
            tensors[f"{name}.lora_B"] = lora_B.detach().float().cpu().contiguous()
            scaling = adapter.scaling
    metadata = {"scaling": str(scaling), "projected": "true"}
    save_file(tensors, str(path), metadata=metadata)
    return time.time() - start


def load_adapter_factors(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="pt") as handle:
        scaling = float(handle.metadata()["scaling"])
    tensors = load_file(str(path))
    factors = {}
    for key, tensor in tensors.items():
        module_name, factor_name = key.rsplit(".", 1)
        factors.setdefault(module_name, {})[factor_name] = tensor
    return factors, scaling


def adapter_delta(factors, module_name, scaling):
    pair = factors[module_name]
    return scaling * (pair["lora_B"].float() @ pair["lora_A"].float())


@torch.no_grad()
def merge_adapter_factors(model, factors, scaling):
    modules = dict(model.named_modules())
    for module_name in factors:
        linear = modules[module_name]
        delta = adapter_delta(factors, module_name, scaling).to(linear.weight.device)
        merged = linear.weight.float() + delta
        linear.weight.copy_(merged.to(linear.weight.dtype))
    return model


@torch.no_grad()
def attach_frozen_adapters(model, factors, scaling):
    """Unmerged form: base weights plus explicit low-rank path (for the merged-vs-unmerged logit check)."""
    for name, parent, target, module in list(target_linears(model)):
        if name not in factors:
            continue
        rank = factors[name]["lora_A"].shape[0]
        adapter = LoRALinear(module, rank, alpha=scaling * rank, dropout=0.0)
        adapter.lora_A.copy_(factors[name]["lora_A"])
        adapter.lora_B.copy_(factors[name]["lora_B"])
        setattr(parent, target, adapter)
    return model
