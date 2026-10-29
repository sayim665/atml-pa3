"""Fixed settings for Task 1 (LLM fine-tuning). Every number here comes from the PA3 manual."""

import os
from dataclasses import dataclass, field, asdict

MODEL_NAME = "Qwen/Qwen3-0.6B"
SEED = 6304

TRAIN_DATASET = "meta-math/MetaMathQA"
GSM8K_DATASET = "openai/gsm8k"
HELLASWAG_DATASET = "Rowan/hellaswag"

NUM_TRAIN_EXAMPLES = 40_000
MAX_SEQ_LEN = 2048

MICRO_BATCH = 2
GRAD_ACCUM = 16
EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
TOTAL_UPDATES = NUM_TRAIN_EXAMPLES // EFFECTIVE_BATCH
WARMUP_UPDATES = 38

ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

CHECKPOINT_UPDATES = (312, 625, 937, 1250)
MIDPOINT_UPDATE = 625
FINAL_UPDATE = 1250

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
LORA_DROPOUT = 0.05

DIAGNOSTIC_LAYERS = (0, 13, 27)
DIAGNOSTIC_MODULE = "down_proj"
PROTECTION_RANK = 128

PROBE_GSM8K = 32
PROBE_HELLASWAG = 32

GSM8K_MAX_NEW_TOKENS = 1024
QUALITATIVE_MAX_NEW_TOKENS = 512

SMOKE_TEST = os.environ.get("ATML_SMOKE_TEST") == "1"
if SMOKE_TEST:
    NUM_TRAIN_EXAMPLES = 16
    GRAD_ACCUM = 2
    EFFECTIVE_BATCH = MICRO_BATCH * GRAD_ACCUM
    TOTAL_UPDATES = 4
    WARMUP_UPDATES = 1
    CHECKPOINT_UPDATES = (1, 2, 3, 4)
    MIDPOINT_UPDATE = 2
    FINAL_UPDATE = 4
    GSM8K_MAX_NEW_TOKENS = 32
    QUALITATIVE_MAX_NEW_TOKENS = 8


@dataclass
class RunConfig:
    name: str
    method: str
    peak_lr: float
    l2sp_lambda: float = 0.0
    lora_rank: int = 0
    lora_alpha: int = 0
    protection_rank: int = 0
    extra: dict = field(default_factory=dict)

    def is_full_ft(self):
        return self.method in ("full_ft", "l2sp")

    def is_adapter(self):
        return self.method in ("lora", "oplora")

    def to_dict(self):
        return asdict(self)


RUNS = {
    "full_ft": RunConfig(name="full_ft", method="full_ft", peak_lr=2e-5),
    "l2sp": RunConfig(name="l2sp", method="l2sp", peak_lr=2e-5, l2sp_lambda=1e-3),
    "lora4": RunConfig(name="lora4", method="lora", peak_lr=2e-4, lora_rank=4, lora_alpha=8),
    "lora16": RunConfig(name="lora16", method="lora", peak_lr=2e-4, lora_rank=16, lora_alpha=32),
    "oplora16": RunConfig(name="oplora16", method="oplora", peak_lr=2e-4, lora_rank=16, lora_alpha=32,
                          protection_rank=PROTECTION_RANK),
}

EVAL_CONDITIONS = ("baseline", "full_ft_mid", "full_ft", "l2sp", "lora4", "lora16", "oplora16")
