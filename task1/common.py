"""Shared helpers: paths, seeding, model/tokenizer loading, chat formatting, JSON I/O."""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from task1.config import MODEL_NAME

REPO_ROOT = Path(__file__).resolve().parent.parent


def output_root():
    return Path(os.environ.get("ATML_OUT", REPO_ROOT / "outputs" / "task1"))


def results_root():
    return Path(os.environ.get("ATML_RESULTS", REPO_ROOT / "results" / "task1"))


def run_dir(run_name):
    path = output_root() / "runs" / run_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def load_tokenizer(path=MODEL_NAME):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_causal_lm(path, dtype, device):
    import transformers
    from transformers import AutoModelForCausalLM

    major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
    if (major, minor) >= (4, 56):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation="sdpa")
    return model.to(device=device, dtype=dtype)


def chat_prompt_text(tokenizer, user_text, system_text=None):
    messages = []
    if system_text is not None:
        messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": user_text})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def force_non_thinking_template(tokenizer):
    """Make every apply_chat_template call (including lm-eval's) use enable_thinking=False."""
    if getattr(tokenizer, "_non_thinking_patched", False):
        return tokenizer
    original = tokenizer.apply_chat_template

    def apply_without_thinking(*args, **kwargs):
        kwargs["enable_thinking"] = False
        return original(*args, **kwargs)

    tokenizer.apply_chat_template = apply_without_thinking
    tokenizer._non_thinking_patched = True
    return tokenizer


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    os.replace(tmp_path, path)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def directory_size_bytes(path):
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def environment_info():
    import platform

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    try:
        import transformers

        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    try:
        import lm_eval

        info["lm_eval"] = getattr(lm_eval, "__version__", "unknown")
    except ImportError:
        pass
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info
