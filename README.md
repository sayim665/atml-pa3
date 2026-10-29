# ATML PA3 (EE-5102 / CS-6304, Fall 2026)

Code, configurations, split IDs and machine-readable results for Programming Assignment 3.

| Task | Folder | Status |
|---|---|---|
| 1. LLM fine-tuning (Full FT, L2-SP, LoRA-4/16, OPLoRA-16) | `task1/` | in progress |
| 2. Vision-language model and visual grounding | `task2/` | not started |
| 3. Test-time compute | `task3/` | not started |

## Task 1: Large Language Model Fine-Tuning

Qwen3-0.6B (thinking disabled) is fine-tuned for one epoch on the first 40,000 eligible MetaMathQA examples with five methods. Each run is evaluated with lm-evaluation-harness on GSM8K, HellaSwag, BoolQ and ARC-Easy, and on 24 fixed qualitative prompts.

### Layout

| File | Purpose |
|---|---|
| `task1/config.py` | All fixed settings from the manual (seed 6304, batch 2x16, 1,250 updates, learning rates, LoRA/OPLoRA ranks) |
| `task1/data.py` | MetaMathQA filtering (empty, >2,048 tokens, GSM8K-test overlap after normalization), tokenization with response-only labels, probe-set sampling |
| `task1/lora.py` | LoRA and OPLoRA adapters (projected factors `B - U_k U_k^T B`, `A - A V_k V_k^T`), export, merging |
| `task1/train.py` | FP16 AMP training loop with FP32 master weights or FP32 adapters, L2-SP penalty, resumable checkpoints at updates 312/625/937/1250 |
| `task1/evaluate.py` | lm-eval benchmarks (zero-shot; GSM8K with the non-thinking chat template, greedy, 1,024 tokens), per-example outputs, truncation flags, qualitative generations, merged-vs-unmerged adapter check |
| `task1/run_queue.py` | Runs train/eval jobs across GPUs with a wall-clock budget (Kaggle 12 h sessions) |
| `kaggle/task1_runner.ipynb` | Thin Kaggle notebook: clone repo, restore previous outputs, run the queue |
| `results/task1/` | Split IDs, scores, per-example outputs, training logs and summaries |

### Reproduce

```bash
pip install -r requirements.txt
python -m task1.data                       # builds the training cache and probe set
python -m task1.train --run full_ft        # full_ft | l2sp | lora4 | lora16 | oplora16
python -m task1.evaluate --condition baseline
```

On Kaggle (GPU T4 x2, internet enabled), run `kaggle/task1_runner.ipynb` with "Save & Run All". To continue an interrupted session, attach the previous version's output as an input and run again; finished jobs are skipped and unfinished runs resume from their last recovery checkpoint.

Set `ATML_SMOKE_TEST=1` to run a 4-update CPU smoke test of the full pipeline.

### External resources

- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) task definitions (`gsm8k_cot`, `hellaswag`, `boolq`, `arc_easy`), used unmodified.
- `task1/data/first_plot_questions.yaml` is copied from the [emergent-misalignment](https://github.com/emergent-misalignment/emergent-misalignment) repository (Betley et al.).
- LoRA (Hu et al.) and OPLoRA (Xiong and Xie) are implemented from the papers' equations; no adapter library is used.
